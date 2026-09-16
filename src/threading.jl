const _PARALLEL_GRAIN = 1024
const _BATCHES_PER_THREAD = 4

"""Return the number of Julia compute threads available to numerical kernels."""
@inline _compute_threads() = Threads.nthreads(:default)

@inline function _parallel_batch_size(number_of_items, maximum_grain)
    target = cld(number_of_items, _BATCHES_PER_THREAD * _compute_threads())
    return max(1, min(maximum_grain, target))
end

"""Traverse block-backed cell data with one dense column range per work item.

Serial execution visits each block once, allowing callers to hoist its dense
arrays outside the local `j, i` loops. Parallel execution keeps the same layout
and subdivides the columns of a block only when there are too few leaf blocks
to supply `_BATCHES_PER_THREAD` work items. This preserves locality for the
usual 8x8 AMR blocks while allowing a large one-block uniform mesh to use all
threads.

`f(block, first_j, last_j)` must own every output in the supplied column range.
"""
function _parallel_block_columns(
        f, mesh; grain::Integer = _PARALLEL_GRAIN,
    )
    nx, ny = mesh.block_size
    number_of_blocks = nblocks(mesh)
    number_of_cells = number_of_blocks * nx * ny
    if _compute_threads() == 1 || number_of_cells < grain
        for block in 1:number_of_blocks
            f(block, 1, ny)
        end
        return nothing
    end

    target_items = _BATCHES_PER_THREAD * _compute_threads()
    requested_tiles = max(1, cld(target_items, number_of_blocks))
    columns_per_tile = max(1, cld(ny, requested_tiles))
    tiles_per_block = cld(ny, columns_per_tile)
    number_of_tiles = number_of_blocks * tiles_per_block
    batch_size = _parallel_batch_size(number_of_tiles, number_of_tiles)
    Polyester.@batch minbatch = batch_size for tile in 1:number_of_tiles
        block, local_tile = divrem(tile - 1, tiles_per_block)
        first_j = local_tile * columns_per_tile + 1
        last_j = min(ny, first_j + columns_per_tile - 1)
        f(block + 1, first_j, last_j)
    end
    return nothing
end

"""Traverse one dense `nx` by `ny` region in contiguous column ranges."""
function _parallel_columns(
        f, nx::Integer, ny::Integer; grain::Integer = _PARALLEL_GRAIN,
    )
    number_of_items = nx * ny
    if _compute_threads() == 1 || number_of_items < grain
        f(1, ny)
        return nothing
    end
    requested_tiles = min(ny, _BATCHES_PER_THREAD * _compute_threads())
    columns_per_tile = max(1, cld(ny, requested_tiles))
    number_of_tiles = cld(ny, columns_per_tile)
    batch_size = _parallel_batch_size(number_of_tiles, number_of_tiles)
    Polyester.@batch minbatch = batch_size for tile in 1:number_of_tiles
        first_j = (tile - 1) * columns_per_tile + 1
        last_j = min(ny, first_j + columns_per_tile - 1)
        f(first_j, last_j)
    end
    return nothing
end

"""Run independent indexed work on Julia's composable task scheduler.

The grain threshold keeps task-launch overhead out of small blocks and coarse
multigrid levels.  Iterations must own their outputs; reductions and scatter
updates require a separate race-free schedule.
"""
function _parallel_for(f, number_of_items::Integer; grain = _PARALLEL_GRAIN)
    number_of_items >= 0 || throw(ArgumentError("number_of_items must be nonnegative"))
    if _compute_threads() == 1 || number_of_items < grain
        for index in 1:number_of_items
            f(index)
        end
    else
        batch_size = _parallel_batch_size(number_of_items, grain)
        Polyester.@batch minbatch = batch_size for index in 1:number_of_items
            f(index)
        end
    end
    return nothing
end

"""Expand an allocation-free parallel loop at its numerical call site."""
macro _parallel(loop)
    loop isa Expr && loop.head === :for ||
        throw(ArgumentError("@_parallel requires a for loop"))
    iterator = loop.args[1]
    iterator isa Expr && iterator.head in (:(=), :in) ||
        throw(ArgumentError("@_parallel requires one loop iterator"))
    variable, range = iterator.args
    body = loop.args[2]
    captured_range = gensym(:parallel_range)
    batch_size = gensym(:batch_size)
    quote_expression = quote
        local $captured_range = $range
        if _compute_threads() == 1 ||
                length($captured_range) < _PARALLEL_GRAIN
            for $variable in $captured_range
                $body
            end
        else
            local $batch_size = _parallel_batch_size(
                length($captured_range), _PARALLEL_GRAIN,
            )
            Polyester.@batch minbatch = $batch_size for $variable in $captured_range
                $body
            end
        end
        nothing
    end
    return esc(quote_expression)
end

"""Use Julia's task scheduler for coarse work that traverses pointer-rich trees.

Polyester is reserved for dense numerical loops. Its array lowering is not a
good fit for recursive tree searches with dictionaries and heterogeneous
objects, where the larger per-iteration cost amortizes ordinary task launch.
"""
macro _parallel_tasks(loop)
    loop isa Expr && loop.head === :for ||
        throw(ArgumentError("@_parallel_tasks requires a for loop"))
    iterator = loop.args[1]
    iterator isa Expr && iterator.head in (:(=), :in) ||
        throw(ArgumentError("@_parallel_tasks requires one loop iterator"))
    variable, range = iterator.args
    body = loop.args[2]
    captured_range = gensym(:parallel_range)
    quote_expression = quote
        local $captured_range = $range
        if _compute_threads() == 1 ||
                length($captured_range) < _PARALLEL_GRAIN
            for $variable in $captured_range
                $body
            end
        else
            Threads.@threads :dynamic for $variable in $captured_range
                $body
            end
        end
        nothing
    end
    return esc(quote_expression)
end

"""Multiply a sparse matrix from a stored transpose, one output row per task."""
function _parallel_sparse_mul_from_transpose!(
        destination::AbstractVector{T},
        matrix_transpose::SparseMatrixCSC{T, Int},
        source::AbstractVector{T},
    ) where {T}
    length(destination) == size(matrix_transpose, 2) ||
        throw(DimensionMismatch("sparse output size differs"))
    length(source) == size(matrix_transpose, 1) ||
        throw(DimensionMismatch("sparse input size differs"))
    column_pointers = matrix_transpose.colptr
    rows = rowvals(matrix_transpose)
    matrix_values = nonzeros(matrix_transpose)
    @_parallel for output in 1:length(destination)
        value = zero(eltype(destination))
        for pointer in column_pointers[output]:(column_pointers[output + 1] - 1)
            @inbounds value += matrix_values[pointer] * source[rows[pointer]]
        end
        @inbounds destination[output] = value
    end
    return destination
end

"""Multiply by a sparse transpose, using each source column as one output row."""
function _parallel_sparse_transpose_mul!(
        destination::AbstractVector{T}, matrix::SparseMatrixCSC{T, Int},
        source::AbstractVector{T},
    ) where {T}
    length(destination) == size(matrix, 2) ||
        throw(DimensionMismatch("sparse-transpose output size differs"))
    length(source) == size(matrix, 1) ||
        throw(DimensionMismatch("sparse-transpose input size differs"))
    column_pointers = matrix.colptr
    rows = rowvals(matrix)
    matrix_values = nonzeros(matrix)
    @_parallel for output in 1:length(destination)
        value = zero(eltype(destination))
        for pointer in column_pointers[output]:(column_pointers[output + 1] - 1)
            @inbounds value += matrix_values[pointer] * source[rows[pointer]]
        end
        @inbounds destination[output] = value
    end
    return destination
end

"""Race-free row multiplication for a structurally symmetric CSC matrix."""
function _parallel_symmetric_sparse_mul!(
        destination::AbstractVector{T}, matrix::SparseMatrixCSC{T, Int},
        source::AbstractVector{T},
    ) where {T}
    size(matrix, 1) == size(matrix, 2) ||
        throw(DimensionMismatch("symmetric sparse action must be square"))
    return _parallel_sparse_transpose_mul!(destination, matrix, source)
end

"""Group contribution indices by a one-based output index."""
function _grouped_output_schedule(targets::AbstractVector{Int}, outputs::Int)
    permutation = sortperm(targets)
    offsets = Vector{Int}(undef, outputs + 1)
    contribution = 1
    for output in 1:outputs
        offsets[output] = contribution
        while contribution <= length(permutation) &&
                targets[permutation[contribution]] == output
            contribution += 1
        end
    end
    offsets[end] = contribution
    contribution == length(permutation) + 1 || throw(
        ArgumentError("contribution target lies outside output range"),
    )
    return offsets, permutation
end
