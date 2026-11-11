"""Work threshold (≈ cell updates) below which a loop runs serially.

A `Ref` so that tests can set it to `1` and force the threaded path on small
meshes; restore the default afterwards.
"""
const _PARALLEL_GRAIN = Ref(1024)
const _BATCHES_PER_THREAD = 4

"""Return the number of Julia compute threads available to numerical kernels."""
@inline _compute_threads() = Threads.nthreads(:default)

"""
    _parallel_batch(number_of_items, work)

Return the Polyester `minbatch` of a threaded launch over `number_of_items`
items of `work` cell updates each, or `0` when the loop runs serially: on one
compute thread, or when `number_of_items * work` is below `_PARALLEL_GRAIN[]`.
"""
@inline function _parallel_batch(number_of_items::Integer, work::Integer)
    number_of_items >= 0 || throw(ArgumentError("number_of_items must be nonnegative"))
    work >= 1 || throw(ArgumentError("work must be positive"))
    grain = _PARALLEL_GRAIN[]
    (_compute_threads() == 1 || number_of_items * work < grain) && return 0
    return max(1, cld(grain, _BATCHES_PER_THREAD * work))
end

"""
    _parallel_for(f, number_of_items; work = 1)

Call `f(index)` for `index in 1:number_of_items`, split into contiguous
`Polyester.@batch` ranges when more than one compute thread is available and
`number_of_items * work` reaches `_PARALLEL_GRAIN[]`; otherwise run serially in
index order. `work` estimates the cost of one item in cell updates (for
example `nx * ny` for one block). This and the `@_parallel` loop form
are the package's only launch primitives.

Every index must own its outputs: results may not depend on which thread or in
which order indices run, so the serial and threaded paths are bitwise
identical. Reductions use fixed chunk schedules instead. `f` must not throw;
record failures in owned outputs and throw after the launch. Polyester runs
`f` on its worker threads, so any user callback reached from `f` must be
thread-safe.

Polyester passes a closure that captures arrays to its workers in a heap box
(a few dozen bytes per threaded launch); per-cell loops therefore use
`@_parallel`, whose threaded path launches the loop body itself.
"""
@inline function _parallel_for(f::F, number_of_items::Integer; work::Integer = 1) where {F}
    minimum_batch = _parallel_batch(number_of_items, work)
    if iszero(minimum_batch)
        for index in 1:number_of_items
            f(index)
        end
    else
        Polyester.@batch minbatch = minimum_batch for index in 1:number_of_items
            f(index)
        end
    end
    return nothing
end

const _NESTED_SCOPE_HEADS = (:function, :->, :quote, :generator, :comprehension)

# A loop body runs once per element on either path: `continue` skips the
# element, while `break` and `return` have no per-element meaning on the
# threaded path and are rejected.
_check_parallel_body(expression, in_nested_loop = false) = nothing
function _check_parallel_body(expression::Expr, in_nested_loop = false)
    head = expression.head
    head in _NESTED_SCOPE_HEADS && return nothing
    head === :return &&
        throw(ArgumentError("@_parallel loop bodies cannot return"))
    !in_nested_loop && head === :break &&
        throw(ArgumentError("@_parallel loop bodies cannot break"))
    nested = in_nested_loop || head in (:for, :while)
    foreach(argument -> _check_parallel_body(argument, nested), expression.args)
    return nothing
end

"""
    @_parallel [work = expression] for index in range ... end

Loop form of `_parallel_for` for a 1-based `AbstractVector` `range` (such as
`1:n` or `eachindex(vector)`), with the same ownership rules and serial
cut-off. The body is expanded twice: a plain `for` loop for the serial path
and the `Polyester.@batch` loop of the threaded path, which captures the
body's arrays directly, so neither path allocates. `continue` skips an
element; `break` and `return` are rejected. Assigning a variable of the
enclosing function from the body is a race on the threaded path; bodies only
write owned array entries.
"""
macro _parallel(arguments...)
    isempty(arguments) && throw(ArgumentError("@_parallel requires a for loop"))
    loop = last(arguments)
    loop isa Expr && loop.head === :for ||
        throw(ArgumentError("@_parallel requires a for loop"))
    iterator = loop.args[1]
    iterator isa Expr && iterator.head in (:(=), :in) ||
        throw(ArgumentError("@_parallel requires one loop iterator"))
    work = 1
    for option in arguments[1:(end - 1)]
        option isa Expr && option.head === :(=) && option.args[1] === :work ||
            throw(ArgumentError("@_parallel accepts only `work = expression`"))
        work = option.args[2]
    end
    variable, range = iterator.args
    variable isa Symbol ||
        throw(ArgumentError("@_parallel requires a single loop variable"))
    body = loop.args[2]
    _check_parallel_body(body)
    items = gensym(:parallel_items)
    position = gensym(:parallel_position)
    minimum_batch = gensym(:minimum_batch)
    # The body is expanded twice on purpose. Polyester passes a closure
    # argument to its workers in a heap box (measured 32 B per threaded launch
    # for a closure over two vectors), while a `@batch` loop over the body
    # itself captures the arrays as pointers and does not allocate.
    # `@batch` is referenced through a GlobalRef so that call sites outside
    # the package need not import Polyester.
    threaded_loop = Expr(
        :macrocall, GlobalRef(Polyester, Symbol("@batch")), __source__,
        :(minbatch = $minimum_batch),
        :(
            for $position in 1:length($items)
                local $variable = @inbounds $items[$position]
                $body
            end
        ),
    )
    expression = quote
        local $items = $range
        local $minimum_batch = $(_parallel_batch)(length($items), $work)
        if iszero($minimum_batch)
            for $variable in $items
                $body
            end
        else
            $threaded_loop
        end
        nothing
    end
    return esc(expression)
end

"""Traverse block-backed cell data with one dense column range per work item.

Serial execution (one thread, or fewer cells than the grain) visits each
block once, allowing callers to hoist its dense arrays outside the local
`j, i` loops. With several threads and fewer than `_BATCHES_PER_THREAD` leaf
blocks per thread, the columns of each block are split into tiles so that a
one-block uniform mesh still uses all threads.

`f(block, first_j, last_j)` must own every output in the supplied column range.
"""
function _parallel_block_columns(f::F, mesh) where {F}
    nx, ny = mesh.block_size
    number_of_blocks = nblocks(mesh)
    threads = _compute_threads()
    if threads == 1 || number_of_blocks * nx * ny < _PARALLEL_GRAIN[]
        for block in 1:number_of_blocks
            f(block, 1, ny)
        end
        return nothing
    end
    requested_tiles = max(1, cld(_BATCHES_PER_THREAD * threads, number_of_blocks))
    columns_per_tile = max(1, cld(ny, requested_tiles))
    tiles_per_block = cld(ny, columns_per_tile)
    # A `@batch` body (not a closure argument) so that only `f` is boxed per
    # launch, as before the launch layer was unified.
    @_parallel work = nx * columns_per_tile for tile in 1:(number_of_blocks * tiles_per_block)
        block, local_tile = divrem(tile - 1, tiles_per_block)
        first_j = local_tile * columns_per_tile + 1
        f(block + 1, first_j, min(ny, first_j + columns_per_tile - 1))
    end
    return nothing
end

"""Work estimate of one leaf block in block-level loops (block copies,
refinement indicators, block transfer): four blocks reach the grain, the
threshold these loops used before the launch layer was unified."""
@inline _block_work() = max(1, cld(_PARALLEL_GRAIN[], 4))

"""Multiply a sparse matrix from a stored transpose, one output row per task."""
function _parallel_csr_row_mul!(
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


"""Build one-based CSR offsets from per-row counts."""
function _csr_offsets(counts, offsets = Vector{Int}(undef, length(counts) + 1))
    resize!(offsets, length(counts) + 1)
    offsets[1] = 1
    for index in eachindex(counts)
        offsets[index + 1] = offsets[index] + counts[index]
    end
    return offsets
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
