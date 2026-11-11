"""Work threshold (≈ cell updates) below which a loop runs serially.

A `Ref` so that tests can set it to `1` and force the threaded path on small
meshes; restore the default afterwards.
"""
const _PARALLEL_GRAIN = Ref(1024)
const _BATCHES_PER_THREAD = 4

"""Return the number of Julia compute threads available to numerical kernels."""
@inline _compute_threads() = Threads.nthreads(:default)

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
"""
function _parallel_for(f::F, number_of_items::Integer; work::Integer = 1) where {F}
    number_of_items >= 0 || throw(ArgumentError("number_of_items must be nonnegative"))
    work >= 1 || throw(ArgumentError("work must be positive"))
    grain = _PARALLEL_GRAIN[]
    if _compute_threads() == 1 || number_of_items * work < grain
        for index in 1:number_of_items
            f(index)
        end
    else
        minimum_batch = max(1, cld(grain, _BATCHES_PER_THREAD * work))
        Polyester.@batch minbatch = minimum_batch for index in 1:number_of_items
            f(index)
        end
    end
    return nothing
end

const _NESTED_SCOPE_HEADS = (:function, :->, :quote, :generator, :comprehension)

# Rewrite a loop body into a closure body: a `continue` that belongs to the
# parallel loop becomes `return nothing`. `break` cannot be expressed per
# iteration and a `return` would leave only the closure; both are rejected.
_closure_loop_body(expression, in_nested_loop = false) = expression
function _closure_loop_body(expression::Expr, in_nested_loop = false)
    head = expression.head
    head in _NESTED_SCOPE_HEADS && return expression
    head === :return &&
        throw(ArgumentError("@_parallel loop bodies cannot return"))
    if !in_nested_loop
        head === :continue && return :(return nothing)
        head === :break &&
            throw(ArgumentError("@_parallel loop bodies cannot break"))
    end
    nested = in_nested_loop || head in (:for, :while)
    return Expr(head, (_closure_loop_body(argument, nested) for argument in expression.args)...)
end

"""
    @_parallel [work = expression] for index in range ... end

Loop form of `_parallel_for`: the body becomes one closure called per
element of `range` (a 1-based `AbstractVector`, such as `1:n` or
`eachindex(vector)`), with the same ownership rules. `continue` skips an
element; `break` and `return` are rejected. As in any closure, assigning a
variable of the enclosing function captures it: on the threaded path that is a
race, and Julia boxes the variable, so bodies only write owned array entries.
"""
macro _parallel(arguments...)
    isempty(arguments) && throw(ArgumentError("@_parallel requires a for loop"))
    loop = last(arguments)
    loop isa Expr && loop.head === :for ||
        throw(ArgumentError("@_parallel requires a for loop"))
    iterator = loop.args[1]
    iterator isa Expr && iterator.head in (:(=), :in) ||
        throw(ArgumentError("@_parallel requires one loop iterator"))
    options = map(arguments[1:(end - 1)]) do option
        option isa Expr && option.head === :(=) && option.args[1] === :work ||
            throw(ArgumentError("@_parallel accepts only `work = expression`"))
        Expr(:kw, :work, option.args[2])
    end
    variable, range = iterator.args
    variable isa Symbol ||
        throw(ArgumentError("@_parallel requires a single loop variable"))
    body = _closure_loop_body(loop.args[2])
    items = gensym(:parallel_items)
    index = gensym(:parallel_index)
    expression = quote
        local $items = $range
        $(_parallel_for)(length($items); $(options...)) do $index
            local $variable = @inbounds $items[$index]
            $body
            return nothing
        end
    end
    return esc(expression)
end

"""Traverse block-backed cell data with one dense column range per work item.

Serial execution visits each block once, allowing callers to hoist its dense
arrays outside the local `j, i` loops. With several threads and fewer than
`_BATCHES_PER_THREAD` leaf blocks per thread, the columns of each block are
split into tiles so that a one-block uniform mesh still uses all threads.

`f(block, first_j, last_j)` must own every output in the supplied column range.
"""
function _parallel_block_columns(f::F, mesh) where {F}
    nx, ny = mesh.block_size
    number_of_blocks = nblocks(mesh)
    threads = _compute_threads()
    requested_tiles = threads == 1 ? 1 :
        max(1, cld(_BATCHES_PER_THREAD * threads, number_of_blocks))
    columns_per_tile = max(1, cld(ny, requested_tiles))
    tiles_per_block = cld(ny, columns_per_tile)
    _parallel_for(
        number_of_blocks * tiles_per_block; work = nx * columns_per_tile,
    ) do tile
        block, local_tile = divrem(tile - 1, tiles_per_block)
        first_j = local_tile * columns_per_tile + 1
        f(block + 1, first_j, min(ny, first_j + columns_per_tile - 1))
    end
    return nothing
end

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
