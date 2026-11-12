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
identical. Reductions use fixed chunk schedules instead.

`f` must not throw: Polyester rethrows an exception from the calling thread's
range as itself but one from a worker's range as a `TaskFailedException`,
while the other ranges keep running. A check whose outcome depends on the data
is therefore fused into the loop: the body records the failure in an owned,
preallocated output (one slot per iteration or per chunk, never `push!`) and
the caller throws after the launch. Cheap input checks may instead run serially
before the loop. User callbacks are never called from `f`; they run serially on
the calling task.

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

"""Map every `NaN` to the canonical `NaN` of its type, so that a maximum
returns the same bits whichever `NaN` term it meets first."""
@inline _canonical_nan(value) = isnan(value) ? oftype(value, NaN) : value

"""
    _fold_chunk_maxima(initial, partials, chunks)

Fold the chunk maxima `partials[1:chunks]` of a fixed-chunk maximum into
`initial`. Each chunk covers `_reduction_range(chunk, n)`; `max` is exact, so
with the canonical `NaN` terms of `_canonical_nan` the result is equal to the
old serial maximum up to the `NaN` payload (`NaN` terms are canonicalised, so
a `NaN` result always has the canonical bits) and is bitwise equal for every
thread count. Threaded maxima that run every step or solver iteration
use this with partials owned by their workspace, which keeps them
allocation-free.
"""
@inline function _fold_chunk_maxima(initial, partials, chunks)
    result = initial
    for chunk in 1:chunks
        result = max(result, @inbounds partials[chunk])
    end
    return result
end

"""Grow `partials` to hold one entry per fixed reduction chunk of `n` items
(allocates only when the topology grows)."""
@inline function _reduction_partials!(partials, n)
    chunks = _reduction_chunks(n)
    length(partials) < chunks && resize!(partials, chunks)
    return partials
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
write owned array entries, including a preallocated failure flag that the
caller inspects after the launch.

Variables used in the body must not shadow a function of the calling module
(a local `values`, `schedule`, `parent`, `time`, ...): Polyester leaves such
names to a closure, which allocates 48–64 B per threaded launch. Bodies that
capture a struct rather than its arrays also allocate (Polyester boxes the
argument tuple), so hot loops capture arrays and scalars only.
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

"""Length of one fixed reduction chunk.

Threaded reductions split `1:n` into chunks of this length whose boundaries
depend only on `n`. Each chunk is reduced in index order and the chunk results
are folded in chunk order, so the result is the same for every thread count;
for `n <= _REDUCTION_CHUNK` it equals the plain left-to-right reduction.
"""
const _REDUCTION_CHUNK = 1024

@inline _reduction_chunks(number_of_items) = cld(number_of_items, _REDUCTION_CHUNK)
@inline _reduction_range(chunk, number_of_items) =
    ((chunk - 1) * _REDUCTION_CHUNK + 1):min(number_of_items, chunk * _REDUCTION_CHUNK)

"""Element stride between the per-chunk blocks of a fixed-chunk reduction:
`labels` rounded up to a multiple of 8 `Float64`, i.e. a whole 64-byte cache
line for the solver's `Float64` sums, so that chunks owned by different tasks
never write the same line (false sharing). Only the layout changes."""
@inline _chunk_stride(labels) = max(1, cld(labels, 8) * 8)

# Terms of the slot sums. Singleton modes keep the threaded chunk loop free of
# closure arguments, which Polyester would box on every launch.
@inline _slot_term(::Val{:plain}, terms, weights, item) = @inbounds terms[item]
@inline _slot_term(::Val{:weighted}, terms, weights, item) =
    @inbounds weights[item] * terms[item]
@inline _slot_term(::Val{:absolute}, terms, weights, item) = abs(@inbounds terms[item])

"""
    _chunked_slot_sums!(sums, partials, slots, terms, weights, mode)

Set `sums[k]` to the sum over the items `i` with `slots[i] == k` (zero skips
the item) of `terms[i]` (`mode = Val(:plain)`), `weights[i] * terms[i]`
(`Val(:weighted)`) or `abs(terms[i])` (`Val(:absolute)`). The items are summed
in index order within fixed chunks of `_REDUCTION_CHUNK` items, and the chunk
sums are added in chunk order: the result does not depend on the thread count
and, for at most one chunk, equals the plain index-order sum; for more chunks
the grouping differs from it by rounding. `partials` is caller-owned scratch
holding one stride of sums per chunk (resized once, then allocation-free); the
stride is rounded up to a multiple of 8 `Float64`, so neighbouring chunks never
write the same cache line. The padding changes no arithmetic.
"""
function _chunked_slot_sums!(
        sums::AbstractVector{T}, partials::Vector{T}, slots::AbstractVector{<:Integer},
        terms::AbstractVector, weights, mode::Val,
    ) where {T}
    fill!(sums, zero(T))
    number_of_items = length(terms)
    chunks = _reduction_chunks(number_of_items)
    if chunks <= 1
        for item in 1:number_of_items
            slot = @inbounds slots[item]
            iszero(slot) && continue
            @inbounds sums[slot] += _slot_term(mode, terms, weights, item)
        end
        return sums
    end
    labels = length(sums)
    chunk_stride = _chunk_stride(labels)
    length(partials) < (chunks - 1) * chunk_stride + labels &&
        resize!(partials, (chunks - 1) * chunk_stride + labels)
    @_parallel work = _REDUCTION_CHUNK for chunk in 1:chunks
        offset = (chunk - 1) * chunk_stride
        if labels == 1
            # One component (the common case): accumulate in a register,
            # the same additions in the same order.
            total = zero(T)
            for item in _reduction_range(chunk, number_of_items)
                iszero(@inbounds slots[item]) && continue
                total += _slot_term(mode, terms, weights, item)
            end
            @inbounds partials[offset + 1] = total
            continue
        end
        for slot in 1:labels
            @inbounds partials[offset + slot] = zero(T)
        end
        for item in _reduction_range(chunk, number_of_items)
            slot = @inbounds slots[item]
            iszero(slot) && continue
            @inbounds partials[offset + slot] += _slot_term(mode, terms, weights, item)
        end
    end
    for chunk in 1:chunks
        offset = (chunk - 1) * chunk_stride
        for slot in 1:labels
            @inbounds sums[slot] += partials[offset + slot]
        end
    end
    return sums
end

"""Traverse block-backed cell data with one dense column range per work item.

Serial execution (one thread, or fewer cells than the grain) visits each
block once, allowing callers to hoist its dense arrays outside the local
`j, i` loops. With several threads and fewer than `_BATCHES_PER_THREAD` leaf
blocks per thread, the columns of each block are split into tiles so that a
one-block uniform mesh still uses all threads.

`f(block, first_j, last_j)` must own every output in the supplied column range.
A check whose outcome depends on the data records its result in such an output,
one slot per column, and the caller throws after the launch.
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

"""
    _parallel_copyto!(destination, destination_offset, source, source_offset, number_of_items)

Set `destination[destination_offset + k] = source[source_offset + k]` for
`k in 1:number_of_items` (offsets are shifts, `0` starts at the first
element), threaded above the grain. The ranges are bounds-checked before the
launch and must not overlap in memory: each `k` owns one output.
"""
function _parallel_copyto!(
        destination::AbstractVector, destination_offset::Integer,
        source::AbstractVector, source_offset::Integer, number_of_items::Integer,
    )
    number_of_items >= 0 || throw(ArgumentError("copy length must be nonnegative"))
    iszero(number_of_items) && return destination
    checkbounds(destination, (destination_offset + 1):(destination_offset + number_of_items))
    checkbounds(source, (source_offset + 1):(source_offset + number_of_items))
    @_parallel for index in 1:number_of_items
        @inbounds destination[destination_offset + index] = source[source_offset + index]
    end
    return destination
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
