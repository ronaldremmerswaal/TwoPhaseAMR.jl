@inline function _fine_location(global_index, n)
    child = (global_index - 1) ÷ n + 1
    local_index = global_index - (child - 1) * n
    return child, local_index
end

"""
    restrict!(parent, children, halo)

Conservatively restrict four dyadic child-block cell averages into their
parent. Children are ordered `(low-low, high-low, low-high, high-high)`.
Only block interiors are written.
"""
function restrict!(parent::AbstractMatrix, children::NTuple{4, <:AbstractMatrix}, halo::Integer)
    nx = size(parent, 1) - 2halo
    ny = size(parent, 2) - 2halo
    halo >= 0 || throw(ArgumentError("halo must be nonnegative"))
    nx > 0 && ny > 0 && iseven(nx) && iseven(ny) ||
        throw(DimensionMismatch("parent interior dimensions must be positive and even"))
    for child in children
        size(child) == size(parent) || throw(DimensionMismatch("parent and child block shapes must match"))
    end
    _parallel_columns(nx, ny) do first_j, last_j
        @inbounds for j in first_j:last_j, i in 1:nx
            value = zero(eltype(parent))
            for fj in (2j - 1):(2j), fi in (2i - 1):(2i)
                cx, ii = _fine_location(fi, nx)
                cy, jj = _fine_location(fj, ny)
                child_index = cx + 2(cy - 1)
                value += children[child_index][ii + halo, jj + halo]
            end
            parent[i + halo, j + halo] = value / 4
        end
    end
    return parent
end

"""
    prolong!(children, parent, halo)

Conservatively prolong parent cell averages to four dyadic children with a
limited piecewise-linear reconstruction. The four subcell values belonging to
one parent cell have exactly the parent average. Centred slopes are clipped by
the monotonized-central limiter; the construction is second-order for smooth
data and falls back to first order at extrema or when `halo == 0`. The child
ordering is the same as for [`restrict!`](@ref).
"""
function prolong!(children::NTuple{4, <:AbstractMatrix}, parent::AbstractMatrix, halo::Integer)
    nx = size(parent, 1) - 2halo
    ny = size(parent, 2) - 2halo
    halo >= 0 || throw(ArgumentError("halo must be nonnegative"))
    nx > 0 && ny > 0 && iseven(nx) && iseven(ny) ||
        throw(DimensionMismatch("parent interior dimensions must be positive and even"))
    for child in children
        size(child) == size(parent) || throw(DimensionMismatch("parent and child block shapes must match"))
    end
    _parallel_columns(2nx, 2ny) do first_fj, last_fj
        @inbounds for fj in first_fj:last_fj, fi in 1:(2nx)
            cx, ii = _fine_location(fi, nx)
            cy, jj = _fine_location(fj, ny)
            child_index = cx + 2(cy - 1)
            pi = (fi + 1) ÷ 2
            pj = (fj + 1) ÷ 2
            value = parent[pi + halo, pj + halo]
            slope_x = halo == 0 ? zero(value) : _mc_slope(
                    value - parent[pi + halo - 1, pj + halo],
                    parent[pi + halo + 1, pj + halo] - value
                )
            slope_y = halo == 0 ? zero(value) : _mc_slope(
                    value - parent[pi + halo, pj + halo - 1],
                    parent[pi + halo, pj + halo + 1] - value
                )
            offset_x = isodd(fi) ? -one(value) / 4 : one(value) / 4
            offset_y = isodd(fj) ? -one(value) / 4 : one(value) / 4
            children[child_index][ii + halo, jj + halo] =
                value + offset_x * slope_x + offset_y * slope_y
        end
    end
    return children
end

@inline function _mc_slope(left_difference, right_difference)
    left_difference * right_difference <= zero(left_difference) &&
        return zero(
        promote_type(
            typeof(left_difference),
            typeof(right_difference)
        )
    )
    centred = (left_difference + right_difference) / 2
    magnitude = min(
        abs(centred),
        2abs(left_difference), 2abs(right_difference)
    )
    return copysign(magnitude, centred)
end
