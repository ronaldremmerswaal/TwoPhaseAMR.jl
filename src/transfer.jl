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
function restrict!(parent::AbstractMatrix, children::NTuple{4,<:AbstractMatrix}, halo::Integer)
    nx = size(parent, 1) - 2halo
    ny = size(parent, 2) - 2halo
    size(parent, 1) == nx + 2halo && size(parent, 2) == ny + 2halo ||
        throw(DimensionMismatch("invalid parent halo shape"))
    for child in children
        size(child) == size(parent) || throw(DimensionMismatch("parent and child block shapes must match"))
    end
    for j in 1:ny, i in 1:nx
        value = zero(eltype(parent))
        for fj in (2j - 1):(2j), fi in (2i - 1):(2i)
            cx, ii = _fine_location(fi, nx)
            cy, jj = _fine_location(fj, ny)
            child_index = cx + 2(cy - 1)
            @inbounds value += children[child_index][ii + halo, jj + halo]
        end
        @inbounds parent[i + halo, j + halo] = value / 4
    end
    return parent
end

"""
    prolong!(children, parent, halo)

Conservatively prolong parent averages piecewise-constantly to four dyadic
children. The child ordering is the same as for [`restrict!`](@ref).
"""
function prolong!(children::NTuple{4,<:AbstractMatrix}, parent::AbstractMatrix, halo::Integer)
    nx = size(parent, 1) - 2halo
    ny = size(parent, 2) - 2halo
    for child in children
        size(child) == size(parent) || throw(DimensionMismatch("parent and child block shapes must match"))
    end
    for fj in 1:(2ny), fi in 1:(2nx)
        cx, ii = _fine_location(fi, nx)
        cy, jj = _fine_location(fj, ny)
        child_index = cx + 2(cy - 1)
        pi = (fi + 1) ÷ 2
        pj = (fj + 1) ÷ 2
        @inbounds children[child_index][ii + halo, jj + halo] = parent[pi + halo, pj + halo]
    end
    return children
end
