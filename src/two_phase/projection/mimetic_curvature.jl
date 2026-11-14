# Conservative height-function (CHF) curvature of the mimetic projection's
# capillary force: the divergence form -(S_{i+½} - S_{i-½})/h of the local
# height functions, the exact derivative of the height-polyline length inside
# a patch of valid columns. See docs/techref/chapters/08_forces.tex,
# "Curvature of the mimetic projection", and the audit note "Surface tension
# for the two-velocity mimetic projection" (sec. chf).

# Half width of a height window: seven cells along the column.
const _MIMETIC_HALF_WINDOW = 3

"""Lattice data of the CHF curvature on the uniform mesh of a mimetic
projection: `cell[I, J]` is the cell index of lattice cell `(I, J)`, `fraction`
the liquid fractions and `orientation[I, J]` `0` for a pure cell and `±1`,
`±2` for a cut cell (heights along x or y, gas towards higher or lower
positions)."""
mutable struct _MimeticCurvature{T}
    topology::Any
    lattice::NTuple{2, Int}
    periodic::NTuple{2, Bool}
    cell::Matrix{Int}
    fraction::Matrix{T}
    orientation::Matrix{Int8}
end

_MimeticCurvature(::Type{T}) where {T} = _MimeticCurvature{T}(
    nothing, (0, 0), (false, false), zeros(Int, 0, 0), zeros(T, 0, 0), zeros(Int8, 0, 0),
)

"""Index of position `k` on a lattice line of `n` cells: wrapped if
`periodic`, else `k` inside the line and `0` beyond a wall."""
@inline _mimetic_wrap(k, n, periodic) = periodic ? mod1(k, n) : (1 <= k <= n ? k : 0)

"""Lattice cell of column `column`, position `position` for orientation `o`
(`2`: heights along y, columns at fixed x; `1`: heights along x)."""
@inline _mimetic_lattice_cell(o, column, position) = o == 2 ? (column, position) : (position, column)

"""Rebuild the lattice of the uniform mesh after a topology change."""
function _mimetic_curvature_lattice!(curvature::_MimeticCurvature{T}, mesh, topology) where {T}
    curvature.topology === topology && return curvature
    level = first(mesh.blocks).key.level
    lattice = Tuple(_cells_at_level(mesh, level))
    nx, ny = mesh.block_size
    curvature.lattice = lattice
    curvature.periodic = mesh.periodic
    curvature.cell = zeros(Int, lattice)
    for (block, entry) in pairs(mesh.blocks), j in 1:ny, i in 1:nx
        curvature.cell[entry.key.i * nx + i, entry.key.j * ny + j] =
            _cell_index(CellRef(block, i, j), nx, ny)
    end
    any(iszero, curvature.cell) && throw(
        ArgumentError("MimeticProjection: the leaf blocks do not tile the uniform lattice"),
    )
    curvature.fraction = zeros(T, lattice)
    curvature.orientation = zeros(Int8, lattice)
    curvature.topology = topology
    return curvature
end

"""Copy the liquid fractions of the cells to the lattice and give every cut
cell its orientation: heights along y if `|n_y| ≥ |n_x|` for the PLIC normal
`n` (outward from the liquid), else along x, signed by the gas side."""
function _mimetic_curvature_data!(curvature::_MimeticCurvature{T}, interface, references) where {T}
    fraction, orientation = curvature.fraction, curvature.orientation
    for index in eachindex(curvature.cell)
        reference = references[curvature.cell[index]]
        c = _conditional_fraction(interface, reference)
        fraction[index] = c
        if zero(T) < c < one(T)
            nx = _get(interface.normal_x, reference)
            ny = _get(interface.normal_y, reference)
            orientation[index] = abs(ny) >= abs(nx) ? (ny > 0 ? Int8(2) : Int8(-2)) :
                (nx > 0 ? Int8(1) : Int8(-1))
        else
            orientation[index] = Int8(0)
        end
    end
    return curvature
end

"""Write the CHF curvature to `kappa[cell]` for every cut cell whose
three-column window is valid; other entries are left unchanged.

For a cut cell of orientation `o` in column `a` at position `r`, the window of
each column `q ∈ {a-1, a, a+1}` is `r-3:r+3` along the columns; it is valid if
it lies inside the domain (wrapped in a periodic direction), its liquid end is
full and its gas end empty. The heights `H_q = Σ_r C_{q,r}` (in cells) give
`κ = -(S(H_{a+1} - H_a) - S(H_a - H_{a-1}))/h`, `S(D) = D/√(1 + D²)`: the
divergence form of the height-function curvature, which inside a patch of
valid columns is `∂L/∂V` of the height-polyline length
`L = Σ h√(1 + (H_{q+1} - H_q)²)`. Since `|S| < 1`, `|κ| < 2/h`."""
function _mimetic_chf_curvature!(kappa, workspace, interface)
    topology = workspace.topology
    curvature = _mimetic_curvature_lattice!(workspace.curvature, interface.mesh, topology)
    _mimetic_curvature_data!(curvature, interface, workspace.references)
    return _mimetic_chf_values!(kappa, curvature, topology.spacing)
end

"""The CHF curvature of the lattice data of `curvature` (cell size `h`), see
[`_mimetic_chf_curvature!`](@ref)."""
function _mimetic_chf_values!(kappa, curvature::_MimeticCurvature, h)
    T = eltype(kappa)
    lattice, periodic = curvature.lattice, curvature.periodic
    fraction = curvature.fraction
    half = _MIMETIC_HALF_WINDOW
    for J in 1:lattice[2], I in 1:lattice[1]
        code = curvature.orientation[I, J]
        iszero(code) && continue
        o = abs(code)
        column, position = o == 2 ? (I, J) : (J, I)
        columns, positions = lattice[3 - o], lattice[o]
        first = _mimetic_wrap(position - half, positions, periodic[o])
        last = _mimetic_wrap(position + half, positions, periodic[o])
        (iszero(first) || iszero(last)) && continue
        liquid, gas = code > 0 ? (first, last) : (last, first)
        valid = true
        heights = (zero(T), zero(T), zero(T))
        for (q, offset) in enumerate(-1:1)
            c = _mimetic_wrap(column + offset, columns, periodic[3 - o])
            valid = !iszero(c) &&
                isone(fraction[_mimetic_lattice_cell(o, c, liquid)...]) &&
                iszero(fraction[_mimetic_lattice_cell(o, c, gas)...])
            valid || break
            total = zero(T)
            for k in (position - half):(position + half)
                total += fraction[_mimetic_lattice_cell(o, c, _mimetic_wrap(k, positions, periodic[o]))...]
            end
            heights = Base.setindex(heights, total, q)
        end
        valid || continue
        S(D) = D / sqrt(1 + D^2)
        kappa[curvature.cell[I, J]] = -(S(heights[3] - heights[2]) - S(heights[2] - heights[1])) / h
    end
    return kappa
end
