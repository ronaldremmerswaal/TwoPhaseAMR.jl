# Cut-cell polygons and the mimetic (MFD) kinetic metric of `MimeticProjection`.
#
# Notation follows docs/techref/chapters/10_projection.tex, section "Mimetic
# cut-cell projection": a cell side k = 1, 2, 3, 4 is its bottom, right, top
# and left side (counter-clockwise), the liquid aperture of a side is the
# package's face aperture, and the chord of a cut cell joins the ends of the
# phase parts of its sides.

# Outward normal of side k as a multiple of the face orientation (+axis).
const _MIMETIC_OUTWARD = (-1, 1, 1, -1)
# A counter-clockwise walk starts at the low tangential end of sides 1 and 2
# and at the high end of sides 3 and 4.
const _MIMETIC_STARTS_LOW = (true, true, false, false)

# Face-side flags of a partially wetted interior face.
const _MIMETIC_LIQUID_LOW = Int8(1)
const _MIMETIC_LIQUID_HIGH = Int8(-1)
const _MIMETIC_SIDE_INCONSISTENT = Int8(2)
const _MIMETIC_SIDE_UNKNOWN = Int8(3)

"""Sides of the cells of a uniform square-cell mesh.

`sides[k, cell]` is the interior face index (positive) or minus the boundary
face index (negative) of side `k`; `face_cells[:, face]` are the minus and plus
cells of an interior face.
"""
struct _MimeticTopology{T}
    spacing::T
    sides::Matrix{Int}
    face_cells::Matrix{Int}
    identity::Base.RefValue{Nothing}
    generation::UInt
end

"""Return the cell size of a uniform square-cell mesh, or throw."""
function _mimetic_uniform_spacing(mesh)
    isempty(mesh.blocks) && throw(ArgumentError("MimeticProjection requires a nonempty mesh"))
    spacing = first(mesh.blocks).spacing
    for block in mesh.blocks
        block.spacing == spacing || throw(
            ArgumentError(
                "MimeticProjection supports only uniform meshes; this mesh has " *
                    "leaf blocks on different levels (hanging faces)",
            ),
        )
    end
    T = eltype(spacing)
    abs(spacing[1] - spacing[2]) <= 64eps(T) * max(spacing...) || throw(
        ArgumentError("MimeticProjection requires square cells"),
    )
    return spacing[1]
end

function _mimetic_topology(mesh)
    h = _mimetic_uniform_spacing(mesh)
    nx, ny = mesh.block_size
    cells = length(mesh.blocks) * nx * ny
    sides = zeros(Int, 4, cells)
    face_cells = zeros(Int, 2, length(mesh.faces))
    for (index, face) in pairs(mesh.faces)
        minus = _cell_index(face.minus, nx, ny)
        plus = _cell_index(face.plus, nx, ny)
        face_cells[1, index] = minus
        face_cells[2, index] = plus
        if face.axis == 1
            sides[2, minus] = index
            sides[4, plus] = index
        else
            sides[3, minus] = index
            sides[1, plus] = index
        end
    end
    for (index, face) in pairs(mesh.boundary_faces)
        cell = _cell_index(face.cell, nx, ny)
        side = face.axis == 1 ? (face.side > 0 ? 2 : 4) : (face.side > 0 ? 3 : 1)
        sides[side, cell] = -index
    end
    any(iszero, sides) && throw(
        ArgumentError("MimeticProjection requires one face per cell side"),
    )
    cache = mesh.topology_cache
    return _MimeticTopology(h, sides, face_cells, cache.identity, cache.generation)
end

_mimetic_topology_current(topology, mesh) =
    !isnothing(topology) && topology.identity === mesh.topology_cache.identity &&
    topology.generation == mesh.topology_cache.generation

"""Cell across side `side` of `cell`, or zero at a domain boundary."""
@inline function _mimetic_neighbour(topology, cell, side)
    face = topology.sides[side, cell]
    face > 0 || return 0
    minus = topology.face_cells[1, face]
    return minus == cell ? topology.face_cells[2, face] : minus
end

# Same-axis interior faces near `face` in the order of the donor search
# (thesis/audit laboratory): first the faces across its two cells, then the
# collinear faces shifted by one cell along the face. Zero entries are absent.
function _mimetic_face_ring(topology, face, axis)
    minus = topology.face_cells[1, face]
    plus = topology.face_cells[2, face]
    sides = topology.sides
    if axis == 1
        below = _mimetic_neighbour(topology, plus, 1)
        above = _mimetic_neighbour(topology, plus, 3)
        return (
            max(sides[4, minus], 0), max(sides[2, plus], 0),
            below == 0 ? 0 : max(sides[4, below], 0),
            above == 0 ? 0 : max(sides[4, above], 0),
        )
    end
    left = _mimetic_neighbour(topology, plus, 4)
    right = _mimetic_neighbour(topology, plus, 2)
    return (
        max(sides[1, minus], 0), max(sides[3, plus], 0),
        left == 0 ? 0 : max(sides[1, left], 0),
        right == 0 ? 0 : max(sides[1, right], 0),
    )
end

# Sign of the tangential PLIC normal of `cell` on a face with local normal
# coordinate `normal_coordinate`, and whether the PLIC line cuts the face
# strictly inside. A positive sign puts the liquid at the low end.
function _mimetic_trace_vote(interface, reference, axis, normal_coordinate)
    nx = _get(interface.normal_x, reference)
    ny = _get(interface.normal_y, reference)
    intercept = _get(interface.intercept, reference)
    normal, tangent = axis == 1 ? (nx, ny) : (ny, nx)
    T = typeof(intercept)
    lo, hi = _clip_line_interval(
        -T(0.5), T(0.5), tangent, normal * normal_coordinate - intercept;
        keep_less = true,
    )
    vote = tangent > 0 ? 1 : (tangent < 0 ? -1 : 0)
    return vote, zero(T) < hi - lo < one(T)
end

"""Classify the wetted end of every partially wetted interior face.

The adjacent cut cells whose PLIC line crosses the face strictly inside decide;
if there is none, a cut neighbour whose line misses the face decides by the
direction of its tangential normal. Disagreement or no cut neighbour leaves
the face unresolved.
"""
function _mimetic_face_sides!(flags, interface, liquid_aperture)
    mesh = interface.mesh
    resize!(flags, length(mesh.faces))
    for (index, face) in pairs(mesh.faces)
        aperture = liquid_aperture[index]
        if !(zero(aperture) < aperture < one(aperture))
            flags[index] = Int8(0)
            continue
        end
        strong = 0
        weak = 0
        inconsistent = false
        for (reference, coordinate) in ((face.minus, 0.5), (face.plus, -0.5))
            fraction = _conditional_fraction(interface, reference)
            zero(fraction) < fraction < one(fraction) || continue
            vote, crosses = _mimetic_trace_vote(
                interface, reference, face.axis, coordinate,
            )
            if crosses
                inconsistent |= strong != 0 && strong != vote
                strong = vote
            elseif weak == 0
                weak = vote
            end
        end
        decision = strong != 0 ? strong : weak
        flags[index] = inconsistent ? _MIMETIC_SIDE_INCONSISTENT :
            decision > 0 ? _MIMETIC_LIQUID_LOW :
            decision < 0 ? _MIMETIC_LIQUID_HIGH : _MIMETIC_SIDE_UNKNOWN
    end
    return flags
end

"""One edge of a phase polygon in unit cell coordinates; side 0 is the chord."""
struct _MimeticEdge{T}
    x1::T
    y1::T
    x2::T
    y2::T
    side::Int
end

@inline function _mimetic_corner(::Type{T}, k) where {T}
    half = T(0.5)
    k == 1 && return (-half, -half)
    k == 2 && return (half, -half)
    k == 3 && return (half, half)
    return (-half, half)
end

"""Build the polygon of one phase from the phase parts of the four sides.

`fractions[k]` is the phase aperture of side `k` and `starts[k]` whether the
phase part begins at the counter-clockwise start of a partially wetted side.
Consecutive parts whose ends differ are joined by a chord. Returns the number
of chords.
"""
function _mimetic_phase_polygon!(edges, segments, fractions, starts)
    T = eltype(fractions)
    empty!(segments)
    for k in 1:4
        b = fractions[k]
        iszero(b) && continue
        ax, ay = _mimetic_corner(T, k)
        bx, by = _mimetic_corner(T, mod1(k + 1, 4))
        if b == one(T)
            push!(segments, _MimeticEdge(ax, ay, bx, by, k))
        elseif starts[k]
            push!(segments, _MimeticEdge(ax, ay, ax + b * (bx - ax), ay + b * (by - ay), k))
        else
            push!(segments, _MimeticEdge(bx - b * (bx - ax), by - b * (by - ay), bx, by, k))
        end
    end
    empty!(edges)
    chords = 0
    count = length(segments)
    for q in 1:count
        current = segments[q]
        next = segments[mod1(q + 1, count)]
        push!(edges, current)
        if hypot(current.x2 - next.x1, current.y2 - next.y1) > T(1.0e-12)
            push!(edges, _MimeticEdge(current.x2, current.y2, next.x1, next.y1, 0))
            chords += 1
        end
    end
    return chords
end

"""Area and centroid of a closed polygon given by its edges."""
function _mimetic_polygon_moments(edges)
    T = eltype(first(edges).x1)
    area = cx = cy = zero(T)
    for edge in edges
        cross = edge.x1 * edge.y2 - edge.x2 * edge.y1
        area += cross / 2
        cx += (edge.x1 + edge.x2) * cross / 6
        cy += (edge.y1 + edge.y2) * cross / 6
    end
    return area, cx / area, cy / area
end

"""Mimetic inner product of one polygon in unit coordinates (eq. mm-local).

Fills `block[1:m, 1:m]` with `R R'/|P| + (1/2) D Q D`, where `R` has rows
`|e| (x_e - x_P)'`, `D = diag(|e|)` and `Q` is the orthogonal projector onto
the complement of the range of `D N` (`N`: outward unit normals).
"""
function _mfd_polygon_block!(block, lengths, edges, area, cx, cy)
    T = eltype(block)
    m = length(edges)
    g11 = g12 = g22 = zero(T)
    for q in 1:m
        edge = edges[q]
        lengths[q] = hypot(edge.x2 - edge.x1, edge.y2 - edge.y1)
        # D N row: |e| n_e with n_e = (t_y, -t_x) for a counter-clockwise walk.
        fx = edge.y2 - edge.y1
        fy = edge.x1 - edge.x2
        g11 += fx * fx
        g12 += fx * fy
        g22 += fy * fy
    end
    determinant = g11 * g22 - g12 * g12
    i11, i12, i22 = g22 / determinant, -g12 / determinant, g11 / determinant
    for q in 1:m, r in 1:m
        eq, er = edges[q], edges[r]
        fqx, fqy = eq.y2 - eq.y1, eq.x1 - eq.x2
        frx, fry = er.y2 - er.y1, er.x1 - er.x2
        projector = (q == r ? one(T) : zero(T)) -
            (fqx * (i11 * frx + i12 * fry) + fqy * (i12 * frx + i22 * fry))
        rqx = lengths[q] * ((eq.x1 + eq.x2) / 2 - cx)
        rqy = lengths[q] * ((eq.y1 + eq.y2) / 2 - cy)
        rrx = lengths[r] * ((er.x1 + er.x2) / 2 - cx)
        rry = lengths[r] * ((er.y1 + er.y2) / 2 - cy)
        block[q, r] = (rqx * rrx + rqy * rry) / area +
            lengths[q] * projector * lengths[r] / 2
    end
    return block
end

"""Degrees of freedom and data of one assembled mimetic metric."""
mutable struct _MimeticAssembly{T}
    revision::UInt
    liquid_dof::Vector{Int}
    gas_dof::Vector{Int}
    dof_face::Vector{Int}
    dof_phase::Vector{Int8}
    dof_aperture::Vector{T}
    dof_mass::Vector{T}
    dof_liquid_weight::Vector{T}
    gridline_faces::Int
    band::Vector{Int}
    band_position::Vector{Int}
    diagonal::Vector{T}
    band_matrix::SparseMatrixCSC{T, Int}
    factor::Any
    pressure_x::Vector{T}
    pressure_y::Vector{T}
    cut_cells::Int
    slaved_values::Int
    stabilised_values::Int
    lumped_values::Int
    lumped_polygons::Int
    full_diagonal::Vector{T}
    rows::Vector{Int}
    columns::Vector{Int}
    values::Vector{T}
    touched::BitVector
    flags::Vector{Int8}
    edges::Vector{_MimeticEdge{T}}
    segments::Vector{_MimeticEdge{T}}
    block::Matrix{T}
    lengths::Vector{T}
end

function _MimeticAssembly(::Type{T}) where {T}
    return _MimeticAssembly{T}(
        typemax(UInt), Int[], Int[], Int[], Int8[], T[], T[], T[], 0, Int[], Int[], T[],
        spzeros(T, 0, 0), nothing, T[], T[], 0, 0, 0, 0, 0, T[], Int[], Int[], T[],
        falses(0), Int8[], _MimeticEdge{T}[], _MimeticEdge{T}[], zeros(T, 8, 8), zeros(T, 8),
    )
end

@inline function _mimetic_add!(assembly, row, column, value)
    push!(assembly.rows, row)
    push!(assembly.columns, column)
    push!(assembly.values, value)
    assembly.touched[row] = true
    assembly.touched[column] = true
    return nothing
end

# Quadratic form weight * (u_row - u_donor)^2.
@inline function _mimetic_add_tie!(assembly, row, donor, weight)
    _mimetic_add!(assembly, row, row, weight)
    _mimetic_add!(assembly, donor, donor, weight)
    _mimetic_add!(assembly, row, donor, -weight)
    _mimetic_add!(assembly, donor, row, -weight)
    return nothing
end

function _mimetic_nonsimple(metric, cell, reason)
    metric.nonsimple_cells === :lumped && return true
    throw(
        ArgumentError(
            "MimeticProjection: cell $cell is not a simple cut cell ($reason); " *
                "the mimetic metric is defined for one chord per cut cell only " *
                "(MFDCutCellMetric(nonsimple_cells = :lumped) selects the " *
                "laboratory's inconsistent lumped fallback)",
        ),
    )
end

"""Assemble the MFD cut-cell metric (thm. mm of the audit, its laboratory rules).

Polygon blocks enter the face values of their phase; a phase value on a face
that the phase does not reach is tied to a donor with its staggered mass, and
with the small-cell stabilisation every value with aperture below one half on a
mixed face is tied to a donor with weight `(1 - 2a)` times its mass.

The geometry enters only through `data` (staggered liquid fractions, phase
apertures and masses of the interior faces), the wet-end `flags` of the
interior faces and the liquid apertures and wet ends (`true`: low end) of
the boundary faces.

`gridline[f]` marks an interior face between a pure liquid and a pure gas cell
(`1`: liquid minus cell, `-1`: liquid plus cell, `0`: neither): the interface
lies on the face, no cell is cut. Both cells are treated as uncut and the two
phase values of the face are identified (normal continuity `u^l = u^g`), the
limit of the cut-cell construction as the chord approaches the face. Its one
value carries the summed staggered mass and the full aperture.
"""
function _assemble_cut_cell_metric!(
        assembly::_MimeticAssembly{T}, metric::MFDCutCellMetric, mesh,
        topology, data, flags, boundary_aperture, boundary_low, density;
        allow_wall_cuts = false, gridline = nothing,
    ) where {T}
    h = topology.spacing
    faces = length(mesh.faces)
    cells = size(topology.sides, 2)
    on_gridline(face) = !isnothing(gridline) && !iszero(gridline[face])
    # Phase degrees of freedom: every phase value on a face with phase volume.
    resize!(assembly.liquid_dof, faces)
    resize!(assembly.gas_dof, faces)
    empty!(assembly.dof_face)
    empty!(assembly.dof_phase)
    empty!(assembly.dof_aperture)
    empty!(assembly.dof_mass)
    empty!(assembly.dof_liquid_weight)
    assembly.gridline_faces = 0
    for face in 1:faces
        fraction = data.liquid_fraction[face]
        volume = mesh.face_dual_volumes[face]
        assembly.liquid_dof[face] = 0
        assembly.gas_dof[face] = 0
        if on_gridline(face)
            liquid, gas = data.liquid_mass[face], data.gas_mass[face]
            push!(assembly.dof_face, face)
            push!(assembly.dof_phase, Int8(3))
            push!(assembly.dof_aperture, one(T))
            push!(assembly.dof_mass, volume * (liquid + gas))
            push!(assembly.dof_liquid_weight, liquid / (liquid + gas))
            assembly.liquid_dof[face] = assembly.gas_dof[face] = length(assembly.dof_face)
            assembly.gridline_faces += 1
            continue
        end
        if fraction > zero(T)
            push!(assembly.dof_face, face)
            push!(assembly.dof_phase, Int8(1))
            push!(assembly.dof_aperture, data.liquid_aperture[face])
            push!(assembly.dof_mass, volume * data.liquid_mass[face])
            push!(assembly.dof_liquid_weight, one(T))
            assembly.liquid_dof[face] = length(assembly.dof_face)
        end
        if fraction < one(T)
            push!(assembly.dof_face, face)
            push!(assembly.dof_phase, Int8(2))
            push!(assembly.dof_aperture, data.gas_aperture[face])
            push!(assembly.dof_mass, volume * data.gas_mass[face])
            push!(assembly.dof_liquid_weight, zero(T))
            assembly.gas_dof[face] = length(assembly.dof_face)
        end
    end
    dofs = length(assembly.dof_face)
    empty!(assembly.rows)
    empty!(assembly.columns)
    empty!(assembly.values)
    resize!(assembly.full_diagonal, dofs)
    fill!(assembly.full_diagonal, zero(T))
    resize!(assembly.touched, dofs)
    fill!(assembly.touched, false)
    resize!(assembly.pressure_x, cells)
    resize!(assembly.pressure_y, cells)
    nx, ny = mesh.block_size
    cut_cells = lumped_polygons = 0
    for cell in 1:cells
        block_index = div(cell - 1, nx * ny) + 1
        local_index = rem(cell - 1, nx * ny)
        i, j = rem(local_index, nx) + 1, div(local_index, nx) + 1
        center = cellcenter(mesh, block_index, i, j)
        assembly.pressure_x[cell] = center[1]
        assembly.pressure_y[cell] = center[2]
        apertures = ntuple(Val(4)) do k
            face = topology.sides[k, cell]
            face > 0 || return boundary_aperture[-face]
            # A grid-line interface face is fully wet from its own cell's side.
            on_gridline(face) || return data.liquid_aperture[face]
            minus = topology.face_cells[1, face] == cell
            return (gridline[face] > 0) == minus ? one(T) : zero(T)
        end
        unresolved = false
        for k in 1:4
            face = topology.sides[k, cell]
            face > 0 && zero(T) < apertures[k] < one(T) || continue
            unresolved |= flags[face] == _MIMETIC_SIDE_INCONSISTENT ||
                flags[face] == _MIMETIC_SIDE_UNKNOWN
        end
        starts = ntuple(Val(4)) do k
            zero(T) < apertures[k] < one(T) || return false
            face = topology.sides[k, cell]
            low = face > 0 ? flags[face] == _MIMETIC_LIQUID_LOW : boundary_low[-face]
            return _MIMETIC_STARTS_LOW[k] ? low : !low
        end
        lumped = unresolved && _mimetic_nonsimple(
            metric, cell, "its PLIC face traces do not determine which end of a face is wet",
        )
        cell_cut = false
        for phase in 1:2
            rho = density[phase]
            fractions = phase == 1 ? apertures : map(a -> one(T) - a, apertures)
            pstarts = phase == 1 ? starts : map(!, starts)
            chords = _mimetic_phase_polygon!(
                assembly.edges, assembly.segments, fractions, pstarts,
            )
            isempty(assembly.segments) && continue
            dof_of(k) = (
                face = topology.sides[k, cell]; face <= 0 ? 0 :
                    (phase == 1 ? assembly.liquid_dof[face] : assembly.gas_dof[face])
            )
            if chords == 0 && !lumped
                # Uncut cell: (1/2) rho h^2 on every face value, the diagonal metric.
                for k in 1:4
                    dof = dof_of(k)
                    dof > 0 && (assembly.full_diagonal[dof] += rho * h^2 / 2)
                end
                continue
            end
            area, cx, cy = chords == 0 ? (one(T), zero(T), zero(T)) :
                _mimetic_polygon_moments(assembly.edges)
            simple = !lumped && chords == 1 && length(assembly.edges) >= 3 &&
                area >= T(1.0e-12)
            if !simple
                lumped || _mimetic_nonsimple(
                    metric, cell, chords > 1 ? "$chords chords in one phase polygon" :
                        "degenerate phase polygon of area $(area)h^2",
                )
                lumped_polygons += 1
                for segment in assembly.segments
                    dof = dof_of(segment.side)
                    dof > 0 && _mimetic_add!(assembly, dof, dof, rho * h^2 / 2)
                end
                continue
            end
            if !allow_wall_cuts && any(e -> e.side > 0 && topology.sides[e.side, cell] < 0, assembly.edges)
                throw(
                    ArgumentError(
                        "MimeticProjection: cut cell $cell touches a domain boundary; " *
                            "the mimetic metric is consistent only for cut cells away " *
                            "from walls",
                    ),
                )
            end
            cell_cut = true
            m = length(assembly.edges)
            m <= size(assembly.block, 1) || (
                assembly.block = zeros(T, m, m);
                resize!(assembly.lengths, m)
            )
            _mfd_polygon_block!(assembly.block, assembly.lengths, assembly.edges, area, cx, cy)
            chord = findfirst(e -> e.side == 0, assembly.edges)
            chord_edge = assembly.edges[chord]
            assembly.pressure_x[cell] = center[1] + h * (chord_edge.x1 + chord_edge.x2) / 2
            assembly.pressure_y[cell] = center[2] + h * (chord_edge.y1 + chord_edge.y2) / 2
            lengths = assembly.lengths
            block = assembly.block
            scale = rho * h^2
            for q in 1:m
                kq = assembly.edges[q].side
                kq > 0 || continue
                rq = dof_of(kq)
                if rq == 0
                    topology.sides[kq, cell] < 0 && continue
                    error("mimetic metric: wetted face without a phase value")
                end
                sq = lengths[q] / lengths[chord]
                for r in 1:m
                    kr = assembly.edges[r].side
                    kr > 0 || continue
                    rr = dof_of(kr)
                    rr > 0 || continue
                    sr = lengths[r] / lengths[chord]
                    # Slaved chord flux (eq. mm-slave): v_G = -sum |e| v_e / |G|.
                    value = block[q, r] - sr * block[q, chord] - sq * block[chord, r] +
                        sq * sr * block[chord, chord]
                    _mimetic_add!(
                        assembly, rq, rr,
                        _MIMETIC_OUTWARD[kq] * _MIMETIC_OUTWARD[kr] * scale * value,
                    )
                end
            end
        end
        cut_cells += cell_cut
    end
    # Additions on the face values (stability parts of the metric).
    slaved = stabilised = lumped_values = 0
    polygon_diagonal = resize!(assembly.diagonal, dofs)
    copyto!(polygon_diagonal, assembly.full_diagonal)
    for index in eachindex(assembly.rows)
        assembly.rows[index] == assembly.columns[index] &&
            (polygon_diagonal[assembly.rows[index]] += assembly.values[index])
    end
    reaches(dof) = dof > 0 && assembly.dof_aperture[dof] > zero(T)
    for dof in 1:dofs
        face = assembly.dof_face[dof]
        phase = assembly.dof_phase[dof]
        aperture = assembly.dof_aperture[dof]
        mass = assembly.dof_mass[dof]
        dof_at(g) = g == 0 ? 0 : (phase == 1 ? assembly.liquid_dof[g] : assembly.gas_dof[g])
        axis = mesh.faces[face].axis
        ring = _mimetic_face_ring(topology, face, axis)
        if aperture > zero(T)
            if !(polygon_diagonal[dof] > zero(T))
                _mimetic_add!(assembly, dof, dof, mass)
                lumped_values += 1
            end
            fraction = data.liquid_fraction[face]
            (
                metric.small_cell_stabilization && zero(T) < fraction < one(T) &&
                    aperture < T(0.5)
            ) || continue
            larger(g) = (
                d = dof_at(g); reaches(d) &&
                    assembly.dof_aperture[d] > aperture + T(1.0e-12)
            )
            donor = _mimetic_donor(topology, mesh, ring, dof_at, larger, assembly.dof_aperture)
            donor == 0 && continue
            _mimetic_add_tie!(assembly, dof, donor, (one(T) - 2aperture) * mass)
            stabilised += 1
            continue
        end
        donor = _mimetic_donor(topology, mesh, ring, dof_at, g -> reaches(dof_at(g)), assembly.dof_aperture)
        if donor == 0
            _mimetic_add!(assembly, dof, dof, mass)
            lumped_values += 1
        else
            _mimetic_add_tie!(assembly, dof, donor, mass)
            slaved += 1
        end
    end
    # Band: every value with a non-diagonal or non-uncut contribution.
    band = assembly.band
    empty!(band)
    resize!(assembly.band_position, dofs)
    fill!(assembly.band_position, 0)
    for dof in 1:dofs
        if assembly.touched[dof]
            push!(band, dof)
            assembly.band_position[dof] = length(band)
        end
    end
    resize!(assembly.diagonal, dofs)
    for dof in 1:dofs
        if assembly.touched[dof]
            assembly.diagonal[dof] = zero(T)
            assembly.full_diagonal[dof] > zero(T) && _mimetic_add!(
                assembly, dof, dof, assembly.full_diagonal[dof],
            )
        else
            assembly.diagonal[dof] = assembly.full_diagonal[dof]
            assembly.diagonal[dof] > zero(T) || throw(
                ArgumentError("MimeticProjection: phase value $dof has no metric"),
            )
        end
    end
    positions = assembly.band_position
    count = length(band)
    for index in eachindex(assembly.rows)
        assembly.rows[index] = positions[assembly.rows[index]]
        assembly.columns[index] = positions[assembly.columns[index]]
    end
    assembly.band_matrix = sparse(assembly.rows, assembly.columns, assembly.values, count, count)
    assembly.cut_cells = cut_cells
    assembly.slaved_values = slaved
    assembly.stabilised_values = stabilised
    assembly.lumped_values = lumped_values
    assembly.lumped_polygons = lumped_polygons
    return assembly
end

"""Donor of a tie: the neighbour of the ring (then of the second ring) whose
phase aperture is largest among those satisfying `admissible`; zero if none."""
function _mimetic_donor(topology, mesh, ring, dof_at, admissible, apertures)
    best = 0
    best_aperture = -one(eltype(apertures))
    for g in ring
        g == 0 && continue
        admissible(g) || continue
        d = dof_at(g)
        if apertures[d] > best_aperture
            best, best_aperture = d, apertures[d]
        end
    end
    best == 0 || return best
    for g0 in ring
        g0 == 0 && continue
        for g in _mimetic_face_ring(topology, g0, mesh.faces[g0].axis)
            g == 0 && continue
            admissible(g) || continue
            d = dof_at(g)
            if apertures[d] > best_aperture
                best, best_aperture = d, apertures[d]
            end
        end
    end
    return best
end
