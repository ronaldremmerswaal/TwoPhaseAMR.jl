# The mimetic (MFD) kinetic metric of `MimeticProjection` on the Gram tensors
# of the cut cells (audit note "Kinetic metrics for shared apertures that close
# no region").
#
# Notation follows docs/techref/chapters/10_projection.tex, section "Mimetic
# cut-cell projection": a cell side k = 1, 2, 3, 4 is its bottom, right, top
# and left side (counter-clockwise), the liquid aperture of a side is the
# package's face aperture, and the data of a cut cell are the wetted part of
# each side and its midpoint, shared with the neighbour across the side. No
# polygon is formed. On an adaptive (2:1 graded) mesh the side of a cell next
# to two finer cells consists of two faces, the two segments of the side. With
# an embedded solid a segment is open over the shared open aperture of its
# face, and a cell cut by the solid has more sides, its solid edges, the
# chords that close the open parts of its segments.

# Outward normal of side k as a multiple of the face orientation (+axis).
const _MIMETIC_OUTWARD = (-1, 1, 1, -1)
# A counter-clockwise walk starts at the low tangential end of sides 1 and 2
# and at the high end of sides 3 and 4.
const _MIMETIC_STARTS_LOW = (true, true, false, false)

# Face-side flags of a partially wetted interior face.
const _MIMETIC_LIQUID_LOW = Int8(1)
const _MIMETIC_LIQUID_HIGH = Int8(-1)

"""Sides of the cells of a mesh of square cells.

`sides[k, cell]` is the interior face index (positive) or minus the boundary
face index (negative) of side `k`; on a side that consists of two faces (a
cell next to two finer cells), it is the face at the low tangential end and
`second[k, cell]` the one at the high end (zero on a side of one face).
`face_cells[:, face]` are the minus and plus cells of an interior face,
`spacing[cell]` the width of a cell and `area[face]` that of a face.
"""
struct _MimeticTopology{T}
    spacing::Vector{T}
    area::Vector{T}
    sides::Matrix{Int}
    second::Matrix{Int}
    face_cells::Matrix{Int}
    identity::Base.RefValue{Nothing}
    generation::UInt
end

"""Throw unless every cell of `mesh` is square."""
function _mimetic_square_cells(mesh)
    isempty(mesh.blocks) && throw(ArgumentError("MimeticProjection requires a nonempty mesh"))
    for block in mesh.blocks
        spacing = block.spacing
        T = eltype(spacing)
        abs(spacing[1] - spacing[2]) <= 64eps(T) * max(spacing...) || throw(
            ArgumentError("MimeticProjection requires square cells"),
        )
    end
    return mesh
end

function _mimetic_topology(mesh)
    _mimetic_square_cells(mesh)
    nx, ny = mesh.block_size
    cells = length(mesh.blocks) * nx * ny
    spacing = Vector{eltype(mesh.origin)}(undef, cells)
    for (index, block) in pairs(mesh.blocks)
        fill!(view(spacing, ((index - 1) * nx * ny + 1):(index * nx * ny)), block.spacing[1])
    end
    area = [face.area for face in mesh.faces]
    sides = zeros(Int, 4, cells)
    second = zeros(Int, 4, cells)
    face_cells = zeros(Int, 2, length(mesh.faces))
    function attach!(k, cell, index)
        first = sides[k, cell]
        if iszero(first)
            sides[k, cell] = index
            return
        end
        (first > 0 && iszero(second[k, cell])) || throw(
            ArgumentError("MimeticProjection requires a 2:1 graded mesh"),
        )
        # Order the two faces of the side by their tangential centres.
        t = 3 - mesh.faces[index].axis
        if mesh.faces[index].center[t] < mesh.faces[first].center[t]
            sides[k, cell], second[k, cell] = index, first
        else
            second[k, cell] = index
        end
        return
    end
    for (index, face) in pairs(mesh.faces)
        minus = _cell_index(face.minus, nx, ny)
        plus = _cell_index(face.plus, nx, ny)
        face_cells[1, index] = minus
        face_cells[2, index] = plus
        if face.axis == 1
            attach!(2, minus, index)
            attach!(4, plus, index)
        else
            attach!(3, minus, index)
            attach!(1, plus, index)
        end
    end
    for (index, face) in pairs(mesh.boundary_faces)
        cell = _cell_index(face.cell, nx, ny)
        side = face.axis == 1 ? (face.side > 0 ? 2 : 4) : (face.side > 0 ? 3 : 1)
        iszero(sides[side, cell]) || throw(
            ArgumentError("MimeticProjection requires one boundary face per cell side"),
        )
        sides[side, cell] = -index
    end
    any(iszero, sides) && throw(
        ArgumentError("MimeticProjection requires a face on every cell side"),
    )
    cache = mesh.topology_cache
    return _MimeticTopology(spacing, area, sides, second, face_cells, cache.identity, cache.generation)
end

_mimetic_topology_current(topology, mesh) =
    !isnothing(topology) && topology.identity === mesh.topology_cache.identity &&
    topology.generation == mesh.topology_cache.generation

"""Cell across side `side` of `cell` (across its low face if the side has
two), or zero at a domain boundary."""
@inline function _mimetic_neighbour(topology, cell, side)
    face = topology.sides[side, cell]
    face > 0 || return 0
    minus = topology.face_cells[1, face]
    return minus == cell ? topology.face_cells[2, face] : minus
end

"""Interval `(lo, hi)` of the tangential cell coordinate (unit cell, centre
zero) covered by face `face` on side `k` of `cell`: the whole side, or its low
or high half on a side of two faces."""
@inline function _mimetic_face_interval(topology, cell, k, face, ::Type{T}) where {T}
    half = T(0.5)
    iszero(topology.second[k, cell]) && return -half, half
    return topology.sides[k, cell] == face ? (-half, zero(T)) : (zero(T), half)
end

# Same-axis interior faces near `face` in the order of the donor search
# (thesis/audit laboratory): first the faces across its two cells, then the
# collinear faces shifted by one cell along the face, then the high faces of
# the sides across its cells that consist of two faces. Zero entries are absent.
function _mimetic_face_ring(topology, face, axis)
    minus = topology.face_cells[1, face]
    plus = topology.face_cells[2, face]
    sides = topology.sides
    second = topology.second
    low, high = axis == 1 ? (4, 2) : (1, 3)
    before, after = axis == 1 ? (1, 3) : (4, 2)
    previous = _mimetic_neighbour(topology, plus, before)
    next = _mimetic_neighbour(topology, plus, after)
    return (
        max(sides[low, minus], 0), max(sides[high, plus], 0),
        previous == 0 ? 0 : max(sides[low, previous], 0),
        next == 0 ? 0 : max(sides[low, next], 0),
        second[low, minus], second[high, plus],
    )
end

# Tangential component of the unit PLIC normal of `cell` on a face with local
# normal coordinate `normal_coordinate` that covers the tangential interval
# `(lo, hi)` of the cell, and whether the PLIC line cuts the face strictly
# inside. A positive component puts the liquid at the low end.
function _mimetic_trace_vote(interface, reference, axis, normal_coordinate, (lo, hi))
    nx = _get(interface.normal_x, reference)
    ny = _get(interface.normal_y, reference)
    intercept = _get(interface.intercept, reference)
    normal, tangent = axis == 1 ? (nx, ny) : (ny, nx)
    T = typeof(intercept)
    low, high = _clip_line_interval(
        T(lo), T(hi), tangent, normal * normal_coordinate - intercept;
        keep_less = true,
    )
    return tangent / hypot(nx, ny), zero(T) < high - low < T(hi) - T(lo)
end

"""Classify the wetted end of every partially wetted interior face.

The adjacent cut cells whose PLIC line crosses the face strictly inside decide
by the sign of the sum of their tangential normal components (one crossing
line: its direction; two that disagree: the more tangential one). If no line
crosses, the first cut neighbour decides by the direction of its tangential
normal; without a cut neighbour, or for a vanishing sum, the liquid is at the
low end. Every face gets one shared decision, which is all the consistency of
the metric needs (the wetted midpoint of a face is the same for both cells).
A face on a side of two faces covers half of that side of its coarser cell.
"""
function _mimetic_face_sides!(flags, interface, topology, liquid_aperture)
    mesh = interface.mesh
    resize!(flags, length(mesh.faces))
    for (index, face) in pairs(mesh.faces)
        aperture = liquid_aperture[index]
        if !(zero(aperture) < aperture < one(aperture))
            flags[index] = Int8(0)
            continue
        end
        strong = weak = zero(aperture)
        crossings = 0
        for (reference, coordinate, end_index) in ((face.minus, 0.5, 1), (face.plus, -0.5, 2))
            fraction = _conditional_fraction(interface, reference)
            zero(fraction) < fraction < one(fraction) || continue
            cell = topology.face_cells[end_index, index]
            k = face.axis == 1 ? (end_index == 1 ? 2 : 4) : (end_index == 1 ? 3 : 1)
            tangent, crosses = _mimetic_trace_vote(
                interface, reference, face.axis, coordinate,
                _mimetic_face_interval(topology, cell, k, index, typeof(aperture)),
            )
            if crosses
                strong += tangent
                crossings += 1
            elseif iszero(weak)
                weak = tangent
            end
        end
        decision = crossings > 0 ? strong : weak
        flags[index] = decision < 0 ? _MIMETIC_LIQUID_HIGH : _MIMETIC_LIQUID_LOW
    end
    return flags
end

@inline function _mimetic_corner(::Type{T}, k) where {T}
    half = T(0.5)
    k == 1 && return (-half, -half)
    k == 2 && return (half, -half)
    k == 3 && return (half, half)
    return (-half, half)
end

"""Outward unit normal of side `k` of a cell."""
@inline function _mimetic_normal(::Type{T}, k) where {T}
    k == 1 && return (zero(T), -one(T))
    k == 2 && return (one(T), zero(T))
    k == 3 && return (zero(T), one(T))
    return (-one(T), zero(T))
end

"""Segment of a cell side in unit cell coordinates (origin at the cell
centre): the part `from ≤ t ≤ to` of side `side` on its counter-clockwise walk
(`t = 0` at the start of the side), the liquid aperture of the segment (its
wetted fraction) and whether the liquid lies at the counter-clockwise start of
its open part. A side of one face is one segment `0 ≤ t ≤ 1`, a side of two
faces two segments of half its length. With an embedded solid, `open` is the
open fraction of the segment (the shared open aperture of its face) and
`solid_start` whether the solid closes its counter-clockwise start; the
liquid aperture is a fraction of the whole segment, at most `open`. The solid
edge of a cut cell is a segment with `side = 0` (`_mimetic_solid_edge`).
`a`, `b` are the end points of the whole segment and `normal` its outward
unit normal."""
struct _MimeticSegment{T}
    side::Int
    from::T
    to::T
    aperture::T
    start::Bool
    open::T
    solid_start::Bool
    a::NTuple{2, T}
    b::NTuple{2, T}
    normal::NTuple{2, T}
end

"""Segment `from ≤ t ≤ to` of grid side `k` (see `_MimeticSegment`)."""
function _MimeticSegment{T}(
        k::Integer, from, to, aperture, start::Bool, open = one(T), solid_start::Bool = false,
    ) where {T}
    cx, cy = _mimetic_corner(T, k)
    dx, dy = _mimetic_corner(T, mod1(k + 1, 4))
    a = (cx + from * (dx - cx), cy + from * (dy - cy))
    b = (cx + to * (dx - cx), cy + to * (dy - cy))
    return _MimeticSegment{T}(
        k, T(from), T(to), T(aperture), start, T(open), solid_start, a, b, _mimetic_normal(T, k),
    )
end

"""Segment of a whole side `k` (one face)."""
_MimeticSegment(k::Integer, aperture::T, start::Bool) where {T} =
    _MimeticSegment{T}(k, zero(T), one(T), aperture, start)

"""Solid edge of a cell from `a` to `b` (counter-clockwise, in unit cell
coordinates) with outward unit `normal` (into the solid), liquid fraction
`aperture` and whether the liquid lies at its start `a`."""
function _mimetic_solid_edge(a::NTuple{2, T}, b::NTuple{2, T}, normal, aperture, start::Bool) where {T}
    return _MimeticSegment{T}(
        0, zero(T), hypot(b[1] - a[1], b[2] - a[2]), aperture, start, one(T), false, a, b, normal,
    )
end

"""Length of a segment in units of the cell width."""
@inline _mimetic_length(segment) = segment.to - segment.from

"""End points of a segment in unit cell coordinates."""
@inline _mimetic_segment_ends(segment::_MimeticSegment) = segment.a, segment.b

"""End points of the open part of a segment (the whole segment without a
solid)."""
@inline function _mimetic_open_ends(segment::_MimeticSegment{T}) where {T}
    (ax, ay), (bx, by) = segment.a, segment.b
    open = segment.open
    open < one(T) || return (ax, ay), (bx, by)
    lo, hi = segment.solid_start ? (one(T) - open, one(T)) : (zero(T), open)
    return (ax + lo * (bx - ax), ay + lo * (by - ay)), (ax + hi * (bx - ax), ay + hi * (by - ay))
end

"""Phase boundary on a segment and the midpoints of its liquid and gas parts
in unit cell coordinates (origin at the cell centre), on the open part of the
segment. A part of length zero has its midpoint at the end of the open
part."""
@inline function _mimetic_segment_parts(segment::_MimeticSegment{T}) where {T}
    (ax, ay), (bx, by) = _mimetic_open_ends(segment)
    start = segment.start
    open = segment.open
    fraction = open > zero(T) ? segment.aperture / open : zero(T)
    s = start ? fraction : one(T) - fraction
    qx, qy = ax + s * (bx - ax), ay + s * (by - ay)
    low = ((ax + qx) / 2, (ay + qy) / 2)
    high = ((qx + bx) / 2, (qy + by) / 2)
    return (qx, qy), start ? low : high, start ? high : low
end

"""Wetted length of `phase` on a segment in units of the cell width."""
@inline _mimetic_wetted(segment, phase) =
    (phase == 1 ? segment.aperture : segment.open - segment.aperture) * _mimetic_length(segment)

"""Eigenvalues `λ₁ ≤ λ₂` of the symmetric matrix `[a b; b d]` and the unit
eigenvector `(c, s)` of `λ₂` (that of `λ₁` is `(-s, c)`)."""
@inline function _mimetic_symmetric_eigen(a, b, d)
    mean = (a + d) / 2
    radius = hypot((a - d) / 2, b)
    angle = atan(2b, a - d) / 2
    return mean - radius, mean + radius, cos(angle), sin(angle)
end

"""Pressure point of a cut cell in unit cell coordinates (eqs. mimetic-point,
mimetic-interval) and the liquid closing vector `η = Σ_k a_k n_k` (eq.
mimetic-gram, in units of `h`), for the `segments` of its sides
(`_MimeticSegment`; `a_k` is the wetted length of segment `k`).

With `M = Σ_k a_k n_k (x^l_k)ᵀ` the liquid Gram tensor at `x*` is
`G^l = M - η x*ᵀ` and the gas one `G^g = F - G^l`, with the Gram tensor `F`
of the open cell: `I` without a solid, the symmetric tensor `fluid =
(f₁₁, f₁₂, f₂₂)` of the open segments and the solid edges of a cell cut by an
embedded solid. For `η ≠ 0` both are symmetric exactly on the line
`η × x* = τ`, `τ = M₁₂ - M₂₁`. The data point is the point of that line
closest to the interface of the data, the mean of the phase boundaries on the
partially wetted segments (if every segment is wetted by one phase only: the
length-weighted mean of the midpoints of the open parts of the segments of
the minority phase). It is kept if `0 ⪯ G^l ⪯ F` there; otherwise it moves
along the line into the admissible interval (`_mimetic_admissible_point`,
with the margin `cap`, by default that of a cell of the finest level). For
`η = 0` the tensors do not depend on `x*`; the point satisfies the kernel
condition instead (`_mimetic_kernel_point`). A point more than `1.5h` from
the centre is replaced by the data mean (the cell is then defective).
"""
function _mimetic_pressure_point(
        segments::AbstractVector{_MimeticSegment{T}}, cap = T(_MIMETIC_POINT_MARGIN),
        fluid = nothing,
    ) where {T}
    ex = ey = sx = sy = zero(T)
    m11 = m12 = m21 = m22 = zero(T)
    crossed = zero(T)
    wetted = perimeter = zero(T)
    for segment in segments
        a = segment.aperture
        b = a * _mimetic_length(segment)
        nx, ny = segment.normal
        (qx, qy), (lx, ly), _ = _mimetic_segment_parts(segment)
        ex += b * nx
        ey += b * ny
        m11 += b * nx * lx
        m12 += b * nx * ly
        m21 += b * ny * lx
        m22 += b * ny * ly
        wetted += b
        perimeter += segment.open * _mimetic_length(segment)
        if zero(T) < a < segment.open
            sx += qx
            sy += qy
            crossed += one(T)
        end
    end
    if iszero(crossed)
        minority = wetted >= perimeter / 2 ? zero(T) : one(T)
        for segment in segments
            segment.open > zero(T) || continue
            segment.aperture == minority * segment.open || continue
            length = segment.open * _mimetic_length(segment)
            (ax, ay), (bx, by) = _mimetic_open_ends(segment)
            sx += length * ((ax + bx) / 2)
            sy += length * ((ay + by) / 2)
            crossed += length
        end
    end
    xr, yr = crossed > zero(T) ? (sx / crossed, sy / crossed) : (zero(T), zero(T))
    e2 = ex^2 + ey^2
    if e2 > T(1.0e-18)
        defect = (m12 - m21) - (ex * yr - ey * xr)
        xs, ys, admissible = _mimetic_admissible_point(
            m11, m12, m21, m22, ex, ey, xr - defect * ey / e2, yr + defect * ex / e2, cap, fluid,
        )
    else
        xs, ys = _mimetic_kernel_point(segments, m11, (m12 + m21) / 2, m22, xr, yr, fluid)
        admissible = false
    end
    !admissible && hypot(xs, ys) > T(1.5) && ((xs, ys) = (xr, yr))
    return xs, ys, ex, ey
end

"""Move the point `(xs, ys)` of the symmetry line along `η` into the interval
where `0 ⪯ G^l ⪯ I` (eq. mimetic-interval); return the point and whether the
interval is non-empty.

In the frame `η̂ = η/|η|`, `e = η̂^⊥` the entries `g₂₂ = eᵀG^l e` and
`g₁₂ = η̂ᵀG^l e` do not change along the line and `g₁₁(s) = g₁₁(0) - s|η|` at
`x* + s η̂`; an admissible point exists if and only if `0 ≤ g₂₂ ≤ 1` and
`g₁₂² ≤ g₂₂(1 - g₂₂)`, and then `g₁₁ ∈ [g₁₂²/g₂₂, 1 - g₁₂²/(1 - g₂₂)]`. The
point is clamped into that interval shrunk at both ends by `min(w/100, cap)`
of its length `w` along the line (`cap`: `_MIMETIC_POINT_MARGIN` times the
finest cell width of the mesh, in units of this cell's width, so that the
clamped points of neighbouring cells of different levels lie on one level
line), where both
tensors are positive definite: the point, hence the metric, depends
continuously on the data, and a phase of vanishing area gets a block of order
its wetted length times the margin instead of a block that vanishes or blows
up at the end of the interval. Without an admissible point it stays (the cell
is defective)."""
function _mimetic_admissible_point(m11, m12, m21, m22, ex, ey, xs, ys, cap, ::Nothing)
    T = typeof(xs)
    tolerance = T(1.0e-12)
    c = sqrt(ex^2 + ey^2)
    hx, hy = ex / c, ey / c
    px, py = -hy, hx
    g11, g22 = m11 - ex * xs, m22 - ey * ys
    g12 = ((m12 - ex * ys) + (m21 - ey * xs)) / 2
    q11 = hx * (g11 * hx + g12 * hy) + hy * (g12 * hx + g22 * hy)
    q12 = hx * (g11 * px + g12 * py) + hy * (g12 * px + g22 * py)
    q22 = px * (g11 * px + g12 * py) + py * (g12 * px + g22 * py)
    -tolerance <= q22 <= one(T) + tolerance || return xs, ys, false
    q22 = clamp(q22, zero(T), one(T))
    lo = q22 > T(1.0e-14) ? q12^2 / q22 : (abs(q12) < tolerance ? zero(T) : T(Inf))
    hi = one(T) - q22 > T(1.0e-14) ? one(T) - q12^2 / (one(T) - q22) :
        (abs(q12) < tolerance ? one(T) : -T(Inf))
    lo > hi + tolerance && return xs, ys, false
    smin, smax = (q11 - hi) / c, (q11 - lo) / c
    margin = min(max(smax - smin, zero(T)) / 100, cap)
    s = clamp(zero(T), smin + margin, max(smin + margin, smax - margin))
    return xs + s * hx, ys + s * hy, true
end

"""`_mimetic_admissible_point` for `0 ⪯ G^l ⪯ F` with the symmetric positive
definite Gram tensor `F = (f₁₁, f₁₂, f₂₂)` of the open part of a cell cut by
an embedded solid. With `F = LLᵀ` (Cholesky) the condition is
`0 ⪯ Q ⪯ I` for `Q = L⁻¹G^l L⁻ᵀ`, and moving `x*` by `s η̂` changes `Q` by
`-s|η||L⁻¹η̂|² v vᵀ`, `v = L⁻¹η̂/|L⁻¹η̂|`: the interval of the unit case in the
frame `(v, v^⊥)` of `Q`, with the rate `|η||L⁻¹η̂|²` of `q₁₁` along the line.
Without a positive definite `F` there is no admissible point."""
function _mimetic_admissible_point(m11, m12, m21, m22, ex, ey, xs, ys, cap, fluid::NTuple{3})
    T = typeof(xs)
    tolerance = T(1.0e-12)
    f11, f12, f22 = fluid
    f11 > zero(T) || return xs, ys, false
    l11 = sqrt(f11)
    l21 = f12 / l11
    d = f22 - l21^2
    d > zero(T) || return xs, ys, false
    l22 = sqrt(d)
    # Rows of L⁻¹ = [i11 0; i21 i22].
    i11, i21, i22 = inv(l11), -l21 / (l11 * l22), inv(l22)
    c = sqrt(ex^2 + ey^2)
    hx, hy = ex / c, ey / c
    g11, g22 = m11 - ex * xs, m22 - ey * ys
    g12 = ((m12 - ex * ys) + (m21 - ey * xs)) / 2
    r11 = i11^2 * g11
    r12 = i11 * (i21 * g11 + i22 * g12)
    r22 = i21^2 * g11 + 2 * i21 * i22 * g12 + i22^2 * g22
    vx, vy = i11 * hx, i21 * hx + i22 * hy
    w = hypot(vx, vy)
    vx, vy = vx / w, vy / w
    px, py = -vy, vx
    rate = c * w^2
    q11 = vx * (r11 * vx + r12 * vy) + vy * (r12 * vx + r22 * vy)
    q12 = vx * (r11 * px + r12 * py) + vy * (r12 * px + r22 * py)
    q22 = px * (r11 * px + r12 * py) + py * (r12 * px + r22 * py)
    -tolerance <= q22 <= one(T) + tolerance || return xs, ys, false
    q22 = clamp(q22, zero(T), one(T))
    lo = q22 > T(1.0e-14) ? q12^2 / q22 : (abs(q12) < tolerance ? zero(T) : T(Inf))
    hi = one(T) - q22 > T(1.0e-14) ? one(T) - q12^2 / (one(T) - q22) :
        (abs(q12) < tolerance ? one(T) : -T(Inf))
    lo > hi + tolerance && return xs, ys, false
    smin, smax = (q11 - hi) / rate, (q11 - lo) / rate
    margin = min(max(smax - smin, zero(T)) / 100, cap)
    s = clamp(zero(T), smin + margin, max(smin + margin, smax - margin))
    return xs + s * hx, ys + s * hy, true
end

"""Largest distance (units of the finest cell width of the mesh) by which
`_mimetic_admissible_point` keeps the pressure point inside the ends of its
admissible interval."""
const _MIMETIC_POINT_MARGIN = 0.01

"""Pressure point of a cut cell with `η = 0`: the Gram tensors `G^l` (symmetric
part `[g11 g12; g12 g22]`) and `G^g = F - G^l` (`F = I` without a solid, else
the Gram tensor `fluid` of the open cell) do not depend on `x*`, and a block
with `m N = R` needs `ker G ⊂ ker R`. For every eigenvector `v` with
eigenvalue `0` of `G^l` or of `G^g` the component `v ⋅ x*` is set to the
wetted-length weighted mean of `v ⋅ x_k` over the segments of that phase,
which satisfies the condition when the `v ⋅ x_k` agree (a film along a grid
line: its mid-height); the other component is that of the data point."""
function _mimetic_kernel_point(
        segments::AbstractVector{_MimeticSegment{T}}, g11, g12, g22, xs, ys, ::Nothing,
    ) where {T}
    lambda1, lambda2, c, s = _mimetic_symmetric_eigen(g11, g12, g22)
    tolerance = T(1.0e-10)
    for (lambda, vx, vy) in ((lambda1, -s, c), (lambda2, c, s))
        for phase in 1:2
            (phase == 1 ? lambda : one(T) - lambda) <= tolerance || continue
            total = weighted = zero(T)
            for segment in segments
                a = segment.aperture
                b = (phase == 1 ? a : one(T) - a) * _mimetic_length(segment)
                b > zero(T) || continue
                _, liquid, gas = _mimetic_segment_parts(segment)
                mx, my = phase == 1 ? liquid : gas
                total += b
                weighted += b * (vx * mx + vy * my)
            end
            total > zero(T) || continue
            shift = weighted / total - (vx * xs + vy * ys)
            xs += shift * vx
            ys += shift * vy
        end
    end
    return xs, ys
end

function _mimetic_kernel_point(
        segments::AbstractVector{_MimeticSegment{T}}, g11, g12, g22, xs, ys, fluid::NTuple{3},
    ) where {T}
    tolerance = T(1.0e-10)
    f11, f12, f22 = fluid
    for phase in 1:2
        a11, a12, a22 = phase == 1 ? (g11, g12, g22) : (f11 - g11, f12 - g12, f22 - g22)
        lambda1, lambda2, c, s = _mimetic_symmetric_eigen(a11, a12, a22)
        for (lambda, vx, vy) in ((lambda1, -s, c), (lambda2, c, s))
            lambda <= tolerance || continue
            total = weighted = zero(T)
            for segment in segments
                b = _mimetic_wetted(segment, phase)
                b > zero(T) || continue
                _, liquid, gas = _mimetic_segment_parts(segment)
                mx, my = phase == 1 ? liquid : gas
                total += b
                weighted += b * (vx * mx + vy * my)
            end
            total > zero(T) || continue
            shift = weighted / total - (vx * xs + vy * ys)
            xs += shift * vx
            ys += shift * vy
        end
    end
    return xs, ys
end

"""Gram block of one phase of a cut cell in unit coordinates (eqs.
mimetic-gram, mimetic-local).

For the `m` segments wetted by `phase` (`sides[1:m]`: their indices in
`segments`, wetted lengths `lengths[1:m]`, outward normals `n_k`, wetted
midpoints `x_k`) and the pressure point `(xs, ys)`, fills `moments[1:m, :]`
with `R = [b_k (x_k - x*)ᵀ]` and `block[1:m, 1:m]` with `R G⁺ Rᵀ + S`,
`G = Σ_k b_k n_k (x_k - x*)ᵀ`. `S` is the stabilisation `½ D Q D` of the
wetted segments closed by one more edge of length `|η|` and outward normal
`∓η/|η|` (liquid, gas), whose flux is slaved to close the region (eq.
mimetic-slave). Returns `m` and whether the block is consistent,
`block N = R`: that needs a symmetric positive semi-definite `G` whose kernel
lies in that of `R`. Otherwise the eigenvalues of the symmetric part of `G`
are raised to `1/50` (least-squares consistency). A pure cell is the phase
that wets all its segments, with `x*` its centre and `η = 0`: then `G = I`.
"""
function _mimetic_gram_block!(
        block, sides, lengths, moments, segments, phase, xs, ys, ex, ey,
    )
    T = eltype(block)
    m = 0
    g11 = g12 = g21 = g22 = zero(T)
    for (index, segment) in pairs(segments)
        b = _mimetic_wetted(segment, phase)
        b > zero(T) || continue
        _, liquid, gas = _mimetic_segment_parts(segment)
        mx, my = phase == 1 ? liquid : gas
        nx, ny = segment.normal
        m += 1
        sides[m] = index
        lengths[m] = b
        rx, ry = b * (mx - xs), b * (my - ys)
        moments[m, 1] = rx
        moments[m, 2] = ry
        g11 += nx * rx
        g12 += nx * ry
        g21 += ny * rx
        g22 += ny * ry
    end
    m == 0 && return 0, true
    lambda1, lambda2, c, s = _mimetic_symmetric_eigen(g11, (g12 + g21) / 2, g22)
    consistent = abs(g12 - g21) <= T(1.0e-10) && lambda1 > -T(1.0e-10)
    if consistent
        _mimetic_fill_gram_block!(block, m, sides, lengths, moments, segments, lambda1, lambda2, c, s, false, phase, ex, ey)
        for q in 1:m
            rx = ry = zero(T)
            for r in 1:m
                nx, ny = segments[sides[r]].normal
                rx += block[q, r] * nx
                ry += block[q, r] * ny
            end
            consistent &= max(abs(rx - moments[q, 1]), abs(ry - moments[q, 2])) <= T(1.0e-8)
        end
    end
    consistent || _mimetic_fill_gram_block!(
        block, m, sides, lengths, moments, segments, lambda1, lambda2, c, s, true, phase, ex, ey,
    )
    return m, consistent
end

"""Floor of the eigenvalues of a Gram tensor without a consistent block, in
units of `h²` (least-squares consistency, see `_mimetic_gram_block!`)."""
const _MIMETIC_GRAM_FLOOR = 0.02

"""`block[1:m, 1:m] = R G⁺ Rᵀ + S` (see `_mimetic_gram_block!`) from the
eigen decomposition of `G`; with `floored`, the eigenvalues are raised to
`_MIMETIC_GRAM_FLOOR` instead of inverting only the positive ones."""
function _mimetic_fill_gram_block!(
        block, m, sides, lengths, moments, segments, lambda1, lambda2, c, s, floored, phase, ex, ey,
    )
    T = eltype(block)
    least = T(_MIMETIC_GRAM_FLOOR)
    weight(lambda) = floored ? inv(max(lambda, least)) :
        (lambda > T(1.0e-10) ? inv(lambda) : zero(T))
    w1, w2 = weight(lambda1), weight(lambda2)
    # G⁺ = w₁ v₁v₁ᵀ + w₂ v₂v₂ᵀ with v₂ = (c, s), v₁ = (-s, c).
    i11 = w1 * s^2 + w2 * c^2
    i12 = (w2 - w1) * c * s
    i22 = w1 * c^2 + w2 * s^2
    # Stabilisation: the projector Q onto the complement of the range of the
    # rows f_q = L_q n_q of the closed edge set, P⁺ the pseudo-inverse of Σ f fᵀ.
    e2 = ex^2 + ey^2
    closing = e2 > T(1.0e-18)
    cl = closing ? sqrt(e2) : zero(T)
    orientation = phase == 1 ? -one(T) : one(T)
    cnx, cny = closing ? (orientation * ex / cl, orientation * ey / cl) : (zero(T), zero(T))
    edge(q) = q <= m ? (lengths[q], segments[sides[q]].normal) : (cl, (cnx, cny))
    p11 = p12 = p22 = zero(T)
    for q in 1:(m + closing)
        L, (nx, ny) = edge(q)
        p11 += (L * nx)^2
        p12 += L^2 * nx * ny
        p22 += (L * ny)^2
    end
    mu1, mu2, pc, ps = _mimetic_symmetric_eigen(p11, p12, p22)
    tolerance = T(1.0e-12) * max(mu2, floatmin(T))
    v1 = mu1 > tolerance ? inv(mu1) : zero(T)
    v2 = mu2 > tolerance ? inv(mu2) : zero(T)
    q11 = v1 * ps^2 + v2 * pc^2
    q12 = (v2 - v1) * pc * ps
    q22 = v1 * pc^2 + v2 * ps^2
    # Index of the closing edge.
    gamma = m + 1
    function stabiliser(q, r)
        Lq, (nqx, nqy) = edge(q)
        Lr, (nrx, nry) = edge(r)
        fqx, fqy, frx, fry = Lq * nqx, Lq * nqy, Lr * nrx, Lr * nry
        projector = (q == r ? one(T) : zero(T)) -
            (fqx * (q11 * frx + q12 * fry) + fqy * (q12 * frx + q22 * fry))
        return Lq * projector * Lr / 2
    end
    # Upper triangle, mirrored: the block is symmetric bitwise.
    for q in 1:m, r in q:m
        rqx, rqy = moments[q, 1], moments[q, 2]
        rrx, rry = moments[r, 1], moments[r, 2]
        value = rqx * (i11 * rrx + i12 * rry) + rqy * (i12 * rrx + i22 * rry) + stabiliser(q, r)
        if closing
            # Slaved closing flux: v_G = -Σ_q L_q v_q / |η|.
            sq, sr = lengths[q] / cl, lengths[r] / cl
            value += -sr * stabiliser(q, gamma) - sq * stabiliser(gamma, r) +
                sq * sr * stabiliser(gamma, gamma)
        end
        block[q, r] = block[r, q] = value
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
    wall_coupling::Vector{T}
    wall_cells::Int
    solid_cells::Int
    cut_cells::Int
    slaved_values::Int
    stabilised_values::Int
    lumped_values::Int
    defective_cells::Int
    full_diagonal::Vector{T}
    rows::Vector{Int}
    columns::Vector{Int}
    values::Vector{T}
    touched::BitVector
    flags::Vector{Int8}
    block::Matrix{T}
    sides::Vector{Int}
    lengths::Vector{T}
    moments::Matrix{T}
    segments::Vector{_MimeticSegment{T}}
    segment_faces::Vector{Int}
end

function _MimeticAssembly(::Type{T}) where {T}
    return _MimeticAssembly{T}(
        typemax(UInt), Int[], Int[], Int[], Int8[], T[], T[], T[], 0, Int[], Int[], T[],
        spzeros(T, 0, 0), nothing, T[], T[], T[], 0, 0, 0, 0, 0, 0, 0, T[], Int[], Int[], T[],
        falses(0), Int8[], zeros(T, 16, 16), zeros(Int, 16), zeros(T, 16), zeros(T, 16, 2),
        _MimeticSegment{T}[], Int[],
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

"""Factor `β` of the aperture-independent tie weight `β ρ_π V_f` of the metric
(`_assemble_cut_cell_metric!`). With `β = 1` the measured equivalence constant
`λ_min(𝓜, diag(ρ_π V_f))` stays above `0.085` (static drop, RT, KH); the
static-drop energy is flat between `β = 1/2` and `2`."""
const _MIMETIC_TIE_BETA = 1.0

"""Embedded-solid data of the metric (see `_assemble_cut_cell_metric!`): the
shared open apertures of the interior and boundary faces, whether the solid
closes the low tangential end of a partially open face (`low`,
`boundary_low`), the open dual volumes of the interior faces, and per cell
its open fraction, the centroid of its open polygon and its liquid PLIC line
`line ⋅ x ≤ line_d` (unit cell coordinates) with its conditional liquid
fraction."""
struct _MimeticSolidData{T}
    open::Vector{T}
    low::Vector{Bool}
    boundary_open::Vector{T}
    boundary_low::Vector{Bool}
    face_volume::Vector{T}
    cell_open::Vector{T}
    centroid_x::Vector{T}
    centroid_y::Vector{T}
    line_x::Vector{T}
    line_y::Vector{T}
    line_d::Vector{T}
    fraction::Vector{T}
end

_MimeticSolidData(::Type{T}) where {T} = _MimeticSolidData{T}(
    T[], Bool[], T[], Bool[], T[], T[], T[], T[], T[], T[], T[], T[],
)

"""Segment `from ≤ t ≤ to` of side `k` of `cell` on interior face `face` (or
boundary face `-face`) with its liquid aperture and wet end, and with an
embedded solid (`solid`, `_MimeticSolidData`) its open fraction and solid
end. A grid-line interface face is fully wet from its own cell's side."""
function _mimetic_segment(
        ::Type{T}, k, from, to, face, cell, topology, data, flags,
        boundary_aperture, boundary_low, gridline, solid = nothing,
    ) where {T}
    open = isnothing(solid) ? one(T) :
        face < 0 ? solid.boundary_open[-face] : solid.open[face]
    aperture = if face < 0
        boundary_aperture[-face]
    elseif isnothing(gridline) || iszero(gridline[face])
        data.liquid_aperture[face]
    else
        minus = topology.face_cells[1, face] == cell
        (gridline[face] > 0) == minus ? open : zero(T)
    end
    start = if zero(T) < aperture < open
        low = face > 0 ? flags[face] == _MIMETIC_LIQUID_LOW : boundary_low[-face]
        _MIMETIC_STARTS_LOW[k] ? low : !low
    else
        false
    end
    isnothing(solid) && return _MimeticSegment{T}(k, from, to, aperture, start)
    solid_start = if zero(T) < open < one(T)
        low = face > 0 ? solid.low[face] : solid.boundary_low[-face]
        _MIMETIC_STARTS_LOW[k] ? low : !low
    else
        false
    end
    return _MimeticSegment{T}(k, from, to, aperture, start, open, solid_start)
end

"""Length below which a chord between the open parts of a cell counts as
closed (no solid edge), in units of the cell width."""
const _MIMETIC_SOLID_EDGE_FLOOR = 1.0e-10

"""Whether the open part of a grid segment reaches its counter-clockwise
start (`start = true`) or end."""
@inline function _mimetic_open_reaches(segment::_MimeticSegment{T}, start::Bool) where {T}
    segment.open > zero(T) || return false
    segment.open >= one(T) && return true
    return segment.solid_start != start
end

"""Close the open parts of the grid segments of `cell` with its solid edges
(eq. mimetic-solid-edge) and return the symmetric Gram tensor `F` of the open
cell, `(f₁₁, f₁₂, f₂₂)` in units of `h²`.

The open parts of consecutive segments along the counter-clockwise walk form
runs (a run continues across a segment end that both open parts reach). A
solid edge joins the free end of each run to the free start of the next run
(of the same run if there is one): a side from `a` to `b` with outward unit
normal `(b - a)^⊥` (into the solid) and midpoint `x_S`. The open parts and the
edges bound the convex hull of the open parts (their end points lie on the
boundary of the square in counter-clockwise order), so
`F = P + Σ_S |S| n_S x_Sᵀ` (`P = Σ_s o_s ℓ_s n_s m_sᵀ`, open lengths
`o_s ℓ_s`, outward normals `n_s`, midpoints `m_s` of the open parts) is `A I`
with the area `A` of that polygon, whatever the shared open apertures:
positive definite unless all open parts lie on one side of the cell. For one
run whose chord is the solid's PLIC segment
(data that bound the open polygon) it is the open polygon's tensor. Several
runs come from a thin solid across the cell or from averaged traces that put
an open part next to a closed side (a closed trace averaged with an open
one). The liquid part of an edge is the part on the liquid side of the cell's
PLIC line (the contact point is their intersection), all of it in a pure
liquid cell, none in a pure gas cell. A chord of length at most
`_MIMETIC_SOLID_EDGE_FLOOR` gives no edge."""
function _mimetic_add_solid_edge!(
        segments::Vector{_MimeticSegment{T}}, segment_faces, solid, cell,
    ) where {T}
    p11 = p12 = p21 = p22 = zero(T)
    grid = length(segments)
    for segment in segments
        o = segment.open * _mimetic_length(segment)
        o > zero(T) || continue
        (ax, ay), (bx, by) = _mimetic_open_ends(segment)
        mx, my = (ax + bx) / 2, (ay + by) / 2
        nx, ny = segment.normal
        p11 += o * nx * mx
        p12 += o * nx * my
        p21 += o * ny * mx
        p22 += o * ny * my
    end
    # The first open part whose start is not joined to the previous open part
    # starts a run; without one the open parts close the cell themselves.
    joined(q) = _mimetic_open_reaches(segments[mod1(q - 1, grid)], false) && _mimetic_open_reaches(segments[q], true)
    first_head = 0
    for q in 1:grid
        segments[q].open > zero(T) && !joined(q) && (first_head = q; break)
    end
    head = first_head
    while head > 0
        tail = head
        while joined(mod1(tail + 1, grid))
            tail = mod1(tail + 1, grid)
        end
        next = mod1(tail + 1, grid)
        while !(segments[next].open > zero(T))
            next = mod1(next + 1, grid)
        end
        _, a = _mimetic_open_ends(segments[tail])
        b, _ = _mimetic_open_ends(segments[next])
        head = next == first_head ? 0 : next
        cx, cy = b[1] - a[1], b[2] - a[2]
        chord = hypot(cx, cy)
        chord > T(_MIMETIC_SOLID_EDGE_FLOOR) || continue
        nx, ny = cy / chord, -cx / chord
        xs, ys = (a[1] + b[1]) / 2, (a[2] + b[2]) / 2
        aperture, start = _mimetic_solid_wetting(solid, cell, a, b)
        push!(segments, _mimetic_solid_edge(a, b, (nx, ny), aperture, start))
        push!(segment_faces, 0)
        p11 += chord * nx * xs
        p12 += chord * nx * ys
        p21 += chord * ny * xs
        p22 += chord * ny * ys
    end
    return p11, (p12 + p21) / 2, p22
end

"""Liquid fraction of the solid edge from `a` to `b` of `cell` and whether the
liquid lies at `a`: the part on the liquid side `line ⋅ x ≤ line_d` of the
cell's PLIC line."""
function _mimetic_solid_wetting(solid, cell, (ax, ay), (bx, by))
    fraction = solid.fraction[cell]
    T = typeof(fraction)
    fraction <= zero(T) && return zero(T), false
    fraction >= one(T) && return one(T), false
    nx, ny, d = solid.line_x[cell], solid.line_y[cell], solid.line_d[cell]
    f0 = nx * ax + ny * ay - d
    f1 = nx * bx + ny * by - d
    f0 <= zero(T) && f1 <= zero(T) && return one(T), false
    f0 > zero(T) && f1 > zero(T) && return zero(T), false
    s = f0 / (f0 - f1)
    return f0 <= zero(T) ? (s, true) : (one(T) - s, false)
end

"""Assemble the mimetic cut-cell metric (Gram-tensor blocks of the cut cells,
with the tie terms of the audit's laboratory).

Every cut cell (a cell whose liquid apertures are not all `0` or all `1`) gets
one pressure point (`_mimetic_pressure_point`) and per phase the Gram block
of the segments of its sides that phase wets (`_mimetic_gram_block!`); a cell
in which a block is only least-squares consistent counts as defective. Uncut
cells with one face per side give their face values `ρh²/2`, the Gram block
of the square; an uncut cell with a side of two faces (next to two finer
cells) gets the Gram block of its phase on its five to eight segments, with
the pressure point at its centre. The stability parts of the metric use the
aperture-independent mass `w_f = β ρ_π V_f` of a value (`β =
_MIMETIC_TIE_BETA`, `V_f` the staggered dual volume of its face, the open dual
volume with a solid): a phase value on a face that the phase does not reach
is tied to a donor with weight `w_f`, and every value with aperture `a < 1/2`
on a mixed face to a donor with weight `(1 - 2a) w_f` (small-cell
stabilisation). The ties vanish on constant fields and are continuous in
`a`; their floor `w_f` as `a → 0` makes `𝓜` uniformly equivalent to
`diag(ρ_π V_f)` (`𝓜 ⪰ c diag(ρ_π V_f)` with `c` independent of the apertures
and fractions).

The geometry enters only through `data` (staggered liquid fractions, phase
apertures and masses of the interior faces), the wet-end `flags` of the
interior faces, the liquid apertures and wet ends (`true`: low end) of
the boundary faces and, with an embedded solid, `solid`
(`_MimeticSolidData`).

A side on the domain boundary (a wall) is a side of the cut cell like any
other: its phase apertures, wet end and midpoint enter the pressure point and
the Gram blocks. Its flux is prescribed, not a value: the block is restricted
to the interior sides (the prescribed flux drops out of `u - u*`), and
`wall_coupling[2r - 2 + k]` (`_mimetic_wall_coupling`) collects the coupling
`Σ ρ h² m_{q,w} (n_w)_k` of value `r` to the wall sides `w`, so that
`𝓜 w + C w` is the Gram moment `R w` of a constant field `w` on every value
(`_mimetic_uniform_acceleration!`). A value that its phase does not reach and
that has no donor ties to a same-axis wall face across one of its cells if
there is one (`_mimetic_wall_donor`): the tie `w_f (u_r - u_w)²` to the
prescribed wall value adds its weight `w_f` to the diagonal, as a lumped value
would, and `-w_f` to the coupling of its axis, so the constant field of that
axis is reproduced (the tie vanishes on it) and gravity gives it no
increment.

`gridline[f]` marks an interior face between a pure liquid and a pure gas cell
(`1`: liquid minus cell, `-1`: liquid plus cell, `0`: neither): the interface
lies on the face, no cell is cut. Both cells are treated as uncut and the two
phase values of the face are identified (normal continuity `u^l = u^g`), the
limit of the cut-cell construction as the chord approaches the face. Its one
value carries the summed staggered mass and the full aperture.

With an embedded solid a segment is open over the shared open aperture of
its face, closed at the solid end decided for the face; the phase apertures
are fractions of the whole face. A cell with a partially closed segment gets
its solid edges, the chords that close the open parts (`_mimetic_add_solid_edge!`):
sides with prescribed zero flux, restricted out of the blocks and coupled to
the flux density `n_S ⋅ w` of a constant field like a wall side, with the
contact point where the cell's PLIC line crosses them. The pressure point of a
cut cell is admissible for `0 ⪯ G^l ⪯ F` with the Gram tensor `F` of the open
cell; a pure cell cut by the solid gets the Gram block of its phase with the
pressure point at the centroid of its open polygon. A value with aperture
below one half on a partially closed face is tied like one on a mixed face,
and a value without aperture and donor next to the solid ties to the solid
(prescribed zero velocity) like a wall value. Faces without open dual volume
have no values.
"""
function _assemble_cut_cell_metric!(
        assembly::_MimeticAssembly{T}, mesh,
        topology, data, flags, boundary_aperture, boundary_low, density;
        gridline = nothing, solid = nothing,
    ) where {T}
    spacing = topology.spacing
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
        volume = isnothing(solid) ? mesh.face_dual_volumes[face] : solid.face_volume[face]
        assembly.liquid_dof[face] = 0
        assembly.gas_dof[face] = 0
        iszero(volume) && continue
        if on_gridline(face)
            liquid, gas = data.liquid_mass[face], data.gas_mass[face]
            push!(assembly.dof_face, face)
            push!(assembly.dof_phase, Int8(3))
            push!(assembly.dof_aperture, isnothing(solid) ? one(T) : solid.open[face])
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
    fill!(resize!(assembly.wall_coupling, 2dofs), zero(T))
    nx, ny = mesh.block_size
    cut_cells = defective_cells = wall_cells = solid_cells = 0
    segments = assembly.segments
    segment_faces = assembly.segment_faces
    # The pressure-point margin is one length on the whole mesh (that of the
    # finest cells), so clamped points of neighbours of different levels are
    # level; on a uniform mesh the factor is one.
    finest = minimum(spacing)
    for cell in 1:cells
        h = spacing[cell]
        block_index = div(cell - 1, nx * ny) + 1
        local_index = rem(cell - 1, nx * ny)
        i, j = rem(local_index, nx) + 1, div(local_index, nx) + 1
        center = cellcenter(mesh, block_index, i, j)
        assembly.pressure_x[cell] = center[1]
        assembly.pressure_y[cell] = center[2]
        # The segments of the sides along the counter-clockwise walk.
        empty!(segments)
        empty!(segment_faces)
        split = false
        for k in 1:4
            low, high = topology.sides[k, cell], topology.second[k, cell]
            if iszero(high)
                push!(segment_faces, low)
                push!(segments, _mimetic_segment(T, k, zero(T), one(T), low, cell, topology, data, flags, boundary_aperture, boundary_low, gridline, solid))
            else
                split = true
                first, last = _MIMETIC_STARTS_LOW[k] ? (low, high) : (high, low)
                half = T(0.5)
                push!(segment_faces, first, last)
                push!(segments, _mimetic_segment(T, k, zero(T), half, first, cell, topology, data, flags, boundary_aperture, boundary_low, gridline, solid))
                push!(segments, _mimetic_segment(T, k, half, one(T), last, cell, topology, data, flags, boundary_aperture, boundary_low, gridline, solid))
            end
        end
        dof_of(q, phase) = (
            face = segment_faces[q]; face <= 0 ? 0 :
                (phase == 1 ? assembly.liquid_dof[face] : assembly.gas_dof[face])
        )
        cap = T(_MIMETIC_POINT_MARGIN) * (finest / h)
        xs = ys = ex = ey = zero(T)
        if !isnothing(solid) && any(segment -> segment.open < one(T), segments)
            # A cell cut by the solid: open segments and the solid edges.
            iszero(solid.cell_open[cell]) && continue
            fluid = _mimetic_add_solid_edge!(segments, segment_faces, solid, cell)
            liquid = all(segment -> segment.aperture == segment.open, segments)
            gas = all(segment -> iszero(segment.aperture), segments)
            liquid && gas && continue
            solid_cells += 1
            if liquid || gas
                xs, ys = solid.centroid_x[cell], solid.centroid_y[cell]
            else
                xs, ys, ex, ey = _mimetic_pressure_point(segments, cap, fluid)
                cut_cells += 1
                wall_cells += any(<(0), segment_faces)
            end
            assembly.pressure_x[cell] = center[1] + h * xs
            assembly.pressure_y[cell] = center[2] + h * ys
        else
            reference = segments[1].aperture
            uncut = all(segment -> segment.aperture == reference, segments) &&
                (iszero(reference) || isone(reference))
            if uncut && !split
                # Uncut cell: (1/2) rho h^2 on every face value, the diagonal metric.
                phase = isone(reference) ? 1 : 2
                for q in 1:4
                    dof = dof_of(q, phase)
                    dof > 0 && (assembly.full_diagonal[dof] += density[phase] * h^2 / 2)
                end
                continue
            end
            if !uncut
                xs, ys, ex, ey = _mimetic_pressure_point(segments, cap)
                assembly.pressure_x[cell] = center[1] + h * xs
                assembly.pressure_y[cell] = center[2] + h * ys
                cut_cells += 1
                wall_cells += any(<(0), segment_faces)
            end
        end
        defective = false
        for phase in 1:2
            m, consistent = _mimetic_gram_block!(
                assembly.block, assembly.sides, assembly.lengths, assembly.moments,
                segments, phase, xs, ys, ex, ey,
            )
            defective |= !consistent
            scale = density[phase] * h^2
            for q in 1:m
                sq = assembly.sides[q]
                rq = dof_of(sq, phase)
                if rq == 0
                    segment_faces[sq] <= 0 && continue
                    error("mimetic metric: wetted face without a phase value")
                end
                for r in 1:m
                    sr = assembly.sides[r]
                    rr = dof_of(sr, phase)
                    value = _MIMETIC_OUTWARD[segments[sq].side] * scale * assembly.block[q, r]
                    if rr > 0
                        _mimetic_add!(assembly, rq, rr, _MIMETIC_OUTWARD[segments[sr].side] * value)
                    elseif segment_faces[sr] <= 0
                        # Wall side or solid edge: coupling to its prescribed
                        # flux density n_w ⋅ w.
                        wx, wy = segments[sr].normal
                        assembly.wall_coupling[2rq - 1] += value * wx
                        assembly.wall_coupling[2rq] += value * wy
                    end
                end
            end
        end
        defective_cells += defective
    end
    # Additions on the face values (stability parts of the metric).
    slaved = stabilised = lumped_values = 0
    block_diagonal = resize!(assembly.diagonal, dofs)
    copyto!(block_diagonal, assembly.full_diagonal)
    for index in eachindex(assembly.rows)
        assembly.rows[index] == assembly.columns[index] &&
            (block_diagonal[assembly.rows[index]] += assembly.values[index])
    end
    reaches(dof) = dof > 0 && assembly.dof_aperture[dof] > zero(T)
    at_solid(face, axis) = !isnothing(solid) && _mimetic_solid_donor(topology, mesh, solid, face, axis)
    for dof in 1:dofs
        face = assembly.dof_face[dof]
        phase = assembly.dof_phase[dof]
        aperture = assembly.dof_aperture[dof]
        mass = assembly.dof_mass[dof]
        # Weight of the ties and lumped fallbacks: β ρ_π V_f (the summed
        # staggered mass, times β, for an identified grid-line value).
        weight = T(_MIMETIC_TIE_BETA) * (
            phase == 3 ? mass :
                density[phase] * (isnothing(solid) ? mesh.face_dual_volumes[face] : solid.face_volume[face])
        )
        dof_at(g) = g == 0 ? 0 : (phase == 1 ? assembly.liquid_dof[g] : assembly.gas_dof[g])
        axis = mesh.faces[face].axis
        if aperture > zero(T)
            # A block that gives the value no inertia beyond round-off (a phase
            # of vanishing area) does not count: lump the tie weight.
            if !(block_diagonal[dof] > T(1.0e-12) * density[phase == 2 ? 2 : 1] * topology.area[face]^2)
                _mimetic_add!(assembly, dof, dof, weight)
                lumped_values += 1
            end
            fraction = data.liquid_fraction[face]
            (zero(T) < fraction < one(T) || (!isnothing(solid) && solid.open[face] < one(T))) &&
                aperture < T(0.5) || continue
            larger(g) = (
                d = dof_at(g); reaches(d) &&
                    assembly.dof_aperture[d] > aperture + T(1.0e-12)
            )
            ring = _mimetic_face_ring(topology, face, axis)
            donor = _mimetic_donor(topology, mesh, ring, dof_at, larger, assembly.dof_aperture)
            donor == 0 && continue
            _mimetic_add_tie!(assembly, dof, donor, (one(T) - 2aperture) * weight)
            stabilised += 1
            continue
        end
        ring = _mimetic_face_ring(topology, face, axis)
        donor = _mimetic_donor(topology, mesh, ring, dof_at, g -> reaches(dof_at(g)), assembly.dof_aperture)
        if donor == 0
            _mimetic_add!(assembly, dof, dof, weight)
            if _mimetic_wall_donor(topology, face, axis) || at_solid(face, axis)
                assembly.wall_coupling[2dof - 2 + axis] -= weight
                slaved += 1
            else
                lumped_values += 1
            end
        else
            _mimetic_add_tie!(assembly, dof, donor, weight)
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
    assembly.wall_cells = wall_cells
    assembly.solid_cells = solid_cells
    assembly.slaved_values = slaved
    assembly.stabilised_values = stabilised
    assembly.lumped_values = lumped_values
    assembly.defective_cells = defective_cells
    return assembly
end

"""Coupling `C[k, r]` of value `r` to the wall flux density of a constant
field along axis `k` (see `_assemble_cut_cell_metric!`)."""
@inline _mimetic_wall_coupling(assembly, k, r) = assembly.wall_coupling[2r - 2 + k]

"""Whether a same-axis face across one of the two cells of interior face
`face` is a domain-boundary face (the wall donor of a value without one)."""
function _mimetic_wall_donor(topology, face, axis)
    minus = topology.face_cells[1, face]
    plus = topology.face_cells[2, face]
    low, high = axis == 1 ? (4, 2) : (1, 3)
    return topology.sides[low, minus] < 0 || topology.sides[high, plus] < 0
end

"""Whether a value on interior face `face` without a donor ties to the
embedded solid (prescribed zero velocity): the face is partially closed or
has a partially solid cell, or a same-axis face across one of its cells is
partially closed by the solid (the solid counterpart of `_mimetic_wall_donor`)."""
function _mimetic_solid_donor(topology, mesh, solid, face, axis)
    T = eltype(solid.open)
    (solid.open[face] < one(T) || solid.face_volume[face] < mesh.face_dual_volumes[face]) && return true
    minus = topology.face_cells[1, face]
    plus = topology.face_cells[2, face]
    low, high = axis == 1 ? (4, 2) : (1, 3)
    for across in (
            topology.sides[low, minus], topology.second[low, minus],
            topology.sides[high, plus], topology.second[high, plus],
        )
        across > 0 && solid.open[across] < one(T) && return true
    end
    return false
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
