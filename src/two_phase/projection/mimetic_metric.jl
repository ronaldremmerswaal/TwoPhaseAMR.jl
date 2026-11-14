# The mimetic (MFD) kinetic metric of `MimeticProjection` on the Gram tensors
# of the cut cells (candidate MMd of the audit note on kinetic metrics).
#
# Notation follows docs/techref/chapters/10_projection.tex, section "Mimetic
# cut-cell projection": a cell side k = 1, 2, 3, 4 is its bottom, right, top
# and left side (counter-clockwise), the liquid aperture of a side is the
# package's face aperture, and the data of a cut cell are the wetted part of
# each side and its midpoint, shared with the neighbour across the side. No
# polygon is formed.

# Outward normal of side k as a multiple of the face orientation (+axis).
const _MIMETIC_OUTWARD = (-1, 1, 1, -1)
# A counter-clockwise walk starts at the low tangential end of sides 1 and 2
# and at the high end of sides 3 and 4.
const _MIMETIC_STARTS_LOW = (true, true, false, false)

# Face-side flags of a partially wetted interior face.
const _MIMETIC_LIQUID_LOW = Int8(1)
const _MIMETIC_LIQUID_HIGH = Int8(-1)

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

# Tangential component of the unit PLIC normal of `cell` on a face with local
# normal coordinate `normal_coordinate`, and whether the PLIC line cuts the face
# strictly inside. A positive component puts the liquid at the low end.
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
    return tangent / hypot(nx, ny), zero(T) < hi - lo < one(T)
end

"""Classify the wetted end of every partially wetted interior face.

The adjacent cut cells whose PLIC line crosses the face strictly inside decide
by the sign of the sum of their tangential normal components (one crossing
line: its direction; two that disagree: the more tangential one). If no line
crosses, the first cut neighbour decides by the direction of its tangential
normal; without a cut neighbour, or for a vanishing sum, the liquid is at the
low end. Every face gets one shared decision, which is all the consistency of
the metric needs (the wetted midpoint of a face is the same for both cells).
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
        strong = weak = zero(aperture)
        crossings = 0
        for (reference, coordinate) in ((face.minus, 0.5), (face.plus, -0.5))
            fraction = _conditional_fraction(interface, reference)
            zero(fraction) < fraction < one(fraction) || continue
            tangent, crosses = _mimetic_trace_vote(
                interface, reference, face.axis, coordinate,
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

"""Phase boundary on side `k` and the midpoints of its liquid and gas parts in
unit cell coordinates (origin at the cell centre), for the liquid aperture `a`
of the side; `start` puts the liquid part at the counter-clockwise start of the
side. A part of length zero has its midpoint at the side's end."""
@inline function _mimetic_side_parts(::Type{T}, k, a, start) where {T}
    ax, ay = _mimetic_corner(T, k)
    bx, by = _mimetic_corner(T, mod1(k + 1, 4))
    s = start ? a : one(T) - a
    qx, qy = ax + s * (bx - ax), ay + s * (by - ay)
    low = ((ax + qx) / 2, (ay + qy) / 2)
    high = ((qx + bx) / 2, (qy + by) / 2)
    return (qx, qy), start ? low : high, start ? high : low
end

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
mimetic-gram, in units of `h`).

With `M = Σ_k a_k n_k (x^l_k)ᵀ` the liquid Gram tensor at `x*` is
`G^l = M - η x*ᵀ` and the gas one `G^g = I - G^l`. For `η ≠ 0` both are
symmetric exactly on the line `η × x* = τ`, `τ = M₁₂ - M₂₁`. The data point
is the point of that line closest to the interface of the data, the mean of
the phase boundaries on the partially wetted sides (if every side is wetted by
one phase only: the mean of the midpoints of the sides of the minority phase).
It is kept if `0 ⪯ G^l ⪯ I` there; otherwise it moves along the line into
the admissible interval (`_mimetic_admissible_point`). For `η = 0` the tensors
do not depend on `x*`; the point satisfies the kernel condition instead
(`_mimetic_kernel_point`). A point more than `1.5h` from the centre is
replaced by the data mean (the cell is then defective).
"""
function _mimetic_pressure_point(apertures::NTuple{4, T}, starts) where {T}
    ex = ey = sx = sy = zero(T)
    m11 = m12 = m21 = m22 = zero(T)
    crossed = 0
    for k in 1:4
        a = apertures[k]
        nx, ny = _mimetic_normal(T, k)
        (qx, qy), (lx, ly), _ = _mimetic_side_parts(T, k, a, starts[k])
        ex += a * nx
        ey += a * ny
        m11 += a * nx * lx
        m12 += a * nx * ly
        m21 += a * ny * lx
        m22 += a * ny * ly
        if zero(T) < a < one(T)
            sx += qx
            sy += qy
            crossed += 1
        end
    end
    if crossed == 0
        minority = sum(apertures) >= 2 ? zero(T) : one(T)
        for k in 1:4
            apertures[k] == minority || continue
            ax, ay = _mimetic_corner(T, k)
            bx, by = _mimetic_corner(T, mod1(k + 1, 4))
            sx += (ax + bx) / 2
            sy += (ay + by) / 2
            crossed += 1
        end
    end
    xr, yr = crossed > 0 ? (sx / crossed, sy / crossed) : (zero(T), zero(T))
    e2 = ex^2 + ey^2
    if e2 > T(1.0e-18)
        defect = (m12 - m21) - (ex * yr - ey * xr)
        xs, ys, admissible = _mimetic_admissible_point(
            m11, m12, m21, m22, ex, ey, xr - defect * ey / e2, yr + defect * ex / e2,
        )
    else
        xs, ys = _mimetic_kernel_point(apertures, starts, m11, (m12 + m21) / 2, m22, xr, yr)
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
point is clamped into that interval shrunk at both ends by
`min(w/100, _MIMETIC_POINT_MARGIN)` of its length `w` along the line, where both
tensors are positive definite: the point, hence the metric, depends
continuously on the data, and a phase of vanishing area gets a block of order
its wetted length times the margin instead of a block that vanishes or blows
up at the end of the interval. Without an admissible point it stays (the cell
is defective)."""
function _mimetic_admissible_point(m11, m12, m21, m22, ex, ey, xs, ys)
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
    margin = min(max(smax - smin, zero(T)) / 100, T(_MIMETIC_POINT_MARGIN))
    s = clamp(zero(T), smin + margin, max(smin + margin, smax - margin))
    return xs + s * hx, ys + s * hy, true
end

"""Largest distance (units of `h`) by which `_mimetic_admissible_point` keeps
the pressure point inside the ends of its admissible interval."""
const _MIMETIC_POINT_MARGIN = 0.01

"""Pressure point of a cut cell with `η = 0`: the Gram tensors `G^l` (symmetric
part `[g11 g12; g12 g22]`) and `G^g = I - G^l` do not depend on `x*`, and a
block with `m N = R` needs `ker G ⊂ ker R`. For every eigenvector `v` with
eigenvalue `0` of `G^l` (or `1`: kernel of `G^g`) the component `v ⋅ x*` is
set to the wetted-length weighted mean of `v ⋅ x_k` over the sides of that
phase, which satisfies the condition when the `v ⋅ x_k` agree (a film along a
grid line: its mid-height); the other component is that of the data point."""
function _mimetic_kernel_point(apertures::NTuple{4, T}, starts, g11, g12, g22, xs, ys) where {T}
    lambda1, lambda2, c, s = _mimetic_symmetric_eigen(g11, g12, g22)
    tolerance = T(1.0e-10)
    for (lambda, vx, vy) in ((lambda1, -s, c), (lambda2, c, s))
        for phase in 1:2
            (phase == 1 ? lambda : one(T) - lambda) <= tolerance || continue
            total = weighted = zero(T)
            for k in 1:4
                b = phase == 1 ? apertures[k] : one(T) - apertures[k]
                b > zero(T) || continue
                _, liquid, gas = _mimetic_side_parts(T, k, apertures[k], starts[k])
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

For the `m` sides wetted by `phase` (`sides[1:m]`, wetted lengths
`lengths[1:m]`, outward normals `n_k`, wetted midpoints `x_k`) and the pressure
point `(xs, ys)`, fills `moments[1:m, :]` with `R = [b_k (x_k - x*)ᵀ]` and
`block[1:m, 1:m]` with `R G⁺ Rᵀ + S`, `G = Σ_k b_k n_k (x_k - x*)ᵀ`. `S` is
the stabilisation `½ D Q D` of the wetted sides closed by one more edge of
length `|η|` and outward normal `∓η/|η|` (liquid, gas), whose flux is slaved
to close the region (eq. mimetic-slave). Returns `m` and whether the block
is consistent, `block N = R`: that needs a symmetric positive semi-definite
`G` whose kernel lies in that of `R`. Otherwise the eigenvalues of the
symmetric part of `G` are raised to `1/50` (least-squares consistency).
"""
function _mimetic_gram_block!(
        block, sides, lengths, moments, apertures, starts, phase, xs, ys, ex, ey,
    )
    T = eltype(block)
    m = 0
    g11 = g12 = g21 = g22 = zero(T)
    for k in 1:4
        b = phase == 1 ? apertures[k] : one(T) - apertures[k]
        b > zero(T) || continue
        _, liquid, gas = _mimetic_side_parts(T, k, apertures[k], starts[k])
        mx, my = phase == 1 ? liquid : gas
        nx, ny = _mimetic_normal(T, k)
        m += 1
        sides[m] = k
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
        _mimetic_fill_gram_block!(block, m, sides, lengths, moments, lambda1, lambda2, c, s, false, phase, ex, ey)
        for q in 1:m
            rx = ry = zero(T)
            for r in 1:m
                nx, ny = _mimetic_normal(T, sides[r])
                rx += block[q, r] * nx
                ry += block[q, r] * ny
            end
            consistent &= max(abs(rx - moments[q, 1]), abs(ry - moments[q, 2])) <= T(1.0e-8)
        end
    end
    consistent || _mimetic_fill_gram_block!(
        block, m, sides, lengths, moments, lambda1, lambda2, c, s, true, phase, ex, ey,
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
        block, m, sides, lengths, moments, lambda1, lambda2, c, s, floored, phase, ex, ey,
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
    edge(q) = q <= m ? (lengths[q], _mimetic_normal(T, sides[q])) : (cl, (cnx, cny))
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
end

function _MimeticAssembly(::Type{T}) where {T}
    return _MimeticAssembly{T}(
        typemax(UInt), Int[], Int[], Int[], Int8[], T[], T[], T[], 0, Int[], Int[], T[],
        spzeros(T, 0, 0), nothing, T[], T[], 0, 0, 0, 0, 0, T[], Int[], Int[], T[],
        falses(0), Int8[], zeros(T, 4, 4), zeros(Int, 4), zeros(T, 4), zeros(T, 4, 2),
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

"""Assemble the MFD cut-cell metric (candidate MMd of the audit note on
kinetic metrics, with the tie terms of the audit's laboratory).

Every cut cell (a cell whose four liquid apertures are not all `0` or all
`1`) gets one pressure point (`_mimetic_pressure_point`) and per phase the
Gram block of the sides that phase wets (`_mimetic_gram_block!`); a cell in
which a block is only least-squares consistent counts as defective. Uncut
cells give their face values `ρh²/2`. A phase value on a face that the phase
does not reach is tied to a donor with its staggered mass, and with the
small-cell stabilisation every value with aperture below one half on a mixed
face is tied to a donor with weight `(1 - 2a)` times its mass.

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
    cut_cells = defective_cells = 0
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
        dof_of(k, phase) = (
            face = topology.sides[k, cell]; face <= 0 ? 0 :
                (phase == 1 ? assembly.liquid_dof[face] : assembly.gas_dof[face])
        )
        uncut = all(==(apertures[1]), apertures) &&
            (iszero(apertures[1]) || isone(apertures[1]))
        if uncut
            # Uncut cell: (1/2) rho h^2 on every face value, the diagonal metric.
            phase = isone(apertures[1]) ? 1 : 2
            for k in 1:4
                dof = dof_of(k, phase)
                dof > 0 && (assembly.full_diagonal[dof] += density[phase] * h^2 / 2)
            end
            continue
        end
        if !allow_wall_cuts && any(k -> topology.sides[k, cell] < 0, 1:4)
            throw(
                ArgumentError(
                    "MimeticProjection: cut cell $cell touches a domain boundary; " *
                        "the mimetic metric is consistent only for cut cells away " *
                        "from walls",
                ),
            )
        end
        starts = ntuple(Val(4)) do k
            zero(T) < apertures[k] < one(T) || return false
            face = topology.sides[k, cell]
            low = face > 0 ? flags[face] == _MIMETIC_LIQUID_LOW : boundary_low[-face]
            return _MIMETIC_STARTS_LOW[k] ? low : !low
        end
        xs, ys, ex, ey = _mimetic_pressure_point(apertures, starts)
        assembly.pressure_x[cell] = center[1] + h * xs
        assembly.pressure_y[cell] = center[2] + h * ys
        cut_cells += 1
        defective = false
        for phase in 1:2
            m, consistent = _mimetic_gram_block!(
                assembly.block, assembly.sides, assembly.lengths, assembly.moments,
                apertures, starts, phase, xs, ys, ex, ey,
            )
            defective |= !consistent
            scale = density[phase] * h^2
            for q in 1:m
                kq = assembly.sides[q]
                rq = dof_of(kq, phase)
                if rq == 0
                    topology.sides[kq, cell] < 0 && continue
                    error("mimetic metric: wetted face without a phase value")
                end
                for r in 1:m
                    kr = assembly.sides[r]
                    rr = dof_of(kr, phase)
                    rr > 0 || continue
                    _mimetic_add!(
                        assembly, rq, rr,
                        _MIMETIC_OUTWARD[kq] * _MIMETIC_OUTWARD[kr] * scale * assembly.block[q, r],
                    )
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
    for dof in 1:dofs
        face = assembly.dof_face[dof]
        phase = assembly.dof_phase[dof]
        aperture = assembly.dof_aperture[dof]
        mass = assembly.dof_mass[dof]
        dof_at(g) = g == 0 ? 0 : (phase == 1 ? assembly.liquid_dof[g] : assembly.gas_dof[g])
        axis = mesh.faces[face].axis
        if aperture > zero(T)
            # A block that gives the value no inertia beyond round-off (a phase
            # of vanishing area) does not count: lump the staggered mass.
            if !(block_diagonal[dof] > T(1.0e-12) * density[phase == 2 ? 2 : 1] * h^2)
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
            ring = _mimetic_face_ring(topology, face, axis)
            donor = _mimetic_donor(topology, mesh, ring, dof_at, larger, assembly.dof_aperture)
            donor == 0 && continue
            _mimetic_add_tie!(assembly, dof, donor, (one(T) - 2aperture) * mass)
            stabilised += 1
            continue
        end
        ring = _mimetic_face_ring(topology, face, axis)
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
    assembly.defective_cells = defective_cells
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
