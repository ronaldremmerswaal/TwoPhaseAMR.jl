# Helpers shared by the audits of Chapter 6 (momentum and staggered transport) of the technical reference.
#
# Not a check by itself.  Needs the TwoPhaseAMR package (repository environment) and is included with
#     isdefined(Main, :MomentumAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "momentum_audit.jl"))
#     using .MomentumAudit
#
# `MomentumAudit.Docs` wraps the fixtures and independent evaluations of docs/techref/figures/{cutcells,vof,momentum}.jl in a
# module of its own (with the polygon helpers of `_shared/techref_geometry.jl` standing in for interface.jl).  On top of it this
# file adds
#   * random composite meshes (periodic or walled, 2:1 balanced by the mesh), random solids and interfaces,
#   * the mesh-only incidence data of the rules (N) and (T) of Chapter 6 (cells, sides, fractions, orientations) and the
#     right-hand side of the divergence identity built from them,
#   * dense assembly of the package's convection operator from `convection!`,
#   * the code's dual schedule in a normalised form (tail, head, axis) -> (primal flux -> weight).

module MomentumAudit

import TwoPhaseAMR
using TwoPhaseAMR
using LinearAlgebra
using Random
using Printf

const AMR = TwoPhaseAMR
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

# ---------------------------------------------------------------------------------------------------------
# The docs fixtures and independent evaluations, in a module of their own.
# ---------------------------------------------------------------------------------------------------------
module Docs
    using TwoPhaseAMR
    using Printf: @sprintf
    const AMR = TwoPhaseAMR
    include(joinpath(@__DIR__, "techref_geometry.jl"))
    using .Geo
    const SQUARE = NTuple{2, Float64}[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
    function fmt(x::Real)
        isinteger(x) && return string(Int(x))
        text = rstrip(rstrip(@sprintf("%.6f", x), '0'), '.')
        return text == "-0" ? "0" : text
    end
    sci(x) = iszero(x) ? "0" : begin
            exponent = floor(Int, log10(abs(x)))
            @sprintf("%.1f\\times10^{%d}", x / 10.0^exponent, exponent)
        end
    fix3(x) = @sprintf("%.3f", x)
    function frac(x::Real)
        q = rationalize(Int, x; tol = 1.0e-12)
        denominator(q) == 1 && return string(numerator(q))
        sign = q < 0 ? "-" : ""
        return "$(sign)\\tfrac{$(abs(numerator(q)))}{$(denominator(q))}"
    end
    const FIGURES = joinpath(@__DIR__, "..", "..", "docs", "techref", "figures")
    include(joinpath(FIGURES, "cutcells.jl"))
    include(joinpath(FIGURES, "vof.jl"))
    include(joinpath(FIGURES, "momentum.jl"))
end # module Docs

const D = Docs

# ---------------------------------------------------------------------------------------------------------
# Random fixtures.
# ---------------------------------------------------------------------------------------------------------

"""Random composite mesh (2:1 balance is enforced by the mesh)."""
function random_mesh(
        rng; periodic = (true, true), levels = 1, base = (12, 12), block = (4, 4),
        extent = (12.0, 12.0), refine_probability = 0.4
    )
    flags = Dict{BlockKey, Bool}()
    refine = (key, origin, ext) ->
    get!(() -> rand(rng) < refine_probability / (key.level + 1), flags, key)
    return BlockTreeMesh(;
        base_mesh = base, block_size = block, extent, periodic,
        max_level = levels, refine
    )
end

"""Smooth random streamfunction `psi(x, y)` on a box of the given extent, vanishing on the walls when `walls`."""
function random_streamfunction(rng, extent; walls = false, modes = 3)
    lx, ly = extent
    a = [randn(rng) for _ in 1:modes, _ in 1:modes]
    p = [2pi * rand(rng) for _ in 1:modes, _ in 1:modes]
    q = [2pi * rand(rng) for _ in 1:modes, _ in 1:modes]
    if walls
        return (x, y) -> sum(
            a[m, n] * sinpi(m * x / lx) * sinpi(n * y / ly) for m in 1:modes, n in 1:modes
        ) * lx / (modes^2)
    end
    return (x, y) -> sum(
        a[m, n] * sin(2pi * m * x / lx + p[m, n]) * sin(2pi * n * y / ly + q[m, n])
            for m in 1:modes, n in 1:modes
    ) * lx / (modes^2)
end

"""Face velocity from a vertex streamfunction (exactly solenoidal on every leaf cell, also at hanging vertices)."""
function stream_velocity(mesh, psi)
    velocity = FaceVelocity(mesh)
    for (index, face) in pairs(mesh.faces)
        x, y = face.center
        h = face.area / 2
        velocity.interior[index] = face.axis == 1 ? (psi(x, y + h) - psi(x, y - h)) / face.area :
            -(psi(x + h, y) - psi(x - h, y)) / face.area
    end
    return velocity
end

"""Random (generally non-solenoidal) face velocity with i.i.d. normal values; zero on every boundary face."""
function random_velocity(rng, mesh; scale = 1.0)
    velocity = FaceVelocity(mesh)
    for index in eachindex(velocity.interior)
        velocity.interior[index] = scale * randn(rng)
    end
    return velocity
end

# ---------------------------------------------------------------------------------------------------------
# Mesh-only incidence data of the rules (N) and (T).
# ---------------------------------------------------------------------------------------------------------

"""
Incidences of the faces of axis `k` in the cells: `sides[cell][s]` is the list of `(local face, fraction)` on the low
(`s = -1`) or high (`s = +1`) side of `cell`; `orient[(cell, local face)]` is `o_{c,f}`.  The local face index is the
position in the component-`k` subsequence of `mesh.faces`.  With `boundary = true` the boundary faces of axis `k` follow as
extra unknowns (the phase schedule); otherwise they are absent.
"""
function component_incidence(mesh, k; boundary = false)
    t = 3 - k
    interior = [i for (i, f) in pairs(mesh.faces) if f.axis == k]
    local_of = Dict(g => l for (l, g) in enumerate(interior))
    sides = Dict{CellRef, Dict{Int, Vector{Tuple{Int, Float64}}}}()
    orient = Dict{Tuple{CellRef, Int}, Int}()
    entry(cell) = get!(sides, cell) do
        Dict(-1 => Tuple{Int, Float64}[], 1 => Tuple{Int, Float64}[])
    end
    for (l, g) in enumerate(interior)
        f = mesh.faces[g]
        for (cell, s) in ((f.minus, 1), (f.plus, -1))
            push!(entry(cell)[s], (l, f.area / mesh.blocks[cell.block].spacing[t]))
            orient[(cell, l)] = s
        end
    end
    n = length(interior)
    boundary_faces = Int[]
    if boundary
        for (b, f) in pairs(mesh.boundary_faces)
            f.axis == k || continue
            n += 1
            push!(boundary_faces, b)
            push!(entry(f.cell)[f.side], (n, f.area / mesh.blocks[f.cell.block].spacing[t]))
            orient[(f.cell, n)] = f.side
        end
    end
    return (; k, interior, local_of, sides, orient, n, boundary_faces)
end

"""All cells of the mesh."""
function all_cell_refs(mesh)
    refs = CellRef[]
    nx, ny = mesh.block_size
    for b in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        push!(refs, CellRef(b, i, j))
    end
    return refs
end

"""Sum of the fractions on side `s` of `cell` for component incidence `inc` (0 for a wall side without unknown)."""
side_total(inc, cell, s) = sum(
    (fr for (_, fr) in get(inc.sides, cell, Dict(-1 => Tuple{Int, Float64}[], 1 => Tuple{Int, Float64}[]))[s]);
    init = 0.0
)

"""
Weights (N) and (T) of Chapter 6 from the incidences alone, as `Dict((tail, head, axis) => Dict(primal flux => weight))`
with tail < head (a reversed orientation changes the sign of the weights).
"""
function rule_weights(mesh, inc)
    k = inc.k
    weights = Dict{NTuple{3, Int}, Dict{Int, Float64}}()
    function add!(from, to, axis, flux, w)
        @assert from != to
        key = from < to ? (from, to, axis) : (to, from, axis)
        d = get!(weights, key, Dict{Int, Float64}())
        d[flux] = get(d, flux, 0.0) + (from < to ? w : -w)
        return nothing
    end
    empty_side = Dict(-1 => Tuple{Int, Float64}[], 1 => Tuple{Int, Float64}[])
    for (index, face) in pairs(mesh.faces)
        if face.axis == k
            l = inc.local_of[index]
            for (cell, sigma) in ((face.minus, 1.0), (face.plus, -1.0)), s in (-1, 1)
                for (g, varpi) in get(inc.sides, cell, empty_side)[s]
                    g == l || add!(g, l, k, index, sigma * varpi / 2)
                end
            end
        else
            for s in (-1, 1)
                for (g, a) in get(inc.sides, face.minus, empty_side)[s],
                        (gp, ap) in get(inc.sides, face.plus, empty_side)[s]
                    add!(g, gp, face.axis, index, a * ap / 2)
                end
            end
        end
    end
    for key in collect(keys(weights))
        d = weights[key]
        for f in collect(keys(d))
            abs(d[f]) <= 1.0e-15 && delete!(d, f)
        end
        isempty(d) && delete!(weights, key)
    end
    return weights
end

"""The package's schedule of a workspace in the same normalised form (tail < head); `scale` multiplies by the aperture."""
function schedule_weights(ws; mesh = nothing)
    schedule = ws.schedule
    weights = Dict{NTuple{3, Int}, Dict{Int, Float64}}()
    if schedule isa AMR.AMRDualAdvectionSchedule
        for face in schedule.faces
            d = Dict{Int, Float64}()
            for p in face.first_weight:(face.first_weight + face.weight_count - 1)
                d[schedule.flux_indices[p]] = schedule.flux_weights[p]
            end
            key = (face.minus, face.plus, face.axis)
            @assert face.minus < face.plus
            weights[key] = d
        end
    else
        @assert mesh !== nothing "the uniform schedule stores areas; pass the mesh"
        for face in schedule.faces
            sign = face.minus < face.plus ? 1.0 : -1.0
            key = (min(face.minus, face.plus), max(face.minus, face.plus), face.axis)
            d = get!(weights, key, Dict{Int, Float64}())
            for flux in (face.flux_a, face.flux_b)
                d[flux] = get(d, flux, 0.0) + sign * face.area / (2mesh.faces[flux].area)
            end
        end
    end
    return weights
end

"""
Largest absolute difference of two normalised weight tables over the keys accepted by `keep` (`Inf` if the accepted key sets
differ); `scale` multiplies the weights of the first table per primal flux (apertures).
"""
function weight_gap(a, b; scale = nothing, keep = key -> true)
    ka = Set(key for key in keys(a) if keep(key))
    kb = Set(key for key in keys(b) if keep(key))
    ka == kb || return Inf
    worst = 0.0
    for key in ka
        d, e = a[key], b[key]
        for flux in union(keys(d), keys(e))
            s = scale === nothing ? 1.0 : scale[flux]
            worst = max(worst, abs(get(d, flux, 0.0) * s - get(e, flux, 0.0)))
        end
    end
    return worst
end

"""
Dual-divergence operator `Q` (rows: unknowns, columns: primal faces of ALL axes) of the dual faces of axis `t`:
`(Q Phi)_f = sum_{g: axis(g) = t} sigma_{f,g} Upsilon_g` with `Upsilon_g = sum_h w_{g,h} Phi_h`, `Phi_h` the primal flux.
"""
function dual_divergence_matrix(weights, n, nfluxes, t)
    Q = zeros(n, nfluxes)
    for ((i, j, axis), d) in weights
        axis == t || continue
        for (h, w) in d
            Q[i, h] += w
            Q[j, h] -= w
        end
    end
    return Q
end

"""
Right-hand side operator `P` of the divergence identity for axis `t`:
`(P Phi)_f = sum_{c in C(f)} (varpi_{c,f}/2) sum_{h in F(c) cap F^(t)} o_{c,h} Phi_h`.
Rows are the unknowns of component `inc.k`; the fractions come from the incidences `inc`.
"""
function identity_rhs_matrix(mesh, inc, nfluxes, t)
    P = zeros(inc.n, nfluxes)
    for (index, face) in pairs(mesh.faces)
        face.axis == t || continue
        for (cell, o) in ((face.minus, 1.0), (face.plus, -1.0))
            entry = get(inc.sides, cell, nothing)
            entry === nothing && continue
            for s in (-1, 1), (g, varpi) in entry[s]
                P[g, index] += varpi / 2 * o
            end
        end
    end
    return P
end

"""Faces of every cell with the orientation `o_{c,h}` (+1 if `h` lies on the high side of `c`): `Dict(cell => [(index, o)])`."""
function faces_of_cells(mesh)
    result = Dict{CellRef, Vector{Tuple{Int, Int}}}()
    for (index, face) in pairs(mesh.faces)
        push!(get!(result, face.minus, Tuple{Int, Int}[]), (index, +1))
        push!(get!(result, face.plus, Tuple{Int, Int}[]), (index, -1))
    end
    return result
end

"""
Closed-form residual `Q - P` of the divergence identity for incomplete side partitions (exact for arbitrary fractions):
axial part (`t == k`): `-(1/2) Phi_f (S_{c^-} - S_{c^+})` in column `f`, transverse part: `(1/2) varpi_{c,f} o_{c,h}
(S^{s(c)}_{c'(h)} - 1)` in column `h`, with `S_c` the total fraction of the axial faces of `c`, `S^s_{c'}` that on side `s`.
"""
function predicted_residual_matrix(mesh, inc, nfluxes, t; cell_faces = faces_of_cells(mesh))
    k = inc.k
    R = zeros(inc.n, nfluxes)
    for (l, g) in enumerate(inc.interior)
        face = mesh.faces[g]
        if t == k
            S(c) = side_total(inc, c, -1) + side_total(inc, c, 1)
            R[l, g] = -(S(face.minus) - S(face.plus)) / 2
        else
            for (c, s) in ((face.minus, +1), (face.plus, -1))
                varpi = face.area / mesh.blocks[c.block].spacing[3 - k]
                for (h, o) in get(cell_faces, c, Tuple{Int, Int}[])
                    hf = mesh.faces[h]
                    hf.axis == t || continue
                    other = hf.minus == c ? hf.plus : hf.minus
                    R[l, h] += varpi / 2 * o * (side_total(inc, other, s) - 1)
                end
            end
        end
    end
    return R
end

"""The printed hypothesis of eq:momentum:identity at interior unknown `g`: both cells have unknowns on both sides."""
hypothesis_h0(mesh, inc, g) = all(
    c -> isapprox(side_total(inc, c, -1), 1; atol = 1.0e-12) && isapprox(side_total(inc, c, 1), 1; atol = 1.0e-12),
    (mesh.faces[g].minus, mesh.faces[g].plus),
)

"""Every cell across a transverse face of a cell of `g` has an unknown on the side where `g` lies in that cell (T-completeness)."""
function t_complete(mesh, inc, g; cell_faces = faces_of_cells(mesh))
    face = mesh.faces[g]
    for (c, s) in ((face.minus, +1), (face.plus, -1))
        for (h, o) in get(cell_faces, c, Tuple{Int, Int}[])
            hf = mesh.faces[h]
            hf.axis == inc.k && continue
            other = hf.minus == c ? hf.plus : hf.minus
            isapprox(side_total(inc, other, s), 1; atol = 1.0e-12) || return false
        end
    end
    return true
end

"""
Class of the interior unknown `g` of component `inc.k` (see the audit of eq:momentum:identity):
`:regular` (hypothesis H0), `:wall` (a wall cell, T-complete, face fully open: repaired by the boundary dual faces),
`:tincomplete` (limitation (i), issue 60), `:cutwall` (limitation (ii), issue 61: a wall cell whose face aperture is below one).
"""
function unknown_class(mesh, inc, g, aperture; cell_faces = faces_of_cells(mesh))
    t_complete(mesh, inc, g; cell_faces) || return :tincomplete
    hypothesis_h0(mesh, inc, g) && return :regular
    return aperture[g] < 1 ? :cutwall : :wall
end

"""Report a result about a configuration that the claim excludes (a known defect); it never fails a run."""
excluded(name, value; note = "") = println("[excluded configuration] ", name, ": ", value, isempty(note) ? "" : "  -- " * note)

"""Random smooth solid: a disc (`kind = :disc`) or a sloped half-plane across the box (`:wall`); returns the level set."""
function random_solid(rng, extent; kind = :disc)
    lx, ly = extent
    cx, cy = lx * (0.3 + 0.4 * rand(rng)), ly * (0.3 + 0.4 * rand(rng))
    r = min(lx, ly) * (0.1 + 0.12 * rand(rng))
    slope = (rand(rng) - 0.5) * 0.8
    off = ly * (0.15 + 0.15 * rand(rng))
    disc = (x, y) -> hypot(x - cx, y - cy) - r
    wall = (x, y) -> y - off - slope * (x - lx / 2)
    kind == :disc && return disc
    kind == :wall && return wall
    error("unknown solid kind $kind")
end

"""Centres (x, y) of the unknowns of component `inc.k`, boundary unknowns last."""
function unknown_centres(mesh, inc)
    centres = NTuple{2, Float64}[mesh.faces[g].center for g in inc.interior]
    for b in inc.boundary_faces
        push!(centres, mesh.boundary_faces[b].center)
    end
    return centres
end

# ---------------------------------------------------------------------------------------------------------
# Dense assembly of the package's convection operator.
# ---------------------------------------------------------------------------------------------------------

"""Transport volumes `V_f` of the unknowns stored in a staggered workspace (uniform schedule: one constant)."""
schedule_volumes(ws, n) = ws.schedule isa AMR.AMRDualAdvectionSchedule ? copy(ws.schedule.volumes) :
    fill(Float64(ws.schedule.volume), n)

"""
Dual divergence `V_f (div Upsilon)_f = sum_g sigma_{f,g} Upsilon_g` (both axes) of the interior dual faces for the primal fluxes
`Phi_h = |h| a_h v_h`, from the rule weights (apertures included in `Phi`).
"""
function dual_divergence(mesh, inc, Phi)
    nf = length(mesh.faces)
    w = rule_weights(mesh, inc)
    return (dual_divergence_matrix(w, inc.n, nf, 1) .+ dual_divergence_matrix(w, inc.n, nf, 2)) * Phi
end

"""
Boundary dual divergence `sum_{b in B(f)} sigma_b Upsilon_b` of eq:momentum:boundary at every unknown of component `k` of an
uncut single-phase schedule: for a wall face `b` normal to `e_k` the opposite faces `f` of the wall cell give
`sigma_b (|f|/2)(v_b + v_f)` at `f`; for a wall face `b` along the other axis every unknown `f` of the wall cell gets
`sigma_b (1/2) varpi_{c,f} |b| v_b`.  Uses the uncut measures, as the text states.
"""
function boundary_dual_divergence(mesh, inc, velocity)
    k = inc.k
    result = zeros(inc.n)
    for (b, bf) in pairs(mesh.boundary_faces)
        cell = bf.cell
        entry = get(inc.sides, cell, nothing)
        entry === nothing && continue
        if bf.axis == k
            for (l, varpi) in entry[-bf.side]
                f = mesh.faces[inc.interior[l]]
                result[l] += bf.side * f.area / 2 * (velocity.boundary[b] + velocity.interior[inc.interior[l]])
            end
        else
            for s in (-1, 1), (l, varpi) in entry[s]
                result[l] += bf.side * varpi / 2 * bf.area * velocity.boundary[b]
            end
        end
    end
    return result
end

"""
Dense matrix `C` of `q -> convection!(...)` for component `k` (columns = unit vectors on the interior unknowns), with the
boundary values of `q` held at `boundary_value`; also returns the constant offset `C(0)` (non-zero only for non-zero boundary
values or inflow walls).
"""
function assemble_convection(
        mesh, geometry, velocity, k; scheme = Centred(), amr_interface_scheme = Centred(), boundary = nothing,
        boundary_value = 0.0, dt = 0.0
    )
    basis = FaceField{k}(mesh)
    column = FaceField{k}(mesh)
    ws = D.staggered_workspace(basis, geometry)
    n = length(basis.interior)
    fill!(basis.boundary, boundary_value)
    fill!(basis.interior, 0.0)
    convection!(column, basis, velocity, dt, ws; scheme, amr_interface_scheme, boundary)
    offset = copy(column.interior)
    C = zeros(n, n)
    for source in 1:n
        fill!(basis.interior, 0.0)
        basis.interior[source] = 1.0
        convection!(column, basis, velocity, dt, ws; scheme, amr_interface_scheme, boundary)
        C[:, source] .= column.interior .- offset
    end
    return C, offset, ws
end

# ---------------------------------------------------------------------------------------------------------
# Two-phase fixtures and the replay of one-velocity sweeps.
# ---------------------------------------------------------------------------------------------------------

"""A random smooth interface level set (liquid where negative): drop, wavy film or tilted ellipse; returns `(levelset, lipschitz)`."""
function random_interface(rng, extent)
    kind = rand(rng, 1:3)
    lx, ly = extent
    if kind == 1
        cx, cy = lx * (0.3 + 0.4 * rand(rng)), ly * (0.3 + 0.4 * rand(rng))
        r = min(lx, ly) * (0.15 + 0.2 * rand(rng))
        sgn = rand(rng, Bool) ? 1.0 : -1.0                       # drop or bubble
        return (x, y) -> sgn * (hypot(x - cx, y - cy) - r), 1
    elseif kind == 2
        y0 = ly * (0.3 + 0.4 * rand(rng)); amp = 0.05 * ly * rand(rng); k = rand(rng, 1:3)
        return (x, y) -> y - y0 - amp * sinpi(2k * x / lx), 1 + ceil(Int, 2pi * k * amp / lx)
    end
    cx, cy = lx * (0.3 + 0.4 * rand(rng)), ly * (0.3 + 0.4 * rand(rng))
    a, b = min(lx, ly) * (0.2 + 0.15 * rand(rng)), min(lx, ly) * (0.1 + 0.1 * rand(rng))
    th = pi * rand(rng)
    c, s = cos(th), sin(th)
    return (x, y) -> begin
            u = c * (x - cx) + s * (y - cy); v = -s * (x - cx) + c * (y - cy)
            (hypot(u / a, v / b) - 1) * min(a, b)
        end, 2
end

"""
Face velocity `u = curl psi + grad phi` (finite differences of a vertex streamfunction and of a cell-centre potential across each
face), zero on the boundary faces; the potential part makes the carrier non-solenoidal.  `solid` (level set, solid where `<= 0`)
makes the streamfunction vanish on solid vertices.  Returns the velocity, `solenoidal` part only if `potential = 0`.
"""
function smooth_velocity(rng, mesh, geometry, solid; potential = 0.4, walls = false, amplitude = 1.0)
    psi = random_streamfunction(rng, mesh.extent; walls)
    velocity = if geometry === nothing
        stream_velocity(mesh, psi)
    else
        D.solid_streamfunction_velocity(mesh, geometry, solid, psi)
    end
    lx, ly = mesh.extent
    a, b = 0.4 + 0.6 * rand(rng), 2pi * rand(rng)
    m, n = rand(rng, 1:2), rand(rng, 1:2)
    phi = walls ? (x, y) -> a * cospi(m * x / lx) * cospi(n * y / ly) :
        (x, y) -> a * sin(2pi * m * x / lx + b) * sin(2pi * n * y / ly + b)
    if potential != 0
        for (index, face) in pairs(mesh.faces)
            k = face.axis
            hm = mesh.blocks[face.minus.block].spacing[k]
            hp = mesh.blocks[face.plus.block].spacing[k]
            lo = k == 1 ? (face.center[1] - hm / 2, face.center[2]) : (face.center[1], face.center[2] - hm / 2)
            hi = k == 1 ? (face.center[1] + hp / 2, face.center[2]) : (face.center[1], face.center[2] + hp / 2)
            velocity.interior[index] += potential * (phi(hi...) - phi(lo...)) / ((hm + hp) / 2)
        end
    end
    velocity.interior .*= amplitude
    geometry === nothing || enforce_geometry_no_slip!(velocity, geometry)
    return velocity
end

"""
A random two-phase fixture `(field, velocity, solid)`: composite mesh (periodic or walled), optional solid (`:disc` or `:wall`),
random interface, smooth non-solenoidal carrier.  The solid and the streamfunction are redrawn until the open fluxes telescope.
"""
function random_two_phase(
        rng; periodic = (true, true), solid_kind = nothing, levels = 1, refine_probability = 0.5,
        fraction_floor = nothing, potential = 0.4
    )
    for attempt in 1:100
        mesh = random_mesh(rng; periodic, levels, refine_probability)
        solid = solid_kind === nothing ? nothing : random_solid(rng, mesh.extent; kind = solid_kind)
        geometry = solid === nothing ? nothing : GeometryField(mesh, solid; lipschitz = 2)
        levelset, lipschitz = random_interface(rng, mesh.extent)
        field = try
            kwargs = fraction_floor === nothing ? (;) : (; fraction_floor)
            InterfaceField(mesh, levelset; geometry, lipschitz, kwargs...)
        catch err
            err isa ErrorException || rethrow()
            continue
        end
        velocity = try
            smooth_velocity(rng, mesh, geometry, solid; potential, walls = !all(periodic))
        catch err
            err isa AssertionError || rethrow()
            continue
        end
        return field, velocity, solid
    end
    error("no admissible two-phase fixture found")
end

"""Dof `i` of a phase schedule: its face (interior or boundary) and its cells."""
function dof_face(mesh, dofs, i)
    n_int = length(dofs.interior)
    return i <= n_int ? mesh.faces[dofs.interior[i]] : mesh.boundary_faces[dofs.boundary_faces[i - n_int]]
end

"""
Replay of the sweeps of one `advect!(...; momentum, density)` of the one-velocity model from the code's primal phase volumes
(`work.open_flux`, `work.liquid_flux`, `work.gas_flux` hold the volumes of both sweeps after the call), the fractions `C0` before
the step and the sweep order.  For every sweep and component it returns, per unknown, the mass-identity defect per unit volume
(old density = density of the stored, snapped fractions; `defect_literal` with the pre-snap predicted density of the previous
sweep), the predicted defect `d_f = (1/V_f)[ -sum_c (w_c - varpi O_c / 2) mu_c + sum_c w_c (rl - rg) Delta_c ]`, the flags
clamp / condition (b) / snap in the previous sweep, the dilation term `R`, and the schedule.  With `velocity` (the carrier before the step) and the `scheme` of the call it also replays the
velocity update of every sweep (independent replica of eq:momentum:one-velocity) and returns `v_before`, `v_after` per component.
"""
function replay_mass_identity(field, work, C0, order, density; velocity = nothing, scheme = nothing)
    mesh, geometry = field.mesh, field.geometry
    cells = D.all_cells(mesh)
    floor = field.fraction_floor
    rho_l, rho_g = density
    mixture(x) = rho_g + (rho_l - rho_g) * x
    chi = Dict(ref => C0[ref] > 0.5 ? 1.0 : 0.0 for ref in cells)
    schedules = [D.phase_schedule(mesh, geometry, k) for k in (1, 2)]
    C = copy(C0)
    prev_pred = nothing
    sweeps = Any[]
    interface_scheme = scheme === nothing ? nothing : default_amr_interface_scheme(scheme)
    values = velocity === nothing ? nothing : [D.component_values(velocity, schedules[k].dofs) for k in (1, 2)]
    sources = values === nothing ? nothing : deepcopy(values)
    for kappa in order
        sigma, dC, _ = D.sweep_cell_terms(field, work, kappa, chi)
        check = Dict(r => C[r] + dC[r] for r in cells)
        pred = Dict(r => clamp(check[r], 0.0, 1.0) for r in cells)
        snapped = Dict(r => D.snap_fraction(check[r], floor) for r in cells)
        rhochi(r) = rho_l * chi[r] + rho_g * (1 - chi[r])
        comps = Any[]
        for k in (1, 2)
            s = schedules[k]
            n = s.dofs.n
            old = D.independent_interpolation(s.interpolation, r -> C[r])
            new = D.independent_interpolation(s.interpolation, r -> pred[r])
            old_literal = prev_pred === nothing ? old : D.independent_interpolation(s.interpolation, r -> prev_pred[r])
            R = D.independent_interpolation(s.interpolation, r -> rhochi(r) * sigma[r])
            mass = zeros(n)
            for (rho, flux) in ((rho_l, work.liquid_flux), (rho_g, work.gas_flux))
                for ((i, j, axis), r) in D.phase_dual_volumes(s, flux, kappa)
                    mass[i] += rho * r / s.V[i]
                    mass[j] -= rho * r / s.V[j]
                end
            end
            defect = mixture.(new) .- mixture.(old) .+ mass .- R
            defect_literal = defect .+ (mixture.(old) .- mixture.(old_literal))
            predicted = zeros(n); b_term = zeros(n); clamp_term = zeros(n)
            clamp_flag = falses(n); b_flag = trues(n); snap_flag = falses(n); delta_clamp = zeros(n)
            for i in 1:n
                face = dof_face(mesh, s.dofs, i)
                total = 0.0; tb = 0.0; tc = 0.0
                for (cell, w) in s.interpolation[i]
                    O = D.open_volume_of(field, cell)
                    O > 0 || continue
                    varpi = face.area / mesh.blocks[cell.block].spacing[3 - k]
                    mu = rhochi(cell) * sigma[cell] - (rho_l - rho_g) * dC[cell]
                    delta = pred[cell] - check[cell]
                    tb += -(w - varpi * O / 2) * mu
                    tc += w * (rho_l - rho_g) * delta
                    abs(delta) > 1.0e-12 && (clamp_flag[i] = true)
                    delta_clamp[i] = max(delta_clamp[i], abs(delta))
                    uncut = isnothing(geometry) || D.open_fraction_of_geometry(geometry, cell) >= 1
                    b_flag[i] &= uncut || abs(varpi - 1) <= 64eps()
                    prev_pred !== nothing && abs(prev_pred[cell] - C[cell]) > 1.0e-13 && (snap_flag[i] = true)
                end
                predicted[i] = (tb + tc) / s.V[i]
                b_term[i] = tb / s.V[i]
                clamp_term[i] = tc / s.V[i]
            end
            v_before = v_after = nothing
            if values !== nothing
                v_before = copy(values[k])
                increment = zeros(n)
                for (rho, flux, liquid) in ((rho_l, work.liquid_flux, true), (rho_g, work.gas_flux, false))
                    phase_new = liquid ? new : 1 .- new
                    increment .+= rho .* D.phase_increment(
                        mesh, s, values[k], flux, kappa, phase_new, work.ctu_threshold, scheme, interface_scheme,
                    )
                end
                for i in 1:n
                    values[k][i] += (increment[i] + R[i] * (sources[k][i] - values[k][i])) / mixture(new[i])
                end
                v_after = copy(values[k])
            end
            push!(comps, (; k, n, defect, defect_literal, predicted, b_term, clamp_term, clamp_flag, b_flag, snap_flag, delta_clamp, R, mass, new, old, schedule = s, v_before, v_after))
        end
        push!(sweeps, (; kappa, check, pred, snapped, sigma, dC, chi, comps, C_old = copy(C)))
        prev_pred = pred
        C = snapped
    end
    return sweeps, C
end

# ---------------------------------------------------------------------------------------------------------
# Partial volumes of the dual faces at an unknown.
# ---------------------------------------------------------------------------------------------------------

"""
Total outflow `O_f` and inflow `I_f` of the term-wise partial volumes `p = w_{g,h} Phi_h` of the dual faces of axis `kappa` at every
unknown of the phase schedule `s` (`flux[h]` = primal volume of face `h`), and the same totals `On`, `In` for the net dual volumes.
"""
function partial_in_out(s, flux, kappa)
    n = s.dofs.n
    O = zeros(n); I = zeros(n); On = zeros(n); In = zeros(n)
    for ((i, j, axis), d) in s.weights
        axis == kappa || continue
        net = 0.0
        for (h, w) in d
            p = w * flux[h]
            net += p
            if p > 0
                O[i] += p; I[j] += p
            elseif p < 0
                I[i] -= p; O[j] -= p
            end
        end
        if net > 0
            On[i] += net; In[j] += net
        else
            In[i] -= net; On[j] -= net
        end
    end
    return O, I, On, In
end

# ---------------------------------------------------------------------------------------------------------
# Worst-case Courant number of a dual face under the WY time-step bound (exact, by small linear programs).
# ---------------------------------------------------------------------------------------------------------

"""
`max c'x` subject to `A x <= b`, `x >= 0`, `b >= 0` (the origin is feasible), by the tableau simplex with Bland's rule.
Returns the optimal value; the feasible set must be bounded in the directions with `c > 0`.
"""
function lp_max(c, A, b)
    m, n = size(A)
    T = zeros(m + 1, n + m + 1)
    T[1:m, 1:n] .= A
    for i in 1:m
        T[i, n + i] = 1.0
    end
    T[1:m, end] .= b
    T[end, 1:n] .= -c
    basis = collect((n + 1):(n + m))
    for iteration in 1:10_000
        j = 0
        for jj in 1:(n + m)
            if T[end, jj] < -1.0e-12
                j = jj
                break
            end
        end
        j == 0 && return T[end, end]
        i = 0
        best = Inf
        for ii in 1:m
            if T[ii, j] > 1.0e-12
                ratio = T[ii, end] / T[ii, j]
                if ratio < best - 1.0e-12 || (abs(ratio - best) <= 1.0e-12 && basis[ii] < basis[i])
                    best = ratio
                    i = ii
                end
            end
        end
        i == 0 && error("unbounded linear program")
        T[i, :] ./= T[i, j]
        for ii in 1:(m + 1)
            ii == i && continue
            T[ii, j] != 0 && (T[ii, :] .-= T[ii, j] .* T[i, :])
        end
        basis[i] = j
    end
    error("simplex did not terminate")
end

"""
Local problem of the worst-case Courant number of the dual face `key` of the phase schedule `sched` (see `solve_dual_courant`):
the weights `w_h` on the involved primal fluxes, per involved face the volumes of its two cells and the extrusion caps
`|h| a_h h_donor` for a positive / negative flux, the incidence of the involved faces in their cells, and `V_min`.
The canonical tuple (rounded) is the signature used to remove duplicates.
"""
function dual_courant_problem(mesh, sched, key; aper = nothing, open_volume = nothing)
    weights = sched.weights[key]
    faces = sort(collect(keys(weights)))
    vol(c) = open_volume === nothing ? mesh.blocks[c.block].volume : open_volume(c)
    a(h) = aper === nothing ? 1.0 : aper[h]
    cells = unique(vcat([[mesh.faces[h].minus, mesh.faces[h].plus] for h in faces]...))
    cell_volume = [vol(c) for c in cells]
    incidence = [Tuple{Int, Float64}[] for _ in cells]          # per cell: (face position, orientation o_{c,h})
    for (i, h) in enumerate(faces)
        f = mesh.faces[h]
        push!(incidence[findfirst(==(f.minus), cells)], (i, 1.0))
        push!(incidence[findfirst(==(f.plus), cells)], (i, -1.0))
    end
    cap_plus = [mesh.faces[h].area * a(h) * mesh.blocks[mesh.faces[h].minus.block].spacing[mesh.faces[h].axis] for h in faces]
    cap_minus = [mesh.faces[h].area * a(h) * mesh.blocks[mesh.faces[h].plus.block].spacing[mesh.faces[h].axis] for h in faces]
    return (;
        w = [weights[h] for h in faces], cell_volume, incidence, cap_plus, cap_minus,
        Vmin = min(sched.V[key[1]], sched.V[key[2]]),
    )
end

"""Canonical, rounded signature of a `dual_courant_problem` (equal local configurations give equal signatures)."""
function problem_signature(p)
    r(x) = round(x; sigdigits = 10)
    order = sortperm(p.w)                                              # order of the involved faces is irrelevant
    inverse = invperm(order)
    cells = sort(
        [
            (r(p.cell_volume[c]), sort([(inverse[i], o) for (i, o) in p.incidence[c]])) for c in eachindex(p.cell_volume)
        ]
    )
    return (r.(p.w[order]), r.(p.cap_plus[order]), r.(p.cap_minus[order]), cells, r(p.Vmin))
end

"""
Exact supremum of `|Upsilon_g| / (nu V_min)` over all carrier volumes `Phi_h = dt |h| a_h v_h` allowed by the WY time-step bound
(eq:vof:cfl: incoming and outgoing volume of every cell at most `nu |c^o|`, extrusion `dt |v_f| <= nu h_donor`).  Faces without a
weight in the dual face are set to zero (optimal: they only tighten the constraints).  Enumerates the sign patterns of the involved
fluxes and solves two small linear programs for each.
"""
function solve_dual_courant(p; nu = 0.5)
    m = length(p.w)
    best = 0.0
    for pattern in 0:(2^m - 1)
        sigma = [(pattern >> (i - 1)) & 1 == 0 ? 1.0 : -1.0 for i in 1:m]
        rows = Vector{Vector{Float64}}()
        rhs = Float64[]
        for (ci, inc) in enumerate(p.incidence)
            inflow = zeros(m); outflow = zeros(m)
            for (i, o) in inc
                o * sigma[i] > 0 ? (outflow[i] = 1.0) : (inflow[i] = 1.0)
            end
            any(!iszero, outflow) && (push!(rows, outflow); push!(rhs, nu * p.cell_volume[ci]))
            any(!iszero, inflow) && (push!(rows, inflow); push!(rhs, nu * p.cell_volume[ci]))
        end
        for i in 1:m
            e = zeros(m); e[i] = 1.0
            push!(rows, e)
            push!(rhs, nu * (sigma[i] > 0 ? p.cap_plus[i] : p.cap_minus[i]))
        end
        A = reduce(vcat, (r' for r in rows))
        for s in (1.0, -1.0)
            best = max(best, lp_max([s * p.w[i] * sigma[i] for i in 1:m], A, rhs) / (nu * p.Vmin))
        end
    end
    return best
end

"""Worst-case Courant ratio of every distinct local configuration among the dual faces of a phase schedule (see above)."""
function worst_courant_by_signature!(table, mesh, sched; aper = nothing, open_volume = nothing, nu = 0.5, skip = key -> false)
    worst = 0.0
    for key in keys(sched.weights)
        skip(key) && continue
        p = dual_courant_problem(mesh, sched, key; aper, open_volume)
        sig = problem_signature(p)
        value = get!(table, sig) do
            solve_dual_courant(p; nu)
        end
        worst = max(worst, value)
    end
    return worst
end

end # module MomentumAudit
