# Helpers shared by the audits of Chapter 5 (VOF and phase-scalar transport) of the technical reference.
#
# Not a check by itself.  Needs the TwoPhaseAMR package (repository environment) and is included with
#     isdefined(Main, :VofAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "vof_audit.jl"))
#     using .VofAudit
#
# `VofAudit.Docs` wraps docs/techref/figures/vof.jl (the independent donating regions, fluxes, replica sweeps
# and fixtures of Chapter 5) in a module of its own, with the polygon helpers of `_shared/techref_geometry.jl`
# standing in for the clipping helpers of docs/techref/figures/interface.jl.  On top of it this file adds
#   * random fixtures (uniform / graded AMR meshes, walls, interfaces, anisotropic cells) and random velocities,
#   * a replica sweep that records the pre-clamp fractions and the quantities of the boundedness argument
#     (outflow and inflow of the conservatively moved phase, rectangle overlaps) instead of asserting them,
#   * the classification of donors (uncut / cut, single-face / hanging sides).

module VofAudit

import TwoPhaseAMR
using TwoPhaseAMR
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
include(joinpath(@__DIR__, "..", "..", "docs", "techref", "figures", "vof.jl"))
end # module Docs

const D = Docs

# ---------------------------------------------------------------------------------------------------------
# Cells, faces, donors.
# ---------------------------------------------------------------------------------------------------------

"""Number of owned face segments on the (low, high) side of a cell normal to `axis`, keyed by (cell, axis)."""
function side_counts(mesh)
    counts = Dict{Tuple{CellRef, Int}, Vector{Int}}()
    for face in mesh.faces
        counts[(face.minus, face.axis)] = get(counts, (face.minus, face.axis), [0, 0]) .+ [0, 1]
        counts[(face.plus, face.axis)] = get(counts, (face.plus, face.axis), [0, 0]) .+ [1, 0]
    end
    return counts
end

"""Donor class: (:uncut | :cut | :covered, hanging) with hanging = a side normal to `axis` has >= 2 segments."""
function donor_class(field, counts, ref, axis)
    alpha = D.solid_fraction_of(field, ref)
    cls = alpha <= 0 ? :uncut : alpha >= 1 ? :covered : :cut
    c = get(counts, (ref, axis), [1, 1])
    return cls, max(c[1], c[2]) >= 2
end

rect_bounds(poly) = (
    minimum(p[1] for p in poly), maximum(p[1] for p in poly),
    minimum(p[2] for p in poly), maximum(p[2] for p in poly),
)

"""Intersection polygon of two axis-aligned rectangles given as polygons (empty if the interiors are disjoint)."""
function rect_intersection(p, q)
    a = rect_bounds(p); b = rect_bounds(q)
    x0, x1 = max(a[1], b[1]), min(a[2], b[2])
    y0, y1 = max(a[3], b[3]), min(a[4], b[4])
    (x1 > x0 && y1 > y0) || return NTuple{2, Float64}[]
    return NTuple{2, Float64}[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
end

# ---------------------------------------------------------------------------------------------------------
# Random fixtures.
# ---------------------------------------------------------------------------------------------------------

"""Random composite mesh (2:1 balance is enforced by the mesh)."""
function random_mesh(rng; periodic = (true, true), levels = 1, base = (12, 12), block = (4, 4),
        extent = (12.0, 12.0), refine_probability = 0.4)
    flags = Dict{BlockKey, Bool}()
    refine = (key, origin, ext) ->
        get!(() -> rand(rng) < refine_probability / (key.level + 1), flags, key)
    return BlockTreeMesh(; base_mesh = base, block_size = block, extent, periodic,
        max_level = levels, refine)
end

"""A random smooth interface level set (drop, wavy film or tilted ellipse) and a Lipschitz bound."""
function random_interface(rng, extent)
    kind = rand(rng, 1:3)
    lx, ly = extent
    if kind == 1
        cx, cy = lx * (0.3 + 0.4 * rand(rng)), ly * (0.3 + 0.4 * rand(rng))
        r = min(lx, ly) * (0.15 + 0.2 * rand(rng))
        return (x, y) -> hypot(x - cx, y - cy) - r, 1.0
    elseif kind == 2
        y0 = ly * (0.3 + 0.4 * rand(rng)); amp = 0.05 * ly * rand(rng); k = rand(rng, 1:3)
        return (x, y) -> y - y0 - amp * sinpi(2k * x / lx), hypot(1, 2pi * k * amp / lx)
    else
        cx, cy = lx * (0.3 + 0.4 * rand(rng)), ly * (0.3 + 0.4 * rand(rng))
        a, b = min(lx, ly) * (0.2 + 0.15 * rand(rng)), min(lx, ly) * (0.1 + 0.1 * rand(rng))
        th = pi * rand(rng)
        c, s = cos(th), sin(th)
        return (x, y) -> begin
            u = c * (x - cx) + s * (y - cy); v = -s * (x - cx) + c * (y - cy)
            (hypot(u / a, v / b) - 1) * min(a, b)
        end, 2.0
    end
end

"""Random straight or gently curved wall: the solid is `phi <= 0`; returns `(phi, lipschitz)`."""
function random_wall(rng, extent)
    lx, ly = extent
    slope = (rand(rng) - 0.5) * 1.6
    off = ly * (0.15 + 0.2 * rand(rng))
    bend = rand(rng) < 0.5 ? 0.03 * ly * (rand(rng) - 0.5) : 0.0
    k = rand(rng, 1:2)
    lip = hypot(1, abs(slope) + abs(bend) * 2pi * k / lx)
    return (x, y) -> y - off - slope * (x - lx / 2) - bend * sinpi(2k * x / lx), lip
end

"""Random fixture `(field, kind)`; `cut = true` puts a random wall in and walls on all four sides."""
function random_field(rng; cut = false, levels = 1, aniso = false, base = (12, 12), block = (4, 4),
        fraction_floor = nothing, reconstruction_method = nothing)
    extent = aniso ? (base[1] * (0.8 + 0.6 * rand(rng)), base[2] * (0.6 + 0.8 * rand(rng))) :
        (Float64(base[1]), Float64(base[2]))
    mesh = random_mesh(rng; periodic = cut ? (false, false) : (true, true), levels, base, block, extent)
    phi, lip = random_interface(rng, extent)
    kw = (; (fraction_floor === nothing ? (;) : (; fraction_floor))...,
        (reconstruction_method === nothing ? (;) : (; reconstruction_method))...)
    if cut
        wphi, wlip = random_wall(rng, extent)
        geometry = GeometryField(mesh, wphi; lipschitz = wlip)
        return InterfaceField(mesh, phi; geometry, lipschitz = lip, kw...), :cut
    end
    return InterfaceField(mesh, phi; lipschitz = lip, kw...), :plain
end

"""Random face velocity (smooth shear plus noise; zero on solid-closed faces is not imposed, apertures do that)."""
function random_velocity(rng, field; noise = 1.0, scale = 1.0)
    mesh = field.mesh
    velocity = FaceVelocity(mesh)
    a = randn(rng, 6)
    lx, ly = mesh.extent
    for (index, face) in pairs(mesh.faces)
        x, y = face.center
        smooth = face.axis == 1 ?
            a[1] + a[2] * sinpi(2y / ly + a[3]) + a[4] * cospi(2x / lx) :
            a[5] + a[6] * sinpi(2x / lx + a[3]) + a[2] * cospi(2y / ly)
        velocity.interior[index] = scale * (smooth + noise * randn(rng))
    end
    return velocity
end

"""Two independent random phase velocities; `opposite` in [0,1] mixes in the negative of the liquid one."""
function random_phase_velocity(rng, field; noise = 1.0, opposite = 0.0)
    ul = random_velocity(rng, field; noise)
    ug = random_velocity(rng, field; noise)
    if opposite > 0
        for index in eachindex(ug.interior)
            ug.interior[index] = (1 - opposite) * ug.interior[index] - opposite * ul.interior[index]
        end
    end
    return ul, ug
end

# ---------------------------------------------------------------------------------------------------------
# Replica sweeps with diagnostics (non-asserting variant of Docs.wy_replica).
# ---------------------------------------------------------------------------------------------------------

"""
Independent replica of a full step (one-velocity or frozen carrier; fluxes by polygon clipping from
`Docs.replica_fluxes`) recording, per sweep and per cell:
  `cpre`            fraction after the update, before clamp and snap;
  `V_cons`          volume of the conservatively moved phase before the sweep (liquid if chi = 0, else gas);
  `out_cons`        its outflow through the rectangles of the faces through which the cell donates;
  `in_cons`, `in_carrier`, `out_carrier`  inflow of that phase / inflow and outflow of the carrier (volumes);
  `nu_in`           cumulative incoming CFL number over the sweeps so far (carrier volumes / open volume);
  `overlap`         largest pairwise overlap area of the donor's rectangles (local units), `overlap_phase` the
                    area of the part of that overlap lying in the open part and in the conservatively moved phase;
  `depth`           largest depth (capped at 1) of the donor's rectangles in this sweep;
  `cls`, `hanging`  donor class.
"""
function sweep_records(field, carrier, dt, order)
    mesh = field.mesh
    copyfield = copy(field)
    reconstruct!(copyfield)
    cells = D.all_cells(mesh)
    incidence = D.cell_faces(mesh)
    counts = side_counts(mesh)
    chi = Dict(ref => D.cond(copyfield, ref) > 0.5 ? 1.0 : 0.0 for ref in cells)
    floor = copyfield.fraction_floor
    records = Vector{Vector{NamedTuple}}()
    nu_cum = Dict(ref => 0.0 for ref in cells)
    dilation_total = snapped = 0.0
    for axis in order
        volumes, liquids, gases, regions = D.replica_fluxes(copyfield, carrier, dt, axis)
        out_c = Dict(ref => 0.0 for ref in cells); in_c = Dict(ref => 0.0 for ref in cells)
        out_l = Dict(ref => 0.0 for ref in cells); in_l = Dict(ref => 0.0 for ref in cells)
        by_donor = Dict{CellRef, Vector{Any}}()
        for region in regions
            r = region.result
            face = mesh.faces[region.index]
            receiver = r.donor == face.minus ? face.plus : face.minus
            carrier_volume = if carrier.kind == :frozen
                abs(dt * carrier.velocity[region.phase].interior[region.index] * face.area *
                    carrier.apertures[region.phase][region.index])
            else
                abs(volumes[region.index])
            end
            liquid_volume = abs(r.liquid)
            out_c[r.donor] += carrier_volume; in_c[receiver] += carrier_volume
            out_l[r.donor] += liquid_volume; in_l[receiver] += liquid_volume
            push!(get!(by_donor, r.donor, Any[]), region)
        end
        recs = NamedTuple[]
        new = Dict{CellRef, Float64}()
        for ref in cells
            open = D.open_volume_of(copyfield, ref)
            V = L = 0.0
            for (index, o) in get(incidence, ref, Tuple{Int, Int}[])
                mesh.faces[index].axis == axis || continue
                V += o * volumes[index]; L += o * liquids[index]
            end
            c0 = D.cond(copyfield, ref)
            cpre = open > 0 ? c0 + (chi[ref] * V - L) / open : 0.0
            s = clamp(cpre, 0.0, 1.0)
            s = s <= floor ? 0.0 : s >= 1 - floor ? 1.0 : s
            snapped += open * (s - cpre)
            dilation_total += chi[ref] * V
            new[ref] = s
            nu_cum[ref] += open > 0 ? in_c[ref] / open : 0.0
            cls, hanging = donor_class(copyfield, counts, ref, axis)
            v_cons = open * (chi[ref] == 1 ? 1 - c0 : c0)
            out_cons = chi[ref] == 1 ? out_c[ref] - out_l[ref] : out_l[ref]
            in_cons = chi[ref] == 1 ? in_c[ref] - in_l[ref] : in_l[ref]
            # rectangle overlaps of this donor
            overlap = overlap_phase = 0.0
            rs = get(by_donor, ref, Any[])
            for a in 1:length(rs), b in (a + 1):length(rs)
                inter = rect_intersection(rs[a].result.rectangle, rs[b].result.rectangle)
                isempty(inter) && continue
                overlap = max(overlap, Docs.shoelace(inter))
                op = D.open_part(copyfield, ref, inter)
                if 0 < c0 < 1 && !isempty(op)
                    n, aa = D.liquid_plane(copyfield, ref)
                    ph = Docs.clip_halfplane(op, n, aa; keep_less = chi[ref] == 0)
                    overlap_phase = max(overlap_phase, Docs.shoelace(ph))
                end
            end
            depth = isempty(rs) ? 0.0 : maximum(min(r.result.depth, 1.0) for r in rs)
            push!(recs, (; ref, axis, chi = chi[ref], c0, cpre, snapped_value = s, open, V, L, cls, hanging,
                v_cons, out_cons, in_cons, in_carrier = in_c[ref], out_carrier = out_c[ref],
                nu_in = nu_cum[ref], overlap, overlap_phase, ndr = length(rs), depth))
        end
        push!(records, recs)
        for ref in cells
            AMR._set!(copyfield.fraction, D.open_fraction_of(copyfield, ref) * new[ref], ref)
        end
        reconstruct!(copyfield)
    end
    return (; records, dilation_total, snapped, chi, field = copyfield)
end


# ---------------------------------------------------------------------------------------------------------
# Carriers: the three transport paths of Chapter 5, prepared exactly as docs/techref/figures/vof.jl does.
# ---------------------------------------------------------------------------------------------------------

"""
Prepare one transport path.  `kind` is `:one` (`velocity` a FaceVelocity), `:frozen` or `:projected`
(`velocity` a `(liquid, gas)` tuple).  Returns `(; kind, work, dt, carrier, velocity)`: `work` is the package workspace
of the path, `dt` its `stable_timestep` (wy bound with cfl = 1/2), `carrier` the specification the replica sweeps use.
"""
function prepare_carrier(kind::Symbol, field, velocity; tolerance = 1.0e-12, cfl = 0.5)
    if kind == :one
        work = WYAdvectionWorkspace(field)
        dt = stable_timestep(velocity, work; cfl)
        return (; kind, work, dt, carrier = D.one_velocity_carrier(velocity), velocity)
    elseif kind == :frozen
        work = WYAdvectionWorkspace(field, velocity; interface_transport = FrozenInterfaceFluxes())
        dt = stable_timestep(velocity, work; cfl)
        apertures = (copy(work.liquid_aperture), copy(work.gas_aperture))
        return (; kind, work, dt, carrier = D.frozen_carrier(velocity, apertures), velocity)
    elseif kind == :projected
        method = ProjectedLiquidVelocity(; tolerance, max_cycles = 1000)
        work = WYAdvectionWorkspace(field, velocity; interface_transport = method)
        dt = stable_timestep(velocity, work; cfl)
        projected = copy(AMR._project_liquid_velocity!(work, velocity))
        gas = copy(AMR._extend_projected_gas_velocity!(work, velocity))
        return (; kind, work, dt, carrier = D.projected_carrier(projected, gas), velocity)
    end
    error("unknown carrier kind $kind")
end

# ---------------------------------------------------------------------------------------------------------
# Independent replica of the phase-scalar update (conservative form (5.18) and fluctuation form (5.19)).
# ---------------------------------------------------------------------------------------------------------

"""
Face state of (5.20) for `:upwind`, `:centred`, `:lw` (Lax-Wendroff).  `flux` is the phase volume of the face (its sign
selects the upwind cell), `cm, cp, nm, np` the phase fractions of the two cells before / after the sweep (before clamp)
and `vm, vp` their open volumes.
"""
function face_state(scheme::Symbol, qm, qp, cm, cp, nm, np, flux, vm, vp)
    upwind = flux >= 0 ? qm : qp
    iszero(flux) && return zero(upwind)
    min(cm, cp, nm, np) >= 1 - 64eps() || return upwind
    downwind = flux >= 0 ? qp : qm
    scheme === :upwind && return upwind
    scheme === :centred && return (qm + qp) / 2
    if scheme === :lw
        courant = abs(flux) / min(vm, vp)
        return upwind + (1 - courant) * (downwind - upwind) / 2
    end
    error("unknown scheme $scheme")
end

"""Initial intrinsic phase values of a `PhaseScalar` as Dicts over cells: `(; liquid, gas)`."""
scalar_values(q::PhaseScalar, mesh) = (;
    liquid = Dict(ref => AMR._get(q.liquid, ref) for ref in D.all_cells(mesh)),
    gas = Dict(ref => AMR._get(q.gas, ref) for ref in D.all_cells(mesh)),
)

"""
Independent replica of the phase scalars of a full step.  Per sweep and cell it evaluates the conservative form (5.18),
the fluctuation form (5.19) with the snapped fraction, and records
`dev   = C_s * a_new - A518`  (what the snap did to the amount) and the snap bookkeeping.
Returns `(; liquid, gas, events, fractions)`; `events` is a Vector of NamedTuples (one per phase, cell, sweep).
"""
function scalar_replica(field, carrier, dt, order, q0; scheme = :upwind)
    mesh = field.mesh
    f = copy(field)
    reconstruct!(f)
    cells = D.all_cells(mesh)
    incidence = D.cell_faces(mesh)
    aux = carrier.kind == :projected
    floor = f.fraction_floor
    chi = Dict(ref => D.cond(f, ref) > 0.5 ? 1.0 : 0.0 for ref in cells)
    q = (liquid = copy(q0.liquid), gas = copy(q0.gas))
    qn = (liquid = q0.liquid, gas = q0.gas)       # frozen dilation values
    events = NamedTuple[]
    sweep = 0
    for axis in order
        sweep += 1
        volumes, liquids, gases, _ = D.replica_fluxes(f, carrier, dt, axis)
        gas_flux(index) = aux ? gases[index] : volumes[index] - liquids[index]
        C = Dict(ref => D.cond(f, ref) for ref in cells)
        gstar = Dict(ref => 1 - C[ref] for ref in cells)          # reset of the auxiliary gas carrier
        open = Dict(ref => D.open_volume_of(f, ref) for ref in cells)
        li = Dict{CellRef, Float64}(); gi = Dict{CellRef, Float64}()
        for ref in cells
            il = ig = 0.0
            for (index, o) in get(incidence, ref, Tuple{Int, Int}[])
                mesh.faces[index].axis == axis || continue
                V, L, G = volumes[index], liquids[index], gas_flux(index)
                il += o * (chi[ref] * V - L)
                ig += o * (aux ? -G : (1 - chi[ref]) * V - G)
            end
            li[ref] = open[ref] > 0 ? il / open[ref] : 0.0
            gi[ref] = open[ref] > 0 ? ig / open[ref] : 0.0
        end
        phase_fraction(ref, phase) = phase == 1 ? C[ref] : (aux ? gstar[ref] : 1 - C[ref])
        increments = (Dict{CellRef, Float64}(), Dict{CellRef, Float64}())
        amounts = (Dict{CellRef, Float64}(), Dict{CellRef, Float64}())
        for ref in cells, phase in 1:2
            incr = aflux = 0.0
            openref = open[ref]
            if openref > 0
                for (index, o) in get(incidence, ref, Tuple{Int, Int}[])
                    face = mesh.faces[index]
                    face.axis == axis || continue
                    qa = phase == 1 ? q.liquid : q.gas
                    cm, cp = phase_fraction(face.minus, phase), phase_fraction(face.plus, phase)
                    nm = cm + (phase == 1 ? li[face.minus] : gi[face.minus])
                    np = cp + (phase == 1 ? li[face.plus] : gi[face.plus])
                    flux = phase == 1 ? liquids[index] : gas_flux(index)
                    state = face_state(scheme, qa[face.minus], qa[face.plus], cm, cp, nm, np, flux,
                        open[face.minus], open[face.plus])
                    intrinsic = qa[ref]
                    chip = phase == 1 ? chi[ref] : (aux ? 0.0 : 1 - chi[ref])
                    src = chip * (phase == 1 ? qn.liquid[ref] : qn.gas[ref])
                    vdil = (aux && phase == 2) ? 0.0 : volumes[index]
                    incr += o / openref * ((src - chip * intrinsic) * vdil + flux * (intrinsic - state))
                    aflux += o / openref * (flux * state - src * vdil)
                end
            end
            increments[phase][ref] = incr
            qprev = (phase == 1 ? q.liquid : q.gas)[ref]
            amounts[phase][ref] = phase_fraction(ref, phase) * qprev - aflux      # conservative form (5.18)
        end
        new_c = Dict{CellRef, Float64}()
        newq = (liquid = copy(q.liquid), gas = copy(q.gas))
        for ref in cells
            cpre = C[ref] + li[ref]
            s = clamp(cpre, 0.0, 1.0)
            s = s <= floor ? 0.0 : s >= 1 - floor ? 1.0 : s
            new_c[ref] = s
            for phase in 1:2
                qprev = (phase == 1 ? q.liquid : q.gas)[ref]
                if phase == 1
                    cphase_pre, cphase = cpre, s
                elseif aux
                    gpre = gstar[ref] + gi[ref]
                    gpre >= -128eps() || error("replica: auxiliary gas out of bounds: $gpre")
                    g = max(gpre, 0.0); g <= floor && (g = 0.0)
                    cphase_pre, cphase = gpre, g
                else
                    cphase_pre, cphase = 1 - cpre, 1 - s
                end
                after = qprev
                if cphase > floor
                    after = qprev + increments[phase][ref] / cphase
                    (phase == 1 ? newq.liquid : newq.gas)[ref] = after
                end
                push!(events, (; ref, sweep, phase, cphase_pre, cphase, qprev, after,
                    A518 = amounts[phase][ref], open = open[ref],
                    dev = cphase * after - amounts[phase][ref],
                    snap = cphase - cphase_pre,
                    snapped_to_zero = cphase == 0,
                    updated = cphase > floor))
            end
        end
        q = newq
        for ref in cells
            AMR._set!(f.fraction, D.open_fraction_of(f, ref) * new_c[ref], ref)
        end
        reconstruct!(f)
    end
    return (; liquid = q.liquid, gas = q.gas, events, field = f)
end

"""Largest |code - replica| of the intrinsic phase values over all cells."""
function scalar_difference(q::PhaseScalar, replica, mesh)
    dl = dg = 0.0
    for ref in D.all_cells(mesh)
        dl = max(dl, abs(AMR._get(q.liquid, ref) - replica.liquid[ref]))
        dg = max(dg, abs(AMR._get(q.gas, ref) - replica.gas[ref]))
    end
    return dl, dg
end

end # module VofAudit
