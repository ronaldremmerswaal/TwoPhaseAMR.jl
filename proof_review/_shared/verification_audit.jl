# Shared helpers of the audit of Chapter 14 (verification) dated 2026-11-10: random walled fixtures (uniform and
# composite meshes, drops and planar interfaces including interfaces on grid lines, sloped / aligned / disc solids), flow
# states with random velocities, and independent evaluations of the diagnostics of eq:verif:energy ...
# eq:verif:divergence-norm written from the text (open strip areas, clamped fraction, polygon moments, PLIC chords).
# Not an audit. Included by the scripts of proof_review/eq-verif-diagnostics/scripts (needs TwoPhaseAMR and ForcesAudit):
#     isdefined(Main, :ForcesAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "forces_audit.jl"))
#     isdefined(Main, :VerifAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "verification_audit.jl"))

module VerifAudit

using Random, LinearAlgebra
using TwoPhaseAMR
const AMR = TwoPhaseAMR
const FA = Main.ForcesAudit

# ---------------------------------------------------------------------------------------------------------------------
# Fixtures: (name, mesh, geometry, field, solid) on the walled 8 x 8 box (cells 0.5 on the left half, 0.25 on the right half
# for the composite meshes; grid lines at multiples of 0.5 / 0.25).

disc(cx, cy, r) = (x, y) -> hypot(x - cx, y - cy) - r

function make_fixture(name, kind, levelset; solid = nothing, lipschitz = 1.0, contact = pi / 2, kwargs...)
    mesh = FA.mesh_of(kind)
    geometry = solid === nothing ? nothing : GeometryField(mesh, solid.shape; lipschitz = solid.lipschitz)
    field = geometry === nothing ? InterfaceField(mesh, levelset; lipschitz, kwargs...) :
        InterfaceField(mesh, levelset; geometry, contact_angle = contact, lipschitz, kwargs...)
    return (; name, mesh, geometry, field, solid = geometry !== nothing)
end

"""The fixtures of the diagnostics audit (deterministic given `seed`)."""
function fixtures(seed = 20261110)
    rng = MersenneTwister(seed)
    out = Any[]
    low = (; shape = FA.solid_low, lipschitz = hypot(1, FA.SOLID_LOW.slope))
    high = (; shape = FA.solid_high, lipschitz = hypot(1, 0.1))
    aligned = (; shape = (x, y) -> y - 1.0, lipschitz = 1.0)        # solid y < 1 on cell faces of both levels
    rdisc = (; shape = FA.disc_levelset, lipschitz = 1.0)
    push!(out, make_fixture("uniform drop", :uniform, disc(3.7 + rand(rng), 4.0 + rand(rng), 1.5 + rand(rng))))
    push!(out, make_fixture("composite drop", :amr, disc(3.9 + 0.4 * rand(rng), 4.1 + rand(rng), 1.8 + rand(rng))))
    push!(out, make_fixture("composite plane on grid line x=4", :amr, (x, y) -> x - 4.0))
    push!(out, make_fixture("uniform plane on grid line y=3.5", :uniform, (x, y) -> y - 3.5))
    push!(out, make_fixture("composite oblique plane", :amr, (x, y) -> 0.6x + 0.8y - 4.4))
    push!(out, make_fixture("composite drop on sloped solid", :amr, disc(4.3, 1.4, 1.5); solid = low))
    push!(out, make_fixture("composite drop across sloped solid", :amr, disc(3.2 + rand(rng), 1.1, 1.6); solid = low))
    push!(out, make_fixture("uniform drop and disc solid", :uniform, disc(5.9, 5.8, 1.45); solid = rdisc, atol = 1.0e-8, maxdepth = 24))
    push!(out, make_fixture("composite oblique plane, high solid", :amr, (x, y) -> 0.3x - y + 3.7; solid = high))
    push!(out, make_fixture("composite plane x=2 on grid line, aligned solid", :amr, (x, y) -> x - 2.0; solid = aligned))
    push!(out, make_fixture("composite plane x=5 on grid line, sloped solid", :amr, (x, y) -> x - 5.0; solid = low))
    return out
end

"""A `FlowSimulation` with random velocities (`two` selects the two-velocity model); `NaN` of absent phases is kept."""
function flow_state(fix; two, density = (1000.0, 1.0), gravity = (0.31, -1.07), sigma = 0.7, rng = MersenneTwister(5))
    physics = TwoPhase(fix.field; density, dynamic_viscosity = (0.01, 0.02), surface_tension = sigma, gravity)
    model = two ? TwoVelocityModel() : OneVelocityModel()
    velocity = two ? ((x, y, t, a) -> 1.0, (x, y, t, a) -> -1.0) : (x, y, t, a) -> 1.0
    flow = fix.solid ?
        FlowSimulation(fix.mesh, physics; geometry = fix.geometry, velocity_model = model, velocity, diagnostics = ()) :
        FlowSimulation(fix.mesh, physics; velocity_model = model, velocity, diagnostics = ())
    for v in (two ? flow.velocity : (flow.velocity,))
        for a in (v.interior, v.boundary), i in eachindex(a)
            isnan(a[i]) || (a[i] = randn(rng))
        end
    end
    return flow, simulation_state(flow)
end

# ---------------------------------------------------------------------------------------------------------------------
# Independent evaluations from the text.

"""Open staggered volume of eq:cutcells:dual: half open strips of the two cells over the tangential extent of the face."""
function open_dual_volume(field, face)
    mesh = field.mesh
    v = sum(0.5 * FA.open_strip_area(field, ref, face) * mesh.blocks[ref.block].volume for ref in (face.minus, face.plus))
    full = sum(0.5 * mesh.blocks[ref.block].volume for ref in (face.minus, face.plus))
    return v > 1.0e-12 * full ? v : 0.0           # clipping round-off of covered faces (e.g. 2e-29) is zero
end

"""Clamped staggered liquid fraction of eq:forces:density (strip weights, conditional fractions, clamp; 0 if both strips vanish)."""
function clamped_fraction(field, face)
    weighted = volume = 0.0
    for ref in (face.minus, face.plus)
        w = 0.5 * FA.open_strip_area(field, ref, face) * field.mesh.blocks[ref.block].volume
        weighted += w * AMR._conditional_fraction(field, ref)
        volume += w
    end
    full = sum(0.5 * field.mesh.blocks[ref.block].volume for ref in (face.minus, face.plus))
    return volume > 1.0e-12 * full ? clamp(weighted / volume, 0.0, 1.0) : 0.0
end

"""Kinetic energy and momentum of eq:verif:energy (one velocity) / eq:verif:energy-two (two velocities), written from the text."""
function kinetic_momentum(field, velocity, density; two)
    ρl, ρg = density
    energy = 0.0
    momentum = [0.0, 0.0]
    for (i, face) in pairs(field.mesh.faces)
        V = open_dual_volume(field, face)
        C = clamped_fraction(field, face)
        if two
            V == 0 && continue      # no open staggered volume: both phase velocities are undefined (NaN), the product has zero weight
            wl, wg = ρl * C, ρg * (1 - C)
            ul = velocity[1].interior[i]
            ug = velocity[2].interior[i]
            # "a product with a vanishing weight is zero, also where the phase velocity is undefined"
            El = wl == 0 ? 0.0 : wl * ul^2
            Eg = wg == 0 ? 0.0 : wg * ug^2
            Ml = wl == 0 ? 0.0 : wl * ul
            Mg = wg == 0 ? 0.0 : wg * ug
            energy += 0.5 * V * (El + Eg)
            momentum[face.axis] += V * (Ml + Mg)
        else
            ρ = ρl * C + ρg * (1 - C)
            u = velocity.interior[i]
            energy += 0.5 * V * ρ * u^2
            momentum[face.axis] += V * ρ * u
        end
    end
    return energy, momentum
end

const SQUARE = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]

"""Open polygon of a cell in local coordinates (the unit square clipped to the open side of its solid plane)."""
function open_polygon(field, ref)
    geometry = field.geometry
    geometry === nothing && return SQUARE
    solid = geometry.solid
    f = AMR._get(solid.fraction, ref)
    f <= 0 && return SQUARE
    f >= 1 && return NTuple{2, Float64}[]
    return FA.clip(SQUARE, (AMR._get(solid.normal_x, ref), AMR._get(solid.normal_y, ref)), AMR._get(solid.intercept, ref); keep_less = false)
end

"""Area and first moment ∫ x dA of a simple polygon (shoelace formulas), local coordinates."""
function polygon_moment(poly)
    m = length(poly)
    m < 3 && return 0.0, 0.0, 0.0
    A = Mx = My = 0.0
    for k in 1:m
        x1, y1 = poly[k]
        x2, y2 = poly[mod1(k + 1, m)]
        cross = x1 * y2 - x2 * y1
        A += cross / 2
        Mx += (x1 + x2) * cross / 6
        My += (y1 + y2) * cross / 6
    end
    s = sign(A)
    return abs(A), s * Mx, s * My
end

"""Liquid region of a cell (local polygon) of the text: the part of the open cell on the liquid side `n.x <= a` of its PLIC
line for 0 < C < 1, the open cell for C = 1, empty for C = 0."""
function liquid_polygon(field, ref)
    C = AMR._conditional_fraction(field, ref)
    op = open_polygon(field, ref)
    C <= 0 && return NTuple{2, Float64}[]
    C >= 1 && return op
    return FA.clip(op, (AMR._get(field.normal_x, ref), AMR._get(field.normal_y, ref)), AMR._get(field.intercept, ref); keep_less = true)
end

"""First moments ∫ x dx (physical coordinates) of the open region and of the reconstructed liquid region, and the maximum
violation of |liquid polygon| = C |open cell| over mixed cells (the convention check of the liquid side)."""
function first_moments(field)
    mesh = field.mesh
    Ml = [0.0, 0.0]
    Mo = [0.0, 0.0]
    area_gap = 0.0
    for ref in AMR._cell_references(mesh)
        block = mesh.blocks[ref.block]
        c = AMR.cellcenter(mesh, ref.block, ref.i, ref.j)
        h = block.spacing
        vol = block.volume
        Ao, mox, moy = polygon_moment(open_polygon(field, ref))
        lp = liquid_polygon(field, ref)
        Al, mlx, mly = polygon_moment(lp)
        C = AMR._conditional_fraction(field, ref)
        0 < C < 1 && (area_gap = max(area_gap, abs(Al - C * Ao)))
        Mo .+= vol .* (c[1] * Ao + h[1] * mox, c[2] * Ao + h[2] * moy)
        Ml .+= vol .* (c[1] * Al + h[1] * mlx, c[2] * Al + h[2] * mly)
    end
    return Ml, Mo, area_gap
end

"""Potential energy of eq:verif:energies from the moments."""
function potential_energy(field, density, gravity)
    Ml, Mo, _ = first_moments(field)
    Mg = Mo .- Ml
    return -sum(gravity[k] * (density[1] * Ml[k] + density[2] * Mg[k]) for k in 1:2)
end

"""Interface measure of eq:verif:energies: PLIC chords of the mixed cells in the open polygons plus separating faces
(`ffloor` rule, aperture weight). Returns (total, segments, faces, n_double_counted, n_skipped_cells)."""
function interface_measure(field)
    mesh = field.mesh
    seg = 0.0
    skipped = 0
    for ref in AMR._cell_references(mesh)
        C = AMR._conditional_fraction(field, ref)
        0 < C < 1 || continue
        n = (AMR._get(field.normal_x, ref), AMR._get(field.normal_y, ref))
        a = AMR._get(field.intercept, ref)
        op = open_polygon(field, ref)
        length(op) >= 3 || (skipped += 1; continue)
        p, q = try
            Main.ForcesAudit.Geo.segment_in(op, n, a)
        catch
            skipped += 1
            continue
        end
        h = mesh.blocks[ref.block].spacing
        seg += hypot(h[1] * (p[1] - q[1]), h[2] * (p[2] - q[2]))
    end
    fl = field.fraction_floor
    facepart = 0.0
    doubled = 0
    for (i, face) in pairs(mesh.faces)
        cm = AMR._conditional_fraction(field, face.minus)
        cp = AMR._conditional_fraction(field, face.plus)
        sep = (cm <= fl && cp >= 1 - fl) || (cp <= fl && cm >= 1 - fl)
        sep || continue
        a = field.geometry === nothing ? 1.0 : field.geometry.aperture[i]
        facepart += face.area * a
        # no PLIC segment may exist in either neighbour (otherwise the interface would be counted twice)
        (0 < cm < 1 || 0 < cp < 1) && (doubled += 1)
    end
    return seg + facepart, seg, facepart, doubled, skipped
end

"""Divergence norm of eq:verif:divergence-norm written from the text: cell sums Σ orient |f| (face flux) / (|c| open)."""
function divergence_norm(field, velocity; two)
    mesh = field.mesh
    nx, ny = mesh.block_size
    geometry = field.geometry
    flux = zeros(length(mesh.blocks) * nx * ny)
    if two
        la, ga = AMR._current_phase_face_apertures!(field)
        for (i, face) in pairs(mesh.faces)
            q = face.area * ((la[i] == 0 ? 0.0 : la[i] * velocity[1].interior[i]) + (ga[i] == 0 ? 0.0 : ga[i] * velocity[2].interior[i]))
            flux[AMR._cell_index(face.minus, nx, ny)] += q
            flux[AMR._cell_index(face.plus, nx, ny)] -= q
        end
        for (i, face) in pairs(mesh.boundary_faces)
            l, g = AMR._phase_boundary_face_apertures(field, i)
            q = face.side * face.area * ((l == 0 ? 0.0 : l * velocity[1].boundary[i]) + (g == 0 ? 0.0 : g * velocity[2].boundary[i]))
            flux[AMR._cell_index(face.cell, nx, ny)] += q
        end
    else
        for (i, face) in pairs(mesh.faces)
            a = geometry === nothing ? 1.0 : geometry.aperture[i]
            q = face.area * a * velocity.interior[i]
            flux[AMR._cell_index(face.minus, nx, ny)] += q
            flux[AMR._cell_index(face.plus, nx, ny)] -= q
        end
        for (i, face) in pairs(mesh.boundary_faces)
            a = geometry === nothing ? 1.0 : geometry.boundary_aperture[i]
            flux[AMR._cell_index(face.cell, nx, ny)] += face.side * face.area * a * velocity.boundary[i]
        end
    end
    total = 0.0
    for ref in AMR._cell_references(mesh)
        block = mesh.blocks[ref.block]
        Ao = polygon_moment(open_polygon(field, ref))[1]       # open fraction of the cell (local unit square)
        open = Ao * block.volume
        open > 1.0e-14 * block.volume || continue
        total += flux[AMR._cell_index(ref, nx, ny)]^2 / open
    end
    return sqrt(total)
end

end # module VerifAudit
