# Chapter 5 (VOF and phase-scalar transport): figure data and spot checks.
# Included by generate.jl after interface.jl and cutcells.jl, whose
# independent polygon helpers (clip_halfplane, shoelace, segment_in) it uses.
# Fractions, planes, fluxes, indicators and time-step bounds are read from an
# InterfaceField and a WYAdvectionWorkspace. The functions marked
# "independent" evaluate the formulas of Chapter 5 by polygon clipping and
# their own root finding, without the package's flux, area or update kernels;
# the differences to the code are the numbers quoted in the text, and the
# assertions fail if they disagree.

const VOF_BOX = 12.0

# Streamfunctions in the 12 x 12 box: periodic, or vanishing on the walls.
# Face velocities from differences of psi at the segment end points are
# exactly solenoidal on every leaf cell of a composite mesh, because the
# corner values telescope (also at hanging vertices).
vof_psi_periodic(x, y) = VOF_BOX / (2pi) * sinpi(2x / VOF_BOX) * sinpi(2y / VOF_BOX)
vof_psi_walls(x, y) = VOF_BOX / pi * sinpi(x / VOF_BOX)^2 * sinpi(y / VOF_BOX)^2

function streamfunction_velocity(mesh, psi; scale = 1.0)
    velocity = FaceVelocity(mesh)
    for (index, face) in pairs(mesh.faces)
        x, y = face.center
        h = face.area / 2
        velocity.interior[index] = scale * (face.axis == 1 ?
            (psi(x, y + h) - psi(x, y - h)) / face.area :
            -(psi(x + h, y) - psi(x - h, y)) / face.area)
    end
    return velocity
end

const VOF_DROP = (; centre = (6.3, 5.8), radius = 2.6)
vof_drop(x, y) = hypot(x - VOF_DROP.centre[1], y - VOF_DROP.centre[2]) - VOF_DROP.radius

# Uniform periodic mesh of unit cells (one block).
function vof_uniform_mesh()
    return BlockTreeMesh(;
        base_mesh = (12, 12), block_size = (12, 12), extent = (VOF_BOX, VOF_BOX),
        periodic = (true, true),
    )
end

# Composite periodic mesh: the central root of a 3 x 3 root grid (4 x 4 unit
# cells each) is refined once, so the drop crosses all four level transitions.
function vof_amr_mesh(; periodic = (true, true))
    return BlockTreeMesh(;
        base_mesh = (12, 12), block_size = (4, 4), extent = (VOF_BOX, VOF_BOX),
        periodic, max_level = 1,
        refine = (key, origin, extent) -> key == BlockKey(0, 1, 1),
    )
end

# Sloped solid wall below y = 2.2625 + 0.25 x; the drop is centred on the wall
# (contact angle 90 degrees), so cut cells, triple cells and pure cut cells
# occur. The wall crosses every horizontal grid line 0.95 cells past a vertical
# one, so no open part is smaller than 0.11 cells. Walls on all four sides
# (zero boundary velocity).
const VOF_WALL = (; offset = 2.2625, slope = 0.25)
vof_wall(x, y) = y - VOF_WALL.offset - VOF_WALL.slope * x
const VOF_SESSILE = (; centre = (6.4, VOF_WALL.offset + VOF_WALL.slope * 6.4), radius = 2.62)
vof_sessile(x, y) = hypot(x - VOF_SESSILE.centre[1], y - VOF_SESSILE.centre[2]) - VOF_SESSILE.radius
# Liquid film between the wall and y = 2.75 + 0.2 x: the interface runs inside
# the cut cells for 0 < x < 9.75, where it meets the wall.
vof_film(x, y) = y - 2.75 - 0.2x

function vof_fixture(kind::Symbol)
    if kind == :uniform
        mesh = vof_uniform_mesh()
        field = InterfaceField(mesh, vof_drop; lipschitz = 1)
        velocity = streamfunction_velocity(mesh, vof_psi_periodic)
    elseif kind == :amr
        mesh = vof_amr_mesh()
        field = InterfaceField(mesh, vof_drop; lipschitz = 1)
        velocity = streamfunction_velocity(mesh, vof_psi_periodic)
    elseif kind == :cut
        mesh = vof_amr_mesh(; periodic = (false, false))
        geometry = GeometryField(mesh, vof_wall; lipschitz = hypot(1, VOF_WALL.slope))
        field = InterfaceField(mesh, vof_sessile; geometry, lipschitz = 1)
        velocity = streamfunction_velocity(mesh, vof_psi_walls)
    elseif kind == :film
        mesh = vof_amr_mesh(; periodic = (false, false))
        geometry = GeometryField(mesh, vof_wall; lipschitz = hypot(1, VOF_WALL.slope))
        field = InterfaceField(mesh, vof_film; geometry, lipschitz = hypot(1, 0.2))
        velocity = streamfunction_velocity(mesh, vof_psi_walls)
    else
        error("unknown fixture $kind")
    end
    return field, velocity
end

# --- Independent donating regions --------------------------------------------

cell_level(mesh, ref) = mesh.blocks[ref.block].key.level
cell_spacing(mesh, ref) = mesh.blocks[ref.block].spacing
cell_volume(mesh, ref) = mesh.blocks[ref.block].volume
cond(field, ref) = AMR._get(field.conditional_fraction, ref)
liquid_plane(field, ref) = (
    (AMR._get(field.normal_x, ref), AMR._get(field.normal_y, ref)), AMR._get(field.intercept, ref),
)
solid_fraction_of(field, ref) = isnothing(field.geometry) ? 0.0 :
    AMR._get(field.geometry.solid.fraction, ref)
solid_plane_of(field, ref) = (
    (AMR._get(field.geometry.solid.normal_x, ref), AMR._get(field.geometry.solid.normal_y, ref)),
    AMR._get(field.geometry.solid.intercept, ref),
)
open_fraction_of(field, ref) = isnothing(field.geometry) ? 1.0 :
    AMR._get(field.geometry.open_fraction, ref)
aperture_of(field, index) = isnothing(field.geometry) ? 1.0 : field.geometry.aperture[index]

wrap(delta, extent, periodic) = periodic ? delta - round(delta / extent) * extent : delta

# Tangential extent [lo, hi] of face f in the local coordinates of `ref`.
function tangent_interval(mesh, ref, face)
    t = 3 - face.axis
    centre = cellcenter(mesh, ref.block, ref.i, ref.j)
    h = cell_spacing(mesh, ref)[t]
    lo = wrap(face.center[t] - face.area / 2 - centre[t], mesh.extent[t], mesh.periodic[t]) / h
    return lo, lo + face.area / h
end

# Local donating rectangle of depth d adjacent to the face side of the donor
# (orientation o = +1: high side, -1: low side), over [lo, hi] tangentially.
function donating_rectangle(axis, o, d, lo, hi)
    klo, khi = o > 0 ? (0.5 - d, 0.5) : (-0.5, -0.5 + d)
    xlo, xhi, ylo, yhi = axis == 1 ? (klo, khi, lo, hi) : (lo, hi, klo, khi)
    return NTuple{2, Float64}[(xlo, ylo), (xhi, ylo), (xhi, yhi), (xlo, yhi)]
end

function open_part(field, ref, polygon)
    alpha = solid_fraction_of(field, ref)
    alpha <= 0 && return polygon
    alpha >= 1 && return NTuple{2, Float64}[]
    n, a = solid_plane_of(field, ref)
    return clip_halfplane(polygon, n, a; keep_less = false)
end

# Depth with |R(d) cap open| = target (local units), by independent bisection;
# Inf if the open part of the whole strip is smaller than the target.
function independent_depth(field, ref, axis, o, lo, hi, target)
    alpha = solid_fraction_of(field, ref)
    alpha <= 0 && return target / (hi - lo)
    area(d) = shoelace(open_part(field, ref, donating_rectangle(axis, o, d, lo, hi)))
    target <= area(1.0) * (1 + 1.0e-12) || return Inf
    a, b = 0.0, 1.0
    for _ in 1:200
        m = (a + b) / 2
        area(m) < target ? (a = m) : (b = m)
        b - a <= 1.0e-17 && break
    end
    return (a + b) / 2
end

# Donating region of an owned face for carrier velocity u over [lo, hi]
# (default: the whole segment) with prescribed swept volume |volume|.
# Returns (liquid volume, local DR rectangle, its open part, its liquid part,
# donor, depth). Pure donors give C_d |volume| without geometry.
function independent_donating_region(field, face, u, dt, volume; interval = nothing)
    mesh = field.mesh
    donor = u >= 0 ? face.minus : face.plus
    o = u >= 0 ? 1 : -1
    lo, hi = isnothing(interval) ? tangent_interval(mesh, donor, face) : interval
    target = abs(volume) / cell_volume(mesh, donor)
    h = cell_spacing(mesh, donor)[face.axis]
    depth = isnothing(field.geometry) ? abs(u) * dt / h :
        independent_depth(field, donor, face.axis, o, lo, hi, target)
    rectangle = donating_rectangle(face.axis, o, min(depth, 1.0), lo, hi)
    open = open_part(field, donor, rectangle)
    c = cond(field, donor)
    if c == 0 || c == 1
        liquid = c == 1 ? open : NTuple{2, Float64}[]
        return (; liquid = sign(u) * c * abs(volume), rectangle, open, liquid_polygon = liquid,
            donor, depth)
    end
    isfinite(depth) || error("swept volume exceeds the open strip of a mixed donor")
    n, a = liquid_plane(field, donor)
    liquid_polygon = clip_halfplane(open, n, a; keep_less = true)
    return (; liquid = sign(u) * shoelace(liquid_polygon) * cell_volume(mesh, donor),
        rectangle, open, liquid_polygon, donor, depth)
end

function face_class(field, index, face, donor)
    involved = aperture_of(field, index) < 1 || 0 < solid_fraction_of(field, donor) < 1
    involved && return :cut
    return cell_level(field.mesh, face.minus) == cell_level(field.mesh, face.plus) ? :uniform : :coarse_fine
end

# One sweep's one-velocity fluxes (both axes, from the same reconstruction)
# against the independent donating regions.
function one_velocity_flux_check(field, velocity, dt)
    work = WYAdvectionWorkspace(field)
    reconstruct!(field)
    worst = Dict(:uniform => 0.0, :coarse_fine => 0.0, :cut => 0.0)
    count = Dict(:uniform => 0, :coarse_fine => 0, :cut => 0)
    mixed = Dict(:uniform => 0, :coarse_fine => 0, :cut => 0)
    partition = 0.0
    for axis in 1:2
        AMR._compute_directional_wy_fluxes!(field, velocity, dt, work, axis)
        for (index, face) in pairs(field.mesh.faces)
            face.axis == axis || continue
            u = velocity.interior[index]
            iszero(u) && continue
            volume = dt * u * face.area * aperture_of(field, index)
            @assert work.open_flux[index] == volume
            result = independent_donating_region(field, face, u, dt, volume)
            class = face_class(field, index, face, result.donor)
            scale = cell_volume(field.mesh, result.donor)
            worst[class] = max(worst[class], abs(work.liquid_flux[index] - result.liquid) / scale)
            count[class] += 1
            0 < cond(field, result.donor) < 1 && (mixed[class] += 1)
            partition = max(partition, abs(work.liquid_flux[index] + work.gas_flux[index] - volume))
        end
    end
    return (; worst, count, mixed, partition)
end

# Frozen phase subfaces (FrozenInterfaceFluxes): independent interval
# placement and donating regions.
function independent_subface(field, liquid_aperture, gas_aperture, index, face, donor, liquid_carrier)
    mesh = field.mesh
    lo, hi = tangent_interval(mesh, donor, face)
    width = hi - lo
    open_aperture = aperture_of(field, index)
    open_lo, open_hi = lo, hi
    if open_aperture < 1
        t = 3 - face.axis
        sn(ref) = solid_plane_of(field, ref)[1][t]
        solid_tangent = (sn(face.minus) + sn(face.plus)) / 2
        open_lo, open_hi = solid_tangent > 0 ? (hi - open_aperture * width, hi) :
            (lo, lo + open_aperture * width)
    end
    phase_aperture = liquid_carrier ? liquid_aperture : gas_aperture
    phase_width = clamp(phase_aperture, 0.0, open_aperture) * width
    phase_width >= open_aperture * width && return (open_lo, open_hi)
    t = 3 - face.axis
    ln(ref) = liquid_plane(field, ref)[1][t]
    liquid_lower = (ln(face.minus) + ln(face.plus)) / 2 >= 0
    lower = liquid_carrier == liquid_lower
    return lower ? (open_lo, open_lo + phase_width) : (open_hi - phase_width, open_hi)
end

function frozen_flux_check(field, velocity::Tuple, dt)
    work = WYAdvectionWorkspace(field, velocity; interface_transport = FrozenInterfaceFluxes())
    AMR._freeze_phase_apertures!(field, work)
    worst = 0.0; faces = 0; split = 0
    for axis in 1:2
        AMR._compute_directional_wy_fluxes!(field, velocity, dt, work, axis)
        for (index, face) in pairs(field.mesh.faces)
            face.axis == axis || continue
            open = liquid = 0.0
            scale = Inf
            parts = 0
            for (phase, aperture) in ((1, work.liquid_aperture[index]), (2, work.gas_aperture[index]))
                iszero(aperture) && continue
                u = velocity[phase].interior[index]
                iszero(u) && continue
                volume = dt * u * face.area * aperture
                donor = u >= 0 ? face.minus : face.plus
                interval = independent_subface(field, work.liquid_aperture[index],
                    work.gas_aperture[index], index, face, donor, phase == 1)
                result = independent_donating_region(field, face, u, dt, volume; interval)
                open += volume; liquid += result.liquid
                scale = min(scale, cell_volume(field.mesh, donor))
                parts += 1
            end
            parts == 0 && continue
            faces += 1
            parts == 2 && (split += 1)
            @assert abs(work.open_flux[index] - open) <= 4eps() * abs(open)
            worst = max(worst, abs(work.liquid_flux[index] - liquid) / scale)
        end
    end
    return (; worst, faces, split)
end

# Equal phase velocities without a solid: the two frozen subface regions tile
# the one-velocity donating region, so the fluxes coincide.
function equal_velocity_reduction_check(field, velocity, dt)
    one = WYAdvectionWorkspace(field)
    two_velocity = (velocity, copy(velocity))
    two = WYAdvectionWorkspace(field, two_velocity; interface_transport = FrozenInterfaceFluxes())
    AMR._freeze_phase_apertures!(field, two)
    worst = 0.0
    for axis in 1:2
        AMR._compute_directional_wy_fluxes!(field, velocity, dt, one, axis)
        AMR._compute_directional_wy_fluxes!(field, two_velocity, dt, two, axis)
        for (index, face) in pairs(field.mesh.faces)
            face.axis == axis || continue
            scale = min(cell_volume(field.mesh, face.minus), cell_volume(field.mesh, face.plus))
            worst = max(worst, abs(one.liquid_flux[index] - two.liquid_flux[index]) / scale,
                abs(one.open_flux[index] - two.open_flux[index]) / scale)
        end
    end
    return worst
end

# --- Velocities for the balance checks -----------------------------------------

# Second streamfunctions, used as the gas velocity of the two-velocity checks.
vof_psi2_periodic(x, y) = -0.6 * VOF_BOX / (2pi) * sinpi(2x / VOF_BOX + 0.35) * sinpi(4y / VOF_BOX)
vof_psi2_walls(x, y) = -0.5 * VOF_BOX / pi * sinpi(x / VOF_BOX)^2 * sinpi(y / VOF_BOX) * sinpi(2y / VOF_BOX)

# Open-solenoidal velocity on a cut mesh: the open flux |f| a_f u_f is the
# difference of a vertex function that vanishes on solid vertices, so it
# telescopes around every open cell and closed faces carry none.
function open_streamfunction_velocity(field, psi)
    mesh = field.mesh
    value(x, y) = vof_wall(x, y) <= 0 ? 0.0 : psi(x, y)
    velocity = FaceVelocity(mesh)
    for (index, face) in pairs(mesh.faces)
        x, y = face.center
        h = face.area / 2
        flux = face.axis == 1 ? value(x, y + h) - value(x, y - h) :
            -(value(x + h, y) - value(x - h, y))
        a = aperture_of(field, index)
        @assert a > 0 || flux == 0
        velocity.interior[index] = a > 0 ? flux / (face.area * a) : 0.0
    end
    return velocity
end

function maximum_open_divergence(field, velocity)
    divergence = CellField(field.mesh)
    isnothing(field.geometry) ? divergence!(divergence, velocity) :
        divergence!(divergence, velocity, field.geometry)
    return maximum(abs(AMR._get(divergence, ref)) for ref in all_cells(field.mesh))
end

# --- Independent WY step (replica) -------------------------------------------

all_cells(mesh) = [CellRef(b, i, j) for b in eachindex(mesh.blocks)
    for j in 1:mesh.block_size[2] for i in 1:mesh.block_size[1]]

function cell_faces(mesh)
    faces = Dict{CellRef, Vector{Tuple{Int, Int}}}()
    for (index, face) in pairs(mesh.faces)
        push!(get!(faces, face.minus, Tuple{Int, Int}[]), (index, 1))
        push!(get!(faces, face.plus, Tuple{Int, Int}[]), (index, -1))
    end
    return faces
end

open_volume_of(field, ref) = open_fraction_of(field, ref) * cell_volume(field.mesh, ref)
liquid_volume(field) = sum(
    cell_volume(field.mesh, ref) * AMR._get(field.fraction, ref) for ref in all_cells(field.mesh)
)

# Carrier specifications of the three transport paths.
one_velocity_carrier(velocity) = (; kind = :one, velocity)
frozen_carrier(velocity, apertures) = (; kind = :frozen, velocity, apertures)
projected_carrier(liquid, gas) = (; kind = :projected, velocity = liquid, gas)

# Fluxes of one sweep, evaluated independently on the current reconstruction:
# carrier volume, liquid volume and (projected method) auxiliary gas volume.
function replica_fluxes(field, carrier, dt, axis)
    mesh = field.mesh
    volumes = Dict{Int, Float64}(); liquids = Dict{Int, Float64}()
    gases = Dict{Int, Float64}(); regions = Any[]
    for (index, face) in pairs(mesh.faces)
        face.axis == axis || continue
        if carrier.kind == :frozen
            V = L = 0.0
            al, ag = carrier.apertures[1][index], carrier.apertures[2][index]
            for (phase, aperture) in ((1, al), (2, ag))
                iszero(aperture) && continue
                u = carrier.velocity[phase].interior[index]
                iszero(u) && continue
                volume = dt * u * face.area * aperture
                donor = u >= 0 ? face.minus : face.plus
                interval = independent_subface(field, al, ag, index, face, donor, phase == 1)
                result = independent_donating_region(field, face, u, dt, volume; interval)
                V += volume; L += result.liquid
                push!(regions, (; index, phase, result))
            end
            volumes[index] = V; liquids[index] = L
            continue
        end
        u = carrier.velocity.interior[index]
        V = dt * u * face.area * aperture_of(field, index)
        volumes[index] = V
        if iszero(u)
            liquids[index] = 0.0
        else
            result = independent_donating_region(field, face, u, dt, V)
            liquids[index] = result.liquid
            push!(regions, (; index, phase = 1, result))
        end
        if carrier.kind == :projected
            ug = carrier.gas.interior[index]
            Vg = dt * ug * face.area * aperture_of(field, index)
            gases[index] = iszero(ug) ? 0.0 :
                Vg - independent_donating_region(field, face, ug, dt, Vg).liquid
        end
    end
    return volumes, liquids, gases, regions
end

snapshot(field, cells, axis, regions, dilation) = (;
    C = Dict(ref => cond(field, ref) for ref in cells),
    plane = Dict(ref => liquid_plane(field, ref) for ref in cells),
    axis, regions, dilation,
)

# Independent sweep sequence: fluxes by clipping, the update of Chapter 5,
# clamp and snap, then the package's reconstruction (Chapter 4). Records the
# intermediate states, the dilation volume, the clamped/snapped volume and,
# for the projected method, the auxiliary gas carrier and its reset.
function wy_replica(field, carrier, dt, order)
    mesh = field.mesh
    copyfield = copy(field)
    reconstruct!(copyfield)
    cells = all_cells(mesh)
    incidence = cell_faces(mesh)
    chi = Dict(ref => cond(copyfield, ref) > 0.5 ? 1.0 : 0.0 for ref in cells)
    floor = copyfield.fraction_floor
    states = Any[snapshot(copyfield, cells, 0, Any[], Dict{CellRef, Float64}())]
    dilation_total = snapped = 0.0
    snaps = 0
    reset_signed = reset_absolute = 0.0
    for axis in order
        volumes, liquids, gases, regions = replica_fluxes(copyfield, carrier, dt, axis)
        dilation = Dict{CellRef, Float64}()
        new = Dict{CellRef, Float64}()
        gas_carrier = Dict{CellRef, Float64}()
        for ref in cells
            open = open_volume_of(copyfield, ref)
            V = L = G = 0.0
            for (index, o) in get(incidence, ref, Tuple{Int, Int}[])
                mesh.faces[index].axis == axis || continue
                V += o * volumes[index]; L += o * liquids[index]
                carrier.kind == :projected && (G += o * gases[index])
            end
            dilation[ref] = chi[ref] * V
            dilation_total += chi[ref] * V
            c = open > 0 ? cond(copyfield, ref) + (chi[ref] * V - L) / open : 0.0
            @assert -128eps() <= c <= 1 + 128eps() "replica out of bounds: $c"
            s = clamp(c, 0.0, 1.0)
            s = s <= floor ? 0.0 : s >= 1 - floor ? 1.0 : s
            if s != c
                snapped += open * (s - c)
                abs(s - c) > 256eps() && (snaps += 1)
            end
            new[ref] = s
            if carrier.kind == :projected
                g = open > 0 ? (1 - cond(copyfield, ref)) - G / open : 0.0
                @assert g >= -128eps() "auxiliary gas out of bounds: $g"
                g = max(g, 0.0)
                g <= floor && (g = 0.0)
                gas_carrier[ref] = g
            end
        end
        for ref in cells
            AMR._set!(copyfield.fraction, open_fraction_of(copyfield, ref) * new[ref], ref)
        end
        reconstruct!(copyfield)
        if carrier.kind == :projected
            for ref in cells
                d = open_volume_of(copyfield, ref) * (gas_carrier[ref] - (1 - cond(copyfield, ref)))
                reset_signed += d; reset_absolute += abs(d)
            end
        end
        push!(states, snapshot(copyfield, cells, axis, regions, dilation))
    end
    return (; field = copyfield, states, dilation_total, snapped, snaps, chi,
        reset_signed, reset_absolute)
end

fraction_difference(a, b) = maximum(
    abs(AMR._get(a.fraction, ref) - AMR._get(b.fraction, ref)) for ref in all_cells(a.mesh)
)

# Full one-velocity step: package advect! against the replica; the volume
# balance V(n+1) - V(n) = dilation + clamped/snapped volume; constants.
function step_balance_check(field, velocity; order = (1, 2))
    work = WYAdvectionWorkspace(field)
    dt = stable_timestep(velocity, work)
    before = liquid_volume(field)
    replica = wy_replica(field, one_velocity_carrier(velocity), dt, order)
    scalar = PhaseScalar(field; liquid = 3.0, gas = 7.0)
    advect!(field, velocity, dt, work; scalars = (scalar,), sweep_order = order)
    after = liquid_volume(field)
    constants = all(AMR._get(scalar.liquid, ref) == 3.0 && AMR._get(scalar.gas, ref) == 7.0
        for ref in all_cells(field.mesh))
    return (; dt, before, change = after - before, dilation = replica.dilation_total,
        snapped = replica.snapped, snaps = replica.snaps,
        residual = abs(after - before - replica.dilation_total - replica.snapped),
        difference = fraction_difference(field, replica.field), constants)
end

function frozen_step_check(field, velocity::Tuple; order = (1, 2))
    work = WYAdvectionWorkspace(field, velocity; interface_transport = FrozenInterfaceFluxes())
    dt = stable_timestep(velocity, work)
    apertures = (copy(work.liquid_aperture), copy(work.gas_aperture))
    before = liquid_volume(field)
    replica = wy_replica(field, frozen_carrier(velocity, apertures), dt, order)
    scalar = PhaseScalar(field; liquid = 3.0, gas = 7.0)
    advect!(field, velocity, dt, work; scalars = (scalar,), sweep_order = order)
    after = liquid_volume(field)
    constants = all(AMR._get(scalar.liquid, ref) == 3.0 && AMR._get(scalar.gas, ref) == 7.0
        for ref in all_cells(field.mesh))
    return (; dt, change = after - before, dilation = replica.dilation_total,
        snapped = replica.snapped, snaps = replica.snaps,
        residual = abs(after - before - replica.dilation_total - replica.snapped),
        difference = fraction_difference(field, replica.field), constants,
        partition = phase_flux_diagnostics(work).maximum_partition_defect)
end

function projected_step_check(field, velocity::Tuple; order = (1, 2), tolerance = 1.0e-12)
    method = ProjectedLiquidVelocity(; tolerance, max_cycles = 1000)
    work = WYAdvectionWorkspace(field, velocity; interface_transport = method)
    dt = stable_timestep(velocity, work)
    projected = copy(AMR._project_liquid_velocity!(work, velocity))
    gas = copy(AMR._extend_projected_gas_velocity!(work, velocity))
    residual_divergence = phase_flux_diagnostics(work).projection_final_divergence
    before = liquid_volume(field)
    replica = wy_replica(field, projected_carrier(projected, gas), dt, order)
    # One-velocity WY with the projected carrier (reduction of the liquid update).
    reduced = copy(field)
    advect!(reduced, projected, dt, WYAdvectionWorkspace(reduced); sweep_order = order)
    scalar = PhaseScalar(field; liquid = 3.0, gas = 7.0)
    advect!(field, velocity, dt, work; scalars = (scalar,), sweep_order = order)
    after = liquid_volume(field)
    diagnostics = phase_flux_diagnostics(work)
    constants = all(AMR._get(scalar.liquid, ref) == 3.0 && AMR._get(scalar.gas, ref) == 7.0
        for ref in all_cells(field.mesh))
    chi_volume = sum(open_volume_of(field, ref) for ref in all_cells(field.mesh) if replica.chi[ref] == 1)
    return (; dt, change = after - before, dilation = replica.dilation_total,
        snapped = replica.snapped, snaps = replica.snaps,
        residual = abs(after - before - replica.dilation_total - replica.snapped),
        difference = fraction_difference(field, replica.field),
        reduction = fraction_difference(field, reduced), constants,
        divergence = residual_divergence,
        dilation_bound = dt * residual_divergence * chi_volume,
        reset = diagnostics.gas_reset_discrepancy, reset_replica = replica.reset_signed,
        reset_absolute = diagnostics.absolute_gas_reset_discrepancy,
        reset_absolute_replica = replica.reset_absolute,
        iterations = diagnostics.projection_iterations)
end

# --- Time-step bound -----------------------------------------------------------

# Independent bound: nu_in, nu_out per open cell, extrusion per face (or
# phase subface), each <= cfl.
function independent_wy_bound(field, rates; cfl = 0.5)
    mesh = field.mesh
    incidence = cell_faces(mesh)
    incoming = outgoing = extrusion = 0.0
    for ref in all_cells(mesh)
        open = open_volume_of(field, ref)
        open > 0 || continue
        rin = rout = 0.0
        for (index, o) in get(incidence, ref, Tuple{Int, Int}[])
            for (u, a) in rates(index)
                rate = o * u * mesh.faces[index].area * a
                rout += max(rate, 0.0); rin += max(-rate, 0.0)
            end
        end
        incoming = max(incoming, rin / open); outgoing = max(outgoing, rout / open)
    end
    for (index, face) in pairs(mesh.faces), (u, a) in rates(index)
        a > 0 || continue
        donor = u >= 0 ? face.minus : face.plus
        extrusion = max(extrusion, abs(u) / cell_spacing(mesh, donor)[face.axis])
    end
    rate = max(incoming, outgoing, extrusion)
    return (; bound = cfl / rate, incoming = cfl / incoming, outgoing = cfl / outgoing,
        extrusion = cfl / extrusion)
end

one_velocity_rates(field, velocity) = index -> ((velocity.interior[index], aperture_of(field, index)),)
frozen_rates(velocity, work) = index -> (
    (velocity[1].interior[index], work.liquid_aperture[index]),
    (velocity[2].interior[index], work.gas_aperture[index]),
)

function accepts(field, velocity, dt)
    trial = copy(field)
    work = velocity isa Tuple ?
        WYAdvectionWorkspace(trial, velocity; interface_transport = FrozenInterfaceFluxes()) :
        WYAdvectionWorkspace(trial)
    try
        advect!(trial, velocity, dt, work)
        return true
    catch error
        error isa ArgumentError || rethrow()
        return false
    end
end

function cfl_check(field, velocity)
    work = WYAdvectionWorkspace(field)
    code = stable_timestep(velocity, work)
    indep = independent_wy_bound(field, one_velocity_rates(field, velocity))
    # Depth of every donating region at the bound (uncut donors: at most 1/2).
    reconstruct!(field)
    deepest_uncut = deepest_cut = 0.0
    for (index, face) in pairs(field.mesh.faces)
        u = velocity.interior[index]
        iszero(u) && continue
        volume = code * u * face.area * aperture_of(field, index)
        region = independent_donating_region(field, face, u, code, volume)
        if 0 < solid_fraction_of(field, region.donor) < 1
            deepest_cut = max(deepest_cut, region.depth)
        else
            deepest_uncut = max(deepest_uncut, region.depth)
        end
    end
    return (; code, indep, relative = abs(code - indep.bound) / indep.bound,
        deepest_uncut, deepest_cut,
        accepted = accepts(field, velocity, code),
        rejected = !accepts(field, velocity, code * (1 + 1.0e-6)))
end

function frozen_cfl_check(field, velocity::Tuple)
    work = WYAdvectionWorkspace(field, velocity; interface_transport = FrozenInterfaceFluxes())
    code = stable_timestep(velocity, work)
    indep = independent_wy_bound(field, frozen_rates(velocity, work))
    return (; code, indep, relative = abs(code - indep.bound) / indep.bound,
        accepted = accepts(field, velocity, code),
        rejected = !accepts(field, velocity, code * (1 + 1.0e-6)))
end

# A cut donor whose open part is the triangle below its anti-diagonal, next to
# an uncut cell: the trace of the shared face is 0 from the donor and 1 from
# the neighbour, so a_f = 1/2. A region of depth d over the whole face has open
# area d^2/2 (local units), the swept volume is |f| a_f |u_f| dt = (1/2) nu |c|
# with nu = |u_f| dt / h, and the outgoing bound nu_out = nu <= 1/2 allows
# d = sqrt(nu) = 1/sqrt(2) > 1/2 (independent closed form).
triangle_depth_example() = sqrt(0.5)

# A coarse cut donor at a level transition whose half-strip behind one fine
# segment is nearly closed by the wall: at the stable time step the swept
# volume of that segment exceeds the open area of its strip, and advect!
# raises an error (limitation stated in Section 5.4; independent of the
# velocity's divergence).
function strip_limitation_check()
    mesh = vof_amr_mesh(; periodic = (false, false))
    geometry = GeometryField(mesh, vof_wall; lipschitz = hypot(1, VOF_WALL.slope))
    field = InterfaceField(mesh, (x, y) -> y - 2.65 - 0.2x; geometry, lipschitz = hypot(1, 0.2))
    velocity = FaceVelocity(mesh, (x, y, t, axis) -> axis == 2 ? 1.0 : 0.0)
    enforce_no_permeation!(velocity)
    work = WYAdvectionWorkspace(field)
    dt = stable_timestep(velocity, work)
    # The segment of the coarse donor [6,7]x[3,4] at x in [6.5, 7], y = 4.
    index = findfirst(f -> f.axis == 2 && f.center == (6.75, 4.0), mesh.faces)
    face = mesh.faces[index]
    donor = face.minus
    lo, hi = tangent_interval(mesh, donor, face)
    strip = shoelace(open_part(field, donor, donating_rectangle(2, 1, 1.0, lo, hi)))
    target = dt * velocity.interior[index] * face.area * aperture_of(field, index) /
        cell_volume(mesh, donor)
    message = try
        advect!(field, velocity, dt, work)
        ""
    catch error
        sprint(showerror, error)
    end
    @assert occursin("exceeds the donor's open subcell volume", message)
    return (; dt, strip, target, open = open_fraction_of(field, donor),
        aperture = aperture_of(field, index), fraction = cond(field, donor))
end

# --- Figure data -----------------------------------------------------------------

polygon_text(points) = join(("($(fmt(p[1])),$(fmt(p[2])))" for p in points), " -- ") * " -- cycle"

function to_physical(mesh, ref, polygon)
    centre = cellcenter(mesh, ref.block, ref.i, ref.j)
    h = cell_spacing(mesh, ref)
    return [(centre[1] + h[1] * p[1], centre[2] + h[2] * p[2]) for p in polygon]
end

function cell_box_of(mesh, ref)
    centre = cellcenter(mesh, ref.block, ref.i, ref.j)
    h = cell_spacing(mesh, ref)
    return (centre[1] - h[1] / 2, centre[2] - h[2] / 2, centre[1] + h[1] / 2, centre[2] + h[2] / 2)
end

in_window(box, window) = box[1] < window[2] && box[3] > window[1] && box[2] < window[4] && box[4] > window[3]

# Cells, liquid polygons, PLIC and solid segments of a state in a window:
#   \fxvcell{x0}{y0}{x1}{y1}{C}{chi}{kind}  kind 0 uncut, 1 cut, 2 covered
#   \fxvliquid{path}  \fxvplic{x1}{y1}{x2}{y2}  \fxvsolid{path}  \fxvwall{x1}{y1}{x2}{y2}
function state_text(io, field, window; C = nothing, plane = nothing, chi = nothing)
    mesh = field.mesh
    for ref in all_cells(mesh)
        box = cell_box_of(mesh, ref)
        in_window(box, window) || continue
        alpha = solid_fraction_of(field, ref)
        kind = alpha <= 0 ? 0 : alpha >= 1 ? 2 : 1
        c = isnothing(C) ? cond(field, ref) : C[ref]
        x = isnothing(chi) ? (c > 0.5 ? 1 : 0) : Int(chi[ref])
        println(io, "\\fxvcell{$(fmt(box[1]))}{$(fmt(box[2]))}{$(fmt(box[3]))}{$(fmt(box[4]))}",
            "{$(@sprintf("%.2f", c))}{$x}{$kind}")
        kind == 2 && continue
        open = open_part(field, ref, SQUARE)
        if kind == 1
            n, a = solid_plane_of(field, ref)
            solid = clip_halfplane(SQUARE, n, a; keep_less = true)
            println(io, "\\fxvsolid{$(polygon_text(to_physical(mesh, ref, solid)))}")
            p, q = to_physical(mesh, ref, segment_in(SQUARE, n, a))
            println(io, "\\fxvwall{$(fmt(p[1]))}{$(fmt(p[2]))}{$(fmt(q[1]))}{$(fmt(q[2]))}")
        end
        c <= 0 && continue
        if c >= 1
            println(io, "\\fxvliquid{$(polygon_text(to_physical(mesh, ref, open)))}")
            continue
        end
        n, a = isnothing(plane) ? liquid_plane(field, ref) : plane[ref]
        liquid = clip_halfplane(open, n, a; keep_less = true)
        println(io, "\\fxvliquid{$(polygon_text(to_physical(mesh, ref, liquid)))}")
        p, q = to_physical(mesh, ref, segment_in(open, n, a))
        println(io, "\\fxvplic{$(fmt(p[1]))}{$(fmt(p[2]))}{$(fmt(q[1]))}{$(fmt(q[2]))}")
    end
end

# Donating regions of faces with an end point in the window:
#   \fxvdr{rectangle}{open part}{liquid part}{depth}
function regions_text(io, field, regions, window)
    mesh = field.mesh
    for region in regions
        face = mesh.faces[region.index]
        r = region.result
        box = cell_box_of(mesh, r.donor)
        in_window(box, window) || continue
        rectangle = to_physical(mesh, r.donor, r.rectangle)
        open = isempty(r.open) ? "" : polygon_text(to_physical(mesh, r.donor, r.open))
        liquid = isempty(r.liquid_polygon) ? "" : polygon_text(to_physical(mesh, r.donor, r.liquid_polygon))
        println(io, "\\fxvdr{$(polygon_text(rectangle))}{$open}{$liquid}{$(@sprintf("%.2f", r.depth))}")
    end
end

# Face velocities (arrows) in the window: \fxvvel{axis}{x}{y}{u}
function velocity_text(io, mesh, velocity, window; axis = 0)
    for (index, face) in pairs(mesh.faces)
        axis == 0 || face.axis == axis || continue
        x, y = face.center
        window[1] <= x <= window[2] && window[3] <= y <= window[4] || continue
        u = velocity.interior[index]
        iszero(u) && continue
        println(io, "\\fxvvel{$(face.axis)}{$(fmt(x))}{$(fmt(y))}{$(@sprintf("%.3f", u))}")
    end
end

# Figure "donating": (a) uniform and (b) coarse-fine faces of an x sweep in
# a sheared flow u = 1 + 0.3 sin(pi y / 4) on a composite mesh, (c) a cut
# donor in a uniform flow over a sloped wall.
vof_psi_shear(x, y) = y - 0.3 * 8 / (2pi) * cospi(2y / 8)

function vof_shear_fixture()
    mesh = BlockTreeMesh(;
        base_mesh = (12, 8), block_size = (4, 4), extent = (12.0, 8.0),
        periodic = (true, true), max_level = 1,
        refine = (key, origin, extent) -> key.level == 0 && key.i == 1,
    )
    field = InterfaceField(mesh, (x, y) -> hypot(x - 4.3, y - 4.2) - 2.35; lipschitz = 1)
    velocity = streamfunction_velocity(mesh, vof_psi_shear)
    return field, velocity
end

function vof_cut_figure_fixture()
    mesh = BlockTreeMesh(;
        base_mesh = (6, 4), block_size = (6, 4), extent = (6.0, 4.0), periodic = (false, false),
    )
    geometry = GeometryField(mesh, (x, y) -> y - 0.2 - 0.1x; lipschitz = hypot(1, 0.1))
    # :lhf_ghf adopts the GHF line in the triple cells (Chapter 4), as the
    # embedded-wall tests of the package do.
    field = InterfaceField(mesh, (x, y) -> x - 2.4 - 0.3y; geometry, lipschitz = hypot(1, 0.3),
        reconstruction_method = :lhf_ghf)
    velocity = FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? 1.0 : 0.0)
    enforce_no_permeation!(velocity)
    return field, velocity
end

const DONATING_WINDOWS = (; uniform = (0.0, 3.0, 3.0, 6.0), coarse_fine = (3.0, 5.0, 5.0, 7.0),
    cut = (1.0, 4.0, 0.0, 2.0))

function donating_data()
    io = IOBuffer()
    info = Dict{Symbol, Any}()
    field, velocity = vof_shear_fixture()
    dt = stable_timestep(velocity, WYAdvectionWorkspace(field))
    replica = wy_replica(field, one_velocity_carrier(velocity), dt, (1, 2))
    regions = replica.states[2].regions
    for panel in (:uniform, :coarse_fine)
        window = DONATING_WINDOWS[panel]
        println(io, "\\ifnum\\fxvpanel=$(panel == :uniform ? 1 : 2)")
        state_text(io, field, window)
        regions_text(io, field, regions, window)
        velocity_text(io, field.mesh, velocity, window; axis = 1)
        println(io, "\\fi")
    end
    info[:shear_dt] = dt
    cut, cut_velocity = vof_cut_figure_fixture()
    cut_dt = stable_timestep(cut_velocity, WYAdvectionWorkspace(cut))
    cut_replica = wy_replica(cut, one_velocity_carrier(cut_velocity), cut_dt, (1, 2))
    window = DONATING_WINDOWS[:cut]
    println(io, "\\ifnum\\fxvpanel=3")
    state_text(io, cut, window)
    regions_text(io, cut, cut_replica.states[2].regions, window)
    velocity_text(io, cut.mesh, cut_velocity, window; axis = 1)
    println(io, "\\fi")
    info[:cut_dt] = cut_dt
    # The shown cut donor: the triple cell [2,3]x[0,1] and its face x = 3.
    index = findfirst(f -> f.axis == 1 && f.center == (3.0, 0.5), cut.mesh.faces)
    region = only(r.result for r in cut_replica.states[2].regions if r.index == index)
    info[:cut_depth] = region.depth
    info[:cut_extrusion] = cut_dt * abs(cut_velocity.interior[index])
    info[:cut_aperture] = aperture_of(cut, index)
    return String(take!(io)), info
end

# Figure "sweeps": one time step on the uniform fixture, window of 4 x 3 cells.
const SWEEP_WINDOW = (3.0, 7.0, 6.0, 9.0)

function sweeps_data()
    io = IOBuffer()
    field, velocity = vof_fixture(:uniform)
    dt = stable_timestep(velocity, WYAdvectionWorkspace(field))
    replica = wy_replica(field, one_velocity_carrier(velocity), dt, (1, 2))
    states = replica.states
    window = SWEEP_WINDOW
    for (k, state) in enumerate(states)
        println(io, "\\ifnum\\fxvpanel=$k")
        state_text(io, field, window; C = state.C, plane = state.plane, chi = replica.chi)
        k < 3 && regions_text(io, field, states[k + 1].regions, window)
        k < 3 && velocity_text(io, field.mesh, velocity, window; axis = k)
        if k > 1
            for (ref, value) in state.dilation
                box = cell_box_of(field.mesh, ref)
                in_window(box, window) && replica.chi[ref] == 1 || continue
                println(io, "\\fxvdil{$(fmt(box[1]))}{$(fmt(box[2]))}{$(@sprintf("%+.3f", value / cell_volume(field.mesh, ref)))}")
            end
        end
        println(io, "\\fi")
    end
    # Dilation of the two sweeps in the chi = 1 cells of the window, and its sum.
    worst_sum = maximum(abs(states[2].dilation[ref] + states[3].dilation[ref])
        for ref in all_cells(field.mesh) if replica.chi[ref] == 1)
    largest = maximum(abs(states[2].dilation[ref]) for ref in all_cells(field.mesh) if replica.chi[ref] == 1)
    return String(take!(io)), (; dt, worst_sum, largest)
end

# Figure "frozen": frozen phase subfaces of FrozenInterfaceFluxes on one face,
# without and with a solid; liquid moves in +x, gas in -x.
function vof_frozen_fixture(with_solid)
    mesh = BlockTreeMesh(;
        base_mesh = (4, 4), block_size = (4, 4), extent = (4.0, 4.0), periodic = (false, false),
    )
    geometry = with_solid ? GeometryField(mesh, (x, y) -> y - 0.25 - 0.1x; lipschitz = hypot(1, 0.1)) :
        nothing
    # Interface y = level - slope (x - 2); with the solid it stays above the
    # wall in the window (contact beyond x = 3), but the cut cells are triple
    # cells and carry the GHF or WY lines of Chapter 4.
    level, slope = with_solid ? (0.85, 0.1) : (1.75, 0.35)
    field = InterfaceField(mesh, (x, y) -> y - level + slope * (x - 2); geometry,
        lipschitz = hypot(1, slope), reconstruction_method = :lhf_ghf)
    velocity = (
        FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? 0.8 : 0.0),
        FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? -0.6 : 0.0),
    )
    enforce_no_permeation!(velocity)
    return field, velocity
end

const FROZEN_WINDOW = (1.0, 3.0, 0.0, 3.0)

function frozen_data()
    io = IOBuffer()
    info = Dict{Int, Any}()
    for (panel, with_solid) in ((1, false), (2, true))
        field, velocity = vof_frozen_fixture(with_solid)
        work = WYAdvectionWorkspace(field, velocity; interface_transport = FrozenInterfaceFluxes())
        dt = stable_timestep(velocity, work)
        apertures = (copy(work.liquid_aperture), copy(work.gas_aperture))
        replica = wy_replica(field, frozen_carrier(velocity, apertures), dt, (1, 2))
        mesh = field.mesh
        println(io, "\\ifnum\\fxvpanel=$panel")
        state_text(io, field, FROZEN_WINDOW)
        # The shown face x = 2 between the cells of row 1 (y in [1,2]) or row 0.
        y = with_solid ? 0.5 : 1.5
        index = findfirst(f -> f.axis == 1 && f.center == (2.0, y), mesh.faces)
        face = mesh.faces[index]
        for region in replica.states[2].regions
            region.index == index || continue
            r = region.result
            rectangle = to_physical(mesh, r.donor, r.rectangle)
            open = isempty(r.open) ? "" : polygon_text(to_physical(mesh, r.donor, r.open))
            liquid = isempty(r.liquid_polygon) ? "" :
                polygon_text(to_physical(mesh, r.donor, r.liquid_polygon))
            println(io, "\\fxvsub{$(region.phase)}{$(polygon_text(rectangle))}{$open}{$liquid}")
            # Subface interval on the face, from the donor's local tangential range.
            centre = cellcenter(mesh, r.donor.block, r.donor.i, r.donor.j)
            h = cell_spacing(mesh, r.donor)[2]
            ys = [p[2] for p in r.rectangle]
            lo, hi = centre[2] + h * minimum(ys), centre[2] + h * maximum(ys)
            println(io, "\\fxvsubface{$(region.phase)}{$(fmt(face.center[1]))}{$(fmt(lo))}{$(fmt(hi))}",
                "{$(@sprintf("%.2f", velocity[region.phase].interior[index]))}")
        end
        println(io, "\\fxvapertures{$(fmt(face.center[1]))}{$(fmt(face.center[2]))}",
            "{$(@sprintf("%.2f", apertures[1][index]))}{$(@sprintf("%.2f", apertures[2][index]))}",
            "{$(@sprintf("%.2f", aperture_of(field, index)))}")
        println(io, "\\fi")
        info[panel] = (; dt, liquid = apertures[1][index], gas = apertures[2][index])
    end
    return String(take!(io)), info
end

# --- Outputs of Chapter 5: figure data and the numbers quoted in the text ------

vof_gas_velocity(field, kind) = kind in (:uniform, :amr) ?
    streamfunction_velocity(field.mesh, vof_psi2_periodic) :
    streamfunction_velocity(field.mesh, vof_psi2_walls)

function vof_outputs(write_output)
    donating, donating_info = donating_data()
    write_output("vof_donating.tex", donating)
    sweeps, sweeps_info = sweeps_data()
    write_output("vof_sweeps.tex", sweeps)
    frozen, frozen_info = frozen_data()
    write_output("vof_frozen.tex", frozen)

    worst = Dict(:uniform => 0.0, :coarse_fine => 0.0, :cut => 0.0)
    count = Dict(:uniform => 0, :coarse_fine => 0, :cut => 0)
    mixed = Dict(:uniform => 0, :coarse_fine => 0, :cut => 0)
    partition = frozen_worst = frozen_cfl = equal = cfl_diff = 0.0
    frozen_faces = frozen_split = 0
    replica = residual = reduction = reset = 0.0
    constants = true
    steps = Dict{Symbol, Float64}()
    cut_info = nothing
    projected_cut = nothing
    open_divergence = 0.0
    deepest = 0.0
    for kind in (:uniform, :amr, :cut, :film)
        field, velocity = vof_fixture(kind)
        dt = stable_timestep(velocity, WYAdvectionWorkspace(field))
        flux = one_velocity_flux_check(copy(field), velocity, dt)
        for class in keys(worst)
            worst[class] = max(worst[class], flux.worst[class])
            count[class] += flux.count[class]; mixed[class] += flux.mixed[class]
        end
        partition = max(partition, flux.partition)
        for order in ((1, 2), (2, 1))
            s = step_balance_check(copy(field), velocity; order)
            replica = max(replica, s.difference); residual = max(residual, s.residual)
            constants &= s.constants
            @assert s.snaps == 0
            kind in (:uniform, :amr) && (steps[kind] = max(get(steps, kind, 0.0), abs(s.change) / s.before))
            kind == :cut && order == (1, 2) && (steps[:cut_plain] = s.change)
        end
        c = cfl_check(copy(field), velocity)
        cfl_diff = max(cfl_diff, c.relative)
        @assert c.accepted && c.rejected
        kind == :uniform && (deepest = c.deepest_uncut)
        kind == :cut && (cut_info = (; c.code, extrusion = c.indep.extrusion,
            smallest = minimum(open_fraction_of(field, r) for r in all_cells(field.mesh)
                if open_fraction_of(field, r) > 0)))
        if kind == :cut
            open = open_streamfunction_velocity(field, vof_psi_walls)
            open_divergence = maximum_open_divergence(field, open)
            s = step_balance_check(copy(field), open)
            replica = max(replica, s.difference); residual = max(residual, s.residual)
            @assert s.snaps == 0
            steps[:cut] = abs(s.change) / s.before
        end
        kind == :film && continue
        two = (velocity, vof_gas_velocity(field, kind))
        work = WYAdvectionWorkspace(copy(field), two; interface_transport = FrozenInterfaceFluxes())
        fdt = stable_timestep(two, work)
        f = frozen_flux_check(copy(field), two, fdt)
        frozen_worst = max(frozen_worst, f.worst)
        frozen_faces += f.faces; frozen_split += f.split
        fc = frozen_cfl_check(copy(field), two)
        frozen_cfl = max(frozen_cfl, fc.relative)
        @assert fc.accepted && fc.rejected
        fs = frozen_step_check(copy(field), two)
        replica = max(replica, fs.difference); residual = max(residual, fs.residual)
        constants &= fs.constants
        p = projected_step_check(copy(field), two)
        replica = max(replica, p.difference); residual = max(residual, p.residual)
        reduction = max(reduction, p.reduction); constants &= p.constants
        reset = max(reset, abs(p.reset_absolute - p.reset_absolute_replica) / p.reset_absolute)
        kind == :cut && (projected_cut = p)
        if kind in (:uniform, :amr)
            equal = max(equal, equal_velocity_reduction_check(copy(field), velocity, dt))
        end
    end
    @assert constants
    @assert max(worst[:uniform], worst[:coarse_fine], worst[:cut], frozen_worst) <= 1.0e-15
    @assert replica <= 1.0e-15 && residual <= 1.0e-14 && reduction == 0
    @assert abs(projected_cut.change) <= projected_cut.dilation_bound + 1.0e-14
    strip = strip_limitation_check()
    @assert strip.target > strip.strip
    println("vof flux check:              ", (; worst, count, mixed, partition))
    println("vof frozen check:            ", (; frozen_worst, frozen_faces, frozen_split, frozen_cfl, equal))
    println("vof balance:                 ", (; replica, residual, reduction, reset, steps, open_divergence))
    println("vof projected (cut):         ", projected_cut)
    println("vof cfl:                     ", (; cfl_diff, deepest, cut_info))
    println("vof strip limitation:        ", strip)
    fix2(x) = @sprintf("%.2f", x)
    return Any[
        ("chkVofFluxUniformFaces", count[:uniform]), ("chkVofFluxUniformMixed", mixed[:uniform]),
        ("chkVofFluxUniformDiff", sci(worst[:uniform])),
        ("chkVofFluxCoarseFaces", count[:coarse_fine]), ("chkVofFluxCoarseMixed", mixed[:coarse_fine]),
        ("chkVofFluxCoarseDiff", sci(worst[:coarse_fine])),
        ("chkVofFluxCutFaces", count[:cut]), ("chkVofFluxCutMixed", mixed[:cut]),
        ("chkVofFluxCutDiff", sci(worst[:cut])), ("chkVofPartition", sci(partition)),
        ("chkVofFrozenFaces", frozen_faces), ("chkVofFrozenSplit", frozen_split),
        ("chkVofFrozenDiff", sci(frozen_worst)), ("chkVofFrozenCflDiff", sci(frozen_cfl)),
        ("chkVofEqualReduction", sci(equal)),
        ("chkVofReplicaDiff", sci(replica)), ("chkVofBalanceResidual", sci(residual)),
        ("chkVofStepUniform", sci(steps[:uniform])), ("chkVofStepAmr", sci(steps[:amr])),
        ("chkVofStepCut", sci(steps[:cut])), ("chkVofCutDilation", sci(steps[:cut_plain])),
        ("chkVofOpenDivergence", sci(open_divergence)),
        ("chkVofProjectedReduction", sci(reduction)), ("chkVofResetDiff", sci(reset)),
        ("chkVofProjectedDivergence", sci(projected_cut.divergence)),
        ("chkVofProjectedChange", sci(abs(projected_cut.change))),
        ("chkVofProjectedBound", sci(projected_cut.dilation_bound)),
        ("chkVofCflDiff", sci(cfl_diff)), ("chkVofCflDepth", @sprintf("%.3f", deepest)),
        ("chkVofCutSmallest", @sprintf("%.3f", cut_info.smallest)),
        ("chkVofCutDt", @sprintf("%.3f", cut_info.code)),
        ("chkVofCutExtrusionDt", @sprintf("%.3f", cut_info.extrusion)),
        ("chkVofStripOpen", @sprintf("%.3f", strip.open)), ("chkVofStripAperture", fix2(strip.aperture)),
        ("chkVofStripArea", @sprintf("%.4f", strip.strip)),
        ("chkVofStripTarget", @sprintf("%.4f", strip.target)),
        ("chkVofStripDt", @sprintf("%.4f", strip.dt)),
        ("chkVofCutAperture", fix2(donating_info[:cut_aperture])),
        ("chkVofCutDepth", @sprintf("%.3f", donating_info[:cut_depth])),
        ("chkVofCutExtrusion", @sprintf("%.3f", donating_info[:cut_extrusion])),
        ("chkVofSweepDt", @sprintf("%.3f", sweeps_info.dt)),
        ("chkVofSweepDilation", fix2(sweeps_info.largest)),
        ("chkVofSweepDilationSum", sci(sweeps_info.worst_sum)),
        ("chkVofFrozenLiquidA", fix2(frozen_info[1].liquid)), ("chkVofFrozenGasA", fix2(frozen_info[1].gas)),
        ("chkVofFrozenLiquidB", fix2(frozen_info[2].liquid)), ("chkVofFrozenGasB", fix2(frozen_info[2].gas)),
        ("chkVofFrozenOpenB", fix2(frozen_info[2].liquid + frozen_info[2].gas)),
    ]
end
