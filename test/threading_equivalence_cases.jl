# Cases of the serial-versus-threaded equivalence test, run in a fresh process
# per thread count by `test/threading_equivalence.jl`:
#
#     julia --project=. --threads=N test/threading_equivalence_cases.jl
#
# The grain is one, so every loop runs its threaded path as soon as a second
# thread exists. One line per case reports the steps, the mesh updates, the
# leaf blocks, the wall time and two SHA-256 digests: every state field
# (velocity components per phase, pressure, liquid fraction) and every recorded
# diagnostic history; the parent test compares those lines across thread
# counts. One `alloc=<case>` line per allocation case reports the bytes a single
# warm step allocated after the warm-up.
using LinearAlgebra
BLAS.set_num_threads(1)
using SHA
using TwoPhaseAMR

# Every loop, however small, takes its threaded branch.
TwoPhaseAMR._PARALLEL_GRAIN[] = 1

const EXAMPLES = joinpath(@__DIR__, "..", "examples")
module Cylinder
    include(joinpath(Main.EXAMPLES, "04_cylinder_flow", "cylinder_flow.jl"))
end
module Stokes3
    include(joinpath(Main.EXAMPLES, "10_stokes3_wave", "stokes3_wave.jl"))
end

const CMDGFM = TwoVelocityModel(pressure_projection = CMDGFMProjection())

"""Diagnostics of the two-phase cases: their reductions run every step inside
the threaded kernels, so the comparison covers them too."""
const TWO_PHASE_DIAGNOSTICS = (
    KineticEnergy(), LinearMomentum(), PhaseVolume(), DivergenceNorm(),
    LeafBlockCount(),
)

"""One 16×16 root block refined to `max_level`, Taylor–Green vortices."""
function taylor_green_flow(max_level)
    mesh = BlockTreeMesh(
        block_size = (16, 16), extent = (2pi, 2pi), max_level = max_level,
        refine = (key, origin, extent) -> key.level < max_level,
    )
    velocity(x, y, time, axis) = exp(-0.04time) *
        (axis == 1 ? sin(x) * cos(y) : -cos(x) * sin(y))
    flow = FlowSimulation(
        mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.02);
        velocity, boundary = FreeSlip(),
    )
    return flow, stable_timestep(flow; cfl = 0.45)
end

"""CMDGFM capillary wave: `k` wavelengths wide, the interface band refined to
`max_level`. `adaptation = false` skips the setup refinement and the
adaptation criteria, leaving the uniform base mesh (the allocation case)."""
function capillary_flow(; k = 4, max_level = 2, adaptation = true, diagnostics = ())
    wavelength = 1.0
    wavenumber = 2pi / wavelength
    density = (1.0, 1.0e-3)
    surface_tension = 1.0
    dynamic_viscosity = density .* sqrt(surface_tension * wavelength / 3000.0)
    levelset = (x, y) -> y - 0.25 - 0.01cos(wavenumber * x)
    mesh = BlockTreeMesh(
        base_mesh = (8k, 24k), block_size = (8, 8),
        origin = (0.0, -1.5), extent = (wavelength, 3.0),
        periodic = (true, false), max_level = max_level,
        adaptation = MeshAdaptationOptions(update_interval = 5, coarsen_delay = 2),
    )
    interface = InterfaceField(mesh, levelset; lipschitz = 1)
    refinement = InterfaceRefinementIndicator(interface; level = max_level)
    if adaptation
        workspace = AMRWorkspace(mesh, interface)
        for pass in 1:(max_level + 1)
            adapt!(workspace, refinement; step = pass, force = true).changed || break
            initialize!(interface, levelset; lipschitz = 1)
        end
    end
    physics = TwoPhase(interface; density, dynamic_viscosity, surface_tension)
    flow = FlowSimulation(
        mesh, physics; velocity_model = CMDGFM, boundary = FreeSlip(), diagnostics,
        adaptivity = adaptation ?
            Adaptivity(criteria = (refinement,)) : Adaptivity(),
    )
    return flow, 0.5 * stable_timestep(flow)
end

"""Uniform capillary wave with the mimetic cut-cell projection."""
function mimetic_capillary_flow(; diagnostics = ())
    levelset = (x, y) -> y - 0.25 - 0.01cos(2pi * x)
    mesh = BlockTreeMesh(
        base_mesh = (32, 96), block_size = (8, 8), origin = (0.0, -1.5),
        extent = (1.0, 3.0), periodic = (true, false),
    )
    interface = InterfaceField(mesh, levelset; lipschitz = 1)
    density = (1.0, 1.0e-3)
    physics = TwoPhase(
        interface; density, surface_tension = 1.0,
        dynamic_viscosity = density .* sqrt(1 / 3000.0),
    )
    model = TwoVelocityModel(pressure_projection = MimeticProjection())
    flow = FlowSimulation(
        mesh, physics; velocity_model = model, boundary = FreeSlip(), diagnostics,
    )
    return flow, 0.5 * stable_timestep(flow)
end

function cylinder_flow(max_level)
    run = Cylinder.run_example(; max_level, final_time = 0.0)
    return run.flow, stable_timestep(run.flow; cfl = 0.45)
end

"""Droplet on an embedded sloped wall (example 09's physics without its
recorder): CMDGFM, gravity, contact angle, interface, curvature and geometry
refinement."""
function droplet_flow(max_level; diagnostics = ())
    wall_angle = pi / 12
    bond = 1.21
    laplace = 5.5e2
    density = (1.0, 1.0e-3)
    surface_tension = 1.0
    radius = 0.16
    wall_point = (0.5, 0.22)
    normal = (-sin(wall_angle), cos(wall_angle))
    wall = (x, y) -> normal[1] * (x - wall_point[1]) +
        normal[2] * (y - wall_point[2])
    contact_angle = pi / 2
    centre = wall_point .- radius * cos(contact_angle) .* normal
    levelset = (x, y) -> hypot(x - centre[1], y - centre[2]) - radius
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = max_level, periodic = (false, false),
        adaptation = MeshAdaptationOptions(update_interval = 2, coarsen_delay = 2),
    )
    geometry = GeometryField(mesh, wall)
    interface = InterfaceField(
        mesh, levelset; geometry, contact_angle, lipschitz = 1,
        atol = 1.0e-7, maxdepth = 24,
    )
    refinement = (
        InterfaceRefinementIndicator(interface; level = min(1, max_level)),
        CurvatureRefinementIndicator(interface; cells_per_radius = 8),
        GeometryRefinementIndicator(geometry; level = min(1, max_level)),
    )
    workspace = AMRWorkspace(mesh, interface, geometry)
    for pass in 1:(max_level + 1)
        adapt!(workspace, refinement; step = pass, force = true).changed || break
        initialize!(interface, levelset; lipschitz = 1)
    end
    gravity = bond * surface_tension / ((density[1] - density[2]) * radius^2)
    liquid_viscosity = sqrt(surface_tension * density[1] * radius / laplace)
    dynamic_viscosity = (liquid_viscosity, 0.0175liquid_viscosity)
    physics = TwoPhase(
        interface; density, dynamic_viscosity, surface_tension,
        gravity = -gravity .* normal,
    )
    flow = FlowSimulation(
        mesh, physics; velocity_model = CMDGFM, geometry, diagnostics,
        boundary = FreeSlip(), adaptivity = Adaptivity(criteria = refinement),
    )
    return flow, 0.1 * stable_timestep(flow)
end

function stokes3_flow(max_level)
    run = Stokes3.run_example(;
        max_level, final_time = 0.0, velocity_model = CMDGFM,
    )
    return run.flow, 0.5 * stable_timestep(run.flow)
end

"""Cases: name, number of fixed-dt steps, and the flow builder."""
const CASES = (
    ("capillary", 10, () -> capillary_flow(diagnostics = TWO_PHASE_DIAGNOSTICS)),
    ("capillary_mimetic", 10, () -> mimetic_capillary_flow(diagnostics = TWO_PHASE_DIAGNOSTICS)),
    ("droplet", 6, () -> droplet_flow(2; diagnostics = TWO_PHASE_DIAGNOSTICS)),
    ("stokes3_two", 6, () -> stokes3_flow(3)),
    ("cylinder", 6, () -> cylinder_flow(2)),
    ("taylor_green", 10, () -> taylor_green_flow(3)),
)

"""Cases whose warm step must stay within the recorded allocation budget."""
const ALLOCATION_CASES = (
    ("taylor_green", () -> taylor_green_flow(3)),
    ("capillary", () -> capillary_flow(adaptation = false)),
)

"""SHA-256 over every state field of `flow`: the velocity components per phase
on the interior and boundary faces, the pressure and the liquid fraction."""
function state_hash(flow)
    context = SHA.SHA256_CTX()
    velocity = flow.velocity
    phases = velocity isa TwoPhaseAMR.TwoPhaseVelocity ? collect(velocity) : (velocity,)
    for phase in phases
        SHA.update!(context, reinterpret(UInt8, phase.interior))
        SHA.update!(context, reinterpret(UInt8, phase.boundary))
    end
    interface = flow.interface
    fields = isnothing(interface) ?
        (flow.pressure,) : (flow.pressure, interface.fraction)
    for field in fields, block in field.blocks
        SHA.update!(context, reinterpret(UInt8, vec(block)))
    end
    return bytes2hex(SHA.digest!(context))
end

"""Hash one diagnostic sample: a number, or a tuple or static vector of them."""
function update_sample!(context, sample::Real)
    SHA.update!(context, reinterpret(UInt8, [Float64(sample)]))
    return nothing
end

function update_sample!(context, sample::Union{Tuple, AbstractArray})
    for component in sample
        update_sample!(context, component)
    end
    return nothing
end

"""SHA-256 over every recorded diagnostic history: for each series, sorted by
name, its name, the time and step of every sample, and the sample values."""
function diagnostics_hash(flow)
    context = SHA.SHA256_CTX()
    names = sort!(diagnostic_names(flow.diagnostics))
    for name in names
        history = diagnostic_history(flow, name)
        SHA.update!(context, codeunits(string(name)))
        SHA.update!(context, reinterpret(UInt8, history.times))
        SHA.update!(context, reinterpret(UInt8, history.steps))
        for sample in history.values
            update_sample!(context, sample)
        end
    end
    return bytes2hex(SHA.digest!(context)), length(names)
end

function emit_case(name, steps, build)
    started = time()
    flow, dt = build()
    for _ in 1:steps
        simulate!(flow, flow.time + dt; dt)
    end
    diagnostics, series = diagnostics_hash(flow)
    println(
        "case=", name, " threads=", Threads.nthreads(:default),
        " steps=", flow.steps, " mesh_updates=", flow.mesh_updates,
        " blocks=", length(flow.pressure.blocks),
        " seconds=", round(time() - started; digits = 2),
        " series=", series, " diagnostics=", diagnostics,
        " sha256=", state_hash(flow),
    )
    flush(stdout)
    return nothing
end

function emit_allocation(name, build)
    flow, dt = build()
    # One step to leave the first-call path, one to reach the steady state of
    # every workspace the step touches.
    simulate!(flow, flow.time + dt; dt)
    simulate!(flow, flow.time + dt; dt)
    GC.gc()
    bytes = @allocated simulate!(flow, flow.time + dt; dt)
    println(
        "alloc=", name, " threads=", Threads.nthreads(:default), " bytes=", bytes,
    )
    flush(stdout)
    return nothing
end

for (name, steps, build) in CASES
    emit_case(name, steps, build)
end
for (name, build) in ALLOCATION_CASES
    emit_allocation(name, build)
end
