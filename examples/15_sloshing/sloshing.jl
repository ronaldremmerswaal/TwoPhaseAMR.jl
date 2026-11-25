using TwoPhaseAMR

Base.include(@__MODULE__, joinpath(@__DIR__, "linear_sloshing.jl"))

"""
    strip_elevation(interface, left, width, depth)

Mean free-surface elevation over the full-height strip `left <= x <= left +
width`: the liquid volume of the cells in the strip, weighted by their overlap,
divided by `width`, minus the still depth. Spray and films in the strip count
as liquid height, so the value stays defined and continuous when the surface
at the wall overturns or breaks up.
"""
function strip_elevation(interface, left, width, depth)
    mesh = interface.mesh
    volume = zero(width)
    for (index, block) in pairs(mesh.blocks)
        block.origin[1] < left + width && block.origin[1] + block.extent[1] > left ||
            continue
        dx, dy = block.spacing
        fractions = interior(interface, index)
        for i in axes(fractions, 1)
            low = block.origin[1] + (i - 1) * dx
            overlap = min(low + dx, left + width) - max(low, left)
            overlap > 0 || continue
            volume += overlap * dy * sum(view(fractions, i, :))
        end
    end
    return volume / width - depth
end

"""
    run_example(; forcing_amplitude=0.002, forcing_frequency=5.29,
                initial_amplitude=0, baffle_height=0, baffle_thickness=0.01,
                max_level=2, final_time=10, velocity_model=OneVelocityModel(),
                numerics=NavierStokesOptions(advection=AdvectionOptions(scheme=MonotonizedCentral())),
                start_studio=nothing, recorder_options=(;), kwargs...)

Run water sloshing in a closed 1 m × 1 m tank, 0.5 m deep, in SI units. The
tank is displaced horizontally by `-forcing_amplitude*sin(forcing_frequency*t)`
from rest, so it starts impulsively while the fluids are still at rest: the
tank-fixed frame starts from the projected relative velocity
`forcing_amplitude*forcing_frequency` and applies the opposite of the tank
acceleration as a body force. The defaults are the near-resonant case of
Faltinsen (1978) and Liu & Lin (2008). `initial_amplitude` adds the first
mode `cos(πx)` to the initial surface for free sloshing. A positive
`baffle_height` adds a central, bottom-mounted baffle as an exact polygonal
solid. Momentum is advected with the monotonized-central limiter by default:
water and air in a 1 m tank are nearly inviscid on these meshes, and
Lax–Wendroff leaves dispersive grid-scale noise in the vorticity along the
surface and behind the baffle. Pass `start_studio=studio` after loading
TwoPhaseAMRStudio to follow the run live. `recorder_options` overrides the
recording schedule; other keywords are passed to [`FlowSimulation`](@ref).
"""
function run_example(;
        forcing_amplitude = 0.002,
        forcing_frequency = 5.29,
        initial_amplitude = 0.0,
        baffle_height = 0.0,
        baffle_thickness = 0.01,
        max_level = 2,
        final_time = 10.0,
        velocity_model = OneVelocityModel(),
        numerics = NavierStokesOptions(
            advection = AdvectionOptions(scheme = MonotonizedCentral()),
        ),
        start_studio = nothing,
        recorder_options = (;),
        kwargs...,
    )
    tank = sloshing_tank(; baffle_height, baffle_thickness)
    L, depth = tank.length, tank.depth
    k = pi / L
    levelset = (x, y) -> y - depth - initial_amplitude * cos(k * x)
    lipschitz = hypot(1, initial_amplitude * k)

    mesh = BlockTreeMesh(
        base_mesh = (32, 32), block_size = (8, 8),
        extent = (L, tank.height), periodic = (false, false),
        max_level = max_level,
        adaptation = MeshAdaptationOptions(
            update_interval = 5, coarsen_delay = 10,
        ),
    )
    baffle = baffle_height > 0 ? PolygonIndicator(
            [
                (L / 2 - baffle_thickness / 2, -0.1),
                (L / 2 + baffle_thickness / 2, -0.1),
                (L / 2 + baffle_thickness / 2, baffle_height),
                (L / 2 - baffle_thickness / 2, baffle_height),
            ]
        ) : nothing
    geometry = isnothing(baffle) ? nothing :
        GeometryField(InterfaceField(mesh, baffle))
    interface = InterfaceField(mesh, levelset; geometry, lipschitz)
    refinement = (
        InterfaceRefinementIndicator(interface; level = max_level),
        (isnothing(geometry) ? () : (GeometryRefinementIndicator(geometry; level = max_level),))...,
    )
    adaptation = isnothing(geometry) ? AMRWorkspace(mesh, interface) :
        AMRWorkspace(mesh, interface, geometry)
    for pass in 1:(max_level + 1)
        adapt!(adaptation, refinement; step = pass, force = true).changed ||
            break
        if !isnothing(geometry)
            initialize!(geometry.solid, baffle)
            refresh_geometry!(geometry)
        end
        initialize!(interface, levelset; lipschitz)
    end

    physics = TwoPhase(
        interface; tank.density, tank.dynamic_viscosity, tank.surface_tension,
        gravity = (0.0, -tank.gravity),
    )
    ω = forcing_frequency
    acceleration(time) = forcing_amplitude * ω^2 * sin(ω * time)
    force = iszero(forcing_amplitude) ? nothing :
        (x, y, time, axis) -> axis == 1 ? -acceleration(time) : zero(time)
    elevation = PhysicalDimension(0, 1, 0)
    width = tank.probe_width
    probes = (
        Diagnostic(
            :left_elevation,
            state -> strip_elevation(state.interface, 0.0, width, depth);
            label = "Left-wall elevation", dimension = elevation,
        ),
        Diagnostic(
            :right_elevation,
            state -> strip_elevation(state.interface, L - width, width, depth);
            label = "Right-wall elevation", dimension = elevation,
        ),
    )
    linear = if isnothing(geometry)
        modes = plain_tank_modes(tank)
        (
            Diagnostic(
                :linear_elevation,
                state -> modal_elevation(
                    state.time, modes; initial_amplitude, forcing_amplitude,
                    forcing_frequency,
                );
                label = "Linear left-wall elevation", dimension = elevation,
            ),
        )
    else
        ()
    end
    power = Diagnostic(
        :forcing_power,
        state -> -acceleration(state.time) *
            diagnostic_value(LinearMomentum(), state)[1];
        label = "Forcing power", dimension = PhysicalDimension(1, 1, -3),
    )
    flow_options = (;
        velocity_model, geometry, numerics, force, boundary = FreeSlip(),
        # Fluids at rest in a tank that starts with velocity -Aω.
        velocity = (forcing_amplitude * ω, 0.0),
        diagnostics = (
            probes..., linear..., KineticEnergy(), PotentialEnergy(),
            SurfaceEnergy(), TotalEnergy(), power, PhaseVolume(),
        ),
        adaptivity = Adaptivity(criteria = max_level > 0 ? refinement : ()),
        kwargs...,
    )
    flow = FlowSimulation(mesh, physics; flow_options...)
    # The impulsive start: walls and incompressibility act through an impulse.
    iszero(forcing_amplitude) || project!(flow)
    fields = VisualizationFieldRegistry()
    register_default_fields!(fields, flow)
    isnothing(geometry) || register_geometry!(fields, geometry)
    recorder = Recorder(
        fields; diagnostics = flow,
        merge(
            (;
                keyframes = EquispacedTimes(21),
                preview = isnothing(start_studio) ? nothing : EveryWallInterval(0.2),
            ),
            recorder_options,
        )...,
    )
    driver = isnothing(start_studio) ? nothing : TwoPhaseAMR.StudioDriver(flow, recorder)
    app = isnothing(start_studio) ? nothing : start_studio(recorder)
    result = try
        isnothing(driver) ? simulate!(flow, final_time; observer = recorder) :
            simulate!(driver, final_time)
    catch
        isnothing(app) || close(app)
        rethrow()
    end
    return (; flow, result, recorder, app)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--studio" in ARGS
        using TwoPhaseAMRStudio
        run = run_example(; start_studio = studio)
    else
        run = run_example()
    end
    print_simulation_summary(run.flow, run.result)
    println(snapshot_store(run.recorder))
    if !isnothing(run.app)
        try
            wait(run.app)
        finally
            close(run.app)
        end
    end
end
