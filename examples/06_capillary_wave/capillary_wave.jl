using TwoPhaseAMR

Base.include(
    @__MODULE__,
    joinpath(
        @__DIR__, "prosperetti_amplitude.jl",
    )
)

"""
    run_example(; density_ratio=1e3, laplace=3000, amplitude=0.01,
                max_level=4, final_time=nothing, start_studio=nothing,
                recorder_options=(;), kwargs...)

Run a small-amplitude standing capillary wave. By default the simulation
advances through one linear capillary period. Pass `start_studio=studio` after
loading TwoPhaseAMRStudio to follow the run live. `recorder_options` overrides
the recording schedule; other keywords are passed to [`FlowSimulation`](@ref),
for example `velocity_model=TwoVelocityModel()`.
"""
function run_example(;
        density_ratio = 1.0e3,
        laplace = 3000.0,
        amplitude = 0.01,
        max_level = 4,
        final_time = nothing,
        start_studio = nothing,
        recorder_options = (;),
        kwargs...,
    )
    wavelength = 1.0
    wavenumber = 2pi / wavelength
    density = (1.0, inv(density_ratio))
    surface_tension = 1.0
    kinematic_viscosity = sqrt(surface_tension * wavelength / laplace)
    dynamic_viscosity = density .* kinematic_viscosity
    levelset = (x, y) ->
    y - 0.25 - amplitude * cos(wavenumber * x)

    mesh = BlockTreeMesh(
        base_mesh = (8, 24), block_size = (8, 8),
        origin = (0.0, -1.5), extent = (wavelength, 3.0),
        periodic = (true, false), max_level = max_level,
        adaptation = MeshAdaptationOptions(
            update_interval = 5, coarsen_delay = 2,
        ),
    )
    interface = InterfaceField(mesh, levelset; lipschitz = 1)
    refinement = InterfaceRefinementIndicator(interface; level = max_level)
    adaptation = AMRWorkspace(mesh, interface)
    for pass in 1:(max_level + 1)
        adapt!(adaptation, refinement; step = pass, force = true).changed ||
            break
        initialize!(interface, levelset; lipschitz = 1)
    end

    physics = TwoPhase(
        interface; density, dynamic_viscosity, surface_tension,
    )
    frequency = sqrt(surface_tension * wavenumber^3 / sum(density))
    analytical_amplitude(time) = amplitude * prosperetti_amplitude(
        time;
        liquid_density = density[1], gas_density = density[2],
        kinematic_viscosity, gravity = 0.0,
        surface_tension, wavenumber,
    )
    capillary_period = 2pi / frequency
    flow_options = (;
        boundary = FreeSlip(),
        diagnostics = (
            Diagnostic(
                :amplitude,
                state -> interface_height(state.interface) - 0.25;
                label = "Numerical amplitude", dimension = PhysicalDimension(0, 1, 0),
            ),
            Diagnostic(
                :analytical_amplitude,
                state -> analytical_amplitude(state.time);
                label = "Linear amplitude", dimension = PhysicalDimension(0, 1, 0),
            ),
        ),
        adaptivity = Adaptivity(
            criteria = max_level > 0 ? (refinement,) : (),
        ),
        kwargs...,
    )
    flow = FlowSimulation(mesh, physics; flow_options...)

    target_time = isnothing(final_time) ? capillary_period : final_time
    fields = VisualizationFieldRegistry()
    register_default_fields!(fields, flow)
    recorder = Recorder(
        fields; diagnostics = flow,
        merge(
            (; keyframes = EquispacedTimes(21),
                preview = isnothing(start_studio) ? nothing : EveryWallInterval(0.2)),
            recorder_options,
        )...,
    )
    app = isnothing(start_studio) ? nothing : start_studio(recorder)
    result = try
        simulate!(flow, target_time; observer = recorder)
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
