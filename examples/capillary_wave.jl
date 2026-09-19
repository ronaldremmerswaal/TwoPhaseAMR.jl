using TwoPhaseAMR

"""
    run_example(; density_ratio=1e3, laplace=3000, amplitude=0.01,
                max_level=2, final_time=nothing, visualize=false, kwargs...)

Run a small-amplitude standing capillary wave. By default the simulation
advances through one linear capillary period. Additional keywords are passed
directly to [`FlowSimulation`](@ref), for example
`velocity_model=TwoVelocityModel()`.
"""
function run_example(;
        density_ratio = 1.0e3,
        laplace = 3000.0,
        amplitude = 0.01,
        max_level = 2,
        final_time = nothing,
        visualize = false,
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
    dispersion_modes = (-frequency, frequency)
    analytical_amplitude(time) = real(amplitude * (
        cis(-dispersion_modes[1] * time) +
            cis(-dispersion_modes[2] * time)
    ) / 2)
    capillary_period = 2pi / frequency
    flow_options = (;
        boundary = FreeSlip(),
        diagnostics = (
            Diagnostic(
                :amplitude,
                state -> interface_height(state.interface) - 0.25,
            ),
            Diagnostic(
                :analytical_amplitude,
                state -> analytical_amplitude(state.time),
            ),
        ),
        adaptivity = Adaptivity(
            criteria = max_level > 0 ? (refinement,) : (),
        ),
        kwargs...,
    )
    flow = FlowSimulation(mesh, physics; flow_options...)

    target_time = isnothing(final_time) ? capillary_period : final_time
    visualization = nothing
    observer = NoVisualization()
    if visualize
        fields = VisualizationFieldRegistry()
        register_default_fields!(fields, flow)
        register_diagnostic!(
            fields, flow, :amplitude;
            label = "Numerical amplitude", color = :dodgerblue,
        )
        register_diagnostic!(
            fields, flow, :analytical_amplitude;
            label = "Linear amplitude", color = :black,
        )
        viewer = TwoPhaseAMR.visualize(
            fields; field = :volume_fraction, show_blocks = true,
        )
        session = VisualizationSession(
            fields; viewer, checkpoints = 21, live_dt = target_time / 200,
            asynchronous = true,
        )
        observer = session
        visualization = (; fields, viewer, session)
    end
    run_simulation() = simulate!(
        flow, target_time; visualization = observer,
    )
    result = isnothing(visualization) ? run_simulation() :
        fetch(Threads.@spawn run_simulation())
    isnothing(visualization) || wait_visualization!(visualization.session)
    return (; flow, result, visualization)
end

if abspath(PROGRAM_FILE) == @__FILE__
    using GLMakie
    GLMakie.activate!()
    run = run_example(visualize = true)
    print_simulation_summary(run.flow, run.result)
    scene = visualization_figure(run.visualization.viewer).scene
    scene.events.window_open[] && wait(scene)
end
