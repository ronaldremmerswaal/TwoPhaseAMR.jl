using TwoPhaseAMR

"""Circular-cap initial condition shared with the sessile-drop tests."""
function sessile_droplet_initial_condition(; radius = 0.2, contact_angle = pi / 2)
    centre = (0.5, -radius * cos(contact_angle))
    levelset = (x, y) -> hypot(x - centre[1], y - centre[2]) - radius
    return (; levelset, centre)
end

"""
    run_example(; bond=1.21, laplace=550, density_ratio=1000,
                contact_angle=pi/2, max_level=3, final_time=nothing,
                visualize=false, kwargs...)

Run the gravity-driven relaxation of a circular sessile droplet. The default
end time is eight capillary times. Additional keywords are passed directly to
[`FlowSimulation`](@ref), including `velocity_model=TwoVelocityModel()`.
"""
function run_example(;
        bond = 1.21,
        laplace = 5.5e2,
        density_ratio = 1.0e3,
        contact_angle = pi / 2,
        max_level = 3,
        final_time = nothing,
        visualize = false,
        kwargs...,
    )
    density = (1.0, inv(density_ratio))
    surface_tension = 1.0
    radius = 0.2
    gravity = bond * surface_tension /
        ((density[1] - density[2]) * radius^2)
    liquid_viscosity = sqrt(surface_tension * density[1] * radius / laplace)
    dynamic_viscosity = (liquid_viscosity, 0.0175liquid_viscosity)
    initial = sessile_droplet_initial_condition(; radius, contact_angle)

    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = max_level,
        periodic = (false, false),
        adaptation = MeshAdaptationOptions(
            update_interval = 5, coarsen_delay = 2,
        ),
    )
    interface = InterfaceField(
        mesh, initial.levelset; lipschitz = 1,
        contact_angle = (nothing, nothing, contact_angle, nothing),
    )
    refinement = (
        InterfaceRefinementIndicator(
            interface; level = min(1, max_level),
        ),
        CurvatureRefinementIndicator(
            interface; cells_per_radius = 8,
            minimum_level = min(1, max_level),
        ),
    )
    adaptation = AMRWorkspace(mesh, interface)
    for pass in 1:(max_level + 1)
        adapt!(adaptation, refinement; step = pass, force = true).changed ||
            break
        initialize!(interface, initial.levelset; lipschitz = 1)
    end

    physics = TwoPhase(
        interface; density, dynamic_viscosity, surface_tension,
        gravity = (0.0, -gravity),
    )
    flow_options = (;
        boundary = NoSlip(),
        diagnostics = (
            KineticEnergy(), LinearMomentum(), PhaseVolume(),
            DivergenceNorm(), LeafBlockCount(),
        ),
        adaptivity = Adaptivity(
            criteria = max_level > 0 ? refinement : (),
        ),
        kwargs...,
    )
    flow = FlowSimulation(mesh, physics; flow_options...)
    capillary_time = sqrt(sum(density) * radius^3 / surface_tension)
    target_time = isnothing(final_time) ? 8capillary_time : final_time
    visualization = nothing
    observer = NoVisualization()
    if visualize
        fields = VisualizationFieldRegistry()
        register_default_fields!(fields, flow)
        register_diagnostic!(fields, flow, :kinetic_energy)
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
    result = simulate!(
        flow, target_time; visualization = observer,
    )
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
