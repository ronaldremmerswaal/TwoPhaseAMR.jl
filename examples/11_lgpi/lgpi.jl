using TwoPhaseAMR

"""Scaled large gas-pocket impact geometry used by the LGPI benchmark."""
function lgpi_initial_condition(; scale = 5.0)
    wavelength = 20.0 / scale
    domain = (wavelength, 3wavelength / 5)
    mean_height = 7.6 / scale
    amplitude = 3.6 / scale
    horizontal_shift = 15.5 / scale
    steepness = 0.44scale
    bathymetry_axes = (18.0 / scale, 2.8 / scale)
    elevation(x) = mean_height + amplitude *
        tanh(steepness * (x - horizontal_shift))
    interface_levelset(x, y) = y - elevation(x)
    bathymetry_levelset(x, y) = hypot(
        x / bathymetry_axes[1], y / bathymetry_axes[2],
    ) - 1
    return (;
        wavelength, domain, interface_levelset, bathymetry_levelset,
        interface_lipschitz = hypot(1, amplitude * steepness),
        bathymetry_lipschitz = maximum(inv, bathymetry_axes),
    )
end

"""
    run_example(; scale=5, max_level=3, final_time=nothing,
                visualize=false, kwargs...)

Run the scaled large gas-pocket impact benchmark. The default end time is
`0.6` gravity-wave periods. Additional keywords are passed directly to
[`FlowSimulation`](@ref), including the velocity model.
"""
function run_example(;
        scale = 5.0,
        max_level = 3,
        final_time = nothing,
        visualize = false,
        kwargs...,
    )
    initial = lgpi_initial_condition(; scale)
    mesh = BlockTreeMesh(
        base_mesh = (40, 24), block_size = (4, 4),
        extent = initial.domain, periodic = (false, false),
        max_level = max_level,
        adaptation = MeshAdaptationOptions(
            update_interval = 5, coarsen_delay = 10,
        ),
    )
    geometry = GeometryField(
        mesh, initial.bathymetry_levelset;
        lipschitz = initial.bathymetry_lipschitz,
        atol = 1.0e-8, maxdepth = 28,
    )
    interface = InterfaceField(
        mesh, initial.interface_levelset; geometry,
        lipschitz = initial.interface_lipschitz,
        atol = 1.0e-8, maxdepth = 28,
    )
    velocity = FaceVelocity(mesh)
    fixed_level = min(1, max_level)
    refinement = (
        InterfaceRefinementIndicator(interface; level = fixed_level),
        VorticityRefinementIndicator(
            velocity; refine_tolerance = 0.25,
            coarsen_tolerance = 0.25 / 3,
            maximum_level = max_level,
        ),
        GeometryRefinementIndicator(geometry; level = fixed_level),
    )
    adaptation = AMRWorkspace(mesh, interface, geometry, velocity)
    for pass in 1:(max_level + 1)
        adapt!(adaptation, refinement; step = pass, force = true).changed ||
            break
        initialize!(
            geometry.solid, initial.bathymetry_levelset;
            lipschitz = initial.bathymetry_lipschitz,
            atol = 1.0e-8, maxdepth = 28,
        )
        refresh_geometry!(geometry)
        initialize!(
            interface, initial.interface_levelset;
            lipschitz = initial.interface_lipschitz,
            atol = 1.0e-8, maxdepth = 28,
        )
        fill!(velocity.interior, 0)
        fill!(velocity.boundary, 0)
    end

    density = (1.0e3, 1.0)
    physics = TwoPhase(
        interface; density,
        dynamic_viscosity = (1.0e-3, 1.75e-5),
        surface_tension = 7.3e-2, gravity = (0.0, -9.81),
    )
    flow_options = (;
        velocity, geometry, boundary = FreeSlip(),
        diagnostics = (
            KineticEnergy(), LinearMomentum(), PhaseVolume(),
            DivergenceNorm(), LeafBlockCount(),
        ),
        adaptivity = Adaptivity(criteria = refinement),
        kwargs...,
    )
    flow = FlowSimulation(mesh, physics; flow_options...)
    gravity_time = sqrt(
        sum(density) / (density[1] - density[2]) *
            2pi * initial.wavelength / 9.81,
    )
    target_time = isnothing(final_time) ? 0.6gravity_time : final_time
    visualization = nothing
    observer = NoVisualization()
    if visualize
        fields = VisualizationFieldRegistry()
        register_default_fields!(fields, flow)
        register_geometry!(fields, geometry)
        register_diagnostic!(fields, flow, :kinetic_energy)
        register_diagnostic!(fields, flow, :leaf_blocks)
        viewer = TwoPhaseAMR.visualize(
            fields; field = :velocity_vorticity, show_blocks = true,
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
