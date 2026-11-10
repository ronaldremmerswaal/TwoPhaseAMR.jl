using TwoPhaseAMR

"""
    run_example(; wall_angle=pi/12, bond=1.21, laplace=550,
                density_ratio=1000, contact_angle=pi/2, max_level=2,
                final_time=nothing, start_studio=nothing,
                recorder_options=(;), kwargs...)

Run a sessile droplet on an embedded inclined wall. The default end time is
four capillary times. Pass `start_studio=studio` after loading TwoPhaseAMRStudio
to follow the run live. `recorder_options` overrides the recording schedule;
other keywords are passed to [`FlowSimulation`](@ref).
"""
function run_example(;
        wall_angle = pi / 12,
        bond = 1.21,
        laplace = 5.5e2,
        density_ratio = 1.0e3,
        contact_angle = pi / 2,
        max_level = 2,
        final_time = nothing,
        start_studio = nothing,
        recorder_options = (;),
        kwargs...,
    )
    density = (1.0, inv(density_ratio))
    surface_tension = 1.0
    radius = 0.16
    wall_point = (0.5, 0.22)
    normal = (-sin(wall_angle), cos(wall_angle))
    wall = (x, y) -> normal[1] * (x - wall_point[1]) +
        normal[2] * (y - wall_point[2])
    centre = wall_point .- radius * cos(contact_angle) .* normal
    levelset = (x, y) -> hypot(x - centre[1], y - centre[2]) - radius

    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = max_level,
        periodic = (false, false),
        adaptation = MeshAdaptationOptions(
            update_interval = 2, coarsen_delay = 2,
        ),
    )
    geometry = GeometryField(mesh, wall)
    interface = InterfaceField(
        mesh, levelset; geometry, contact_angle, lipschitz = 1,
        atol = 1.0e-7, maxdepth = 24,
    )
    refinement = (
        InterfaceRefinementIndicator(
            interface; level = min(1, max_level),
        ),
        CurvatureRefinementIndicator(interface; cells_per_radius = 8),
        GeometryRefinementIndicator(
            geometry; level = min(1, max_level),
        ),
    )
    adaptation = AMRWorkspace(mesh, interface, geometry)
    for pass in 1:(max_level + 1)
        adapt!(adaptation, refinement; step = pass, force = true).changed ||
            break
        initialize!(interface, levelset; lipschitz = 1)
    end

    gravity = bond * surface_tension /
        ((density[1] - density[2]) * radius^2)
    liquid_viscosity = sqrt(surface_tension * density[1] * radius / laplace)
    dynamic_viscosity = (liquid_viscosity, 0.0175liquid_viscosity)
    physics = TwoPhase(
        interface; density, dynamic_viscosity, surface_tension,
        gravity = -gravity .* normal,
    )
    flow_options = (;
        geometry, boundary = FreeSlip(),
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
    target_time = isnothing(final_time) ? 4capillary_time : final_time
    fields = VisualizationFieldRegistry()
    register_default_fields!(fields, flow)
    register_geometry!(fields, geometry)
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
        isnothing(driver) ? simulate!(flow, target_time; observer = recorder) :
            simulate!(driver, target_time)
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
