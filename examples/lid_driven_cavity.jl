using TwoPhaseAMR

"""
    run_example(; reynolds=100, max_level=1, final_time=15,
                visualize=false, kwargs...)

Run the nondimensional lid-driven cavity. The mesh is uniformly refined to
`max_level`; additional keywords are passed directly to
[`FlowSimulation`](@ref).
"""
function run_example(;
        reynolds = 100.0,
        max_level = 1,
        final_time = 15.0,
        visualize = false,
        kwargs...,
    )
    mesh = BlockTreeMesh(
        block_size = (16, 16), periodic = (false, false),
        max_level = max_level,
        refine = (key, origin, extent) -> key.level < max_level,
    )
    lid = NoSlip(
        (x, y, time, axis) -> axis == 1 && y > 0.5 ? 1.0 : 0.0,
    )
    physics = SinglePhase(
        density = 1.0, dynamic_viscosity = inv(reynolds),
    )
    flow_options = (;
        boundary = lid,
        diagnostics = (
            KineticEnergy(), LinearMomentum(), DivergenceNorm(),
            LeafBlockCount(),
        ),
        kwargs...,
    )
    flow = FlowSimulation(mesh, physics; flow_options...)
    visualization = nothing
    observer = NoVisualization()
    if visualize
        fields = VisualizationFieldRegistry()
        register_default_fields!(fields, flow)
        register_diagnostic!(fields, flow, :kinetic_energy)
        viewer = TwoPhaseAMR.visualize(
            fields; field = :velocity_magnitude, show_blocks = true,
        )
        session = VisualizationSession(
            fields; viewer, checkpoints = 21, live_dt = final_time / 200,
            asynchronous = true,
        )
        observer = session
        visualization = (; fields, viewer, session)
    end
    run_simulation() = simulate!(
        flow, final_time; visualization = observer,
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
