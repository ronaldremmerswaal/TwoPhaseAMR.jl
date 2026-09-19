using TwoPhaseAMR

"""
    run_example(; reynolds=50, max_level=2, final_time=0.2,
                visualize=false, kwargs...)

Run the periodic Taylor--Green vortex on a fixed mixed-level mesh. Additional
keywords are passed directly to [`FlowSimulation`](@ref).
"""
function run_example(;
        reynolds = 50.0,
        max_level = 2,
        final_time = 0.2,
        visualize = false,
        kwargs...,
    )
    viscosity = inv(reynolds)
    mesh = BlockTreeMesh(
        block_size = (12, 12), extent = (2pi, 2pi),
        max_level = max_level,
        refine = (key, origin, extent) ->
        key.level == 0 ||
            (key.level < max_level && key.i == 0 && key.j == 0),
    )
    velocity = (x, y, time, axis) -> exp(-2viscosity * time) *
        (axis == 1 ? sin(x) * cos(y) : -cos(x) * sin(y))
    physics = SinglePhase(
        density = 1.0, dynamic_viscosity = viscosity,
    )
    flow_options = (;
        velocity, boundary = FreeSlip(),
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
            fields; field = :velocity_vorticity, show_blocks = true,
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
