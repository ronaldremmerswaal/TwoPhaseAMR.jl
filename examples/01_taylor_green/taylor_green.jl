using TwoPhaseAMR

"""
    run_example(; reynolds=50, max_level=2, final_time=0.2, kwargs...)

Run the periodic Taylor--Green vortex on a fixed mixed-level mesh. Additional
keywords are passed directly to [`FlowSimulation`](@ref).
"""
function run_example(;
        reynolds = 50.0,
        max_level = 2,
        final_time = 0.2,
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
    fields = VisualizationFieldRegistry()
    register_default_fields!(fields, flow)
    recorder = Recorder(
        fields; diagnostics = flow, keyframes = EquispacedTimes(21), preview = nothing,
    )
    result = simulate!(flow, final_time; observer = recorder)
    return (; flow, result, recorder)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run = run_example()
    print_simulation_summary(run.flow, run.result)
    println(snapshot_store(run.recorder))
end
