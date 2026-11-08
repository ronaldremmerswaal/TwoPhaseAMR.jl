using TwoPhaseAMR

"""
    run_example(; reynolds=100, max_level=1, final_time=15, kwargs...)

Run the nondimensional lid-driven cavity. The mesh is uniformly refined to
`max_level`; additional keywords are passed directly to
[`FlowSimulation`](@ref).
"""
function run_example(;
        reynolds = 100.0,
        max_level = 1,
        final_time = 15.0,
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
