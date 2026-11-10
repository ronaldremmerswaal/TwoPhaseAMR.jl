using TwoPhaseAMR

"""
    run_example(; reynolds=100, max_level=1, final_time=15,
                start_studio=nothing, recorder_options=(;), kwargs...)

Run the nondimensional lid-driven cavity on a mesh uniformly refined to
`max_level`. Pass `start_studio=studio` after loading TwoPhaseAMRStudio to
follow the run live. `recorder_options` overrides the recording schedule;
other keywords are passed to [`FlowSimulation`](@ref).
"""
function run_example(;
        reynolds = 100.0,
        max_level = 1,
        final_time = 15.0,
        start_studio = nothing,
        recorder_options = (;),
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
