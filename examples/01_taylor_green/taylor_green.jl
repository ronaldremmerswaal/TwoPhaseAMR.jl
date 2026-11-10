using TwoPhaseAMR

"""
    run_example(; reynolds=50, max_level=2, final_time=0.2,
                start_studio=nothing, recorder_options=(;), kwargs...)

Run the periodic Taylor--Green vortex on a fixed mixed-level mesh. Pass
`start_studio=studio` after loading TwoPhaseAMRStudio to follow the run live.
`recorder_options` overrides the recording schedule; other keywords are passed
to [`FlowSimulation`](@ref).
"""
function run_example(;
        reynolds = 50.0,
        max_level = 2,
        final_time = 0.2,
        start_studio = nothing,
        recorder_options = (;),
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
