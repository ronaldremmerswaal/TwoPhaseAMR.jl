active_project = Base.active_project()
studio_project = normpath(joinpath(@__DIR__, "..", "..", "Project.toml"))
@assert !isnothing(active_project) && realpath(active_project) == realpath(studio_project) "Run this fixture with the Studio project active"

using LinearAlgebra
BLAS.set_num_threads(1)
using TwoPhaseAMR
using TwoPhaseAMRStudio

root = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
push!(LOAD_PATH, joinpath(root, "examples"))
include(joinpath(root, "examples", "06_capillary_wave", "capillary_wave.jl"))

# Run the example's real Studio opt-in path. Pause at the initial state so the
# browser can connect before the first run; the UI resumes it. Slow the recorder
# clock slightly to let clients observe both running transitions, while keeping
# the example's actual simulation, fields, and diagnostic histories intact.
ready = Channel{Any}(1)
task = Threads.@spawn try
    run_example(;
        max_level = 1, final_time = 0.002, dt = 0.0002,
        velocity_model = TwoVelocityModel(),
        recorder_options = (;
            keyframes = EveryNSteps(5), preview = EveryStep(),
            retention = Retention(preview = :all), max_capture_overhead = 1.0,
            clock = () -> (sleep(0.04); 0.0),
        ),
        start_studio = recorder -> begin
            pause!(recorder)
            app = redirect_stdout(stderr) do
                studio(recorder; port = 0, open = :none, name = "Capillary continuation")
            end
            put!(ready, app)
            app
        end,
    )
catch exception
    isready(ready) || put!(ready, Base.CapturedException(exception, catch_backtrace()))
    rethrow()
end

app = take!(ready)
app isa Base.CapturedException && fetch(task)
try
    println("STUDIO_URL ", url(app))
    flush(stdout)
    for command in eachline(stdin)
        strip(command) == "quit" && break
    end
finally
    try
        state = run_state(app.recorder)
        if state in (:created, :running, :pause_requested, :paused, :stopping)
            try
                request_stop!(app.recorder)
            catch exception
                exception isa ArgumentError && run_state(app.recorder) in (:completed, :failed) || rethrow()
            end
        end
        timedwait(() -> istaskdone(task), 60.0) === :ok || error("Initial simulation did not stop")
        fetch(task)
        timedwait(() -> run_state(app.recorder) in (:completed, :failed), 60.0) === :ok ||
            error("Continued simulation did not stop")
    finally
        close(app)
    end
end
