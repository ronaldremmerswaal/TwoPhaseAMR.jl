@assert startswith(Base.active_project(), normpath(joinpath(@__DIR__, "..", "..")))

using TwoPhaseAMR
using TwoPhaseAMRStudio

# A real, small flow with a deterministic initial pause and enough completed
# steps to exercise timeline navigation while it is still running. The delay is
# in the physical force callback once per step, not in the Studio server.
mesh = BlockTreeMesh(block_size = (4, 4))
flow = FlowSimulation(
    mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.05);
    boundary = FreeSlip(), dt = 0.01, diagnostics = (KineticEnergy(),),
    velocity = (x, y, t, axis) -> axis == 1 ? sinpi(2y) : 0.0,
)
fields = VisualizationFieldRegistry()
register_field!(fields, :pressure, flow.pressure)
register_field!(fields, :velocity_x, flow.velocity, Face{1}())
recorder = Recorder(
    fields; diagnostics = flow, keyframes = EveryNSteps(5), preview = EveryStep(),
    retention = Retention(preview = :all), max_capture_overhead = 1.0,
    clock = () -> 0.0,
)
last_midpoint = Ref(-Inf)
force_lock = ReentrantLock()
function slow_force(x, y, midpoint, axis)
    Base.@lock force_lock begin
        if midpoint > last_midpoint[]
            last_midpoint[] = midpoint
            sleep(0.04)
        end
    end
    return 0.0
end

pause!(recorder)
task = Threads.@spawn simulate!(flow, 100.0; observer = recorder, force = slow_force)
app = nothing
try
    status = timedwait(() -> run_state(recorder) === :paused || istaskdone(task), 120.0)
    status === :ok || error("Simulation did not reach its initial pause")
    istaskdone(task) && fetch(task) # Propagate an early simulation error.
    run_state(recorder) === :paused || error("Simulation ended before its initial pause")
    global app = studio(recorder; port = 0, open = :none)
    println("STUDIO_URL ", url(app))
    flush(stdout)
    # The parent owns lifetime; it closes the browser before sending this line.
    readline(stdin)
finally
    try
        if run_state(recorder) in (:created, :running, :pause_requested, :paused, :stopping)
            request_stop!(recorder)
        end
        timedwait(() -> istaskdone(task), 30.0) === :ok ||
            error("Fixture simulation did not stop")
        fetch(task)
    finally
        isnothing(app) || close(app)
    end
end
