@assert startswith(Base.active_project(), normpath(joinpath(@__DIR__, "..", "..")))

using TwoPhaseAMR
using TwoPhaseAMRStudio

# A real, small flow with a deterministic initial pause. The marker follows
# the public regrid example; delaying its activation puts the regrid among the
# captured steps rather than at the very start of the timeline.
mesh = BlockTreeMesh(
    block_size = (4, 4), max_level = 1,
    adaptation = MeshAdaptationOptions(coarsen_delay = 0),
)
marker = CellField(mesh, Float64)
initialize!(marker, (x, y) -> 1.0 + 0.2 * sinpi(4x) * sinpi(4y))
criterion = InterpolationErrorIndicator(
    marker; E_refine = 1.0e-3, E_coarsen = 1.0e-4, t_init = 0.105,
)
flow = FlowSimulation(
    mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.05);
    boundary = FreeSlip(), dt = 0.01, diagnostics = (KineticEnergy(),),
    velocity = (x, y, t, axis) -> axis == 1 ? sinpi(2y) : 0.0,
    adaptivity = Adaptivity(criteria = criterion),
)
fields = VisualizationFieldRegistry()
register_field!(fields, :pressure, flow.pressure)
register_field!(fields, :velocity_x, flow.velocity, Face{1}())
recorder = Recorder(
    fields; diagnostics = flow, keyframes = EveryNSteps(5), preview = EveryStep(),
    retention = Retention(preview = :all), max_capture_overhead = 1.0,
    clock = () -> 0.0,
)
# Give the browser time to issue controls between completed solver steps; this
# delay is in the physical force callback, not in the Studio server. The
# finite horizon keeps the timeline markers separated at screenshot time.
last_midpoint = Ref(-Inf)
force_lock = ReentrantLock()
function slow_force(x, y, midpoint, axis)
    Base.@lock force_lock begin
        if midpoint > last_midpoint[]
            last_midpoint[] = midpoint
            sleep(0.2)
        end
    end
    return 0.0
end

pause!(recorder)
task = Threads.@spawn simulate!(flow, 1.0; observer = recorder, force = slow_force)
app = nothing
try
    status = timedwait(() -> run_state(recorder) === :paused || istaskdone(task), 120.0)
    status === :ok || error("Simulation did not reach its initial pause")
    istaskdone(task) && fetch(task) # Propagate an early simulation error.
    run_state(recorder) === :paused || error("Simulation ended before its initial pause")
    global app = studio(recorder; port = 0, open = :none, name = "Adaptive flow")
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
