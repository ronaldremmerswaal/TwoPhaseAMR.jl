active_project = Base.active_project()
studio_project = normpath(joinpath(@__DIR__, "..", "..", "Project.toml"))
@assert !isnothing(active_project) && realpath(active_project) == realpath(studio_project) "Run this fixture with the Studio project active"

using LinearAlgebra
BLAS.set_num_threads(1)
using TwoPhaseAMR
using TwoPhaseAMRStudio

# A real AMR run with CFL-adaptive time steps: the recorded frame times are
# irregular, both before and during the exports.
mesh = BlockTreeMesh(
    block_size = (4, 4), max_level = 1,
    adaptation = MeshAdaptationOptions(coarsen_delay = 0),
)
marker = CellField(mesh, Float64)
initialize!(marker, (x, y) -> 1.0 + 0.2 * sinpi(4x) * sinpi(4y))
criterion = InterpolationErrorIndicator(
    marker; E_refine = 1.0e-3, E_coarsen = 1.0e-4, t_init = 0.04,
)
flow = FlowSimulation(
    mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.05);
    boundary = FreeSlip(), dt = nothing,
    diagnostics = (KineticEnergy(),),
    velocity = (x, y, t, axis) -> axis == 1 ? sinpi(2y) : 0.0,
    adaptivity = Adaptivity(criteria = criterion),
)
fields = VisualizationFieldRegistry()
register_field!(fields, :pressure, flow.pressure)
register_field!(fields, :velocity_x, flow.velocity, Face{1}())
recorder = Recorder(
    fields; diagnostics = flow, keyframes = EveryNSteps(2), preview = EveryStep(),
    retention = Retention(preview = :all), max_capture_overhead = 1.0,
    clock = () -> 0.0,
)

# Keep every physical step slow enough for the browser to act between steps.
const last_midpoint = Ref(-Inf)
const force_lock = ReentrantLock()
function slow_force(x, y, midpoint, axis)
    Base.@lock force_lock begin
        if midpoint > last_midpoint[]
            last_midpoint[] = midpoint
            sleep(0.05)
        end
    end
    return axis == 1 ? 0.2 * cospi(2y) * cospi(midpoint / 2) : 0.0
end

pause!(recorder)
task = Threads.@spawn simulate!(
    flow, 100.0; observer = recorder, force = slow_force, cfl = 0.25,
    tolerance = 1.0e-9, max_cycles = 1000,
)
store = snapshot_store(recorder)
app = nothing
try
    status = timedwait(() -> run_state(recorder) === :paused || istaskdone(task), 120.0)
    status === :ok || error("Simulation did not reach its initial pause")
    istaskdone(task) && fetch(task)
    run_state(recorder) === :paused || error("Simulation ended before its initial pause")
    # Accumulate irregular frames before the browser opens.
    resume!(recorder)
    reached = timedwait(
        () -> istaskdone(task) || (!isempty(frames(store)) && newest_frame(store).time >= 0.12),
        120.0,
    )
    reached === :ok || error("Simulation did not reach the pre-run target")
    pause!(recorder)
    timedwait(() -> run_state(recorder) === :paused || istaskdone(task), 120.0) === :ok ||
        error("Simulation did not pause before the Studio session")
    istaskdone(task) && fetch(task)
    run_state(recorder) === :paused || error("Simulation ended before the Studio session")
    app = redirect_stdout(stderr) do
        studio(recorder; port = 0, open = :none, name = "Irregular AMR")
    end
    println("STUDIO_URL ", url(app))
    flush(stdout)
    for command in eachline(stdin)
        strip(command) == "quit" && break
    end
finally
    try
        if !istaskdone(task) && run_state(recorder) in
                (:created, :running, :pause_requested, :paused, :stopping)
            try
                request_stop!(recorder)
            catch exception
                exception isa ArgumentError && run_state(recorder) in (:completed, :failed) || rethrow()
            end
        end
        timedwait(() -> istaskdone(task), 30.0) === :ok ||
            error("Fixture simulation did not stop")
        fetch(task)
    finally
        isnothing(app) || close(app)
    end
end
