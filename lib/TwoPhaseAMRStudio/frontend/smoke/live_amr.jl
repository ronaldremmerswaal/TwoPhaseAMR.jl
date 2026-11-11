active_project = Base.active_project()
studio_project = normpath(joinpath(@__DIR__, "..", "..", "Project.toml"))
@assert !isnothing(active_project) && realpath(active_project) == realpath(studio_project) "Run this fixture with the Studio project active"

using LinearAlgebra
BLAS.set_num_threads(1)
using TwoPhaseAMR
using TwoPhaseAMRStudio

# The interface starts on one coarse block. Activating its indicator after
# several recorded steps makes the split happen while the browser is live.
mesh = BlockTreeMesh(
    block_size = (4, 4), max_level = 1, periodic = (true, false),
    adaptation = MeshAdaptationOptions(update_interval = 1, coarsen_delay = 0),
)
interface = InterfaceField(mesh, (x, y) -> y - 0.43; lipschitz = 1)
indicator = InterfaceRefinementIndicator(interface; level = 1, t_init = 0.055)

# The phases have different tangential shear, but share the initial normal
# velocity. The x-dependent perturbation is divergence-free in the continuum
# and has zero normal velocity at the free-slip walls.
liquid_velocity(x, y, t, axis) = axis == 1 ?
    0.12 + 0.04cospi(2y) + 0.01sinpi(2x) * cospi(y) :
    -0.02cospi(2x) * sinpi(y)
gas_velocity(x, y, t, axis) = axis == 1 ?
    0.07 + 0.03cospi(2y) + 0.01sinpi(2x) * cospi(y) :
    -0.02cospi(2x) * sinpi(y)
flow = FlowSimulation(
    mesh,
    TwoPhase(interface; density = (2.0, 1.0), dynamic_viscosity = (0.0, 0.0));
    velocity_model = TwoVelocityModel(
        pressure_projection = CMDGFMProjection(),
    ),
    velocity = (liquid_velocity, gas_velocity), boundary = FreeSlip(), dt = 0.01,
    adaptivity = Adaptivity(criteria = (indicator,)),
    diagnostics = (KineticEnergy(), PhaseVolume()),
)
fields = VisualizationFieldRegistry()
register_default_fields!(fields, flow)
recorder = Recorder(
    fields; diagnostics = flow, keyframes = nothing, preview = EveryStep(),
    retention = Retention(preview = :all), max_capture_overhead = 1.0,
    # Disable overhead-based preview skipping, including during JIT/regridding.
    clock = () -> 0.0,
)

# A physical, time-dependent shear acceleration keeps velocity and vorticity
# evolving. Sleep once per midpoint so the browser can observe live frames.
const last_midpoint = Ref(-Inf)
const force_lock = ReentrantLock()
function slow_force(x, y, midpoint, axis)
    Base.@lock force_lock begin
        if midpoint > last_midpoint[]
            last_midpoint[] = midpoint
            sleep(0.2)
        end
    end
    return axis == 1 ? 0.12cospi(2y) * cospi(midpoint / 2) : 0.0
end

pause!(recorder)
task = Threads.@spawn simulate!(
    flow, 20.0; observer = recorder, force = slow_force,
    tolerance = 1.0e-9, max_cycles = 1000,
)
app = nothing
try
    status = timedwait(() -> run_state(recorder) === :paused || istaskdone(task), 540.0)
    status === :ok || error("Simulation did not reach its initial pause")
    istaskdone(task) && fetch(task)
    run_state(recorder) === :paused || error("Simulation ended before its initial pause")
    # Studio normally prints connection instructions; reserve stdout for the
    # single readiness line consumed by the parent process.
    global app = redirect_stdout(stderr) do
        studio(recorder; port = 0, open = :none, name = "Live two-phase AMR")
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
                # A simulation can complete or fail between the state query
                # and the stop request; its task still reports any failure.
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
