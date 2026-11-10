# Check 02: the run sequence of simulate! -- sec:overview:run (set-up, step size, step, regrid, events, end), the observer events
# :initial / :step / :completed / :failed and the stop polling.
#
# Label:  sec:overview:run
# Source: docs/techref/chapters/01_overview.tex:478-546; src/navier_stokes/simulate.jl, src/observation/observer.jl,
#         src/simulation/diagnostics/history.jl (_diagnostic_due!), src/navier_stokes/flow_simulation.jl
# Method: INSTRUMENTED RUN (see step01): a recording observer plus the call log of the stage functions of simulate!; each statement of the
#         chapter's list is checked on a run that exhibits it.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/ch-overview/scripts/step02_run_sequence.jl

include(joinpath(@__DIR__, "_fixtures.jl"))

mutable struct Obs <: A.AbstractObserver
    events::Vector{Any}
    stop_after::Int            # stop_requested returns true from poll number stop_after + 1 on
    polls::Int
    throw_at::Union{Nothing, Symbol}
    throw_value::Any
    poll_throws::Bool
    dynamic_ok::Bool
end
Obs(; stop_after = typemax(Int), throw_at = nothing, throw_value = ErrorException("observer failure"), poll_throws = false, dynamic_ok = true) =
    Obs(Any[], stop_after, 0, throw_at, throw_value, poll_throws, dynamic_ok)

function A.observe!(o::Obs, event::Symbol, state, info)
    Tracer.note!(:observe!, "script", (:event,), (event,))
    push!(
        o.events, (
            event = event, step = state.step, time = state.time, dt = info.dt, topology = info.topology_changed, wall = info.wall_time,
            exception = info.exception, has_procedure = !isnothing(info.step_procedure), final_time = info.final_time,
        ),
    )
    o.throw_at === event && throw(o.throw_value)
    return nothing
end
function A.stop_requested(o::Obs)
    Tracer.note!(:stop_requested, "script", (), ())
    o.polls += 1
    o.poll_throws && error("poll failure")
    return o.polls > o.stop_after
end
A._supports_dynamic_topology(o::Obs) = o.dynamic_ok

evnames(o) = Symbol[e.event for e in o.events]
approx(a, b; atol = 1.0e-13) = isapprox(a, b; atol)
quiet(f) = with_logger(f, NullLogger())

dt = 1.0e-3

println("== A. events of an ordinary run with a shortened last step")
flow = single_flow(diagnostics = (A.KineticEnergy(),))
obs = Obs()
res, log = Tracer.record() do
    A.simulate!(flow, 0.0025; dt, observer = obs)
end
check("events: :initial, :step x3, :completed", evnames(obs) == [:initial, :step, :step, :step, :completed], string(evnames(obs)))
check("step counters 0,1,2,3,3 and times 0, dt, 2dt, 0.0025, 0.0025", [e.step for e in obs.events] == [0, 1, 2, 3, 3] && all(approx.([e.time for e in obs.events], [0, 1.0e-3, 2.0e-3, 2.5e-3, 2.5e-3])))
check("StepInfo.dt: 0 at :initial, the step size at :step (last step shortened), 0 at :completed", all(approx.([e.dt for e in obs.events], [0, 1.0e-3, 1.0e-3, 5.0e-4, 0])))
check("wall time > 0 at :step; topology_changed false; independent-step procedure only at :initial", all(e -> e.event !== :step || e.wall > 0, obs.events) && all(e -> !e.topology, obs.events) && [e.has_procedure for e in obs.events] == [true, false, false, false, false])
check("the run carries final_time in every event", all(e -> approx(e.final_time, 0.0025), obs.events))
check("result: time, steps, stopped = false", approx(res.time, 0.0025) && res.steps == 3 && res.stopped === false)
# order of calls: stop poll, then per step: step!, diagnostics(:step), observer(:step), poll; end: diagnostics(:completed), observer(:completed)
seqn = Symbol[]
for e in log
    if e.name === :_record_diagnostics!
        push!(seqn, Symbol("diag_", e.args.event))
    elseif e.name === :observe!
        push!(seqn, Symbol("observe_", e.args.event))
    elseif e.name === :_observe_safely!
        push!(seqn, Symbol("safely_", e.args.event))
    elseif e.name in (:_observer_stop_requested, :step!, :adapt!)
        push!(seqn, e.name)
    end
end
# observe!(:initial) is called directly, :step and :completed through _observe_safely! (which calls observe!, logged as observe_*)
expected = [
    :diag_initial, :observe_initial, :_observer_stop_requested,
    :step!, :diag_step, :safely_step, :observe_step, :_observer_stop_requested,
    :step!, :diag_step, :safely_step, :observe_step, :_observer_stop_requested,
    :step!, :diag_step, :safely_step, :observe_step, :_observer_stop_requested,
    :diag_completed, :safely_completed, :observe_completed,
]
check("call order: diagnostics before each observer event; stop polled before the first step and after every step", seqn == expected, join(seqn, " "))
hist = A.diagnostic_history(flow.diagnostics, :kinetic_energy)
check("diagnostics recorded at :initial and after each step; :completed at the same step adds no duplicate", hist.steps == [0, 1, 2, 3], string(hist.steps))

println("\n== B. stop polling")
flow = single_flow()
obs = Obs(stop_after = 2)
res, log = Tracer.record() do
    A.simulate!(flow, 0.01; dt, observer = obs)
end
check(
    "stop after step 2: events :initial, :step, :step, :completed; stopped = true; time = 2 dt (not the final time) in the :completed state",
    evnames(obs) == [:initial, :step, :step, :completed] && res.stopped && res.steps == 2 && approx(res.time, 2.0e-3) && approx(obs.events[end].time, 2.0e-3),
)
check("polls: once before the first step and once after each step", obs.polls == 3)
flow = single_flow()
obs = Obs(stop_after = 0)
res, log = Tracer.record() do
    A.simulate!(flow, 0.01; dt, observer = obs)
end
check("stop requested before the first step: no step, events :initial, :completed, stopped = true", evnames(obs) == [:initial, :completed] && res.steps == 0 && res.stopped && !(:step! in names_of(log)))
flow = single_flow()
obs = Obs(poll_throws = true)
res = quiet(() -> A.simulate!(flow, 0.002; dt, observer = obs))
check("an exception while polling is logged and treated as 'continue'", res.steps == 2 && !res.stopped)

println("\n== C. observer exceptions")
flow = single_flow()
obs = Obs(throw_at = :initial)
err = try
    A.simulate!(flow, 0.002; dt, observer = obs); nothing
catch e
    e
end
check("exception at :initial: propagates, ends the call before the first step, no :failed event", err isa ErrorException && evnames(obs) == [:initial] && flow.steps == 0)
flow = single_flow()
obs = Obs(throw_at = :step)
res = quiet(() -> A.simulate!(flow, 0.002; dt, observer = obs))
check("exception at :step: logged, the run continues to :completed", res.steps == 2 && evnames(obs) == [:initial, :step, :step, :completed])
flow = single_flow()
obs = Obs(throw_at = :completed)
res = quiet(() -> A.simulate!(flow, 0.002; dt, observer = obs))
check("exception at :completed: logged, not rethrown", res.steps == 2 && evnames(obs)[end] === :completed)
flow = single_flow()
obs = Obs(throw_at = :step, throw_value = InterruptException())
err = try
    A.simulate!(flow, 0.002; dt, observer = obs); nothing
catch e
    e
end
check("InterruptException at :step propagates, the observer receives :failed first", err isa InterruptException && evnames(obs) == [:initial, :step, :failed] && obs.events[end].exception isa InterruptException)

println("\n== D. :failed")
flow = single_flow()
obs = Obs()
err = try
    A.simulate!(flow, 1.0; dt = 10.0, observer = obs); nothing
catch e
    e
end
check(
    "a step above the bounds is not retried: ArgumentError rethrown, events :initial, :failed with the exception, step counter 0",
    err isa ArgumentError && occursin("exceeds", err.msg) && evnames(obs) == [:initial, :failed] && obs.events[end].exception === err && obs.events[end].step == 0,
    err isa Exception ? first(split(sprint(showerror, err), '\n')) : "no error",
)
flow = drop_flow(
    viscosity = (0.0, 0.0), sigma = 0.0, gravity = (0.0, 0.0),
    velocity = (x, y, t, axis) -> axis == 1 ? 0.2 * sinpi(x) : 0.0,
)
obs = Obs()
err = try
    A.simulate!(flow, 0.002; dt, observer = obs, max_cycles = 0, tolerance = 1.0e-14)
    nothing
catch e
    e
end
check(
    "an unconverged projection stops the run with an ErrorException (:failed delivered, step not counted)",
    err isa ErrorException && occursin("pressure projection", err.msg) && evnames(obs)[end] === :failed && obs.events[end].step == 0,
    err isa Exception ? first(split(sprint(showerror, err), '\n')) : "no error (projection converged with max_cycles = 0)",
)
flow = single_flow()
obs = Obs()
err = try
    A.simulate!(flow, 0.002; dt, observer = obs, max_cycles = 0, tolerance = 1.0e-14)
    nothing
catch e
    e
end
check(
    "an unconverged viscous solve also stops the run with an ErrorException (before the projection; the chapter names only the projection)",
    err isa ErrorException && occursin("diffusion solve", err.msg) && evnames(obs)[end] === :failed,
    err isa Exception ? first(split(sprint(showerror, err), '\n')) : "no error",
)
flow = single_flow()
obs = Obs(throw_at = :failed)
check(
    "an error of the observer at :failed does not replace the original error", (
        try
            A.simulate!(flow, 1.0; dt = 10.0, observer = obs); false
        catch e
            e isa ArgumentError
        end
    )
)

println("\n== E. step size: automatic proposal, cfl, never increased, last step")
flow = single_flow()
A._enforce_current_velocity_boundary!(flow.velocity, flow.system, 0.0; preserve_open = true)
bound45 = A.stable_timestep(flow; cfl = 0.45)
obs = Obs()
A.simulate!(flow, 10 * bound45; observer = obs)
check("automatic first step = stable_timestep at CFL number 0.45 (default)", approx(obs.events[2].dt, bound45; atol = 1.0e-15), "$(obs.events[2].dt) vs $bound45")
flow = single_flow()
A._enforce_current_velocity_boundary!(flow.velocity, flow.system, 0.0; preserve_open = true)
b30 = A.stable_timestep(flow; cfl = 0.3)
obs = Obs()
A.simulate!(flow, 10 * b30; cfl = 0.3, observer = obs)
check("cfl keyword scales the proposal (0.3/0.45)", approx(obs.events[2].dt / bound45, 0.3 / 0.45; atol = 1.0e-12))
# a flow whose bound grows: strong viscous decay of the velocity
flow = single_flow(viscosity = 5.0)
obs = Obs()
A.simulate!(flow, 40 * bound45; observer = obs, tolerance = 1.0e-10)
dts = [e.dt for e in obs.events if e.event === :step]
later_bound = A.stable_timestep(flow; cfl = 0.45)
check("the step is never increased within a run although the bound grows (except the stretched/shortened last step)", all(diff(dts[1:(end - 1)]) .<= 1.0e-15) && later_bound > 1.01 * dts[1], "first $(dts[1]), last bound $later_bound")
# stretched last step: remaining - proposal < 1e-6 proposal -> one step of the remaining size
flow = single_flow(); obs = Obs()
A.simulate!(flow, 3 * dt + 5.0e-10; dt, observer = obs)
dts = [e.dt for e in obs.events if e.event === :step]
check("a remainder below 1e-6 of the proposal is absorbed: 3 steps, the last stretched", length(dts) == 3 && approx(dts[3], dt + 5.0e-10; atol = 1.0e-15), string(dts))
flow = single_flow(); obs = Obs()
A.simulate!(flow, 3 * dt + 5.0e-8; dt, observer = obs)
dts = [e.dt for e in obs.events if e.event === :step]
check("a larger remainder gets its own (tiny) step: 4 steps", length(dts) == 4 && approx(dts[4], 5.0e-8; atol = 1.0e-15), string(dts))

println("\n== F. retained history")
flow = single_flow()
A._enforce_current_velocity_boundary!(flow.velocity, flow.system, 0.0; preserve_open = true)
res = A.simulate!(flow, 1.5 * bound45)       # automatic: one full step and one shortened step
sel = flow.workspace.simulation_step
check(
    "after an automatic run: the unmodified proposal and the last actual step are retained",
    isfinite(sel.step) && sel.last_dt < sel.step && approx(sel.last_dt, 0.5 * bound45; atol = 1.0e-9) || (isfinite(sel.step) && sel.last_dt > 0),
    "proposal $(sel.step), last step $(sel.last_dt), defect step $(sel.defect_dt)",
)
A.step!(flow.velocity, flow.pressure, flow.system, 1.0e-4, flow.workspace; time = flow.time)
check("a standalone step! ends the retained history (proposal infinite, no previous run)", isinf(sel.step) && sel.velocity === nothing)
flow = single_flow(); A.simulate!(flow, 1.0e-3; dt = 1.0e-3)
check("a prescribed step retains the last step but no proposal", isinf(flow.workspace.simulation_step.step) && approx(flow.workspace.simulation_step.last_dt, 1.0e-3))
A.project!(flow)
check("project!(flow) ends the retained history and leaves the physical pressure unchanged", flow.workspace.simulation_step.velocity === nothing && isinf(flow.workspace.simulation_step.step))
p0 = copy(flow.pressure); A.project!(flow)
check("project!(flow): pressure unchanged", all(i -> flow.pressure.blocks[i] == p0.blocks[i], eachindex(p0.blocks)))

println("\n== G. regrid (sec:overview:run item 4)")
function regrid_flow(two_velocity; interval = 1)
    mesh = A.BlockTreeMesh(block_size = (4, 4), max_level = 1, adaptation = A.MeshAdaptationOptions(coarsen_delay = 0, update_interval = interval))
    marker = A.CellField(mesh)
    A.initialize!(marker, (x, y) -> 1 + 0.2sinpi(4x) * sinpi(4y))
    criterion = A.InterpolationErrorIndicator(marker; E_refine = 1.0e-3, E_coarsen = 1.0e-4)
    adaptivity = A.Adaptivity(criteria = criterion)
    numerics = A.NavierStokesOptions(multigrid = A.MultigridOptions(coarse_size = 8))
    if two_velocity
        interface = A.InterfaceField(mesh, (x, y) -> x - 0.43; lipschitz = 1)
        return A.FlowSimulation(
            mesh, A.TwoPhase(interface; density = (2.0, 1.0), dynamic_viscosity = (0.0, 0.0));
            velocity = (0.0, 0.0), boundary = A.FreeSlip(), adaptivity, numerics,
            velocity_model = A.TwoVelocityModel(pressure_projection = A.CMDGFMProjection()),
        )
    end
    return A.FlowSimulation(
        mesh, A.SinglePhase(density = 1.0, dynamic_viscosity = 0.01);
        velocity = (x, y, time, axis) -> axis == 1 ? 0.05 + 0.001sinpi(2x) : 0.0, boundary = A.FreeSlip(), adaptivity, numerics,
    )
end
function regrid_case(tv)
    flow = regrid_flow(tv); obs = Obs()
    res, log = Tracer.record() do
        A.simulate!(flow, 0.01; dt = 0.1, observer = obs, tolerance = 1.0e-10, max_cycles = 100)
    end
    nm = Symbol[e.name for e in log if e.name in (:_navier_stokes_amr_workspace, :step!, :adapt!, :_rebuild_topology_dependent_state!, :_project_regridded_two_velocity!, :_record_diagnostics!, :_safely_dummy, :_observe_safely!)]
    expected = tv ?
        [:_navier_stokes_amr_workspace, :_record_diagnostics!, :step!, :adapt!, :_rebuild_topology_dependent_state!, :_project_regridded_two_velocity!, :_record_diagnostics!, :_observe_safely!, :_record_diagnostics!, :_observe_safely!] :
        [:_navier_stokes_amr_workspace, :_record_diagnostics!, :step!, :adapt!, :_rebuild_topology_dependent_state!, :_record_diagnostics!, :_observe_safely!, :_record_diagnostics!, :_observe_safely!]
    label = tv ? "two velocities" : "one velocity"
    check("$label: set-up registers the AMR workspace, step, adapt!, rebuild, [projection of the transferred velocities], diagnostics, :step event", nm == expected, join(nm, " > "))
    ad = only(e for e in log if e.name === :adapt!)
    check("$label: adapt! gets the completed-step count and the time at the end of the step", ad.args.step == 1 && approx(ad.args.time, 0.01))
    check("$label: the :step event reports topology_changed = true and mesh_updates = 1", obs.events[2].topology === true && res.mesh_updates == 1)
    if tv
        pr = only(e for e in log if e.name === :_project_regridded_two_velocity!)
        check("two velocities: the regrid projection uses the unshortened proposal as dt (0.1) at the end-of-step time", approx(pr.args.dt, 0.1) && approx(pr.args.time, 0.01))
    end
    check("the transfer defect is associated with the proposal dtprop = 0.1 (defect_dt)", approx(flow.workspace.simulation_step.defect_dt, 0.1) && approx(flow.workspace.simulation_step.last_dt, 0.01))
    return nothing
end
foreach(regrid_case, (false, true))
# update_interval: adapt! is called after every step but acts at most every update_interval steps
flow = regrid_flow(false; interval = 2); obs = Obs()
_, log = Tracer.record() do
    A.simulate!(flow, 0.004; dt = 1.0e-3, observer = obs, tolerance = 1.0e-10)
end
calls = [e.args.step for e in log if e.name === :adapt!]
last_update = flow.system.mesh.adaptation.last_update_step
check("update_interval = 2: adapt! is called with steps 1,2,3,4 and acts at steps 1 and 3 (last_update_step = 3)", calls == [1, 2, 3, 4] && last_update == 3, "calls $calls, last_update_step $last_update")
flow = regrid_flow(false)
err = try
    A.simulate!(flow, 0.002; dt = 1.0e-3, observer = Obs(dynamic_ok = false)); nothing
catch e
    e
end
check("dynamic adaptation rejects an observer that advances copies of the state (exact keyframes)", err isa ArgumentError && occursin("exact keyframes", err.msg))

finish("step02_run_sequence")
