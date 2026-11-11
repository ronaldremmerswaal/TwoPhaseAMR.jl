"""
    simulate!(velocity, pressure, solver, final_time, workspace;
              time=0, dt=nothing, cfl=0.45, force=nothing,
              diagnostics=NoDiagnostics(), step_offset=0,
              observer=NoObserver(),
              tolerance=sqrt(eps(T)), max_cycles=nothing)

Advance to `final_time`. If `dt` is omitted, a stable step is
chosen initially and retained while it remains below the current bound, which
also reuses the shifted viscous hierarchy and bottom factorization. The `cfl`
keyword must lie in `(0, 1]` for one phase and `(0, 1/2]` for two phases; each
step is checked against the same bound at the top of that range, so every step
the bound returns is accepted. Consecutive
automatic calls using the same fields, solver, workspace and topology retain
the unmodified proposal, not the last step adjusted to `final_time`. If the
next step would leave less than `1e-6` of the proposal, it is stretched to
reach the target instead of leaving a tiny final step. Formulations that retain
inherited divergence use the largest of the proposal, the previous actual step,
and the current step as the reference; the first step uses its proposal. The
previous step is retained across fixed and automatic calls when the field and
solver identities, time and mesh topology still match. A field-level `step!`
or `project!(flow)` invalidates that history and the automatic proposal. A
full `step!(flow, dt)` retains the defect reference but resets the automatic
proposal. After a regrid, the transfer may create a new defect. Its reference
is the current unshortened proposal, also used for the two-velocity regrid
projection, until the next physical step establishes a new actual step.
A chunk boundary between physical steps nevertheless introduces a shortened
step and can change the trajectory. If the initial bound is infinite, the step
is capped at one 128th of the requested simulation interval, ensuring at least
128 steps before the target time. Returns the final time, number of steps,
latest physical-step projection statistics, the number of topology changes,
`unresolved_curvature_faces`, the sum of unresolved surface-tension faces
across completed steps in this call, and `stopped`, which is `true` when the
observer ended the run early. The latest step's unresolved-curvature count is
available as `workspace.unresolved_curvature_faces[]`. When the solver has
refinement indicators, they are evaluated after a successful physical step.
After a topology change, two-velocity flows project the transferred velocities
onto the new mixture incompressibility constraint before the next step; the
transferred physical pressure is retained.
For one-velocity flows, the returned pressure statistics include `compatible`,
which is independent of linear-solve convergence. An incompatible pressure
projection warns once per workspace, including across topology rebuilds and
successive calls using that workspace.

The `observer` (an [`AbstractObserver`](@ref), such as a [`Recorder`](@ref))
receives the initial state, every completed step, and the final state through
[`observe!`](@ref); diagnostics are recorded before each event. At `:initial`
the [`StepInfo`](@ref) carries an independent-step procedure that advances
copied state without replacing or subdividing a step on the actual simulation
path. Where applicable, the independent step uses that physical step's
reference for the inherited defect. After every step `simulate!` polls
[`stop_requested`](@ref); when it returns `true`, the run ends at that step
boundary.
"""
function simulate!(
        velocity::Union{FaceVelocity, TwoPhaseVelocity},
        pressure::CellField{T},
        solver::NavierStokesSystem{T}, final_time,
        workspace::NavierStokesWorkspace{T};
        time = zero(T), dt = nothing, cfl = 0.45,
        force = nothing,
        diagnostics::Union{NoDiagnostics, DiagnosticCollection} = NoDiagnostics(),
        step_offset = 0,
        observer::AbstractObserver = NoObserver(),
        tolerance = sqrt(eps(T)), max_cycles = nothing
    ) where {T}
    target_time = _finite_time(T, final_time, "final_time")
    current_time = _finite_time(T, time)
    initial_step = _nonnegative_count(step_offset, "step_offset")
    flow = _simulation_flow(
        velocity, pressure, solver, workspace, current_time, initial_step,
        diagnostics, force,
    )
    return _simulate!(
        flow, target_time; dt, cfl, force, observer, tolerance, max_cycles,
    )
end

function _simulate!(
        flow::FlowSimulation{T}, final_time;
        dt = flow.default_dt, cfl = 0.45, force = flow.force,
        observer::AbstractObserver = NoObserver(),
        tolerance = sqrt(eps(T)), max_cycles = nothing,
    ) where {T}
    velocity, pressure, solver, workspace =
        flow.velocity, flow.pressure, flow.system, flow.workspace
    target_time = _finite_time(T, final_time, "final_time")
    current_time = _finite_time(T, flow.time)
    initial_step = _nonnegative_count(flow.steps, "step_offset")
    initial_mesh_updates = flow.mesh_updates
    initial_unresolved = flow.unresolved_curvature_faces
    target_time >= current_time ||
        throw(ArgumentError("final_time must not precede the initial time"))
    _check_cfl_range(cfl, solver.two_phase)
    dimensionless_tolerance = _dimensionless_tolerance(T, tolerance)
    cycle_limit = _navier_stokes_cycle_limit(solver, max_cycles)
    options = (; tolerance = dimensionless_tolerance, max_cycles = cycle_limit)
    step_count = 0
    stats = velocity isa FaceVelocity ?
        _one_velocity_pressure_solver!(solver, workspace).stats :
        solver.pressure_solver.stats
    stats_ref = observer isa NoObserver ? nothing : Ref{Any}(stats)
    selection = workspace.simulation_step
    same_run = _same_simulation_step(flow)
    previous_matches = same_run && selection.last_dt > zero(T)
    defect_step = previous_matches ? selection.defect_dt : zero(T)
    last_step = previous_matches ? selection.last_dt : zero(T)
    selected_step = if isnothing(dt)
        same_run ? selection.step : T(Inf)
    else
        _positive_finite_step(T, dt)
    end
    time_tolerance = T(16) * eps(T) *
        max(one(T), abs(target_time), abs(current_time))
    independent_step = _observer_step_procedure(
        observer, solver, force, options, workspace,
    )
    amr_workspace = _flow_amr_workspace!(flow)
    isnothing(amr_workspace) || _supports_dynamic_topology(observer) ||
        throw(
        ArgumentError(
            "exact keyframes are not supported while dynamic mesh adaptation is enabled"
        )
    )
    initial_state = SimulationState(
        velocity, pressure, solver, current_time, initial_step, workspace,
    )
    _record_diagnostics!(flow.diagnostics, initial_state; event = :initial)
    observe!(
        observer, :initial, initial_state,
        StepInfo(;
            dt = zero(T), projection = stats, final_time = target_time,
            step_procedure = independent_step,
        ),
    )
    stopped = _observer_stop_requested(observer)
    next_time = current_time
    try
        while !stopped && target_time - current_time > time_tolerance
            step_started = time_ns()
            workspace.transport_geometry_prepared = false
            if isnothing(dt)
                # Boundary values must be imposed before the transport CFL.
                if velocity isa FaceVelocity
                    _enforce_current_velocity_boundary!(
                        velocity, solver, current_time; preserve_open = true,
                    )
                else
                    _enforce_current_velocity_boundary!(velocity, solver, current_time)
                end
                bound = stable_timestep(velocity, solver, workspace; cfl)
                selected_step = min(selected_step, bound)
                if step_count == 0 && !isfinite(bound) && !isfinite(selected_step)
                    selected_step = (target_time - current_time) / T(128)
                end
                if velocity isa TwoPhaseVelocity
                    # Reuse the frozen apertures prepared by this CFL query.
                    workspace.transport_geometry_prepared =
                        solver.two_phase.interface_transport isa FrozenInterfaceFluxes
                end
            end
            remaining = target_time - current_time
            proposed = isfinite(selected_step) ? selected_step : remaining
            step_size = remaining - proposed < T(1.0e-6) * proposed ?
                remaining : proposed
            reference_step = max(proposed, defect_step, step_size)
            step_size > zero(T) ||
                throw(ArgumentError("time-step selection produced a non-positive step"))
            next_time = step_size == remaining ? target_time :
                min(target_time, current_time + step_size)
            stats = step!(
                flow, step_size; force, observer, options...,
                _proposal = proposed, _reference_dt = reference_step,
                _current_time = current_time, _next_time = next_time,
                _final_time = target_time, _independent_step = independent_step,
                _step_started = step_started, _stats_ref = stats_ref,
                _defer_selection_save = true,
            )
            last_step = T(step_size)
            defect_step = selection.defect_dt
            current_time = next_time
            step_count += 1
            if _observer_stop_requested(observer)
                stopped = true
                break
            end
        end
        stopped || (current_time = target_time)
        flow.time = current_time
        final_state = SimulationState(
            velocity, pressure, solver, current_time, initial_step + step_count, workspace,
        )
        _record_diagnostics!(flow.diagnostics, final_state; event = :completed)
        _observe_safely!(
            observer, :completed, final_state,
            StepInfo(; dt = zero(T), projection = stats, final_time = target_time),
        )
    catch exception
        try
            failure = exception isa Exception ? exception :
                ErrorException(sprint(showerror, exception))
            failure_time = flow.steps == initial_step + step_count ? current_time : next_time
            failure_stats = isnothing(stats_ref) ? stats : stats_ref[]
            observe!(
                observer, :failed,
                SimulationState(
                    velocity, pressure, solver, failure_time, flow.steps, workspace,
                ),
                StepInfo(;
                    dt = zero(T), projection = failure_stats, final_time = target_time,
                    exception = failure,
                ),
            )
        catch observer_exception
            observer_exception isa InterruptException && rethrow()
            # Preserve the original simulation error for ordinary observer failures.
        end
        rethrow()
    finally
        workspace.transport_geometry_prepared = false
    end
    if step_count > 0
        _save_simulation_step!(
            flow, isnothing(dt) ? selected_step : T(Inf), last_step, defect_step,
        )
    elseif isnothing(dt) && isfinite(selected_step)
        # Preserve a no-op proposal without inventing an earlier defect.
        selection.time = current_time
        selection.step = selected_step
        if !previous_matches
            selection.last_dt = zero(T)
            selection.defect_dt = zero(T)
        end
        selection.velocity = velocity
        selection.pressure = pressure
        selection.solver = solver
    end
    return (
        time = current_time, steps = step_count, pressure = stats,
        mesh_updates = flow.mesh_updates - initial_mesh_updates,
        unresolved_curvature_faces = flow.unresolved_curvature_faces - initial_unresolved,
        stopped,
    )
end
