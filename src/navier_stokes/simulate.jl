"""
    simulate!(velocity, pressure, solver, final_time, workspace;
              time=0, dt=nothing, cfl=0.45,
              diagnostics=NoDiagnostics(), step_offset=0,
              observer=NoObserver(),
              tolerance=sqrt(eps(T)), max_cycles=nothing)

Advance to `final_time`. If `dt` is omitted, a stable step is
chosen initially and retained while it remains below the current bound, which
also reuses the shifted viscous hierarchy and bottom factorization. Consecutive
automatic calls using the same fields, solver, workspace and topology retain
the unmodified proposal, not the last step adjusted to `final_time`. If the
next step would leave less than `1e-6` of the proposal, it is stretched to
reach the target instead of leaving a tiny final step. Formulations that retain
inherited divergence use the largest of the proposal, the previous actual step,
and the current step as the reference; the first step uses its proposal. The
previous step is retained across fixed and automatic calls when the field and
solver identities, time and mesh topology still match. A standalone `step!` or
`project!(flow)` invalidates that history and the automatic proposal. After a
regrid, the transfer may create a new defect. Its reference is the current
unshortened proposal, also used for the two-velocity regrid projection, until
the next physical step establishes a new actual step.
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
        solver::IncompressibleNavierStokesSolver{T}, final_time,
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
    target_time >= current_time ||
        throw(ArgumentError("final_time must not precede the initial time"))
    dimensionless_tolerance = _dimensionless_tolerance(T, tolerance)
    cycle_limit = _navier_stokes_cycle_limit(solver, max_cycles)
    options = (;
        tolerance = dimensionless_tolerance,
        max_cycles = cycle_limit,
    )
    step_count = 0
    mesh_updates = 0
    unresolved_curvature_faces = 0
    stats = velocity isa FaceVelocity ?
        _one_velocity_pressure_solver!(solver, workspace).stats :
        solver.pressure_solver.stats
    selection = workspace.simulation_step
    same_run = selection.velocity === velocity && selection.pressure === pressure &&
        selection.solver === solver && selection.time == current_time &&
        selection.topology_generation == solver.mesh.topology_cache.generation
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
        observer, solver, force, options, workspace
    )
    amr_workspace = isempty(solver.refinement_indicators) ? nothing :
        _navier_stokes_amr_workspace(
            velocity, pressure, solver
        )
    isnothing(amr_workspace) || _supports_dynamic_topology(observer) ||
        throw(
        ArgumentError(
            "exact keyframes are not supported while dynamic mesh adaptation is enabled"
        )
    )
    initial_state = SimulationState(
        velocity, pressure, solver, current_time, initial_step, workspace,
    )
    _record_diagnostics!(diagnostics, initial_state; event = :initial)
    observe!(
        observer, :initial, initial_state,
        StepInfo(;
            dt = zero(T), projection = stats, final_time = target_time,
            step_procedure = independent_step,
        ),
    )
    stopped = _observer_stop_requested(observer)
    try
        while !stopped && target_time - current_time > time_tolerance
            step_started = time_ns()
            transport_geometry_prepared = false
            if isnothing(dt)
                # Initializers and AMR transfer operate on intrinsic fields;
                # impose the configured boundary values before querying the
                # transport CFL, just as `step!` does before advection.
                if velocity isa FaceVelocity
                    _enforce_current_velocity_boundary!(
                        velocity, solver, current_time; preserve_open = true,
                    )
                else
                    _enforce_current_velocity_boundary!(velocity, solver, current_time)
                end
                bound = stable_timestep(
                    velocity, solver, workspace; cfl
                )
                selected_step = min(selected_step, bound)
                if step_count == 0 && !isfinite(bound) && !isfinite(selected_step)
                    selected_step = (target_time - current_time) / T(128)
                end
                if velocity isa TwoPhaseVelocity
                    # The frozen-interface CFL query has ensured the
                    # time-step-start reconstruction and frozen its
                    # apertures. The immediately following step may use this
                    # preparation; AMR after the step will build a new state.
                    transport_geometry_prepared =
                        solver.two_phase.interface_transport isa
                        FrozenInterfaceFluxes
                end
            end
            remaining = target_time - current_time
            proposed = isfinite(selected_step) ? selected_step : remaining
            step_size = remaining - proposed < T(1.0e-6) * proposed ?
                remaining : proposed
            reference_step = max(proposed, defect_step, step_size)
            step_size > zero(T) ||
                throw(
                ArgumentError(
                    "time-step selection produced a non-positive step"
                )
            )

            _prepare_independent_step!(
                independent_step, solver, workspace, reference_step,
            )
            stats = step!(
                velocity, pressure, solver, step_size, workspace;
                time = current_time, force, options...,
                _transport_geometry_prepared = transport_geometry_prepared,
                _reference_dt = reference_step,
            )
            stats.converged ||
                throw(
                ErrorException(
                    _solver_failure_message(
                        "pressure projection", stats,
                        dimensionless_tolerance;
                        details = "dt=$(step_size), time=$(current_time), " *
                            "max_cycles=$(cycle_limit)",
                    )
                )
            )
            last_step = T(step_size)
            defect_step = last_step
            current_time = step_size == remaining ? target_time :
                min(target_time, current_time + step_size)
            step_count += 1
            unresolved_curvature_faces += workspace.unresolved_curvature_faces[]
            solver.mesh.adaptation.completed_steps += 1
            topology_changed = false
            if !isnothing(amr_workspace)
                amr_stats = adapt!(
                    amr_workspace, solver.refinement_indicators;
                    step = solver.mesh.adaptation.completed_steps,
                    time = current_time
                )
                if amr_stats.changed
                    topology_changed = true
                    mesh_updates += 1
                    # Seed the replacement viscous hierarchy with the current
                    # shift. Fixed-step runs therefore do not build it twice
                    # after every topology change.
                    _rebuild_topology_dependent_state!(
                        solver, workspace; dt = step_size
                    )
                    # Transfer creates a fresh defect with no intrinsic time
                    # scale. Associate it with the current untruncated
                    # proposal, which is also the legacy two-velocity
                    # regrid-projection gauge; do not carry an older defect's
                    # reference across the topology change.
                    defect_step = T(proposed)
                    if velocity isa TwoPhaseVelocity
                        _project_regridded_two_velocity!(
                            velocity, solver, workspace, proposed, current_time;
                            tolerance = dimensionless_tolerance,
                            max_iterations = cycle_limit,
                        )
                    end
                end
            end
            reported_step = initial_step + step_count
            step_state = SimulationState(
                velocity, pressure, solver, current_time, reported_step,
                workspace,
            )
            _record_diagnostics!(diagnostics, step_state; event = :step)
            _observe_safely!(
                observer, :step, step_state,
                StepInfo(;
                    dt = step_size, projection = stats, topology_changed,
                    final_time = target_time,
                    wall_time = (time_ns() - step_started) / 1.0e9,
                ),
            )
            if _observer_stop_requested(observer)
                stopped = true
                break
            end
        end
        stopped || (current_time = target_time)
        final_state = SimulationState(
            velocity, pressure, solver, current_time,
            initial_step + step_count, workspace,
        )
        _record_diagnostics!(diagnostics, final_state; event = :completed)
        _observe_safely!(
            observer, :completed, final_state,
            StepInfo(; dt = zero(T), projection = stats, final_time = target_time),
        )
    catch exception
        try
            failure = exception isa Exception ? exception :
                ErrorException(sprint(showerror, exception))
            observe!(
                observer, :failed,
                SimulationState(
                    velocity, pressure, solver, current_time,
                    initial_step + step_count, workspace,
                ),
                StepInfo(;
                    dt = zero(T), projection = stats, final_time = target_time,
                    exception = failure,
                ),
            )
        catch observer_exception
            observer_exception isa InterruptException && rethrow()
            # Preserve the original simulation error for ordinary observer failures.
        end
        rethrow()
    end
    if step_count > 0
        selection.topology_generation = solver.mesh.topology_cache.generation
        selection.time = current_time
        selection.step = isnothing(dt) ? selected_step : T(Inf)
        selection.last_dt = last_step
        selection.defect_dt = defect_step
        selection.velocity = velocity
        selection.pressure = pressure
        selection.solver = solver
    elseif isnothing(dt) && isfinite(selected_step)
        # Preserve the old no-op proposal behavior while requiring a previous
        # physical step before retaining its defect time scale.
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
        mesh_updates, unresolved_curvature_faces, stopped,
    )
end
