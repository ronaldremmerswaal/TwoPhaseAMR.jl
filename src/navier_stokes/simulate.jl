"""
    simulate!(velocity, pressure, solver, final_time, workspace;
              time=0, dt=nothing, cfl=0.45,
              diagnostics=NoDiagnostics(), step_offset=0,
              observer=NoObserver(),
              tolerance=sqrt(eps(T)), max_cycles=nothing)

Advance to `final_time`. If `dt` is omitted, a stable step is
chosen initially and retained while it remains below the current bound, which
also reuses the shifted viscous hierarchy and bottom factorization. Consecutive
calls using the same fields, solver and workspace retain the unshortened
proposal, not the last step clipped to `final_time`. A chunk boundary between
physical steps nevertheless introduces a shortened step and can change the
trajectory. If the initial bound is infinite, the step is capped at one 128th
of the requested simulation interval, ensuring at least 128 steps before the
target time. Returns the final time, number of steps, latest physical-step
projection statistics, the number of topology changes,
`unresolved_curvature_faces`, the sum of unresolved surface-tension faces
across completed steps in this call, and `stopped`, which is `true` when the
observer ended the run early. The latest step's unresolved-curvature count is
available as `workspace.unresolved_curvature_faces[]`. When the solver has
refinement indicators, they are evaluated after a successful physical step.
After a topology change, two-velocity flows project the transferred velocities
onto the new mixture incompressibility constraint before the next step; the
transferred physical pressure is retained.

The `observer` (an [`AbstractObserver`](@ref), such as a [`Recorder`](@ref))
receives the initial state, every completed step, and the final state through
[`observe!`](@ref); diagnostics are recorded before each event. At `:initial`
the [`StepInfo`](@ref) carries an independent-step procedure that advances
copied state without replacing or subdividing a step on the actual simulation
path. After every step `simulate!` polls [`stop_requested`](@ref); when it
returns `true`, the run ends at that step boundary.
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
    stats = solver.pressure_solver.stats
    selection = workspace.simulation_step
    selected_step = if isnothing(dt)
        selection.velocity === velocity && selection.pressure === pressure &&
            selection.solver === solver && selection.time == current_time ?
            selection.step : T(Inf)
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
    stopped = stop_requested(observer)
    try
        while !stopped && target_time - current_time > time_tolerance
            step_started = time_ns()
            transport_geometry_prepared = false
            if isnothing(dt)
                # Initializers and AMR transfer operate on intrinsic fields;
                # impose the configured boundary values before querying the
                # transport CFL, just as `step!` does before advection.
                _enforce_current_velocity_boundary!(
                    velocity, solver, current_time,
                )
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
            proposed = isfinite(selected_step) ? selected_step :
                target_time - current_time
            step_size = min(proposed, target_time - current_time)
            step_size > zero(T) ||
                throw(
                ArgumentError(
                    "time-step selection produced a non-positive step"
                )
            )

            _prepare_independent_step!(
                independent_step, solver, workspace
            )
            stats = step!(
                velocity, pressure, solver, step_size, workspace;
                time = current_time, force, options...,
                _transport_geometry_prepared = transport_geometry_prepared,
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
            current_time = min(target_time, current_time + step_size)
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
                    if velocity isa TwoPhaseVelocity
                        # Use the untruncated step as the defect gauge so a
                        # tiny final step does not weaken the constraint for
                        # a resumed run.
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
            observe!(
                observer, :step, step_state,
                StepInfo(;
                    dt = step_size, projection = stats, topology_changed,
                    final_time = target_time,
                    wall_time = (time_ns() - step_started) / 1.0e9,
                ),
            )
            if stop_requested(observer)
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
        observe!(
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
        catch
            # Preserve the simulation or observer error that stopped the run.
        end
        rethrow()
    end
    if isnothing(dt) && isfinite(selected_step)
        selection.time = current_time
        selection.step = selected_step
        selection.velocity = velocity
        selection.pressure = pressure
        selection.solver = solver
    end
    return (
        time = current_time, steps = step_count, pressure = stats,
        mesh_updates, unresolved_curvature_faces, stopped,
    )
end
