"""
    simulate!(velocity, pressure, solver, final_time, workspace;
              time=0, dt=nothing, cfl=0.45,
              diagnostics=NoDiagnostics(), step_offset=0,
              visualization=NoVisualization(),
              tolerance=sqrt(eps(T)), max_cycles=nothing)

Advance to `final_time`. If `dt` is omitted, a stable step is
chosen initially and retained while it remains below the current bound, which
also reuses the shifted viscous hierarchy and bottom factorization. Consecutive
automatic calls using the same fields, solver, workspace and topology retain
the unmodified proposal, not the last step adjusted to `final_time`. If the
next step would leave less than `1e-6` of the proposal, it is stretched to
reach the target instead of leaving a tiny final step.
Formulations that retain
inherited divergence use the larger of the proposal, the previous actual
step and the current step as the reference; the first step uses its proposal.
The previous step is
retained across fixed and automatic calls when the field and solver identities,
time and mesh topology still match. A standalone `step!` or `project!(flow)`
invalidates that history and the automatic proposal.
After a regrid, the transfer may create a new defect. Its reference is the
current unshortened proposal, also used for the two-velocity regrid projection,
until the next physical step establishes a new actual step.
A chunk boundary between physical steps nevertheless introduces a shortened
step and can change the trajectory. If the initial bound is infinite, the
step is capped at one 128th of the requested simulation interval, ensuring
at least 128 steps before the target time. Returns the final time, number of
steps, latest physical-step projection statistics, the number of topology changes, and
`unresolved_curvature_faces`, the sum of unresolved
surface-tension faces across completed steps in this call. The latest step's
count is available as `workspace.unresolved_curvature_faces[]`. When the solver
has refinement indicators, they are evaluated after a successful physical step.
After a topology change, two-velocity flows project the transferred velocities
onto the new mixture incompressibility constraint before the next step; the
transferred physical pressure is retained. A supplied
visualization receives the initial state, every completed step, and the final
state. Each call initializes the visualization; reusing a `VisualizationSession`
starts new session-local checkpoint and diagnostic histories. During
initialization it is also given a lazy independent-step procedure. A
visualization may use that procedure to advance copied state without replacing
or subdividing a step on the actual simulation path; where applicable, it uses
that physical step's reference for the inherited defect.
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
        visualization::AbstractVisualization = NoVisualization(),
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
    independent_step = _visualization_step_procedure(
        visualization, solver, force, options, workspace
    )
    amr_workspace = isempty(solver.refinement_indicators) ? nothing :
        _navier_stokes_amr_workspace(
            velocity, pressure, solver
        )
    isnothing(amr_workspace) || _supports_dynamic_topology(visualization) ||
        throw(
        ArgumentError(
            "exact visualization checkpoints are not supported while dynamic mesh adaptation is enabled"
        )
    )
    _record_diagnostics!(
        diagnostics,
        SimulationState(
            velocity, pressure, solver, current_time, initial_step, workspace,
        ); event = :initial,
    )
    initialize_visualization!(
        visualization; time = current_time, step = initial_step,
        final_time = target_time, velocity, pressure, solver,
        step_procedure = independent_step
    )
    try
        _yield_visualization!(visualization)
        while target_time - current_time > time_tolerance
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
            if !isnothing(amr_workspace)
                amr_stats = adapt!(
                    amr_workspace, solver.refinement_indicators;
                    step = solver.mesh.adaptation.completed_steps,
                    time = current_time
                )
                if amr_stats.changed
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
            _record_diagnostics!(
                diagnostics,
                SimulationState(
                    velocity, pressure, solver, current_time, reported_step,
                    workspace,
                ); event = :step,
            )
            update_visualization!(
                visualization; event = :step, time = current_time,
                step = reported_step, final_time = target_time,
                velocity, pressure, solver
            )
            _yield_visualization!(visualization)
        end
        current_time = target_time
        reported_step = initial_step + step_count
        _record_diagnostics!(
            diagnostics,
            SimulationState(
                velocity, pressure, solver, current_time, reported_step,
                workspace,
            ); event = :completed,
        )
        update_visualization!(
            visualization; event = :completed, time = current_time,
            step = reported_step, final_time = current_time,
            velocity, pressure, solver
        )
        _yield_visualization!(visualization)
    catch exception
        try
            update_visualization!(
                visualization; event = :failed, time = current_time,
                step = initial_step + step_count, final_time = target_time,
                velocity, pressure, solver, exception
            )
        catch
            # Preserve the simulation or visualization error that stopped the run.
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
        mesh_updates, unresolved_curvature_faces,
    )
end
