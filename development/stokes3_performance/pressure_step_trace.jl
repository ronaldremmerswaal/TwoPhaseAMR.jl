# Development-only trace of the standard automatic-step Stokes3 loop. This
# mirrors `simulate!` so the projection's returned stats are available at
# every step; compare the step/AMR history with pressure_probe.jl to validate.
# Usage: julia --project=. development/stokes3_performance/pressure_step_trace.jl 0.5
include(joinpath(@__DIR__, "pressure_probe.jl"))

function describe(values)
    isempty(values) && return (NaN, NaN, NaN)
    return (median(values), quantile(values, 0.95), maximum(values))
end

function trace_pressure!(flow, final_time)
    solver = flow.solver
    velocity = flow.velocity
    pressure = flow.pressure
    workspace = flow.workspace
    T = eltype(pressure)
    current = flow.time
    target = T(final_time)
    selected_step = T(Inf)
    tolerance = sqrt(eps(T))
    cycle_limit = M._navier_stokes_cycle_limit(solver, nothing)
    amr_workspace = M._navier_stokes_amr_workspace(
        velocity, pressure, solver)
    records = NamedTuple[]
    previous_amr = false
    while target - current > T(16) * eps(T) *
            max(one(T), abs(target), abs(current))
        M._enforce_current_velocity_boundary!(velocity, solver, current)
        bound = M.stable_timestep(velocity, solver, workspace; cfl = 0.45)
        selected_step = min(selected_step, bound)
        dt = min(selected_step, target - current)
        prepared = velocity isa TwoPhaseVelocity &&
            solver.two_phase.interface_transport isa M.FrozenInterfaceFluxes
        step = @timed M.step!(velocity, pressure, solver, dt, workspace;
            time = current, tolerance, max_cycles = cycle_limit,
            _transport_geometry_prepared = prepared)
        stats = step.value
        stats.converged || error("projection failed at t=$current")
        projection = workspace.two_phase.two_velocity[].projection_workspace
        jump_diagonal = diag(projection.jump_block)
        diagonal_scale = isempty(jump_diagonal) ? one(T) :
            max(one(T), maximum(abs, jump_diagonal))
        tiny_diagonals = count(value ->
            abs(value) <= T(1e-12) * diagonal_scale, jump_diagonal)
        last_krylov = isnothing(projection.krylov) ? 0 :
            M.Krylov.iteration_count(projection.krylov)
        current = min(target, current + dt)
        flow.steps += 1
        solver.mesh.adaptation.completed_steps += 1
        push!(records, (;
            step = flow.steps, time = current, dt, faces = length(solver.mesh.faces),
            post_amr = previous_amr, step_seconds = step.time,
            step_bytes = step.bytes, step_gc = step.gctime,
            iterations = stats.iterations, initial_residual =
                stats.initial_residual, final_residual = stats.final_residual,
            mixture_volume_defect = stats.mixture_volume_defect,
            normal_velocity_defect = stats.normal_velocity_defect,
            momentum_identity_error = stats.momentum_identity_error,
            minimum_jump_diagonal_margin = stats.minimum_jump_diagonal_margin,
            underresolved_jump_rows = stats.underresolved_jump_rows,
            pressure_unknowns = stats.pressure_unknowns,
            jump_unknowns = stats.jump_unknowns,
            tiny_diagonals,
            last_krylov, correction_restarted =
                stats.iterations > last_krylov,
            pressure_reused = stats.pressure_hierarchy_reused,
            jump_reused = stats.jump_factor_reused,
            initial_guess_used = projection.initial_guess_used,
        ))
        amr = M.adapt!(amr_workspace, solver.refinement_indicators;
            step = solver.mesh.adaptation.completed_steps, time = current)
        previous_amr = amr.changed
        if amr.changed
            flow.mesh_updates += 1
            M._rebuild_topology_dependent_state!(solver, workspace; dt)
        end
        M._record_diagnostics!(flow.diagnostics,
            M.SimulationState(velocity, pressure, solver, current,
                flow.steps, workspace); event = :step)
    end
    flow.time = current
    return records
end

function print_trace(records, flow)
    @printf("TRACE_SUMMARY steps=%d amr=%d faces=%d time=%.8f\n",
        flow.steps, flow.mesh_updates, length(flow.solver.mesh.faces),
        flow.time)
    for (name, extract) in (
            (:iterations, r -> r.iterations),
            (:initial_residual, r -> r.initial_residual),
            (:final_residual, r -> r.final_residual),
            (:mixture_volume_defect, r -> r.mixture_volume_defect),
            (:normal_velocity_defect, r -> r.normal_velocity_defect),
            (:momentum_identity_error, r -> r.momentum_identity_error),
            (:minimum_jump_diagonal_margin,
                r -> r.minimum_jump_diagonal_margin),
            (:jump_unknowns, r -> r.jump_unknowns),
            (:tiny_jump_diagonals, r -> r.tiny_diagonals),
            (:step_seconds, r -> r.step_seconds),
            (:step_bytes, r -> r.step_bytes),
        )
        values = [extract(r) for r in records]
        mid, p95, peak = describe(values)
        @printf("TRACE_DIST metric=%s median=%.8g p95=%.8g max=%.8g\n",
            name, mid, p95, peak)
    end
    @printf("TRACE_FLAGS pressure_reused=%d jump_reused=%d initial_guess_used=%d underresolved_steps=%d correction_restarts=%d tiny_diagonal_steps=%d\n",
        count(r -> r.pressure_reused, records),
        count(r -> r.jump_reused, records),
        count(r -> r.initial_guess_used, records),
        count(r -> r.underresolved_jump_rows > 0, records),
        count(r -> r.correction_restarted, records),
        count(r -> r.tiny_diagonals > 0, records))
    for (label, subset) in (
            (:ordinary, filter(r -> !r.post_amr, records[2:end])),
            (:post_amr, filter(r -> r.post_amr, records)),
        )
        isempty(subset) && continue
        mid, p95, peak = describe([r.step_seconds for r in subset])
        bytes, _, _ = describe([r.step_bytes for r in subset])
        @printf("TRACE_CONTEXT name=%s count=%d step_median=%.6f step_p95=%.6f step_max=%.6f median_bytes=%.0f\n",
            label, length(subset), mid, p95, peak, bytes)
    end
    if get(ENV, "PRESSURE_TRACE_ROWS", "0") == "1"
        for r in records
            @printf("TRACE_ROW step=%d time=%.8f dt=%.8g faces=%d iterations=%d final_residual=%.8g mixture_defect=%.8g normal_defect=%.8g margin=%.8g jump=%d underresolved=%d pressure_reused=%s jump_reused=%s\n",
                r.step, r.time, r.dt, r.faces, r.iterations,
                r.final_residual, r.mixture_volume_defect,
                r.normal_velocity_defect,
                r.minimum_jump_diagonal_margin,
                r.jump_unknowns, r.underresolved_jump_rows,
                r.pressure_reused, r.jump_reused)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    target = parse(Float64, get(ARGS, 1, "0.5"))
    model = TwoVelocityModel(pressure_projection = CMDGFMProjection())
    warm = run_example(velocity_model = model, final_time = 0.03)
    @printf("WARM steps=%d\n", warm.result.steps)
    flow = run_example(velocity_model = model, final_time = 0.0).flow
    trace = @timed trace_pressure!(flow, target)
    @printf("TRACE_ADVANCE seconds=%.6f bytes=%d gc=%.6f compile=%.6f\n",
        trace.time, trace.bytes, trace.gctime, trace.compile_time)
    print_trace(trace.value, flow)
end
