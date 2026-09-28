# Exact, non-overlapping pressure refresh/projection time and allocation on
# Stokes3's moving interface. This mirrors its staged step (no external body
# force), including AMR and diagnostics. Validate the complete mesh/step
# history against pressure_probe.jl; sampled stage estimates alone cannot
# attribute the large whole-run allocation to the pressure solver.
include(joinpath(@__DIR__, "pressure_probe.jl"))

function trace_pressure_stages!(flow, final_time)
    solver, workspace = flow.solver, flow.workspace
    velocity, pressure, model = flow.velocity, flow.pressure, flow.solver.two_phase
    two = velocity isa TwoPhaseVelocity
    adaptation = M._navier_stokes_amr_workspace(velocity, pressure, solver)
    records = NamedTuple[]
    selected = Inf
    current = flow.time
    tolerance = sqrt(eps(Float64))
    limit = M._navier_stokes_cycle_limit(solver, nothing)
    post_amr = false
    while current < final_time - 16eps(final_time)
        M._enforce_current_velocity_boundary!(velocity, solver, current)
        selected = min(selected, stable_timestep(flow; cfl = 0.45))
        dt = min(selected, final_time - current)
        state = two ? M._two_velocity_flow_workspace!(workspace.two_phase, model) : nothing
        old = two ? state.old_velocity : workspace.old_velocity
        M._copy_velocity!(old, velocity)
        M._enforce_current_velocity_boundary!(old, solver, current)
        convected = M._transport_substep!(velocity, solver, workspace, dt, current, model)
        M._enforce_current_velocity_boundary!(convected, solver, current + dt / 2)
        M._diffusion_substep!(velocity, solver, workspace, dt, current, model;
            tolerance, max_cycles = limit)
        M._enforce_current_velocity_boundary!(velocity, solver, current + dt)
        refresh = @timed M._refresh_pressure_substep!(solver, workspace, model)
        M._phase_force_substep!(velocity, solver, workspace, dt, model)
        projection = @timed if two
            M._project_two_velocity!(velocity, pressure, solver, workspace, state, dt;
                pressure_prepared = true, tolerance, max_iterations = limit)
        else
            project!(velocity, pressure, solver.pressure_solver, dt, workspace.projection;
                velocity_boundary = solver.boundary, boundary_time = current + dt,
                tolerance, max_cycles = limit)
        end
        stats = projection.value
        stats.converged || error("unconverged pressure at t=$current")
        iterations = two ? stats.iterations : stats.cycles
        push!(records, (; post_amr, faces = length(solver.mesh.faces), iterations,
            refresh_seconds = refresh.time, refresh_bytes = refresh.bytes,
            projection_seconds = projection.time, projection_bytes = projection.bytes,
            compilation = refresh.compile_time + projection.compile_time,
            gc = refresh.gctime + projection.gctime,
            initial_guess_used = two && state.projection_workspace.initial_guess_used))
        current = min(final_time, current + dt)
        flow.time = current
        flow.steps += 1
        solver.mesh.adaptation.completed_steps += 1
        amr = adapt!(adaptation, solver.refinement_indicators;
            step = solver.mesh.adaptation.completed_steps, time = current)
        post_amr = amr.changed
        if post_amr
            flow.mesh_updates += 1
            M._rebuild_topology_dependent_state!(solver, workspace; dt)
        end
        record_diagnostics!(flow)
    end
    return records
end

function pressure_stage_report(mode; final_time = 0.5)
    model = mode == :one ? OneVelocityModel() : TwoVelocityModel(
        pressure_projection = CMDGFMProjection())
    run_example(velocity_model = model, final_time = 0.03)
    flow = run_example(velocity_model = model, final_time = 0.0).flow
    advance = @timed trace_pressure_stages!(flow, final_time)
    records = advance.value
    @printf("PRESSURE_HISTORY mode=%s time=%.8g steps=%d amr=%d faces=%d total_seconds=%.6f total_bytes=%d\n",
        mode, flow.time, flow.steps, flow.mesh_updates, length(flow.solver.mesh.faces),
        advance.time, advance.bytes)
    @printf("PRESSURE_TOTAL mode=%s refresh_seconds=%.6f refresh_bytes=%d projection_seconds=%.6f projection_bytes=%d compile_seconds=%.6f gc_seconds=%.6f warm_guesses=%d\n",
        mode, sum(r.refresh_seconds for r in records), sum(r.refresh_bytes for r in records),
        sum(r.projection_seconds for r in records), sum(r.projection_bytes for r in records),
        sum(r.compilation for r in records), sum(r.gc for r in records),
        count(r -> r.initial_guess_used, records))
    for (label, subset) in ((:ordinary, filter(r -> !r.post_amr, records)),
            (:post_amr, filter(r -> r.post_amr, records)))
        isempty(subset) && continue
        @printf("PRESSURE_CONTEXT mode=%s kind=%s calls=%d time_median=%.8g time_p95=%.8g bytes_median=%.8g iterations_median=%.8g iterations_p95=%.8g iterations_max=%d\n",
            mode, label, length(subset), median([r.projection_seconds for r in subset]),
            quantile([r.projection_seconds for r in subset], 0.95),
            median([r.projection_bytes for r in subset]), median([r.iterations for r in subset]),
            quantile([r.iterations for r in subset], 0.95), maximum(r.iterations for r in subset))
    end
    flush(stdout)
    return flow, records
end

if abspath(PROGRAM_FILE) == @__FILE__
    mode = Symbol(get(ARGS, 1, "cmdgfm"))
    mode in (:one, :cmdgfm) || error("mode must be one or cmdgfm")
    pressure_stage_report(mode; final_time = parse(Float64, get(ARGS, 2, "0.5")))
end
