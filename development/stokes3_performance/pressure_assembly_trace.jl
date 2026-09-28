# Attribute *first* assembly on a moved interface / new AMR topology, not just
# repeated frozen assemblies. The predictor mirrors Stokes3's force-free step
# wrapper, then the production projection is used unchanged. The staged
# assembly is an extra read-only assembly; do not call this a timing benchmark
# of the uninstrumented advance. Compare its mesh/step history explicitly.
include(joinpath(@__DIR__, "pressure_initial_guess.jl"))

function trace_assembly(; final_time = 0.08)
    model = TwoVelocityModel(pressure_projection = CMDGFMProjection())
    run_example(velocity_model = model, final_time = 0.03)
    flow = run_example(velocity_model = model, final_time = 0.0).flow
    adaptation = M._navier_stokes_amr_workspace(flow.velocity, flow.pressure, flow.solver)
    totals = Dict{Symbol,Tuple{Float64,Int}}()
    current = 0.0
    selected = Inf
    post_amr = false
    while current < final_time - 16eps(final_time)
        M._enforce_current_velocity_boundary!(flow.velocity, flow.solver, current)
        selected = min(selected, stable_timestep(flow; cfl = 0.45))
        dt = min(selected, final_time - current)
        state = pressure_predictor!(flow, dt)
        samples = decompose_cached_cmdgfm(flow; dt)
        for sample in samples
            time, bytes = get(totals, sample.label, (0.0, 0))
            totals[sample.label] = (time + sample.seconds, bytes + sample.bytes)
            @printf("ASSEMBLY_TRACE step=%d post_amr=%s stage=%s seconds=%.8g bytes=%d compile=%.6f\n",
                flow.steps + 1, post_amr, sample.label, sample.seconds, sample.bytes,
                sample.compile)
        end
        stats = M._project_two_velocity!(flow.velocity, flow.pressure, flow.solver,
            flow.workspace, state, dt; pressure_prepared = true,
            tolerance = sqrt(eps()), max_iterations = 1000)
        stats.converged || error("unconverged projection at t=$current")
        current = min(final_time, current + dt)
        flow.time = current
        flow.steps += 1
        flow.solver.mesh.adaptation.completed_steps += 1
        amr = adapt!(adaptation, flow.solver.refinement_indicators;
            step = flow.steps, time = current)
        post_amr = amr.changed
        if post_amr
            flow.mesh_updates += 1
            M._rebuild_topology_dependent_state!(flow.solver, flow.workspace; dt)
        end
        record_diagnostics!(flow)
        flush(stdout)
    end
    @printf("ASSEMBLY_HISTORY time=%.8g steps=%d amr=%d faces=%d\n",
        flow.time, flow.steps, flow.mesh_updates, length(flow.solver.mesh.faces))
    for (label, (time, bytes)) in sort!(collect(totals); by = pair -> last(pair)[1], rev = true)
        @printf("ASSEMBLY_TOTAL stage=%s seconds=%.8g bytes=%d\n", label, time, bytes)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    trace_assembly(final_time = parse(Float64, get(ARGS, 1, "0.08")))
end
