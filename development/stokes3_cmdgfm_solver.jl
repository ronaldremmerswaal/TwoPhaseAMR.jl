using TwoPhaseAMR
using Printf
using LinearAlgebra
using SparseArrays

include(joinpath(@__DIR__, "..", "examples", "stokes3_wave.jl"))

# Run one physical step per simulate! call so the projection statistics remain
# available even when a later step fails. Retain the minimum CFL step, as the
# regular simulate! driver does, and let it perform the usual dynamic AMR.
function benchmark(; mode = :default, final_time = 0.5, max_level = 5)
    projection = if mode == :default
        CMDGFMProjection()
    elseif mode == :fresh
        CMDGFMProjection(preconditioner_options = (
            refresh_pressure_coefficients = true,
            reuse_jump_factorization = false,
        ))
    elseif mode == :pressure_refresh
        CMDGFMProjection(preconditioner_options = (
            refresh_pressure_coefficients = true,
        ))
    elseif mode == :stale_pressure
        CMDGFMProjection(preconditioner_options = (
            refresh_pressure_coefficients = false,
        ))
    elseif mode == :ilu
        CMDGFMProjection(preconditioner = :ilu)
    elseif mode == :none
        CMDGFMProjection(preconditioner = nothing)
    else
        error("unknown solver mode: $mode")
    end
    setup = setup_stokes3(;
        velocity_model = TwoVelocityModel(pressure_projection = projection),
        final_time, max_level,
    )
    flow = setup.flow
    @printf("mode=%s max_level=%d final_time=%.3f initial_iterations=%d\n",
        mode, max_level, final_time, setup.initial_projection.iterations)
    flush(stdout)
    selected_step = Inf
    total_wall = 0.0
    while flow.time < final_time - 16eps(Float64)
        selected_step = min(selected_step, stable_timestep(flow))
        step = min(selected_step, final_time - flow.time)
        before = flow.time
        result = nothing
        wall = @elapsed try
            result = simulate!(flow, before + step; dt = step)
        catch error
            state = flow.workspace.two_phase.two_velocity[]
            work = state.projection_workspace
            @printf("FAILED at t=%.9f dt=%.9g step=%d completed_wall=%.3f\n",
                before, step, flow.steps + 1, total_wall)
            @printf("system: n=%d nnz=%d jump_n=%d jump_nnz=%d\n",
                size(work.matrix, 1), nnz(work.matrix),
                size(work.jump_block, 1), nnz(work.jump_block))
            flush(stdout)
            rethrow(error)
        end
        total_wall += wall
        stats = result.pressure
        if flow.steps <= 10 || flow.steps % 10 == 0 ||
                stats.iterations >= 100 || flow.time >= final_time - 16eps(Float64)
            @printf("step=%d t=%.9f dt=%.9g wall=%.3f it=%d p=%9.2e j=%9.2e np=%d nx=%d unresolved=%d reused=(%s,%s)\n",
                flow.steps, flow.time, step, wall, stats.iterations,
                stats.mixture_volume_defect, stats.normal_velocity_defect,
                stats.pressure_unknowns, stats.jump_unknowns,
                stats.underresolved_jump_rows,
                stats.pressure_hierarchy_reused, stats.jump_factor_reused)
            flush(stdout)
        end
    end
    @printf("DONE mode=%s time=%.9f steps=%d updates=%d wall=%.3f\n",
        mode, flow.time, flow.steps, flow.mesh_updates, total_wall)
    return flow
end

if abspath(PROGRAM_FILE) == @__FILE__
    benchmark(;
        mode = Symbol(get(ARGS, 1, "default")),
        final_time = parse(Float64, get(ARGS, 2, "0.5")),
        max_level = parse(Int, get(ARGS, 3, "5")),
    )
end
