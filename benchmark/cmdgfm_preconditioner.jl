using BenchmarkTools
using Statistics: median
using TwoPhaseAMR

include(joinpath(@__DIR__, "..", "examples", "two_velocity_instabilities.jl"))

const CMDGFM_SAMPLES = parse(
    Int, get(ENV, "CMDGFM_PRECONDITIONER_SAMPLES", "10"),
)

function copy_velocity(case)
    return map(case.flow.velocity) do source
        destination = FaceVelocity(case.mesh)
        copyto!(destination.interior, source.interior)
        copyto!(destination.boundary, source.boundary)
        destination
    end
end

function projection_sample!(case, solver, workspace)
    stats = project!(
        copy_velocity(case), CellField(case.mesh), solver, 0.01, workspace;
        tolerance = 1.0e-7, max_iterations = 3000,
    )
    stats.converged || error("CMDGFM benchmark projection failed: $stats")
    return stats
end

function benchmark_preconditioner(; mixed_level)
    case = two_velocity_instability_setup(;
        weber = 0.5, bond = 0.0, cells_per_wavelength = 32,
        mixed_level,
    )
    density = case.dispersion.density
    cached_solver = CMDGFMProjectionSolver(case.interface; density)
    cached_workspace = CMDGFMProjectionWorkspace(cached_solver)
    projection_sample!(case, cached_solver, cached_workspace)

    refreshed_solver = CMDGFMProjectionSolver(
        case.interface; density,
        preconditioner_options = (;
            refresh_pressure_coefficients = true,
            reuse_jump_factorization = false,
        ),
    )
    refreshed_workspace = CMDGFMProjectionWorkspace(refreshed_solver)
    projection_sample!(case, refreshed_solver, refreshed_workspace)

    cold = @benchmark projection_sample!(
        $case, $cached_solver, CMDGFMProjectionWorkspace($cached_solver),
    ) samples = CMDGFM_SAMPLES evals = 1
    cached = @benchmark projection_sample!(
        $case, $cached_solver, $cached_workspace,
    ) samples = CMDGFM_SAMPLES evals = 1
    refreshed = @benchmark projection_sample!(
        $case, $refreshed_solver, $refreshed_workspace,
    ) samples = CMDGFM_SAMPLES evals = 1

    stats = projection_sample!(case, cached_solver, cached_workspace)
    return (; cold, cached, refreshed, stats)
end

function print_trial(name, trial)
    estimate = median(trial)
    return println(
        rpad(name, 24),
        lpad(round(estimate.time / 1.0e6; digits = 3), 10), " ms  ",
        lpad(round(estimate.memory / 2.0^20; digits = 3), 10), " MiB  ",
        lpad(estimate.allocs, 12), " allocations",
    )
end

println("CMDGFM block-preconditioner benchmark")
println("Julia ", VERSION, ", threads=", Threads.nthreads())
for mixed_level in (false, true)
    result = benchmark_preconditioner(; mixed_level)
    println("\n", mixed_level ? "fixed mixed-level mesh" : "uniform mesh")
    print_trial("cold block setup + solve", result.cold)
    print_trial("cached block solve", result.cached)
    print_trial("forced numeric refresh", result.refreshed)
    println(
        "iterations=", result.stats.iterations,
        ", pressure_hierarchy_reused=",
        result.stats.pressure_hierarchy_reused,
        ", jump_factor_reused=", result.stats.jump_factor_reused,
        ", active_jumps=", result.stats.jump_unknowns,
    )
end
