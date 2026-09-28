# Pressure-performance changes must not change CMDGFM's solution. This is a
# development check, not another model or a replacement for the integration
# convergence suite. Run from the repository root with --project=.
using TwoPhaseAMR, Test, LinearAlgebra, Printf

module PressurePoissonProblem
include(joinpath(@__DIR__, "..", "..", "examples", "manufactured_poisson.jl"))
end

module PressureWaveProblem
include(joinpath(@__DIR__, "..", "..", "test", "integration",
    "interfacial_instability_problem.jl"))
include(joinpath(@__DIR__, "..", "..", "test", "two_phase", "two_velocity", "helpers.jl"))
end

function poisson_pressure_gate()
    @testset "CMDGFM manufactured pressure: warm versus zero start" begin
        for density in ((1000.0, 1.0), (1.0, 1000.0)), resolution in (8, 16, 32)
            run = PressurePoissonProblem.run_example(; resolution, density,
                method = :cmdgfm, tolerance = 1e-10)
            cold_velocity = (copy(run.predictor[1]), copy(run.predictor[2]))
            cold_pressure = CellField(run.mesh)
            cold_work = CMDGFMProjectionWorkspace(run.solver)
            cold_stats = project!(cold_velocity, cold_pressure, run.solver, 1.0,
                cold_work; tolerance = 1e-10, max_iterations = 5000, warm_start = false)
            cold = PressurePoissonProblem._manufactured_poisson_fields(
                cold_pressure, run.solver, cold_work, density)
            @test run.stats.converged && cold_stats.converged
            @test run.workspace.solution ≈ cold_work.solution rtol = 2e-7 atol = 2e-7
            for key in (:pressure, :gradient), norm_name in (:l2, :linf)
                @test getproperty(getproperty(run.errors, key), norm_name) ≈
                    getproperty(getproperty(cold.errors, key), norm_name) rtol = 2e-7 atol = 2e-7
            end
            @printf("POISSON n=%d rhoL=%g rhoG=%g pressure_l2=%.8g pressure_linf=%.8g gradient_l2=%.8g gradient_linf=%.8g xi_x_l2=%.8g xi_x_linf=%.8g xi_y_l2=%.8g xi_y_linf=%.8g warm_iterations=%d cold_iterations=%d solution_linf_difference=%.8g\n",
                resolution, density..., run.errors.pressure.l2, run.errors.pressure.linf,
                run.errors.gradient.l2, run.errors.gradient.linf,
                run.errors.xi.x.l2, run.errors.xi.x.linf,
                run.errors.xi.y.l2, run.errors.xi.y.linf,
                run.stats.iterations, cold_stats.iterations,
                norm(run.workspace.solution - cold_work.solution, Inf))
            flush(stdout)
        end
    end
end

function wave_pressure_gate(; final_time = 0.2)
    # A bounded KH regression with mesh and step-size changes. The default
    # short endpoint probes solver/NaN robustness, NOT long-time dispersion
    # convergence; set final_time=nothing for the original integration endpoint.
    @testset "CMDGFM KH pressure robustness" begin
        for density_ratio in (1.0, 1000.0), weber in (0.5, 5.0)
            errors = Float64[]
            for (resolution, factor) in ((8, 1.0), (16, 1.0), (16, 0.5), (16, 0.25))
                run = PressureWaveProblem.interfacial_instability(;
                    cells_per_wavelength = resolution, density_ratio, weber,
                    bond = 0.0, liquid_above = false,
                    pressure_projection = CMDGFMProjection(), final_time,
                    timestep_factor = factor, tolerance = 1e-7, max_cycles = 3000)
                error = PressureWaveProblem.normalized_amplitude_error(
                    run.times, run.amplitudes, run.analytical)
                push!(errors, error)
                @test isfinite(error)
                @test run.flow.time == run.final_time
                @test PressureWaveProblem._phase_storage_contract(
                    run.flow.velocity, run.interface)
                @printf("KH_PRESSURE rho=%g We=%g n=%d factor=%g time=%.8g error=%.8g steps=%d\n",
                    density_ratio, weber, resolution, factor, run.final_time, error,
                    run.flow.steps)
                flush(stdout)
            end
            # Spatial error need not monotonically decrease at this short
            # endpoint. Temporal robustness is bounded error, not zero error.
            @test maximum(errors[2:end]) <= 1.5errors[2] + 1e-10
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    poisson_pressure_gate()
    raw_time = get(ENV, "PRESSURE_WAVE_TIME", "0.2")
    wave_pressure_gate(final_time = raw_time == "default" ? nothing : parse(Float64, raw_time))
end
