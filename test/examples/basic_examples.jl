using Test, TwoPhaseAMR

@testset "Standalone single-phase examples" begin
    for name in (:taylor_green, :lid_driven_cavity, :decaying_turbulence)
        example = load_example(name)
        run = call_example(
            example, :run_example; max_level = 0, final_time = 1.0e-6,
        )
        @test run.flow.velocity isa FaceVelocity
        @test run.result.time == 1.0e-6
        @test run.result.steps == 1
        @test run.result.pressure.converged
        @test :kinetic_energy in diagnostic_names(run.flow)
    end
end

@testset "Standalone VOF transport example" begin
    example = load_example(:vortex_reverse)
    run = call_example(
        example, :run_example; max_level = 0, final_time = 1.0e-3,
    )
    @test run.result.time ≈ 1.0e-3
    @test run.result.steps == 1
    @test run.result.volume_error <= 128eps()
end

@testset "Manufactured Poisson example" begin
    example = load_example(:manufactured_poisson)
    for method in (:commuting, :cmdgfm)
        run = call_example(
            example, :run_example; resolution = 8, method,
        )
        @test run.method == method
        @test run.stats.converged
        @test isnothing(run.visualization)
        @test all(isfinite, values(run.errors.pressure))
        @test all(isfinite, values(run.errors.gradient))
        @test all(isfinite, values(run.errors.xi.x))
        @test all(isfinite, values(run.errors.xi.y))
        @test run.errors.pressure.l2 <= run.errors.pressure.linf
        @test run.errors.gradient.l2 <= run.errors.gradient.linf
        @test run.errors.xi.x.l2 <= run.errors.xi.x.linf
        @test run.errors.xi.y.l2 <= run.errors.xi.y.linf
    end
end
