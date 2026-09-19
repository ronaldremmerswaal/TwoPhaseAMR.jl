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
