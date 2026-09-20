using Test, TwoPhaseAMR

@testset "Standalone Stokes-3 initial condition" begin
    example = load_example(:stokes3_wave)
    initial = call_example(example, :stokes3_initial_condition)
    @test initial.levelset(0.0, initial.elevation(0.0)) ≈ 0.0 atol = eps()
    @test initial.liquid_velocity(0.0, 0.49, 0.0, 1) == 0.0
end
