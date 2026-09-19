using Test, TwoPhaseAMR

@testset "Standalone LGPI initial condition" begin
    example = load_example(:lgpi)
    initial = call_example(example, :lgpi_initial_condition)
    @test initial.domain == (4.0, 2.4)
    @test initial.interface_levelset(3.1, 1.52) == 0.0
    @test initial.bathymetry_levelset(3.6, 0.0) ≈ 0.0 atol = 8eps()
end
