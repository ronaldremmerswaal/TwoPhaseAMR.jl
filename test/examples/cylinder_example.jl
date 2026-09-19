using Test, TwoPhaseAMR

@testset "Standalone cylinder example" begin
    example = load_example(:cylinder_flow)
    run = call_example(
        example, :run_example;
        reynolds = 100, max_level = 0, final_time = 1.0e-6,
    )
    @test run.result.time == 1.0e-6
    @test run.result.pressure.converged
    @test !isnothing(run.flow.solver.geometry)
    @test :pressure_drag in diagnostic_names(run.flow)
    @test all(isfinite, diagnostic_history(
        run.flow, :pressure_drag,
    ).values)
end
