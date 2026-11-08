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
    pressure_drag = diagnostic_history(run.flow, :pressure_drag)
    # The example suppresses its impulsive-start force spike with a NaN at
    # the initial sample; require at least one finite completed-step sample.
    @test length(pressure_drag) >= 2
    @test first(pressure_drag.times) == 0
    @test isnan(first(pressure_drag.values))
    @test all(>(0), Iterators.drop(pressure_drag.steps, 1))
    @test all(isfinite, Iterators.drop(pressure_drag.values, 1))
end

@testset "Adaptive cylinder example" begin
    example = load_example(:cylinder_flow)
    # The example disables dynamic adaptation at level 1; level 2 enables
    # its geometry and vorticity criteria.
    run = call_example(
        example, :run_example;
        reynolds = 100, max_level = 2, final_time = 8.0e-4,
        dt = 1.0e-4, visualize = false,
    )
    @test run.result.time ≈ 8.0e-4
    @test run.result.steps == 8
    @test run.result.pressure.converged
    mesh = run.flow.solver.mesh
    @test any(block -> block.key.level == 2, mesh.blocks)
    @test mesh.adaptation.last_update_step > 2
    test_finite_example_fields(run.flow)
end
