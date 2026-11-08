using Test, TwoPhaseAMR

@testset "Standalone Stokes-3 initial condition" begin
    example = load_example(:stokes3_wave)
    initial = call_example(example, :stokes3_initial_condition)
    @test initial.levelset(0.0, initial.elevation(0.0)) ≈ 0.0 atol = eps()
    @test initial.liquid_velocity(0.0, 0.49, 0.0, 1) == 0.0
end

@testset "Stokes-wave refinement window" begin
    example = load_example(:stokes3_wave)
    # A flat, zero-surface-tension wave has no curvature or vorticity forcing.
    # Once the interface indicator's t_end=0.01 window expires, its initially
    # refined mesh can coarsen during this eight-step run.
    run = call_example(
        example, :run_example;
        max_level = 1, steepness = 0.0, bond = Inf,
        final_time = 0.016, dt = 0.002, visualize = false,
    )
    @test run.result.time ≈ 0.016
    @test run.result.steps == 8
    @test run.result.mesh_updates > 0
    @test run.result.pressure.converged
    test_finite_example_fields(run.flow)
    test_phase_volume_conservation(run.flow)
end
