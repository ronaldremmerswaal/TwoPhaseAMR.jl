using Test, TwoPhaseAMR

@testset "Taylor–Culick example: Studio and velocity models" begin
    example = load_example(:taylor_culick)
    for model in (TwoVelocityModel(), OneVelocityModel())
        run = call_example(
            example, :run_example;
            max_level = 1, final_time = 1.0e-6, velocity_model = model,
            start_studio = example_start_probe,
        )
        @test isopen(run.app)
        close(run.app)
        @test run.flow.velocity isa
            (model isa OneVelocityModel ? FaceVelocity : TwoPhaseVelocity)
        @test run.result.steps == 1
        @test run.result.pressure.converged
        @test first(diagnostic_history(run.flow, :tip_position).values) ≈ 24 atol = 0.05
    end
end

@testset "Taylor–Culick retraction at four cells per thickness" begin
    example = load_example(:taylor_culick)
    run = call_example(example, :run_example; max_level = 2, final_time = 2.0)
    flow = run.flow
    history(name) = diagnostic_history(flow, name).values
    @test run.result.time ≈ 2.0
    @test run.result.pressure.converged
    @test any(block -> block.key.level == 2, flow.system.mesh.blocks)
    # No sheet surface lies on a cell face, so every face with a surface
    # tension jump has a curvature.
    @test run.result.total_unresolved_curvature_faces == 0
    # The two flat surfaces pull the liquid with 2σ = 1 in the example's
    # units, so its momentum is -t from rest; four cells across the sheet
    # give 0.956 of it at t = 2.
    @test 0.93 < -last(history(:liquid_momentum)) / 2 < 0.98
    # Measured: tip 22.786 after a retraction from 24, largest thickness 1.903.
    @test 22.7 < last(history(:tip_position)) < 22.87
    @test 1.85 < last(history(:edge_thickness)) < 1.95
    # The released surface energy feeds the kinetic energy and dissipation.
    released = first(history(:surface_energy)) - last(history(:surface_energy))
    @test 0 < last(history(:kinetic_energy)) < released
    test_finite_example_fields(flow)
    test_phase_volume_conservation(flow)
end
