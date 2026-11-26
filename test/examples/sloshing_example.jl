using Test, TwoPhaseAMR

@testset "Sloshing linear theory" begin
    example = load_example(:sloshing)
    # Zero thickness aligns the unused baffle face on both plain-tank grids.
    tank = call_example(example, :sloshing_tank; baffle_thickness = 0.0)
    plain = Base.invokelatest(example.plain_tank_modes, tank)
    # Single-fluid limit of the first mode: g k tanh(k h) with k = π; the
    # air layer lowers it by 0.12%.
    @test plain[1].frequency ≈ sqrt(9.81pi * tanh(pi / 2)) rtol = 2.0e-3
    # The bilinear FEM converges to the cosine modes; its higher modes and
    # strip-integrated amplitudes have O(h²) spatial error.
    coarse = Base.invokelatest(
        example.baffled_tank_modes, tank; count = 3, spacing = 0.01,
    )
    fem = Base.invokelatest(
        example.baffled_tank_modes, tank; count = 3, spacing = 0.005,
    )
    for (rough, numerical, exact) in zip(coarse, fem, plain)
        @test numerical.frequency ≈ exact.frequency rtol = 1.0e-3
        @test abs(numerical.frequency - exact.frequency) <
            0.4 * abs(rough.frequency - exact.frequency)
        @test numerical.probe * numerical.forcing ≈
            exact.probe * exact.forcing rtol = 1.0e-3
        @test abs(numerical.probe * numerical.forcing - exact.probe * exact.forcing) <
            0.4 * abs(rough.probe * rough.forcing - exact.probe * exact.forcing)
    end
    @test fem[1].probe * fem[1].initial ≈ plain[1].probe * plain[1].initial rtol = 3.0e-5
    @test abs(fem[1].probe * fem[1].initial - plain[1].probe * plain[1].initial) <
        0.4 * abs(coarse[1].probe * coarse[1].initial - plain[1].probe * plain[1].initial)
    baffled = call_example(example, :sloshing_tank; baffle_height = 0.375)
    fem = Base.invokelatest(
        example.baffled_tank_modes, baffled; count = 1, spacing = 0.005,
    )
    # A baffle reaching three quarters of the depth lowers the first mode by
    # about a fifth (Yu, Xue & Zhu 2020, Fig. 8, read as 0.81).
    @test 0.79 < fem[1].frequency / plain[1].frequency < 0.83
end

@testset "Forced sloshing follows linear theory" begin
    example = load_example(:sloshing)
    # The default near-resonant forcing from the impulsive start; over 1.5 s
    # the wall elevation grows to about 7 mm and stays linear.
    run = call_example(example, :run_example; max_level = 1, final_time = 1.5)
    @test run.result.time ≈ 1.5
    @test run.result.pressure.converged
    @test any(block -> block.key.level == 1, run.flow.system.mesh.blocks)
    test_finite_example_fields(run.flow)
    test_phase_volume_conservation(run.flow)
    left = diagnostic_history(run.flow, :left_elevation)
    right = diagnostic_history(run.flow, :right_elevation)
    linear = diagnostic_history(run.flow, :linear_elevation)
    @test maximum(abs, linear.values) > 0.006
    # Coarse level-1 resolution and finite-amplitude effects allow sub-mm error.
    @test maximum(abs.(left.values .- linear.values)) < 1.0e-3
    # The first mode is antisymmetric up to finite-amplitude effects.
    @test maximum(abs.(left.values .+ right.values)) < 1.0e-3
end

@testset "Baffled free sloshing" begin
    example = load_example(:sloshing)
    run = call_example(
        example, :run_example;
        max_level = 1, baffle_height = 0.375, forcing_amplitude = 0.0,
        initial_amplitude = 0.01, final_time = 3.0,
    )
    @test run.result.pressure.converged
    test_finite_example_fields(run.flow)
    test_phase_volume_conservation(run.flow)
    # Exact polygon clipping: the open area is that of the tank minus the
    # 1 cm × 0.375 m baffle, whatever the cut cells.
    volume = diagnostic_history(run.flow, :phase_volume)
    @test sum(first(volume.values)) ≈ 1 - 0.01 * 0.375 rtol = 1.0e-12
    left = diagnostic_history(run.flow, :left_elevation)
    right = diagnostic_history(run.flow, :right_elevation)
    η = (left.values .- right.values) ./ 2
    crossings = [
        left.times[i - 1] - η[i - 1] * (left.times[i] - left.times[i - 1]) / (η[i] - η[i - 1])
            for i in 2:length(η) if (η[i - 1] < 0 <= η[i]) || (η[i - 1] > 0 >= η[i])
    ]
    frequency = pi * (length(crossings) - 2) / (crossings[end] - crossings[2])
    # The baffled first mode is 4.315 rad/s (finite elements); the plain tank's
    # is 5.310 rad/s.
    @test abs(frequency / 4.315 - 1) < 0.05
end

@testset "Sloshing example with Studio and both velocity models" begin
    example = load_example(:sloshing)
    for model in (OneVelocityModel(), TwoVelocityModel(pressure_projection = CMDGFMProjection()))
        run = call_example(
            example, :run_example;
            max_level = 0, final_time = 1.0e-3, baffle_height = 0.375,
            velocity_model = model, start_studio = example_start_probe,
        )
        @test isopen(run.app)
        close(run.app)
        @test run.flow.velocity isa (model isa OneVelocityModel ? FaceVelocity : TwoPhaseVelocity)
        @test run.result.time == 1.0e-3
        @test run.result.pressure.converged
        test_finite_example_fields(run.flow)
    end
end
