using Test, TwoPhaseAMR

@testset "Coalescence example: inertial bridge growth" begin
    example = load_example(:coalescence)
    # A level-4 neck resolves the fillet of an r0 = 0.25 bridge by 2.7 cells
    # (the default r0 = 0.15 needs level 5); the band is coarse.
    bridge = 0.25
    final_time = 0.04
    run = call_example(
        example, :run_example;
        band_level = 2, max_level = 4, bridge, final_time,
        start_studio = example_start_probe,
    )
    @test isopen(run.app)
    close(run.app)
    @test run.result.time ≈ final_time
    @test run.result.pressure.converged
    test_finite_example_fields(run.flow)
    test_phase_volume_conservation(run.flow)
    width = diagnostic_history(run.flow, :bridge_half_width).values
    surface = diagnostic_history(run.flow, :surface_energy).values
    kinetic = diagnostic_history(run.flow, :kinetic_energy).values
    deformation = diagnostic_history(run.flow, :deformation).values
    # The PLIC half-width on x = 0 starts at the bridge.
    @test first(width) ≈ bridge atol = 0.01
    # Inertial growth r_b² = r0² + C²t with C between 1.2 and the
    # asymptotic 1.62 of Duchemin et al. (2003); measured: 0.367.
    @test sqrt(bridge^2 + 1.2^2 * final_time) < last(width) <
        sqrt(bridge^2 + 1.62^2 * final_time)
    # Surface tension shortens the interface and sets the liquid in motion; a
    # resolved fillet leaves no grid-scale gas debris that would lengthen it.
    @test last(surface) < first(surface)
    @test last(kinetic) > 0
    # The far halves of the drops have not moved yet: two touching circles
    # have (I_xx - I_yy)/(I_xx + I_yy) = 2/3.
    @test all(d -> isapprox(d, 2 / 3; atol = 0.01), deformation)
end

@testset "Coalescence example: two-velocity variation" begin
    example = load_example(:coalescence)
    run = call_example(
        example, :run_example;
        band_level = 2, max_level = 4, bridge = 0.25, final_time = 0.01,
        velocity_model = TwoVelocityModel(pressure_projection = CMDGFMProjection()),
    )
    @test run.flow.velocity isa TwoPhaseVelocity
    @test run.result.time ≈ 0.01
    @test run.result.pressure.converged
    test_finite_example_fields(run.flow)
    test_phase_volume_conservation(run.flow)
    width = diagnostic_history(run.flow, :bridge_half_width).values
    @test last(width) > first(width)
end
