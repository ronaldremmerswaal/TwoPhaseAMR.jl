using Test, TwoPhaseAMR

@testset "Standalone disc wave example" begin
    example = load_example(:disc_wave)
    bessel_zero = Base.invokelatest(getproperty, example, :bessel_zero)
    # Abramowitz and Stegun, table 9.5.
    for (m, k, value) in (
            (0, 1, 2.404825557695773), (1, 1, 3.831705970207512),
            (2, 1, 5.135622301840683), (3, 2, 9.76102312998167),
        )
        @test Base.invokelatest(bessel_zero, m, k) ≈ value rtol = 4eps()
    end
    relative(errors, name) = getproperty(errors, name) / getproperty(errors.norms, name)

    # Short trajectories refined in space and time together (Δt ∝ h): t = 1/4
    # in 20 steps at 32 cells and 40 steps at 64. `regression_study` in
    # make_figures.jl refines this trajectory to 256 cells, where the relative
    # errors converge at second order (orders 1.96–2.03 from 128 to 256). Each
    # error is bounded by C (32/n)² plus 5 %, C the largest measured
    # e(n) (n/32)² over 32–256 cells; measured at 32 cells: 7.002e-3,
    # 1.030e-2, 2.799e-2, 3.017e-2.
    constants = (;
        displacement_l2 = 9.619e-3, displacement_max = 1.3885e-2,
        rate_l2 = 2.7991e-2, rate_max = 3.017e-2,
    )
    short(resolution; options...) = call_example(
        example, :run_example; resolution, final_time = 0.25,
        steps = 20 * resolution ÷ 32, tolerance = 1.0e-12, keyframes = nothing, options...,
    )
    lsstag = (short(32), short(64))
    wall = (short(32; closure = WallDistanceClosure()), short(64; closure = WallDistanceClosure()))
    for run in (lsstag..., wall...)
        # Each solve changes the energy by at most about its relative tolerance.
        @test maximum(abs, run.energy ./ first(run.energy) .- 1) <= run.steps * 1.0e-12
    end
    for name in keys(constants)
        for (run, resolution) in zip(lsstag, (32, 64))
            @test relative(run.errors, name) <= 1.05 * constants[name] * (32 / resolution)^2
        end
        # Measured orders from 32 to 64 cells: 1.70, 1.71, 2.19, 2.21.
        @test log2(relative(lsstag[1].errors, name) / relative(lsstag[2].errors, name)) >= 1.6
    end
    frequency_errors(run) = [abs(entry.spatial / entry.exact - 1) for entry in run.frequencies]
    # Measured orders 1.91, 1.88, 1.91, 2.01.
    @test all(log2.(frequency_errors(lsstag[1]) ./ frequency_errors(lsstag[2])) .>= 1.8)
    # The legacy wall-distance closure is first order at the curved wall:
    # measured order 0.99, and 5.2 times the LS-STAG error at 32 cells.
    wall_errors = map(run -> relative(run.errors, :displacement_l2), wall)
    @test log2(wall_errors[1] / wall_errors[2]) < 1.3
    @test wall_errors[1] > 3 * relative(lsstag[1].errors, :displacement_l2)
    @test frequency_errors(wall[1])[1] > 10 * frequency_errors(lsstag[1])[1]

    # Steps of 6.4 times the explicit limit of the composite mesh, recorded:
    # one keyframe per step over one period of the (0,1) mode. Measured
    # maximum displacement error 2.735e-2.
    large = call_example(example, :run_example; resolution = 32, composite = true, steps = 64)
    @test large.dt > 6 * large.explicit_limit
    @test maximum(abs, large.energy ./ first(large.energy) .- 1) <= large.steps * 1.0e-10
    @test large.errors.displacement_max < 0.03
    @test length(keyframes(snapshot_store(large.recorder))) == 65

    sampled = call_example(
        example, :run_example; resolution = 16, steps = 4, initial_data = :sampled, keyframes = nothing,
    )
    @test sampled.initial_errors.displacement_max == 0
    @test lsstag[1].initial_errors.displacement_max > 0
end
