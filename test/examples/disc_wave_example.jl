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

    # The README's spatial study at 32 cells: t = 1 in 2000 steps.
    options = (; resolution = 32, final_time = 1.0, steps = 2000, tolerance = 1.0e-12, keyframes = nothing)
    lsstag = call_example(example, :run_example; options...)
    wall = call_example(example, :run_example; options..., closure = WallDistanceClosure())
    for run in (lsstag, wall)
        # Each solve changes the energy by at most about its relative tolerance.
        @test maximum(abs, run.energy ./ first(run.energy) .- 1) <= run.steps * 1.0e-12
    end
    # Second order from 256 cells back to 32: each relative error is at most
    # the measured 256-cell value of the README times (256/32)^2. Measured at
    # 32 cells: 2.028e-2, 3.026e-2, 0.1725, 0.2719 and frequency errors
    # 6.00e-4, 5.565e-3, 2.240e-3, 1.687e-2.
    bound = (256 / 32)^2
    for (name, finest) in (
            (:displacement_l2, 3.424e-4), (:displacement_max, 5.275e-4),
            (:rate_l2, 2.803e-3), (:rate_max, 4.423e-3),
        )
        @test relative(lsstag.errors, name) <= finest * bound
    end
    for (entry, finest) in zip(lsstag.frequencies, (1.005e-5, 9.034e-5, 3.874e-5, 2.744e-4))
        @test abs(entry.spatial / entry.exact - 1) <= finest * bound
    end
    # The legacy wall-distance closure is first order at the curved wall.
    @test relative(wall.errors, :displacement_l2) > 3 * relative(lsstag.errors, :displacement_l2)
    @test abs(wall.frequencies[1].spatial / wall.frequencies[1].exact - 1) >
        10 * abs(lsstag.frequencies[1].spatial / lsstag.frequencies[1].exact - 1)

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
    @test lsstag.initial_errors.displacement_max > 0
end
