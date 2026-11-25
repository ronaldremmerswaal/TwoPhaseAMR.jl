using Test, TwoPhaseAMR

@testset "Rising bubble: setup, models and Studio hook" begin
    example = load_example(:rising_bubble)
    hysing_parameters = Base.invokelatest(getproperty, example, :hysing_parameters)
    parameters = Base.invokelatest(hysing_parameters, 2)
    @test parameters.density == (1000.0, 1.0)
    @test_throws ArgumentError Base.invokelatest(hysing_parameters, 3)
    for model in (OneVelocityModel(), TwoVelocityModel())
        run = call_example(
            example, :run_example;
            case = 2, max_level = 0, final_time = 1.0e-6, velocity_model = model,
            start_studio = example_start_probe,
        )
        @test isopen(run.app)
        close(run.app)
        @test run.flow.velocity isa
            (model isa OneVelocityModel ? FaceVelocity : TwoPhaseVelocity)
        @test run.result.steps == 1
        @test run.result.pressure.converged
        @test diagnostic_names(run.flow) ==
            [:phase_volume, :circularity, :centroid, :rise_velocity]
        @test all(isfinite, diagnostic_history(run.flow, :rise_velocity).values)
    end
end

@testset "Rising bubble: early rise of case 1" begin
    example = load_example(:rising_bubble)
    for model in (OneVelocityModel(), TwoVelocityModel())
        # 16 cells per diameter at the interface, the far field at level 0.
        run = call_example(
            example, :run_example;
            case = 1, max_level = 2, final_time = 0.5, velocity_model = model,
        )
        @test run.result.time ≈ 0.5
        @test run.result.pressure.converged
        test_finite_example_fields(run.flow)
        test_phase_volume_conservation(run.flow)
        circularity = diagnostic_history(run.flow, :circularity)
        centroid = diagnostic_history(run.flow, :centroid)
        velocity = diagnostic_history(run.flow, :rise_velocity)
        @test first(centroid.values) ≈ 0.5 atol = 1.0e-12
        @test first(velocity.values) == 0
        # TP2D's finest grid gives V_c = 0.1963 and y_c - 0.5 = 0.0546 at
        # t = 0.5 (Hysing et al. 2009). At 16 cells per diameter one velocity
        # gives +1.0 % and -0.3 %, two velocities +1.7 % and +0.8 %.
        rise = last(centroid.values) - first(centroid.values)
        @test last(velocity.values) ≈ 0.1963 rtol = 0.03
        @test rise ≈ 0.0546 rtol = 0.02
        # The bubble stays round; at this resolution the PLIC perimeter is up
        # to 2.8 % short where the circle touches grid lines (c(0) = 1.028).
        @test all(c -> 0.98 < c < 1.03, circularity.values)
        # The centroid moves with the mean bubble velocity: its rise equals
        # the time integral of V_c (measured: within 1.1 %, halving with h).
        times, values = velocity.times, velocity.values
        integral = sum(
            (times[k + 1] - times[k]) * (values[k + 1] + values[k]) / 2
                for k in 1:(length(times) - 1)
        )
        @test rise ≈ integral rtol = 0.02
    end
end
