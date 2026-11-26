using Test, TwoPhaseAMR

@testset "Rayleigh–Taylor linear theory" begin
    example = load_example(:rayleigh_taylor)
    rate(k; kwargs...) = Base.invokelatest(example.linear_growth_rate, k; kwargs...)
    k, gravity, density = 2pi, 1.0, (3.0, 1.0)
    inviscid = sqrt(0.5 * gravity * k)
    @test rate(k; density, dynamic_viscosity = (0.0, 0.0), gravity) ≈ inviscid
    # Capillarity: n² = Agk - σk³/(ρ_h + ρ_l), zero beyond the cutoff.
    sigma = 0.01
    @test rate(k; density, dynamic_viscosity = (0.0, 0.0), gravity, surface_tension = sigma) ≈
        sqrt(inviscid^2 - sigma * k^3 / sum(density))
    @test iszero(rate(k; density, dynamic_viscosity = (0.0, 0.0), gravity, surface_tension = 1.0))
    # The viscous root satisfies the implicit form of the two-fluid relation
    # (Xie, Tao & Li 2017, eqs. 5–6, after Chandrasekhar §94), also for
    # unequal kinematic viscosities.
    for (ν_upper, ν_lower) in ((5.7735e-4, 5.7735e-4), (1.0e-3, 5.0e-3))
        μ = (density[1] * ν_upper, density[2] * ν_lower)
        n = rate(k; density, dynamic_viscosity = μ, gravity)
        sum_term = 1 / (μ[2] * k + sqrt(μ[1]^2 * k^2 + density[1] * n * μ[1])) +
            1 / (μ[1] * k + sqrt(μ[2]^2 * k^2 + density[2] * n * μ[2]))
        residual = (-gravity * (density[1] - density[2]) * k + sum(density) * n^2) *
            sum_term + 4n * k
        @test 0 < n < inviscid
        @test abs(residual) <= 1.0e-10 * 4n * k
    end
    # At Re_p = 1000 the explicit equal-ν approximation √(Agk + ν²k⁴) - νk²
    # overestimates the exact rate by 4 %.
    ν = sqrt(0.5 / 1.5) / 1000
    exact = rate(k; density, dynamic_viscosity = density .* ν, gravity)
    @test exact ≈ 1.68219 atol = 5.0e-6
    @test (sqrt(inviscid^2 + ν^2 * k^4) - ν * k^2) / exact ≈ 1.04 atol = 1.0e-3
    # Goncharov's 2-D Froude number is 1/√(3π) at every Atwood number; at A=1
    # the velocity is Layzer's √(g/(3k)).
    for atwood in (0.2, 0.5, 0.9)
        U = Base.invokelatest(example.bubble_velocity; atwood, gravity, wavelength = 1.0)
        @test U / sqrt(atwood * gravity / (1 + atwood)) ≈ 1 / sqrt(3pi)
    end
    @test Base.invokelatest(example.bubble_velocity; atwood = 1.0, gravity, wavelength = 1.0) ≈
        sqrt(gravity / (3k))
end

@testset "Rayleigh–Taylor initial conditions" begin
    example = load_example(:rayleigh_taylor)
    single = call_example(example, :rayleigh_taylor_initial_condition; case = :single_mode)
    @test single.modes == [1]
    @test single.elevation(0.0) ≈ 0.002
    @test single.elevation(0.5) ≈ -0.002
    @test single.levelset(0.0, 0.001) > 0 > single.levelset(0.0, 0.003)
    multimode = call_example(example, :rayleigh_taylor_initial_condition)
    @test multimode.modes == collect(8:16)
    samples = multimode.elevation.((0:4095) ./ 4096)
    @test sqrt(sum(abs2, samples) / length(samples)) ≈ 0.001 rtol = 1.0e-12
    # The seed fixes the phases.
    @test multimode.phases ==
        call_example(example, :rayleigh_taylor_initial_condition; seed = 2026).phases
    @test multimode.phases !=
        call_example(example, :rayleigh_taylor_initial_condition; seed = 1).phases
end

@testset "Single-mode Rayleigh–Taylor linear growth" begin
    example = load_example(:rayleigh_taylor)
    # λ/Δx = 32 with a fixed step: the measured linear rate is 4.25 % below
    # the viscous theory (4.27 % with the gallery's dt = 0.005; the error
    # falls with resolution, see the README).
    run = call_example(
        example, :run_example;
        case = :single_mode, max_level = 2, final_time = 2.3, dt = 0.01,
    )
    @test run.result.steps == 230
    bubble = diagnostic_history(run.flow, :bubble_height)
    spike = diagnostic_history(run.flow, :spike_height)
    amplitude = (bubble.values .+ spike.values) ./ 2
    # The axis cell column averages the cosine over its width.
    @test first(amplitude) ≈ 0.002 * sin(2pi / 32) / (2pi / 32) rtol = 1.0e-6
    window = findall(a -> 6 * first(amplitude) <= a <= 20 * first(amplitude), amplitude)
    t, y = bubble.times[window], log.(amplitude[window])
    rate = sum((t .- sum(t) / length(t)) .* y) / sum(abs2, t .- sum(t) / length(t))
    p = run.parameters
    theory = Base.invokelatest(
        example.linear_growth_rate, 2pi;
        density = p.density, dynamic_viscosity = p.dynamic_viscosity,
        gravity = p.gravity,
    )
    @test -0.05 < rate / theory - 1 < -0.035
    @test run.result.pressure.converged
    test_finite_example_fields(run.flow)
    test_phase_volume_conservation(run.flow)
end

@testset "Multimode Rayleigh–Taylor dye, Studio and velocity models" begin
    example = load_example(:rayleigh_taylor)
    initial = call_example(example, :run_example; max_level = 2, final_time = 0.0)
    dye = (integrate(initial.dye, :liquid), integrate(initial.dye, :gas))
    # The automatic step is the gravity-wave bound here (dt = 0.14), so a
    # fixed step gives the transport and the regrids some work.
    run = call_example(
        example, :run_example;
        max_level = 2, final_time = 0.5, dt = 0.01, start_studio = example_start_probe,
    )
    @test isopen(run.app)
    close(run.app)
    @test run.result.steps == 50
    @test run.result.pressure.converged
    @test run.flow.system.mesh.adaptation.last_update_step > 2
    @test diagnostic_names(run.flow) == [
        :bubble_height, :spike_height, :spike_origin, :bubble_origin,
        :phase_volume, :kinetic_energy, :potential_energy, :cells,
    ]
    bubble = diagnostic_history(run.flow, :bubble_height)
    spike = diagnostic_history(run.flow, :spike_height)
    @test all(isfinite, bubble.values) && all(isfinite, spike.values)
    @test last(bubble.values) > first(bubble.values)
    @test last(spike.values) > first(spike.values)
    origin = diagnostic_history(run.flow, :spike_origin)
    @test last(origin.values) > 0
    # The dye integrals of both phases are conserved up to the cleanup and
    # transfer budget of the phase volumes.
    tolerance = max(2 * run.flow.steps + run.flow.mesh_updates, 1) *
        run.flow.interface.fraction_floor * prod(run.flow.system.mesh.extent)
    @test integrate(run.dye, :liquid) ≈ dye[1] atol = tolerance
    @test integrate(run.dye, :gas) ≈ dye[2] atol = tolerance
    store = snapshot_store(run.recorder)
    @test :dye in field_names(store[last(keyframes(store))])
    test_finite_example_fields(run.flow)
    test_phase_volume_conservation(run.flow)

    two = call_example(
        example, :run_example;
        max_level = 1, final_time = 0.05, velocity_model = TwoVelocityModel(),
    )
    @test two.flow.velocity isa TwoPhaseVelocity
    @test two.flow.system.two_phase.pressure_projection isa CMDGFMProjection
    # One vorticity criterion per phase velocity.
    @test count(
        criterion -> criterion isa VorticityRefinementIndicator,
        two.flow.system.adaptivity.criteria,
    ) == 2
    @test two.result.projection.converged
    test_finite_example_fields(two.flow)
    test_phase_volume_conservation(two.flow)
end
