# Code check 01: the time-step bounds of Section 13.1 and the step check of Section 13.2 against src/.
#
# Label:  eq:stab:flow-bound, eq:stab:check, dev:stab:gravity (ch:stability-accuracy)
# Source: docs/techref/chapters/13_stability_accuracy.tex:59-128 (bounds, speed-corrected capillary bound, gravity), 235-263 (step check);
#         src/two_phase/stepping.jl:40-60 (_shared_velocity_timestep_bound), :120-125 (check), src/two_phase/two_velocity/stepping.jl:48-90,
#         src/single_phase/stepping.jl:17-20, src/vof/workspace.jl:256-300 (WY bound), src/operators/advection/staggered_update.jl:426-443
# Checks:
#   C1  uniform translation of a flat interface (U, 0): dt_wy = cfl h/U exactly (cfl = 0.45); flow bound = min{cfl h/U, h/(2 c_sigma)} from the code;
#       the ratio to the Denner--van Wachem dynamic bound h/(sqrt2 c_sigma + U) is cfl + 2^(-1/2) = 1.157 at u = U/c_sigma = 2 cfl (maximum over a
#       u-sweep through the code), equal bounds there; ratio 2^(-1/2) at U = 0; one- and two-velocity (frozen) models;
#   C2  dt_cap = h/(2 c_sigma) = 2^(-1/2) T(lambda = h) at rest on the composite capillary-wave mesh (h = 1/64, the smallest leaf width);
#   C3  dt_grav of the code = thesis (7.4)/sqrt2 at rest; step! names the active bound ("capillary", "gravity", "advective");
#   C4  step!: accepts dt = (1 + 0.9e-6) dt(cfl = 1/2), rejects (1 + 2e-6) dt(cfl = 1/2) [capillary- and gravity-limited: relative to the bound itself;
#       advective-limited: margin 0.5/0.45 = 1.11 relative to the step selected at cfl = 0.45], one and two velocities;
#   C5  cfl ranges: two phases (0, 1/2], one phase (0, 1]; single-phase step! uses cfl = 1/2 in its check, so simulate!(cfl = 0.8) of a single phase fails;
#   C6  bounds are infinite without rates (sigma = 0 or rho_l = rho_g, g = 0, U = 0).
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/ch-stability-accuracy/scripts/code_01_bounds.jl

isdefined(Main, :STAB_CFL) || include(joinpath(@__DIR__, "_prelude.jl"))

@testset "ch13 code: bounds and step check" begin
    h = 1 / 16
    sigma = 0.1
    rho = 1.001
    c_sigma = sqrt(pi * sigma / (rho * h))
    cfl = STAB_CFL

    @testset "C1 uniform translation, speed-corrected excess" begin
        for model in (OneVelocityModel(), two_velocity_model())
            ratios = Float64[]
            us = (0.0, 0.3, 0.6, 2cfl, 1.2, 2.0, 5.0)
            for u in us
                U = u * c_sigma
                case = uniform_two_phase(; sigma, U, velocity_model = model)
                b = stab_bounds(case; cfl)
                @test b.h == h
                if u > 0
                    @test b.advective ≈ cfl * h / U rtol = 1.0e-13                  # dt_wy = cfl h/U
                else
                    @test isinf(b.advective)
                end
                @test b.capillary ≈ h / (2c_sigma) rtol = 1.0e-13                    # dt_cap = h/(2 c_sigma)
                @test b.code ≈ min(cfl * h / max(U, eps()), h / (2c_sigma)) rtol = 1.0e-13
                dynamic = h / (sqrt(2) * c_sigma + U)
                push!(ratios, b.code / dynamic)
                if u == 2cfl
                    @test b.advective ≈ b.capillary rtol = 1.0e-13                    # equal bounds at U = 2 cfl c_sigma
                end
            end
            @test ratios[1] ≈ 1 / sqrt(2) rtol = 1.0e-13                             # static: sqrt2 stricter than Denner--van Wachem
            @test maximum(ratios) ≈ cfl + 1 / sqrt(2) rtol = 1.0e-13
            @test argmax(ratios) == findfirst(==(2cfl), us)
            @test round(maximum(ratios); digits = 2) == 1.16
            println("C1 ", nameof(typeof(model)), ": ratios selected/dynamic over u = ", us, ": ", round.(ratios; digits = 4))
        end
    end

    @testset "C2 capillary bound on the composite capillary-wave mesh" begin
        wave = stab_capillary_wave()
        b = stab_bounds(wave)
        hw = stab_min_width(wave.mesh)
        @test hw == 1 / 64 && b.h == hw
        cs = sqrt(pi * wave.sigma / (sum(wave.density) * hw))
        @test b.capillary ≈ hw / (2cs) rtol = 1.0e-13
        period = 2pi / sqrt(wave.sigma * (2pi / hw)^3 / sum(wave.density))         # T(lambda = h), omega^2 = sigma k^3/rho
        @test b.capillary ≈ period / sqrt(2) rtol = 1.0e-13
        @test b.code == b.capillary && b.active == :capillary && isinf(b.advective)
        println("C2 wave: dt_cap = ", b.capillary, ", T(lambda=h)/sqrt2 = ", period / sqrt(2), ", Denner static = ", sqrt(2) * b.capillary)
    end

    @testset "C3/C4 gravity, names and the check" begin
        g = 1.0
        for model in (OneVelocityModel(), two_velocity_model())
            # gravity-limited at rest
            case = uniform_two_phase(; gravity = g, velocity_model = model)
            b = stab_bounds(case)
            thesis74 = sqrt((1.0 + 1.0e-3) / (1.0 - 1.0e-3) * h / g)
            @test b.gravity ≈ thesis74 / sqrt(2) rtol = 1.0e-13
            @test b.active == :gravity
            hard = stable_timestep(case.flow; cfl = 0.5)
            @test hard == b.gravity                                       # not scaled by cfl
            @test occursin("gravity", step_message(uniform_two_phase(; gravity = g, velocity_model = model).flow, (1 + 2.0e-6) * hard))
            @test step_message(uniform_two_phase(; gravity = g, velocity_model = model).flow, (1 + 0.9e-6) * hard) == ""
            # capillary-limited at rest
            case = uniform_two_phase(; sigma, velocity_model = model)
            hard = stable_timestep(case.flow; cfl = 0.5)
            @test hard == stab_bounds(case).capillary
            @test occursin("capillary", step_message(uniform_two_phase(; sigma, velocity_model = model).flow, (1 + 2.0e-6) * hard))
            @test step_message(uniform_two_phase(; sigma, velocity_model = model).flow, (1 + 0.9e-6) * hard) == ""
            # advective-limited: selected at 0.45, check at 0.5
            U = 0.3
            case = uniform_two_phase(; U, velocity_model = model)
            selected = stable_timestep(case.flow; cfl)
            hard = stable_timestep(case.flow; cfl = 0.5)
            @test hard / selected ≈ 0.5 / 0.45 rtol = 1.0e-13
            @test selected ≈ cfl * h / U rtol = 1.0e-13
            @test step_message(uniform_two_phase(; U, velocity_model = model).flow, selected * 0.5 / 0.45 * (1 - 1.0e-9)) == ""
            @test step_message(uniform_two_phase(; U, velocity_model = model).flow, (1 + 0.9e-6) * hard) == ""
            msg = step_message(uniform_two_phase(; U, velocity_model = model).flow, (1 + 2.0e-6) * hard)
            @test occursin("advective", msg)
            # a larger simultaneous capillary term: the name is the active (smaller) one
            case = uniform_two_phase(; sigma, U = 5c_sigma, velocity_model = model)
            hard = stable_timestep(case.flow; cfl = 0.5)
            @test occursin("advective", step_message(uniform_two_phase(; sigma, U = 5c_sigma, velocity_model = model).flow, (1 + 2.0e-6) * hard))
        end
    end

    @testset "C5 cfl ranges" begin
        two = uniform_two_phase(; sigma, U = 0.3)
        @test isfinite(stable_timestep(two.flow; cfl = 0.5))
        @test_throws ArgumentError stable_timestep(two.flow; cfl = 0.51)
        @test_throws ArgumentError stable_timestep(two.flow; cfl = 0.0)
        one = stab_taylor_green()
        @test isfinite(stable_timestep(one; cfl = 1.0))
        @test_throws ArgumentError stable_timestep(one; cfl = 1.01)
        @test stable_timestep(one; cfl = 1.0) ≈ 2 * stable_timestep(one; cfl = 0.5) rtol = 1.0e-13
        # the check of step! uses cfl = 1/2: a single-phase simulate! with cfl in (1/2, 1] stops at its first step
        err = try
            simulate!(stab_taylor_green(), 0.05; cfl = 0.8)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("advective stability bound", err.msg)
        ok = simulate!(stab_taylor_green(), 0.05; cfl = 0.5)
        @test ok.steps >= 1
        println("C5 single-phase simulate!(cfl = 0.8): ", err isa ArgumentError ? err.msg : err)
    end

    @testset "C6 bounds without rates are infinite" begin
        @test isinf(stab_bounds(uniform_two_phase()).code)                                        # no velocity, sigma = g = 0
        @test isinf(stab_bounds(uniform_two_phase(; sigma = 0.0, gravity = 1.0, density = (1.0, 1.0))).code)
        @test stab_bounds(uniform_two_phase(; sigma, density = (1.0, 1.0))).capillary ≈ sqrt(2h^3 / (4pi * sigma)) rtol = 1.0e-13
    end
end
