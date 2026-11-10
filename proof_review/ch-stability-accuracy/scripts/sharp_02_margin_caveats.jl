# Sharpness experiment 02: caveats of the empirical margins of Section 13.3 -- (a) the Kelvin--Helmholtz margin is relative to the INITIAL bound
# while the bound falls during the run (effective WY cfl of the last stable step), (b) the capillary-wave margin is a finite-horizon (40 steps) criterion.
#
# Label:  sec:stab:measured (ch:stability-accuracy)
# Source: docs/techref/chapters/13_stability_accuracy.tex:331-368; docs/techref/figures/stability.jl (stab_margin_run)
# Checks:
#   H1  KH wave, multiples 1.35 (stable) and 1.5 (fails in step 5): the ratio dt/dtflow(1/2) at the start of every step (the effective WY cfl is 0.5 x ratio)
#       is recorded; the stable run's last-step effective cfl exceeds the nominal 0.45 x 1.35 = 0.61;
#   H2  capillary wave, multiples 2.0 and 2.5: 40 steps (the chapter's horizon) against 160 steps (4x): does the largest velocity stay below a*omega?
#   H3  capillary wave at the unstable multiple 2.75: the run completes its 40 steps without an error ("during the 40-step run", l. 354), the speed first
#       exceeds a*omega after 14 steps and reaches 36.9 times it
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/ch-stability-accuracy/scripts/sharp_02_margin_caveats.jl

isdefined(Main, :STAB_CFL) || include(joinpath(@__DIR__, "_prelude.jl"))

# stab_margin_run with a per-step record of dt / dtflow(1/2) (taken after the bound evaluation that precedes every step)
function margin_run_recorded(build, m; steps)
    case = build()
    flow = case.flow
    AMR._enforce_current_velocity_boundary!(flow.velocity, flow.system, flow.time)
    dt = m * stable_timestep(flow; cfl = STAB_CFL)
    ratios, speeds = Float64[], Float64[]
    failure = ""
    for _ in 1:steps
        try
            AMR._enforce_current_velocity_boundary!(flow.velocity, flow.system, flow.time)
            push!(ratios, dt / stable_timestep(flow; cfl = 0.5))
            flow.workspace.transport_geometry_prepared = true
            stats = step!(
                flow.velocity, flow.pressure, flow.system, dt, flow.workspace;
                time = flow.time,
            )
            stats.converged || (failure = "projection did not converge"; break)
        catch error
            failure = first(split(sprint(showerror, error), '\n'))
            break
        end
        flow.time += dt
        state = SimulationState(flow.velocity, flow.pressure, flow.system, flow.time, 0, flow.workspace)
        push!(speeds, diagnostic_value(MaximumVelocity(), state))
        isfinite(last(speeds)) || (failure = "non-finite velocity"; break)
    end
    return (; m, ratios, speeds, failure)
end

@testset "ch13 sharpness: margin caveats" begin
    @testset "H1 KH: effective cfl relative to the current bound" begin
        stable = margin_run_recorded(stab_kelvin_helmholtz, 1.35; steps = 6)
        failing = margin_run_recorded(stab_kelvin_helmholtz, 1.5; steps = 6)
        @test isempty(stable.failure) && !isempty(failing.failure)
        println("H1 m = 1.35: dt/dtflow(1/2) per step ", round.(stable.ratios; digits = 3), " (nominal ", round(1.35 * 0.45 / 0.5; digits = 3), ")")
        println("H1 m = 1.5 : dt/dtflow(1/2) per step ", round.(failing.ratios; digits = 3), ", failure: ", first(failing.failure, 60))
        @test maximum(stable.ratios) >= 1.35 * 0.45 / 0.5 * (1 - 1.0e-9)
        @test maximum(failing.ratios) > 1                                                      # the failing run steps above dtflow(1/2) (check not applied)
    end
    @testset "H2 capillary wave: horizon" begin
        build = () -> stab_capillary_wave(velocity_model = TwoVelocityModel())
        scale = STAB_WAVE.amplitude * build().omega
        for m in (2.0, 2.5)
            short = margin_run_recorded(build, m; steps = 40)
            long = margin_run_recorded(build, m; steps = 160)
            println(
                "H2 m = ", m, ": max speed / (a omega) over 40 steps ", maximum(short.speeds) / scale, ", over 160 steps ",
                maximum(long.speeds) / scale, "; failures '", short.failure, "' / '", long.failure, "'"
            )
            @test isempty(short.failure) && maximum(short.speeds) <= scale
        end
    end
    @testset "H3 unstable capillary run completes its steps" begin
        build = () -> stab_capillary_wave(velocity_model = TwoVelocityModel())
        scale = STAB_WAVE.amplitude * build().omega
        run = margin_run_recorded(build, 2.75; steps = 40)
        @test isempty(run.failure) && length(run.speeds) == 40
        @test findfirst(>(scale), run.speeds) == 14
        @test isapprox(maximum(run.speeds) / scale, 36.9, atol = 0.1)
        println("H3 m = 2.75: ", length(run.speeds), " steps, first above a omega at step ", findfirst(>(scale), run.speeds), ", maximum ", maximum(run.speeds) / scale)
    end
end
