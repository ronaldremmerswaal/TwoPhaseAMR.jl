# Sharpness experiment 01: how sharp are the empirical stability margins of Section 13.3? Bisection of the largest stable step multiple between
# the tested multiples (the chapter's "largest tested multiple below the first unstable one" is a bracket, not the limit).
#
# Label:  sec:stab:measured (ch:stability-accuracy)
# Source: docs/techref/chapters/13_stability_accuracy.tex:331-368; docs/techref/figures/stability.jl (stab_margin_run, stability_margin_check)
# Checks:
#   B1  capillary wave (two velocities, 40 steps): the multiple bracketed by 2.5 (stable) and 2.75 (unstable) is refined by bisection to 0.01;
#       the limit exceeds the Denner--van Wachem static factor sqrt2 and the code's bound (factor 1);
#   B2  Kelvin--Helmholtz wave (6 steps): bracket 1.35 .. 1.5 refined to 0.01 (WY cfl 0.45 m); the stable WY cfl exceeds 1/2.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/ch-stability-accuracy/scripts/sharp_01_margin_bisection.jl

isdefined(Main, :STAB_CFL) || include(joinpath(@__DIR__, "_prelude.jl"))

function bisect(stable, lo, hi; tol)
    @assert stable(lo) && !stable(hi)
    while hi - lo > tol
        mid = (lo + hi) / 2
        stable(mid) ? (lo = mid) : (hi = mid)
    end
    return lo, hi
end

@testset "ch13 sharpness: margins" begin
    wave_build = () -> stab_capillary_wave(velocity_model = TwoVelocityModel())
    scale = STAB_WAVE.amplitude * wave_build().omega
    wave_stable(m) = let r = stab_margin_run(wave_build, m; steps = 40)
        isempty(r.failure) && length(r.speeds) == 40 && maximum(r.speeds) <= scale
    end
    lo, hi = bisect(wave_stable, 2.5, 2.75; tol = 0.01)
    @test sqrt(2) < lo
    println("B1 capillary wave: largest stable multiple in [", lo, ", ", hi, "] (chapter: 2.5 stable, 2.75 unstable); vs Denner static sqrt2 = ", sqrt(2))
    kh_stable(m) = let r = stab_margin_run(stab_kelvin_helmholtz, m; steps = 6)
        isempty(r.failure) && length(r.speeds) == 6
    end
    lo, hi = bisect(kh_stable, 1.35, 1.5; tol = 0.01)
    @test STAB_CFL * lo > 0.5
    println("B2 KH: largest stable multiple in [", lo, ", ", hi, "], WY cfl in [", STAB_CFL * lo, ", ", STAB_CFL * hi, "] (bound 0.5)")
end
