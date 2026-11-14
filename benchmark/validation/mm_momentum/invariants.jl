# 𝓜-momentum and 𝓜-energy of the transport stage versus dt at fixed grid:
# drop R = 0.2 in U = (1, 0.37) plus a vortex of amplitude 0.3, n = 32, inviscid,
# no surface tension, Lax-Wendroff without the CTU switch, t = 0.25, fixed
# dt = f * (initial full-cell bound at CFL 0.45), f = 1, 1/2, 1/4.
# Reported per run: Σ transport energy changes / E0, max |ΔP_mix| per step and
# |Σ ΔP_mix| relative to Σ m|u|, steps.
#   MM_ACTIVITY=... julia --project=<wt> invariants.jl
include(joinpath(@__DIR__, "common.jl"))

function invariants(design, match, factor; density = (1.0, 1.0e-3), n = 32, final = 0.25, ctu = floatmin())
    set_design!(design)
    TP._MIMETIC_MASS_MATCH[] = match
    U = (1.0, 0.37)
    v = (x, y, t, axis) -> U[axis] + 0.3 * (axis == 1 ? sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y))
    flow, interface = make_flow((x, y) -> hypot(x - 0.5, y - 0.47) - 0.2; n, density, velocity = (v, v), ctu_threshold = ctu)
    _, E0, W0 = metric_invariants(flow)
    scale = abs(W0[1]) + abs(W0[2])
    dt = factor * stable_timestep(flow.velocity, flow.system, flow.workspace; cfl = 0.45)
    steps = round(Int, final / dt)
    dt = final / steps
    TP._MIMETIC_TRANSPORT_LOG[] = Any[]
    status = "ok"
    try
        for _ in 1:steps
            do_step!(flow, dt)
        end
    catch err
        status = "error: " * first(sprint(showerror, err), 120)
    end
    lg = TP._MIMETIC_TRANSPORT_LOG[]
    TP._MIMETIC_TRANSPORT_LOG[] = nothing
    dE = sum((x.energy_after - x.energy_before for x in lg); init = 0.0) / E0
    dPs = [hypot(x.momentum_after[5] - x.momentum_before[5], x.momentum_after[6] - x.momentum_before[6]) for x in lg]
    sumP = hypot(sum(x.momentum_after[5] - x.momentum_before[5] for x in lg), sum(x.momentum_after[6] - x.momentum_before[6] for x in lg))
    corr = maximum((x.correction for x in lg); init = 0.0)
    set_design!(:D0)
    TP._MIMETIC_MASS_MATCH[] = false
    return (; dE, maxP = maximum(dPs; init = 0.0) / scale, sumP = sumP / scale, steps = length(lg), corr, status)
end

open(joinpath(OUT, "invariants" * SUFFIX * ".txt"), "w") do io
    variants = ACTIVITY === :aperture ?
        ((:D0, false), (:DI, false), (:DP, true), (:DG, true), (:DGP, true), (:DS, false)) :
        ((:D0, false), (:DI, false), (:DP, true))
    for density in ((1.0, 1.0), (1.0, 1.0e-3)), (design, match) in variants, factor in (1.0, 0.5, 0.25)
        r = invariants(design, match, factor; density)
        s = @sprintf("rho_g=%.0e %-4s%-3s dtf=%.2f steps=%4d  Σ dE_transport/E0=%+.3e  max|ΔP|/S=%.2e  |ΣΔP|/S=%.2e  maxcorr=%.2e %s",
            density[2], design, match ? "+mm" : "", factor, r.steps, r.dE, r.maxP, r.sumP, r.corr, r.status)
        println(io, s); flush(io); println(s); flush(stdout)
    end
end
