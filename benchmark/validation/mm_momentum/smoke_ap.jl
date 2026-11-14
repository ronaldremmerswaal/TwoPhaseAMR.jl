# Quick check of the aperture-activity prototype: constants, one vortex run per design.
include(joinpath(@__DIR__, "common.jl"))
const R = 0.2
println("activity=", ACTIVITY, " ties=", TIES)
for (design, match) in ((:D0, false), (:DI, false), (:DP, true), (:DS, false), (:DF, false), (:DG, true))
    set_design!(design)
    TP._MIMETIC_MASS_MATCH[] = match
    for (label, U, amp) in (("const", (1.0, 0.37), 0.0), ("vortex", (1.0, 0.37), 0.3))
        TP._MIMETIC_TRANSPORT_LOG[] = Any[]
        v = (x, y, t, axis) -> U[axis] + amp * (axis == 1 ? sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y))
        flow, interface = make_flow((x, y) -> hypot(x - 0.52, y - 0.47) - R; n = 32, density = (1.0, 1.0e-3), velocity = (v, v))
        status = "ok"
        try
            for _ in 1:8
                dt = stable_timestep(flow.velocity, flow.system, flow.workspace; cfl = 0.45)
                do_step!(flow, dt)
            end
        catch err
            status = first(sprint(showerror, err), 200)
        end
        lg = TP._MIMETIC_TRANSPORT_LOG[]
        dev = constant_deviation(state_of(flow).convected, interface.mesh, U)
        if isempty(lg)
            println(design, match ? "+mm " : " ", label, " ", status); continue
        end
        k = argmax([x.correction for x in lg])
        dP = [x.momentum_after[5] - x.momentum_before[5] for x in lg]
        dE = [x.energy_after - x.energy_before for x in lg]
        println(rpad(string(design, match ? "+mm" : ""), 6), " ", rpad(label, 6), " dev=", @sprintf("%.2e", dev),
            " maxcorr=", @sprintf("%.3e", lg[k].correction), " max|dPx|=", @sprintf("%.2e", maximum(abs, dP)),
            " dE=", @sprintf("%+.2e", sum(dE)), " merge=", lg[end].merge, " ", status)
        amp > 0 && println("        worst=", lg[k].worst)
    end
    TP._MIMETIC_TRANSPORT_LOG[] = nothing
end
