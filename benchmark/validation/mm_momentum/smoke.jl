include(joinpath(@__DIR__, "common.jl"))
const R = 0.2
for match in (false, true), design in (:D0, :DI, :DS, :DF, :DG)
    set_design!(design)
    TP._MIMETIC_MASS_MATCH[] = match
    TP._MIMETIC_TRANSPORT_LOG[] = Any[]
    U = (1.0, 0.37)
    shared = (x, y, t, axis) -> U[axis]
    vortex = (x, y, t, axis) -> U[axis] + 0.3 * (axis == 1 ? sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y))
    flow, interface = make_flow((x, y) -> hypot(x - 0.52, y - 0.47) - R; n = 32, density = (1.0, 1.0e-3), velocity = (shared, shared))
    for _ in 1:5
        dt = stable_timestep(flow.velocity, flow.system, flow.workspace; cfl = 0.45)
        do_step!(flow, dt)
    end
    d = constant_deviation(state_of(flow).convected, interface.mesh, U)
    l = TP._MIMETIC_TRANSPORT_LOG[][end]
    th = l.theta
    dP = l.momentum_after[5] - l.momentum_before[5]
    dW = l.w_momentum_transport[1] - l.w_momentum_before[1]
    dE = l.excess_after[5] - l.excess_before[5]
    orph = l.orphan[1] + l.orphan[3]
    pred = dW + th * dE - (1 - th) * orph
    println(match ? "MATCH " : "      ", design, " dev=", d, " dPx=", dP, " pred=", pred, " dW=", dW, " dEnergy=", l.energy_after - l.energy_before,
        " red=", (l.redistribution.iterations, l.redistribution.residual, l.redistribution.obstruction), " band=", l.band,
        " t=", (l.assembly_time, l.solve_time, l.redistribution.time), " mm=", l.mass_match)
    # vortex field: correction size
    TP._MIMETIC_TRANSPORT_LOG[] = Any[]
    flow, interface = make_flow((x, y) -> hypot(x - 0.52, y - 0.47) - R; n = 32, density = (1.0, 1.0e-3), velocity = (vortex, vortex))
    for _ in 1:5
        dt = stable_timestep(flow.velocity, flow.system, flow.workspace; cfl = 0.45)
        do_step!(flow, dt)
    end
    lg = TP._MIMETIC_TRANSPORT_LOG[]
    k = argmax([x.correction for x in lg])
    println("     vortex: maxcorr=", lg[k].correction, " at step ", k, " worst=", lg[k].worst,
        " red=", (lg[k].redistribution.iterations, lg[k].redistribution.residual, lg[k].redistribution.obstruction),
        "\n       dPx=", [x.momentum_after[5] - x.momentum_before[5] for x in lg],
        "\n       dE=", [x.energy_after - x.energy_before for x in lg])
    TP._MIMETIC_TRANSPORT_LOG[] = nothing
end
