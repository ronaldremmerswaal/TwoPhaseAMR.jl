# Small cut cells and the time step: a drop carried by U = (1, 0.37) plus a
# solenoidal vortex of amplitude 0.3 crosses the periodic box once (t = 1),
# so the interface sweeps through near-empty faces; the step is a fraction
# `cfl` of the full-cell VOF bound (WY: cfl <= 0.5). Reported: max phase |u|
# over all values and over values whose phase aperture is at least 0.05 (or
# bulk), relative to the initial max; 𝓜-energy max/initial; min fraction.
include(joinpath(@__DIR__, "common.jl"))

const VARIANTS = ((:D0, false), (:DI, false), (:DP, true), (:DG, true), (:DF, false))

function sweep(design, match, cfl; n = 32, density = (1.0, 1.0e-3), floor = sqrt(eps()), final = 1.0)
    set_design!(design)
    TP._MIMETIC_MASS_MATCH[] = match
    U = (1.0, 0.37)
    v = (x, y, t, axis) -> U[axis] + 0.3 * (axis == 1 ? sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y))
    flow, interface = make_flow((x, y) -> hypot(x - 0.5, y - 0.47) - 0.2; n, density, velocity = (v, v), fraction_floor = floor)
    umax0 = maximum(abs, filter(isfinite, flow.velocity[1].interior))
    _, E0, _ = metric_invariants(flow)
    umax = umax0
    uwet = umax0
    Emax = E0
    fmin = 1.0
    status = "ok"
    steps = 0
    wall = @elapsed try
        while flow.time < final - 1.0e-12
            dt = min(stable_timestep(flow.velocity, flow.system, flow.workspace; cfl), final - flow.time)
            do_step!(flow, dt)
            steps += 1
            la, ga = TP._current_phase_face_apertures!(interface)
            for phase in 1:2, (index, u) in pairs(flow.velocity[phase].interior)
                isfinite(u) || continue
                umax = max(umax, abs(u))
                (phase == 1 ? la[index] : ga[index]) >= 0.05 && (uwet = max(uwet, abs(u)))
            end
            _, E, _ = metric_invariants(flow)
            Emax = max(Emax, E)
            fmin = min(fmin, min_fraction(interface))
            (isfinite(E) && E < 100E0 && umax < 1.0e4 * umax0) || (status = "blow-up"; break)
            steps < 5000 || (status = "dt collapse (>5000 steps)"; break)
        end
    catch err
        status = "error: " * first(sprint(showerror, err), 120)
    end
    set_design!(:D0)
    TP._MIMETIC_MASS_MATCH[] = false
    return (; umax = umax / umax0, uwet = uwet / umax0, E = Emax / E0, fmin, status, steps, wall)
end

const CFLS = isempty(ARGS) ? (0.1, 0.25, 0.5) : Tuple(parse.(Float64, ARGS))
open(joinpath(OUT, "smallcell.txt"), "w") do io
    for density in ((1.0, 1.0), (1.0, 1.0e-3)), floor in (sqrt(eps()), 1.0e-12), cfl in CFLS, (design, match) in VARIANTS
        r = sweep(design, match, cfl; floor, density)
        s = @sprintf("rho_g=%.0e floor=%.0e cfl=%.2f %-3s%-3s max|u|/u0 all=%9.3f wet=%7.4f Emax/E0=%9.4f minC=%.1e steps=%d wall/step=%.2ems %s",
            density[2], floor, cfl, design, match ? "+mm" : "", r.umax, r.uwet, r.E, r.fmin, r.steps, 1.0e3r.wall / max(r.steps, 1), r.status)
        println(io, s); flush(io); println(s); flush(stdout)
    end
end
