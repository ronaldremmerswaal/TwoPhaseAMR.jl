# Drop impact on a pool (setup of mimetic-projection-audit/scripts/mimetic_st_pkg.jl
# group `impact`, package CGFM force): pool y < 0.3, drop R = 0.08 at
# (0.5 + 0.17/n, 0.55 + 0.11/n), drop velocity -2 (y), DR 1000, sigma 0.05,
# mu = (1e-3, 1e-6), periodic x, free-slip walls in y, t_end 0.25.
#   MM_ACTIVITY=volume|aperture julia --project=<wt> impact.jl <design> [match] [n]
include(joinpath(@__DIR__, "common.jl"))
const DESIGN = Symbol(ARGS[1])
const MATCH = length(ARGS) >= 2 && ARGS[2] == "match"
const N = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 32
set_design!(DESIGN)
TP._MIMETIC_MASS_MATCH[] = MATCH
const TAG = "$(DESIGN)$(MATCH ? "+mm" : "")$(SUFFIX)"

"Largest |u| over the phase values and where it sits."
function speed_max(flow)
    interface = flow.system.two_phase.interface
    la, ga = TP._current_phase_face_apertures!(interface)
    best = (0.0, 0, 0)
    for phase in 1:2, (index, u) in pairs(flow.velocity[phase].interior)
        isfinite(u) && abs(u) > best[1] && (best = (abs(u), index, phase))
    end
    _, index, phase = best
    index == 0 && return (0.0, "")
    face = interface.mesh.faces[index]
    s = TP._staggered_liquid_fraction(interface, nothing, face)
    a = phase == 1 ? la[index] : ga[index]
    sp = phase == 1 ? s : 1 - s
    status = a > 0 ? (a < 0.05 ? "small" : "wetted") : "slaved"
    return best[1], @sprintf("face %d axis %d phase %d aperture %.2e staggered-fraction %.2e %s", index, face.axis, phase, a, sp, status)
end

function impact(n)
    h = 1 / n
    xc, yc, R = 0.5 + 0.17 / n, 0.55 + 0.11 / n, 0.08
    ls = (x, y) -> min(hypot(x - xc, y - yc) - R, y - 0.3)
    vy(x, y) = hypot(x - xc, y - yc) < R + 1.5h ? -2.0 : 0.0
    v = (x, y, t, a) -> a == 2 ? vy(x, y) : 0.0
    flow, interface = make_flow(ls; n, density = (1.0, 1.0e-3), sigma = 0.05, viscosity = (1.0e-3, 1.0e-6),
        periodic = (true, false), velocity = (v, v))
    _, E0, _ = metric_invariants(flow)
    Emax = E0
    umax = 0.0
    where = ""
    status = "ok"
    steps = 0
    series = open(joinpath(OUT, "impact_series_$(TAG)_n$(n).txt"), "w")
    wall = @elapsed try
        while flow.time < 0.25 - 1.0e-12 && steps < 20000
            dt = min(stable_timestep(flow.velocity, flow.system, flow.workspace; cfl = 0.45), 0.25 - flow.time)
            stats = do_step!(flow, dt)
            steps += 1
            _, E, _ = metric_invariants(flow)
            Emax = max(Emax, E)
            u, w = speed_max(flow)
            u > umax && ((umax, where) = (u, @sprintf("t=%.4f %s", flow.time, w)))
            println(series, @sprintf("t=%.5f E=%.6e umax=%.4e %s", flow.time, E, u, w))
            isfinite(E) || (status = "non-finite"; break)
            stats.converged || (status = "projection not converged"; break)
        end
    catch err
        status = "error: " * first(sprint(showerror, err), 160)
    end
    close(series)
    _, E1, _ = metric_invariants(flow)
    return (; umax, where, E1 = E1 / E0, Emax = Emax / E0, steps, wall, status, t = flow.time)
end

open(joinpath(OUT, "impact_$(TAG)_n$(N).txt"), "w") do io
    r = impact(N)
    s = @sprintf("impact n=%d %-14s t=%.4f steps=%d wall=%.0fs max|u|=%.3e [%s] E_end/E0=%.4f maxE/E0=%.4f %s",
        N, TAG, r.t, r.steps, r.wall, r.umax, r.where, r.E1, r.Emax, r.status)
    println(io, s); println(s)
end
