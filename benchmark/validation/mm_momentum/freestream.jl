# Free-stream preservation: uniform U in both phases carrying a drop or an
# oblique band through the grid, at several CFL numbers (fraction of the
# full-cell VOF bound), both density ratios, tiny cut volumes (cell fraction
# floor sqrt(eps) and 1e-12). Max |u - U|/|U| after the transport stage and
# after the full step; 𝓜-momentum change of the run relative to ρV|U|.
include(joinpath(@__DIR__, "common.jl"))

const R = 0.2
"Drop whose top enters the next cell row by a cap of fraction ≈ `cap`."
function drop_levelset(n, cap)
    h = 1 / n
    eta = (cap * h^2 * 3 / (4 * sqrt(2R)))^(2 / 3)
    yc = round((0.5 + R) * n) / n - R + eta
    xc = 0.5 + 0.123h
    return (x, y) -> hypot(x - xc, y - yc) - R
end
"Band between the lines x + 2y = c ± w; one line a distance `cap` h above a grid vertex."
function band_levelset(n, cap)
    h = 1 / n
    p = (1.0, 2.0)
    w = 0.17
    c = 1.5 - w + cap * h * hypot(p...)
    return (x, y) -> (abs(mod(p[1] * x + p[2] * y - c + 0.5, 1) - 0.5) - w) / hypot(p...)
end

const VARIANTS = isempty(ARGS) ? ((:D0, false), (:DI, false), (:DP, true), (:DG, true), (:DS, false), (:DF, false)) :
    ((:D0, false), (:DI, false), (:DP, true))
const FS_CFLS = isempty(ARGS) ? (0.1, 0.45, 0.9) : Tuple(parse.(Float64, ARGS))

function run_case(design, match, shape, density, U, cfl; n = 32, steps = 30, cap = 1.0e-6, floor = sqrt(eps()))
    set_design!(design)
    TP._MIMETIC_MASS_MATCH[] = match
    levelset = shape === :drop ? drop_levelset(n, cap) : band_levelset(n, cap)
    shared = (x, y, t, axis) -> U[axis]
    flow, interface = make_flow(levelset; n, density, velocity = (shared, shared), fraction_floor = floor)
    mesh = interface.mesh
    fmin = min_fraction(interface)
    after_transport = after_step = 0.0
    P0, _, W0 = metric_invariants(flow)
    status = "ok"
    try
        for _ in 1:steps
            dt = stable_timestep(flow.velocity, flow.system, flow.workspace; cfl)
            do_step!(flow, dt)
            after_transport = max(after_transport, constant_deviation(state_of(flow).convected, mesh, U))
            after_step = max(after_step, constant_deviation(flow.velocity, mesh, U))
            fmin = min(fmin, min_fraction(interface))
        end
    catch err
        status = "error: " * first(sprint(showerror, err), 120)
    end
    P1, _, _ = metric_invariants(flow)
    rel = maximum(abs, (P1[5] - P0[5], P1[6] - P0[6])) / (abs(W0[1]) + abs(W0[2]))
    set_design!(:D0)
    TP._MIMETIC_MASS_MATCH[] = false
    return (; after_transport, after_step, fmin, rel, status)
end

open(joinpath(OUT, isempty(ARGS) ? "freestream.txt" : "freestream_extra.txt"), "w") do io
    for shape in (:drop, :band), density in ((1.0, 1.0), (1.0, 1.0e-3)), U in ((1.0, 0.37), (0.0, 1.0)),
            cfl in FS_CFLS, floor in (sqrt(eps()), 1.0e-12)
        shape === :band && floor != sqrt(eps()) && continue
        for (design, match) in VARIANTS
            r = run_case(design, match, shape, density, U, cfl; floor)
            s = @sprintf("%-5s rho_g=%-6.0e U=%-11s cfl=%.2f floor=%.0e %-3s%-3s transport=%.2e step=%.2e dPmix/(ρV|U|)=%.2e minC=%.1e %s",
                shape, density[2], U, cfl, floor, design, match ? "+mm" : "", r.after_transport, r.after_step, r.rel, r.fmin, r.status)
            println(io, s); flush(io); println(s); flush(stdout)
        end
    end
end
