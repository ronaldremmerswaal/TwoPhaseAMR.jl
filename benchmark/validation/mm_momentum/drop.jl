# Static drop (test/integration/static_droplet_stability.jl, MimeticProjection)
# and the same drop translating with U in a periodic box (Galilean invariance:
# spurious currents measured relative to U), per transport design.
#   julia --project=<wt> drop.jl <design> [match]
include(joinpath(@__DIR__, "common.jl"))
using Test
const TESTS = joinpath(WT, "test", "integration")
let
    src = read(joinpath(TESTS, "static_droplet_stability.jl"), String)
    include_string(Main, src[1:(first(findfirst("@testset \"Static-droplet stability\"", src)) - 1)])
end
const DESIGN = Symbol(ARGS[1])
const MATCH = length(ARGS) >= 2 && ARGS[2] == "match"
set_design!(DESIGN)
TP._MIMETIC_MASS_MATCH[] = MATCH
const TAG = "$(DESIGN)$(MATCH ? "+mm" : "")"

"Max |u_π - U| on mixed faces with phase aperture above the floor of the test."
function relative_speed(velocity, interface, U)
    la, ga = TP._current_phase_face_apertures!(interface)
    best = 0.0
    for (index, face) in pairs(interface.mesh.faces)
        c = (TP._conditional_fraction(interface, face.minus) + TP._conditional_fraction(interface, face.plus)) / 2
        0 < c < 1 || continue
        for (phase, a) in ((1, la[index]), (2, ga[index]))
            a > STATIC_DROP_PHASE_FLOOR || continue
            u = velocity[phase].interior[index]
            isfinite(u) && (best = max(best, abs(u - U[face.axis])))
        end
    end
    return best
end

function translating(ratio, n, U; laplace = STATIC_DROP_LAPLACE, final_time = 0.4)
    density = _static_drop_density(ratio)
    shared = (x, y, t, axis) -> U[axis]
    flow, interface = make_flow((x, y) -> hypot(x - 0.5, y - 0.5) - STATIC_DROP_RADIUS; n, density,
        sigma = STATIC_DROP_SIGMA, viscosity = _static_drop_viscosity(ratio, laplace), velocity = (shared, shared))
    early = late = 0.0
    P0, E0, W0 = metric_invariants(flow)
    steps = 0
    status = "ok"
    wall = @elapsed try
        while flow.time < final_time - 1.0e-12
            dt = min(stable_timestep(flow.velocity, flow.system, flow.workspace; cfl = 0.45), final_time - flow.time)
            do_step!(flow, dt)
            steps += 1
            s = relative_speed(flow.velocity, interface, U)
            flow.time <= final_time / 2 ? (early = max(early, s)) : (late = max(late, s))
        end
    catch err
        status = "error: " * first(sprint(showerror, err), 120)
    end
    P1, _, _ = metric_invariants(flow)
    dP = maximum(abs, (P1[5] - P0[5], P1[6] - P0[6])) / max(abs(W0[1]) + abs(W0[2]), 1.0e-300)
    return (; early, late, dP, steps, wall, status)
end

open(joinpath(OUT, "drop_$(TAG)$(SUFFIX).txt"), "w") do io
    say(s) = (println(io, s); flush(io); println(s); flush(stdout))
    for ratio in (1.0, 1000.0), n in (16, 32), laplace in (Inf, STATIC_DROP_LAPLACE)
        wall = @elapsed r = try
            static_droplet_case(:mimetic, ratio, n; laplace)
        catch err
            say("static DR=$ratio n=$n La=$laplace $TAG ERROR " * first(sprint(showerror, err), 160)); continue
        end
        say(@sprintf("static     DR=%-6g n=%2d La=%-6g %-6s liquid %.3e->%.3e gas %.3e->%.3e Ca=%.3e Elate/Eearly=%.3f peaks %.3e->%.3e steps=%d wall=%.1fs",
            ratio, n, laplace, TAG, r.liquid_early, r.liquid_late, r.gas_early, r.gas_late, r.capillary,
            r.energy_late / r.energy_early, r.first_peak, r.last_peak, r.steps, wall))
    end
    for ratio in (1.0, 1000.0), n in (16, 32), U in ((0.0, 0.0), (1.0, 0.37))
        r = translating(ratio, n, U)
        say(@sprintf("periodic   DR=%-6g n=%2d U=%-11s %-6s max|u-U| early=%.3e late=%.3e dP/(ρV|U|)=%.2e steps=%d wall=%.1fs %s",
            ratio, n, U, TAG, r.early, r.late, r.dP, r.steps, r.wall, r.status))
    end
end
