# Time continuity of the phase apertures along a translating interface: a drop
# (R = 0.2) and an oblique band carried by U = (1, 0.37) (no surface tension,
# rho = (1, 1e-3)), n = 32, 60 steps at CFL 0.45. Per step: births and deaths of
# aperture-active values (a: 0 <-> > 0), the aperture just after a birth and
# just before a death, values with 0 < a < 1e-6, the largest |Δa| of a face
# that stays active, and the changes of the small-tie status (a crossing 1/2).
#   MM_APERTURE=current|plain julia --project=<wt> transitions.jl
include(joinpath(@__DIR__, "common.jl"))

function history(levelset; n = 32, steps = 60)
    U = (1.0, 0.37)
    shared = (x, y, t, axis) -> U[axis]
    flow, interface = make_flow(levelset; n, density = (1.0, 1.0e-3), velocity = (shared, shared))
    la, ga = TP._current_phase_face_apertures!(interface)
    previous = (copy(la), copy(ga))
    stats = (births = Int[], deaths = Int[], birth_a = Float64[], death_a = Float64[], tiny = Int[],
        jump = Float64[], tie_switches = Int[], volume_births = Int[], volume_deaths = Int[])
    fraction(face) = TP._staggered_liquid_fraction(interface, nothing, face)
    vprev = [fraction(f) for f in interface.mesh.faces]
    for _ in 1:steps
        dt = stable_timestep(flow.velocity, flow.system, flow.workspace; cfl = 0.45)
        do_step!(flow, dt)
        la, ga = TP._current_phase_face_apertures!(interface)
        b = d = tiny = ties = vb = vd = 0
        jump = 0.0
        for phase in 1:2
            new = phase == 1 ? la : ga
            old = previous[phase]
            for i in eachindex(new)
                if old[i] == 0 && new[i] > 0
                    b += 1; push!(stats.birth_a, new[i])
                elseif old[i] > 0 && new[i] == 0
                    d += 1; push!(stats.death_a, old[i])
                elseif old[i] > 0 && new[i] > 0
                    jump = max(jump, abs(new[i] - old[i]))
                end
                0 < new[i] < 1.0e-6 && (tiny += 1)
                (old[i] < 0.5) != (new[i] < 0.5) && old[i] > 0 && new[i] > 0 && (ties += 1)
            end
        end
        for (i, face) in pairs(interface.mesh.faces)
            v = fraction(face)
            for (o, w) in ((vprev[i], v), (1 - vprev[i], 1 - v))
                o == 0 && w > 0 && (vb += 1)
                o > 0 && w == 0 && (vd += 1)
            end
            vprev[i] = v
        end
        previous = (copy(la), copy(ga))
        push!(stats.births, b); push!(stats.deaths, d); push!(stats.tiny, tiny); push!(stats.jump, jump)
        push!(stats.tie_switches, ties); push!(stats.volume_births, vb); push!(stats.volume_deaths, vd)
    end
    return stats
end

q(v, p) = isempty(v) ? NaN : sort(v)[clamp(round(Int, p * length(v)), 1, length(v))]

open(joinpath(OUT, "transitions" * SUFFIX * ".txt"), "w") do io
    for (name, ls) in (
            ("drop", (x, y) -> hypot(x - 0.52, y - 0.47) - 0.2),
            ("band", (x, y) -> (abs(mod(x + 2y - 1.33 + 0.5, 1) - 0.5) - 0.17) / sqrt(5)),
        )
        s = history(ls)
        line = @sprintf("%-4s aperture=%-7s births/step %.2f (max %d) deaths/step %.2f (max %d) | a at birth median %.2e min %.2e max %.2e | a at death median %.2e min %.2e max %.2e | values 0<a<1e-6 per step %.2f (max %d) | max|Δa| of surviving values %.3f | tie switches/step %.2f | volume births/deaths per step %.2f/%.2f",
            name, PLAIN ? "plain" : "current", mean(s.births), maximum(s.births), mean(s.deaths), maximum(s.deaths),
            q(s.birth_a, 0.5), q(s.birth_a, 0.0), q(s.birth_a, 1.0), q(s.death_a, 0.5), q(s.death_a, 0.0), q(s.death_a, 1.0),
            mean(s.tiny), maximum(s.tiny), maximum(s.jump), mean(s.tie_switches), mean(s.volume_births), mean(s.volume_deaths))
        println(io, line); flush(io); println(line)
    end
end
