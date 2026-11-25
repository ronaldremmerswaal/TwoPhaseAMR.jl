# Tier 3: drop impact (local only, about two minutes for all methods). The V = 2
# "similar" drop/pool impact of the MM impact studies: a drop of radius 0.08
# falling at V = 2 onto a pool of depth 0.3 in the unit box, ρ = (1000, 1),
# μ = (0.32, 0.0064), σ = 6.4 (CGFM), g = 9.81, free-slip walls, interphase
# drag, t = 0.12, `simulate!` with the package step controller (CFL 0.45).
# Members 1–5 shift the drop and the pool by fractions of a cell of the
# n = 32 mesh. Inviscid impact is singular, so the maxima are compared with a
# loose tolerance; they measure the small-phase-value problem of MM (gas
# values whose aperture closes reach tens of V) against CMDGFM and the
# one-velocity model.

const IMPACT_R = 0.08
const IMPACT_V = 2.0
const IMPACT_DENSITY = (1000.0, 1.0)
const IMPACT_VISCOSITY = (0.32, 0.0064)
const IMPACT_SIGMA = 6.4
const IMPACT_GRAVITY = (0.0, -9.81)
const IMPACT_SHIFTS = ((0.0, 0.0), (0.5, 0.25), (0.25, 0.5), (0.13, 0.77), (0.37, 0.11))
const IMPACT_TIME = 0.12
# Gas values with a phase aperture up to this are "small"; the bottleneck
# apertures a ≈ √(ρ_g/ρ_l) ≈ 0.03 of the MM spikes lie in this band.
const IMPACT_SMALL_APERTURE = 0.05

function impact_flow(m::Method, n, member)
    sx, sy = IMPACT_SHIFTS[member] ./ 32
    xc, H = 0.5 + sx, 0.3 + sy
    yc = H + 0.04 + IMPACT_R
    mesh = BlockTreeMesh(; base_mesh = (n, n), block_size = (8, 8), periodic = (false, false))
    interface = InterfaceField(mesh, (x, y) -> min(hypot(x - xc, y - yc) - IMPACT_R, y - H); lipschitz = 1)
    physics = TwoPhase(
        interface; density = IMPACT_DENSITY, dynamic_viscosity = IMPACT_VISCOSITY,
        surface_tension = SurfaceTension(IMPACT_SIGMA; method = CGFM()), gravity = IMPACT_GRAVITY,
    )
    drop = (x, y, t, axis) -> axis == 2 && hypot(x - xc, y - yc) < IMPACT_R + 0.02 ? -IMPACT_V : 0.0
    rest = (x, y, t, axis) -> 0.0
    flow = FlowSimulation(
        mesh, physics; velocity_model = velocity_model(m), velocity = two_velocity(m) ? (drop, rest) : drop,
        boundary = FreeSlip(),
    )
    checked(project!(flow))
    return flow
end

"""Maxima over the run. A value is present where its phase has staggered
volume (the single field of the one-velocity model: where the phase has
fraction); it is resolved where its phase fraction is at least ½ and its
aperture positive, and its class is slaved (aperture 0), small
(0 < a ≤ 0.05) or regular by the aperture of its own phase."""
mutable struct ImpactObserver <: AbstractObserver
    steps::Int
    min_dt::Float64
    raw::Vector{Float64}                 # liquid, gas
    resolved::Vector{Float64}
    gas_class::Vector{Float64}           # slaved, small, regular
    peak::NamedTuple{(:u, :aperture, :fraction, :time), NTuple{4, Float64}}
    slaved_values::Int
    small_values::Int
    fast_small_steps::Int
end

ImpactObserver() = ImpactObserver(
    0, Inf, zeros(2), zeros(2), zeros(3), (; u = 0.0, aperture = NaN, fraction = NaN, time = NaN), 0, 0, 0,
)

function TwoPhaseAMR.observe!(o::ImpactObserver, event::Symbol, state, info)
    event === :step || return nothing
    o.steps += 1
    o.min_dt = min(o.min_dt, Float64(info.dt))
    interface, velocity = state.interface, state.velocity
    apertures = TP._current_phase_face_apertures!(interface)
    fast_small = false
    for (i, face) in pairs(interface.mesh.faces)
        alpha = TP._staggered_liquid_fraction(interface, nothing, face)
        for phase in 1:2
            fraction = phase == 1 ? alpha : 1 - alpha
            u = velocity isa FaceVelocity ? velocity.interior[i] : velocity[phase].interior[i]
            isfinite(u) && fraction > 0 || continue
            speed, aperture = abs(u), apertures[phase][i]
            o.raw[phase] = max(o.raw[phase], speed)
            fraction >= 0.5 && aperture > 0 && (o.resolved[phase] = max(o.resolved[phase], speed))
            phase == 2 || continue
            class = iszero(aperture) ? 1 : aperture <= IMPACT_SMALL_APERTURE ? 2 : 3
            o.gas_class[class] = max(o.gas_class[class], speed)
            class == 1 && (o.slaved_values += 1)
            class == 2 && (o.small_values += 1)
            class < 3 && speed > 5IMPACT_V && (fast_small = true)
            speed > o.peak.u && (o.peak = (; u = speed, aperture, fraction, time = Float64(state.time)))
        end
    end
    o.fast_small_steps += fast_small
    return nothing
end

METRIC_DESCRIPTIONS["impact/gas_raw"] = "largest gas value over the run"
METRIC_DESCRIPTIONS["impact/gas_resolved"] = "largest gas value with gas fraction ≥ ½ and aperture > 0"
METRIC_DESCRIPTIONS["impact/liquid_raw"] = "largest liquid value over the run"
METRIC_DESCRIPTIONS["impact/liquid_resolved"] = "largest liquid value with liquid fraction ≥ ½ and aperture > 0"
METRIC_DESCRIPTIONS["impact/gas_slaved"] = "largest gas value without aperture (a = 0)"
METRIC_DESCRIPTIONS["impact/gas_small"] = "largest gas value with 0 < a ≤ 0.05 (small and bottleneck values)"
METRIC_DESCRIPTIONS["impact/gas_regular"] = "largest gas value with a > 0.05"
METRIC_DESCRIPTIONS["impact/peak_aperture"] = "gas aperture of the value of the raw gas peak"
METRIC_DESCRIPTIONS["impact/peak_fraction"] = "gas fraction of the value of the raw gas peak"
METRIC_DESCRIPTIONS["impact/peak_time"] = "time of the raw gas peak"
METRIC_DESCRIPTIONS["impact/slaved_per_step"] = "mean number of gas values without aperture per step"
METRIC_DESCRIPTIONS["impact/small_per_step"] = "mean number of gas values with 0 < a ≤ 0.05 per step"
METRIC_DESCRIPTIONS["impact/fast_small_steps"] = "steps in which a slaved or small gas value exceeds 5V"
METRIC_DESCRIPTIONS["impact/steps"] = "time steps of `simulate!`"
METRIC_DESCRIPTIONS["impact/min_dt"] = "smallest time step"
METRIC_DESCRIPTIONS["impact/ensemble_gas_raw_median"] = "median over the n = 32 members of the raw gas maximum"
METRIC_DESCRIPTIONS["impact/ensemble_gas_resolved_median"] = "median over the n = 32 members of the resolved gas maximum"

median_of(values) = (s = sort(values); isodd(length(s)) ? s[(end + 1) ÷ 2] : (s[end ÷ 2] + s[end ÷ 2 + 1]) / 2)

const IMPACT_CASES = [(32, 1), (32, 2), (32, 3), (32, 4), (32, 5), (64, 1)]

function run_tier3!(suite; cases = IMPACT_CASES)
    for m in suite.methods
        println("Tier 3, ", m.label)
        flush(stdout)
        ensemble = Dict("gas_raw" => Float64[], "gas_resolved" => Float64[])
        for (n, member) in cases
            case_name = "V=2 member $member"
            guarded(suite, 3, "impact", case_name, n, m) do
                flow = impact_flow(m, n, member)
                o = ImpactObserver()
                simulate!(flow, IMPACT_TIME; cfl = 0.45, observer = o)
                rec(name, value; goal = :min) = record!(suite, 3, "impact", case_name, n, m, name, value; goal)
                rec("gas_raw", o.raw[2])
                rec("gas_resolved", o.resolved[2])
                rec("liquid_raw", o.raw[1])
                rec("liquid_resolved", o.resolved[1])
                rec("gas_slaved", o.gas_class[1])
                rec("gas_small", o.gas_class[2])
                rec("gas_regular", o.gas_class[3])
                rec("peak_aperture", o.peak.aperture; goal = :info)
                rec("peak_fraction", o.peak.fraction; goal = :info)
                rec("peak_time", o.peak.time; goal = :info)
                rec("slaved_per_step", o.slaved_values / o.steps; goal = :info)
                rec("small_per_step", o.small_values / o.steps; goal = :info)
                rec("fast_small_steps", o.fast_small_steps; goal = :info)
                rec("steps", o.steps; goal = :info)
                rec("min_dt", o.min_dt; goal = :info)
                if n == 32
                    push!(ensemble["gas_raw"], o.raw[2])
                    push!(ensemble["gas_resolved"], o.resolved[2])
                end
            end
        end
        if length(ensemble["gas_raw"]) == 5
            for (name, values) in ensemble
                record!(suite, 3, "impact", "V=2 members 1-5", 32, m, "ensemble_$(name)_median", median_of(values))
            end
        end
    end
    return suite
end
