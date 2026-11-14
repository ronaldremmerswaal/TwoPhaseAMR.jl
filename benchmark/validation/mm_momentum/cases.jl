# Integration cases per transport design (MimeticProjection throughout):
#   julia --project=<wt> cases.jl <group> <design> [match]
# groups: kh, khdt, rt, waves, timing. Output: cases_<group>_<design>[_mm].txt
include(joinpath(@__DIR__, "common.jl"))
using Test
const TESTS = joinpath(WT, "test", "integration")
include(joinpath(TESTS, "interfacial_instability_problem.jl"))

const GROUP = ARGS[1]
const DESIGN = Symbol(ARGS[2])
const MATCH = length(ARGS) >= 3 && ARGS[3] == "match"
const LOGGING = GROUP != "timing"
set_design!(DESIGN)
TP._MIMETIC_MASS_MATCH[] = MATCH

mutable struct Tracker <: AbstractObserver
    flow::Any
    P0::NTuple{6, Float64}
    Pprev::NTuple{6, Float64}
    E0::Float64
    Eprev::Float64
    sum_dP::Vector{Float64}     # Σ_steps |ΔP| (mixture x, y)
    transport_dP::Vector{Float64}
    transport_dE::Float64       # Σ transport energy changes (signed)
    projection_dE::Float64
    other_dE::Float64
    W0::NTuple{2, Float64}
    scale::Float64              # Σ m |u| (x)
    umax::Float64
    jump::Float64
    corr::Float64
    defective::Int
    status::String
    solve::Float64
    band::Int
    steps::Int
    final::Any
end
Tracker(flow) = Tracker(flow, ntuple(_ -> NaN, 6), ntuple(_ -> NaN, 6), NaN, NaN, zeros(2), zeros(2), 0.0, 0.0, 0.0,
    (NaN, NaN), NaN, 0.0, 0.0, 0.0, 0, "ok", 0.0, 0, 0, nothing)

function scale_of(flow)
    s = 0.0
    interface = flow.system.two_phase.interface
    rho = flow.system.two_phase.density
    for (index, face) in pairs(flow.system.mesh.faces)
        face.axis == 1 || continue
        c = (TP._conditional_fraction(interface, face.minus) + TP._conditional_fraction(interface, face.plus)) / 2
        for (phase, m) in ((1, rho[1] * c), (2, rho[2] * (1 - c)))
            u = flow.velocity[phase].interior[index]
            isfinite(u) && (s += flow.system.mesh.face_dual_volumes[index] * m * abs(u))
        end
    end
    return s
end

function track!(t::Tracker, event, stats = nothing)
    LOGGING || return nothing
    if event === :initial
        P, E, W = metric_invariants(t.flow)
        t.P0 = t.Pprev = P
        t.E0 = t.Eprev = E
        t.W0 = W
        t.scale = scale_of(t.flow)
        TP._MIMETIC_TRANSPORT_LOG[] = Any[]
    elseif event === :step
        t.steps += 1
        stats isa MimeticProjectionStats && (t.defective = max(t.defective, stats.defective_cells))
        P, E, _ = metric_invariants(t.flow)
        isfinite(E) || (t.status = "non-finite energy")
        log = TP._MIMETIC_TRANSPORT_LOG[]
        if !isempty(log)
            l = log[end]
            t.transport_dP[1] += abs(l.momentum_after[5] - l.momentum_before[5])
            t.transport_dP[2] += abs(l.momentum_after[6] - l.momentum_before[6])
            t.transport_dE += l.energy_after - l.energy_before
            t.other_dE += (E - t.Eprev) - (l.energy_after - l.energy_before)
            t.corr = max(t.corr, l.correction)
            t.solve += l.solve_time + l.redistribution.time
            t.band = max(t.band, l.band)
            empty!(log)
        end
        t.sum_dP[1] += abs(P[5] - t.Pprev[5])
        t.sum_dP[2] += abs(P[6] - t.Pprev[6])
        t.Pprev = P
        t.Eprev = E
        for phase in 1:2, u in t.flow.velocity[phase].interior
            isfinite(u) && (t.umax = max(t.umax, abs(u)))
        end
    elseif event === :completed
        t.final = metric_invariants(t.flow)
        TP._MIMETIC_TRANSPORT_LOG[] = nothing
    end
    return nothing
end

function TwoPhaseAMR.observe!(t::Tracker, event::Symbol, state, info)
    event === :failed && (t.status = "failed")
    return track!(t, event, event === :step ? info.projection : nothing)
end

function tracked_instability(parameters; n, timestep_factor = 1.0, cfl = 0.35, tolerance = 1.0e-7, max_cycles = 3000, ctu = nothing)
    case = two_velocity_instability_setup(; cells_per_wavelength = n, parameters..., pressure_projection = MimeticProjection())
    target = _instability_target(case, nothing)
    flow = case.flow
    isnothing(ctu) || (flow.workspace.two_phase.advection = WYAdvectionWorkspace(flow.system.two_phase.interface; ctu_threshold = ctu))
    for phase in flow.velocity
        enforce_velocity_boundary!(phase, flow.system.boundary; time = flow.time, geometry = flow.system.geometry)
    end
    tracker = Tracker(flow)
    track!(tracker, :initial)
    wall = @elapsed while flow.time < target - 16eps(target)
        dt = min(timestep_factor * stable_timestep(flow; cfl), target - flow.time)
        stats = try
            step!(flow.velocity, flow.pressure, flow.system, dt, flow.workspace; time = flow.time, tolerance, max_cycles)
        catch err
            tracker.status = "error: " * first(sprint(showerror, err), 160)
            break
        end
        flow.time = min(target, flow.time + dt)
        flow.steps += 1
        record_diagnostics!(flow)
        track!(tracker, :step, stats)
        tracker.status == "ok" || break
        stats.converged || (tracker.status = "not converged at step $(flow.steps)"; break)
    end
    flow.time = min(flow.time, target)
    record_diagnostics!(flow; event = :completed)
    track!(tracker, :completed)
    real_history = diagnostic_history(flow, :amplitude_real)
    imaginary_history = diagnostic_history(flow, :amplitude_imaginary)
    times = real_history.times
    amplitudes = ComplexF64.(real_history.values .+ im .* imaginary_history.values)
    analytical = [instability_linear_amplitude(t, case.dispersion; initial_amplitude = case.initial_amplitude) for t in times]
    error = tracker.status == "ok" ? normalized_amplitude_error(times, amplitudes, analytical) : NaN
    return (; error, steps = flow.steps, wall, tracker)
end

# Two-phase waves: the test's wave_case with a Tracker observer.
let
    src = read(joinpath(TESTS, "two_phase_waves.jl"), String)
    prefix = src[1:(first(findfirst("\"\"\"Run a quarter-period", src)) - 1)]
    body = src[first(findfirst("function wave_case(", src)):(first(findfirst("@testset \"Two-phase wave integration\"", src)) - 1)]
    old = "    simulate!(\n        flow, final_time; dt = selected_dt,"
    @assert count(old, body) == 1
    body = replace(body, old => "    tracker = Tracker(flow)\n    track!(tracker, :initial)\n    simulate!(\n        flow, final_time; dt = selected_dt, observer = tracker,")
    old = "        steps = flow.steps,\n    )"
    @assert count(old, body) == 1
    body = replace(body, old => "        steps = flow.steps, tracker,\n    )")
    old = "    println(\n        \"wave model="
    @assert count(old, body) == 1
    body = replace(body, old => "    false && println(\n        \"wave model=")
    include_string(Main, replace(prefix, "include(joinpath(@__DIR__," => "include(joinpath(\"$(TESTS)\","))
    include_string(Main, body)
end

function line(name, n, factor, error, steps, wall, t::Tracker)
    LOGGING || return @sprintf("%-14s %-3s%s n=%3d dtf=%.3f err=%.5e steps=%d wall=%.2fs wall/step=%.2ems status=%s",
        name, DESIGN, MATCH ? "+mm" : "", n, factor, error, steps, wall, 1.0e3wall / max(steps, 1), t.status)
    P1, E1, _ = isnothing(t.final) ? (ntuple(_ -> NaN, 6), NaN, nothing) : t.final
    return @sprintf(
        "%-14s %-3s%s n=%3d dtf=%.3f err=%.5e steps=%d wall=%.1fs E0=%.5e E1=%.5e dE_transport=%+.3e dE_other=%+.3e dPx_end/S=%+.3e dPy_end/S=%+.3e sum|dPx|/S=%.3e sum|dPx_tr|/S=%.3e umax=%.3e corr=%.3e band=%d solve=%.2fs defective=%d status=%s",
        name, DESIGN, MATCH ? "+mm" : "", n, factor, error, steps, wall, t.E0, E1, t.transport_dE, t.other_dE,
        (P1[5] - t.P0[5]) / t.scale, (P1[6] - t.P0[6]) / t.scale, t.sum_dP[1] / t.scale, t.transport_dP[1] / t.scale,
        t.umax, t.corr, t.band, t.solve, t.defective, t.status,
    )
end

function report(io, name, parameters, n; timestep_factor = 1.0, ctu = nothing)
    r = tracked_instability(parameters; n, timestep_factor, ctu)
    s = line(name, n, timestep_factor, r.error, r.steps, r.wall, r.tracker)
    println(io, s); flush(io); println(s); flush(stdout)
    return r
end

function report_wave(io, density_ratio, n; timestep_factor = 1.0)
    r = nothing
    failures = ""
    wall = @elapsed try
        @testset "wave" begin
            r = wave_case(
                :uniform, n, density_ratio, :capillary; formulation = CGFM(),
                velocity_model = :two_velocity, two_velocity_projection = MimeticProjection(), timestep_factor,
            )
        end
    catch err
        failures = " TESTFAIL: " * first(replace(sprint(showerror, err), '\n' => ' '), 160)
    end
    isnothing(r) && (println(io, "wave DR=$density_ratio n=$n $DESIGN ERROR$failures"); return nothing)
    s = line("wave DR=$(Int(density_ratio))", n, timestep_factor, r.error, r.steps, wall, r.tracker) * failures
    println(io, s); flush(io); println(s); flush(stdout)
    return r
end

kh(d, w) = (; density_ratio = d, weber = w, bond = 0.0, liquid_above = false)
rt(d, b) = (; density_ratio = d, weber = 0.0, bond = b, liquid_above = true)
const COMBINED = (density_ratio = 3.0, weber = 0.5, bond = 0.75, mean_interface_velocity = 0.1, liquid_above = true)

open(joinpath(OUT, "cases_$(GROUP)_$(DESIGN)$(MATCH ? "_mm" : "").txt"), "w") do io
    if GROUP == "kh"
        for n in (16, 32)
            report(io, "KH1/0.5", kh(1.0, 0.5), n)
            report(io, "KH1000/0.5", kh(1000.0, 0.5), n; timestep_factor = 0.25)
        end
    elseif GROUP == "khdt"
        # Energy versus dt at fixed grid, Lax-Wendroff without the CTU switch.
        for f in (1.0, 0.5, 0.25, 0.125)
            report(io, "KH1/0.5 noCTU", kh(1.0, 0.5), 16; timestep_factor = f, ctu = floatmin())
        end
        for f in (1.0, 0.5, 0.25)
            report(io, "KH1/0.5", kh(1.0, 0.5), 16; timestep_factor = f)
        end
    elseif GROUP == "rt"
        report(io, "RT3/0.5", rt(3.0, 0.5), 32)
        report(io, "RT1000/0.5", rt(1000.0, 0.5), 32)
        report(io, "combined", COMBINED, 32)
    elseif GROUP == "waves"
        for dr in (1.0, 1000.0), n in (16, 32, 64)
            report_wave(io, dr, n)
        end
    elseif GROUP == "timing"
        report(io, "RT1000/0.5", rt(1000.0, 0.5), 32)
        report(io, "KH1/0.5", kh(1.0, 0.5), 32)
        report_wave(io, 1000.0, 32)
        report_wave(io, 1.0, 64)
    end
end
