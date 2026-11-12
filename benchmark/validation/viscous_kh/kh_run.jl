# One TwoPhaseAMR simulation of a viscous Kelvin–Helmholtz case (see kh_cases.jl).
#
# The interface and velocity are initialised with the t = 0 Orr–Sommerfeld eigenmode
# (amplitude |η̂(0)| = case.amplitude) on the composite error-function base flow. The
# staggered velocities are face-control-volume averages of the exact field:
#   one-velocity: mass-weighted mixture average  Σ_π ρ_π ∫_π u_π / Σ_π ρ_π |π|,
#   two-velocity: phase average ∫_π u_π / |π| over the part of the control volume
#                 occupied by phase π (flat interface at η(x) within the volume).
# The mesh is static: level L within |y| ≤ w of the interface, one level coarser per
# doubling of the distance (2:1 balanced), base 16 × 16 cells on (0,1) × (−1/2,1/2).
using TwoPhaseAMR
using Printf
include(joinpath(@__DIR__, "kh_cases.jl"))

const BASE_CELLS = 16
const RUNS = joinpath(@__DIR__, "results", "runs")
const FIELDS = joinpath(@__DIR__, "results", "fields")

function build_mesh(c::KHCase, n)
    L = round(Int, log2(n / BASE_CELLS))
    2^L * BASE_CELLS == n || throw(ArgumentError("n must be 16·2^L"))
    target(d) = d <= c.band ? L : max(0, L - ceil(Int, log2(d / c.band)))
    refine = (key, origin, extent) -> begin
        y0, y1 = origin[2], origin[2] + extent[2]
        d = y0 <= 0 <= y1 ? 0.0 : min(abs(y0), abs(y1))
        key.level < target(d)
    end
    H = c.height
    return BlockTreeMesh(;
        base_mesh = (BASE_CELLS, round(Int, 2H * BASE_CELLS)), block_size = (8, 8),
        origin = (0.0, -H), extent = (WAVELENGTH, 2H), periodic = (true, false),
        max_level = L, refine,
    )
end

"Nodal stream-function data of the eigenmode for fast evaluation of (φ, Dφ) per phase."
struct ModeProfile
    problem::TwoLayerProblem
    nodal::NTuple{2, NTuple{5, Vector{ComplexF64}}}
    eta::ComplexF64
end

function ModeProfile(p::TwoLayerProblem, X)
    nodal = ntuple(2) do phase
        lay = layer(p, phase)
        φ = X[layer_range(p, phase)]
        (φ, lay.D1 * φ, lay.D2 * φ, lay.D3 * φ, lay.D4 * φ)
    end
    return ModeProfile(p, nodal, X[end])
end

"(φ, Dφ) of layer `phase` at y (cubic Taylor extension across y = 0, only used for |y| ≤ |η̂|)."
function streamfunction(m::ModeProfile, phase, y)
    lay = layer(m.problem, phase)
    φ, Dφ, D2φ, D3φ, D4φ = m.nodal[phase]
    if lay.side * y >= 0
        ξ = _imap(min(abs(y), lay.H), lay.H, lay.stretch)
        return chebinterp(lay.ξ, φ, ξ), chebinterp(lay.ξ, Dφ, ξ)
    end
    return φ[end] + y * Dφ[end] + y^2 / 2 * D2φ[end] + y^3 / 6 * D3φ[end],
        Dφ[end] + y * D2φ[end] + y^2 / 2 * D3φ[end] + y^3 / 6 * D4φ[end]
end

"∫_a^b U∞ erf(y/δ) dy."
erf_integral(U∞, δ, a, b) = U∞ * (
    (b * erf(b / δ) + δ / sqrt(pi) * exp(-(b / δ)^2)) -
        (a * erf(a / δ) + δ / sqrt(pi) * exp(-(a / δ)^2))
)

const GAUSS4 = (
    (-0.8611363115940526, 0.34785484513745385), (-0.33998104358485626, 0.6521451548625461),
    (0.33998104358485626, 0.6521451548625461), (0.8611363115940526, 0.34785484513745385),
)

"∫_a^b of velocity component `axis` of phase `phase` at abscissa x (exact base, Gauss perturbation)."
function phase_integral(c::KHCase, m::ModeProfile, phase, axis, x, a, b, resolution)
    b > a || return 0.0
    k = WAVENUMBER
    e = cis(k * x)
    base = base_flow(c)
    total = axis == 1 ? erf_integral(base.U∞[phase], base.δ0[phase], a, b) : 0.0
    pieces = max(1, ceil(Int, (b - a) / resolution))
    w = (b - a) / pieces
    for q in 1:pieces, (s, g) in GAUSS4
        y = a + (q - 0.5) * w + s * w / 2
        φ, Dφ = streamfunction(m, phase, y)
        total += g * w / 2 * (axis == 1 ? real(Dφ * e) : real(-im * k * φ * e))
    end
    return total
end

"Point value of velocity component `axis` of phase `phase` at (x, y)."
function phase_point(c::KHCase, m::ModeProfile, phase, axis, x, y)
    φ, Dφ = streamfunction(m, phase, y)
    e = cis(WAVENUMBER * x)
    axis == 1 || return real(-im * WAVENUMBER * φ * e)
    base = base_flow(c)
    return base.U∞[phase] * erf(y / base.δ0[phase]) + real(Dφ * e)
end

"""
Initial staggered velocity (callback(s) `(x, y, t, axis)`) for `model`; `h` is the
interface cell size (control-volume height of faces near the interface).
"""
function initial_velocity(c::KHCase, m::ModeProfile, model, h)
    η(x) = real(m.eta * cis(WAVENUMBER * x))
    resolution = minimum(initial_thicknesses(c)) / 2
    parts(x, y) = (lo = y - h / 2; hi = y + h / 2; s = clamp(η(x), lo, hi); (lo, s, hi))
    if model isa OneVelocityModel
        ρ = c.density
        return (x, y, t, axis) -> begin
            lo, s, hi = parts(x, y)
            mom = ρ[1] * phase_integral(c, m, 1, axis, x, lo, s, resolution) +
                ρ[2] * phase_integral(c, m, 2, axis, x, s, hi, resolution)
            mom / (ρ[1] * (s - lo) + ρ[2] * (hi - s))
        end
    end
    tiny = 1.0e-9 * h
    function phase_velocity(phase)
        return (x, y, t, axis) -> begin
            lo, s, hi = parts(x, y)
            a, b = phase == 1 ? (lo, s) : (s, hi)
            if b - a > tiny
                return phase_integral(c, m, phase, axis, x, a, b, resolution) / (b - a)
            end
            # phase absent from this control volume: value at the interface (inactive
            # faces are overwritten with NaN by FlowSimulation)
            return phase_point(c, m, phase, axis, x, s)
        end
    end
    return (phase_velocity(1), phase_velocity(2))
end

"Complex Fourier coefficient η̂ (η = Re(η̂ e^{ikx})) from the liquid volume fractions."
function interface_mode(interface, h)
    mesh = interface.mesh
    k = WAVENUMBER
    s = zero(ComplexF64)
    nx, ny = mesh.block_size
    for (b, leaf) in pairs(mesh.blocks)
        dx, dy = leaf.spacing
        f = interior(interface.fraction, b)
        for j in 1:ny, i in 1:nx
            fij = f[i, j]
            iszero(fij) && continue
            x, _ = cellcenter(mesh, b, i, j)
            s += fij * dy * (cis(-k * (x - dx / 2)) - cis(-k * (x + dx / 2))) / (im * k)
        end
    end
    z = k * h / 2
    return 2 / WAVELENGTH * s / (sin(z) / z)^2
end

mutable struct ModeObserver <: AbstractObserver
    h::Float64
    times::Vector{Float64}
    eta::Vector{ComplexF64}
    last_print::Float64
    urange::Vector{Float64}       # extrema over the run of the finite x-velocities
    frange::Vector{Float64}       # extrema over the run of the volume fractions
    infinite::Int                 # number of ±Inf velocity entries seen
end
ModeObserver(h) = ModeObserver(h, Float64[], ComplexF64[], time(), [Inf, -Inf], [Inf, -Inf], 0)

function track_bounds!(o::ModeObserver, state)
    fields = state.velocity isa Tuple ? state.velocity : (state.velocity,)
    mesh = state.interface.mesh
    for f in fields, (i, face) in pairs(mesh.faces)
        face.axis == 1 || continue
        u = f.interior[i]
        isinf(u) && (o.infinite += 1)
        isfinite(u) || continue
        o.urange[1] = min(o.urange[1], u)
        o.urange[2] = max(o.urange[2], u)
    end
    for b in eachindex(mesh.blocks)
        fmin, fmax = extrema(interior(state.interface.fraction, b))
        o.frange[1] = min(o.frange[1], fmin)
        o.frange[2] = max(o.frange[2], fmax)
    end
    return nothing
end

function TwoPhaseAMR.observe!(o::ModeObserver, event::Symbol, state, info)
    (event === :initial || event === :step) || return nothing
    push!(o.times, state.time)
    push!(o.eta, interface_mode(state.interface, o.h))
    track_bounds!(o, state)
    if time() - o.last_print > 60
        o.last_print = time()
        @printf("    t = %.4f  |η̂| = %.4e\n", state.time, abs(o.eta[end]))
        flush(stdout)
    end
    return nothing
end

"Rows of x-faces: (y, mean u) per phase field; NaN entries (absent phase) are skipped."
function row_profiles(flow, fields)
    mesh = flow.system.mesh
    rows = Dict{Float64, Vector{Vector{Float64}}}()
    for (i, face) in pairs(mesh.faces)
        face.axis == 1 || continue
        y = round(face.center[2]; digits = 12)
        acc = get!(() -> [zeros(2) for _ in fields], rows, y)
        for (q, f) in enumerate(fields)
            u = f.interior[i]
            isfinite(u) || continue
            acc[q][1] += face.area * u
            acc[q][2] += face.area
        end
    end
    ys = sort(collect(keys(rows)))
    return ys, [[rows[y][q][2] > 0 ? rows[y][q][1] / rows[y][q][2] : NaN for y in ys] for q in eachindex(fields)]
end

"x-face velocities near the interface: (x, y, face length = local h, u per field)."
function band_field(flow, fields; halfwidth = 0.1)
    mesh = flow.system.mesh
    out = Vector{Vector{Float64}}()
    for (i, face) in pairs(mesh.faces)
        face.axis == 1 && abs(face.center[2]) <= halfwidth || continue
        push!(out, [face.center[1], face.center[2], face.area, (f.interior[i] for f in fields)...])
    end
    return out
end

model_label(model) = model isa OneVelocityModel ? "one" : "two"

"""
    set_ctu_threshold!(flow, θ)

Replace the WY advection workspace of a two-phase flow by one with `ctu_threshold = θ`
(the threshold is not reachable from the public `FlowSimulation` options; the
workspace is rebuilt from the public `WYAdvectionWorkspace` constructor and stored in
the mutable flow workspaces).
"""
function set_ctu_threshold!(flow, θ)
    two_phase = flow.workspace.two_phase
    advection = WYAdvectionWorkspace(flow.interface; ctu_threshold = θ)
    two_phase.advection = advection
    state = two_phase.two_velocity[]
    isnothing(state) || (state.advection = advection)
    return flow
end

"""
    run_case(c, model, n; save_field=false, tag="", scheme=LaxWendroff(),
             ctu_threshold=0.5, runs=RUNS, skip_existing=false) -> NamedTuple

Simulate case `c` on the mesh with interface resolution λ/h = n and write
<runs>/<case>_<model>_n<n><tag>.csv (time series) and _profile.csv
(x-averaged tangential velocity at the final time); with `save_field` also the
x-face velocities near the interface to results/fields/. `scheme` is the bulk
staggered momentum flux, `ctu_threshold` the phase fraction below which the CTU
(upwind) flux is used.
"""
function run_case(
        c::KHCase, model, n; save_field = false, tag = "", scheme = LaxWendroff(),
        ctu_threshold = 0.5, runs = RUNS, skip_existing = false
    )
    label = "$(c.name)_$(model_label(model))_n$(n)$(tag)"
    if skip_existing && isfile(joinpath(runs, label * ".csv")) &&
            (!save_field || isfile(joinpath(FIELDS, label * "_field.csv")))
        println("  skip $label (exists)")
        return nothing
    end
    mkpath(runs)
    h = WAVELENGTH / n
    p, ω0, X = initial_mode(c)
    mode = ModeProfile(p, X)
    mesh = build_mesh(c, n)
    A0 = c.amplitude
    interface = InterfaceField(mesh, (x, y) -> y - A0 * cos(WAVENUMBER * x); lipschitz = 1)
    physics = TwoPhase(
        interface; density = c.density, dynamic_viscosity = dynamic_viscosities(c),
        surface_tension = surface_tension(c),
    )
    flow = FlowSimulation(
        mesh, physics; velocity_model = model,
        velocity = initial_velocity(c, mode, model, h), boundary = FreeSlip(),
        numerics = NavierStokesOptions(; advection = AdvectionOptions(; scheme)),
    )
    ctu_threshold == 0.5 || set_ctu_threshold!(flow, ctu_threshold)
    observer = ModeObserver(h)
    cells = ncells(mesh)
    @printf("  run %s: %d cells, h = 1/%d\n", label, cells, n)
    flush(stdout)
    failure = ""
    wall = @elapsed result = try
        simulate!(flow, c.final_time; observer)
    catch err
        failure = sprint(showerror, err)[1:min(end, 200)]
        @printf("  FAILED %s: %s\n", label, failure)
        (; steps = length(observer.times) - 1)
    end
    fields = model isa OneVelocityModel ? (flow.velocity,) : flow.velocity
    ys, profiles = row_profiles(flow, fields)
    open(joinpath(runs, label * ".csv"), "w") do io
        @printf(
            io, "# case=%s model=%s n=%d cells=%d steps=%d wall=%.3f umin=%.6g umax=%.6g fmin=%.6g fmax=%.6g infinite=%d failed=%s\n",
            c.name, model_label(model), n, cells, result.steps, wall, observer.urange..., observer.frange...,
            observer.infinite, isempty(failure) ? "no" : "yes"
        )
        isempty(failure) || println(io, "# failure: ", replace(failure, '\n' => ' '))
        println(io, "t,eta_re,eta_im")
        for (t, e) in zip(observer.times, observer.eta)
            @printf(io, "%.8f,%.12e,%.12e\n", t, real(e), imag(e))
        end
    end
    open(joinpath(runs, label * "_profile.csv"), "w") do io
        println(io, model isa OneVelocityModel ? "y,u" : "y,u_liquid,u_gas")
        for (r, y) in enumerate(ys)
            println(io, join((y, (pr[r] for pr in profiles)...), ","))
        end
    end
    if save_field
        mkpath(FIELDS)
        open(joinpath(FIELDS, label * "_field.csv"), "w") do io
            println(io, model isa OneVelocityModel ? "x,y,h,u" : "x,y,h,u_liquid,u_gas")
            for row in band_field(flow, fields)
                println(io, join(row, ","))
            end
        end
    end
    @printf("  done %s: %d steps, %.1f s, |η̂(T)| = %.5e\n", label, result.steps, wall, abs(observer.eta[end]))
    flush(stdout)
    return (; label, cells, steps = result.steps, wall)
end

"Copy of case `c` with some fields replaced."
modify(c::KHCase; kw...) = KHCase(; (f => get(kw, f, getfield(c, f)) for f in fieldnames(KHCase))...)

# Command line:
#   julia --project=. kh_run.jl <case[,case…]> <one|two|both> <n[,n…]> [field] [uniform] [amp=<A0>] [skip]
# `uniform`: finest level everywhere (checks the graded mesh); `amp=`: initial amplitude
# (checks linearity); both add a tag to the output names. `skip`: do not redo runs
# whose output exists.
if abspath(PROGRAM_FILE) == @__FILE__
    names = split(ARGS[1], ',')
    models = ARGS[2] == "both" ? (OneVelocityModel(), TwoVelocityModel()) :
        ARGS[2] == "one" ? (OneVelocityModel(),) : (TwoVelocityModel(),)
    resolutions = parse.(Int, split(ARGS[3], ','))
    flags = ARGS[4:end]
    save_field = "field" in flags
    skip_existing = "skip" in flags
    for name in names, n in resolutions, model in models
        c = CASES[name]
        tag = ""
        if "uniform" in flags
            c = modify(c; band = Inf)
            tag *= "_uniform"
        end
        for f in flags
            startswith(f, "amp=") || continue
            c = modify(c; amplitude = parse(Float64, f[5:end]))
            tag *= "_" * f
        end
        run_case(c, model, n; save_field, tag, skip_existing)
    end
end
