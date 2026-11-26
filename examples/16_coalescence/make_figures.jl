using TwoPhaseAMR, CairoMakie, Printf

Base.include(@__MODULE__, joinpath(@__DIR__, "coalescence.jl"))

# Inviscid bridge law r_b = C (σR/ρ)^(1/4) t^(1/2), with R = σ = ρ = 1:
# C = √(2·10/7.6) from the reconnection constants of Duchemin, Eggers &
# Josserand (2003); experiments and simulations give C ≈ 1.5 (Eggers,
# Sprittles & Snoeijer 2025).
const DUCHEMIN_PREFACTOR = sqrt(20 / 7.6)
const MEASURED_PREFACTOR = 1.5
# Fit window in r_b: after the fillet, which starts at rest, has accelerated,
# and before the bridge reaches the drop scale.
const FIT_WINDOW = (0.2, 0.45)
const BRIDGE = 0.15
const BRIDGE_TIME = 0.1

function linear_fit(x, y)
    mx, my = sum(x) / length(x), sum(y) / length(y)
    slope = sum((x .- mx) .* (y .- my)) / sum(abs2, x .- mx)
    return my - slope * mx, slope
end

# Fits over FIT_WINDOW: `r_b = A (t + t0)^β` in log-log, with the virtual
# origin `t0 = (r0/1.62)²` of a law through the initial bridge, and
# `r_b² = C²(t - t_v)` with the exponent fixed at 1/2.
function bridge_fits(t, r)
    keep = (FIT_WINDOW[1] .<= r .<= FIT_WINDOW[2]) .& (t .> 0)
    shift = (BRIDGE / DUCHEMIN_PREFACTOR)^2
    a, exponent = linear_fit(log.(t[keep] .+ shift), log.(r[keep]))
    b, slope = linear_fit(t[keep], r[keep] .^ 2)
    return (;
        exponent, amplitude = exp(a), prefactor = sqrt(slope), origin = -b / slope,
        shift, window = extrema(t[keep]),
    )
end

# Local prefactor sqrt(d(r_b²)/dt): least-squares slopes over sliding windows
# of `width` in time.
function local_prefactor(t, r; width = 0.01)
    radii = Float64[]
    values = Float64[]
    for start in range(t[2], last(t) - width; step = width / 4)
        keep = start .<= t .<= start + width
        count(keep) > 4 || continue
        _, slope = linear_fit(t[keep], r[keep] .^ 2)
        push!(radii, sum(r[keep]) / count(keep))
        push!(values, sqrt(max(slope, 0)))
    end
    return (; r = radii, prefactor = values)
end

# The bridge phase `t <= BRIDGE_TIME` of a run whose neck is at `level`.
function bridge_summary(flow, level)
    history = diagnostic_history(flow, :bridge_half_width)
    keep = history.times .<= BRIDGE_TIME * (1 + 1.0e-9)
    t, r = history.times[keep], history.values[keep]
    surface = diagnostic_history(flow, :surface_energy).values[keep]
    fits = bridge_fits(t, r)
    @printf(
        "  level %d (h = R/%d): fit window t = [%.4f, %.4f], r_b = %.4f (t + %.5f)^%.4f\n",
        level, 4 * 2^level, fits.window..., fits.amplitude, fits.shift, fits.exponent,
    )
    @printf(
        "  exponent 1/2: C = %.4f, origin %.5f; r_b(%.3f) = %.4f; E_s %.5f -> %.5f\n",
        fits.prefactor, fits.origin, last(t), last(r), first(surface), last(surface),
    )
    return (; level, t, r, fits)
end

# The bridge phase at a finer neck level, with close-ups of the neck.
function bridge_case(level; snapshots)
    local run, recorder, result
    elapsed = @elapsed begin
        run = run_example(; max_level = level, final_time = 0.0)
        fields = VisualizationFieldRegistry()
        register_default_fields!(fields, run.flow)
        recorder = Recorder(
            fields; diagnostics = run.flow, keyframes = AtTimes(snapshots),
            preview = nothing,
        )
        result = simulate!(run.flow, BRIDGE_TIME; observer = recorder)
    end
    println(
        "coalescence bridge: level ", level, ", ", result.steps, " steps, ",
        ncells(run.flow.system.mesh), " final cells, runtime ", round(elapsed; digits = 1), " s",
    )
    return (; bridge_summary(run.flow, level)..., store = snapshot_store(recorder))
end

# The default run: merger and two linear mode-2 periods of the merged drop,
# with keyframes every 0.01 during the bridge phase and 61 for the rest.
function merger_case()
    local run, recorder, result, period
    elapsed = @elapsed begin
        run = run_example(; final_time = 0.0)
        radius = sqrt(integrate(run.flow.interface) / pi)
        period = mode2_period(;
            radius, density = run.flow.system.two_phase.density, surface_tension = 1.0,
        )
        final_time = 2period
        fields = VisualizationFieldRegistry()
        register_default_fields!(fields, run.flow)
        recorder = Recorder(
            fields; diagnostics = run.flow, preview = nothing,
            keyframes = AtTimes(
                vcat(range(0, 0.3; length = 31), range(0.3, final_time; length = 62)[2:end]),
            ),
        )
        result = simulate!(run.flow, final_time; observer = recorder)
    end
    println("coalescence merger: runtime ", round(elapsed; digits = 1), " s")
    print_simulation_summary(run.flow, result)
    return (; flow = run.flow, recorder, period)
end

function bridge_figure(path, cases)
    figure = Figure(size = (1000, 400))
    axis = Axis(
        figure[1, 1]; xscale = log10, yscale = log10,
        xlabel = "t + t₀,  t₀ = (r₀/1.62)²", ylabel = "r_b / R",
        title = "Bridge half-width",
    )
    shift = (BRIDGE / DUCHEMIN_PREFACTOR)^2
    times = 10 .^ range(log10(shift), log10(BRIDGE_TIME + shift); length = 50)
    for case in cases
        lines!(
            axis, case.t .+ shift, case.r;
            label = @sprintf(
                "h = R/%d: β = %.2f, C = %.2f", 4 * 2^case.level,
                case.fits.exponent, case.fits.prefactor,
            ),
        )
    end
    lines!(
        axis, times, DUCHEMIN_PREFACTOR .* sqrt.(times); color = :black,
        linestyle = :dash, label = "1.62 √t (Duchemin et al.)",
    )
    lines!(
        axis, times, MEASURED_PREFACTOR .* sqrt.(times); color = :gray50,
        linestyle = :dot, label = "1.5 √t (experiments)",
    )
    hspan!(axis, FIT_WINDOW...; color = (:gray, 0.12))
    axislegend(axis; position = :lt, labelsize = 11)
    axis = Axis(
        figure[1, 2]; xlabel = "r_b / R", ylabel = "√(d r_b²/dt)",
        title = "Local prefactor",
    )
    for case in cases
        values = local_prefactor(case.t, case.r)
        lines!(axis, values.r, values.prefactor; label = "h = R/$(4 * 2^case.level)")
    end
    hlines!(axis, [DUCHEMIN_PREFACTOR]; color = :black, linestyle = :dash)
    hlines!(axis, [MEASURED_PREFACTOR]; color = :gray50, linestyle = :dot)
    vspan!(axis, FIT_WINDOW...; color = (:gray, 0.12))
    axislegend(axis; position = :rb, labelsize = 11)
    save(path, figure; px_per_unit = 1.5)
    return nothing
end

function neck_figure(path, store)
    infos = keyframes(store)
    figure = Figure(size = (200 * length(infos) + 90, 380))
    plot = nothing
    for (col, info) in enumerate(infos)
        frame = store[info]
        axis = fieldaxis(
            figure[1, col]; camera = (-0.3, 0.3, -0.65, 0.65),
            title = @sprintf("t = %.3f", frame.time),
        )
        plot = amrfield!(axis, frame, :volume_fraction; blocks = true, colorrange = (0, 1))
    end
    Colorbar(figure[1, length(infos) + 1], plot; label = "Volume fraction")
    save(path, figure)
    return nothing
end

# Extrema of the deformation after `after`, each at the vertex of the
# parabola through its three samples.
function deformation_extrema(t, d; after)
    result = NamedTuple{(:time, :value), Tuple{Float64, Float64}}[]
    for k in 2:(length(d) - 1)
        t[k] > after || continue
        (d[k] - d[k - 1]) * (d[k + 1] - d[k]) <= 0 || continue
        t0, t1, t2 = t[k - 1], t[k], t[k + 1]
        d0, d1, d2 = d[k - 1], d[k], d[k + 1]
        a = (t1 - t0) * (d1 - d2)
        b = (t1 - t2) * (d1 - d0)
        vertex = iszero(a - b) ? t1 : t1 - ((t1 - t0) * a - (t1 - t2) * b) / (2(a - b))
        push!(result, (; time = vertex, value = d1))
    end
    return result
end

function oscillation_figure(path, deformation, surface, kinetic; period)
    t = deformation.times
    extrema = deformation_extrema(t, deformation.values; after = 1)
    figure = Figure(size = (1000, 380))
    axis = Axis(
        figure[1, 1]; xlabel = "t / √(ρR³/σ)", ylabel = "(I_xx − I_yy) / (I_xx + I_yy)",
        title = "Mode-2 deformation",
    )
    lines!(axis, t, deformation.values; color = :steelblue)
    scatter!(axis, [e.time for e in extrema], [e.value for e in extrema]; color = :black)
    if length(extrema) >= 3
        # Linear-theory period started at the first extremum.
        start = first(extrema).time
        vlines!(
            axis, start .+ period .* (0:0.5:((last(t) - start) / period));
            color = :gray50, linestyle = :dash, label = "linear half-periods",
        )
        axislegend(axis; position = :rt, labelsize = 11)
    end
    axis = Axis(
        figure[1, 2]; xlabel = "t / √(ρR³/σ)", ylabel = "energy / (σR)",
        title = "Energy budget",
    )
    released = first(surface) .- surface
    lines!(axis, t, released; label = "released surface energy")
    lines!(axis, t, kinetic; label = "kinetic energy")
    lines!(axis, t, released .- kinetic; label = "energy residual")
    axislegend(axis; position = :lt, labelsize = 11)
    # Leave headroom below the gallery's 150 kB PNG limit.
    save(path, figure; px_per_unit = 1.4)
    return extrema
end

# Speed and interface of the merger. The keyframes are dense during the
# bridge phase and sparse during the oscillation; the title gives the time.
function merger_animation(path, store; levels = 15)
    frames = [store[info] for info in keyframes(store)]
    frame = Observable(first(frames))
    figure = Figure(size = (560, 410))
    axis = fieldaxis(
        figure[1, 1]; camera = (-2.3, 2.3, -1.6, 1.6),
        title = lift(current -> @sprintf("t = %.2f √(ρR³/σ)", current.time), frame),
    )
    colormap = cgrad(:viridis, levels; categorical = true)
    plot = amrfield!(
        axis, frame, :velocity_magnitude; colorrange = (0, 2.5), colormap,
        blocks = false, interfacecolor = :white,
    )
    Colorbar(figure[1, 2], plot; label = "speed / √(σ/(ρR))")
    # A compact categorical palette and no block overlay keep the full animation small.
    record(figure, path, frames; framerate = 12, px_per_unit = 1) do current
        frame[] = current
    end
    return nothing
end

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    started = time()
    merger = merger_case()
    period = merger.period
    # The bridge phase once more with the neck at half the cell size.
    fine = bridge_case(6; snapshots = [0.0, 0.01, 0.03, 0.06, 0.1])
    coarse = bridge_summary(merger.flow, 5)
    bridge_figure(joinpath(outdir, "bridge.png"), [coarse, fine])
    neck_figure(joinpath(outdir, "neck.png"), fine.store)
    deformation = diagnostic_history(merger.flow, :deformation)
    surface = diagnostic_history(merger.flow, :surface_energy).values
    kinetic = diagnostic_history(merger.flow, :kinetic_energy).values
    extrema = oscillation_figure(joinpath(outdir, "oscillation.png"), deformation, surface, kinetic; period)
    for e in extrema
        @printf("  deformation extremum at t = %.4f: %.5f\n", e.time, e.value)
    end
    volume = diagnostic_history(merger.flow, :phase_volume).values
    @printf(
        "  linear mode-2 period %.4f; E_s %.5f -> %.5f, E_k %.5f; liquid volume %.10f -> %.10f\n",
        period, first(surface), last(surface), last(kinetic), first(volume)[1], last(volume)[1],
    )
    merger_animation(joinpath(outdir, "merger.gif"), snapshot_store(merger.recorder))
    println("coalescence figures: total ", round(time() - started; digits = 1), " s")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
