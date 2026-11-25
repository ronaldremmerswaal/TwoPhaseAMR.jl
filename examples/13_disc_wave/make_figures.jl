using TwoPhaseAMR, CairoMakie, Printf

Base.include(@__MODULE__, joinpath(@__DIR__, "disc_wave.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "disc_wave_analysis.jl"))

const STUDY_TIME = 1.0
const STUDY_STEPS = 2000
const STUDY_RESOLUTIONS = (32, 64, 128, 256)
const STUDY_CASES = (
    (; label = "LS-STAG", closure = LSStagClosure(), composite = false, sampled = false),
    (; label = "LS-STAG, composite", closure = LSStagClosure(), composite = true, sampled = false),
    (; label = "wall distance", closure = WallDistanceClosure(), composite = false, sampled = false),
    (; label = "wall distance, composite", closure = WallDistanceClosure(), composite = true, sampled = false),
    (; label = "LS-STAG, sampled data", closure = LSStagClosure(), composite = false, sampled = true),
)

const ERROR_NAMES = (:displacement_l2, :displacement_max, :rate_l2, :rate_max)

rate_label(name, coarse, fine) =
    @sprintf("%.2f", log2(getproperty(coarse, name) / getproperty(fine, name)))

# `run_example` with the measurements of the studies: errors against the exact
# solution at t = 0 and at the end, mode frequencies and the explicit limit.
function analysed_run(; options...)
    tracker = PhaseTracker()
    initial_errors = Ref{Any}(nothing)
    function on_step(state)
        state.step == 0 && (
            initial_errors[] = disc_wave_errors(
                state.operator, state.displacement, state.rate, state.modes, 0.0;
                state.radius, state.speed,
            )
        )
        return tracker(state)
    end
    run = run_example(; merge((; keyframes = nothing), (; options...))..., on_step)
    return (;
        run..., errors = disc_wave_errors(run), initial_errors = initial_errors[],
        frequencies = mode_frequencies(tracker, run),
        explicit_limit = explicit_limit(run.operator; run.speed),
        times = collect(range(0.0, run.time; length = run.steps + 1)),
    )
end

# Spatial convergence at t = 1 with 2000 steps (dt = 5e-4), small enough that
# the time error stays below 1 % of the spatial error at 256 cells. Errors are
# relative to the same norm of the exact solution at t = 1.
function spatial_study(; resolutions = STUDY_RESOLUTIONS, steps = STUDY_STEPS)
    results = Dict{String, Vector{Any}}()
    for case in STUDY_CASES
        series = results[case.label] = Any[]
        for resolution in resolutions
            run = analysed_run(;
                resolution, case.closure, case.composite, final_time = STUDY_TIME, steps,
                tolerance = 1.0e-12,
                initial_displacement! = case.sampled ? sample_initial_displacement! :
                    project_initial_displacement!,
            )
            relative = NamedTuple{ERROR_NAMES}(relative_error.(Ref(run.errors), ERROR_NAMES))
            push!(
                series, (;
                    resolution, run.errors, relative, run.initial_errors,
                    cells = ncells(run.geometry.mesh),
                    frequency = [entry.spatial / entry.exact - 1 for entry in run.frequencies],
                    drift = maximum(abs, run.energy ./ first(run.energy) .- 1),
                    run.iterations, run.run_seconds,
                )
            )
            line = @sprintf(
                "%-26s n=%3d cells=%6d relative u L2/max %.3e/%.3e  v L2/max %.3e/%.3e  drift %.1e  %.1f s",
                case.label, resolution, series[end].cells, relative.displacement_l2,
                relative.displacement_max, relative.rate_l2, relative.rate_max,
                series[end].drift, run.run_seconds,
            )
            if length(series) > 1
                line *= "  orders " * join(
                    (rate_label(name, series[end - 1].relative, relative) for name in ERROR_NAMES), "/",
                )
            end
            println(line)
            println(
                "    initial u L2/max ",
                @sprintf("%.2e/%.2e", run.initial_errors.displacement_l2, run.initial_errors.displacement_max),
                ", spatial frequency errors ", join((@sprintf("%.3e", value) for value in series[end].frequency), ", "),
                ", CG iterations per step ", @sprintf("%.2f", run.iterations / run.steps),
            )
            flush(stdout)
        end
    end
    println("exact norms at t = 1 (u L2/max, ∂u/∂t L2/max): ", first(results["LS-STAG"]).errors.norms)
    return results
end

# Open dual volume of every interior face that carries a fluid unknown.
function fluid_weights(run)
    weights = zeros(length(run.displacement.interior))
    for axis in 1:2
        nodes = value_points(run.operator, axis)
        weights[nodes.faces] .= nodes.volumes
    end
    return weights
end

# Temporal self-convergence on one mesh: differences to a run with 5120 steps,
# relative to the exact norms at t = 1.
function temporal_study(; resolution = 128, steps = (20, 40, 80, 160, 320, 640))
    options = (; resolution, final_time = STUDY_TIME, tolerance = 1.0e-12)
    reference = analysed_run(; options..., steps = 5120)
    norms = reference.errors.norms
    weights = fluid_weights(reference)
    open = weights .> 0
    function difference(first, second, l2, maximum_norm)
        error = abs.(first.interior .- second.interior)[open]
        return sqrt(sum(weights[open] .* error .^ 2) / sum(weights[open])) / l2, maximum(error) / maximum_norm
    end
    series = map(steps) do count
        run = analysed_run(; options..., steps = count)
        u_l2, u_max = difference(run.displacement, reference.displacement, norms.displacement_l2, norms.displacement_max)
        v_l2, v_max = difference(run.rate, reference.rate, norms.rate_l2, norms.rate_max)
        (; dt = run.dt, u_l2, u_max, v_l2, v_max, exact = relative_error(run.errors, :displacement_l2))
    end
    for (index, entry) in pairs(series)
        line = @sprintf(
            "dt=%.5f  relative self u L2/max %.3e/%.3e  v L2/max %.3e/%.3e  exact u L2 %.3e",
            entry.dt, entry.u_l2, entry.u_max, entry.v_l2, entry.v_max, entry.exact,
        )
        if index > 1
            line *= "  orders " * join(
                (rate_label(name, series[index - 1], entry) for name in (:u_l2, :u_max, :v_l2, :v_max)), "/",
            )
        end
        println(line)
    end
    println("reference (5120 steps): relative u L2 against exact ", relative_error(reference.errors, :displacement_l2))
    return series
end

# Energy over four periods of the (0,1) mode at 128 steps per period, for
# several solver tolerances.
function energy_study(; resolution = 128, periods = 4, tolerances = (1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12))
    return map(tolerances) do tolerance
        run = analysed_run(;
            resolution, tolerance, final_time = periods * 2pi / DISC_WAVE_MODES[1].root,
            steps = 128periods,
        )
        change = run.energy ./ first(run.energy) .- 1
        println(
            @sprintf(
                "tolerance %.0e: dt %.5f (%.1f × leapfrog limit), largest |E/E0 - 1| %.2e, ",
                tolerance, run.dt, run.dt / run.explicit_limit, maximum(abs, change),
            ),
            @sprintf("%.2f CG iterations per step, %.2f s", run.iterations / run.steps, run.run_seconds),
        )
        (; tolerance, times = run.times, change, period = run.time / periods)
    end
end

# Both displacement components over one period of the (0,1) mode, one output
# frame per keyframe; 31 colour levels keep the GIF small.
function wave_animation(path, run; size)
    store = snapshot_store(run.recorder)
    frames = [store[info] for info in keyframes(store)]
    names = (:displacement_x, :displacement_y)
    limit = maximum(frame.stats[name].absmax for frame in frames, name in names)
    current = Observable(first(frames))
    figure = Figure(; size, figure_padding = 4)
    plot = nothing
    for (column, name) in pairs(names)
        axis = fieldaxis(
            figure[1, column]; title = lift(
                frame -> rich("u", subscript("xy"[column:column]), @sprintf(",  t = %.2f", frame.time)),
                current,
            ),
        )
        plot = amrfield!(
            axis, current, name; colorrange = (-limit, limit),
            colormap = cgrad(:balance, 31; categorical = true), geometrycolor = :gray35,
        )
        # The exact wall r = R over the PLIC solid.
        arc!(axis, Point2f(0, 0), 1, 0, 2pi; color = :black, linewidth = 1)
        colsize!(figure.layout, column, Aspect(1, 1.0))
    end
    Colorbar(figure[1, 3], plot)
    resize_to_layout!(figure)
    record(figure, path, frames; framerate = 12, px_per_unit = 1.2) do frame
        current[] = frame
    end
    return limit
end

# Ticks at whole decades for the logarithmic error axes.
const DECADES = (10.0 .^ (-8:1), [rich("10", superscript(string(power))) for power in -8:1])

# Errors against the cells per side of the box, `h = 2.5R/n`, on log-log axes.
cells_axis(position; kwargs...) = Axis(
    position; xscale = log2, yscale = log10, xticks = collect(STUDY_RESOLUTIONS),
    yticks = DECADES, xlabel = "cells per side n", kwargs...,
)

# A dotted reference slope `x^order` through `(x[end], y)`.
reference_slope!(axis, x, y, order; label = nothing) = lines!(
    axis, x, y .* (x ./ last(x)) .^ order; color = :gray45, linestyle = :dot, label,
)

# A legend below the panel in `column`.
legend_below!(figure, column, axis) = Legend(
    figure[2, column], axis; framevisible = false, labelsize = 11, nbanks = 2,
    rowgap = 0, tellwidth = false, tellheight = true,
)

function convergence_figure(path, spatial; size)
    figure = Figure(; size)
    colors = Makie.wong_colors()
    resolutions(series) = [entry.resolution for entry in series]
    n = resolutions(spatial["LS-STAG"])

    displacement = cells_axis(
        figure[1, 1]; ylabel = "relative error of u at t = 1", title = "u: L2 (solid), max (dashed)",
    )
    for (index, case) in pairs(STUDY_CASES[1:4])
        series = spatial[case.label]
        values(name) = [getproperty(entry.relative, name) for entry in series]
        scatterlines!(displacement, resolutions(series), values(:displacement_l2); color = colors[index], label = case.label)
        scatterlines!(displacement, resolutions(series), values(:displacement_max); color = colors[index], linestyle = :dash)
    end
    reference_slope!(displacement, n, spatial["LS-STAG"][end].relative.displacement_l2 / 3, -2; label = "n⁻², n⁻¹")
    reference_slope!(displacement, n, spatial["wall distance"][end].relative.displacement_max * 3, -1)
    legend_below!(figure, 1, displacement)

    frequency = cells_axis(figure[1, 2]; ylabel = "|ω_h/ω − 1|", title = "Frequency: LS-STAG (solid), wall (dashed)")
    for (index, mode) in pairs(DISC_WAVE_MODES)
        label = rich(@sprintf("(%d,%d) in u", mode.m, mode.k), subscript("xy"[mode.axis:mode.axis]))
        for (case, linestyle) in (("LS-STAG", :solid), ("wall distance", :dash))
            series = spatial[case]
            scatterlines!(
                frequency, resolutions(series), [abs(entry.frequency[index]) for entry in series];
                color = colors[index], linestyle, label = linestyle === :solid ? label : nothing,
            )
        end
    end
    reference_slope!(frequency, n, abs(spatial["LS-STAG"][end].frequency[1]) / 3, -2; label = "n⁻²")
    legend_below!(figure, 2, frequency)

    rate = cells_axis(
        figure[1, 3]; ylabel = "relative max error of ∂u/∂t at t = 1",
        title = "∂u/∂t: projected or sampled u(0)",
    )
    for (index, label) in ((1, "LS-STAG"), (3, "wall distance"), (5, "LS-STAG, sampled data"))
        series = spatial[label]
        scatterlines!(rate, resolutions(series), [entry.relative.rate_max for entry in series]; color = colors[index], label)
    end
    reference_slope!(rate, n, spatial["LS-STAG"][end].relative.rate_max / 3, -2; label = "n⁻²")
    legend_below!(figure, 3, rate)
    save(path, figure; px_per_unit = 1)
    return path
end

function time_figure(path, temporal, energy; size)
    figure = Figure(; size)
    colors = Makie.wong_colors()
    steps = [round(Int, STUDY_TIME / entry.dt) for entry in temporal]
    timing = Axis(
        figure[1, 1]; xscale = log2, yscale = log10, xticks = steps,
        yticks = DECADES, xlabel = "steps to t = 1",
        ylabel = "relative difference at t = 1", title = "128 cells: against 5120 steps",
    )
    for (index, (name, label)) in pairs(((:u_l2, "u, L2"), (:u_max, "u, max"), (:v_l2, "∂u/∂t, L2"), (:v_max, "∂u/∂t, max")))
        scatterlines!(timing, steps, [getproperty(entry, name) for entry in temporal]; color = colors[index], label)
    end
    scatterlines!(timing, steps, [entry.exact for entry in temporal]; color = :black, linestyle = :dash, label = "u, L2, against exact")
    reference_slope!(timing, steps, last(temporal).u_l2 / 3, -2; label = "Δt²")
    legend_below!(figure, 1, timing)

    drift = Axis(
        figure[1, 2]; yscale = log10, xlabel = "t / T₀₁", ylabel = "|E(t)/E(0) − 1|",
        title = "Energy: 128 cells, Δt = T₀₁/128",
    )
    for (index, entry) in pairs(energy)
        values = abs.(entry.change)
        keep = values .> 0
        lines!(
            drift, entry.times[keep] ./ entry.period, values[keep];
            color = colors[index], label = @sprintf("tolerance %.0e", entry.tolerance),
        )
    end
    legend_below!(figure, 2, drift)
    save(path, figure; px_per_unit = 1)
    return path
end

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    docs = size === :docs
    started = time()
    hero = analysed_run(; keyframes = EquispacedTimes(65))
    println(
        "hero: 128 cells, ", hero.steps, " steps, dt = ", hero.dt, " = ",
        hero.dt / hero.explicit_limit, " × leapfrog limit ", hero.explicit_limit,
        "; relative errors at t = ", hero.time, ": ", NamedTuple{ERROR_NAMES}(relative_error.(Ref(hero.errors), ERROR_NAMES)),
        "; largest |E/E0 - 1| ", maximum(abs, hero.energy ./ first(hero.energy) .- 1),
        "; ", hero.iterations, " CG iterations; setup ", hero.setup_seconds, " s, stepping ",
        hero.run_seconds, " s",
    )
    for entry in hero.frequencies
        println(
            "  mode (", entry.mode.m, ",", entry.mode.k, "): frequency error ",
            entry.discrete / entry.exact - 1, ", spatial part ", entry.spatial / entry.exact - 1,
        )
    end
    limit = wave_animation(joinpath(outdir, "wave.gif"), hero; size = docs ? (640, 300) : (860, 400))
    println("animation colour range ±", limit)
    spatial = spatial_study()
    temporal = temporal_study()
    energy = energy_study()
    convergence_figure(joinpath(outdir, "convergence.png"), spatial; size = docs ? (1020, 470) : (1360, 620))
    time_figure(joinpath(outdir, "time.png"), temporal, energy; size = docs ? (860, 430) : (1150, 570))
    println("figure script: ", time() - started, " s")
    return (; hero, spatial, temporal, energy)
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
