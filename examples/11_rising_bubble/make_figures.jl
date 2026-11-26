using TwoPhaseAMR, CairoMakie, Printf

Base.include(@__MODULE__, joinpath(@__DIR__, "rising_bubble.jl"))

const GROUPS = ("TP2D", "FreeLIFE", "MooNMD")
const REFERENCE_STYLES = Dict(
    "TP2D" => (color = :black, linestyle = :dash),
    "FreeLIFE" => (color = :gray45, linestyle = :dot),
    "MooNMD" => (color = :gray45, linestyle = :dashdot),
)
const BENCHMARK_MARKERS = Dict("TP2D" => :circle, "FreeLIFE" => :rect, "MooNMD" => :utriangle)
# The finest grids, whose series `hysing_reference.csv` holds; MooNMD counts
# the degrees of freedom on its interface.
reference_grid(case, group) = group == "TP2D" ? "1/h = $(case == 1 ? 320 : 640)" :
    group == "FreeLIFE" ? "1/h = 160" : "900 interface DOF"

# Benchmark quantities of the three groups on their finest grids, Hysing et
# al. (2009), Tables XII (case 1) and XVI (case 2): `(time, value)` pairs, one
# rise-velocity maximum in case 1 and two in case 2.
const BENCHMARK = Dict(
    (1, "TP2D") => (;
        circularity_min = (1.9041, 0.9013), velocity_max = ((0.9213, 0.2417),),
        centroid_end = 1.0813,
    ),
    (1, "FreeLIFE") => (;
        circularity_min = (1.875, 0.9011), velocity_max = ((0.9313, 0.2421),),
        centroid_end = 1.0799,
    ),
    (1, "MooNMD") => (;
        circularity_min = (1.9, 0.9013), velocity_max = ((0.9239, 0.2417),),
        centroid_end = 1.0817,
    ),
    (2, "TP2D") => (;
        circularity_min = (2.4004, 0.5869),
        velocity_max = ((0.7332, 0.2524), (2.0705, 0.2434)), centroid_end = 1.138,
    ),
    (2, "FreeLIFE") => (;
        circularity_min = (3.0, 0.4647),
        velocity_max = ((0.7281, 0.2514), (1.9844, 0.244)), centroid_end = 1.1249,
    ),
    (2, "MooNMD") => (;
        circularity_min = (3.0, 0.5144),
        velocity_max = ((0.7317, 0.2502), (2.06, 0.2393)), centroid_end = 1.1376,
    ),
)

"""Numeric rows of one case in a reference CSV of this folder, by group."""
function reference_rows(file, case)
    rows = Dict{String, Vector{Vector{Float64}}}()
    for line in eachline(joinpath(@__DIR__, file))
        (startswith(line, '#') || startswith(line, "case")) && continue
        fields = split(line, ',')
        parse(Int, fields[1]) == case || continue
        push!(get!(rows, fields[2], Vector{Float64}[]), parse.(Float64, fields[3:end]))
    end
    return rows
end

"""Linear interpolation of the sampled `values` at `time`, held at the ends."""
function interpolate(times, values, time)
    k = searchsortedlast(times, time)
    k == 0 && return first(values)
    k == length(times) && return last(values)
    weight = (time - times[k]) / (times[k + 1] - times[k])
    return (1 - weight) * values[k] + weight * values[k + 1]
end

"""
    error_norms(series, reference; final_time, spacing=1.0e-3)

Relative error norms of Hysing et al. (2009, §2.6) of a sampled `series`
against a `reference`, both `(times, values)`: the l1, l2 and l∞ norms of the
difference over those of the reference. Both are linearly interpolated onto
`t_k = k * spacing`, `k = 1, …, final_time / spacing`; like the paper's sums
over time steps, this leaves out the initial state.
"""
function error_norms(series, reference; final_time, spacing = 1.0e-3)
    grid = spacing .* (1:round(Int, final_time / spacing))
    q = [interpolate(series..., time) for time in grid]
    r = [interpolate(reference..., time) for time in grid]
    e = abs.(q .- r)
    return (sum(e) / sum(abs, r), sqrt(sum(abs2, e) / sum(abs2, r)), maximum(e) / maximum(abs, r))
end

# Print the error norms of `runs` (successive levels, `h` halving) against the
# finest-grid series of each benchmark code, with the rates of convergence
# between levels; against the finest run, as the paper measures its codes; and
# those of the codes against each other for scale. Case 2 stops at t = 1.75,
# where the codes part ways.
function print_error_norms(case, runs)
    final_time = case == 1 ? 3.0 : 1.75
    reference = reference_rows("hysing_reference.csv", case)
    series(group, column) = ([r[1] for r in reference[group]], [r[column] for r in reference[group]])
    show_norms(label, norms, previous) = @printf(
        "  %-46s %.2e %.2e %.2e%s\n", label, norms...,
        isnothing(previous) ? "" :
            @sprintf("   ROC %.2f %.2f %.2f", log2.(previous ./ norms)...)
    )
    println("Relative error norms l1, l2, l∞ for case $case, t ≤ $final_time:")
    for (column, name) in enumerate((:circularity, :centroid, :rise_velocity))
        println(" ", name)
        for group in GROUPS
            previous = nothing
            for run in runs
                norms = error_norms(run.histories[name], series(group, column + 1); final_time)
                show_norms("$(run_label(run)) vs $group", norms, previous)
                previous = norms
            end
        end
        previous = nothing
        for run in runs[1:(end - 1)]
            norms = error_norms(run.histories[name], last(runs).histories[name]; final_time)
            show_norms("level $(run.level) vs level $(last(runs).level)", norms, previous)
            previous = norms
        end
        for (group, other) in (("FreeLIFE", "TP2D"), ("MooNMD", "TP2D"), ("MooNMD", "FreeLIFE"))
            norms = error_norms(series(group, column + 1), series(other, column + 1); final_time)
            show_norms("$group vs $other", norms, nothing)
        end
    end
    return nothing
end

"""The `(times, values)` histories of the benchmark quantities of a flow."""
benchmark_histories(flow) = Dict(
    name => (diagnostic_history(flow, name).times, diagnostic_history(flow, name).values)
        for name in (:circularity, :centroid, :rise_velocity)
)

"""
    running_median(times, values; width=0.04)

The median of `values` over the samples within `width/2` of each time. The
circularity dips for a step or two where the interface touches a grid line
(see README); a window of 0.04 removes these dips but no physical change.
"""
function running_median(times, values; width = 0.04)
    first_sample = last_sample = 1
    return map(eachindex(times)) do k
        while times[first_sample] < times[k] - width / 2
            first_sample += 1
        end
        while last_sample < lastindex(times) && times[last_sample + 1] <= times[k] + width / 2
            last_sample += 1
        end
        window = sort(values[first_sample:last_sample])
        n = length(window)
        isodd(n) ? window[(n + 1) ÷ 2] : (window[n ÷ 2] + window[n ÷ 2 + 1]) / 2
    end
end

"""
Benchmark quantities of one run's histories, in the layout of `BENCHMARK`, and
the minimum of the circularity's running median.
"""
function benchmark_quantities(histories, case)
    times, circularity = histories[:circularity]
    i = argmin(circularity)
    smoothed = running_median(times, circularity)
    j = argmin(smoothed)
    # The second maximum of case 2 follows a minimum near t = 1.5.
    velocity_times, velocity = histories[:rise_velocity]
    windows = case == 1 ? (eachindex(velocity_times),) :
        (findall(<(1.5), velocity_times), findall(>=(1.5), velocity_times))
    velocity_max = map(filter(!isempty, collect(windows))) do window
        k = window[argmax(velocity[window])]
        (velocity_times[k], velocity[k])
    end
    return (;
        circularity_min = (times[i], circularity[i]),
        smoothed_circularity_min = (times[j], smoothed[j]), velocity_max,
        centroid_end = last(last(histories[:centroid])),
    )
end

model_name(model) = model isa TwoVelocityModel ? "two velocities" : "one velocity"
run_label(run) = "level $(run.level) (1/h = $(8 * 2^run.level)), $(model_name(run.model))"

function bubble_figure_case(case, level; final_time = 3.0, keyframes = 7)
    local run, recorder, result
    elapsed = @elapsed begin
        run = run_example(; case, max_level = level, final_time = 0.0)
        # The TP2D rise velocity, a reference curve for the animation.
        tp2d = reference_rows("hysing_reference.csv", case)["TP2D"]
        times, velocities = first.(tp2d), last.(tp2d)
        add_diagnostic!(
            run.flow, :tp2d_rise_velocity,
            state -> interpolate(times, velocities, state.time);
            label = "TP2D rise velocity", dimension = PhysicalDimension(0, 1, -1)
        )
        fields = VisualizationFieldRegistry()
        register_default_fields!(fields, run.flow)
        recorder = Recorder(
            fields; diagnostics = run.flow, keyframes = EquispacedTimes(keyframes),
            preview = nothing
        )
        result = simulate!(run.flow, final_time; observer = recorder)
    end
    model = run.flow.velocity isa TwoPhaseVelocity ? TwoVelocityModel() : OneVelocityModel()
    histories = benchmark_histories(run.flow)
    quantities = benchmark_quantities(histories, case)
    gas = [value[2] for value in diagnostic_history(run.flow, :phase_volume).values]
    println(
        "rising_bubble: case=$case, level=$level, $(model_name(model)), runtime=$elapsed s, ",
        "steps=$(result.steps), final cells=$(ncells(run.flow.system.mesh)), ",
        "gas area change=$(last(gas) - first(gas))"
    )
    @printf(
        "  c_min = %.4f at t = %.4f (running median %.4f at t = %.4f); y_c(%.2f) = %.4f\n",
        quantities.circularity_min[2], quantities.circularity_min[1],
        quantities.smoothed_circularity_min[2], quantities.smoothed_circularity_min[1],
        result.time, quantities.centroid_end
    )
    for (k, (time, value)) in enumerate(quantities.velocity_max)
        @printf("  V_c,max%d = %.4f at t = %.4f\n", k, value, time)
    end
    print_simulation_summary(run.flow, result)
    return (; case, level, model, flow = run.flow, recorder, result, elapsed, histories, quantities)
end

# One row of full histories (circularity, centroid, rise velocity) and one of
# close-ups around the benchmark quantities in `zooms`, `(xmin, xmax, ymin,
# ymax)` per column, against the groups' finest-grid series and values.
function validation_figure(path, case, runs, zooms)
    reference = reference_rows("hysing_reference.csv", case)
    names = (:circularity, :centroid, :rise_velocity)
    labels = ("Circularity", "Centroid height", "Rise velocity")
    colors = Makie.wong_colors()
    figure = Figure(size = (900, 580), fontsize = 13)
    axes = [
        Axis(
            figure[row, column]; xlabel = "t", ylabel = labels[column],
            title = row == 1 ? "" : "close-up"
        ) for row in 1:2, column in 1:3
    ]
    for row in 1:2, (column, name) in enumerate(names)
        axis = axes[row, column]
        # No converged reference after t ≈ 1.75 in case 2 (see README).
        case == 2 && vspan!(axis, 1.75, 3.0; color = (:gray, 0.12))
        for group in GROUPS
            data = reference[group]
            lines!(
                axis, [r[1] for r in data], [r[column + 1] for r in data];
                label = "$group, $(reference_grid(case, group))", linewidth = 1.2,
                REFERENCE_STYLES[group]...
            )
        end
        for (k, run) in enumerate(runs)
            lines!(axis, run.histories[name]...; color = colors[k], label = run_label(run))
        end
        for group in GROUPS
            benchmark = BENCHMARK[(case, group)]
            points = name === :circularity ? [benchmark.circularity_min] :
                name === :centroid ? [(3.0, benchmark.centroid_end)] :
                collect(benchmark.velocity_max)
            scatter!(
                axis, points; color = :black, marker = BENCHMARK_MARKERS[group],
                markersize = 8, label = "$group benchmark value (Hysing et al.)"
            )
        end
        row == 2 && limits!(axis, zooms[column]...)
    end
    Legend(
        figure[3, 1:3], axes[1, 1]; orientation = :horizontal, nbanks = 3,
        framevisible = false, labelsize = 11, unique = true
    )
    save(path, figure)
    return nothing
end

# Final PLIC interfaces against the groups' shapes at t = 3, one panel per case.
function shapes_figure(path, cases)
    colors = Makie.wong_colors()
    figure = Figure(size = (700, 470), fontsize = 13)
    axes = Axis[]
    for (column, runs) in enumerate(cases)
        case = first(runs).case
        axis = Axis(
            figure[1, column]; aspect = DataAspect(), xlabel = "x", ylabel = "y",
            title = "case $case at t = 3, TP2D $(reference_grid(case, "TP2D"))"
        )
        push!(axes, axis)
        reference = reference_rows("hysing_shapes.csv", case)
        for group in GROUPS
            haskey(reference, group) || continue
            rows = reference[group]
            for curve in unique(first.(rows))
                points = [Point2f(r[2], r[3]) for r in rows if r[1] == curve]
                lines!(
                    axis, points; linewidth = 1.2, REFERENCE_STYLES[group]...,
                    label = group == "TP2D" ? group : "$group, $(reference_grid(case, group))"
                )
            end
        end
        for (k, run) in enumerate(runs)
            points = [Point2f(p[1], p[2]) for segment in plic_segments(run.flow.interface) for p in segment]
            linesegments!(axis, points; color = colors[k], label = run_label(run))
        end
        limits!(axis, 0.1, 0.9, 0.5, 1.45)
    end
    Legend(
        figure[2, 1:2], axes[2]; orientation = :horizontal, nbanks = 2,
        framevisible = false, labelsize = 11, unique = true
    )
    save(path, figure)
    return nothing
end

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    started = time()
    levels = size === :docs ? (3, 4) : (3, 4, 5)
    case1 = [bubble_figure_case(1, level) for level in levels]
    # The finest case-2 run is the animation: 81 uniformly spaced keyframes.
    case2 = [
        bubble_figure_case(2, level; keyframes = level == last(levels) ? 81 : 7)
            for level in levels
    ]
    print_error_norms(1, case1)
    print_error_norms(2, case2)

    # 1.1 device pixels per unit keep the step-to-step jitter of the
    # circularity within the gallery's 150 kB per PNG (1.25 gave 157 kB).
    CairoMakie.activate!(px_per_unit = 1.1)
    validation_figure(
        joinpath(outdir, "case1.png"), 1, case1,
        ((1.5, 2.3, 0.885, 0.91), (2.5, 3.0, 0.98, 1.1), (0.6, 1.3, 0.234, 0.246))
    )
    validation_figure(
        joinpath(outdir, "case2.png"), 2, case2,
        ((1.0, 2.0, 0.6, 0.95), (2.5, 3.0, 1.0, 1.15), (0.5, 2.5, 0.22, 0.26))
    )
    shapes_figure(joinpath(outdir, "shapes.png"), (case1, case2))
    store = snapshot_store(last(case2).recorder)
    window = (0.0, 1.0, 0.2, 1.45)
    # The gas inside the skirt reaches |ω| ≈ 40; ±8 resolves the liquid wake.
    vorticity = (colorrange = (-8.0, 8.0), colormap = cgrad(:balance, 31; categorical = true))
    export_figure(
        joinpath(outdir, "hero.png"), store; field = :velocity_vorticity,
        at = [0.75, 1.5, 2.25, 3.0], camera = window, blocks = true, vorticity...
    )
    CairoMakie.activate!(px_per_unit = 2)
    # One output frame per keyframe: a uniform time step and no held frames.
    export_animation(
        joinpath(outdir, "case2.gif"), store; field = :velocity_vorticity,
        camera = window, blocks = true, diagnostics = [:rise_velocity],
        references = Dict(:rise_velocity => :tp2d_rise_velocity), fps = 10,
        nframes = length(keyframes(store)), size = (900, 520), vorticity...
    )
    println("Whole figure script: ", time() - started, " s (package loading excluded)")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
