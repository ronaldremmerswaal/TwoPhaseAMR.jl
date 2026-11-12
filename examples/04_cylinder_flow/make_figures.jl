using CairoMakie
using Statistics
Base.include(@__MODULE__, joinpath(@__DIR__, "cylinder_flow.jl"))

function readme_animations(outdir, store, window)
    camera = (-1.5, 8.0, -2.5, 2.5)
    frames = [store[info] for info in keyframes(store) if window[1] <= info.time <= window[2]]
    magnitudes = Float32[]
    for frame in frames, value in rasterize(frame, :vorticity, camera, (475, 250))
        isfinite(value) && push!(magnitudes, abs(value))
    end
    limit = sort!(magnitudes)[ceil(Int, 0.999length(magnitudes))]
    println("README: $(length(frames)) frames, vorticity range ±$limit")
    function animation(path, colormap; blockcolor, geometrycolor)
        figure = Figure(size = (760, 400), figure_padding = 0)
        axis = fieldaxis(figure[1, 1]; camera, decorations = :none)
        current = Observable(first(frames))
        amrfield!(
            axis, current, :vorticity; colorrange = (-limit, limit),
            colormap = cgrad(colormap, 31; categorical = true), blocks = true,
            blockcolor, geometrycolor
        )
        return record(figure, path, frames; framerate = 12) do frame
            current[] = frame
        end
    end
    animation(
        joinpath(outdir, "readme_light.gif"), :balance;
        blockcolor = (:gray25, 0.35), geometrycolor = :gray20
    )
    muted = [get(CairoMakie.Makie.ColorSchemes.berlin, x) for x in range(0.15, 0.85; length = 31)]
    with_theme(theme_dark(); textcolor = :gray80) do
        animation(
            joinpath(outdir, "readme_dark.gif"), muted;
            blockcolor = (:white, 0.25), geometrycolor = :gray75
        )
    end
    return nothing
end

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    started = time_ns()
    max_level, final_time = 4, 50.0
    window = (final_time - 20, final_time)
    flow = run_example(; reynolds = 200.0, max_level, final_time = 0.0, perturbation = 0.1).flow
    fields = VisualizationFieldRegistry()
    register_field!(
        fields, :vorticity, flow.velocity, Vorticity();
        label = "Vorticity", colormap = :balance
    )
    register_geometry!(fields, flow.system.geometry)
    recorder = Recorder(
        fields; diagnostics = flow, keyframes = AtTimes(collect(range(window...; length = 49))),
        preview = nothing
    )
    result = simulate!(flow, final_time; observer = recorder)
    simulation_seconds = (time_ns() - started) / 1.0e9
    print_simulation_summary(flow, result)
    println("Final cells: ", TwoPhaseAMR.ncells(flow.system.mesh))
    for name in (:pressure_drag, :pressure_lift)
        history = diagnostic_history(flow, name)
        values = [v for (t, v) in zip(history.times, history.values) if t >= window[1] && isfinite(v)]
        println(
            name, ": mean = ", mean(values), ", rms = ", sqrt(mean(abs2, values)),
            ", fluctuation rms = ", std(values; corrected = false)
        )
    end
    lift = diagnostic_history(flow, :pressure_lift)
    drag = diagnostic_history(flow, :pressure_drag)
    store = snapshot_store(recorder)
    sample_times = collect(range(window...; step = 0.05))
    samples = [lift.values[searchsortedlast(lift.times, t)] for t in sample_times]
    smoothed = [mean(@view samples[max(1, i - 5):min(end, i + 5)]) for i in eachindex(samples)]
    lift_mean = mean(smoothed)
    crossings = Float64[]
    for i in 2:length(smoothed)
        if smoothed[i - 1] < lift_mean <= smoothed[i]
            fraction = (lift_mean - smoothed[i - 1]) / (smoothed[i] - smoothed[i - 1])
            push!(crossings, sample_times[i - 1] + fraction * (sample_times[i] - sample_times[i - 1]))
        end
    end
    amplitude = (maximum(smoothed) - minimum(smoothed)) / 2
    noise = std(samples .- smoothed; corrected = false)
    println("Smoothed lift upward crossings: ", crossings)
    println("Strouhal from crossings: ", length(crossings) >= 2 ? inv(mean(diff(crossings))) : NaN)
    println("Lift residual noise RMS: ", noise, "; noise / smoothed amplitude: ", noise / amplitude)
    println("Setup + simulation: ", simulation_seconds, " s (package loading excluded)")
    export_figure(
        joinpath(outdir, "vorticity.png"), store; field = :vorticity,
        colorrange = (-3, 3), blocks = true, title = "Vorticity at t = $final_time"
    )
    figure = Figure(size = size === :docs ? (700, 420) : (900, 540))
    axis = only(diagnosticsgrid!(figure[1, 1], store; names = [:pressure_drag]))
    xlims!(axis, window...)
    drag_values = [v for (t, v) in zip(drag.times, drag.values) if t >= window[1] && isfinite(v)]
    lo, hi = extrema(drag_values)
    margin = 0.1 * (hi - lo)
    ylims!(axis, lo - margin, hi + margin)
    save(joinpath(outdir, "pressure_drag.png"), figure)
    export_animation(
        joinpath(outdir, "shedding.gif"), store; field = :vorticity, window,
        blocks = true, colorrange = (-3, 3), colormap = cgrad(:balance, 31; categorical = true),
        fps = 12, duration = 4.0, size = (700, 310)
    )
    readme_animations(outdir, store, window)
    println("Including figures: ", (time_ns() - started) / 1.0e9, " s (package loading excluded)")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
