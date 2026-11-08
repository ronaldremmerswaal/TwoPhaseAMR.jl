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
        amrfield!(axis, current, :vorticity; colorrange = (-limit, limit),
            colormap = cgrad(colormap, 31; categorical = true), blocks = true,
            blockcolor, geometrycolor)
        record(figure, path, frames; framerate = 12) do frame
            current[] = frame
        end
    end
    animation(joinpath(outdir, "readme_light.gif"), :balance;
        blockcolor = (:gray25, 0.35), geometrycolor = :gray20)
    muted = [get(CairoMakie.Makie.ColorSchemes.berlin, x) for x in range(0.15, 0.85; length = 31)]
    with_theme(theme_dark(); textcolor = :gray80) do
        animation(joinpath(outdir, "readme_dark.gif"), muted;
            blockcolor = (:white, 0.25), geometrycolor = :gray75)
    end
    return nothing
end

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    started = time_ns()
    max_level, final_time = size === :docs ? (3, 60.0) : (4, 60.0)
    window = (final_time - 16, final_time)
    flow = run_example(; reynolds = 200.0, max_level, final_time = 0.0).flow
    fields = VisualizationFieldRegistry()
    register_field!(fields, :vorticity, flow.velocity, Vorticity();
        label = "Vorticity", colormap = :balance)
    register_geometry!(fields, flow.solver.geometry)
    recorder = Recorder(fields; diagnostics = flow, keyframes = AtTimes(collect(range(window...; length = 49))),
        preview = nothing)
    result = simulate!(flow, final_time; observer = recorder)
    simulation_seconds = (time_ns() - started) / 1e9
    print_simulation_summary(flow, result)
    println("Final cells: ", TwoPhaseAMR.ncells(flow.solver.mesh))
    for name in (:pressure_drag, :pressure_lift)
        history = diagnostic_history(flow, name)
        values = [v for (t, v) in zip(history.times, history.values) if t >= window[1] && isfinite(v)]
        println(name, ": mean = ", mean(values), ", rms = ", sqrt(mean(abs2, values)),
            ", fluctuation rms = ", std(values; corrected = false))
    end
    lift = diagnostic_history(flow, :pressure_lift)
    peaks = [lift.times[i] for i in 2:(length(lift.times) - 1) if lift.times[i] >= window[1] &&
        lift.values[i] > lift.values[i - 1] && lift.values[i] >= lift.values[i + 1]]
    println("Lift peak times: ", peaks)
    println("Setup + simulation: ", simulation_seconds, " s (package loading excluded)")
    store = snapshot_store(recorder)
    export_figure(joinpath(outdir, "vorticity.png"), store; field = :vorticity,
        blocks = true, title = "Vorticity at t = $final_time")
    figure = Figure(size = (700, 420))
    diagnosticsgrid!(figure[1, 1], store; names = [:pressure_drag])
    save(joinpath(outdir, "pressure_drag.png"), figure)
    export_animation(joinpath(outdir, "shedding.gif"), store; field = :vorticity, window,
        blocks = true, colorrange = (-3, 3), framerate = 12, size = (900, 400))
    readme_animations(outdir, store, window)
    println("Including figures: ", (time_ns() - started) / 1e9, " s (package loading excluded)")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
