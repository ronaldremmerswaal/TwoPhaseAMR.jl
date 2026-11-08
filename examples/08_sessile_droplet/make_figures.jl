using TwoPhaseAMR, CairoMakie

Base.include(@__MODULE__, joinpath(@__DIR__, "sessile_droplet.jl"))

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    level = size === :docs ? 2 : 3
    target_time = 8sqrt((1 + 1.0e-3) * 0.2^3)
    local run, recorder, result
    elapsed = @elapsed begin
        run = run_example(max_level = level, final_time = 0.0)
        fields = VisualizationFieldRegistry()
        register_default_fields!(fields, :volume_fraction, run.flow.interface)
        recorder = Recorder(fields; keyframes = EquispacedTimes(3), preview = nothing)
        result = simulate!(run.flow, target_time; observer = recorder)
    end
    println("sessile_droplet: size=$size, runtime=$elapsed s")
    print_simulation_summary(run.flow, result)
    energy = diagnostic_history(run.flow, :kinetic_energy)
    volume = diagnostic_history(run.flow, :phase_volume)
    liquid = [value[1] for value in volume.values]
    println("initial/final kinetic energy = ", (first(energy.values), last(energy.values)),
        "; maximum kinetic energy = ", maximum(energy.values),
        "; initial/final (liquid, gas) volumes = ",
        (first(volume.values), last(volume.values)),
        "; maximum sampled liquid-volume error = ", maximum(abs.(liquid .- first(liquid))))

    store = snapshot_store(recorder)
    checkpoints = [store[info] for info in keyframes(store)]
    figure = Figure(size = (900, 330))
    for (i, frame) in enumerate(checkpoints)
        axis = snapshot_figure!(figure[1, i], frame; field = :volume_fraction,
            title = "t = $(round(frame.time; digits = 3))", colorbar = i == length(checkpoints))
        xlims!(axis, 0, 1)
        ylims!(axis, 0, 0.5)
    end
    save(joinpath(outdir, "snapshots.png"), figure)

    figure = Figure(size = (850, 350))
    axis = Axis(figure[1, 1]; title = "Kinetic energy", xlabel = "t", ylabel = "KE")
    plot_diagnostics!(axis, run.flow, :kinetic_energy; labels = "kinetic energy")
    axis = Axis(figure[1, 2]; title = "Liquid volume", xlabel = "t", ylabel = "Vₗ")
    lines!(axis, volume.times, liquid; color = :dodgerblue)
    save(joinpath(outdir, "diagnostics.png"), figure)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
