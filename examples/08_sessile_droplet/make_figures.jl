using TwoPhaseAMR, CairoMakie

Base.include(@__MODULE__, joinpath(@__DIR__, "sessile_droplet.jl"))

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    level = 3
    target_time = 8sqrt((1 + 1.0e-3) * 0.2^3)
    local run, recorder, result
    elapsed = @elapsed begin
        run = run_example(max_level = level, final_time = 0.0)
        fields = VisualizationFieldRegistry()
        register_default_fields!(fields, run.flow)
        recorder = Recorder(
            fields; diagnostics = run.flow, keyframes = EquispacedTimes(41),
            preview = nothing
        )
        result = simulate!(run.flow, target_time; observer = recorder)
    end
    println("sessile_droplet: size=$size, runtime=$elapsed s")
    print_simulation_summary(run.flow, result)
    energy = diagnostic_history(run.flow, :kinetic_energy)
    volume = diagnostic_history(run.flow, :phase_volume)
    liquid = [value[1] for value in volume.values]
    println(
        "initial/final kinetic energy = ", (first(energy.values), last(energy.values)),
        "; maximum kinetic energy = ", maximum(energy.values),
        "; initial/final (liquid, gas) volumes = ",
        (first(volume.values), last(volume.values)),
        "; maximum sampled liquid-volume error = ", maximum(abs.(liquid .- first(liquid)))
    )

    # The lower half of the domain, which holds the droplet, with square cells.
    store = snapshot_store(recorder)
    export_figure(
        joinpath(outdir, "snapshots.png"), store; field = :volume_fraction,
        at = [info.time for info in keyframes(store)][[1, cld(end, 2), end]],
        camera = (0.0, 1.0, 0.0, 0.5), blocks = true
    )
    export_animation(
        joinpath(outdir, "relaxation.gif"), store; field = :velocity_magnitude,
        camera = (0.0, 1.0, 0.0, 0.5), diagnostics = [:kinetic_energy], framerate = 10,
        size = (1100, 380)
    )

    figure = Figure(size = (850, 350))
    diagnosticsgrid!(
        figure[1, 1], store; names = [:kinetic_energy, :phase_volume],
        direction = :row
    )
    save(joinpath(outdir, "diagnostics.png"), figure)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
