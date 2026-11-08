using TwoPhaseAMR, CairoMakie

Base.include(@__MODULE__, joinpath(@__DIR__, "capillary_wave.jl"))

function capillary_figure_case(model, target_time)
    local run, recorder, result
    elapsed = @elapsed begin
        run = run_example(max_level = 2, final_time = 0.0, velocity_model = model)
        fields = VisualizationFieldRegistry()
        register_default_fields!(fields, :volume_fraction, run.flow.interface)
        recorder = Recorder(fields; keyframes = EquispacedTimes(3), preview = nothing)
        result = simulate!(run.flow, target_time; observer = recorder)
    end
    println("capillary_wave: model=$(typeof(model)), runtime=$elapsed s")
    print_simulation_summary(run.flow, result)
    numerical = diagnostic_history(run.flow, :amplitude)
    reference = diagnostic_history(run.flow, :analytical_amplitude)
    println("final time = ", last(numerical.times), "; numerical amplitude = ",
        last(numerical.values), "; Prosperetti amplitude = ", last(reference.values),
        "; maximum sampled absolute error = ",
        maximum(abs.(numerical.values .- reference.values)))
    return (; flow = run.flow, recorder)
end

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    # The example's default is already the docs-size case: L=2 and one period.
    target_time = 2pi / sqrt((2pi)^3 / (1 + 1.0e-3))
    one = capillary_figure_case(OneVelocityModel(), target_time)
    two = capillary_figure_case(TwoVelocityModel(), target_time)

    figure = Figure(size = (760, 400))
    axis = Axis(figure[1, 1]; xlabel = "t", ylabel = "interface displacement at x = 0",
        title = "Capillary-wave amplitude")
    plot_diagnostics!(axis, one.flow, :amplitude; labels = "one velocity")
    plot_diagnostics!(axis, two.flow, :amplitude; labels = "two velocities")
    reference = diagnostic_history(one.flow, :analytical_amplitude)
    lines!(axis, reference.times, reference.values; color = :black,
        linestyle = :dash, label = "Prosperetti")
    axislegend(axis)
    save(joinpath(outdir, "amplitude.png"), figure)

    store = snapshot_store(one.recorder)
    checkpoints = [store[info] for info in keyframes(store)]
    figure = Figure(size = (900, 280))
    for (i, frame) in enumerate((first(checkpoints), last(checkpoints)))
        axis = snapshot_figure!(figure[1, i], frame; field = :volume_fraction,
            title = "t = $(round(frame.time; digits = 3))", colorbar = i == 2)
        axis.aspect[] = nothing
        xlims!(axis, 0, 1)
        ylims!(axis, 0.16, 0.34)
    end
    save(joinpath(outdir, "interface.png"), figure)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
