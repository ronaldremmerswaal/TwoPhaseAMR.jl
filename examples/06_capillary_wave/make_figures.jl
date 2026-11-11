using TwoPhaseAMR, CairoMakie

Base.include(@__MODULE__, joinpath(@__DIR__, "capillary_wave.jl"))

function capillary_figure_case(model, target_time)
    local run, recorder, result
    elapsed = @elapsed begin
        run = run_example(max_level = 4, final_time = 0.0, velocity_model = model)
        fields = VisualizationFieldRegistry()
        register_default_fields!(fields, run.flow)
        recorder = Recorder(
            fields; diagnostics = run.flow, keyframes = EquispacedTimes(41),
            preview = nothing
        )
        result = simulate!(run.flow, target_time; observer = recorder)
    end
    println("capillary_wave: model=$(typeof(model)), runtime=$elapsed s")
    print_simulation_summary(run.flow, result)
    numerical = diagnostic_history(run.flow, :amplitude)
    reference = diagnostic_history(run.flow, :analytical_amplitude)
    println(
        "final time = ", last(numerical.times), "; numerical amplitude = ",
        last(numerical.values), "; Prosperetti amplitude = ", last(reference.values),
        "; maximum sampled absolute error = ",
        maximum(abs.(numerical.values .- reference.values))
    )
    return (; flow = run.flow, recorder)
end

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    # The example's default is already the docs-size case: L=4 and one period.
    target_time = 2pi / sqrt((2pi)^3 / (1 + 1.0e-3))
    one = capillary_figure_case(OneVelocityModel(), target_time)
    two = capillary_figure_case(TwoVelocityModel(), target_time)

    figure = Figure(size = (760, 400))
    axis = only(
        diagnosticsgrid!(
            figure[1, 1], snapshot_store(one.recorder);
            names = [:amplitude], references = Dict(:amplitude => :analytical_amplitude),
            legend = nothing
        )
    )
    amplitude = diagnostic_history(two.flow, :amplitude)
    lines!(axis, amplitude.times, amplitude.values; label = "Numerical amplitude, two velocities")
    axislegend(axis; labelsize = 11)
    save(joinpath(outdir, "amplitude.png"), figure)

    # One wavelength by 0.4 wavelengths around the interface, with square cells.
    store = snapshot_store(one.recorder)
    times = [info.time for info in keyframes(store)]
    export_figure(
        joinpath(outdir, "interface.png"), store; field = :volume_fraction,
        at = [first(times), last(times)], camera = (0.0, 1.0, 0.05, 0.45), blocks = true
    )
    export_animation(
        joinpath(outdir, "wave.gif"), store; field = :velocity_magnitude,
        camera = (0.0, 1.0, 0.05, 0.45), diagnostics = [:amplitude],
        references = Dict(:amplitude => :analytical_amplitude), fps = 10, duration = 4.0,
        size = (1150, 340)
    )
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
