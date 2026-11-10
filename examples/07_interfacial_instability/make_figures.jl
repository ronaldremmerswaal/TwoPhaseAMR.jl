using TwoPhaseAMR, CairoMakie, Printf

Base.include(@__MODULE__, joinpath(@__DIR__, "interfacial_instability.jl"))

function instability_figure_case(model, target_time)
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
    println("interfacial_instability: model=$(typeof(model)), runtime=$elapsed s")
    print_simulation_summary(run.flow, result)
    numerical = diagnostic_history(run.flow, :amplitude)
    reference = diagnostic_history(run.flow, :analytical_amplitude)
    println(
        "final time = ", last(numerical.times), "; numerical amplitude = ",
        last(numerical.values), "; linear amplitude = ", last(reference.values),
        "; maximum sampled absolute difference from linear theory = ",
        maximum(abs.(numerical.values .- reference.values))
    )
    return (; flow = run.flow, recorder)
end

# Repository README animation: Kelvin–Helmholtz roll-up with two velocities.
# Without gravity (`bond = 0`), at `weber = 3` and `density_ratio = 1.2`, the
# wavelength-one mode grows at the linear rate `sqrt(weber - 1)`; the
# level-`level` run continues to `t = 4.5`, well into the non-linear roll-up.
# The horizontal velocity is the liquid's in the liquid (initially -0.25) and
# the gas's in the gas (0.30), so the shear layer is a sharp jump at the
# interface. Light and dark variants share the frames and one range, in which
# only the peaks at the fold tips saturate; 31 colour levels keep the GIFs
# small. The dark variant is muted for GitHub's dark theme: `:berlin` without
# its lightest ends (a dark neutral), grey text, interface, and faint blocks.
function readme_animations(outdir; level = 5)
    weber = 3.0
    final_time = 4.5
    run = run_example(;
        max_level = level, weber, bond = 0.0, density_ratio = 1.2,
        final_time = 0.0
    )
    fields = VisualizationFieldRegistry()
    register_default_fields!(fields, run.flow)
    recorder = Recorder(fields; keyframes = EquispacedTimes(55), preview = nothing)
    seconds = @elapsed result = simulate!(run.flow, final_time; observer = recorder)
    println(
        "README animation: level ", level, ", ", result.steps, " steps to t = ",
        final_time, " (", final_time * sqrt(weber - 1), " linear e-folding times) in ",
        seconds, " s; ", TwoPhaseAMR.ncells(run.flow.solver.mesh), " final cells"
    )
    levels = 31
    animation(path, colormap; kwargs...) = export_animation(
        path, recorder;
        field = :velocity_x, camera = (0.0, 1.0, -0.4, 0.4), colorrange = (-0.6, 0.6),
        colormap = cgrad(colormap, levels; categorical = true), blocks = true,
        title = frame -> @sprintf("t = %.2f", frame.time), framerate = 12,
        size = (640, 480), kwargs...
    )
    animation(joinpath(outdir, "readme_light.gif"), :balance)
    muted = [get(CairoMakie.Makie.ColorSchemes.berlin, x) for x in range(0.15, 0.85; length = levels)]
    with_theme(theme_dark(); textcolor = :gray80) do
        animation(
            joinpath(outdir, "readme_dark.gif"), muted; interfacecolor = :gray75,
            blockcolor = (:white, 0.15)
        )
    end
    return nothing
end

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    # With We=0.5 and Bo=0.75 the linear growth rate is 0.5.
    target_time = acosh(2) / 0.5
    two = instability_figure_case(TwoVelocityModel(), target_time)
    one = instability_figure_case(OneVelocityModel(), target_time)

    figure = Figure(size = (760, 400))
    axis = only(
        diagnosticsgrid!(
            figure[1, 1], snapshot_store(two.recorder);
            names = [:amplitude], references = Dict(:amplitude => :analytical_amplitude),
            legend = nothing
        )
    )
    amplitude = diagnostic_history(one.flow, :amplitude)
    lines!(axis, amplitude.times, amplitude.values; label = "Numerical amplitude, one velocity")
    axislegend(axis; position = :lt, labelsize = 11)
    save(joinpath(outdir, "amplitude.png"), figure)

    # One wavelength by 0.4 wavelengths around the interface, with square cells.
    window = (0.0, 1.0, -0.2, 0.2)
    two_store = snapshot_store(two.recorder)
    one_store = snapshot_store(one.recorder)
    first_frame = two_store[first(keyframes(two_store))]
    panels = (
        (first_frame, "initial"),
        (two_store[last(keyframes(two_store))], "two velocities"),
        (one_store[last(keyframes(one_store))], "one velocity"),
    )
    figure = Figure(size = (1000, 300))
    plot = nothing
    for (i, (frame, label)) in enumerate(panels)
        axis = fieldaxis(
            figure[1, i]; camera = window,
            title = "$label, t = $(round(frame.time; digits = 3))"
        )
        plot = amrfield!(axis, frame, :volume_fraction; blocks = true)
        colsize!(figure.layout, i, Aspect(1, 2.5))
    end
    Colorbar(figure[1, 4], plot; label = "Volume fraction")
    resize_to_layout!(figure)
    save(joinpath(outdir, "interfaces.png"), figure)
    export_animation(
        joinpath(outdir, "instability.gif"), two_store; field = :velocity_x,
        camera = window, diagnostics = [:amplitude],
        references = Dict(:amplitude => :analytical_amplitude), framerate = 10,
        size = (1150, 340)
    )
    readme_animations(outdir)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
