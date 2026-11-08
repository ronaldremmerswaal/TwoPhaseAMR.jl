using TwoPhaseAMR, CairoMakie

Base.include(@__MODULE__, joinpath(@__DIR__, "interfacial_instability.jl"))

function instability_figure_case(model, target_time)
    local run, session, result
    elapsed = @elapsed begin
        run = run_example(max_level = 1, final_time = 0.0, velocity_model = model)
        fields = VisualizationFieldRegistry()
        register_default_fields!(fields, :volume_fraction, run.flow.interface)
        session = VisualizationSession(fields; checkpoints = 3, live_every = typemax(Int))
        result = simulate!(run.flow, target_time; visualization = session)
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
    return (; flow = run.flow, session)
end

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    # With We=0.5 and Bo=0.75 the linear growth rate is 0.5.
    target_time = acosh(2) / 0.5
    two = instability_figure_case(TwoVelocityModel(), target_time)
    one = instability_figure_case(OneVelocityModel(), target_time)

    figure = Figure(size = (760, 400))
    axis = Axis(
        figure[1, 1]; xlabel = "t", ylabel = "interface height at x = 0",
        title = "Interfacial instability"
    )
    plot_diagnostics!(axis, two.flow, :amplitude; labels = "two velocities")
    plot_diagnostics!(axis, one.flow, :amplitude; labels = "one velocity")
    reference = diagnostic_history(two.flow, :analytical_amplitude)
    lines!(
        axis, reference.times, reference.values; color = :black,
        linestyle = :dash, label = "linear theory"
    )
    axislegend(axis)
    save(joinpath(outdir, "amplitude.png"), figure)

    two_checkpoints = visualization_checkpoints(two.session)
    one_checkpoints = visualization_checkpoints(one.session)
    panels = (
        (first(two_checkpoints), "initial"),
        (last(two_checkpoints), "two velocities"),
        (last(one_checkpoints), "one velocity"),
    )
    figure = Figure(size = (900, 270))
    for (i, (checkpoint, label)) in enumerate(panels)
        frame = checkpoint.frame
        axis = snapshot_figure!(
            figure[1, i], frame; field = :volume_fraction,
            title = "$label, t = $(round(frame.time; digits = 3))", colorbar = i == 3
        )
        axis.aspect[] = nothing
        xlims!(axis, 0, 1)
        ylims!(axis, -0.08, 0.08)
    end
    save(joinpath(outdir, "interfaces.png"), figure)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
