using CairoMakie

Base.include(@__MODULE__, joinpath(@__DIR__, "stokes3_wave.jl"))

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    initial = stokes3_initial_condition()
    target_time = 2pi / initial.frequency
    level = size == :docs ? 3 : 5
    local flow, result, session
    elapsed = @elapsed begin
        flow = run_example(max_level = level, final_time = 0).flow
        fields = VisualizationFieldRegistry()
        register_field!(
            fields, :velocity_vorticity, flow.velocity, Vorticity();
            label = "Vorticity", colormap = :balance
        )
        register_interface!(fields, flow.interface)
        session = VisualizationSession(fields; checkpoints = 21, live_every = typemax(Int))
        result = simulate!(flow, target_time; visualization = session)
    end
    println("Setup and simulation time (s; includes first-call compilation): ", elapsed)
    print_simulation_summary(flow, result)

    xs = range(-0.5, 0.5; length = 201)[1:(end - 1)]
    numerical = [interface_height(flow.interface; position = x) for x in xs]
    reference = initial.elevation.(xs)
    errors = numerical .- reference
    println(
        "Final interface versus third-order initial profile: sample RMS = ",
        sqrt(sum(abs2, errors) / length(errors)), ", sample maximum = ", maximum(abs, errors)
    )
    figure = Figure(size = size == :docs ? (800, 440) : (1000, 550))
    axis = Axis(figure[1, 1]; xlabel = "x", ylabel = "Interface elevation")
    lines!(axis, xs, numerical; label = "Computed after one period")
    lines!(axis, xs, reference; label = "Third-order initial profile", linestyle = :dash)
    axislegend(axis)
    save(joinpath(outdir, "hero.png"), figure)
    checkpoints = visualization_checkpoints(session)
    save(
        joinpath(outdir, "vorticity.png"), snapshot_figure(
            last(checkpoints);
            field = :velocity_vorticity, colormap = :balance,
            size = size == :docs ? (700, 500) : (900, 650),
            title = "Vorticity at t = $(round(last(checkpoints).frame.time; sigdigits = 4))"
        )
    )
    record_checkpoints(
        joinpath(outdir, "wave.gif"), session;
        field = :velocity_vorticity, colormap = :balance, framerate = 8,
        size = size == :docs ? (600, 420) : (800, 560)
    )
    return (; flow, result, session, errors)
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
