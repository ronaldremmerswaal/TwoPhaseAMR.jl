using CairoMakie
Base.include(@__MODULE__, joinpath(@__DIR__, "lid_driven_cavity.jl"))

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    started = time_ns()
    dt = size === :docs ? 0.005 : nothing
    flow = run_example(; max_level = 1, final_time = 0.0, dt).flow
    fields = VisualizationFieldRegistry()
    register_field!(fields, :speed, flow.velocity, CentredMagnitude(); label = "Speed")
    session = VisualizationSession(fields; live_every = typemax(Int))
    result = simulate!(flow, 15.0; visualization = session)
    simulation_seconds = (time_ns() - started) / 1.0e9
    print_simulation_summary(flow, result)

    figure = snapshot_figure(
        latest_frame(session); field = :speed, blocks = true,
        size = (650, 550), title = "Speed at t = 15"
    )
    save(joinpath(outdir, "speed.png"), figure)
    figure = Figure(size = (700, 420))
    axis = Axis(figure[1, 1]; xlabel = "Time", ylabel = "Total kinetic energy")
    plot_diagnostics!(axis, flow, :kinetic_energy)
    save(joinpath(outdir, "kinetic_energy.png"), figure)
    println(
        "Setup + simulation: ", simulation_seconds, " s; including figures: ",
        (time_ns() - started) / 1.0e9, " s (package loading excluded)"
    )
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
