using CairoMakie
Base.include(@__MODULE__, joinpath(@__DIR__, "decaying_turbulence.jl"))

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    started = time_ns()
    max_level, final_time = size === :docs ? (3, 2.0) : (4, 10.0)
    flow = run_example(; max_level, final_time = 0.0).flow
    # The example projects after installing its diagnostics; sample this one after projection.
    add_diagnostic!(
        flow, :projected_kinetic_energy,
        state -> diagnostic_value(KineticEnergy(), state)
    )
    fields = VisualizationFieldRegistry()
    register_field!(
        fields, :vorticity, flow.velocity, Vorticity();
        label = "Vorticity", colormap = :balance
    )
    session = VisualizationSession(fields; checkpoints = 11, live_every = typemax(Int))
    result = simulate!(flow, final_time; visualization = session)
    simulation_seconds = (time_ns() - started) / 1.0e9
    print_simulation_summary(flow, result)

    checkpoints = visualization_checkpoints(session)
    figure = Figure(size = (900, 410))
    for (i, checkpoint) in enumerate((first(checkpoints), last(checkpoints)))
        snapshot_figure!(
            figure[1, i], checkpoint; field = :vorticity,
            blocks = true, colormap = :balance,
            title = "t = $(round(checkpoint.frame.time; digits = 3))"
        )
    end
    save(joinpath(outdir, "vorticity.png"), figure)
    figure = Figure(size = (900, 360))
    energy_axis = Axis(figure[1, 1]; xlabel = "Time", ylabel = "Total kinetic energy")
    cells_axis = Axis(figure[1, 2]; xlabel = "Time", ylabel = "Active cells")
    plot_diagnostics!(energy_axis, flow, :projected_kinetic_energy)
    plot_diagnostics!(cells_axis, flow, :cells)
    save(joinpath(outdir, "energy_cells.png"), figure)
    record_checkpoints(
        joinpath(outdir, "vorticity.gif"), session;
        field = :vorticity, blocks = true, colormap = :balance,
        size = (400, 300), framerate = 4
    )
    println(
        "Setup + simulation: ", simulation_seconds, " s; including figures: ",
        (time_ns() - started) / 1.0e9, " s (package loading excluded)"
    )
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
