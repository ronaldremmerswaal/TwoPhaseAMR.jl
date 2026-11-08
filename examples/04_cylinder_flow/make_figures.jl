using CairoMakie
Base.include(@__MODULE__, joinpath(@__DIR__, "cylinder_flow.jl"))

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    started = time_ns()
    max_level, final_time = size === :docs ? (2, 10.0) : (3, 40.0)
    flow = run_example(; max_level, final_time = 0.0).flow
    fields = VisualizationFieldRegistry()
    register_field!(fields, :vorticity, flow.velocity, Vorticity();
        label = "Vorticity", colormap = :balance)
    register_geometry!(fields, flow.solver.geometry)
    session = VisualizationSession(fields; live_every = typemax(Int))
    result = simulate!(flow, final_time; visualization = session)
    simulation_seconds = (time_ns() - started) / 1e9
    print_simulation_summary(flow, result)

    figure = snapshot_figure(latest_frame(session); field = :vorticity,
        blocks = true, geometry = true, colormap = :balance,
        size = (900, 520), title = "Vorticity at t = $final_time")
    save(joinpath(outdir, "vorticity.png"), figure)
    figure = Figure(size = (700, 420))
    axis = Axis(figure[1, 1]; xlabel = "Time", ylabel = "Pressure drag coefficient")
    plot_diagnostics!(axis, flow, :pressure_drag)
    save(joinpath(outdir, "pressure_drag.png"), figure)
    println("Setup + simulation: ", simulation_seconds, " s; including figures: ",
        (time_ns() - started) / 1e9, " s (package loading excluded)")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
