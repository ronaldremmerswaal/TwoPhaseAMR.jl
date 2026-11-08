using CairoMakie
Base.include(@__MODULE__, joinpath(@__DIR__, "cylinder_flow.jl"))

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    started = time_ns()
    max_level, final_time = size === :docs ? (2, 80.0) : (3, 80.0)
    # Shedding develops after t ≈ 40; animate the last 16 time units.
    window = (final_time - 16, final_time)
    flow = run_example(; max_level, final_time = 0.0).flow
    fields = VisualizationFieldRegistry()
    register_field!(fields, :vorticity, flow.velocity, Vorticity();
        label = "Vorticity", colormap = :balance)
    register_geometry!(fields, flow.solver.geometry)
    recorder = Recorder(fields; diagnostics = flow, keyframes = AtTimes(collect(range(window...; length = 33))),
        preview = nothing)
    result = simulate!(flow, final_time; observer = recorder)
    simulation_seconds = (time_ns() - started) / 1e9
    print_simulation_summary(flow, result)

    store = snapshot_store(recorder)
    export_figure(joinpath(outdir, "vorticity.png"), store; field = :vorticity,
        blocks = true, title = "Vorticity at t = $final_time")
    figure = Figure(size = (700, 420))
    axis = only(diagnosticsgrid!(figure[1, 1], store; names = [:pressure_drag]))
    # The impulsive start gives a huge first force sample; show the developed range.
    ylims!(axis, 0.5, 1.0)
    save(joinpath(outdir, "pressure_drag.png"), figure)
    export_animation(joinpath(outdir, "shedding.gif"), store; field = :vorticity, window,
        blocks = true, colorrange = (-3, 3), framerate = 10, size = (900, 400))
    println("Setup + simulation: ", simulation_seconds, " s; including figures: ",
        (time_ns() - started) / 1e9, " s (package loading excluded)")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
