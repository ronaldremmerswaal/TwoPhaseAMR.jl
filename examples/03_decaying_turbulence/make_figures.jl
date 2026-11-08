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
        state -> diagnostic_value(KineticEnergy(), state);
        label = "Kinetic energy", dimension = PhysicalDimension(1, 1, -2)
    )
    fields = VisualizationFieldRegistry()
    register_field!(
        fields, :vorticity, flow.velocity, Vorticity();
        label = "Vorticity", colormap = :balance
    )
    recorder = Recorder(
        fields; diagnostics = flow, keyframes = EquispacedTimes(41),
        preview = nothing
    )
    result = simulate!(flow, final_time; observer = recorder)
    simulation_seconds = (time_ns() - started) / 1.0e9
    print_simulation_summary(flow, result)

    store = snapshot_store(recorder)
    times = [info.time for info in keyframes(store)]
    export_figure(
        joinpath(outdir, "vorticity.png"), store; field = :vorticity,
        at = [first(times), last(times)], blocks = true
    )
    figure = Figure(size = (900, 360))
    diagnosticsgrid!(
        figure[1, 1], store; names = [:projected_kinetic_energy, :cells],
        direction = :row
    )
    save(joinpath(outdir, "energy_cells.png"), figure)
    export_animation(
        joinpath(outdir, "vorticity.gif"), store; field = :vorticity,
        blocks = true, diagnostics = [:projected_kinetic_energy, :cells], framerate = 8
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
