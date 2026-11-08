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
    recorder = Recorder(
        fields; diagnostics = flow, keyframes = EquispacedTimes(2),
        preview = nothing
    )
    result = simulate!(flow, 15.0; observer = recorder)
    simulation_seconds = (time_ns() - started) / 1.0e9
    print_simulation_summary(flow, result)

    store = snapshot_store(recorder)
    export_figure(
        joinpath(outdir, "speed.png"), store; field = :speed, blocks = true,
        title = "Speed at t = 15"
    )
    figure = Figure(size = (700, 420))
    diagnosticsgrid!(figure[1, 1], store; names = [:kinetic_energy])
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
