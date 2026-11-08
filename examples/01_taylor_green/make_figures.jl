using CairoMakie
Base.include(@__MODULE__, joinpath(@__DIR__, "taylor_green.jl"))

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    started = time_ns()
    final_time = size === :docs ? 5.0 : 0.2
    reynolds = 50.0
    flow = run_example(; reynolds, max_level = 2, final_time = 0.0).flow
    fields = VisualizationFieldRegistry()
    register_field!(fields, :vorticity, flow.velocity, Vorticity();
        label = "Vorticity", colormap = :balance)
    recorder = Recorder(fields; diagnostics = flow, keyframes = EquispacedTimes(3),
        preview = nothing)
    result = simulate!(flow, final_time; observer = recorder)
    simulation_seconds = (time_ns() - started) / 1e9
    print_simulation_summary(flow, result)

    store = snapshot_store(recorder)
    export_figure(joinpath(outdir, "vorticity.png"), store; field = :vorticity,
        at = [info.time for info in keyframes(store)], blocks = true)

    exact_energy(t) = pi^2 * exp(-4t / reynolds)
    figure = Figure(size = (700, 420))
    diagnosticsgrid!(figure[1, 1], store; names = [:kinetic_energy],
        references = Dict(:kinetic_energy => exact_energy))
    save(joinpath(outdir, "kinetic_energy.png"), figure)
    energy = diagnostic_history(flow, :kinetic_energy)
    println("Exact kinetic energy: initial = ", exact_energy(first(energy.times)),
        ", final = ", exact_energy(last(energy.times)))
    println("Final kinetic-energy error (numerical - exact) = ",
        last(energy.values) - exact_energy(last(energy.times)))
    println("Maximum absolute kinetic-energy error = ",
        maximum(abs(value - exact_energy(time)) for (time, value) in zip(energy.times, energy.values)))
    println("Setup + simulation: ", simulation_seconds, " s; including figures: ",
        (time_ns() - started) / 1e9, " s (package loading excluded)")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
