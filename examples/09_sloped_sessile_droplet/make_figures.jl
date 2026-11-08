using CairoMakie

Base.include(@__MODULE__, joinpath(@__DIR__, "sloped_sessile_droplet.jl"))

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    target_time = 4sqrt((1 + 1 / 1000) * 0.16^3)
    local flow, result, session
    elapsed = @elapsed begin
        flow = run_example(final_time = 0).flow
        fields = VisualizationFieldRegistry()
        register_default_fields!(fields, :volume_fraction, flow.interface)
        register_geometry!(fields, flow.solver.geometry)
        session = VisualizationSession(fields; checkpoints = 3, live_every = typemax(Int))
        result = simulate!(flow, target_time; visualization = session)
    end
    println("Setup and simulation time (s; includes first-call compilation): ", elapsed)
    print_simulation_summary(flow, result)

    figure = Figure(size = size == :docs ? (900, 340) : (1200, 440))
    for (index, checkpoint) in enumerate(visualization_checkpoints(session))
        snapshot_figure!(
            figure[1, index], checkpoint; field = :volume_fraction,
            title = "t = $(round(checkpoint.frame.time; sigdigits = 4))"
        )
    end
    save(joinpath(outdir, "hero.png"), figure)
    energy = Figure(size = size == :docs ? (700, 420) : (900, 540))
    axis = Axis(energy[1, 1]; xlabel = "Time", ylabel = "Kinetic energy")
    plot_diagnostics!(axis, flow, :kinetic_energy)
    save(joinpath(outdir, "kinetic_energy.png"), energy)
    return (; flow, result, session)
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
