using CairoMakie

Base.include(@__MODULE__, joinpath(@__DIR__, "sloped_sessile_droplet.jl"))

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    target_time = 4sqrt((1 + 1 / 1000) * 0.16^3)
    local flow, result, recorder
    elapsed = @elapsed begin
        flow = run_example(max_level = 3, final_time = 0).flow
        fields = VisualizationFieldRegistry()
        register_default_fields!(fields, :volume_fraction, flow.interface)
        register_geometry!(fields, flow.solver.geometry)
        recorder = Recorder(
            fields; diagnostics = flow, keyframes = EquispacedTimes(41),
            preview = nothing
        )
        result = simulate!(flow, target_time; observer = recorder)
    end
    println("Setup and simulation time (s; includes first-call compilation): ", elapsed)
    print_simulation_summary(flow, result)

    store = snapshot_store(recorder)
    export_figure(
        joinpath(outdir, "hero.png"), store; field = :volume_fraction,
        at = [info.time for info in keyframes(store)][[1, cld(end, 2), end]]
    )
    energy = Figure(size = size == :docs ? (700, 420) : (900, 540))
    diagnosticsgrid!(energy[1, 1], store; names = [:kinetic_energy])
    save(joinpath(outdir, "kinetic_energy.png"), energy)
    export_animation(
        joinpath(outdir, "relaxation.gif"), store; field = :volume_fraction,
        diagnostics = [:kinetic_energy], framerate = 10, size = (900, 380)
    )
    return (; flow, result, recorder)
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
