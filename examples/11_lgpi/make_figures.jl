using CairoMakie

Base.include(@__MODULE__, joinpath(@__DIR__, "lgpi.jl"))

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    initial = lgpi_initial_condition()
    target_time = 0.6sqrt((1000 + 1) / (1000 - 1) * 2pi * initial.wavelength / 9.81)
    level = size == :docs ? 2 : 3
    local flow, result, recorder
    elapsed = @elapsed begin
        flow = run_example(max_level = level, final_time = 0).flow
        fields = VisualizationFieldRegistry()
        register_field!(fields, :velocity_vorticity, flow.velocity, Vorticity();
            label = "Vorticity", colormap = :balance)
        register_interface!(fields, flow.interface)
        register_geometry!(fields, flow.solver.geometry)
        recorder = Recorder(fields; keyframes = EquispacedTimes(21), preview = nothing)
        result = simulate!(flow, target_time; observer = recorder)
    end
    println("Setup and simulation time (s; includes first-call compilation): ", elapsed)
    print_simulation_summary(flow, result)

    store = snapshot_store(recorder)
    checkpoints = [store[info] for info in keyframes(store)]
    figure = Figure(size = size == :docs ? (900, 300) : (1200, 400))
    for (panel, index) in enumerate((1, 11, 21))
        frame = checkpoints[index]
        snapshot_figure!(figure[1, panel], frame; field = :velocity_vorticity,
            colormap = :balance, title = "t = $(round(frame.time; sigdigits = 4))")
    end
    save(joinpath(outdir, "hero.png"), figure)
    record_checkpoints(joinpath(outdir, "impact.gif"), recorder;
        field = :velocity_vorticity, colormap = :balance, framerate = 8,
        size = size == :docs ? (600, 400) : (800, 530))
    return (; flow, result, recorder)
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
