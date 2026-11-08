using TwoPhaseAMR, CairoMakie
CairoMakie.activate!(visible = false)

Base.include(@__MODULE__, joinpath(@__DIR__, "vortex_reverse.jl"))

function frame_volume(frame)
    nx, ny = frame.topology.block_size
    fractions = frame[:volume_fraction]
    return sum(
        fractions[(block - 1) * nx * ny + (j - 1) * nx + i] *
            spacing[1] * spacing[2]
            for (block, spacing) in enumerate(frame.topology.spacings)
            for j in 1:ny for i in 1:nx
    )
end

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    level = size === :docs ? 3 : 5
    # This prescribed-velocity example owns its time loop and its recorder; it does not return a FlowSimulation to pass to simulate!.
    local run
    elapsed = @elapsed run = run_example(max_level = level, final_time = 8.0)
    store = snapshot_store(run.recorder)
    frames = [store[info] for info in keyframes(store)]
    times = [frame.time for frame in frames]
    volumes = [frame_volume(frame) for frame in frames]
    errors = abs.(volumes .- first(volumes))
    println(
        "vortex_reverse: size=$size, runtime=$elapsed s, cells=",
        ncells(run.interface.mesh), ", result=", run.result
    )
    println(
        "checkpoint times = ", times, "; liquid volumes = ", volumes,
        "; absolute volume errors = ", errors
    )

    export_figure(
        joinpath(outdir, "snapshots.png"), store; field = :volume_fraction,
        at = times[[1, 11, end]]
    )

    figure = Figure(size = (760, 340))
    axis = fieldaxis(
        figure[1, 1]; camera = (0, 1, 0, 1), decorations = :full,
        title = "Interface recovery"
    )
    for (frame, color, label) in (
            (first(frames), :dodgerblue, "initial"),
            (last(frames), :orangered, "final"),
        )
        segments = (segment for phase in frame.phases for segment in phase.segments)
        points = Point2f[Point2f(point...) for segment in segments for point in segment]
        linesegments!(axis, points; color, linewidth = 2, label)
    end
    axislegend(axis; position = :lb)
    axis = Axis(
        figure[1, 2]; title = "Liquid volume error", xlabel = "t",
        ylabel = "|V(t) − V(0)|"
    )
    lines!(axis, times, errors; color = :dodgerblue)
    scatter!(axis, times, errors; color = :dodgerblue, markersize = 5)
    save(joinpath(outdir, "recovery.png"), figure)
    export_animation(
        joinpath(outdir, "reversal.gif"), store; field = :volume_fraction,
        framerate = 6, size = (520, 450)
    )
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
