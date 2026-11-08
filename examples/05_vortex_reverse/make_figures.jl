using TwoPhaseAMR, CairoMakie
CairoMakie.activate!(visible = false)

Base.include(@__MODULE__, joinpath(@__DIR__, "vortex_reverse.jl"))

# A frame stores one fraction and four physical vertices per leaf cell.
function checkpoint_volume(frame)
    points = frame.topology.cell_geometry.points
    fractions = frame[:volume_fraction]
    return sum(
        fractions[i] * (points[4i - 1][1] - points[4i - 3][1]) *
            (points[4i - 1][2] - points[4i - 3][2])
            for i in eachindex(fractions)
    )
end

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    level = size === :docs ? 3 : 5
    # This prescribed-velocity example owns its time loop and its checkpoint
    # session; it does not return a FlowSimulation to pass to simulate!.
    local run
    elapsed = @elapsed run = run_example(max_level = level, final_time = 8.0, visualize = true)
    checkpoints = visualization_checkpoints(run.visualization.session)
    times = [checkpoint.frame.time for checkpoint in checkpoints]
    volumes = [checkpoint_volume(checkpoint.frame) for checkpoint in checkpoints]
    errors = abs.(volumes .- first(volumes))
    println(
        "vortex_reverse: size=$size, runtime=$elapsed s, cells=",
        ncells(run.interface.mesh), ", result=", run.result
    )
    println(
        "checkpoint times = ", times, "; liquid volumes = ", volumes,
        "; absolute volume errors = ", errors
    )

    figure = Figure(size = (900, 330))
    for (i, index) in enumerate((1, 11, length(checkpoints)))
        frame = checkpoints[index].frame
        snapshot_figure!(
            figure[1, i], frame; field = :volume_fraction,
            title = "t = $(round(frame.time; digits = 3))", colorbar = i == 3
        )
    end
    save(joinpath(outdir, "snapshots.png"), figure)

    figure = Figure(size = (760, 340))
    axis = Axis(
        figure[1, 1]; aspect = DataAspect(), title = "Interface recovery",
        xlabel = "x", ylabel = "y"
    )
    for (checkpoint, color, label) in (
            (first(checkpoints), :dodgerblue, "initial"),
            (last(checkpoints), :orangered, "final"),
        )
        segments = checkpoint.frame.primitives[run.interface]
        points = Point2f[Point2f(point...) for segment in segments for point in segment]
        linesegments!(axis, points; color, linewidth = 2, label)
    end
    xlims!(axis, 0, 1)
    ylims!(axis, 0, 1)
    axislegend(axis; position = :lb)
    axis = Axis(
        figure[1, 2]; title = "Liquid volume error", xlabel = "t",
        ylabel = "|V(t) − V(0)|"
    )
    lines!(axis, times, errors; color = :dodgerblue)
    scatter!(axis, times, errors; color = :dodgerblue, markersize = 5)
    save(joinpath(outdir, "recovery.png"), figure)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
