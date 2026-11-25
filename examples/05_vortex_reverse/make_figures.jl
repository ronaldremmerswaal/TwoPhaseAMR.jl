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
function frame_scalar_content(frame)
    nx, ny = frame.topology.block_size
    fractions = frame[:volume_fraction]
    scalar = frame[:liquid_scalar].liquid
    return sum(
        fractions[k] * scalar[k] * spacing[1] * spacing[2]
            for (block, spacing) in enumerate(frame.topology.spacings)
            for j in 1:ny for i in 1:nx
            for k in ((block - 1) * nx * ny + (j - 1) * nx + i,)
    )
end

function scalar_recovery(run)
    mesh = run.interface.mesh
    shape(x, y) = hypot(x - 0.5, y - 0.75) - 0.15
    reference = InterfaceField(mesh, shape; lipschitz = 1, atol = 1.0e-7, maxdepth = 24)
    initial = PhaseScalar(reference; liquid = run.scalar_profile, gas = 0)
    return sum(
        block.volume * sum(
            abs.(
                interior(run.interface, b) .* interior(run.scalar.liquid, b) .-
                    interior(reference, b) .* interior(initial.liquid, b)
            )
        ) for (b, block) in pairs(mesh.blocks)
    )
end


function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    level = size === :docs ? 4 : 6
    # This prescribed-velocity example owns its time loop and its recorder; it does not return a FlowSimulation to pass to simulate!.
    local run
    # 81 keyframes, one per GIF frame; the snapshots and the volume-error plot
    # use every fourth, the 21 times of the example's default schedule.
    elapsed = @elapsed run = run_example(
        max_level = level, final_time = 8.0,
        recorder_options = (; keyframes = EquispacedTimes(81))
    )
    store = snapshot_store(run.recorder)
    frames = [store[info] for info in keyframes(store)][1:4:end]
    times = [frame.time for frame in frames]
    volumes = [frame_volume(frame) for frame in frames]
    errors = abs.(volumes .- first(volumes))
    contents = frame_scalar_content.(frames)
    scalar_error = scalar_recovery(run)
    println(
        "vortex_reverse: size=$size, runtime=$elapsed s, cells=",
        ncells(run.interface.mesh), ", result=", run.result
    )
    println(
        "keyframe times = ", times, "; liquid volumes = ", volumes,
        "; absolute volume errors = ", errors
    )
    println(
        "scalar L1 recovery error = ", scalar_error,
        "; contents = ", contents,
        "; maximum retained content drift = ", maximum(abs.(contents .- first(contents)))
    )

    export_figure(
        joinpath(outdir, "snapshots.png"), store; field = :volume_fraction,
        at = times[[1, 11, end]], blocks = true
    )
    export_figure(
        joinpath(outdir, "scalar_snapshots.png"), store; field = :liquid_scalar,
        at = times[[1, 11, end]], blocks = true, colorrange = (0, 1.8),
        phase_rendering = :plic
    )

    figure = Figure(size = (1000, 340))
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
    content_drift = abs.(contents .- first(contents))
    axis = Axis(
        figure[1, 2]; title = "Conservation", xlabel = "t",
        ylabel = "absolute drift"
    )
    lines!(axis, times, errors; color = :dodgerblue, label = "liquid volume |V(t) − V(0)|")
    scatter!(axis, times, errors; color = :dodgerblue, markersize = 5)
    lines!(axis, times, content_drift; color = :orangered, label = "dye content |∫fs(t) − ∫fs(0)|")
    scatter!(axis, times, content_drift; color = :orangered, markersize = 5)
    Legend(figure[1, 3], axis; framevisible = false)
    save(joinpath(outdir, "recovery.png"), figure)
    export_animation(
        joinpath(outdir, "scalar_reversal.gif"), store; field = :liquid_scalar,
        fps = 10, nframes = length(keyframes(store)), size = (520, 450),
        blocks = true, colorrange = (0, 1.8), phase_rendering = :plic
    )
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
