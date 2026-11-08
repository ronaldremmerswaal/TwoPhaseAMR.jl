using CairoMakie

Base.include(@__MODULE__, joinpath(@__DIR__, "stokes3_wave.jl"))

# Number of connected gas regions (liquid fraction below 1/2) on a uniform
# raster, periodic in x; regions smaller than `minimum` pixels are ignored.
# A second region is a gas pocket entrapped by the plunging jet.
function gas_regions(frame; resolution = (400, 400), minimum = 12)
    fraction = rasterize(frame, :volume_fraction, (-0.5, 0.5, -0.5, 0.5), resolution)
    gas = [isfinite(value) && value < 0.5 for value in fraction]
    label = zeros(Int, size(gas))
    regions = Tuple{Int, Tuple{Float64, Float64}}[]
    w, h = size(gas)
    for start in CartesianIndices(gas)
        (gas[start] && iszero(label[start])) || continue
        id = length(regions) + 1
        stack = [start]
        label[start] = id
        count, sx, sy = 0, 0.0, 0.0
        while !isempty(stack)
            cell = pop!(stack)
            count += 1
            sx += cell[1]
            sy += cell[2]
            for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                i, j = mod1(cell[1] + di, w), cell[2] + dj
                1 <= j <= h || continue
                if gas[i, j] && iszero(label[i, j])
                    label[i, j] = id
                    push!(stack, CartesianIndex(i, j))
                end
            end
        end
        push!(regions, (count, (-0.5 + (sx / count - 0.5) / w, -0.5 + (sy / count - 0.5) / h)))
    end
    return filter(region -> first(region) >= minimum, regions)
end

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    initial = stokes3_initial_condition()
    period = 2pi / initial.frequency
    # Two nominal wave periods.
    target_time = 2period
    level = size == :docs ? 4 : 6
    local flow, result, recorder
    elapsed = @elapsed begin
        flow = run_example(max_level = level, final_time = 0).flow
        fields = VisualizationFieldRegistry()
        register_field!(
            fields, :velocity_vorticity, flow.velocity, Vorticity();
            label = "Vorticity", colormap = :balance
        )
        register_default_fields!(fields, :volume_fraction, flow.interface)
        # Every step is kept as a preview to locate the breaking impact; the
        # animation uses the 41 keyframes only.
        recorder = Recorder(
            fields; diagnostics = flow, keyframes = EquispacedTimes(41),
            preview = EveryStep(), max_capture_overhead = 1.0,
            retention = Retention(preview = :all)
        )
        result = simulate!(flow, target_time; observer = recorder)
    end
    println("Setup and simulation time (s; includes first-call compilation): ", elapsed)
    print_simulation_summary(flow, result)

    xs = range(-0.5, 0.5; length = 201)[1:(end - 1)]
    numerical = [interface_height(flow.interface; position = x) for x in xs]
    reference = initial.elevation.(xs)
    errors = numerical .- reference
    println(
        "Final interface after two periods versus third-order initial profile: sample RMS = ",
        sqrt(sum(abs2, errors) / length(errors)), ", sample maximum = ", maximum(abs, errors)
    )
    figure = Figure(size = size == :docs ? (800, 440) : (1000, 550))
    axis = Axis(figure[1, 1]; xlabel = "x", ylabel = "Interface elevation")
    lines!(axis, xs, numerical; label = "Computed after two periods")
    lines!(axis, xs, reference; label = "Third-order initial profile", linestyle = :dash)
    axislegend(axis)
    save(joinpath(outdir, "profile.png"), figure)

    # Impact: the first stored state in which the plunging jet has reconnected
    # with the free surface, so that a gas pocket is entrapped.
    store = snapshot_store(recorder)
    stored = [store[info] for info in TwoPhaseAMR.frames(store)]
    impact = findfirst(frame -> length(gas_regions(frame)) >= 2, stored)
    isnothing(impact) && error("no entrapped gas pocket found; the wave did not break")
    pocket = last(sort(gas_regions(stored[impact]); by = first)[1:(end - 1)])
    centre = last(pocket)
    window = (centre[1] - 0.25, centre[1] + 0.25, centre[2] - 0.15, centre[2] + 0.15)
    title(frame) = "t/T = $(round(frame.time / period; digits = 3))"
    println(
        "Breaking impact: first entrapped gas pocket at t = ", stored[impact].time,
        " (t/T = ", stored[impact].time / period, "), pocket centre = ", centre,
        ", previous stored state t = ", stored[impact - 1].time
    )
    candidates = stored[[impact - 1, impact, min(impact + 2, end), min(impact + 6, end)]]
    figure = Figure(size = (1100, 800))
    for (k, frame) in enumerate(candidates)
        axis = fieldaxis(figure[fld1(k, 2), mod1(k, 2)]; camera = window, title = title(frame))
        amrfield!(axis, frame, :velocity_vorticity; colorrange = (-60, 60))
    end
    save(joinpath(outdir, "impact_candidates.png"), figure)
    export_figure(
        joinpath(outdir, "hero.png"), stored[impact]; field = :velocity_vorticity,
        camera = window, title = frame -> "Breaking impact, " * title(frame)
    )
    export_figure(
        joinpath(outdir, "vorticity.png"), store; field = :velocity_vorticity,
        title = frame -> "Vorticity at t = $(round(frame.time; sigdigits = 4))"
    )
    export_animation(
        joinpath(outdir, "wave.gif"), store; field = :velocity_vorticity,
        blocks = true, diagnostics = [:kinetic_energy], framerate = 8
    )
    return (; flow, result, recorder, errors)
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
