using CairoMakie
using Printf

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

# A level-`level` run of the wave up to `final_time` with keyframes at `times`:
# its store and the stored keyframes.
function keyframe_run(level, final_time, times)
    flow = run_example(max_level = level, final_time = 0).flow
    fields = VisualizationFieldRegistry()
    register_field!(
        fields, :velocity_vorticity, flow.velocity, Vorticity();
        label = "Vorticity", colormap = :balance
    )
    register_default_fields!(fields, :volume_fraction, flow.interface)
    recorder = Recorder(fields; keyframes = AtTimes(times), preview = nothing)
    simulate!(flow, final_time; observer = recorder)
    store = snapshot_store(recorder)
    return store, [store[info] for info in keyframes(store)]
end

# The first state of a level-`level` run within the first period in which the
# plunging jet has reconnected with the free surface (a trapped gas region).
# Keyframes of a first-period run bracket it; a second run up to the bracket
# stores every state inside it as keyframes, which are never skipped or
# thinned. Also returns the store of the first-period run, whose keyframes
# are sparse while the wave steepens and dense and evenly spaced from
# `t = T/2` on, over plunging, impact, and splash-up.
function breaking_impact(level, period)
    period_store, coarse = keyframe_run(
        level, period, vcat(
            range(0, 0.5period; length = 11), range(0.5period, period; length = 49),
        )
    )
    bracket = findfirst(frame -> length(gas_regions(frame)) >= 2, coarse)
    (isnothing(bracket) || bracket == 1) &&
        error("no entrapped gas pocket found in the first period")
    times = collect(range(coarse[bracket - 1].time, coarse[bracket].time; length = 81))
    stored = filter(frame -> frame.time >= first(times) - 1.0e-12, last(keyframe_run(level, last(times), times)))
    impact = findfirst(frame -> length(gas_regions(frame)) >= 2, stored)
    (isnothing(impact) || impact == 1) &&
        error("the impact was not resolved inside the keyframe bracket")
    return stored, impact, period_store
end

# The symmetric range `(-q, q)` of a field over frames and a view, with `q`
# the `fraction` quantile of its magnitude: rarer peaks saturate.
function quantile_range(frames, field, view, fraction)
    magnitudes = Float32[]
    for frame in frames, value in rasterize(frame, field, view, (500, 225))
        isfinite(value) && push!(magnitudes, abs(value))
    end
    limit = sort!(magnitudes)[ceil(Int, fraction * length(magnitudes))]
    return (-limit, limit)
end

# Repository README animation of a first-period run: its keyframes from
# `t = T/2` on, evenly spaced and each stored within a step of its requested
# time, over plunging, impact, and splash-up. Light and dark variants share
# the frames and one vorticity range, the 99.9 % quantile of its magnitude,
# so that the impact peaks saturate instead of fading the vortex sheets. The
# dark variant stays muted for GitHub's dark theme: `:berlin` without its
# lightest ends (a dark neutral), grey text, interface, and faint blocks.
function readme_animations(outdir, store, period)
    window = (0.475period, period)
    camera = (-0.5, 0.5, -0.2, 0.25)
    frames = [store[info] for info in keyframes(store) if window[1] <= info.time <= window[2]]
    colorrange = quantile_range(frames, :velocity_vorticity, camera, 0.999)
    println(
        "README animation: ", length(frames), " keyframes, t/T = ",
        first(frames).time / period, " to ", last(frames).time / period,
        ", vorticity range ", colorrange
    )
    animation(path; kwargs...) = export_animation(
        path, store; field = :velocity_vorticity,
        window, camera, colorrange, blocks = true,
        title = frame -> @sprintf("t/T = %.3f", frame.time / period), framerate = 12,
        size = (900, 400), kwargs...
    )
    animation(joinpath(outdir, "readme_light.gif"))
    muted = [get(CairoMakie.Makie.ColorSchemes.berlin, x) for x in range(0.15, 0.85; length = 256)]
    with_theme(theme_dark(); textcolor = :gray80) do
        animation(
            joinpath(outdir, "readme_dark.gif"); colormap = muted,
            interfacecolor = :gray75, blockcolor = (:white, 0.15)
        )
    end
    return nothing
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
        recorder = Recorder(
            fields; diagnostics = flow, keyframes = EquispacedTimes(41),
            preview = nothing
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

    store = snapshot_store(recorder)
    # The hero comes from a separate, finer run of the first period.
    hero_level = size == :docs ? 5 : 6
    hero_seconds = @elapsed stored, impact, period_store = breaking_impact(hero_level, period)
    println("Hero run: level ", hero_level, ", first period, ", hero_seconds, " s")
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
    candidates_path = joinpath(tempdir(), "10_impact_candidates.png")
    save(candidates_path, figure)
    println("Impact candidates: ", candidates_path)
    export_figure(
        joinpath(outdir, "hero.png"), stored[impact]; field = :velocity_vorticity,
        camera = window, title = frame -> "Breaking impact, " * title(frame)
    )
    readme_animations(outdir, period_store, period)
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
