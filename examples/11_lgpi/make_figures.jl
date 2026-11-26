using CairoMakie, Printf

Base.include(@__MODULE__, joinpath(@__DIR__, "lgpi.jl"))

# The hero resolves the impact at level 4 and includes the full domain height,
# from the submerged bathymetry to the top wall.
const HERO_WINDOW = (0.42, 0.62)
const HERO_CAMERA = (0.0, 0.5, 0.0, 0.6)
const HERO_FRAMES = 101
const HERO_FPS = 10
const HERO_SIZE = (620, 740)
const IMPACT_FPS = 8
const COLOR_LEVELS = 31

function render_hero(
        path, store; window = HERO_WINDOW, camera = HERO_CAMERA,
        nframes = HERO_FRAMES, fps = HERO_FPS, size = HERO_SIZE
    )
    infos = [
        info for info in keyframes(store)
            if any(t -> window[1] <= t <= window[2], info.requested_times)
    ]
    length(infos) >= 2 || error("The hero window needs at least two stored states")
    frames = [store[info] for info in infos]
    magnitudes = Float32[]
    for frame in frames, value in rasterize(frame, :velocity_vorticity, camera, (250, 300))
        isfinite(value) && push!(magnitudes, abs(value))
    end
    limit = sort!(magnitudes)[ceil(Int, 0.999length(magnitudes))]
    limit > 0 || (limit = 1.0f0)
    # Colour the liquid pale blue, and make near-zero vorticity transparent so
    # both phases remain visible beneath the balance-coloured vorticity.
    colors = [
        begin
            c = get(CairoMakie.Makie.ColorSchemes.balance, x)
            RGBAf(c.r, c.g, c.b, 0.9 * min(1, 5abs(2x - 1)))
        end for x in range(0, 1; length = COLOR_LEVELS)
    ]
    figure = Figure(; size)
    current = Observable(first(frames))
    axis = fieldaxis(
        figure[1, 1]; camera,
        title = lift(frame -> @sprintf("t = %.3f", frame.time), current),
    )
    amrfield!(
        axis, current, :volume_fraction;
        colormap = [RGBf(1, 1, 1), RGBf(0.72, 0.86, 0.96)],
        colorrange = (0, 1), interface = false, geometry = false,
    )
    vort = amrfield!(
        axis, current, :velocity_vorticity; colormap = cgrad(colors, COLOR_LEVELS; categorical = true),
        colorrange = (-limit, limit), blocks = true,
    )
    Colorbar(figure[1, 2], vort; label = "Vorticity")
    times = range(first(infos).time, last(infos).time; length = nframes)
    record(figure, path, times; framerate = fps, px_per_unit = 1) do time
        current[] = frame_at(store, time)
    end
    println(
        "Hero: level 4, ", nframes, " frames, t = ", first(times), "–", last(times),
        ", vorticity limit = ", limit, ", camera = ", camera
    )
    return path
end

function make_hero(;
        outdir = joinpath(@__DIR__, "assets"), window = HERO_WINDOW,
        camera = HERO_CAMERA, nframes = HERO_FRAMES, fps = HERO_FPS, size = HERO_SIZE
    )
    mkpath(outdir)
    local flow, recorder, result
    elapsed = @elapsed begin
        flow = run_example(max_level = 4, final_time = 0).flow
        fields = VisualizationFieldRegistry()
        register_field!(
            fields, :velocity_vorticity, flow.velocity, flow.interface, Vorticity();
            label = "Vorticity", colormap = :balance,
        )
        register_field!(
            fields, :volume_fraction, flow.interface.fraction;
            label = "Liquid fraction", colorrange = (0, 1),
        )
        register_interface!(fields, flow.interface)
        recorder = Recorder(
            fields; diagnostics = flow,
            keyframes = AtTimes(range(window[1], window[2]; length = nframes)),
            preview = nothing,
        )
        result = simulate!(flow, window[2]; observer = recorder)
    end
    println("Level-4 hero setup and simulation time (s): ", elapsed)
    print_simulation_summary(flow, result)
    render_hero(
        joinpath(outdir, "impact_hero.gif"), snapshot_store(recorder);
        window, camera, nframes, fps, size
    )
    return (; flow, result, recorder)
end

function make_figures(;
        outdir = joinpath(@__DIR__, "assets"), size = :docs,
        final_time = nothing, display_time = nothing,
        hero_window = HERO_WINDOW, hero_camera = HERO_CAMERA,
        hero_frames = HERO_FRAMES, hero_fps = HERO_FPS, hero_size = HERO_SIZE,
    )
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    initial = lgpi_initial_condition()
    full_time = 1.2sqrt((1000 + 1) / (1000 - 1) * 2pi * initial.wavelength / 9.81)
    target_time = isnothing(final_time) ? full_time : final_time
    level = size == :docs ? 3 : 4
    hero_requests = collect(range(hero_window[1], hero_window[2]; length = hero_frames))
    requested = unique!(
        sort!(
            vcat(
                collect(range(0.0, target_time; length = 81)),
                filter(time -> 0 <= time <= target_time, hero_requests),
            ),
        ),
    )
    local flow, result, recorder
    elapsed = @elapsed begin
        flow = run_example(max_level = level, final_time = 0).flow
        fields = VisualizationFieldRegistry()
        register_field!(
            fields, :velocity_vorticity, flow.velocity, flow.interface, Vorticity();
            label = "Vorticity", colormap = :balance
        )
        register_interface!(fields, flow.interface)
        register_geometry!(fields, flow.system.geometry)
        recorder = Recorder(
            fields; diagnostics = flow, keyframes = AtTimes(requested),
            preview = nothing
        )
        result = simulate!(flow, target_time; observer = recorder)
    end
    println("Setup and simulation time (s; includes first-call compilation): ", elapsed)
    print_simulation_summary(flow, result)
    println("Unresolved curvature faces (cumulative): ", result.total_unresolved_curvature_faces)
    solid_levels = [
        block.key.level for block in flow.velocity[1].mesh.blocks
            if initial.bathymetry_levelset(block.origin...) <= 0
    ]
    println("Maximum level of blocks touching solid at final time: ", maximum(solid_levels))
    store = snapshot_store(recorder)
    display_end = isnothing(display_time) ? target_time : min(target_time, display_time)
    selected = filter(info -> info.time <= display_end, keyframes(store))
    isempty(selected) && error("No recorded frames at or before the requested display time")
    times = [info.time for info in selected]
    view = (0.0, initial.domain[1], 0.0, initial.domain[2])
    resolution = (400, 240)
    magnitudes = Float32[]
    peak = (value = 0.0, time = 0.0, x = 0.0, y = 0.0)
    for info in selected
        values = rasterize(store[info], :velocity_vorticity, view, resolution)
        for index in CartesianIndices(values)
            value = abs(values[index])
            isfinite(value) || continue
            push!(magnitudes, value)
            if value > peak.value
                peak = (
                    value = value, time = info.time,
                    x = (index[1] - 0.5) * initial.domain[1] / resolution[1],
                    y = (index[2] - 0.5) * initial.domain[2] / resolution[2],
                )
            end
        end
    end
    quantile = sort!(magnitudes)[ceil(Int, 0.999length(magnitudes))]
    limit = quantile > 0 && peak.value > 3quantile ? quantile : peak.value
    println("Vorticity peak ", peak, "; 99.9% magnitude = ", quantile, "; display limit = ", limit)
    colormap = cgrad(:balance, COLOR_LEVELS; categorical = true)
    CairoMakie.activate!(px_per_unit = 1)
    export_figure(
        joinpath(outdir, "hero.png"), store; field = :velocity_vorticity,
        at = [first(times), (first(times) + last(times)) / 2, last(times)],
        blocks = true, size = (1100, 230), colorrange = (-limit, limit)
    )
    CairoMakie.activate!(px_per_unit = 2)
    # Sample the full run uniformly, including the dense impact candidates.
    export_animation(
        joinpath(outdir, "impact.gif"), store; field = :velocity_vorticity,
        window = (first(times), last(times)), blocks = true, diagnostics = [:kinetic_energy],
        fps = IMPACT_FPS, nframes = max(2, round(Int, 80 * last(times) / target_time)),
        size = (900, 355), colorrange = (-limit, limit), colormap
    )
    make_hero(;
        outdir, window = hero_window, camera = hero_camera,
        nframes = hero_frames, fps = hero_fps, size = hero_size
    )
    return (; flow, result, recorder, store)
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
