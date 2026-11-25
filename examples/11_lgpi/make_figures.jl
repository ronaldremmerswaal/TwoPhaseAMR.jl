using CairoMakie, Printf

Base.include(@__MODULE__, joinpath(@__DIR__, "lgpi.jl"))

# README hero: the impact and wall jet, slowed down. Request 101 times in the
# impact window in addition to the 81 full-run requests. Output times select
# the nearest stored states, which can differ slightly from the requests.
const HERO_WINDOW = (0.42, 0.62)
const HERO_CAMERA = (0.0, 0.5, 0.08, 0.6)
const HERO_FRAMES = 101
const HERO_FPS = 10
const HERO_SIZE = (700, 620)
const IMPACT_FPS = 8
const COLOR_LEVELS = 31

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
    hero_display = filter(
        info -> hero_window[1] - 1.0e-9 <= info.time <= min(hero_window[2], display_end),
        selected,
    )
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
    if length(hero_display) < 2
        @warn "No hero frames in the requested window; skipping the hero animation"
    else
        export_animation(
            joinpath(outdir, "impact_hero.gif"), store; field = :velocity_vorticity,
            window = (first(hero_display).time, last(hero_display).time),
            camera = hero_camera, blocks = true, fps = hero_fps,
            nframes = min(hero_frames, length(hero_display)), size = hero_size,
            colorrange = (-limit, limit), colormap,
            title = frame -> @sprintf("t = %.3f", frame.time)
        )
        println(
            "Hero animation: ", min(hero_frames, length(hero_display)), " frames from t = ",
            first(hero_display).time, " to ", last(hero_display).time, " at ",
            hero_fps, " fps, camera ", hero_camera
        )
    end
    return (; flow, result, recorder, store)
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
