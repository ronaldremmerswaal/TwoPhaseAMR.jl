using CairoMakie
using Serialization

Base.include(@__MODULE__, joinpath(@__DIR__, "lgpi.jl"))

function make_figures(;
        outdir = joinpath(@__DIR__, "assets"), size = :docs,
        final_time = nothing, display_time = 0.97, recording = nothing,
    )
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    initial = lgpi_initial_condition()
    full_time = 1.2sqrt((1000 + 1) / (1000 - 1) * 2pi * initial.wavelength / 9.81)
    target_time = isnothing(final_time) ? full_time : final_time
    level = size == :docs ? 2 : 4
    flow = result = recorder = nothing
    local store
    if isnothing(recording)
        elapsed = @elapsed begin
            flow = run_example(max_level = level, final_time = 0).flow
            fields = VisualizationFieldRegistry()
            register_field!(
                fields, :velocity_vorticity, flow.velocity, flow.interface, Vorticity();
                label = "Vorticity", colormap = :balance
            )
            register_interface!(fields, flow.interface)
            register_geometry!(fields, flow.solver.geometry)
            recorder = Recorder(
                fields; diagnostics = flow, keyframes = EquispacedTimes(41),
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
        serialize(joinpath(tempdir(), "viz-lgpi-recording.jls"), store)
    else
        store = deserialize(recording)
    end
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
    CairoMakie.activate!(px_per_unit = 1)
    export_figure(
        joinpath(outdir, "hero.png"), store; field = :velocity_vorticity,
        at = times[[1, cld(length(times), 2), end]], blocks = true, size = (1100, 230),
        colorrange = (-limit, limit)
    )
    CairoMakie.activate!(px_per_unit = 2)
    export_animation(
        joinpath(outdir, "impact.gif"), store; field = :velocity_vorticity,
        window = (first(times), last(times)), blocks = true, diagnostics = [:kinetic_energy],
        framerate = 8, colorrange = (-limit, limit)
    )
    return (; flow, result, recorder, store)
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
