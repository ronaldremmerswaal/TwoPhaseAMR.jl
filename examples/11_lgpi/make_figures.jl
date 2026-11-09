using CairoMakie
using Serialization

Base.include(@__MODULE__, joinpath(@__DIR__, "lgpi.jl"))

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    initial = lgpi_initial_condition()
    target_time = 1.2sqrt((1000 + 1) / (1000 - 1) * 2pi * initial.wavelength / 9.81)
    level = size == :docs ? 2 : 4
    local flow, result, recorder
    elapsed = @elapsed begin
        flow = run_example(max_level = level, final_time = 0).flow
        fields = VisualizationFieldRegistry()
        register_field!(
            fields, :velocity_vorticity, flow.velocity, Vorticity();
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
        block.key.level for block in flow.velocity.mesh.blocks
            if initial.bathymetry_levelset(block.origin...) <= 0
    ]
    println("Maximum level of blocks touching solid at final time: ", maximum(solid_levels))

    store = snapshot_store(recorder)
    times = [info.time for info in keyframes(store)]
    serialize(joinpath(tempdir(), "viz-lgpi-recording.jls"), store)
    view = (0.0, initial.domain[1], 0.0, initial.domain[2])
    resolution = (400, 240)
    magnitudes = Float32[]
    peak = (value = 0.0, time = 0.0, x = 0.0, y = 0.0)
    for info in keyframes(store)
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
        blocks = true, diagnostics = [:kinetic_energy], framerate = 8, colorrange = (-limit, limit)
    )
    return (; flow, result, recorder)
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end
