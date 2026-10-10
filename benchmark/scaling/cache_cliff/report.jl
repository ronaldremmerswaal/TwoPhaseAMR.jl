"""Aggregate and plot the controlled LW cache-cliff diagnostic phases."""
module CacheCliffReport

include(joinpath(@__DIR__, "..", "Support.jl"))
using .ScalingSupport: readjson, readjsonl, writejson, writecsv, sha256file
using CairoMakie
using Random
using Statistics

function boot_median(values, seed; repetitions=10000)
    rng = MersenneTwister(seed)
    replicas = sort!([median(rand(rng, values, length(values))) for _ in 1:repetitions])
    return [replicas[floor(Int,.025*(repetitions-1))+1],
            replicas[floor(Int,.975*(repetitions-1))+1]]
end

function geometry(n; tx=512, ty=16)
    widths = [min(tx, n-i) for i in 0:tx:n-1]
    heights = [min(ty, n-j) for j in 0:ty:n-1]
    cells = [width*height for height in heights for width in widths]
    return Dict("tile_count"=>length(cells), "x_tile_widths"=>widths,
                "y_tile_heights"=>heights, "min_tile_cells"=>minimum(cells),
                "max_tile_cells"=>maximum(cells), "five_array_bytes"=>5n*n*8)
end

function save_figure(figure, directory, stem)
    save(joinpath(directory, "$stem.svg"), figure)
    save(joinpath(directory, "$stem.png"), figure; px_per_unit=2)
end

function add_series!(axis, sequence, label, color; bounds=true)
    x = Float64[row["nx"] for row in sequence]
    y = Float64[row["median_ns_per_unknown_step"] for row in sequence]
    if bounds
        ci = [row["ci95_ns_per_unknown_step"] for row in sequence]
        errorbars!(axis, x, y, y .- first.(ci), last.(ci) .- y;
                   color, whiskerwidth=8)
    end
    lines!(axis, x, y; color, label)
    scatter!(axis, x, y; color)
end

function plot_ladder(rows, directory)
    figure = Figure(size=(900, 500))
    axis = Axis(figure[1,1], title="p16, spread across two CCDs, tile 512×16",
                xlabel="Grid width N (N×N)",
                ylabel="ns per unknown per complete timestep",
                xticks=(sort(unique(row["nx"] for row in rows)),
                        string.(sort(unique(row["nx"] for row in rows)))))
    for (language, color) in (("julia", "#2563eb"), ("fortran", "#d24b28"))
        sequence = sort([row for row in rows if row["language"] == language];
                        by=row->row["nx"])
        add_series!(axis, sequence, uppercasefirst(language), color)
    end
    for (n,label,color) in ((sqrt(64*1024^2/40), "5 arrays / 64 MiB", "#636363"),
                            (sqrt(64*1024^2/24), "3 arrays / 64 MiB", "#999999"))
        vlines!(axis, [n]; color, linestyle=:dash, label)
    end
    axislegend(axis)
    save_figure(figure, directory, "ladder-p16")
end

function plot_placement(rows, directory)
    figure = Figure(size=(1100, 480))
    for (column, language) in enumerate(("julia", "fortran"))
        axis = Axis(figure[1,column], title="$(uppercasefirst(language)) p8",
                    xlabel="Grid width N",
                    ylabel=column == 1 ? "ns per unknown per complete timestep" : "",
                    xticks=(sort(unique(row["nx"] for row in rows)),
                            string.(sort(unique(row["nx"] for row in rows)))))
        for (placement, color) in (("compact", "#d24b28"), ("spread", "#2563eb"))
            sequence = sort([row for row in rows if row["language"] == language &&
                        row["placement"] == placement]; by=row->row["nx"])
            add_series!(axis, sequence, placement, color)
        end
        axislegend(axis)
    end
    save_figure(figure, directory, "placement-p8")
end

function plot_steps(rows, directory)
    figure = Figure(size=(1000, 480))
    for (column, n) in enumerate((1024, 2048))
        axis = Axis(figure[1,column], title="$(n)×$(n), p16",
                    xlabel="Complete steps per fresh process",
                    ylabel=column == 1 ? "ns per unknown per complete timestep" : "",
                    xticks=([64,256], ["64","256"]))
        for (language, color) in (("julia", "#2563eb"), ("fortran", "#d24b28"))
            sequence = sort([row for row in rows if row["nx"] == n &&
                        row["language"] == language]; by=row->row["steps"])
            x = Float64[row["steps"] for row in sequence]
            y = Float64[row["median_ns_per_unknown_step"] for row in sequence]
            lines!(axis, x, y; color, label=uppercasefirst(language))
            scatter!(axis, x, y; color)
        end
        axislegend(axis)
    end
    save_figure(figure, directory, "steps-p16")
end

function summarize(directory)
    directory = abspath(directory)
    config = readjson(joinpath(directory, "configuration.json"))
    raw = readjsonl(joinpath(directory, "raw.jsonl"))
    phase = config["phase"]
    grouped = Dict{Tuple{Int,Int,String,Int,String},Vector{Any}}()
    for row in raw
        row["field_mismatches"] == 0 || error("parity failure in raw data")
        key = (row["nx"], row["workers"], row["placement"], row["steps"], row["language"])
        push!(get!(grouped, key, Any[]), row)
    end
    rows = Dict{String,Any}[]
    for (key, samples) in sort!(collect(grouped); by=first)
        n,p,placement,steps,language = key
        length(samples) == config["repeats"] &&
            Set(row["repeat"] for row in samples) == Set(0:config["repeats"]-1) ||
            error("incomplete group: $key")
        times = Float64[row["time_seconds"] for row in samples]
        values = Float64[row["ns_per_unknown_step"] for row in samples]
        row = Dict{String,Any}(
            "nx"=>n, "ny"=>n, "workers"=>p, "placement"=>placement,
            "steps"=>steps, "language"=>language, "samples"=>length(samples),
            "median_seconds"=>median(times),
            "median_ns_per_unknown_step"=>median(values),
            "ci95_ns_per_unknown_step"=>boot_median(values,
                n+steps+p+(language == "julia" ? 1 : 2)),
            "min_seconds"=>minimum(times), "max_seconds"=>maximum(times),
            "short_samples"=>count(sample->sample["short_diagnostic"], samples),
            "max_field_delta"=>maximum(sample["field_max_abs_error"] for sample in samples),
            "min_active_workers"=>minimum(minimum(sample["active_worker_counts"])
                                          for sample in samples))
        merge!(row, geometry(n))
        push!(rows, row)
    end
    summary = Dict{String,Any}(
        "schema"=>1, "phase"=>phase, "samples"=>length(raw), "rows"=>rows,
        "analysis_script_sha256"=>sha256file(@__FILE__),
        "configuration_sha256"=>config["configuration_sha256"])
    writejson(joinpath(directory, "summary.json"), summary)
    columns = String["nx", "workers", "placement", "steps", "language", "samples",
        "median_seconds", "median_ns_per_unknown_step", "ci95_low", "ci95_high",
        "min_seconds", "max_seconds", "short_samples", "max_field_delta",
        "min_active_workers", "tile_count", "min_tile_cells", "max_tile_cells",
        "five_array_bytes"]
    flat_rows = Dict{String,Any}[]
    for row in rows
        flat = copy(row)
        flat["ci95_low"], flat["ci95_high"] = row["ci95_ns_per_unknown_step"]
        push!(flat_rows, flat)
    end
    writecsv(joinpath(directory, "summary.csv"), flat_rows; fields=columns)
    phase == "ladder" && plot_ladder(rows, directory)
    phase == "placement" && plot_placement(rows, directory)
    phase == "steps" && plot_steps(rows, directory)
    return summary
end

function main(args=ARGS)
    length(args) == 1 || error("usage: julia report.jl RESULT_DIRECTORY")
    result = summarize(args[1])
    println("$(result["phase"]): $(length(result["rows"])) rows, $(result["samples"]) samples")
end

end # module

abspath(PROGRAM_FILE) == (@__FILE__) && CacheCliffReport.main()
