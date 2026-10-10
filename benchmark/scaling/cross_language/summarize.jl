"""Summarize matched Julia/Fortran scaling samples into CSV, JSON, and figures."""
module CrossLanguageSummary

include(joinpath(@__DIR__, "..", "Support.jl"))
using .ScalingSupport: readjson, readjsonl, writejson, writecsv, sha256file
using CairoMakie
using Random
using Statistics

function interval(values, seed; count=10000)
    rng = MersenneTwister(seed)
    replicas = sort!([median(rand(rng, values, length(values))) for _ in 1:count])
    return [replicas[floor(Int, .025*(count-1))+1],
            replicas[floor(Int, .975*(count-1))+1]]
end

function ratio_interval(first, second, seed; count=10000)
    length(first) == length(second) || error("paired sample count mismatch")
    rng = MersenneTwister(seed)
    replicas = Vector{Float64}(undef, count)
    for k in eachindex(replicas)
        indices = rand(rng, 1:length(first), length(first))
        replicas[k] = median(first[indices]) / median(second[indices])
    end
    sort!(replicas)
    return [replicas[floor(Int, .025*(count-1))+1],
            replicas[floor(Int, .975*(count-1))+1]]
end

function plots(rows, nx, directory)
    local_rows = sort([row for row in rows if row["nx"] == nx]; by=row->row["workers"])
    workers = Float64[row["workers"] for row in local_rows]
    colors = Dict("julia"=>"#2563eb", "fortran"=>"#d24b28")
    for kind in ("comparison", "speedup")
        figure = Figure(size=(840, 480))
        axis = Axis(figure[1,1],
            title="$(nx)×$(nx) periodic Lax–Wendroff, tile 512×16",
            xlabel="Threads (16 physical cores; 32 uses SMT)",
            ylabel=kind == "speedup" ? "Self speedup (same language, 1 thread / p threads)" :
                   "Wall-clock ns per unknown per complete timestep",
            xscale=log2, yscale=kind == "comparison" ? log10 : identity,
            xticks=(workers, string.(Int.(workers))))
        for language in ("julia", "fortran")
            metric = kind == "comparison" ? "ns_per_unknown_step" : "speedup"
            bound_metric = kind == "comparison" ? "ns_ci95" : "speedup_ci95"
            values = Float64[row["$(language)_$(metric)"] for row in local_rows]
            bounds = [row["$(language)_$(bound_metric)"] for row in local_rows]
            lower = Float64[max(0, value - bound[1]) for (value,bound) in zip(values,bounds)]
            upper = Float64[max(0, bound[2] - value) for (value,bound) in zip(values,bounds)]
            errorbars!(axis, workers, values, lower, upper;
                       color=colors[language], whiskerwidth=8)
            lines!(axis, workers, values; color=colors[language], linewidth=2,
                   label=uppercasefirst(language))
            scatter!(axis, workers, values; color=colors[language], markersize=9)
        end
        if kind == "speedup"
            lines!(axis, workers, workers; color="#666666", linestyle=:dash, label="Ideal")
            ylims!(axis, low=0)
        end
        xlims!(axis, .9, 36)
        axislegend(axis)
        save(joinpath(directory, "$kind-$nx.svg"), figure)
        save(joinpath(directory, "$kind-$nx.png"), figure; px_per_unit=2)
    end
end

function summarize_study(directory)
    directory = abspath(directory)
    config = readjson(joinpath(directory, "configuration.json"))
    workload = Dict(parse(Int,n)=>Int(steps)
                    for (n,steps) in readjson(joinpath(directory, "workload.json")))
    samples = readjsonl(joinpath(directory, "raw.jsonl"))
    grouped = Dict{Tuple{Int,Int,String},Dict{Int,Any}}()
    for sample in samples
        key = (sample["nx"], sample["workers"], sample["language"])
        group = get!(grouped, key, Dict{Int,Any}())
        repeat = sample["repeat"]
        !haskey(group, repeat) || error("duplicate sample $key repeat $repeat")
        sample["field_mismatches"] == 0 || error("summary contains failed field parity")
        group[repeat] = sample
    end
    sizes, counts, repeats = config["sizes"], config["threads"], config["repeats"]
    expected = Set(0:repeats-1)
    rows = Dict{String,Any}[]
    for nx in sizes, p in counts
        record = Dict{String,Any}(
            "nx"=>nx, "ny"=>nx, "steps"=>workload[nx], "workers"=>p,
            "samples_per_language"=>repeats,
            "tile_x"=>config["tile"][1], "tile_y"=>config["tile"][2])
        timings = Dict{String,Vector{Float64}}()
        for language in ("julia", "fortran")
            key = (nx,p,language)
            group = get(grouped, key, Dict{Int,Any}())
            Set(keys(group)) == expected || error("incomplete group $key")
            samples_for_group = [group[r] for r in 0:repeats-1]
            times = Float64[item["time_seconds"] for item in samples_for_group]
            timings[language] = times
            med = median(times)
            divisor = nx*nx*workload[nx]
            record["$(language)_seconds"] = med
            record["$(language)_seconds_ci95"] = interval(times, 17+nx+p)
            record["$(language)_ns_per_unknown_step"] = 1e9*med/divisor
            record["$(language)_ns_ci95"] =
                [1e9*x/divisor for x in record["$(language)_seconds_ci95"]]
            base = [grouped[(nx,1,language)][r]["time_seconds"] for r in 0:repeats-1]
            record["$(language)_speedup"] = median(base)/med
            record["$(language)_speedup_ci95"] = p == 1 ? [1.0,1.0] :
                ratio_interval(base, times, 43+nx+p)
            record["$(language)_efficiency"] = record["$(language)_speedup"]/p
            record["$(language)_efficiency_ci95"] =
                record["$(language)_speedup_ci95"] ./ p
            record["$(language)_field_max_abs_error"] =
                maximum(item["field_max_abs_error"] for item in samples_for_group)
            record["$(language)_active_workers_min"] =
                minimum(minimum(item["active_worker_counts"]) for item in samples_for_group)
            record["$(language)_sample_seconds_min"] = minimum(times)
            record["$(language)_sample_seconds_max"] = maximum(times)
        end
        record["julia_over_fortran_time"] =
            record["julia_seconds"]/record["fortran_seconds"]
        record["julia_over_fortran_ci95"] =
            ratio_interval(timings["julia"], timings["fortran"], 71+nx+p)
        push!(rows, record)
    end
    summary = Dict{String,Any}(
        "schema"=>1,
        "metric"=>"wall-clock ns per scalar unknown per complete three-sweep timestep",
        "config_sha256"=>config["configuration_sha256"], "samples"=>length(samples),
        "analysis_script_sha256"=>sha256file(@__FILE__), "rows"=>rows)
    writejson(joinpath(directory, "summary.json"), summary)
    columns = String["nx", "workers", "steps", "julia_seconds", "fortran_seconds",
        "julia_ns_per_unknown_step", "fortran_ns_per_unknown_step",
        "julia_ns_ci95_low", "julia_ns_ci95_high", "fortran_ns_ci95_low",
        "fortran_ns_ci95_high", "julia_speedup", "fortran_speedup",
        "julia_efficiency", "fortran_efficiency", "julia_speedup_ci95_low",
        "julia_speedup_ci95_high", "fortran_speedup_ci95_low", "fortran_speedup_ci95_high",
        "julia_over_fortran_time", "julia_over_fortran_ci95_low",
        "julia_over_fortran_ci95_high", "julia_active_workers_min",
        "fortran_active_workers_min"]
    flat_rows = Dict{String,Any}[]
    for row in rows
        flat = copy(row)
        for language in ("julia", "fortran"), metric in ("ns", "speedup")
            low, high = row["$(language)_$(metric)_ci95"]
            flat["$(language)_$(metric)_ci95_low"] = low
            flat["$(language)_$(metric)_ci95_high"] = high
        end
        flat["julia_over_fortran_ci95_low"], flat["julia_over_fortran_ci95_high"] =
            row["julia_over_fortran_ci95"]
        push!(flat_rows, flat)
    end
    writecsv(joinpath(directory, "comparison.csv"), flat_rows; fields=columns)
    for nx in sizes
        plots(rows, nx, directory)
    end
    return summary
end

function main(args=ARGS)
    length(args) == 1 || error("usage: julia summarize.jl RESULT_DIRECTORY")
    summary = summarize_study(args[1])
    println("wrote $(length(summary["rows"])) comparison rows")
end

end # module

abspath(PROGRAM_FILE) == (@__FILE__) && CrossLanguageSummary.main()
