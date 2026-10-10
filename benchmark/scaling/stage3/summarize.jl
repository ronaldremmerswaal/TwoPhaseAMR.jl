# Summarize the five-repeat Stage 3 general-upwind matrix.
include(joinpath(@__DIR__, "..", "Support.jl"))
using .ScalingSupport
using Statistics

function main(args=ARGS)
    if args == ["--help"] || args == ["-h"]
        println("usage: julia summarize.jl MATRIX_DIRECTORY")
        return
    end
    length(args) == 1 || error("usage: julia summarize.jl MATRIX_DIRECTORY")
    directory = args[1]
    config = readjson(joinpath(directory, "configuration.json"))
    raw = readjsonl(joinpath(directory, "raw.jsonl"))
    groups = Dict{Tuple, Vector{Any}}()
    for row in raw
        key = (row["n"], row["block"], row["threads"], row["variant"])
        push!(get!(groups, key, Any[]), row)
        row["tolerance_violations"] == 0 && row["max_abs_vs_oracle"] == 0 ||
            error("failed full-field parity in $key")
        row["time_seconds"] >= config["minimum_timed_seconds"] ||
            error("short timed sample in $key")
    end
    expected = Set((n, b, p, v) for n in config["sizes"]
                   for b in (n, 8) for p in config["workers"]
                   for v in ("frozen", "current"))
    Set(keys(groups)) == expected ||
        error("missing/unexpected groups: $(setdiff(expected, Set(keys(groups)))) / $(setdiff(Set(keys(groups)), expected))")
    for (key, samples) in groups
        length(samples) == config["repeats"] &&
            Set(row["repeat"] for row in samples) == Set(0:(config["repeats"] - 1)) ||
            error("repeat count differs in $key")
        length(Set(row["steps"] for row in samples)) == 1 ||
            error("step count changed in $key")
        length(Set(row["advection_source_sha256"] for row in samples)) == 1 ||
            error("source hash changed in $key")
    end
    result = Any[]
    for n in config["sizes"], b in (n, 8)
        serial = Dict(v => [r["ns_per_unknown_step"] for r in groups[(n, b, 1, v)]]
                      for v in ("frozen", "current"))
        strongest = median(serial["frozen"]) <= median(serial["current"]) ?
                    "frozen" : "current"
        strongest_times = serial[strongest]
        for p in config["workers"], v in ("frozen", "current")
            samples = groups[(n, b, p, v)]
            times = [r["ns_per_unknown_step"] for r in samples]
            mid = median(times)
            own_efficiency = median(serial[v]) / (p * mid)
            strongest_efficiency = median(strongest_times) / (p * mid)
            interval = p == 1 && v == strongest ? [1.0, 1.0] :
                bootstrap_ratio(strongest_times, times, p;
                    seed=20260930 + n + b + p + (v == "current"), draws=5000)
            gate = p == 1 ? "not_applicable" :
                   interval[1] >= 0.5 ? "supported" :
                   interval[2] < 0.5 ? "missed" : "inconclusive"
            push!(result, Dict(
                "n" => n, "block" => b, "workers" => p, "variant" => v,
                "steps" => samples[1]["steps"], "samples" => length(samples),
                "median_ns_per_unknown_step" => mid,
                "mad_ns_per_unknown_step" => median([abs(x - mid) for x in times]),
                "min_timed_seconds" => minimum(r["time_seconds"] for r in samples),
                "max_field_abs_error" => maximum(r["max_abs_vs_oracle"] for r in samples),
                "own_serial_efficiency" => own_efficiency,
                "strongest_serial_variant" => strongest,
                "strongest_serial_efficiency" => strongest_efficiency,
                "strongest_serial_efficiency_ci95" => interval,
                "efficiency_gate_50pct" => gate,
            ))
        end
    end
    writejson(joinpath(directory, "summary.json"),
              Dict("schema" => 1, "config" => config,
                   "raw_samples" => length(raw), "rows" => result))
    writecsv(joinpath(directory, "summary.csv"), result)
    println("validated $(length(raw)) samples, $(length(result)) curves")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
