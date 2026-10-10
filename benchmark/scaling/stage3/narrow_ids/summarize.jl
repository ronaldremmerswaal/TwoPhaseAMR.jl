#!/usr/bin/env julia
module BlockInteriorSummary

include(joinpath(@__DIR__, "..", "..", "Support.jl"))
using .ScalingSupport

function main(args=ARGS)
    if any(arg -> arg in ("-h","--help"),args)
        println("usage: summarize.jl COHORT_DIRECTORY")
        return
    end
    length(args) == 1 || error("usage: summarize.jl COHORT_DIRECTORY")
    directory = abspath(args[1])
    config = readjson(joinpath(directory,"configuration.json"))
    rows = readjsonl(joinpath(directory,"raw.jsonl"))
    groups = Dict{Tuple{Int,Int,Int,String},Vector{Any}}()
    for row in rows
        key = (row["n"],row["block"],row["threads"],row["variant"])
        push!(get!(groups,key,Any[]),row)
        row["tolerance_violations"] == 0 && row["bitwise_equal"] ||
            error("full-field mismatch in $key")
        row["time_seconds"] >= config["minimum_timed_seconds"] ||
            error("short timed interval in $key")
    end
    expected = Set((n,b,p,variant) for (n,b) in config["cases"] for p in (1,4,8)
                   for variant in ("baseline","current"))
    Set(keys(groups)) == expected || error("missing or unexpected sample group")
    for (key,samples) in groups
        length(samples) == config["repeats"] || error("repeat count differs in $key")
        Set(row["repeat"] for row in samples) == Set(0:(config["repeats"]-1)) ||
            error("repeat IDs differ in $key")
        for field in ("source_sha256","steps","initial_sha256","oracle_sha256")
            length(Set(row[field] for row in samples)) == 1 ||
                error("$field differs in $key")
        end
    end
    result = Dict{String,Any}[]
    accepted = config["phase"] == "matrix"
    for (n,b) in config["cases"]
        serial = Dict(variant => [row["ns_per_unknown_step"] for row in groups[n,b,1,variant]]
                      for variant in ("baseline","current"))
        strongest = median_value(serial["baseline"]) <= median_value(serial["current"]) ?
                    "baseline" : "current"
        for p in (1,4,8)
            baseline = [row["ns_per_unknown_step"] for row in groups[n,b,p,"baseline"]]
            current = [row["ns_per_unknown_step"] for row in groups[n,b,p,"current"]]
            base_med, cur_med = median_value(baseline), median_value(current)
            efficiency = median_value(serial[strongest])/(p*cur_med)
            efficiency_ci = accepted ? bootstrap_ratio(serial[strongest],current,p;
                                                       seed=20261010+n+b+p) : nothing
            speedup_ci = accepted ? bootstrap_ratio(baseline,current;
                                                    seed=20261011+n+b+p) : nothing
            gate = !accepted ? "pilot" : p == 1 ? "not_applicable" :
                   efficiency_ci[1] >= 0.5 ? "supported" :
                   efficiency_ci[2] < 0.5 ? "missed" : "inconclusive"
            push!(result,Dict(
                "n" => n, "block" => b, "workers" => p,
                "steps" => groups[n,b,p,"current"][1]["steps"],
                "repeats" => config["repeats"],
                "baseline_median_ns_per_unknown_step" => base_med,
                "current_median_ns_per_unknown_step" => cur_med,
                "baseline_mad_ns_per_unknown_step" => mad_value(baseline),
                "current_mad_ns_per_unknown_step" => mad_value(current),
                "same_worker_baseline_over_current" => base_med/cur_med,
                "same_worker_speedup_ci95" => speedup_ci,
                "strongest_serial_variant" => strongest,
                "current_strongest_serial_efficiency" => efficiency,
                "current_efficiency_ci95" => efficiency_ci,
                "efficiency_gate_50pct" => gate,
                "minimum_timed_seconds" => minimum(row["time_seconds"] for variant in
                    ("baseline","current") for row in groups[n,b,p,variant])))
        end
    end
    writejson(joinpath(directory,"summary.json"),Dict(
        "schema" => 1, "config" => config, "samples" => length(rows), "rows" => result))
    writecsv(joinpath(directory,"summary.csv"),result)
    println("validated $(length(rows)) full-field samples; $(length(result)) paired groups")
    for row in result
        println("n$(row["n"]) b$(row["block"]) p$(row["workers"]): ",
                round(row["baseline_median_ns_per_unknown_step"];digits=3)," -> ",
                round(row["current_median_ns_per_unknown_step"];digits=3),
                " ns/cell-step; E=",
                round(row["current_strongest_serial_efficiency"];digits=3),
                " ($(row["efficiency_gate_50pct"]))")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
