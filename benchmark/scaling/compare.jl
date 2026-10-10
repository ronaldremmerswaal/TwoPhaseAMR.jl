# Compare independent scaling runs at a frozen numerical workload.
include(joinpath(@__DIR__, "Support.jl"))
using .ScalingSupport

function workload_signature(row)
    diagnostics, fixture, runtime = row["diagnostics"], row["fixture"], row["runtime"]
    return (row["steps"], get(diagnostics, "dt", nothing),
            get(diagnostics, "final_time", nothing), diagnostics["numerical_contract"],
            fixture["cells"], get(fixture, "block_nx", nothing),
            get(fixture, "block_ny", nothing), get(fixture, "diagnostics_mode", nothing),
            get(fixture, "diagnostic_interval", nothing), runtime["precision"],
            runtime["blas_threads"], runtime["julia_version"], row["placement"],
            get(row, "pin_workers", false))
end

function load_run(directory)
    config = readjson(joinpath(directory, "configuration.json"))
    raw = readjsonl(joinpath(directory, "raw.jsonl"))
    summary = readjson(joinpath(directory, "summary.json"))
    signatures = Dict{Tuple, Set{Any}}()
    for row in raw
        key = (row["case"], row["nx"], row["ny"])
        push!(get!(signatures, key, Set{Any}()), workload_signature(row))
    end
    any(length(values) != 1 for values in values(signatures)) &&
        error("inconsistent workload within $directory")
    return (config, raw, summary,
            Dict(key => first(values) for (key, values) in signatures))
end

function main(args=ARGS)
    if any(arg -> arg in ("--help", "-h"), args)
        println("usage: julia compare.jl RUN_DIRECTORY [RUN_DIRECTORY ...] [--output PATH]")
        return
    end
    runs = String[]
    output = nothing
    index = 1
    while index <= length(args)
        if args[index] == "--output"
            index < length(args) || error("--output needs a path")
            output = args[index + 1]
            index += 2
        else
            startswith(args[index], "--") && error("unknown argument: $(args[index])")
            push!(runs, args[index])
            index += 1
        end
    end
    isempty(runs) && error("provide at least one results directory")
    loaded = load_run.(runs)
    length(Set(item[1]["runner_sha256"] for item in loaded)) == 1 ||
        error("benchmark runner changed across runs; use one harness revision")
    reference = loaded[1][4]
    for (_, _, _, signatures) in loaded[2:end]
        for key in intersect(keys(reference), keys(signatures))
            reference[key] == signatures[key] ||
                error("different physical workload for $key")
        end
    end
    one_times = Dict{Tuple, Vector{Tuple}}()
    original_one = Dict{Tuple, Any}()
    for (run_index, (_, _, rows, _)) in enumerate(loaded), row in rows
        row["workers"] == 1 || continue
        key = (row["case"], row["nx"], row["ny"], row["placement"])
        push!(get!(one_times, key, Tuple[]),
              (row["time_median_seconds"], run_index, row["backend"]))
        run_index == 1 && (original_one[key] = row["time_median_seconds"])
    end
    comparisons = Any[]
    for (run_index, (_, _, rows, _)) in enumerate(loaded), row in rows
        key = (row["case"], row["nx"], row["ny"], row["placement"])
        fastest, fastest_run, fastest_backend = minimum(one_times[key])
        baseline = get(original_one, key, nothing)
        push!(comparisons, Dict(
            "run" => runs[run_index], "case" => row["case"],
            "nx" => row["nx"], "ny" => row["ny"], "workers" => row["workers"],
            "backend" => row["backend"], "placement" => row["placement"],
            "steps" => row["steps"], "time_median_seconds" => row["time_median_seconds"],
            "self_speedup" => row["speedup"], "self_efficiency" => row["efficiency"],
            "self_speedup_ci95" => row["speedup_ci95_bootstrap"],
            "absolute_speedup_vs_fastest_serial" => fastest / row["time_median_seconds"],
            "fastest_serial_run" => runs[fastest_run],
            "fastest_serial_backend" => fastest_backend,
            "speedup_vs_starting_revision_serial" =>
                isnothing(baseline) ? nothing : baseline / row["time_median_seconds"],
            "mass_error_max" => row["mass_error_max"],
            "l2_error_max" => row["l2_error_max"],
        ))
    end
    destination = isnothing(output) ? joinpath(runs[1], "comparison.json") : output
    writejson(destination, Dict("schema_version" => 1, "runs" => runs,
                                "comparisons" => comparisons))
    println("Compared $(length(runs)) runs with matched physical workloads: $destination")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
