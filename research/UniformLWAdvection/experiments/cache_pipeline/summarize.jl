"""Summarize accepted independent-process toy/pipeline cohorts and plot them."""
module CachePipelineSummary

include(joinpath(@__DIR__, "..", "..", "..", "..", "benchmark", "scaling", "Support.jl"))
using .ScalingSupport: readjson, readjsonl, writejson, writecsv, sha256file, percentile
using CairoMakie
using Random
using Statistics

function options(args)
    values = Dict{String,Vector{String}}()
    index = 1
    while index <= length(args)
        startswith(args[index], "--") || error("expected --option")
        key = args[index][3:end]
        key in ("cohorts", "output", "bootstrap", "seed") || error("unknown option $key")
        index += 1
        values[key] = String[]
        while index <= length(args) && !startswith(args[index], "--")
            push!(values[key], args[index])
            index += 1
        end
    end
    cohorts = abspath.(get(values, "cohorts", String[]))
    !isempty(cohorts) || error("at least one --cohorts directory required")
    length(get(values, "output", String[])) == 1 || error("one --output directory required")
    output = abspath(only(values["output"]))
    bootstrap = parse(Int, only(get(values, "bootstrap", ["20000"])))
    seed = parse(Int, only(get(values, "seed", ["30092027"])))
    !ispath(output) || error("output already exists")
    bootstrap >= 1000 || error("at least 1000 bootstrap replications required")
    return (; cohorts, output, bootstrap, seed)
end

function interval_ratio(numerator, denominator, rng, replications)
    values = Vector{Float64}(undef, replications)
    for k in eachindex(values)
        values[k] = median(rand(rng, numerator, length(numerator))) /
                    median(rand(rng, denominator, length(denominator)))
    end
    sort!(values)
    return [percentile(values, 0.025), percentile(values, 0.975)]
end

function check_cohort!(groups, all_records, cohort, signature)
    configuration = readjson(joinpath(cohort, "configuration.json"))
    records = readjsonl(joinpath(cohort, "raw.jsonl"))
    commands = readjsonl(joinpath(cohort, "commands.jsonl"))
    failures = filter(command -> get(command, "returncode", 0) != 0, commands)
    length(commands) == length(records) + 2length(configuration["sizes"]) + length(failures) ||
        error("launch receipt count differs in $cohort")
    if !isempty(failures)
        recovery_path = joinpath(cohort, "recovery-receipt.json")
        length(failures) == 1 && isfile(recovery_path) ||
            error("unaccounted failed launch in $cohort")
        recovery = readjson(recovery_path)
        original = recovery["original_failure_file"]
        recovery["status"] == "complete" && recovery["original_failure_pre_timer"] &&
            sha256file(original) == recovery["original_failure_sha256"] &&
            readjson(original) == failures[1] && failures[1]["returncode"] == -11 &&
            (get(failures[1], "termsignal", 0) == 11 ||
             occursin("signal (11", get(failures[1], "stderr", ""))) ||
            error("failed launch is not an accepted pre-timer retry in $cohort")
        [row["sample_index"] for row in records] == collect(0:length(records)-1) ||
            error("recovered sample indices are incomplete")
    end
    cohort_signature = Dict(basename(path)=>digest for (path,digest) in configuration["sources"])
    signature === nothing || cohort_signature == signature ||
        error("source manifest differs between cohorts")
    for row in records
        n, p, mode = row["nx"], row["workers"], row["mode"]
        mode in ("toy", "pipeline") || error("unexpected measurement mode")
        row["ny"] == n && row["steps"] == configuration["steps"][string(n)] ||
            error("trajectory mismatch")
        row["field_mismatches"] == 0 && row["field_sha256"] == row["oracle_sha256"] ||
            error("full-field check failed")
        row["q_alias_preserved"] && all(==(p), row["active_worker_counts"]) ||
            error("alias or participation check failed")
        [a["observed_cpu"] for a in row["affinity"]] == row["requested_cpus"] ||
            error("affinity check failed")
        row["time_seconds"] >= configuration["minimum_sample_seconds"] ||
            error("short sample in accepted cohort")
        row["source_sha256"]["pipeline"] == cohort_signature["CachePipeline.jl"] ||
            error("pipeline source changed")
        push!(get!(groups, (n,p,mode), Float64[]), row["ns_per_unknown_step"])
        push!(all_records, row)
    end
    return cohort_signature, configuration, length(failures)
end

function chart(rows, sizes, counts, output)
    figure = Figure(size=(500length(sizes), 400))
    colors = Dict("toy"=>"#2d5f8b", "pipeline"=>"#c06426")
    labels = Dict("toy"=>"Frozen toy", "pipeline"=>"Tile pipeline")
    for (column, n) in enumerate(sizes)
        axis = Axis(figure[1, column], title="$(n) × $(n)", xlabel="Julia workers",
                    ylabel=column == 1 ? "ns per cell per complete step" : "",
                    xscale=log2, xticks=(counts, string.(counts)))
        for mode in ("toy", "pipeline")
            subset = sort([row for row in rows if row["n"] == n]; by=row->row["workers"])
            x = Float64[row["workers"] for row in subset]
            y = Float64[row["$(mode)_median_ns"] for row in subset]
            lower = Float64[row["$(mode)_iqr_ns"][1] for row in subset]
            upper = Float64[row["$(mode)_iqr_ns"][2] for row in subset]
            band!(axis, x, lower, upper; color=(colors[mode], 0.16))
            lines!(axis, x, y; color=colors[mode], label=labels[mode])
            scatter!(axis, x, y; color=colors[mode])
        end
        axislegend(axis; position=:rt)
    end
    save(joinpath(output, "absolute-time.svg"), figure)
    save(joinpath(output, "absolute-time.png"), figure; px_per_unit=2)
end

fmt(value) = string(round(value; digits=3))
ci(values) = "$(fmt(values[1]))–$(fmt(values[2]))"

function report_text(summary)
    rows = summary["rows"]
    count = summary["accepted_samples"]
    lines = String[
        "# Full-step tile pipeline scaling study", "",
        "$count accepted fresh-process samples; minimum timed duration " *
        "$(fmt(summary["minimum_sample_seconds"])) s. Every final field matched " *
        "its serial toy oracle bitwise, and each requested worker participated in both measured phases.", "",
        "Pre-timer failed launches retained: $(summary["pre_timing_failed_launches"]). " *
        "Any recovered failure is documented in its cohort's command log and recovery receipt; " *
        "it contributes no timing sample.", "",
        "The pipeline uses tile `(128,32)` with halo work and public-`q` copyback charged " *
        "on every step. The frozen toy uses tile `(512,16)`. Each method uses the same " *
        "input, timestep, and fixed step count at a given mesh size. Ratios are medians " *
        "of independent process samples; 95% intervals resample numerator and denominator " *
        "groups independently.", "",
        "| N | Workers | Toy ns [IQR] | Pipeline ns [IQR] | Same-p toy/pipeline [95%] | Toy Eₚ [95%] | Pipeline Eₚ [95%] | Matched toy serial / pipeline Sₚ [95%] | Matched toy serial / pipeline Eₚ [95%] |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows
        push!(lines, "| $(row["n"]) | $(row["workers"]) | " *
            "$(fmt(row["toy_median_ns"])) [$(ci(row["toy_iqr_ns"]))] | " *
            "$(fmt(row["pipeline_median_ns"])) [$(ci(row["pipeline_iqr_ns"]))] | " *
            "$(fmt(row["same_worker_toy_over_pipeline"])) [$(ci(row["same_worker_gain_bootstrap95"]))] | " *
            "$(fmt(row["toy_self_efficiency"])) [$(ci(row["toy_self_efficiency_bootstrap95"]))] | " *
            "$(fmt(row["pipeline_self_efficiency"])) [$(ci(row["pipeline_self_efficiency_bootstrap95"]))] | " *
            "$(fmt(row["matched_toy_serial_to_pipeline_speedup"])) " *
            "[$(ci(row["matched_toy_serial_to_pipeline_speedup_bootstrap95"]))] | " *
            "$(fmt(row["matched_toy_serial_to_pipeline_efficiency"])) " *
            "[$(ci(row["matched_toy_serial_to_pipeline_efficiency_bootstrap95"]))] |")
    end
    append!(lines, ["", "Efficiency Eₚ is speedup Sₚ divided by p. Own-method Eₚ uses " *
        "that method's one-worker median. Same-worker absolute time and the toy/pipeline ratio " *
        "are the direct gain measures. Matched toy-serial columns use the frozen toy one-worker " *
        "median as a within-study reference. The fastest observed toy serial sample remains " *
        "in summary.json as a conservative check. Earlier Stage 2 observable halo paths " *
        "reached 2.446 ns at 2048² and 2.674 ns at 4096² in different cohorts; they do not " *
        "enter these confidence intervals. The timings do not measure DRAM bytes.", ""])
    return join(lines, "\n")
end

function main(args=ARGS)
    options = CachePipelineSummary.options(args)
    groups = Dict{Tuple{Int,Int,String},Vector{Float64}}()
    configurations = Any[]
    all_records = Any[]
    source_signature = nothing
    failed_launches = 0
    for cohort in options.cohorts
        source_signature, configuration, failures =
            check_cohort!(groups, all_records, cohort, source_signature)
        push!(configurations, configuration)
        failed_launches += failures
    end
    sizes = sort(unique(key[1] for key in keys(groups)))
    counts = sort(unique(key[2] for key in keys(groups)))
    for n in sizes, p in counts, mode in ("toy", "pipeline")
        length(get(groups, (n,p,mode), Float64[])) == 5 ||
            error("expected five processes for $((n,p,mode))")
    end
    rng = MersenneTwister(options.seed)
    rows = Dict{String,Any}[]
    for n in sizes
        serial_toy = groups[n,1,"toy"]
        serial_pipeline = groups[n,1,"pipeline"]
        strongest_serial_sample = minimum(serial_toy)
        for p in counts
            toy, pipe = groups[n,p,"toy"], groups[n,p,"pipeline"]
            tmed, pmed = median(toy), median(pipe)
            toy_e, pipe_e = median(serial_toy)/tmed, median(serial_pipeline)/pmed
            toy_e_ci = p == 1 ? [1.0,1.0] :
                interval_ratio(serial_toy, toy, rng, options.bootstrap)
            pipe_e_ci = p == 1 ? [1.0,1.0] :
                interval_ratio(serial_pipeline, pipe, rng, options.bootstrap)
            matched_ci = interval_ratio(serial_toy, pipe, rng, options.bootstrap)
            matched_gain = median(serial_toy)/pmed
            push!(rows, Dict{String,Any}(
                "n"=>n, "workers"=>p, "samples_per_mode"=>length(toy),
                "toy_median_ns"=>tmed,
                "toy_iqr_ns"=>[percentile(sort(toy), .25), percentile(sort(toy), .75)],
                "pipeline_median_ns"=>pmed,
                "pipeline_iqr_ns"=>[percentile(sort(pipe), .25), percentile(sort(pipe), .75)],
                "same_worker_toy_over_pipeline"=>tmed/pmed,
                "same_worker_gain_bootstrap95"=>interval_ratio(toy,pipe,rng,options.bootstrap),
                "toy_self_speedup"=>toy_e, "toy_self_speedup_bootstrap95"=>toy_e_ci,
                "toy_self_efficiency"=>toy_e/p,
                "toy_self_efficiency_bootstrap95"=>toy_e_ci ./ p,
                "pipeline_self_speedup"=>pipe_e,
                "pipeline_self_speedup_bootstrap95"=>pipe_e_ci,
                "pipeline_self_efficiency"=>pipe_e/p,
                "pipeline_self_efficiency_bootstrap95"=>pipe_e_ci ./ p,
                "matched_toy_serial_to_pipeline_speedup"=>matched_gain,
                "matched_toy_serial_to_pipeline_speedup_bootstrap95"=>matched_ci,
                "matched_toy_serial_to_pipeline_efficiency"=>matched_gain/p,
                "matched_toy_serial_to_pipeline_efficiency_bootstrap95"=>matched_ci ./ p,
                "fastest_matched_toy_serial_sample_ns"=>strongest_serial_sample,
                "fastest_matched_toy_serial_over_pipeline"=>strongest_serial_sample/pmed))
        end
    end
    mkpath(options.output)
    summary = Dict{String,Any}(
        "cohorts"=>options.cohorts, "source_sha256"=>source_signature,
        "bootstrap_replications"=>options.bootstrap, "bootstrap_seed"=>options.seed,
        "rows"=>rows, "pre_timing_failed_launches"=>failed_launches,
        "minimum_sample_seconds"=>minimum(row["time_seconds"] for row in all_records),
        "maximum_sample_seconds"=>maximum(row["time_seconds"] for row in all_records),
        "accepted_samples"=>length(all_records))
    writejson(joinpath(options.output, "summary.json"), summary)
    writecsv(joinpath(options.output, "summary.csv"), rows)
    write(joinpath(options.output, "report.md"), report_text(summary))
    chart(rows, sizes, counts, options.output)
end

end # module

abspath(PROGRAM_FILE) == (@__FILE__) && CachePipelineSummary.main()
