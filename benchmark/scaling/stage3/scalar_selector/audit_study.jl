#!/usr/bin/env julia
"""Read-only receipt and performance audit of one completed selector cohort."""
module AuditScalarSelectorStudy

include(joinpath(@__DIR__, "..", "..", "Support.jl"))
using .ScalingSupport
using JSON
using TOML

const ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
const RELATIVE = joinpath("src", "operators", "advection", "cell_advection.jl")
const VARIANTS = ("control", "scalar_selector")
const WORKERS = (1, 4, 8)

require_equal(actual, expected, label) =
    actual == expected || error("$label differs: $actual versus $expected")

function audit_sources(output, config)
    cohort = joinpath(ROOT, "benchmark", "scaling", "results",
                      "20261010-narrow-ids-final")
    original = joinpath(cohort, "current-source")
    require_equal(source_hash(original), config["source_tree_sha256"]["control"],
                  "frozen cohort/control source tree")
    require_equal(sha256file(joinpath(original, RELATIVE)),
                  config["control_advection_sha256"], "frozen cohort advection")
    for variant in VARIANTS
        copied = joinpath(output, variant)
        require_equal(source_hash(copied), config["source_tree_sha256"][variant],
                      "$variant copied source tree")
        require_equal(sha256file(joinpath(copied, RELATIVE)),
                      config[variant == "control" ? "control_advection_sha256" :
                             "candidate_advection_sha256"], "$variant advection")
    end
    receipt = readjson(joinpath(output, "scalar_selector", "scalar-selector-receipt.json"))
    require_equal(receipt, config["probe_receipt"], "probe receipt")
    require_equal(receipt["clean_tree_sha256"],
                  config["source_tree_sha256"]["control"], "probe clean tree")
    require_equal(receipt["candidate_tree_sha256"],
                  config["source_tree_sha256"]["scalar_selector"], "probe candidate tree")
    require_equal(receipt["clean_source_sha256"],
                  config["control_advection_sha256"], "probe clean advection")
    require_equal(receipt["candidate_source_sha256"],
                  config["candidate_advection_sha256"], "probe candidate advection")
end

function audit_tooling(output, config)
    for (path, expected) in config["locked_files_sha256"]
        isfile(path) || error("locked file missing: $path")
        require_equal(sha256file(path), expected, "locked $path")
    end
    provenance = TOML.parsefile(joinpath(output, "suite-provenance.toml"))
    require_equal(provenance, config["suite_provenance"], "suite provenance")
    for variant in VARIANTS
        suite = TOML.parsefile(joinpath(output, "suite-$variant.toml"))
        expected_sha = config[variant == "control" ? "control_advection_sha256" :
                              "candidate_advection_sha256"]
        require_equal(suite["status"], "passed", "$variant suite status")
        require_equal(suite["order_checks"], 2304, "$variant all-code checks")
        require_equal(suite["packed_code_count"], 256, "$variant code count")
        require_equal(suite["loaded_advection_sha256"], expected_sha,
                      "$variant suite source hash")
        require_equal(suite["selector_expected"], variant == "scalar_selector",
                      "$variant selector presence")
        require_equal(suite["original_test_sha256"],
                      sha256file(joinpath(ROOT, "test", "operators", "advection",
                                          "cell_advection_compact.jl")),
                      "$variant compact test")
        source = if haskey(provenance, "runs")
            only(run["source"] for run in provenance["runs"] if
                 run["variant"] == variant)
        else
            joinpath(output, variant)
        end
        advection = realpath(joinpath(source, RELATIVE))
        require_equal(suite["loaded_advection_file"], advection,
                      "$variant loaded method source")
        require_equal(suite["ordered_sum_method_file"], advection,
                      "$variant ordered method source")
        require_equal(suite["scalar_selector_method_file"],
                      variant == "control" ? "" : advection,
                      "$variant selector method source")
        if haskey(provenance, "runs")
            run = only(run for run in provenance["runs"] if run["variant"] == variant)
            require_equal(run["exit_code"], 0, "$variant reused-suite exit")
            require_equal(run["source_tree_sha256"], source_hash(source),
                          "$variant reused-suite source tree")
            require_equal(run["source_advection_sha256"], expected_sha,
                          "$variant reused-suite advection")
            require_equal(run["report_sha256"],
                          sha256file(joinpath(output, "suite-$variant.toml")),
                          "$variant reused-suite report")
        end
    end
    oracle = joinpath(output, "oracle.bin")
    require_equal(sha256file(oracle), config["oracle_sha256"], "oracle file")
    oracle_report = TOML.parsefile(joinpath(output, "oracle.toml"))
    require_equal(oracle_report["field_sha256"], config["oracle_sha256"],
                  "oracle report field")
    require_equal(oracle_report["source"], joinpath(output, "control"),
                  "oracle control source")
    require_equal(oracle_report["steps"], config["steps"], "oracle steps")
    return oracle_report
end

function audit_commands(output, config)
    commands = readjsonl(joinpath(output, "commands.jsonl"))
    jobs = readjson(joinpath(output, "job_order.json"))
    require_equal(length(jobs), length(WORKERS)*config["repeats"], "job count")
    expected_jobs = Set((p, repeat) for p in WORKERS for
                        repeat in 0:config["repeats"]-1)
    require_equal(Set((job["workers"],job["repeat"]) for job in jobs),
                  expected_jobs, "job coverage")
    expected = (haskey(config["suite_provenance"], "runs") ? 1 : 3) +
               length(VARIANTS)*length(WORKERS)*config["repeats"]
    require_equal(length(commands), expected, "child launch count")
    first_timed = expected - 2length(jobs) + 1
    expected_order = [(variant, job["workers"], job["repeat"])
                      for job in jobs for variant in job["order"]]
    require_equal(length(expected_order), 2length(jobs), "job variant count")
    for (index, entry) in pairs(commands)
        require_equal(entry["exit_code"], 0, "launch $index exit")
        command = entry["command"]
        require_equal(command[1], config["julia"], "launch $index Julia")
        require_equal(command[3], "--project=$(config["project"])",
                      "launch $index numerical project")
        p = only(parse(Int, split(arg, '='; limit=2)[2]) for arg in command
                 if startswith(arg, "--threads="))
        require_equal(entry["cpus"], config["placement"][string(p)],
                      "launch $index placement")
        if index >= first_timed
            variant, workers, repeat = expected_order[index-first_timed+1]
            require_equal(p, workers, "launch $index worker order")
            stem = "$variant-n$(config["case"][1])-b$(config["case"][2])-p$p-r$repeat"
            require_equal(command[6], joinpath(output,"checked_driver.jl"),
                          "launch $index driver")
            require_equal(command[7:end], [
                "source=$(joinpath(output,variant))",
                "mode=$(config["mode"])", "kernel=$(config["kernel"])",
                "n=$(config["case"][1])", "block=$(config["case"][2])",
                "steps=$(config["steps"])",
                "cpus=$(join(config["placement"][string(p)],','))",
                "input=$(joinpath(output,"initial.bin"))",
                "output=$(joinpath(output,"$stem.bin"))",
                "report=$(joinpath(output,"$stem.toml"))"],
                "launch $index workload/order")
        end
    end
    return length(commands)
end

function audit_rows(output, config)
    rows = readjsonl(joinpath(output, "raw.jsonl"))
    n, block = config["case"]
    expected_count = length(VARIANTS)*length(WORKERS)*config["repeats"]
    require_equal(length(rows), expected_count, "timed row count")
    expected_keys = Set((variant,p,repeat) for variant in VARIANTS,
                        p in WORKERS, repeat in 0:config["repeats"]-1)
    actual_keys = Set((row["variant"],row["threads"],row["repeat"]) for row in rows)
    require_equal(actual_keys, expected_keys, "timed case coverage")
    for row in rows
        variant, p = row["variant"], row["threads"]
        label = "$variant p$p r$(row["repeat"])"
        require_equal(row["source"], joinpath(output, variant), "$label source")
        require_equal(row["source_sha256"],
                      config[variant == "control" ? "control_advection_sha256" :
                             "candidate_advection_sha256"], "$label source hash")
        require_equal((row["n"],row["block"],row["steps"]),
                      (n,block,config["steps"]), "$label workload")
        require_equal((row["mode"],row["kernel"]),
                      (config["mode"],config["kernel"]), "$label mode")
        require_equal(row["initial_sha256"],
                      sha256file(joinpath(output,"initial.bin")), "$label input hash")
        require_equal(row["oracle_sha256"], config["oracle_sha256"],
                      "$label oracle hash")
        require_equal(row["field_sha256"], config["oracle_sha256"],
                      "$label full-field hash")
        row["bitwise_equal"] && row["tolerance_violations"] == 0 &&
            row["max_abs_vs_oracle"] == 0 || error("$label field mismatch")
        row["time_seconds"] >= config["minimum_seconds"] &&
            isfinite(row["ns_per_unknown_step"]) &&
            isapprox(row["ns_per_unknown_step"],
                     1e9row["time_seconds"]/(n*n*config["steps"]); rtol=1e-12) ||
            error("$label timing/normalization differs")
        cpus = config["placement"][string(p)]
        require_equal([entry["cpu"] for entry in row["affinity"]], cpus,
                      "$label CPU placement")
        all(entry["observed_cpu"] == entry["cpu"] for entry in row["affinity"]) ||
            error("$label observed CPU differs")
        if config["workload"] == "fixed"
            require_equal(row["dt"], 0.35/n, "$label fixed dt")
        else
            require_equal(row["cfl_scans_per_timed_step"], 1,
                          "$label timed CFL scans")
            require_equal(row["dt"], 0.4row["cfl_bound"], "$label CFL dt")
        end
    end
    summary = readjson(joinpath(output,"summary.json"))
    require_equal(summary["samples"], expected_count, "summary row count")
    require_equal(summary["case"], config["case"], "summary case")
    require_equal(summary["steps"], config["steps"], "summary steps")
    require_equal(summary["repeats"], config["repeats"], "summary repeats")
    require_equal(summary["workload"], config["workload"], "summary workload")
    serial_control = only(group["control_median_ns_per_unknown_step"] for group in
                          summary["groups"] if group["workers"] == 1)
    serial_candidate = only(group["candidate_median_ns_per_unknown_step"] for group in
                            summary["groups"] if group["workers"] == 1)
    strong = min(serial_control, serial_candidate)
    serial_variant = serial_control <= serial_candidate ? "control" : "scalar_selector"
    serial_samples = [row["ns_per_unknown_step"] for row in rows if
                      row["variant"] == serial_variant && row["threads"] == 1]
    results = Any[]
    for p in WORKERS
        control = [only(row["ns_per_unknown_step"] for row in rows if
                        row["variant"] == "control" && row["threads"] == p &&
                        row["repeat"] == repeat) for repeat in 0:config["repeats"]-1]
        candidate = [only(row["ns_per_unknown_step"] for row in rows if
                          row["variant"] == "scalar_selector" && row["threads"] == p &&
                          row["repeat"] == repeat) for repeat in 0:config["repeats"]-1]
        group = only(group for group in summary["groups"] if group["workers"] == p)
        require_equal(group["control_samples"], control, "p$p control samples")
        require_equal(group["candidate_samples"], candidate, "p$p candidate samples")
        require_equal(group["paired_ratios"], control ./ candidate,
                      "p$p paired ratios")
        cmed, xmed = median_value(control), median_value(candidate)
        require_equal(group["control_median_ns_per_unknown_step"], cmed,
                      "p$p control median")
        require_equal(group["candidate_median_ns_per_unknown_step"], xmed,
                      "p$p candidate median")
        require_equal(group["strongest_serial_efficiency"], strong/(p*xmed),
                      "p$p strongest serial efficiency")
        push!(results, Dict("workers" => p,
                            "control_ns_per_cell_step" => cmed,
                            "candidate_ns_per_cell_step" => xmed,
                            "control_over_candidate" => cmed/xmed,
                            "control_over_candidate_95_interval" =>
                                bootstrap_ratio(control,candidate;
                                                seed=20261011+p),
                            "strongest_serial_efficiency" => strong/(p*xmed),
                            "strongest_serial_efficiency_95_interval" =>
                                p == 1 ? nothing :
                                    bootstrap_ratio(
                                        serial_samples,
                                        candidate,p; seed=20261111+p),
                            "paired_ratios" => control ./ candidate))
    end
    return rows, results
end

function main(args=ARGS)
    length(args) == 1 || error("usage: audit_study.jl COMPLETED_OUTPUT_DIRECTORY")
    output = abspath(only(args))
    config = readjson(joinpath(output,"configuration.json"))
    audit_sources(output,config)
    audit_tooling(output,config)
    commands = audit_commands(output,config)
    rows, results = audit_rows(output,config)
    JSON.print(stdout, Dict("status" => "passed", "output" => output,
                            "case" => config["case"],
                            "workload" => config["workload"],
                            "steps" => config["steps"],
                            "repeats" => config["repeats"],
                            "commands" => commands,
                            "timed_rows" => length(rows),
                            "oracle_sha256" => config["oracle_sha256"],
                            "results" => results), 2)
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end
