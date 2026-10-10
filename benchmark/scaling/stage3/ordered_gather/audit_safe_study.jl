#!/usr/bin/env julia
"""Read-only receipt and serial-baseline audit for a completed safe study."""
module AuditSafeOrderedGather

include(joinpath(@__DIR__, "..", "..", "Support.jl"))
using .ScalingSupport
using JSON
using TOML

const ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
const COHORT = joinpath(ROOT, "benchmark", "scaling", "results",
                        "20261010-narrow-ids-final")
const RELATIVE = joinpath("src", "operators", "advection", "cell_advection.jl")
const VARIANTS = ("control", "pair_select")
const WORKERS = (1, 4, 8)

require_equal(actual, expected, label) =
    actual == expected || error("$label differs: $actual versus $expected")

function audit_sources(output, config)
    cohort_config = readjson(joinpath(COHORT, "configuration.json"))
    cohort_summary = readjson(joinpath(COHORT, "summary.json"))
    cohort_raw = readjsonl(joinpath(COHORT, "raw.jsonl"))
    require_equal(cohort_config["phase"], "matrix", "cohort phase")
    require_equal(cohort_summary["samples"], length(cohort_raw), "cohort rows")
    all(row["bitwise_equal"] && row["tolerance_violations"] == 0 for row in cohort_raw) ||
        error("cohort has a mismatched field")
    cohort_source = joinpath(COHORT, "current-source")
    require_equal(source_hash(cohort_source), config["source_tree_sha256"]["control"],
                  "original cohort/control source tree")
    cohort_manifest = readjson(joinpath(COHORT, "manifest.json"))
    original_advection = joinpath(ROOT, RELATIVE)
    locked_advection = get(cohort_manifest, original_advection,
                           get(cohort_manifest, RELATIVE, nothing))
    require_equal(locked_advection, config["control_advection_sha256"],
                  "cohort advection lock")

    for variant in VARIANTS
        source = joinpath(output, variant)
        require_equal(source_hash(source), config["source_tree_sha256"][variant],
                      "$variant source tree")
        require_equal(sha256file(joinpath(source, RELATIVE)),
                      config[variant == "control" ? "control_advection_sha256" :
                             "candidate_advection_sha256"], "$variant advection")
        suite = TOML.parsefile(joinpath(output, "suite-$variant.toml"))
        require_equal(suite, config["copied_source_suites"][variant],
                      "$variant copied suite receipt")
        require_equal(suite["status"], "passed", "$variant suite status")
        require_equal(suite["threads"], variant == "control" ? 1 : 4,
                      "$variant suite threads")
        for (key, expected) in
            (("loaded_advection_file", joinpath(source, RELATIVE)),
             ("ordered_sum_method_file", joinpath(source, RELATIVE)),
             ("copied_module_file", joinpath(source, "src", "TwoPhaseAMR.jl")),
             ("loaded_advection_sha256", sha256file(joinpath(source, RELATIVE))),
             ("copied_module_sha256", sha256file(joinpath(source, "src", "TwoPhaseAMR.jl"))),
             ("original_test_sha256", config["compact_test_sha256"]),
             ("order_checks", 222), ("canonical_fast_checks", 36),
             ("generic_fallback_checks", 186), ("permutation_count", 24))
            require_equal(suite[key], expected, "$variant $key")
        end
        require_equal(suite["generic_fallback_method_file"],
                      variant == "control" ? "" : joinpath(source, RELATIVE),
                      "$variant fallback method")
    end
    receipt = readjson(joinpath(output, "pair_select", "ordered-gather-receipt.json"))
    require_equal(receipt["clean_source_sha256"], config["control_advection_sha256"],
                  "probe input hash")
    require_equal(receipt["candidate_source_sha256"],
                  config["candidate_advection_sha256"], "probe output hash")
    require_equal(receipt["candidate_tree_sha256"],
                  config["source_tree_sha256"]["pair_select"], "probe tree hash")
    return cohort_manifest
end

function audit_tooling(output, config, cohort_manifest)
    locked = ((joinpath(output, "checked_driver.jl"), "driver_sha256"),
              (joinpath(@__DIR__, "..", "main_driver.jl"), "source_driver_sha256"),
              (joinpath(@__DIR__, "make_checked_driver.jl"), "generator_sha256"),
              (joinpath(@__DIR__, "safe_study.jl"), "study_sha256"),
              (joinpath(@__DIR__, "validation_driver.jl"), "validation_driver_sha256"),
              (joinpath(ROOT, "test", "operators", "advection",
                        "cell_advection_compact.jl"), "compact_test_sha256"))
    for (path, key) in locked
        require_equal(sha256file(path), config[key], key)
    end
    project = config["project"]
    for name in ("Project.toml", "Manifest.toml")
        path = joinpath(project, name)
        require_equal(sha256file(path), cohort_manifest[path], "numerical $name")
    end
    require_equal(sha256file(config["julia"]), cohort_manifest[config["julia"]],
                  "numerical Julia binary")
    for name in ("Project.toml", "Manifest.toml")
        path = joinpath(ROOT, "benchmark", "scaling", name)
        require_equal(sha256file(path), cohort_manifest[path], "tooling $name")
    end
    initial = joinpath(output, "initial-n$(config["case"][1]).bin")
    require_equal(sha256file(initial), config["initial_sha256"], "initial field")
    oracle = joinpath(output, "oracle.bin")
    oracle_receipt = TOML.parsefile(joinpath(output, "oracle.toml"))
    oracle_hash = sha256file(oracle)
    require_equal(oracle_receipt["field_sha256"], oracle_hash, "oracle field")
    require_equal(oracle_receipt["source"], joinpath(output, "control"),
                  "oracle source")
    require_equal(oracle_receipt["steps"], config["steps"], "oracle steps")
    return oracle_hash
end

function audit_commands(output, config)
    commands = readjsonl(joinpath(output, "commands.jsonl"))
    expected_count = 3 + length(VARIANTS) * length(WORKERS) * config["repeats"]
    require_equal(length(commands), expected_count, "launch count")
    for (index, entry) in pairs(commands)
        require_equal(entry["exit_code"], 0, "launch $index exit")
        command = entry["command"]
        require_equal(command[1], config["julia"], "launch $index Julia")
        require_equal(command[3], "--project=$(config["project"])",
                      "launch $index project")
        threads = only(parse(Int, split(part, '='; limit=2)[2]) for part in command
                       if startswith(part, "--threads="))
        require_equal(entry["cpus"], config["placement"][string(threads)],
                      "launch $index CPU placement")
    end
    return length(commands)
end

function audit_rows(output, config, oracle_hash)
    rows = readjsonl(joinpath(output, "raw.jsonl"))
    n, block = config["case"]
    expected_count = length(VARIANTS) * length(WORKERS) * config["repeats"]
    require_equal(length(rows), expected_count, "timed row count")
    expected_keys = Set((variant, p, repeat) for variant in VARIANTS,
                        p in WORKERS, repeat in 0:config["repeats"]-1)
    actual_keys = Set((row["variant"], row["threads"], row["repeat"]) for row in rows)
    require_equal(actual_keys, expected_keys, "timed case coverage")
    for row in rows
        label = "$(row["variant"]) p$(row["threads"]) r$(row["repeat"])"
        variant = row["variant"]
        require_equal(row["source"], joinpath(output, variant), "$label source")
        require_equal(row["source_sha256"],
                      config[variant == "control" ? "control_advection_sha256" :
                             "candidate_advection_sha256"], "$label source hash")
        require_equal((row["n"], row["block"], row["steps"]),
                      (n, block, config["steps"]), "$label workload")
        require_equal(row["initial_sha256"], config["initial_sha256"],
                      "$label input hash")
        require_equal(row["oracle_sha256"], oracle_hash, "$label oracle hash")
        require_equal(row["field_sha256"], oracle_hash, "$label full field hash")
        row["bitwise_equal"] && row["tolerance_violations"] == 0 &&
            row["max_abs_vs_oracle"] == 0 || error("$label field differs")
        row["time_seconds"] >= config["minimum_seconds"] &&
            isfinite(row["ns_per_unknown_step"]) || error("$label short/invalid timing")
        isapprox(row["ns_per_unknown_step"],
                 1e9row["time_seconds"] / (n*n*config["steps"]); rtol=1e-12) ||
            error("$label ns/cell-step normalization differs")
        cpus = config["placement"][string(row["threads"])]
        require_equal([entry["cpu"] for entry in row["affinity"]], cpus,
                      "$label affinity")
        all(entry["observed_cpu"] == entry["cpu"] for entry in row["affinity"]) ||
            error("$label observed affinity differs")
    end
    summary = readjson(joinpath(output, "summary.json"))
    require_equal(summary["samples"], expected_count, "summary row count")
    require_equal(summary["case"], config["case"], "summary case")
    require_equal(summary["steps"], config["steps"], "summary steps")
    require_equal(summary["repeats"], config["repeats"], "summary repeats")
    groups = Dict{Int,Any}()
    for p in WORKERS
        control = [only(row["ns_per_unknown_step"] for row in rows if
                        row["variant"] == "control" && row["threads"] == p &&
                        row["repeat"] == r) for r in 0:config["repeats"]-1]
        candidate = [only(row["ns_per_unknown_step"] for row in rows if
                          row["variant"] == "pair_select" && row["threads"] == p &&
                          row["repeat"] == r) for r in 0:config["repeats"]-1]
        reported = only(group for group in summary["groups"] if group["workers"] == p)
        require_equal(reported["control_samples"], control, "p$p control samples")
        require_equal(reported["candidate_samples"], candidate, "p$p candidate samples")
        require_equal(reported["paired_ratios"], control ./ candidate,
                      "p$p paired ratios")
        groups[p] = (; control, candidate, control_median=median_value(control),
                     candidate_median=median_value(candidate))
    end
    strongest_serial = min(groups[1].control_median, groups[1].candidate_median)
    results = [Dict("workers" => p,
                    "control_ns_per_cell_step" => groups[p].control_median,
                    "candidate_ns_per_cell_step" => groups[p].candidate_median,
                    "control_over_candidate" => groups[p].control_median /
                                                 groups[p].candidate_median,
                    "candidate_self_efficiency" => groups[1].candidate_median /
                                                   (p*groups[p].candidate_median),
                    "strongest_serial_efficiency" => strongest_serial /
                                                     (p*groups[p].candidate_median),
                    "paired_ratios" => groups[p].control ./ groups[p].candidate)
               for p in WORKERS]
    return rows, results, strongest_serial
end

function main(args=ARGS)
    length(args) == 1 || error("usage: audit_safe_study.jl COMPLETED_OUTPUT_DIRECTORY")
    output = abspath(only(args))
    config = readjson(joinpath(output, "configuration.json"))
    cohort_manifest = audit_sources(output, config)
    oracle_hash = audit_tooling(output, config, cohort_manifest)
    commands = audit_commands(output, config)
    rows, results, strongest_serial = audit_rows(output, config, oracle_hash)
    JSON.print(stdout, Dict("status" => "passed", "output" => output,
                            "case" => config["case"], "steps" => config["steps"],
                            "repeats" => config["repeats"], "commands" => commands,
                            "timed_rows" => length(rows), "oracle_sha256" => oracle_hash,
                            "strongest_serial_ns_per_cell_step" => strongest_serial,
                            "results" => results), 2)
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end
