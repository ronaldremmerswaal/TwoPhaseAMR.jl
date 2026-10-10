#!/usr/bin/env julia
"""Read-only receipt and performance audit of a completed copied-source cohort."""
module AuditCopiedSourceStudy

include(joinpath(@__DIR__, "..", "..", "Support.jl"))
using .ScalingSupport
using JSON
using TOML

const ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
const RELATIVE = joinpath("src", "operators", "advection", "cell_advection.jl")
const WORKERS = (1, 4, 8)

require_equal(actual, expected, label) =
    actual == expected || error("$label differs: $actual versus $expected")

function audit_sources(output, config, label)
    frozen = joinpath(ROOT, "benchmark", "scaling", "results",
                      "20261010-narrow-ids-final", "current-source")
    require_equal(source_hash(frozen), config["source_tree_sha256"]["control"],
                  "frozen control tree")
    require_equal(sha256file(joinpath(frozen, RELATIVE)),
                  config["control_advection_sha256"], "frozen control advection")
    for variant in ("control", label)
        copied = joinpath(output, variant)
        require_equal(source_hash(copied), config["source_tree_sha256"][variant],
                      "$variant copied tree")
        expected = config[variant == "control" ? "control_advection_sha256" :
                          "candidate_advection_sha256"]
        require_equal(sha256file(joinpath(copied, RELATIVE)), expected,
                      "$variant copied advection")
    end
    receipt = readjson(joinpath(output, "probe-receipt.json"))
    require_equal(receipt, config["probe_receipt"], "probe receipt")
    require_equal(receipt["source"], frozen, "probe control source")
    require_equal(receipt["output"], config["candidate_source"],
                  "probe prepared candidate")
    require_equal(receipt["clean_tree_sha256"],
                  config["source_tree_sha256"]["control"], "probe control tree")
    require_equal(receipt["candidate_tree_sha256"],
                  config["source_tree_sha256"][label], "probe candidate tree")
    require_equal(receipt["clean_source_sha256"],
                  config["control_advection_sha256"], "probe control advection")
    require_equal(receipt["candidate_source_sha256"],
                  config["candidate_advection_sha256"], "probe candidate advection")
    if haskey(receipt, "parent")
        parent = receipt["parent"]
        require_equal(sha256file(joinpath(parent, RELATIVE)),
                      receipt["parent_source_sha256"], "probe parent advection")
        require_equal(source_hash(parent), receipt["parent_tree_sha256"],
                      "probe parent tree")
        require_equal(sha256file(joinpath(parent, "rolling-flux-receipt.json")),
                      receipt["parent_receipt_sha256"], "probe parent receipt")
        verification = readjson(joinpath(output, "parent-verification.json"))
        require_equal(verification["schema"], 1, "parent verification schema")
        require_equal(verification["candidate_receipt_sha256"],
                      sha256file(joinpath(output, "probe-receipt.json")),
                      "parent verification candidate receipt")
        for key in ("parent_source_sha256", "parent_tree_sha256",
                    "parent_receipt_sha256")
            require_equal(verification[key], receipt[key],
                          "parent verification $key")
        end
        require_equal(verification["wrapper_sha256"],
                      sha256file(joinpath(@__DIR__, "..", "rolling_flux_fused",
                                          "study.jl")), "fused wrapper hash")
        require_equal(verification["generic_study_sha256"],
                      sha256file(joinpath(@__DIR__, "study.jl")),
                      "generic study hash")
    end
    require_equal(sha256file(joinpath(output, "probe.patch")),
                  receipt["patch_sha256"], "probe patch")
end

function audit_tooling(output, config, label)
    for (path, expected) in config["locked_files_sha256"]
        isfile(path) || error("locked file missing: $path")
        require_equal(sha256file(path), expected, "locked $path")
    end
    provenance = TOML.parsefile(joinpath(output, "suite-provenance.toml"))
    require_equal(provenance, config["suite_provenance"], "suite provenance")
    require_equal(provenance["schema"], 1, "suite provenance schema")
    require_equal(provenance["status"], "both passed", "suite provenance status")
    require_equal(provenance["test_sha256"],
                  sha256file(joinpath(ROOT, "test", "operators", "advection",
                                      "cell_advection_compact.jl")),
                  "compact test hash")
    require_equal(provenance["validator_sha256"],
                  sha256file(provenance["validator"]), "validator hash")
    require_equal(length(provenance["runs"]), 2, "suite run count")
    for variant in ("control", label)
        run = only(run for run in provenance["runs"] if run["variant"] == variant)
        prepared = variant == "control" ?
            joinpath(ROOT, "benchmark", "scaling", "results",
                     "20261010-narrow-ids-final", "current-source") :
            config["candidate_source"]
        require_equal(run["source"], prepared, "$variant prepared source")
        require_equal(run["source_tree_sha256"], source_hash(prepared),
                      "$variant prepared tree")
        require_equal(run["source_tree_sha256"],
                      config["source_tree_sha256"][variant],
                      "$variant copied tree in suite")
        expected = config[variant == "control" ? "control_advection_sha256" :
                          "candidate_advection_sha256"]
        require_equal(run["source_advection_sha256"], expected,
                      "$variant suite advection")
        require_equal(run["exit_code"], 0, "$variant suite exit")
        report_path = joinpath(output, "suite-$variant.toml")
        require_equal(sha256file(report_path), run["report_sha256"],
                      "$variant suite report hash")
        report = TOML.parsefile(report_path)
        require_equal(report["status"], "passed", "$variant suite status")
        require_equal(report["threads"], 1, "$variant suite workers")
        require_equal(report["julia_version"], "1.10.12", "$variant suite Julia")
        require_equal(report["original_test_sha256"], provenance["test_sha256"],
                      "$variant suite test")
        require_equal(report["loaded_advection_sha256"], expected,
                      "$variant loaded advection hash")
        advection = realpath(joinpath(prepared, RELATIVE))
        require_equal(report["loaded_advection_file"], advection,
                      "$variant loaded advection path")
        methods = [key for key in keys(report) if endswith(key, "_method_file")]
        isempty(methods) && error("$variant suite reports no method path")
        for key in methods
            expected_method = (key == "ordered_sum_method_file" ||
                               variant == label) ? advection : ""
            require_equal(report[key], expected_method, "$variant $key")
        end
        for (key, value) in report
            if endswith(key, "_expected") && value isa Bool
                require_equal(value, variant == label, "$variant $key")
            elseif endswith(key, "_test_sha256") &&
                   key ∉ ("original_test_sha256", "adapted_test_sha256")
                pathkey = replace(key, "_sha256" => "_file")
                haskey(report, pathkey) || error("$variant $pathkey missing")
                require_equal(value, sha256file(report[pathkey]),
                              "$variant $key")
            end
        end
        require_equal(report["copied_module_file"],
                      joinpath(prepared, "src", "TwoPhaseAMR.jl"),
                      "$variant copied module path")
    end
    if haskey(provenance, "threaded_reports")
        extra_workers = [extra["threads"] for extra in provenance["threaded_reports"]]
        length(Set(extra_workers)) == length(extra_workers) && all(>=(1), extra_workers) ||
            error("extra suite worker list differs")
        for extra in provenance["threaded_reports"]
            p = extra["threads"]
            require_equal(extra["exit_code"], 0, "extra p$p suite exit")
            path = joinpath(output, "suite-threaded-p$p.toml")
            require_equal(sha256file(path), extra["report_sha256"],
                          "extra p$p suite report hash")
            report = TOML.parsefile(path)
            require_equal(report["status"], "passed", "extra p$p suite status")
            require_equal(report["threads"], p, "extra p$p suite workers")
            require_equal(report["loaded_advection_sha256"],
                          config["candidate_advection_sha256"],
                          "extra p$p suite source hash")
            advection = realpath(joinpath(config["candidate_source"], RELATIVE))
            methods = [key for key in keys(report) if endswith(key,"_method_file")]
            isempty(methods) && error("extra p$p reports no method path")
            for key in methods
                require_equal(report[key], advection, "extra p$p $key")
            end
            for (key,value) in report
                if endswith(key,"_expected") && value isa Bool
                    require_equal(value, true, "extra p$p $key")
                elseif endswith(key,"_test_sha256") &&
                       key ∉ ("original_test_sha256","adapted_test_sha256")
                    pathkey = replace(key,"_sha256" => "_file")
                    haskey(report,pathkey) || error("extra p$p $pathkey missing")
                    require_equal(value,sha256file(report[pathkey]),
                                  "extra p$p $key")
                end
            end
        end
    end
    oracle = joinpath(output, "oracle.bin")
    require_equal(sha256file(oracle), config["oracle_sha256"], "oracle file")
    report = TOML.parsefile(joinpath(output, "oracle.toml"))
    require_equal(report["field_sha256"], config["oracle_sha256"],
                  "oracle report field")
    require_equal(report["source"], joinpath(output, "control"),
                  "oracle control source")
    require_equal(report["steps"], config["steps"], "oracle steps")
end

function expected_command(output, config, variant, p, stem)
    cpus = config["placement"][string(p)]
    return [config["julia"], "--startup-file=no",
            "--project=$(config["project"])", "--threads=$p", "--gcthreads=1",
            joinpath(output, "checked_driver.jl"),
            "source=$(joinpath(output,variant))", "mode=$(config["mode"])",
            "kernel=$(config["kernel"])", "n=$(config["case"][1])",
            "block=$(config["case"][2])", "steps=$(config["steps"])",
            "cpus=$(join(cpus,','))", "input=$(joinpath(output,"initial.bin"))",
            "output=$(joinpath(output,"$stem.bin"))",
            "report=$(joinpath(output,"$stem.toml"))"]
end

function audit_commands(output, config, label)
    commands = readjsonl(joinpath(output, "commands.jsonl"))
    jobs = readjson(joinpath(output, "job_order.json"))
    require_equal(length(jobs), length(WORKERS)*config["repeats"], "job count")
    expected_jobs = Set((p, repeat) for p in WORKERS for
                        repeat in 0:config["repeats"]-1)
    require_equal(Set((job["workers"],job["repeat"]) for job in jobs),
                  expected_jobs, "job coverage")
    require_equal(length(commands), 1+2length(jobs), "child launch count")
    order = [(variant, job["workers"], job["repeat"]) for job in jobs for
             variant in job["order"]]
    require_equal(length(order), 2length(jobs), "job variant count")
    oracle = expected_command(output, config, "control", 1, "oracle")
    require_equal(commands[1]["command"], oracle, "oracle command")
    for (index, entry) in pairs(commands)
        require_equal(entry["exit_code"], 0, "launch $index exit")
        if index > 1
            variant, p, repeat = order[index-1]
            variant in ("control", label) || error("invalid variant in job order")
            stem = "$variant-n$(config["case"][1])-b$(config["case"][2])-p$p-r$repeat"
            require_equal(entry["command"],
                          expected_command(output, config, variant, p, stem),
                          "launch $index command")
        end
        command = entry["command"]
        p = parse(Int, split(command[4], '='; limit=2)[2])
        require_equal(entry["cpus"], config["placement"][string(p)],
                      "launch $index placement")
    end
    return length(commands)
end

function audit_rows(output, config, label)
    rows = readjsonl(joinpath(output, "raw.jsonl"))
    n, block = config["case"]
    expected_count = 2length(WORKERS)*config["repeats"]
    require_equal(length(rows), expected_count, "timed row count")
    expected_keys = Set((variant,p,repeat) for variant in ("control",label),
                        p in WORKERS, repeat in 0:config["repeats"]-1)
    require_equal(Set((row["variant"],row["threads"],row["repeat"]) for row in rows),
                  expected_keys, "timed case coverage")
    for row in rows
        variant,p = row["variant"],row["threads"]
        stem = "$variant-n$n-b$block-p$p-r$(row["repeat"])"
        require_equal(row["source"], joinpath(output, variant), "$stem source")
        require_equal(row["source_sha256"],
                      config[variant == "control" ? "control_advection_sha256" :
                             "candidate_advection_sha256"], "$stem source hash")
        require_equal((row["n"],row["block"],row["steps"]),
                      (n,block,config["steps"]), "$stem workload")
        require_equal((row["mode"],row["kernel"]),
                      (config["mode"],config["kernel"]), "$stem mode")
        require_equal(row["initial_sha256"],
                      sha256file(joinpath(output,"initial.bin")), "$stem input")
        require_equal(row["oracle_sha256"], config["oracle_sha256"],
                      "$stem oracle")
        require_equal(row["field_sha256"], config["oracle_sha256"],
                      "$stem full field")
        row["bitwise_equal"] && row["tolerance_violations"] == 0 &&
            row["max_abs_vs_oracle"] == 0 || error("$stem field mismatch")
        row["time_seconds"] >= config["minimum_seconds"] &&
            isfinite(row["ns_per_unknown_step"]) &&
            isapprox(row["ns_per_unknown_step"],
                     1e9row["time_seconds"]/(n*n*config["steps"]); rtol=1e-12) ||
            error("$stem timing/normalization differs")
        cpus = config["placement"][string(p)]
        require_equal([entry["cpu"] for entry in row["affinity"]], cpus,
                      "$stem CPU placement")
        all(entry["observed_cpu"] == entry["cpu"] for entry in row["affinity"]) ||
            error("$stem observed CPU differs")
        if config["workload"] == "fixed"
            require_equal(row["dt"], 0.35/n, "$stem fixed dt")
        else
            require_equal(row["cfl_scans_per_timed_step"], 1,
                          "$stem timed CFL scans")
            require_equal(row["dt"], 0.4row["cfl_bound"], "$stem CFL dt")
        end
    end
    summary = readjson(joinpath(output,"summary.json"))
    for key in ("case","steps","repeats","workload","candidate_label")
        require_equal(summary[key], config[key], "summary $key")
    end
    require_equal(summary["samples"], expected_count, "summary samples")
    serial_control = only(group["control_median_ns_per_unknown_step"] for group in
                          summary["groups"] if group["workers"] == 1)
    serial_candidate = only(group["candidate_median_ns_per_unknown_step"] for group in
                            summary["groups"] if group["workers"] == 1)
    strong = min(serial_control,serial_candidate)
    serial_variant = serial_control <= serial_candidate ? "control" : label
    serial_samples = [row["ns_per_unknown_step"] for row in rows if
                      row["variant"] == serial_variant && row["threads"] == 1]
    results = Any[]
    for p in WORKERS
        control = [only(row["ns_per_unknown_step"] for row in rows if
                        row["variant"] == "control" && row["threads"] == p &&
                        row["repeat"] == repeat) for repeat in 0:config["repeats"]-1]
        candidate = [only(row["ns_per_unknown_step"] for row in rows if
                          row["variant"] == label && row["threads"] == p &&
                          row["repeat"] == repeat) for repeat in 0:config["repeats"]-1]
        group = only(group for group in summary["groups"] if group["workers"] == p)
        require_equal(group["control_samples"], control, "p$p control samples")
        require_equal(group["candidate_samples"], candidate, "p$p candidate samples")
        require_equal(group["paired_ratios"], control ./ candidate,
                      "p$p paired ratios")
        cmed,xmed = median_value(control),median_value(candidate)
        require_equal(group["control_median_ns_per_unknown_step"], cmed,
                      "p$p control median")
        require_equal(group["candidate_median_ns_per_unknown_step"], xmed,
                      "p$p candidate median")
        require_equal(group["control_over_candidate_ratio"], cmed/xmed,
                      "p$p median ratio")
        require_equal(group["strongest_serial_efficiency"], strong/(p*xmed),
                      "p$p strongest serial efficiency")
        push!(results,Dict("workers" => p,
            "control_ns_per_cell_step" => cmed,
            "candidate_ns_per_cell_step" => xmed,
            "control_over_candidate" => cmed/xmed,
            "control_over_candidate_95_interval" =>
                bootstrap_ratio(control,candidate;seed=20261013+p),
            "strongest_serial_efficiency" => strong/(p*xmed),
            "strongest_serial_efficiency_95_interval" => p == 1 ? nothing :
                bootstrap_ratio(serial_samples,candidate,p;seed=20261113+p),
            "paired_ratios" => control ./ candidate))
    end
    return rows,results
end

function main(args=ARGS)
    length(args) == 1 || error("usage: audit_study.jl COMPLETED_OUTPUT_DIRECTORY")
    output = abspath(only(args))
    config = readjson(joinpath(output,"configuration.json"))
    label = config["candidate_label"]
    audit_sources(output,config,label)
    audit_tooling(output,config,label)
    commands = audit_commands(output,config,label)
    rows,results = audit_rows(output,config,label)
    JSON.print(stdout,Dict("status" => "passed","output" => output,
        "candidate_label" => label,"case" => config["case"],
        "workload" => config["workload"],"steps" => config["steps"],
        "repeats" => config["repeats"],"commands" => commands,
        "timed_rows" => length(rows),"oracle_sha256" => config["oracle_sha256"],
        "results" => results),2)
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end
