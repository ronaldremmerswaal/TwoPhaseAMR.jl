#!/usr/bin/env julia
"""Independently audit a gated copied-source CFL tile-max cohort."""
module AuditCFLTileMaxStudy

include(joinpath(@__DIR__, "..", "copied_source", "audit_study.jl"))
using .AuditCopiedSourceStudy
using JSON
using Random
using TOML

const Generic = AuditCopiedSourceStudy
const Support = Generic.ScalingSupport
const LABEL = "cfl_tile_max"
const WORKERS = (1, 4, 8)
const COHORT = joinpath(Generic.ROOT, "benchmark", "scaling", "results",
                       "20261010-narrow-ids-final")
const CONTROL_SHA = "047457e224baa1c5b5dfa26376eff71acd61de02b225ca7e13360df6c3184541"
const CONTROL_TREE = "0bb8826fb52d143ae0fa5f84e043f4ed4d4feeadb4c7328d6fbda33cbf7095a2"

equal(actual, expected, label) =
    actual == expected || error("$label differs")
valid(condition, label) = condition || error("$label differs")

function paired_ratio_interval(ratios, workers)
    rng = MersenneTwister(20261015 + workers)
    draws = [Support.median_value(rand(rng, ratios, length(ratios)))
             for _ in 1:5000]
    sort!(draws)
    return [Support.percentile(draws, 0.025),
            Support.percentile(draws, 0.975)]
end

function setup_command(gate, summary, variant, p)
    source = joinpath(gate, variant)
    driver = joinpath(gate, "checked_driver.jl")
    report = joinpath(gate, "setup-$variant-p$p.toml")
    return [summary["julia"], "--startup-file=no",
            "--project=$(summary["project"])", "--threads=$p", "--gcthreads=1",
            driver, "source=$source", "mode=setup", "kernel=upwind_cfl",
            "n=1024", "block=1024", "report=$report"]
end

function audit_gate(gate)
    gate = abspath(gate)
    summary_path = joinpath(gate, "setup-summary.json")
    summary = Support.readjson(summary_path)
    equal(summary["schema"], 1, "setup schema")
    equal(summary["status"], "passed", "setup status")
    equal(summary["case"], [1024, 1024], "setup case")
    equal(summary["workers"], collect(WORKERS), "setup workers")
    equal(summary["placement"],
          Support.readjson(joinpath(COHORT, "placement.json")), "setup placement")
    equal(summary["control_advection_sha256"], CONTROL_SHA, "setup control")
    equal(summary["control_tree_sha256"], CONTROL_TREE, "setup control tree")
    equal(Support.source_hash(joinpath(COHORT, "current-source")),
          CONTROL_TREE, "frozen control tree")
    equal(Support.sha256file(joinpath(COHORT, "current-source", Generic.RELATIVE)),
          CONTROL_SHA, "frozen control advection")
    receipt = Support.readjson(joinpath(gate, "probe-receipt.json"))
    equal(summary["candidate_receipt_sha256"],
          Support.sha256file(joinpath(gate, "probe-receipt.json")),
          "setup candidate receipt")
    equal(receipt["source"], joinpath(COHORT, "current-source"),
          "probe frozen source path")
    equal(receipt["clean_source_sha256"], CONTROL_SHA, "probe control")
    equal(receipt["clean_tree_sha256"], CONTROL_TREE, "probe control tree")
    equal(summary["candidate_advection_sha256"],
          receipt["candidate_source_sha256"], "setup candidate")
    equal(summary["candidate_tree_sha256"],
          receipt["candidate_tree_sha256"], "setup candidate tree")
    equal(Support.sha256file(joinpath(@__DIR__, "make_probe.jl")),
          receipt["generator_sha256"], "probe generator")
    equal(Support.sha256file(joinpath(gate, "probe.patch")),
          receipt["patch_sha256"], "probe patch")
    for (variant, expected_tree, expected_source) in
        (("control", CONTROL_TREE, CONTROL_SHA),
         (LABEL, receipt["candidate_tree_sha256"],
          receipt["candidate_source_sha256"]))
        source = joinpath(gate, variant)
        equal(Support.source_hash(source), expected_tree, "$variant copied tree")
        equal(Support.sha256file(joinpath(source, Generic.RELATIVE)),
              expected_source, "$variant copied source")
        equal(summary["source_trees"][variant], expected_tree,
              "$variant reported tree")
    end
    equal(Support.source_hash(receipt["output"]),
          receipt["candidate_tree_sha256"], "prepared candidate tree")
    equal(summary["validation_manifest_sha256"],
          Support.sha256file(joinpath(gate, "suite-provenance.toml")),
          "setup suite manifest")
    equal(summary["study_wrapper_sha256"],
          Support.sha256file(joinpath(@__DIR__, "study.jl")),
          "setup study wrapper")
    equal(summary["generic_study_sha256"],
          Support.sha256file(joinpath(@__DIR__, "..", "copied_source", "study.jl")),
          "setup generic study")
    equal(summary["driver_sha256"],
          Support.sha256file(joinpath(gate, "checked_driver.jl")),
          "setup driver")
    equal(summary["setup_rows_sha256"],
          Support.sha256file(joinpath(gate, "setup-rows.jsonl")),
          "setup rows hash")
    equal(summary["commands_sha256"],
          Support.sha256file(joinpath(gate, "commands.jsonl")),
          "setup commands hash")
    locked = summary["locked_files_sha256"]
    manifest = TOML.parsefile(joinpath(gate, "suite-provenance.toml"))
    for path in (joinpath(@__DIR__, "make_probe.jl"),
                 joinpath(@__DIR__, "setup_gate.jl"),
                 joinpath(@__DIR__, "study.jl"),
                 joinpath(gate, "checked_driver.jl"),
                 joinpath(gate, "suite-provenance.toml"),
                 joinpath(COHORT, "placement.json"),
                 summary["julia"],
                 joinpath(summary["project"], "Project.toml"),
                 joinpath(summary["project"], "Manifest.toml"),
                 manifest["validator"], manifest["test"])
        haskey(locked, path) || error("setup lock missing: $path")
    end
    if haskey(manifest, "focused_test")
        haskey(locked, manifest["focused_test"]) ||
            error("setup focused test lock missing")
    end
    for (path, hash) in locked
        valid(isfile(path), "setup locked file missing: $path")
        equal(Support.sha256file(path), hash, "setup lock $path")
    end
    rows = Support.readjsonl(joinpath(gate, "setup-rows.jsonl"))
    commands = Support.readjsonl(joinpath(gate, "commands.jsonl"))
    expected = [(variant, p) for p in WORKERS for variant in ("control", LABEL)]
    equal(length(rows), length(expected), "setup row count")
    equal(length(commands), length(expected), "setup child count")
    for (index, (variant, p)) in enumerate(expected)
        entry, row = commands[index], rows[index]
        equal(entry["command"], setup_command(gate, summary, variant, p),
              "setup $variant p$p command")
        equal(entry["cpus"], summary["placement"][string(p)],
              "setup $variant p$p placement")
        equal(entry["exit_code"], 0, "setup $variant p$p exit")
        report = TOML.parsefile(joinpath(gate, "setup-$variant-p$p.toml"))
        equal(row, merge(report, Dict("variant" => variant)),
              "setup $variant p$p report")
        for (key, value) in (("source", joinpath(gate, variant)),
                             ("mode", "setup"), ("kernel", "upwind_cfl"),
                             ("n", 1024), ("block", 1024), ("threads", p),
                             ("blocks", 1), ("faces", 2*1024*1024),
                             ("julia_version", "1.10.12"))
            equal(row[key], value, "setup $variant p$p $key")
        end
        valid(isfinite(row["mesh_setup_seconds"]) &&
              isfinite(row["workspace_setup_seconds"]) &&
              row["mesh_setup_seconds"] >= 0 &&
              row["workspace_setup_seconds"] >= 0,
              "setup $variant p$p durations")
    end
    return summary, Support.sha256file(summary_path)
end

function audit_timed(output, gate)
    output = abspath(output)
    gate = abspath(gate)
    summary, gate_hash = audit_gate(gate)
    config = Support.readjson(joinpath(output, "configuration.json"))
    equal(config["schema"], 1, "timed schema")
    equal(config["candidate_label"], LABEL, "timed candidate label")
    equal(config["case"], [1024, 1024], "timed case")
    equal(config["steps"], 256, "timed steps")
    valid(config["repeats"] in (2, 3), "timed exploratory repetitions")
    equal(config["workload"], "cfl", "timed workload")
    equal((config["mode"], config["kernel"]),
          ("measure_cfl", "upwind_cfl"), "timed mode/kernel")
    equal(config["control_advection_sha256"], CONTROL_SHA,
          "timed control")
    equal(config["candidate_advection_sha256"],
          summary["candidate_advection_sha256"], "timed candidate")
    equal(config["source_tree_sha256"]["control"],
          summary["control_tree_sha256"], "timed control tree")
    equal(config["source_tree_sha256"][LABEL],
          summary["candidate_tree_sha256"], "timed candidate tree")
    equal(Support.sha256file(joinpath(output, "probe-receipt.json")),
          summary["candidate_receipt_sha256"], "timed probe receipt")
    equal(Support.sha256file(joinpath(output, "suite-provenance.toml")),
          summary["validation_manifest_sha256"], "timed validation manifest")
    equal(config["placement"], summary["placement"], "timed placement")
    equal(Support.sha256file(joinpath(output, "initial.bin")),
          Support.sha256file(joinpath(COHORT, "initial-n1024.bin")),
          "timed frozen initial field")
    for key in ("julia", "project", "depot")
        equal(config[key], summary[key], "timed $key")
    end
    equal(Support.sha256file(joinpath(output, "checked_driver.jl")),
          summary["driver_sha256"], "timed driver")
    Generic.audit_sources(output, config, LABEL)
    Generic.audit_tooling(output, config, LABEL)
    launches = Generic.audit_commands(output, config, LABEL)
    rows, results = Generic.audit_rows(output, config, LABEL)
    for group in results
        group["paired_ratio_median"] = Support.median_value(group["paired_ratios"])
        group["paired_ratio_median_95_interval"] =
            paired_ratio_interval(group["paired_ratios"], group["workers"])
    end
    oracle = TOML.parsefile(joinpath(output, "oracle.toml"))
    valid(isfinite(oracle["cfl_bound"]) && oracle["cfl_bound"] > 0,
          "oracle CFL bound")
    equal(oracle["dt"], 0.4*oracle["cfl_bound"], "oracle dt")
    for row in rows
        stem = "$(row["variant"]) p$(row["threads"]) r$(row["repeat"])"
        equal(row["cfl_bound"], oracle["cfl_bound"], "$stem CFL bound")
        equal(row["dt"], oracle["dt"], "$stem dt")
        equal(row["field_sha256"], config["oracle_sha256"], "$stem field")
    end
    verification = Support.readjson(joinpath(output, "tile-max-verification.json"))
    for (key, value) in (("schema", 1), ("status", "passed"),
                         ("candidate_label", LABEL), ("setup_gate", gate),
                         ("setup_gate_sha256", gate_hash),
                         ("driver_sha256", summary["driver_sha256"]),
                         ("oracle_cfl_bound", oracle["cfl_bound"]),
                         ("oracle_field_sha256", config["oracle_sha256"]),
                         ("timed_rows", length(rows)),
                         ("study_wrapper_sha256",
                          Support.sha256file(joinpath(@__DIR__, "study.jl"))),
                         ("generic_study_sha256",
                          Support.sha256file(joinpath(@__DIR__, "..", "copied_source",
                                                  "study.jl"))))
        equal(verification[key], value, "tile-max verification $key")
    end
    return Dict("status" => "passed", "output" => output,
                "setup_gate" => gate, "setup_children" => 6,
                "timed_children" => launches, "timed_rows" => length(rows),
                "oracle_sha256" => config["oracle_sha256"],
                "oracle_cfl_bound" => oracle["cfl_bound"],
                "results" => results)
end

function main(args=ARGS)
    if length(args) == 2 && args[1] == "setup"
        summary, hash = audit_gate(args[2])
        JSON.print(stdout, Dict("status" => "passed", "setup_gate" => abspath(args[2]),
                                "setup_gate_sha256" => hash,
                                "candidate_advection_sha256" =>
                                    summary["candidate_advection_sha256"],
                                "setup_children" => 6), 2)
    elseif length(args) == 3 && args[1] == "measure"
        JSON.print(stdout, audit_timed(args[2], args[3]), 2)
    else
        error("usage: audit_study.jl setup PASSED_GATE | measure COMPLETED_OUTPUT PASSED_GATE")
    end
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end
