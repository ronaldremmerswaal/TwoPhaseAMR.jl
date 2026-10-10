#!/usr/bin/env julia
"""Gate a copied tile-max candidate, then reuse the frozen paired CFL study."""
module CFLTileMaxStudy

include(joinpath(@__DIR__, "..", "copied_source", "study.jl"))
using .CopiedSourceStudy
using TOML

const Generic = CopiedSourceStudy
const Support = Generic.Support
const LABEL = "cfl_tile_max"
const BASELINE_SHA256 = Generic.Shared.Probe.EXPECTED_CLEAN_SHA256
const BASELINE_TREE_SHA256 =
    "0bb8826fb52d143ae0fa5f84e043f4ed4d4feeadb4c7328d6fbda33cbf7095a2"

function options(args)
    iseven(length(args)) && length(unique(args[1:2:end])) == length(args)÷2 ||
        error("usage: study.jl GENERIC_OPTIONS --setup-gate PASSED_DIR")
    pairs = Dict(args[i] => args[i+1] for i in 1:2:length(args))
    haskey(pairs,"--setup-gate") || error("passed setup gate required")
    gate = abspath(pairs["--setup-gate"])
    isdir(gate) || error("setup gate directory missing")
    generic_args = String[]
    for i in 1:2:length(args)
        args[i] == "--setup-gate" && continue
        append!(generic_args,args[i:i+1])
    end
    o = Generic.options(generic_args)
    o.label == LABEL && o.n == 1024 && o.block == 1024 &&
        o.workload == "cfl" && o.steps == 256 && o.repeats in (2,3) ||
        error("unsupported tile-max pilot workload")
    return o,gate,generic_args
end

function verify_gate(o,gate)
    summary_path = joinpath(gate,"setup-summary.json")
    summary = Support.readjson(summary_path)
    receipt = Support.readjson(o.receipt_path)
    manifest_path = joinpath(o.suite_dir,"validation-runs.toml")
    control = joinpath(gate,"control")
    candidate = joinpath(gate,LABEL)
    driver = joinpath(gate,"checked_driver.jl")
    summary["status"] == "passed" && summary["case"] == [o.n,o.block] &&
        summary["workers"] == [1,4,8] &&
        summary["control_advection_sha256"] == BASELINE_SHA256 &&
        summary["control_tree_sha256"] == BASELINE_TREE_SHA256 &&
        summary["candidate_advection_sha256"] == receipt["candidate_source_sha256"] &&
        summary["candidate_tree_sha256"] == receipt["candidate_tree_sha256"] &&
        summary["candidate_receipt_sha256"] == Support.sha256file(o.receipt_path) &&
        summary["validation_manifest_sha256"] == Support.sha256file(manifest_path) &&
        summary["setup_rows_sha256"] ==
            Support.sha256file(joinpath(gate,"setup-rows.jsonl")) &&
        summary["commands_sha256"] ==
            Support.sha256file(joinpath(gate,"commands.jsonl")) &&
        summary["study_wrapper_sha256"] == Support.sha256file(@__FILE__) &&
        summary["generic_study_sha256"] ==
            Support.sha256file(joinpath(@__DIR__,"..","copied_source","study.jl")) &&
        summary["driver_sha256"] == Support.sha256file(driver) &&
        Support.source_hash(control) == BASELINE_TREE_SHA256 &&
        Support.source_hash(candidate) == receipt["candidate_tree_sha256"] &&
        Support.sha256file(joinpath(candidate,Generic.RELATIVE)) ==
            receipt["candidate_source_sha256"] ||
        error("tile-max setup gate identity differs")
    files = summary["locked_files_sha256"]
    all(isfile(path) && Support.sha256file(path) == hash for (path,hash) in files) ||
        error("tile-max setup gate inputs changed")
    rows = Support.readjsonl(joinpath(gate,"setup-rows.jsonl"))
    commands = Support.readjsonl(joinpath(gate,"commands.jsonl"))
    expected = [(variant,p) for p in (1,4,8) for variant in ("control",LABEL)]
    length(rows) == length(commands) == length(expected) &&
        [(row["variant"],row["threads"]) for row in rows] == expected &&
        all(command["exit_code"] == 0 for command in commands) &&
        all(rows[index]["source"] ==
            joinpath(gate,expected[index][1]) for index in eachindex(rows)) ||
        error("tile-max setup gate coverage differs")
    return Support.sha256file(summary_path),summary["driver_sha256"]
end

function verify_timed(o,gate,gate_hash,driver_hash)
    config = Support.readjson(joinpath(o.output,"configuration.json"))
    report = TOML.parsefile(joinpath(o.output,"oracle.toml"))
    rows = Support.readjsonl(joinpath(o.output,"raw.jsonl"))
    expected_count = 2*3*o.repeats
    config["workload"] == "cfl" && config["candidate_label"] == LABEL &&
        config["case"] == [o.n,o.block] && config["steps"] == o.steps &&
        Support.sha256file(joinpath(o.output,"checked_driver.jl")) == driver_hash &&
        length(rows) == expected_count &&
        all(row["cfl_bound"] == report["cfl_bound"] &&
            row["dt"] == report["dt"] && row["field_sha256"] ==
            config["oracle_sha256"] for row in rows) ||
        error("tile-max timed CFL bound or full field differs from oracle")
    verify_gate(o,gate)[1] == gate_hash ||
        error("tile-max setup gate changed during timing")
    Support.writejson(joinpath(o.output,"tile-max-verification.json"),Dict(
        "schema"=>1,"status"=>"passed","candidate_label"=>LABEL,
        "setup_gate"=>gate,"setup_gate_sha256"=>gate_hash,
        "driver_sha256"=>driver_hash,"oracle_cfl_bound"=>report["cfl_bound"],
        "oracle_field_sha256"=>config["oracle_sha256"],
        "timed_rows"=>expected_count,
        "study_wrapper_sha256"=>Support.sha256file(@__FILE__),
        "generic_study_sha256"=>
            Support.sha256file(joinpath(@__DIR__,"..","copied_source","study.jl"))))
end

function main(args=ARGS)
    o,gate,generic_args = options(args)
    gate_hash,driver_hash = verify_gate(o,gate)
    Generic.main(generic_args)
    verify_timed(o,gate,gate_hash,driver_hash)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end
