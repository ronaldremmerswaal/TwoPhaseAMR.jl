#!/usr/bin/env julia
"""One checked, untimed setup process per source and worker count."""

include(joinpath(@__DIR__, "study.jl"))
using .CFLTileMaxStudy
using TOML

const Tile = CFLTileMaxStudy
const Generic = Tile.Generic
const Support = Tile.Support
const Study = Generic.Study

length(ARGS) == 4 || error(
    "usage: setup_gate.jl CANDIDATE_SOURCE SUITE_DIR PROBE_RECEIPT NEW_OUTPUT")
candidate_source,suite_dir,receipt_path,output = ARGS
options = Generic.options([
    "--candidate-source",candidate_source,
    "--candidate-label",Tile.LABEL,
    "--suite-dir",suite_dir,
    "--probe-receipt",receipt_path,
    "--case","1024:1024",
    "--steps","256",
    "--repeats","3",
    "--workload","cfl",
    "--output",output])

Study.verify_control()
frozen = joinpath(Generic.COHORT,"current-source")
Support.sha256file(joinpath(frozen,Generic.RELATIVE)) == Tile.BASELINE_SHA256 &&
    Support.source_hash(frozen) == Tile.BASELINE_TREE_SHA256 ||
    error("frozen production control differs")
placement_path = joinpath(Generic.COHORT,"placement.json")
placement = Support.readjson(placement_path)
mkpath(options.output)
control,candidate,receipt = Generic.prepare_sources(options,frozen)
manifest = Generic.reused_suites(options,frozen,control,candidate,receipt)
for (variant,prepared,expected) in (("control",frozen,false),
                                    (Tile.LABEL,options.candidate_source,true))
    report = TOML.parsefile(joinpath(options.output,"suite-$variant.toml"))
    report["tile_max_expected"] == expected &&
        report["tile_max_method_file"] ==
            (expected ? realpath(joinpath(prepared,Generic.RELATIVE)) : "") ||
        error("$variant tile-max method identity differs")
end
mode,kernel,make_driver = Generic.Shared.workload_spec("cfl")
mode == "measure_cfl" && kernel == "upwind_cfl" ||
    error("CFL driver workload differs")
driver = joinpath(options.output,"checked_driver.jl")
make_driver(driver)

function locked_files()
    paths = String[Study.JULIA,joinpath(Study.PROJECT,"Project.toml"),
        joinpath(Study.PROJECT,"Manifest.toml"),@__FILE__,
        joinpath(@__DIR__,"study.jl"),
        joinpath(@__DIR__,"..","copied_source","study.jl"),
        joinpath(@__DIR__,"..","scalar_selector","study.jl"),
        joinpath(@__DIR__,"..","ordered_gather","make_checked_driver.jl"),
        joinpath(@__DIR__,"..","ordered_gather","make_cfl_driver.jl"),
        joinpath(@__DIR__,"..","ordered_gather","make_checked_cfl_driver.jl"),
        joinpath(@__DIR__,"make_probe.jl"),
        placement_path,options.receipt_path,
        manifest["validator"],manifest["test"],
        joinpath(options.suite_dir,"validation-runs.toml"),
        joinpath(options.output,"probe-receipt.json"),
        joinpath(options.output,"probe.patch"),
        joinpath(options.output,"suite-provenance.toml"),
        joinpath(options.output,"suite-control.toml"),
        joinpath(options.output,"suite-$(Tile.LABEL).toml"),driver]
    if haskey(manifest,"focused_test")
        push!(paths,manifest["focused_test"])
    end
    for variant in ("control",Tile.LABEL)
        report = TOML.parsefile(joinpath(options.output,"suite-$variant.toml"))
        append!(paths,[value for (key,value) in report if
                       endswith(key,"_test_file") && value isa String])
    end
    return Dict(path=>Support.sha256file(path) for path in paths)
end

files = locked_files()
trees = Dict("control"=>Support.source_hash(control),
             Tile.LABEL=>Support.source_hash(candidate))
for p in (1,4,8), (variant,source) in (("control",control),
                                       (Tile.LABEL,candidate))
    locked_files() == files &&
        Support.source_hash(control) == trees["control"] &&
        Support.source_hash(candidate) == trees[Tile.LABEL] ||
        error("tile-max setup source or inputs changed")
    report_path = joinpath(options.output,"setup-$variant-p$p.toml")
    Generic.Shared.launch(options.output,driver,p,placement[string(p)],
        ["source=$source","mode=setup","kernel=upwind_cfl","n=1024",
         "block=1024","report=$report_path"],"setup-$variant-p$p")
    report = TOML.parsefile(report_path)
    for (key,expected) in (("source",source),("mode","setup"),
                           ("kernel","upwind_cfl"),("n",1024),
                           ("block",1024),("threads",p),("blocks",1),
                           ("faces",2*1024*1024),
                           ("julia_version","1.10.12"))
        report[key] == expected || error("setup $variant p$p $key differs")
    end
    report["variant"] = variant
    Support.appendjsonl(joinpath(options.output,"setup-rows.jsonl"),report)
end
locked_files() == files && Support.source_hash(control) == trees["control"] &&
    Support.source_hash(candidate) == trees[Tile.LABEL] ||
    error("tile-max setup source or inputs changed")
Support.writejson(joinpath(options.output,"setup-summary.json"),Dict(
    "schema"=>1,"status"=>"passed","case"=>[1024,1024],
    "workers"=>[1,4,8],"source_trees"=>trees,
    "control_advection_sha256"=>Tile.BASELINE_SHA256,
    "control_tree_sha256"=>trees["control"],
    "candidate_advection_sha256"=>receipt["candidate_source_sha256"],
    "candidate_tree_sha256"=>trees[Tile.LABEL],
    "candidate_receipt_sha256"=>Support.sha256file(options.receipt_path),
    "validation_manifest_sha256"=>
        Support.sha256file(joinpath(options.suite_dir,"validation-runs.toml")),
    "study_wrapper_sha256"=>Support.sha256file(joinpath(@__DIR__,"study.jl")),
    "generic_study_sha256"=>
        Support.sha256file(joinpath(@__DIR__,"..","copied_source","study.jl")),
    "driver_sha256"=>Support.sha256file(driver),
    "setup_rows_sha256"=>Support.sha256file(joinpath(options.output,"setup-rows.jsonl")),
    "commands_sha256"=>Support.sha256file(joinpath(options.output,"commands.jsonl")),
    "locked_files_sha256"=>files,
    "julia"=>Study.JULIA,"project"=>Study.PROJECT,"depot"=>Study.DEPOT,
    "placement"=>placement))
