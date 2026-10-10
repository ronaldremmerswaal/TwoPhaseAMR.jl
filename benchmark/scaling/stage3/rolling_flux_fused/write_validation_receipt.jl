#!/usr/bin/env julia
"""Pin completed row-fused correctness for an independent timing run."""

include(joinpath(@__DIR__, "..", "..", "Support.jl"))
using .ScalingSupport: sha256file, source_hash
using TOML

length(ARGS) == 5 || error(
    "usage: write_validation_receipt.jl CANDIDATE CONTROL PROJECT JULIA_BINARY VALIDATION_DEPOT")
candidate, control, project, julia_binary = abspath.(ARGS[1:4])
validation_depot = ARGS[5]
test_file = abspath(joinpath(@__DIR__, "..", "..", "..", "..", "test",
                            "operators", "advection", "cell_advection_compact.jl"))
focused_file = abspath(joinpath(@__DIR__, "..", "rolling_flux",
                               "focused_validation.jl"))
validator = abspath(joinpath(@__DIR__, "validation_driver.jl"))
relative_advection = joinpath("src", "operators", "advection", "cell_advection.jl")
runs = Dict{String,Any}[]
for (variant, source, report_name, fused) in (
        ("control", control, "suite-control-p1.toml", false),
        ("rolling_flux_fused", candidate, "suite-fused-p1.toml", true))
    report_file = joinpath(candidate, report_name)
    report = TOML.parsefile(report_file)
    source_sha = sha256file(joinpath(source, relative_advection))
    report["status"] == "passed" &&
        report["loaded_advection_sha256"] == source_sha &&
        report["original_test_sha256"] == sha256file(test_file) &&
        report["focused_test_sha256"] == sha256file(focused_file) &&
        report["fused_expected"] == fused &&
        report["threads"] == 1 ||
        error("suite report identity differs for $variant")
    push!(runs, Dict{String,Any}(
        "variant" => variant,
        "source" => source,
        "source_advection_sha256" => source_sha,
        "source_tree_sha256" => source_hash(source),
        "expect_fused" => fused,
        "report" => report_name,
        "report_sha256" => sha256file(report_file),
        "exit_code" => 0,
    ))
end

threaded = Dict{String,Any}[]
for workers in (3, 4, 7, 8)
    name = "suite-fused-p$(workers).toml"
    path = joinpath(candidate, name)
    report = TOML.parsefile(path)
    report["status"] == "passed" && report["fused_expected"] === true &&
        report["threads"] == workers &&
        report["loaded_advection_sha256"] == runs[2]["source_advection_sha256"] &&
        report["focused_test_sha256"] == sha256file(focused_file) ||
        error("threaded suite report differs for $workers workers")
    push!(threaded, Dict{String,Any}(
        "threads" => workers, "report" => name,
        "report_sha256" => sha256file(path), "exit_code" => 0,
    ))
end

receipt = Dict{String,Any}(
    "schema" => 1,
    "validator" => validator,
    "validator_sha256" => sha256file(validator),
    "test" => test_file,
    "test_sha256" => sha256file(test_file),
    "focused_test" => focused_file,
    "focused_test_sha256" => sha256file(focused_file),
    "julia" => julia_binary,
    "julia_sha256" => sha256file(julia_binary),
    "project" => project,
    "project_toml_sha256" => sha256file(joinpath(project, "Project.toml")),
    "project_manifest_sha256" => sha256file(joinpath(project, "Manifest.toml")),
    "depot" => validation_depot,
    "threads" => 1,
    "status" => "both passed",
    "assertions_per_run" => sum((223, 10819, 11, 516, 128, 603, 19313)),
    "runs" => runs,
    "threaded_reports" => threaded,
    "generator_sha256" => sha256file(joinpath(@__DIR__, "make_probe.jl")),
    "candidate_receipt_sha256" => sha256file(joinpath(candidate, "rolling-flux-fused-receipt.json")),
    "candidate_patch_sha256" => sha256file(joinpath(candidate, "rolling-flux-fused.patch")),
)
path = joinpath(candidate, "validation-runs.toml")
open(path, "w") do io
    TOML.print(io, receipt)
end
println(path)
