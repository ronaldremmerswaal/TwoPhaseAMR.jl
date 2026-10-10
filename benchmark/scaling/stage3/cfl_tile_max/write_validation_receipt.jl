#!/usr/bin/env julia
"""Pin passed copied-source CFL tile-max suites for a later timing study."""

include(joinpath(@__DIR__, "..", "..", "Support.jl"))
using .ScalingSupport: sha256file, source_hash
using TOML

length(ARGS) == 5 || error(
    "usage: write_validation_receipt.jl CANDIDATE CONTROL PROJECT JULIA_BINARY DEPOT")
candidate, control, project, julia_binary = abspath.(ARGS[1:4])
depot = ARGS[5]
test_file = abspath(joinpath(@__DIR__, "..", "..", "..", "..", "test",
                            "operators", "advection", "cell_advection_compact.jl"))
focused_file = abspath(joinpath(@__DIR__, "focused_validation.jl"))
validator = abspath(joinpath(@__DIR__, "validation_driver.jl"))
relative_advection = joinpath("src", "operators", "advection", "cell_advection.jl")
candidate_receipt = joinpath(candidate, "cfl-tile-max-receipt.json")
candidate_patch = joinpath(candidate, "cfl-tile-max.patch")
runs = Dict{String,Any}[]
focused_counts = Int[]
for (variant, source, expected) in
        (("control", control, false), ("cfl_tile_max", candidate, true))
    name = "suite-$(variant)-p1.toml"
    path = joinpath(candidate, name)
    report = TOML.parsefile(path)
    source_sha = sha256file(joinpath(source, relative_advection))
    report["status"] == "passed" &&
        report["loaded_advection_sha256"] == source_sha &&
        report["original_test_sha256"] == sha256file(test_file) &&
        report["focused_test_sha256"] == sha256file(focused_file) &&
        report["tile_max_expected"] === expected &&
        report["threads"] == 1 &&
        report["julia_version"] == "1.10.12" ||
        error("p1 suite identity differs for $variant")
    push!(focused_counts, report["focused_assertions"])
    push!(runs, Dict{String,Any}(
        "variant" => variant,
        "source" => source,
        "source_advection_sha256" => source_sha,
        "source_tree_sha256" => source_hash(source),
        "expect_tile_max" => expected,
        "report" => name,
        "report_sha256" => sha256file(path),
        "exit_code" => 0,
    ))
end

threaded = Dict{String,Any}[]
for workers in (3, 4, 7, 8)
    name = "suite-cfl_tile_max-p$(workers).toml"
    path = joinpath(candidate, name)
    report = TOML.parsefile(path)
    report["status"] == "passed" && report["tile_max_expected"] === true &&
        report["threads"] == workers &&
        report["loaded_advection_sha256"] == runs[2]["source_advection_sha256"] &&
        report["focused_test_sha256"] == sha256file(focused_file) &&
        report["julia_version"] == "1.10.12" ||
        error("threaded suite identity differs for $workers workers")
    push!(focused_counts, report["focused_assertions"])
    push!(threaded, Dict{String,Any}(
        "threads" => workers,
        "report" => name,
        "report_sha256" => sha256file(path),
        "exit_code" => 0,
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
    "depot" => depot,
    "threads" => 1,
    "status" => "both passed",
    "assertions_per_run" => minimum(focused_counts),
    "assertion_scope" => "focused assertions only; compact suite adds further checks",
    "runs" => runs,
    "threaded_reports" => threaded,
    "generator_sha256" => sha256file(joinpath(@__DIR__, "make_probe.jl")),
    "candidate_receipt_sha256" => sha256file(candidate_receipt),
    "candidate_patch_sha256" => sha256file(candidate_patch),
)
path = joinpath(candidate, "validation-runs.toml")
open(path, "w") do io
    TOML.print(io, receipt)
end
println(path)
