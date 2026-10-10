#!/usr/bin/env julia
"""Check frozen fused-study inputs without constructing a mesh or timing a step."""

include(joinpath(@__DIR__, "study.jl"))
using .RollingFluxFusedStudy

length(ARGS) == 4 || error(
    "usage: preflight.jl CANDIDATE_SOURCE SUITE_DIR PROBE_RECEIPT NEW_OUTPUT")
candidate_source, suite_dir, receipt_path, output = ARGS
Generic = RollingFluxFusedStudy.Generic
Support = RollingFluxFusedStudy.Support
options = Generic.options([
    "--candidate-source", candidate_source,
    "--candidate-label", "rolling_flux_fused",
    "--suite-dir", suite_dir,
    "--probe-receipt", receipt_path,
    "--case", "1024:8",
    "--steps", "128",
    "--repeats", "3",
    "--workload", "fixed",
    "--output", output])
Generic.Study.verify_control()
frozen = joinpath(Generic.COHORT, "current-source")
Support.sha256file(joinpath(frozen, Generic.RELATIVE)) ==
    Generic.Shared.Probe.EXPECTED_CLEAN_SHA256 || error("frozen control differs")
parent = RollingFluxFusedStudy.check_parent(options)
mkpath(options.output)
control, candidate, receipt = Generic.prepare_sources(options, frozen)
manifest = Generic.reused_suites(options, frozen, control, candidate, receipt)
Support.writejson(joinpath(options.output, "preflight.json"), Dict(
    "schema" => 1,
    "status" => "passed",
    "parent" => parent,
    "candidate_source_sha256" => receipt["candidate_source_sha256"],
    "validation_manifest_sha256" =>
        Support.sha256file(joinpath(options.output, "suite-provenance.toml")),
    "validation_status" => manifest["status"],
    "preflight_script_sha256" => Support.sha256file(@__FILE__)))
