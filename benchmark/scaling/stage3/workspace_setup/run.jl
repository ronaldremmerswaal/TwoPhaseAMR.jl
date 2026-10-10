#!/usr/bin/env julia
"""Run exactly one setup-only child for each frozen copied source."""

include(joinpath(@__DIR__, "..", "..", "Support.jl"))
using .ScalingSupport: readjson, writejson, sha256file, source_hash, run_capture
using TOML

length(ARGS) == 6 || error(
    "usage: run.jl CONTROL_PROBE FUSED_PROBE JULIA_BINARY PROJECT DEPOT NEW_OUTPUT")
control, fused, julia_binary, project, depot, output = ARGS
control, fused, julia_binary, project, output =
    abspath.((control, fused, julia_binary, project, output))
!ispath(output) || error("output already exists: $output")
driver = abspath(joinpath(@__DIR__, "validate_setup.jl"))
cpus = [0, 8, 1, 9, 2, 10, 3, 11]
records = Dict{String,Any}[]
mkpath(output)
for (variant, source) in (("control", control), ("rolling_flux_fused", fused))
    receipt_file = joinpath(source, "setup-check-receipt.json")
    receipt = readjson(receipt_file)
    receipt["variant"] == variant && receipt["output"] == source &&
        receipt["instrumented_tree_sha256"] == source_hash(source) &&
        receipt["instrumented_threading_sha256"] ==
            sha256file(joinpath(source, "src", "threading.jl")) &&
        receipt["instrumented_advection_sha256"] == sha256file(joinpath(source,
            "src", "operators", "advection", "cell_advection.jl")) ||
        error("instrumented source identity differs for $variant")
    report = joinpath(output, "$(variant)-setup.toml")
    command = [julia_binary, "--startup-file=no", "--project=$project",
               "--threads=8", "--gcthreads=1", driver, source,
               receipt["instrumented_threading_sha256"],
               receipt["instrumented_advection_sha256"], variant, report]
    result = run_capture(command; cpus, env=Dict(
        "JULIA_DEPOT_PATH" => depot,
        "OPENBLAS_NUM_THREADS" => "1",
        "MKL_NUM_THREADS" => "1",
        "JULIA_NUM_GC_THREADS" => "1"))
    stdout_file = joinpath(output, "$(variant)-stdout.txt")
    stderr_file = joinpath(output, "$(variant)-stderr.txt")
    write(stdout_file, result.stdout)
    write(stderr_file, result.stderr)
    passed_report = if isfile(report)
        try
            TOML.parsefile(report)["status"] == "passed"
        catch
            false
        end
    else
        false
    end
    push!(records, Dict{String,Any}(
        "variant" => variant,
        "source" => source,
        "source_receipt_sha256" => sha256file(receipt_file),
        "instrumented_tree_sha256" => source_hash(source),
        "command" => command,
        "exit_code" => result.exitcode,
        "signal" => result.termsignal,
        "stdout" => basename(stdout_file),
        "stdout_sha256" => sha256file(stdout_file),
        "stderr" => basename(stderr_file),
        "stderr_sha256" => sha256file(stderr_file),
        "report" => isfile(report) ? basename(report) : "",
        "report_sha256" => isfile(report) ? sha256file(report) : "",
        "passed" => result.exitcode == 0 && passed_report,
    ))
end
receipt = Dict{String,Any}(
    "schema" => 1,
    "status" => all(record["passed"] for record in records) ? "both_passed" :
        "at_least_one_failed",
    "scope" => "one setup-only 1024²/8² eight-worker child per instrumented source",
    "cpus" => cpus,
    "driver_sha256" => sha256file(driver),
    "runner_sha256" => sha256file(@__FILE__),
    "julia" => julia_binary,
    "julia_sha256" => sha256file(julia_binary),
    "project" => project,
    "project_toml_sha256" => sha256file(joinpath(project, "Project.toml")),
    "project_manifest_sha256" => sha256file(joinpath(project, "Manifest.toml")),
    "depot" => depot,
    "runs" => records,
)
writejson(joinpath(output, "setup-validation-receipt.json"), receipt)
println("setup-only result: ", receipt["status"])
