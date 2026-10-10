"""Isolate one frozen mesh setup crash, then retry its exact failed batch once."""
module Stage3MeshRecovery

include(joinpath(@__DIR__, "main_study.jl"))
include(joinpath(@__DIR__, "..", "Support.jl"))
using .ScalingSupport: readjson, readjsonl, writejson, sha256file
using JSON
using TOML

const Study = MainStudy

function main(args=ARGS)
    length(args) == 2 || error("usage: julia recover_mesh_setup.jl DIRECTORY FAILURE")
    directory, failure = abspath.(args)
    receipt_path = joinpath(directory, "mesh-setup-recovery.json")
    !ispath(receipt_path) || error("this frozen mesh-setup failure was already investigated")
    manifest = readjson(joinpath(directory, "source-manifest.json"))
    Study.source_manifest() == manifest || error("source changed before isolation")
    event = readjson(failure)
    command = event["command"]
    expected_prefix = String[Study.JULIA, "--startup-file=no",
        "--project=$(Study.FROZEN_PROJECT)", "--threads=8", "--gcthreads=1",
        Study.DRIVER]
    command[1:6] == expected_prefix && event["exit_code"] == -11 &&
        occursin("source-snapshot/src/mesh/topology/mesh.jl:809", event["stderr"]) ||
        error("not the documented frozen p8 mesh-packing SIGSEGV")
    batch_path = joinpath(directory, "batch-r0-p8-frozen.toml")
    expected_args = String["source=$(Study.FROZEN_SOURCE)", "batch=$batch_path",
                           "kernel=upwind", "cpus=0,8,1,9,2,10,3,11"]
    command[7:end] == expected_args || error("failed command differs from exact batch retry")
    cases = TOML.parsefile(batch_path)["cases"]
    [(case["n"],case["block"],case["steps"]) for case in cases] ==
        [(512,8,1024), (2048,2048,16), (1024,8,128), (1024,1024,128),
         (512,512,1024), (2048,8,16)] ||
        error("failed batch order or trajectory changed")
    raw_path = joinpath(directory, "raw.jsonl")
    original_raw = read(raw_path, String)
    prior = readjsonl(raw_path)
    length(prior) == 36 && !any(row->row["repeat"] == 0 && row["threads"] == 8 &&
        row["variant"] == "frozen", prior) ||
        error("accepted prefix differs from six complete batches")
    attempts = joinpath(directory, "mesh-setup-attempts")
    !ispath(attempts) || error("mesh setup attempts directory already exists")
    mkpath(attempts)
    for case in cases[1:end-1], key in ("report", "output")
        source = case[key]
        isfile(source) || error("partial failed-batch artifact missing: $source")
        cp(source, joinpath(attempts, basename(source)); force=true)
    end
    !isfile(cases[end]["report"]) || error("last case unexpectedly has a report")
    cpus = [0,8,1,9,2,10,3,11]
    isolated = cases[end]
    isolation_field = joinpath(attempts, "isolated-2048-b8-p8.bin")
    isolation_report = joinpath(attempts, "isolated-2048-b8-p8.toml")
    receipt = Dict{String,Any}(
        "status"=>"isolation_started", "original_failure"=>failure,
        "original_failure_sha256"=>sha256file(failure),
        "original_command"=>command, "failed_batch_sha256"=>sha256file(batch_path),
        "source_manifest_sha256"=>sha256file(joinpath(directory, "source-manifest.json")),
        "recovery_script_sha256"=>sha256file(@__FILE__),
        "partial_first_attempt"=>readdir(attempts))
    writejson(receipt_path, receipt)
    ENV["STAGE3_FAILURE_DIR"] = directory
    log = Any[]
    try
        Study.run_julia(Study.DRIVER, Study.FROZEN_PROJECT, 8,
            String["source=$(Study.FROZEN_SOURCE)", "mode=measure", "kernel=upwind",
                   "n=2048", "block=8", "steps=16", "cpus=0,8,1,9,2,10,3,11",
                   "input=$(isolated["input"])", "output=$isolation_field",
                   "report=$isolation_report"], cpus, log)
        Study.source_manifest() == manifest || error("source changed during isolated process")
        isolated_oracle = joinpath(directory, "upwind-oracle-2048-b8-s16.bin")
        comparison = Study.full_field(isolation_field, isolated_oracle, 2048)
        report = TOML.parsefile(isolation_report)
        comparison["tolerance_violations"] == 0 &&
            comparison["max_abs_vs_oracle"] == 0 &&
            report["time_seconds"] >= 0.2 && report["threads"] == 8 &&
            report["steps"] == 16 && report["block"] == 8 ||
            error("isolated process failed numerical, duration, or identity gate")
        merge!(receipt, Dict(
            "status"=>"isolation_passed", "isolation_comparison"=>comparison,
            "isolation_timed_seconds"=>report["time_seconds"],
            "isolation_field_sha256"=>sha256file(isolation_field)))
        writejson(receipt_path, receipt)
        Study.run_julia(Study.DRIVER, Study.FROZEN_PROJECT, 8, expected_args, cpus, log)
        log[end]["command"] == command || error("retry launch was not byte-for-byte identical")
        Study.source_manifest() == manifest &&
            sha256file(batch_path) == receipt["failed_batch_sha256"] ||
            error("source or batch changed during exact retry")
        validated = Any[]
        for case in cases
            n,b,steps = case["n"],case["block"],case["steps"]
            row = Study.collect_sample(directory, "frozen", n,b,8,steps,cpus,
                case["input"], joinpath(directory, "toy-oracle-$n-s$steps.bin"),
                0,log,0.2)
            row["max_abs_vs_oracle"] == 0 ||
                error("retry differs bitwise from frozen oracle")
            push!(validated, row)
        end
        temporary = joinpath(directory, "raw-recovered.tmp")
        open(temporary, "w") do io
            write(io, original_raw)
            for row in validated
                println(io, JSON.json(row))
            end
        end
        mv(temporary, raw_path; force=true)
        merge!(receipt, Dict(
            "status"=>"retry_passed", "accepted_rows"=>length(validated),
            "min_timed_seconds"=>minimum(row["time_seconds"] for row in validated)))
    catch exception
        receipt["status"] = "stopped"
        receipt["error"] = sprint(showerror, exception)
        rethrow()
    finally
        writejson(receipt_path, receipt)
    end
    println(receipt)
end

end # module

abspath(PROGRAM_FILE) == (@__FILE__) && Stage3MeshRecovery.main()
