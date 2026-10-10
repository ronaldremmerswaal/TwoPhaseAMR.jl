"""Make one exact pre-timer Julia sysimage-abort retry and retain its valid batch."""
module Stage3BatchRecovery

include(joinpath(@__DIR__, "main_study.jl"))
include(joinpath(@__DIR__, "..", "Support.jl"))
using .ScalingSupport: readjson, readjsonl, writejson, appendjsonl, sha256file
using TOML

const Study = MainStudy

function main(args=ARGS)
    length(args) == 2 || error("usage: julia recover_batch.jl DIRECTORY FAILURE")
    directory, failure = abspath.(args)
    receipt_path = joinpath(directory, "recovery.json")
    !ispath(receipt_path) || error("this cohort already used its one retry")
    event = readjson(failure)
    command = event["command"]
    event["exit_code"] == -6 && occursin("jl_restore_system_image", event["stderr"]) ||
        error("failure is not the documented pre-timer sysimage abort")
    expected_prefix = String[Study.JULIA, "--startup-file=no",
        "--project=$(Study.FROZEN_PROJECT)", "--threads=8", "--gcthreads=1",
        Study.DRIVER]
    command[1:6] == expected_prefix || error("failed command differs from frozen driver")
    arguments = command[7:end]
    expected_args = String["source=$(Study.ROOT)",
        "batch=$(joinpath(directory, "batch-r0-p8-current.toml"))", "kernel=upwind",
        "cpus=0,8,1,9,2,10,3,11"]
    arguments == expected_args || error("failed batch arguments differ from retry")
    Study.source_manifest() == readjson(joinpath(directory, "source-manifest.json")) ||
        error("source changed before retry")
    raw_path = joinpath(directory, "raw.jsonl")
    old = readjsonl(raw_path)
    length(old) == 42 && !any(row->row["repeat"] == 0 && row["threads"] == 8 &&
        row["variant"] == "current", old) ||
        error("accepted prefix differs from documented seven batches")
    batch_path = joinpath(directory, "batch-r0-p8-current.toml")
    cases = TOML.parsefile(batch_path)["cases"]
    length(cases) == 6 || error("failed batch case count differs")
    cpus = [0,8,1,9,2,10,3,11]
    receipt = Dict{String,Any}(
        "original_failure"=>failure, "original_failure_sha256"=>sha256file(failure),
        "original_command"=>command, "batch_sha256"=>sha256file(batch_path),
        "recovery_script_sha256"=>sha256file(@__FILE__), "status"=>"retry_started")
    writejson(receipt_path, receipt)
    ENV["STAGE3_FAILURE_DIR"] = directory
    log = Any[]
    try
        Study.run_julia(Study.DRIVER, Study.FROZEN_PROJECT, 8, arguments, cpus, log)
        Study.source_manifest() == readjson(joinpath(directory, "source-manifest.json")) ||
            error("source changed during exact retry")
        validated = Any[]
        for case in cases
            n,b,steps = case["n"],case["block"],case["steps"]
            initial = case["input"]
            toy = joinpath(directory, "toy-oracle-$n-s$steps.bin")
            push!(validated, Study.collect_sample(directory, "current", n,b,8,steps,cpus,
                initial,toy,0,log,0.2))
        end
        for row in validated
            appendjsonl(raw_path, row)
        end
        receipt["status"] = "retry_passed"
        receipt["accepted_rows"] = length(validated)
        receipt["min_timed_seconds"] = minimum(row["time_seconds"] for row in validated)
    catch exception
        receipt["status"] = "retry_failed"
        receipt["error"] = sprint(showerror, exception)
        rethrow()
    finally
        writejson(receipt_path, receipt)
    end
    println(receipt)
end

end # module

abspath(PROGRAM_FILE) == (@__FILE__) && Stage3BatchRecovery.main()
