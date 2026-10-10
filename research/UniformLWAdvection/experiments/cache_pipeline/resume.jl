"""Resume one interrupted Julia pilot cohort after exactly one failed-job retry."""
module CachePipelineResume

include(joinpath(@__DIR__, "pilot.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "benchmark", "scaling", "Support.jl"))
using .ScalingSupport
using Dates

const Pilot = CachePipelinePilot

function accepted_row!(record, job, configuration, manifest, cpus, index)
    repeat, n, count, mode, layout = job
    (record["nx"], record["ny"], record["workers"], record["mode"],
     [record["tile_x"], record["tile_y"]], record["steps"]) ==
        (n, n, count, mode, layout, configuration["steps"][string(n)]) ||
        error("resumed job report differs from frozen order")
    record["field_mismatches"] == 0 &&
        record["field_sha256"] == record["oracle_sha256"] ||
        error("resumed full field differs from serial oracle")
    record["q_alias_preserved"] &&
        all(==(count), record["active_worker_counts"]) ||
        error("resumed alias or worker-participation check failed")
    [item["observed_cpu"] for item in record["affinity"]] == cpus ||
        error("resumed worker pinning differs")
    record["time_seconds"] >= configuration["minimum_sample_seconds"] ||
        error("resumed sample shorter than frozen minimum")
    record["source_sha256"]["pipeline"] == manifest[Pilot.PIPELINE] ||
        error("resumed kernel source differs")
    isfinite(record["ns_per_unknown_step"]) || error("resumed normalized time invalid")
    merge!(record, Dict("repeat"=>repeat, "sample_index"=>index,
                        "placement"=>configuration["placement"],
                        "requested_cpus"=>cpus, "source_manifest"=>manifest))
    return record
end

function options(args)
    parsed = Pilot.cli(args)
    Set(keys(parsed)) == Set(("cohort", "failed-receipt")) ||
        error("usage: julia resume.jl --cohort DIR --failed-receipt FILE")
    return abspath(Pilot.single(parsed, "cohort")),
           abspath(Pilot.single(parsed, "failed-receipt"))
end

function main(args=ARGS)
    cohort, failed_receipt = options(args)
    configuration = readjson(joinpath(cohort, "configuration.json"))
    jobs = readjson(joinpath(cohort, "job_order.json"))
    raw_path = joinpath(cohort, "raw.jsonl")
    accepted = readjsonl(raw_path)
    next_index = length(accepted)
    0 < next_index < length(jobs) || error("no interrupted, resumable cohort prefix")
    [row["sample_index"] for row in accepted] == collect(0:next_index-1) ||
        error("accepted sample indices are not a contiguous prefix")
    for (i, row) in enumerate(accepted)
        [row["repeat"], row["nx"], row["workers"], row["mode"],
         [row["tile_x"], row["tile_y"]]] == jobs[i] ||
            error("accepted prefix differs from frozen job order")
    end
    before = configuration["sources"]
    runner = (; julia=configuration["julia"], project=configuration["project"],
              depot=configuration["depot"], output=cohort)
    Pilot.sources(runner) == before || error("frozen source manifest changed before resume")
    failure = readjson(failed_receipt)
    job = jobs[next_index+1]
    repeat, n, count, mode, layout = job
    (failure["n"], failure["workers"], failure["mode"]) == (n, count, mode) ||
        error("failed receipt does not match next frozen job")
    failure["returncode"] == -11 &&
        (get(failure, "termsignal", 0) == 11 ||
         occursin("signal", lowercase(failure["stderr"]))) ||
        error("this recovery path only covers the recorded signal-11 launch")
    pwd() == failure["cwd"] || error("resume must use the original working directory")
    cpus = configuration["cpu_orders"][string(count)]
    fields = joinpath(cohort, "fields")
    expected_command = String[runner.julia, "--startup-file=no",
        "--project=$(runner.project)", "--threads=$count", "--gcthreads=1",
        Pilot.DRIVER, "mode=$mode", "nx=$n", "ny=$n",
        "output=$(joinpath(fields, "sample.bin"))",
        "steps=$(configuration["steps"][string(n)])",
        "input=$(joinpath(fields, "q0-$n.bin"))",
        "report=$(joinpath(fields, "sample.toml"))", "cpus=$(join(cpus, ','))",
        "oracle=$(joinpath(fields, "oracle-$n.bin"))",
        "tile-x=$(layout[1])", "tile-y=$(layout[2])"]
    expected_command == failure["command"] && cpus == failure["cpus"] ||
        error("retry command differs from failed argv")
    commands = readjsonl(joinpath(cohort, "commands.jsonl"))
    length(commands) == next_index + 2length(configuration["sizes"]) + 1 &&
        commands[end] == failure || error("failed launch is not the final recorded command")
    receipt_path = joinpath(cohort, "recovery-receipt.json")
    !ispath(receipt_path) || error("recovery already attempted")
    receipt = Dict{String,Any}(
        "schema"=>1, "status"=>"retrying", "cohort"=>cohort,
        "resume_script"=>abspath(@__FILE__), "resume_script_sha256"=>sha256file(@__FILE__),
        "argv"=>vcat([PROGRAM_FILE], args),
        "configuration_sha256"=>sha256file(joinpath(cohort, "configuration.json")),
        "job_order_sha256"=>sha256file(joinpath(cohort, "job_order.json")),
        "original_failure_file"=>failed_receipt,
        "original_failure_sha256"=>sha256file(failed_receipt),
        "original_failure_pre_timer"=>true,
        "original_failure_evidence"=>"Julia signal 11 in compiler/GC before timing report",
        "accepted_prefix_count"=>next_index,
        "accepted_prefix_indices"=>collect(0:next_index-1),
        "retried_job_index"=>next_index, "retried_job"=>job,
        "started_utc"=>Pilot.utcstamp())
    writejson(receipt_path, receipt)
    try
        for i in next_index+1:length(jobs)
            Pilot.sources(runner) == before || error("frozen source manifest changed during resume")
            repeat, n, count, mode, layout = jobs[i]
            cpus = configuration["cpu_orders"][string(count)]
            output, report = joinpath(fields, "sample.bin"), joinpath(fields, "sample.toml")
            record = Pilot.launch(runner, mode, n, configuration["steps"][string(n)],
                cpus, joinpath(fields, "q0-$n.bin"), output, report;
                oracle=joinpath(fields, "oracle-$n.bin"), tile_size=layout)
            Pilot.sources(runner) == before || error("frozen source manifest changed after resumed sample")
            accepted_row!(record, jobs[i], configuration, before, cpus, i-1)
            appendjsonl(raw_path, record)
            rm(output)
            rm(report)
            receipt["last_accepted_index"] = i-1
            writejson(receipt_path, receipt)
            println("resumed $i/$(length(jobs)) $mode $(n)² p$count " *
                    "$(round(record["ns_per_unknown_step"]; digits=3)) ns")
        end
    catch exception
        receipt["status"] = "stopped"
        receipt["error"] = sprint(showerror, exception)
        receipt["stopped_utc"] = Pilot.utcstamp()
        writejson(receipt_path, receipt)
        rethrow()
    end
    receipt["status"] = "complete"
    receipt["completed_utc"] = Pilot.utcstamp()
    receipt["accepted_total"] = length(jobs)
    writejson(receipt_path, receipt)
    Pilot.summarize_raw(cohort)
end

end # module

abspath(PROGRAM_FILE) == (@__FILE__) && CachePipelineResume.main()
