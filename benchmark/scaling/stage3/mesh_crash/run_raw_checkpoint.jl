#!/usr/bin/env julia
"""One capped fresh-2048 RawSide checkpoint on the frozen Julia 1.10 runtime."""
module MeshCrashRawCheckpointStudy

include(joinpath(@__DIR__, "run.jl"))
include(joinpath(@__DIR__, "make_raw_checkpoint.jl"))
using .MeshCrashStudy
using .MeshCrashRawCheckpoint
using TOML

const Study = MeshCrashStudy
const Probe = MeshCrashRawCheckpoint
const Support = Study.ScalingSupport
const METHOD_ANCHOR = """
realpath(String(method.file)) == realpath(MESH_FILE) ||
    error("loaded dual-volume method is outside the copied source")
"""
const METHOD_PATCH = """
realpath(String(method.file)) == realpath(MESH_FILE) ||
    error("loaded dual-volume method is outside the copied source")
face_schedule_method = only(collect(methods(TwoPhaseAMR._build_face_schedules)))
realpath(String(face_schedule_method.file)) == realpath(MESH_FILE) ||
    error("loaded face-schedule method is outside copied source")
"""
const EVENT_ANCHOR =
    "        \"dual_volume_method_file\" => realpath(String(method.file)),"
const EVENT_PATCH = EVENT_ANCHOR * "\n" *
    "        \"face_schedule_method_file\" => realpath(String(face_schedule_method.file)),"

function main(args=ARGS)
    options = Study.parse_args(args)
    options === nothing && return
    timeout_binary = Sys.which("timeout")
    timeout_binary === nothing && error("system timeout utility is required")
    files = [@__FILE__,joinpath(@__DIR__,"run.jl"),
        joinpath(@__DIR__,"make_raw_checkpoint.jl"),Study.DRIVER,
        Study.JULIA,timeout_binary,
        joinpath(Study.PROJECT,"Project.toml"),
        joinpath(Study.PROJECT,"Manifest.toml"),
        joinpath(options.source,Study.RELATIVE_ADVECTION),
        joinpath(options.source,Probe.RELATIVE)]
    all(isfile,files) || error("checkpoint input missing")
    Support.sha256file(files[end-1]) == Study.EXPECTED_ADVECTION_SHA256 &&
        Support.sha256file(files[end]) == Probe.CLEAN_SHA256 &&
        Support.sha256file(Study.JULIA) == Study.EXPECTED_JULIA_SHA256 &&
        Support.sha256file(joinpath(Study.PROJECT,"Project.toml")) ==
            Study.EXPECTED_PROJECT_SHA256 &&
        Support.sha256file(joinpath(Study.PROJECT,"Manifest.toml")) ==
            Study.EXPECTED_MANIFEST_SHA256 ||
        error("frozen source or numerical environment changed")
    clean_tree = Support.source_hash(options.source)
    mkpath(options.output)
    copied = joinpath(options.output,"copied-source")
    probe = Probe.make_probe(options.source,copied)
    driver_original = read(Study.DRIVER,String)
    length(split(driver_original,METHOD_ANCHOR)) == 2 &&
        length(split(driver_original,EVENT_ANCHOR)) == 2 ||
        error("driver method anchor missing or repeated")
    driver_copy = replace(driver_original,METHOD_ANCHOR=>METHOD_PATCH;count=1)
    driver_copy = replace(driver_copy,EVENT_ANCHOR=>EVENT_PATCH;count=1)
    driver = joinpath(options.output,"mesh_driver.jl")
    write(driver,driver_copy)
    diff = Support.run_capture(["diff","-u","--label","mesh_driver.jl",
        "--label","mesh_driver.jl.raw-checkpoint",Study.DRIVER,driver])
    diff.exitcode == 1 || error("expected copied driver patch")
    driver_patch = joinpath(options.output,"raw-checkpoint-driver.patch")
    write(driver_patch,diff.stdout)
    append!(files,[driver,driver_patch,joinpath(copied,Probe.RELATIVE),
        joinpath(copied,"raw-checkpoint.patch"),
        joinpath(copied,"raw-checkpoint-receipt.json")])
    hashes = Dict(file=>Support.sha256file(file) for file in files)
    copied_tree = Support.source_hash(copied)
    clean_tree == probe["clean_tree_sha256"] &&
        copied_tree == probe["checkpoint_tree_sha256"] ||
        error("source copy differs from checkpoint receipt")
    Support.writejson(joinpath(options.output,"configuration.json"),Dict(
        "schema"=>1,"source"=>options.source,"cpu"=>0,
        "child_limit_seconds"=>180,
        "julia_version_expected"=>"1.10.12",
        "clean_source_tree_sha256"=>clean_tree,
        "copied_source_tree_sha256"=>copied_tree,
        "probe_receipt"=>probe,"locked_files_sha256"=>hashes))
    all(Support.sha256file(file)==hash for (file,hash) in hashes) &&
        Support.source_hash(options.source)==clean_tree &&
        Support.source_hash(copied)==copied_tree ||
        error("checkpoint inputs changed before launch")
    events = joinpath(options.output,"events.toml")
    checkpoint = joinpath(options.output,"raw-checkpoint.toml")
    command = [timeout_binary,"--signal=INT","--kill-after=5s","180s",
        Study.JULIA,"--startup-file=no","--project=$(Study.PROJECT)",
        "--threads=1","--gcthreads=1",driver,copied,events,"fresh2048",
        probe["checkpoint_mesh_sha256"]]
    environment = Dict("JULIA_DEPOT_PATH"=>Study.DEPOT,
        "OPENBLAS_NUM_THREADS"=>"1","MKL_NUM_THREADS"=>"1",
        "JULIA_NUM_GC_THREADS"=>"1",
        "TWOPHASE_RAW_CHECKPOINT_OUTPUT"=>checkpoint)
    result = Support.run_capture(command;env=environment,cpus=[0])
    all(Support.sha256file(file)==hash for (file,hash) in hashes) &&
        Support.source_hash(options.source)==clean_tree &&
        Support.source_hash(copied)==copied_tree ||
        error("checkpoint inputs changed during launch")
    phases = isfile(events) ?
        [entry["phase"] for entry in TOML.parsefile(events)["events"]] : String[]
    summary = isfile(checkpoint) ? TOML.parsefile(checkpoint) : nothing
    Support.writejson(joinpath(options.output,"receipt.json"),Dict(
        "schema"=>1,"command"=>command,"cpu"=>0,
        "exit_code"=>result.exitcode,"termsignal"=>result.termsignal,
        "seconds"=>result.seconds,"stdout"=>result.stdout,
        "stderr"=>result.stderr,"event_phases"=>phases,
        "checkpoint_file"=>checkpoint,"checkpoint"=>summary,
        "intentional_stop"=>summary !== nothing &&
            occursin("RawSide checkpoint stopped",result.stderr),
        "time_limit_reached"=>result.exitcode==124))
    println("RawSide checkpoint exit=$(result.exitcode) report=$(summary !== nothing)")
    summary === nothing || println("invalid=$(summary["invalid_count"]) populations=$(summary["counts"])")
end

if abspath(PROGRAM_FILE)==@__FILE__
    main()
end

end # module
