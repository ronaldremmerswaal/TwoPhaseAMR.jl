#!/usr/bin/env julia
"""Three bounded fresh-2048 controls with the archived unmodified driver."""
module MeshCrashCleanControls

include(joinpath(@__DIR__,"run.jl"))
using .MeshCrashStudy
using TOML

const Study = MeshCrashStudy
const Support = Study.ScalingSupport
const DRIVER = joinpath(@__DIR__,"mesh_driver_clean_v1.jl")
const DRIVER_SHA256 =
    "75ae5c1c3f7917caa06d9eb62a6e9a1a237f300af173f113550d7db4d63ff05b"
const JULIA112 =
    "/home/ronald/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia"
const JULIA112_SHA256 =
    "582718c20c563d824b88a08e2a9d0afcf5015b3053f425697cecc79ec4e9ea36"
const SOURCE_TREE_SHA256 =
    "0bb8826fb52d143ae0fa5f84e043f4ed4d4feeadb4c7328d6fbda33cbf7095a2"

function main(args=ARGS)
    options=Study.parse_args(args)
    options===nothing && return
    timeout_binary=Sys.which("timeout")
    timeout_binary===nothing && error("system timeout utility is required")
    files=[@__FILE__,joinpath(@__DIR__,"run.jl"),DRIVER,
        Study.JULIA,JULIA112,timeout_binary,
        joinpath(Study.PROJECT,"Project.toml"),
        joinpath(Study.PROJECT,"Manifest.toml"),
        joinpath(options.source,Study.RELATIVE_ADVECTION),
        joinpath(options.source,"src","mesh","topology","mesh.jl")]
    all(isfile,files) || error("a clean-control input is missing")
    Support.sha256file(DRIVER)==DRIVER_SHA256 &&
        Support.sha256file(Study.JULIA)==Study.EXPECTED_JULIA_SHA256 &&
        Support.sha256file(JULIA112)==JULIA112_SHA256 &&
        Support.sha256file(files[end-1])==Study.EXPECTED_ADVECTION_SHA256 &&
        Support.sha256file(files[end])==Study.EXPECTED_MESH_SHA256 &&
        Support.sha256file(joinpath(Study.PROJECT,"Project.toml"))==
            Study.EXPECTED_PROJECT_SHA256 &&
        Support.sha256file(joinpath(Study.PROJECT,"Manifest.toml"))==
            Study.EXPECTED_MANIFEST_SHA256 ||
        error("clean-control source/runtime/project differs from frozen evidence")
    tree=Support.source_hash(options.source)
    tree==SOURCE_TREE_SHA256 || error("frozen source tree hash differs")
    mkpath(options.output)
    hashes=Dict(file=>Support.sha256file(file) for file in files)
    labels=("julia110_default","julia110_noinline","julia112_default")
    Support.writejson(joinpath(options.output,"configuration.json"),Dict(
        "schema"=>1,"source"=>options.source,
        "source_tree_sha256"=>tree,"driver_sha256"=>DRIVER_SHA256,
        "cpu"=>0,"child_limit_seconds"=>180,
        "cases"=>collect(labels),"locked_files_sha256"=>hashes,
        "interpretation_limit"=>"after_mesh reports face count only; no face or dual-value validation"))
    environment=Dict("JULIA_DEPOT_PATH"=>Study.DEPOT,
        "OPENBLAS_NUM_THREADS"=>"1","MKL_NUM_THREADS"=>"1",
        "JULIA_NUM_GC_THREADS"=>"1")
    for label in labels
        all(Support.sha256file(file)==hash for (file,hash) in hashes) &&
            Support.source_hash(options.source)==tree ||
            error("clean-control inputs changed before launch")
        executable=label=="julia112_default" ? JULIA112 : Study.JULIA
        events=joinpath(options.output,"events-$label.toml")
        command=[timeout_binary,"--signal=INT","--kill-after=5s","180s",
            executable,"--startup-file=no","--project=$(Study.PROJECT)",
            "--threads=1","--gcthreads=1"]
        label=="julia110_noinline" && push!(command,"--inline=no")
        append!(command,[DRIVER,options.source,events,"fresh2048"])
        result=Support.run_capture(command;env=environment,cpus=[0])
        all(Support.sha256file(file)==hash for (file,hash) in hashes) &&
            Support.source_hash(options.source)==tree ||
            error("clean-control inputs changed during launch")
        entries=isfile(events) ? TOML.parsefile(events)["events"] : Any[]
        phases=[entry["phase"] for entry in entries]
        completed_counts=[entry["face_count"] for entry in entries
            if entry["phase"]=="after_mesh"]
        face_count=isempty(completed_counts) ? -1 : last(completed_counts)
        receipt=Dict(
            "schema"=>1,"case"=>label,"command"=>command,"cpu"=>0,
            "driver_sha256"=>DRIVER_SHA256,
            "julia_sha256"=>Support.sha256file(executable),
            "source_tree_sha256"=>tree,
            "exit_code"=>result.exitcode,"termsignal"=>result.termsignal,
            "seconds"=>result.seconds,"stdout"=>result.stdout,
            "stderr"=>result.stderr,"events_file"=>events,
            "event_phases"=>phases,"completed_face_count"=>face_count,
            "time_limit_reached"=>result.exitcode==124,
            "integrity_validation_performed"=>false)
        Support.writejson(joinpath(options.output,"receipt-$label.json"),receipt)
        println("$label exit=$(result.exitcode) signal=$(result.termsignal) phases=$(join(phases,',')) faces=$face_count seconds=$(round(result.seconds;digits=3))")
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    main()
end

end # module
