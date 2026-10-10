#!/usr/bin/env julia
"""One isolated fresh-2048 constructor with Julia bounds checks forced on."""
module MeshCrashCheckedStudy

include(joinpath(@__DIR__, "run.jl"))
using .MeshCrashStudy
using TOML

const Study = MeshCrashStudy
const Support = Study.ScalingSupport

function main(args=ARGS)
    options = Study.parse_args(args)
    options === nothing && return
    paths = [@__FILE__, joinpath(@__DIR__, "run.jl"), Study.DRIVER,
        Study.JULIA, joinpath(Study.PROJECT, "Project.toml"),
        joinpath(Study.PROJECT, "Manifest.toml"),
        joinpath(options.source, Study.RELATIVE_ADVECTION),
        joinpath(options.source, "src", "mesh", "topology", "mesh.jl")]
    all(isfile, paths) || error("a required checked-run input is missing")
    Support.sha256file(paths[end - 1]) == Study.EXPECTED_ADVECTION_SHA256 &&
        Support.sha256file(paths[end]) == Study.EXPECTED_MESH_SHA256 &&
        Support.sha256file(Study.JULIA) == Study.EXPECTED_JULIA_SHA256 &&
        Support.sha256file(joinpath(Study.PROJECT, "Project.toml")) ==
            Study.EXPECTED_PROJECT_SHA256 &&
        Support.sha256file(joinpath(Study.PROJECT, "Manifest.toml")) ==
            Study.EXPECTED_MANIFEST_SHA256 ||
        error("source or numerical environment differs from clean reproducer")
    hashes = Dict(path => Support.sha256file(path) for path in paths)
    tree_hash = Support.source_hash(options.source)
    mkpath(options.output)
    Support.writejson(joinpath(options.output, "configuration.json"), Dict(
        "schema" => 1, "mode" => "fresh2048", "check_bounds" => "yes",
        "source" => options.source, "source_tree_sha256" => tree_hash,
        "cpu" => 0, "locked_files_sha256" => hashes))
    events = joinpath(options.output, "events-fresh2048.toml")
    command = [Study.JULIA, "--startup-file=no", "--check-bounds=yes",
               "--project=$(Study.PROJECT)", "--threads=1", "--gcthreads=1",
               Study.DRIVER, options.source, events, "fresh2048"]
    environment = Dict("JULIA_DEPOT_PATH" => Study.DEPOT,
                       "OPENBLAS_NUM_THREADS" => "1", "MKL_NUM_THREADS" => "1",
                       "JULIA_NUM_GC_THREADS" => "1")
    result = Support.run_capture(command; env=environment, cpus=[0])
    all(Support.sha256file(path) == digest for (path,digest) in hashes) &&
        Support.source_hash(options.source) == tree_hash ||
        error("frozen diagnostic input changed during launch")
    phases = isfile(events) ?
        [entry["phase"] for entry in TOML.parsefile(events)["events"]] : String[]
    Support.writejson(joinpath(options.output, "receipt.json"), Dict(
        "command" => command, "exit_code" => result.exitcode,
        "termsignal" => result.termsignal, "seconds" => result.seconds,
        "stdout" => result.stdout, "stderr" => result.stderr,
        "events_file" => events, "event_phases" => phases))
    println("fresh2048 --check-bounds=yes exit=$(result.exitcode)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
