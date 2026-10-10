#!/usr/bin/env julia
"""Launch bounded copied-source staged face-validity diagnostics."""
module MeshCrashStagedStudy

include(joinpath(@__DIR__, "run.jl"))
using .MeshCrashStudy
using TOML

const Study = MeshCrashStudy
const Support = Study.ScalingSupport
const DRIVER = joinpath(@__DIR__, "staged_driver.jl")

function parse_options(args)
    if any(arg -> arg in ("-h", "--help"), args)
        println("usage: run_staged.jl --source FROZEN_SOURCE --output NEW_DIRECTORY [--mode staged|phase_scan --repeats 1|2]")
        return nothing
    end
    iseven(length(args)) || error("expected key/value options")
    options = Dict(args[i] => args[i + 1] for i in 1:2:length(args))
    Set(keys(options)) ⊆ Set(("--source", "--output", "--mode", "--repeats")) ||
        error("unknown diagnostic option")
    haskey(options, "--source") && haskey(options, "--output") ||
        error("--source and --output are required")
    base = Study.parse_args(["--source", options["--source"],
                             "--output", options["--output"]])
    mode = get(options, "--mode", "staged")
    repeats = parse(Int, get(options, "--repeats", "1"))
    ((mode == "staged" && repeats == 1) ||
     (mode == "phase_scan" && repeats == 2)) ||
        error("only one staged or two phase-scan processes are permitted")
    return merge(base, (; mode, repeats))
end

function main(args=ARGS)
    options = parse_options(args)
    options === nothing && return
    paths = [@__FILE__, joinpath(@__DIR__, "run.jl"), DRIVER,
        Study.JULIA, joinpath(Study.PROJECT, "Project.toml"),
        joinpath(Study.PROJECT, "Manifest.toml"),
        joinpath(options.source, Study.RELATIVE_ADVECTION),
        joinpath(options.source, "src", "mesh", "topology", "mesh.jl")]
    all(isfile, paths) || error("a required staged-run input is missing")
    Support.sha256file(paths[end - 1]) == Study.EXPECTED_ADVECTION_SHA256 &&
        Support.sha256file(paths[end]) == Study.EXPECTED_MESH_SHA256 &&
        Support.sha256file(Study.JULIA) == Study.EXPECTED_JULIA_SHA256 &&
        Support.sha256file(joinpath(Study.PROJECT, "Project.toml")) ==
            Study.EXPECTED_PROJECT_SHA256 &&
        Support.sha256file(joinpath(Study.PROJECT, "Manifest.toml")) ==
            Study.EXPECTED_MANIFEST_SHA256 ||
        error("source or numerical environment differs from clean reproducer")
    mkpath(options.output)
    driver = joinpath(options.output, "staged_driver.jl")
    cp(DRIVER, driver)
    push!(paths, driver)
    hashes = Dict(path => Support.sha256file(path) for path in paths)
    tree_hash = Support.source_hash(options.source)
    Support.writejson(joinpath(options.output, "configuration.json"), Dict(
        "schema" => 1, "mode" => options.mode, "repeats" => options.repeats,
        "source" => options.source, "source_tree_sha256" => tree_hash,
        "cpu" => 0, "locked_files_sha256" => hashes))
    environment = Dict("JULIA_DEPOT_PATH" => Study.DEPOT,
                       "OPENBLAS_NUM_THREADS" => "1", "MKL_NUM_THREADS" => "1",
                       "JULIA_NUM_GC_THREADS" => "1")
    for repeat in 1:options.repeats
        all(Support.sha256file(path) == digest for (path,digest) in hashes) &&
            Support.source_hash(options.source) == tree_hash ||
            error("frozen diagnostic input changed before launch")
        stem = options.repeats == 1 ? "" : "-r$repeat"
        events = joinpath(options.output, "events$stem.toml")
        command = [Study.JULIA, "--startup-file=no",
                   "--project=$(Study.PROJECT)", "--threads=1", "--gcthreads=1",
                   driver, options.source, events, options.mode]
        result = Support.run_capture(command; env=environment, cpus=[0])
        all(Support.sha256file(path) == digest for (path,digest) in hashes) &&
            Support.source_hash(options.source) == tree_hash ||
            error("frozen diagnostic input changed during launch")
        entries = isfile(events) ? TOML.parsefile(events)["events"] : Any[]
        phases = [entry["phase"] for entry in entries]
        scans = Dict(entry["phase"] => Dict(
            "invalid_count" => entry["invalid_count"],
            "field_fnv64_hex" => entry["field_fnv64_hex"],
            "field_sum64_hex" => entry["field_sum64_hex"])
            for entry in entries if entry["phase"] in
                ("scanned_after_faces", "scanned_after_halo"))
        Support.writejson(joinpath(options.output, "receipt$stem.json"), Dict(
            "repeat" => repeat, "command" => command,
            "exit_code" => result.exitcode,
            "termsignal" => result.termsignal, "seconds" => result.seconds,
            "stdout" => result.stdout, "stderr" => result.stderr,
            "events_file" => events, "event_phases" => phases,
            "face_scans" => scans))
        println("$(options.mode) r$repeat exit=$(result.exitcode) phases=$(join(phases, ','))")
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
