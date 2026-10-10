#!/usr/bin/env julia
"""Run one fresh 2048² copied-source face-production diagnostic."""
module MeshCrashFaceProducerStudy

include(joinpath(@__DIR__, "run.jl"))
include(joinpath(@__DIR__, "make_face_producer_probe.jl"))
using .MeshCrashStudy
using .MeshCrashFaceProducerProbe
using TOML

const Study = MeshCrashStudy
const Probe = MeshCrashFaceProducerProbe
const Support = Study.ScalingSupport
const DRIVER_METHOD_ANCHOR = """
realpath(String(method.file)) == realpath(MESH_FILE) ||
    error("loaded dual-volume method is outside the copied source")
"""
const DRIVER_METHOD_PATCH = """
realpath(String(method.file)) == realpath(MESH_FILE) ||
    error("loaded dual-volume method is outside the copied source")
face_schedule_method = only(collect(methods(TwoPhaseAMR._build_face_schedules)))
match_sides_method = only(collect(methods(TwoPhaseAMR._match_sides!)))
for loaded_method in (face_schedule_method, match_sides_method)
    realpath(String(loaded_method.file)) == realpath(MESH_FILE) ||
        error("loaded face-production method is outside the copied source")
end
"""
const DRIVER_EVENT_ANCHOR =
    "        \"dual_volume_method_file\" => realpath(String(method.file)),"
const DRIVER_EVENT_PATCH = join((
    DRIVER_EVENT_ANCHOR,
    "        \"face_schedule_method_file\" => realpath(String(face_schedule_method.file)),",
    "        \"match_sides_method_file\" => realpath(String(match_sides_method.file)),",
), "\n")

function main(args=ARGS)
    options = Study.parse_args(args)
    options === nothing && return
    files = [@__FILE__, joinpath(@__DIR__, "run.jl"),
        joinpath(@__DIR__, "make_face_producer_probe.jl"), Study.DRIVER,
        Study.JULIA, joinpath(Study.PROJECT, "Project.toml"),
        joinpath(Study.PROJECT, "Manifest.toml"),
        joinpath(options.source, Study.RELATIVE_ADVECTION),
        joinpath(options.source, Probe.RELATIVE)]
    all(isfile, files) || error("a required face-probe input is missing")
    Support.sha256file(files[end - 1]) == Study.EXPECTED_ADVECTION_SHA256 &&
        Support.sha256file(files[end]) == Probe.CLEAN_SHA256 &&
        Support.sha256file(Study.JULIA) == Study.EXPECTED_JULIA_SHA256 &&
        Support.sha256file(joinpath(Study.PROJECT, "Project.toml")) ==
            Study.EXPECTED_PROJECT_SHA256 &&
        Support.sha256file(joinpath(Study.PROJECT, "Manifest.toml")) ==
            Study.EXPECTED_MANIFEST_SHA256 ||
        error("source or numerical environment differs from frozen control")
    clean_tree = Support.source_hash(options.source)
    mkpath(options.output)
    copied = joinpath(options.output, "copied-source")
    probe_receipt = Probe.make_probe(options.source, copied)
    driver = joinpath(options.output, "mesh_driver.jl")
    driver_original = read(Study.DRIVER, String)
    length(split(driver_original, DRIVER_METHOD_ANCHOR)) == 2 &&
        length(split(driver_original, DRIVER_EVENT_ANCHOR)) == 2 ||
        error("copied driver method anchors are absent or repeated")
    driver_copy = replace(driver_original,
        DRIVER_METHOD_ANCHOR => DRIVER_METHOD_PATCH; count=1)
    driver_copy = replace(driver_copy,
        DRIVER_EVENT_ANCHOR => DRIVER_EVENT_PATCH; count=1)
    write(driver, driver_copy)
    driver_diff = Support.run_capture(["diff", "-u", "--label", "mesh_driver.jl",
        "--label", "mesh_driver.jl.face-producer", Study.DRIVER, driver])
    driver_diff.exitcode == 1 || error("expected copied driver method diff")
    driver_patch = joinpath(options.output, "face-driver-methods.patch")
    write(driver_patch, driver_diff.stdout)
    append!(files, [driver, joinpath(copied, Probe.RELATIVE),
                    joinpath(copied, "face-producer.patch"),
                    joinpath(copied, "face-producer-receipt.json"),
                    driver_patch])
    hashes = Dict(file => Support.sha256file(file) for file in files)
    copied_tree = Support.source_hash(copied)
    copied_tree == probe_receipt["probe_tree_sha256"] &&
        clean_tree == probe_receipt["clean_tree_sha256"] ||
        error("copied source differs from probe receipt")
    Support.writejson(joinpath(options.output, "configuration.json"), Dict(
        "schema" => 1, "source" => options.source,
        "clean_source_tree_sha256" => clean_tree,
        "copied_source_tree_sha256" => copied_tree,
        "copied_mesh_sha256" => probe_receipt["probe_mesh_sha256"],
        "julia_version_expected" => "1.10.12", "cpu" => 0,
        "locked_files_sha256" => hashes,
        "probe_receipt" => probe_receipt))
    all(Support.sha256file(file) == hash for (file, hash) in hashes) &&
        Support.source_hash(options.source) == clean_tree &&
        Support.source_hash(copied) == copied_tree ||
        error("probe input changed before launch")
    events = joinpath(options.output, "events.toml")
    probe_log = joinpath(options.output, "producer-probe.toml")
    command = [Study.JULIA, "--startup-file=no",
               "--project=$(Study.PROJECT)", "--threads=1", "--gcthreads=1",
               driver, copied, events, "fresh2048",
               probe_receipt["probe_mesh_sha256"]]
    environment = Dict("JULIA_DEPOT_PATH" => Study.DEPOT,
                       "OPENBLAS_NUM_THREADS" => "1", "MKL_NUM_THREADS" => "1",
                       "JULIA_NUM_GC_THREADS" => "1",
                       "TWOPHASE_FACE_PROBE_OUTPUT" => probe_log)
    result = Support.run_capture(command; env=environment, cpus=[0])
    all(Support.sha256file(file) == hash for (file, hash) in hashes) &&
        Support.source_hash(options.source) == clean_tree &&
        Support.source_hash(copied) == copied_tree ||
        error("probe input changed during launch")
    phases = isfile(events) ?
        [entry["phase"] for entry in TOML.parsefile(events)["events"]] : String[]
    summary = isfile(probe_log) ? TOML.parsefile(probe_log) : nothing
    Support.writejson(joinpath(options.output, "receipt.json"), Dict(
        "schema" => 1, "command" => command,
        "exit_code" => result.exitcode, "termsignal" => result.termsignal,
        "seconds" => result.seconds, "stdout" => result.stdout,
        "stderr" => result.stderr, "events_file" => events,
        "event_phases" => phases, "probe_file" => probe_log,
        "probe_complete" => summary !== nothing,
        "anomaly_counts" => summary === nothing ? nothing : summary["counts"],
        "intentional_probe_stop" => summary !== nothing &&
            occursin("face producer probe stopped", result.stderr)))
    println("face producer exit=$(result.exitcode) probe_complete=$(summary !== nothing)")
    summary === nothing || println("anomaly counts: ", summary["counts"])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
