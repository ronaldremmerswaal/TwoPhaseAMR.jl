#!/usr/bin/env julia
"""One matched frozen/hoisted fresh-2048 mesh pair, without prevalidation."""
module MeshCrashConstructorStudy

include(joinpath(@__DIR__, "run.jl"))
include(joinpath(@__DIR__, "make_constructor_probe.jl"))
using .MeshCrashStudy
using .MeshCrashConstructorProbe
using TOML

const Study = MeshCrashStudy
const Probe = MeshCrashConstructorProbe
const Support = Study.ScalingSupport

function main(args=ARGS)
    options = Study.parse_args(args)
    options === nothing && return
    source_file = joinpath(options.source, Study.RELATIVE_ADVECTION)
    mesh_file = joinpath(options.source, Probe.RELATIVE)
    Support.sha256file(source_file) == Study.EXPECTED_ADVECTION_SHA256 &&
        Support.sha256file(mesh_file) == Probe.EXPECTED_CLEAN_SHA256 ||
        error("control source differs from frozen narrow-ID cohort")
    fixed_paths = [@__FILE__, joinpath(@__DIR__, "run.jl"),
        joinpath(@__DIR__, "make_constructor_probe.jl"), Study.DRIVER,
        Study.JULIA, joinpath(Study.PROJECT, "Project.toml"),
        joinpath(Study.PROJECT, "Manifest.toml"), source_file, mesh_file]
    all(isfile, fixed_paths) || error("a required probe input is missing")
    Support.sha256file(Study.JULIA) == Study.EXPECTED_JULIA_SHA256 &&
        Support.sha256file(joinpath(Study.PROJECT, "Project.toml")) ==
            Study.EXPECTED_PROJECT_SHA256 &&
        Support.sha256file(joinpath(Study.PROJECT, "Manifest.toml")) ==
            Study.EXPECTED_MANIFEST_SHA256 ||
        error("Julia executable or frozen numerical project differs")
    mkpath(options.output)
    control = joinpath(options.output, "control")
    candidate = joinpath(options.output, "hoisted")
    mkpath(control)
    cp(joinpath(options.source, "src"), joinpath(control, "src"))
    probe_receipt = Probe.make_probe(options.source, candidate)
    driver = joinpath(options.output, "mesh_driver.jl")
    cp(Study.DRIVER, driver)
    Support.source_hash(control) == Support.source_hash(options.source) ||
        error("control copy differs from frozen source")
    roots = Dict("control" => control, "hoisted" => candidate)
    mesh_hashes = Dict("control" => Probe.EXPECTED_CLEAN_SHA256,
                       "hoisted" => probe_receipt["hoisted_mesh_sha256"])
    tree_hashes = Dict(label => Support.source_hash(path) for (label,path) in roots)
    locked = Dict(path => Support.sha256file(path) for path in vcat(fixed_paths,
        [driver, joinpath(control, Probe.RELATIVE),
         joinpath(candidate, Probe.RELATIVE),
         joinpath(candidate, "constructor-hoist.patch"),
         joinpath(candidate, "constructor-hoist-receipt.json")]))
    Support.writejson(joinpath(options.output, "configuration.json"), Dict(
        "schema" => 1, "source" => options.source, "cpu" => 0,
        "julia_version_expected" => "1.10.12",
        "cases" => ["control", "hoisted"],
        "source_tree_sha256" => tree_hashes,
        "mesh_sha256" => mesh_hashes,
        "locked_files_sha256" => locked,
        "probe_receipt" => probe_receipt))
    environment = Dict("JULIA_DEPOT_PATH" => Study.DEPOT,
                       "OPENBLAS_NUM_THREADS" => "1", "MKL_NUM_THREADS" => "1",
                       "JULIA_NUM_GC_THREADS" => "1")
    exits = Dict{String,Int}()
    for label in ("control", "hoisted")
        all(Support.sha256file(path) == digest for (path,digest) in locked) &&
            all(Support.source_hash(path) == tree_hashes[name]
                for (name,path) in roots) ||
            error("diagnostic source or executable changed")
        source = roots[label]
        events = joinpath(options.output, "events-$label.toml")
        command = [Study.JULIA, "--startup-file=no",
                   "--project=$(Study.PROJECT)", "--threads=1", "--gcthreads=1",
                   driver, source, events, "fresh2048", mesh_hashes[label]]
        result = Support.run_capture(command; env=environment, cpus=[0])
        phases = isfile(events) ?
            [entry["phase"] for entry in TOML.parsefile(events)["events"]] : String[]
        Support.writejson(joinpath(options.output, "receipt-$label.json"), Dict(
            "variant" => label, "command" => command,
            "exit_code" => result.exitcode, "termsignal" => result.termsignal,
            "seconds" => result.seconds, "stdout" => result.stdout,
            "stderr" => result.stderr, "events_file" => events,
            "event_phases" => phases))
        all(Support.sha256file(path) == digest for (path,digest) in locked) &&
            all(Support.source_hash(path) == tree_hashes[name]
                for (name,path) in roots) ||
            error("diagnostic source or executable changed during launch")
        exits[label] = result.exitcode
        println("$label exit=$(result.exitcode) phases=$(join(phases, ','))")
        flush(stdout)
    end
    Support.writejson(joinpath(options.output, "comparison.json"), Dict(
        "schema" => 1, "exit_codes" => exits,
        "interpretation" => "single copied-source liveness/codegen diagnostic; no timing inference"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
