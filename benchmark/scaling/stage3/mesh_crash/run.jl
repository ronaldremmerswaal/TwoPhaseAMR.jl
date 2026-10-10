#!/usr/bin/env julia
"""Two bounded fresh-process reproductions of the 2048 mesh SIGSEGV."""
module MeshCrashStudy

include(joinpath(@__DIR__, "..", "..", "Support.jl"))
using .ScalingSupport
using TOML

const ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
const CONTROL = joinpath(ROOT, "benchmark", "scaling", "results",
                         "20261010-stage3-narrow-ids-control")
const JULIA = "/tmp/julia-1.10.12/bin/julia"
const PROJECT = joinpath(CONTROL, "benchmark-project")
const DEPOT = "/tmp/twophase-julia110-depot:/home/ronald/.julia"
const DRIVER = joinpath(@__DIR__, "mesh_driver.jl")
const RELATIVE_ADVECTION = joinpath("src", "operators", "advection", "cell_advection.jl")
const EXPECTED_ADVECTION_SHA256 =
    "047457e224baa1c5b5dfa26376eff71acd61de02b225ca7e13360df6c3184541"
const EXPECTED_MESH_SHA256 =
    "d62cebb9cb5775f3b72de974ec55e2d4bf572f6daefdb28b1a89ff46eccf86fd"
const EXPECTED_JULIA_SHA256 =
    "1c7ea3ba2e49e6c4a96ca8b80e34f984f65c1dbcdc1423f33a63f2bc14a69a5b"
const EXPECTED_PROJECT_SHA256 =
    "01d49519e37d146088a17f8830bff96f2e941e07c1f79ea895c2df199f1f1e91"
const EXPECTED_MANIFEST_SHA256 =
    "d2bfdb90a95191cd2e7ed74a32cc827e3413f05ef231de44edac5260e92f90d0"

function parse_args(args)
    if any(arg -> arg in ("-h", "--help"), args)
        println("usage: run.jl --source FROZEN_CONTROL_SOURCE --output NEW_DIRECTORY")
        return nothing
    end
    length(args) == 4 || error("--source and --output are required")
    options = Dict(args[i] => args[i + 1] for i in 1:2:length(args))
    Set(keys(options)) == Set(("--source", "--output")) ||
        error("expected only --source and --output")
    source, output = abspath(options["--source"]), abspath(options["--output"])
    isdir(source) || error("missing source: $source")
    !ispath(output) || error("output must be a new directory: $output")
    return (; source, output)
end

function main(args=ARGS)
    options = parse_args(args)
    options === nothing && return
    source_file = joinpath(options.source, RELATIVE_ADVECTION)
    sha256file(source_file) == EXPECTED_ADVECTION_SHA256 ||
        error("source is not frozen narrow-ID control")
    mesh_file = joinpath(options.source, "src", "mesh", "topology", "mesh.jl")
    sha256file(mesh_file) == EXPECTED_MESH_SHA256 ||
        error("mesh source differs from both failed cohorts")
    paths = [DRIVER, @__FILE__, JULIA, source_file,
        mesh_file,
        joinpath(PROJECT, "Project.toml"), joinpath(PROJECT, "Manifest.toml")]
    all(isfile, paths) || error("a frozen source, executable, or environment is missing")
    sha256file(JULIA) == EXPECTED_JULIA_SHA256 &&
        sha256file(joinpath(PROJECT, "Project.toml")) == EXPECTED_PROJECT_SHA256 &&
        sha256file(joinpath(PROJECT, "Manifest.toml")) == EXPECTED_MANIFEST_SHA256 ||
        error("Julia executable or frozen numerical project differs")
    mkpath(options.output)
    hashes = Dict(path => sha256file(path) for path in paths)
    tree_hash = source_hash(options.source)
    writejson(joinpath(options.output, "configuration.json"), Dict(
        "schema" => 1, "source" => options.source,
        "source_tree_sha256" => tree_hash,
        "julia_version_expected" => "1.10.12", "cpu" => 0,
        "sequences" => ["fresh2048", "seq1024_2048"],
        "locked_files_sha256" => hashes))
    for mode in ("fresh2048", "seq1024_2048")
        all(sha256file(path) == digest for (path,digest) in hashes) &&
            source_hash(options.source) == tree_hash ||
            error("frozen diagnostic input changed")
        events = joinpath(options.output, "events-$mode.toml")
        command = [JULIA, "--startup-file=no", "--project=$PROJECT",
                   "--threads=1", "--gcthreads=1", DRIVER,
                   options.source, events, mode]
        environment = Dict("JULIA_DEPOT_PATH" => DEPOT,
                           "OPENBLAS_NUM_THREADS" => "1",
                           "MKL_NUM_THREADS" => "1",
                           "JULIA_NUM_GC_THREADS" => "1")
        result = run_capture(command; env=environment, cpus=[0])
        all(sha256file(path) == digest for (path,digest) in hashes) &&
            source_hash(options.source) == tree_hash ||
            error("frozen diagnostic input changed during launch")
        phases = isfile(events) ?
            [entry["phase"] for entry in TOML.parsefile(events)["events"]] : String[]
        receipt = Dict("mode" => mode, "command" => command,
                       "exit_code" => result.exitcode,
                       "termsignal" => result.termsignal,
                       "seconds" => result.seconds,
                       "stdout" => result.stdout, "stderr" => result.stderr,
                       "events_file" => events, "event_phases" => phases)
        writejson(joinpath(options.output, "receipt-$mode.json"), receipt)
        println("$mode exit=$(result.exitcode) seconds=$(round(result.seconds; digits=2))")
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
