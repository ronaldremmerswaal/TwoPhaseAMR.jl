#!/usr/bin/env julia
"""Match the Julia 1.12 diagnostic driver under frozen Julia 1.10.12."""
module MeshCrashRuntimeMatch

include(joinpath(@__DIR__, "run.jl"))
include(joinpath(@__DIR__, "run_runtime_check.jl"))
using .MeshCrashStudy
using TOML

const Study = MeshCrashStudy
const Support = Study.ScalingSupport
const DriverGenerator = MeshCrashRuntimeCheck
const EXPECTED_DRIVER_SHA256 =
    "24c6ac0599d4f21b30a8e0417222cdc90268dafb6b8b5091f0e083acf1055077"
const EXPECTED_SOURCE_TREE_SHA256 =
    "0bb8826fb52d143ae0fa5f84e043f4ed4d4feeadb4c7328d6fbda33cbf7095a2"

function canonical_driver()
    generated = read(Study.DRIVER, String)
    generated = DriverGenerator.replace_once(generated,
        DriverGenerator.METHOD_ANCHOR, DriverGenerator.METHOD_PATCH)
    generated = DriverGenerator.replace_once(generated,
        DriverGenerator.CONSTRUCT_ANCHOR, DriverGenerator.CONSTRUCT_PATCH)
    return DriverGenerator.replace_once(generated,
        DriverGenerator.VALIDATOR_ANCHOR,
        DriverGenerator.VALIDATOR * DriverGenerator.VALIDATOR_ANCHOR)
end

function parse_options(args)
    if any(arg -> arg in ("-h", "--help"), args)
        println("usage: run_runtime_match.jl --source FROZEN_SOURCE --output NEW_DIRECTORY --variant default|noinline")
        return nothing
    end
    iseven(length(args)) || error("expected key/value options")
    options = Dict(args[i] => args[i+1] for i in 1:2:length(args))
    Set(keys(options)) == Set(("--source", "--output", "--variant")) ||
        error("--source, --output, and --variant are required")
    variant = options["--variant"]
    variant in ("default", "noinline") || error("unknown runtime variant")
    basic = Study.parse_args(["--source", options["--source"],
                              "--output", options["--output"]])
    return merge(basic, (; variant))
end

function main(args=ARGS)
    options = parse_options(args)
    options === nothing && return
    files = [@__FILE__, joinpath(@__DIR__, "run.jl"),
        joinpath(@__DIR__, "run_runtime_check.jl"), Study.DRIVER,
        Study.JULIA, joinpath(Study.PROJECT, "Project.toml"),
        joinpath(Study.PROJECT, "Manifest.toml"),
        joinpath(options.source, Study.RELATIVE_ADVECTION),
        joinpath(options.source, "src", "mesh", "topology", "mesh.jl")]
    all(isfile, files) || error("runtime comparison input missing")
    Support.sha256file(Study.JULIA) == Study.EXPECTED_JULIA_SHA256 &&
        Support.sha256file(files[end-1]) == Study.EXPECTED_ADVECTION_SHA256 &&
        Support.sha256file(files[end]) == Study.EXPECTED_MESH_SHA256 &&
        Support.sha256file(joinpath(Study.PROJECT,"Project.toml")) ==
            Study.EXPECTED_PROJECT_SHA256 &&
        Support.sha256file(joinpath(Study.PROJECT,"Manifest.toml")) ==
            Study.EXPECTED_MANIFEST_SHA256 ||
        error("frozen runtime comparison input differs")
    tree = Support.source_hash(options.source)
    tree == EXPECTED_SOURCE_TREE_SHA256 ||
        error("source tree differs from frozen narrow-ID candidate")
    mkpath(options.output)
    driver = joinpath(options.output, "runtime_driver.jl")
    write(driver, canonical_driver())
    Support.sha256file(driver) == EXPECTED_DRIVER_SHA256 ||
        error("generated driver differs from validated Julia 1.12 diagnostic")
    push!(files, driver)
    hashes = Dict(file => Support.sha256file(file) for file in files)
    Support.writejson(joinpath(options.output,"configuration.json"), Dict(
        "schema" => 1, "variant" => options.variant,
        "source" => options.source, "source_tree_sha256" => tree,
        "driver_matches_julia112_sha256" => EXPECTED_DRIVER_SHA256,
        "julia_version_expected" => "1.10.12", "cpu" => 0,
        "depot" => Study.DEPOT, "locked_files_sha256" => hashes))
    all(Support.sha256file(file) == hash for (file,hash) in hashes) &&
        Support.source_hash(options.source) == tree ||
        error("runtime input changed before launch")
    events = joinpath(options.output,"events.toml")
    validation = joinpath(options.output,"validation.toml")
    command = [Study.JULIA, "--startup-file=no",
        "--project=$(Study.PROJECT)", "--threads=1", "--gcthreads=1"]
    options.variant == "noinline" && push!(command,"--inline=no")
    append!(command,[driver, options.source, events, "fresh2048"])
    environment = Dict("JULIA_DEPOT_PATH" => Study.DEPOT,
        "OPENBLAS_NUM_THREADS" => "1", "MKL_NUM_THREADS" => "1",
        "JULIA_NUM_GC_THREADS" => "1",
        "TWOPHASE_RUNTIME_VALIDATION" => validation)
    result = Support.run_capture(command; env=environment, cpus=[0])
    all(Support.sha256file(file) == hash for (file,hash) in hashes) &&
        Support.source_hash(options.source) == tree ||
        error("runtime input changed during launch")
    phases = isfile(events) ?
        [entry["phase"] for entry in TOML.parsefile(events)["events"]] : String[]
    summary = isfile(validation) ? TOML.parsefile(validation) : nothing
    Support.writejson(joinpath(options.output,"receipt.json"), Dict(
        "schema" => 1, "variant" => options.variant,
        "command" => command, "exit_code" => result.exitcode,
        "termsignal" => result.termsignal, "seconds" => result.seconds,
        "stdout" => result.stdout, "stderr" => result.stderr,
        "event_phases" => phases, "validation" => summary))
    println("Julia1.10 $(options.variant) exit=$(result.exitcode) phases=$(join(phases,','))")
    summary === nothing || println("face/dual errors: ",
        summary["bad_face_count"], "/", summary["bad_dual_count"])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
