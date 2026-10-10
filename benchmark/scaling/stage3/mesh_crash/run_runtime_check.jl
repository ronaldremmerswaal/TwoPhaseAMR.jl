#!/usr/bin/env julia
"""One unchanged-source fresh-2048 constructor on Julia 1.12.7."""
module MeshCrashRuntimeCheck

include(joinpath(@__DIR__, "run.jl"))
using .MeshCrashStudy
using TOML

const Study = MeshCrashStudy
const Support = Study.ScalingSupport
const JULIA112 = "/home/ronald/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia"

const METHOD_ANCHOR = """
realpath(String(method.file)) == realpath(MESH_FILE) ||
    error("loaded dual-volume method is outside the copied source")
"""
const METHOD_PATCH = """
realpath(String(method.file)) == realpath(MESH_FILE) ||
    error("loaded dual-volume method is outside the copied source")
for f in (TwoPhaseAMR._build_face_schedules, TwoPhaseAMR._match_sides!)
    loaded_method = only(collect(methods(f)))
    realpath(String(loaded_method.file)) == realpath(MESH_FILE) ||
        error("loaded face-production method is outside the frozen source")
end
"""
const CONSTRUCT_ANCHOR =
    "    event!(\"after_mesh\"; n, face_count=length(mesh.faces))"
const CONSTRUCT_PATCH = CONSTRUCT_ANCHOR * "\n    validate_original_mesh(mesh, n)\n    event!(\"validated_mesh\"; n, face_count=length(mesh.faces))"
const VALIDATOR_ANCHOR = "function construct(n)"
const VALIDATOR = raw"""
function validate_original_mesh(mesh, n)
    length(mesh.blocks) == 1 || error("unexpected block count")
    length(mesh.faces) == 2n*n || error("unexpected face count")
    length(mesh.face_dual_volumes) == length(mesh.faces) ||
        error("dual-volume length mismatch")
    bad_faces = bad_duals = 0
    first_bad_faces = Any[]
    first_bad_duals = Any[]
    expected_dual = 1.0 / (n*n)
    dual_sum = 0.0
    for (index, face) in pairs(mesh.faces)
        minus, plus = face.minus, face.plus
        valid = minus.block == 1 && plus.block == 1 &&
                1 <= minus.i <= n && 1 <= minus.j <= n &&
                1 <= plus.i <= n && 1 <= plus.j <= n &&
                face.axis in (1,2) && isfinite(face.area) && face.area > 0 &&
                all(isfinite, face.center)
        if !valid
            bad_faces += 1
            if length(first_bad_faces) < 10
                push!(first_bad_faces, Dict("index" => index,
                    "minus" => [minus.block, minus.i, minus.j],
                    "plus" => [plus.block, plus.i, plus.j],
                    "axis" => face.axis, "center" => collect(face.center),
                    "area" => face.area))
            end
        end
        dual = mesh.face_dual_volumes[index]
        if !isfinite(dual) || dual != expected_dual
            bad_duals += 1
            if length(first_bad_duals) < 10
                push!(first_bad_duals, Dict("index" => index,
                    "value" => dual, "bits" => string(reinterpret(UInt64,dual); base=16)))
            end
        end
        dual_sum += dual
    end
    report = Dict("schema" => 1, "n" => n,
        "blocks" => length(mesh.blocks), "faces" => length(mesh.faces),
        "dual_count" => length(mesh.face_dual_volumes),
        "expected_dual" => expected_dual, "dual_sum" => dual_sum,
        "bad_face_count" => bad_faces, "bad_dual_count" => bad_duals,
        "first_bad_faces" => first_bad_faces,
        "first_bad_duals" => first_bad_duals)
    output = ENV["TWOPHASE_RUNTIME_VALIDATION"]
    open(output, "w") do io
        TOML.print(io, report)
        flush(io)
        ccall(:fsync, Cint, (Cint,), Base.fd(io)) == 0 || error("validation fsync")
    end
    bad_faces == 0 && bad_duals == 0 || error("constructed mesh validation failed")
    return nothing
end

"""

function replace_once(source, before, after)
    length(split(source, before)) == 2 || error("runtime driver anchor missing or repeated")
    return replace(source, before => after; count=1)
end

function main(args=ARGS)
    options = Study.parse_args(args)
    options === nothing && return
    files = [@__FILE__, joinpath(@__DIR__, "run.jl"), Study.DRIVER,
        JULIA112, joinpath(Study.PROJECT, "Project.toml"),
        joinpath(Study.PROJECT, "Manifest.toml"),
        joinpath(options.source, Study.RELATIVE_ADVECTION),
        joinpath(options.source, "src", "mesh", "topology", "mesh.jl")]
    all(isfile, files) || error("runtime check input missing")
    Support.sha256file(files[end - 1]) == Study.EXPECTED_ADVECTION_SHA256 &&
        Support.sha256file(files[end]) == Study.EXPECTED_MESH_SHA256 &&
        Support.sha256file(joinpath(Study.PROJECT,"Project.toml")) ==
            Study.EXPECTED_PROJECT_SHA256 &&
        Support.sha256file(joinpath(Study.PROJECT,"Manifest.toml")) ==
            Study.EXPECTED_MANIFEST_SHA256 ||
        error("frozen source or numerical project changed")
    clean_tree = Support.source_hash(options.source)
    mkpath(options.output)
    original = read(Study.DRIVER, String)
    copied = replace_once(original, METHOD_ANCHOR, METHOD_PATCH)
    copied = replace_once(copied, CONSTRUCT_ANCHOR, CONSTRUCT_PATCH)
    copied = replace_once(copied, VALIDATOR_ANCHOR, VALIDATOR * VALIDATOR_ANCHOR)
    driver = joinpath(options.output, "runtime_driver.jl")
    write(driver, copied)
    diff = Support.run_capture(["diff", "-u", "--label", "mesh_driver.jl",
        "--label", "runtime_driver.jl", Study.DRIVER, driver])
    diff.exitcode == 1 || error("expected runtime-driver diff")
    patch = joinpath(options.output, "runtime-driver.patch")
    write(patch, diff.stdout)
    append!(files, [driver, patch])
    hashes = Dict(file => Support.sha256file(file) for file in files)
    Support.writejson(joinpath(options.output, "configuration.json"), Dict(
        "schema" => 1, "source" => options.source,
        "source_tree_sha256" => clean_tree,
        "julia_version_expected" => "1.12.7", "cpu" => 0,
        "frozen_project_unchanged" => true,
        "locked_files_sha256" => hashes))
    all(Support.sha256file(file) == hash for (file,hash) in hashes) &&
        Support.source_hash(options.source) == clean_tree ||
        error("runtime check input changed before launch")
    events = joinpath(options.output, "events.toml")
    validation = joinpath(options.output, "validation.toml")
    command = [JULIA112, "--startup-file=no",
        "--project=$(Study.PROJECT)", "--threads=1", "--gcthreads=1",
        driver, options.source, events, "fresh2048"]
    environment = Dict("JULIA_DEPOT_PATH" => Study.DEPOT,
        "OPENBLAS_NUM_THREADS" => "1", "MKL_NUM_THREADS" => "1",
        "JULIA_NUM_GC_THREADS" => "1",
        "TWOPHASE_RUNTIME_VALIDATION" => validation)
    result = Support.run_capture(command; env=environment, cpus=[0])
    all(Support.sha256file(file) == hash for (file,hash) in hashes) &&
        Support.source_hash(options.source) == clean_tree ||
        error("runtime check input changed during launch")
    phases = isfile(events) ?
        [entry["phase"] for entry in TOML.parsefile(events)["events"]] : String[]
    summary = isfile(validation) ? TOML.parsefile(validation) : nothing
    Support.writejson(joinpath(options.output, "receipt.json"), Dict(
        "schema" => 1, "command" => command,
        "exit_code" => result.exitcode, "termsignal" => result.termsignal,
        "seconds" => result.seconds, "stdout" => result.stdout,
        "stderr" => result.stderr, "event_phases" => phases,
        "validation" => summary))
    println("Julia1.12 fresh2048 exit=$(result.exitcode) phases=$(join(phases, ','))")
    summary === nothing || println("face/dual errors: ",
        summary["bad_face_count"], "/", summary["bad_dual_count"])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
