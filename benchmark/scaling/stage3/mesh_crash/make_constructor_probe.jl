#!/usr/bin/env julia
"""Copy the frozen source and hoist only its mesh dual-volume call."""
module MeshCrashConstructorProbe

include(joinpath(@__DIR__, "..", "..", "Support.jl"))
using .ScalingSupport: sha256file, source_hash, run_capture, writejson

const RELATIVE = joinpath("src", "mesh", "topology", "mesh.jl")
const EXPECTED_CLEAN_SHA256 =
    "d62cebb9cb5775f3b72de974ec55e2d4bf572f6daefdb28b1a89ff46eccf86fd"
const ORIGINAL = join((
    "    return BlockTreeMesh(",
    "        root, blocks, mesh_base, (nx, ny), halo, mesh_origin, mesh_extent,",
    "        mesh_periodic, max_level, faces, _face_dual_volumes(blocks, faces),",
    "        boundaries,",
), "\n")
const HOISTED = join((
    "    dual_volumes = _face_dual_volumes(blocks, faces)",
    "    return BlockTreeMesh(",
    "        root, blocks, mesh_base, (nx, ny), halo, mesh_origin, mesh_extent,",
    "        mesh_periodic, max_level, faces, dual_volumes,",
    "        boundaries,",
), "\n")

function make_probe(source, output)
    source, output = abspath(source), abspath(output)
    !ispath(output) || error("probe output already exists: $output")
    clean = joinpath(source, RELATIVE)
    isfile(clean) && sha256file(clean) == EXPECTED_CLEAN_SHA256 ||
        error("source mesh is not the frozen control")
    original = read(clean, String)
    length(split(original, ORIGINAL)) == 2 ||
        error("constructor anchor is missing or not unique")
    candidate = replace(original, ORIGINAL => HOISTED; count=1)
    candidate != original || error("constructor transformation made no change")
    mkpath(output)
    cp(joinpath(source, "src"), joinpath(output, "src"))
    target = joinpath(output, RELATIVE)
    write(target, candidate)
    diff = run_capture(["diff", "-u", "--label", RELATIVE,
                        "--label", RELATIVE * ".hoisted", clean, target])
    diff.exitcode == 1 || error("expected one copied-source diff")
    patch = joinpath(output, "constructor-hoist.patch")
    write(patch, diff.stdout)
    receipt = Dict{String,Any}(
        "schema" => 1, "source" => source, "output" => output,
        "clean_mesh_sha256" => sha256file(clean),
        "hoisted_mesh_sha256" => sha256file(target),
        "clean_tree_sha256" => source_hash(source),
        "hoisted_tree_sha256" => source_hash(output),
        "generator_sha256" => sha256file(@__FILE__),
        "patch_sha256" => sha256file(patch),
        "scope" => "only bind _face_dual_volumes before final BlockTreeMesh expression")
    writejson(joinpath(output, "constructor-hoist-receipt.json"), receipt)
    return receipt
end

function main(args=ARGS)
    if any(arg -> arg in ("-h", "--help"), args)
        println("usage: make_constructor_probe.jl --source FROZEN_SOURCE --output NEW_DIRECTORY")
        return nothing
    end
    length(args) == 4 || error("--source and --output are required")
    options = Dict(args[i] => args[i + 1] for i in 1:2:length(args))
    Set(keys(options)) == Set(("--source", "--output")) ||
        error("expected only --source and --output")
    receipt = make_probe(options["--source"], options["--output"])
    println("created copied constructor probe: ", receipt["output"])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
