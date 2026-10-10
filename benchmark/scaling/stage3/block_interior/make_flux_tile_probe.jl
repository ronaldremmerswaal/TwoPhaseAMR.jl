#!/usr/bin/env julia
"""Make a copied general-AMR regular-face flux traversal with 128×8 tiles."""
module BlockInteriorFluxTileProbe

include(joinpath(@__DIR__, "..", "..", "Support.jl"))
using .ScalingSupport: writejson, sha256file, source_hash, run_capture

const RELATIVE = joinpath("src", "operators", "advection", "cell_advection.jl")
const EXPECTED_CLEAN_SHA256 = "cecb8463cccada2e16d90f5245b506842d9cc90f6df9c380bb69fee673e4f115"
const START_ANCHOR = "function _cell_advection_regular_fluxes!("
const END_ANCHOR = "\nfunction _cell_advection_regular_rates!("

# Only the iteration order changes. Every original x/y slot is still written
# once, and live face-ID-indexed velocity and aperture arrays are read inside
# the pass. The outer column partition retains the production ownership rule.
const TILED_HELPER = raw"""
function _cell_advection_regular_fluxes!(
        scalar::CellField, velocity::FaceVelocity, dt,
        workspace::AdvectionWorkspace, geometry
    )
    mesh = scalar.mesh
    nx, ny = mesh.block_size
    h = mesh.halo
    stride = nx + 2h
    _parallel_block_columns(mesh) do b, first_j, last_j
        values = @inbounds scalar.blocks[b]
        x_area = @inbounds workspace.x_areas[b]
        y_area = @inbounds workspace.y_areas[b]
        @inbounds for jt in first_j:8:last_j
            jlast = min(last_j, jt + 7)
            for it in 1:128:nx
                ilast = min(nx, it + 127)
                for j in jt:jlast
                    row = (j + h - 1) * stride + h
                    for i in it:min(ilast, nx - 1)
                        slot = _cell_advection_x_slot(b, i, j, nx, ny)
                        face_id = workspace.x_face_ids[slot]
                        speed = velocity.interior[face_id]
                        state = speed >= 0 ? values[row + i] : values[row + i + 1]
                        workspace.x_flux[slot] =
                            _cell_advection_regular_flux(dt, speed, state, x_area, face_id, geometry)
                    end
                    if j < ny
                        for i in it:ilast
                            slot = _cell_advection_y_slot(b, i, j, nx, ny)
                            face_id = workspace.y_face_ids[slot]
                            speed = velocity.interior[face_id]
                            state = speed >= 0 ? values[row + i] : values[row + i + stride]
                            workspace.y_flux[slot] =
                                _cell_advection_regular_flux(dt, speed, state, y_area, face_id, geometry)
                        end
                    end
                end
            end
        end
    end
    return nothing
end
"""

function tiled_source(original)
    first_range = findfirst(START_ANCHOR, original)
    first_range === nothing && error("regular flux helper anchor missing")
    last_range = findnext(END_ANCHOR, original, last(first_range) + 1)
    last_range === nothing && error("regular rate helper anchor missing")
    length(split(original, START_ANCHOR)) == 2 ||
        error("regular flux helper anchor is not unique")
    before = original[firstindex(original):prevind(original, first(first_range))]
    after = original[first(last_range):lastindex(original)]
    return before * TILED_HELPER * after
end

function make_probe(source, output)
    source, output = abspath(source), abspath(output)
    !ispath(output) || error("probe output already exists: $output")
    clean = joinpath(source, RELATIVE)
    isfile(clean) || error("clean advection source missing: $clean")
    sha256file(clean) == EXPECTED_CLEAN_SHA256 ||
        error("source is not the frozen block-interior implementation")
    original = read(clean, String)
    candidate = tiled_source(original)
    candidate != original || error("tile transformation made no change")
    mkpath(output)
    cp(joinpath(source, "src"), joinpath(output, "src"))
    target = joinpath(output, RELATIVE)
    write(target, candidate)
    diff = run_capture(["diff", "-u", "--label", RELATIVE,
                        "--label", RELATIVE * ".tile128x8", clean, target])
    diff.exitcode == 1 || error("expected exactly one copied-source diff")
    patch = joinpath(output, "flux-tile.patch")
    write(patch, diff.stdout)
    receipt = Dict{String,Any}(
        "schema" => 1, "source" => source, "output" => output,
        "clean_source_sha256" => sha256file(clean),
        "candidate_source_sha256" => sha256file(target),
        "clean_tree_sha256" => source_hash(source),
        "candidate_tree_sha256" => source_hash(output),
        "generator_sha256" => sha256file(@__FILE__),
        "patch_sha256" => sha256file(patch),
        "tile" => [128, 8],
        "scope" => "copied regular x/y face traversal only; every AMR leaf",
        "policy" => "one flux per owned regular face; live face IDs, velocities, and apertures")
    writejson(joinpath(output, "flux-tile-receipt.json"), receipt)
    return receipt
end

function main(args=ARGS)
    if any(arg -> arg in ("-h", "--help"), args)
        println("usage: make_flux_tile_probe.jl --source SOURCE --output NEW_DIRECTORY")
        return
    end
    length(args) == 4 || error("usage: --source SOURCE --output NEW_DIRECTORY")
    options = Dict(args[i] => args[i + 1] for i in 1:2:length(args))
    Set(keys(options)) == Set(("--source", "--output")) ||
        error("expected only --source and --output")
    receipt = make_probe(options["--source"], options["--output"])
    println("created copied 128×8 flux tile: ", receipt["output"])
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end

end # module
