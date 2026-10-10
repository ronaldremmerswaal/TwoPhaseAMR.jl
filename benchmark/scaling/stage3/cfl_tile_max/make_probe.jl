#!/usr/bin/env julia
"""Make a copied general-AMR CFL rate-write/tile-maximum probe."""
module CFLTileMaxProbe

include(joinpath(@__DIR__, "..", "..", "Support.jl"))
using .ScalingSupport: writejson, sha256file, source_hash, run_capture

const RELATIVE = joinpath("src", "operators", "advection", "cell_advection.jl")
const EXPECTED_CLEAN_SHA256 =
    "047457e224baa1c5b5dfa26376eff71acd61de02b225ca7e13360df6c3184541"
const EXPECTED_CLEAN_TREE_SHA256 =
    "0bb8826fb52d143ae0fa5f84e043f4ed4d4feeadb4c7328d6fbda33cbf7095a2"

function replace_once(source, original, replacement)
    anchor = findfirst(original, source)
    anchor === nothing && error("source anchor missing: $original")
    findnext(original, source, last(anchor) + 1) === nothing ||
        error("source anchor repeated: $original")
    return replace(source, original => replacement; count=1)
end

const TILE_LAYOUT = raw"""
# Match the ordinary block-column traversal's serial and parallel tile rules.
@inline function _cell_advection_rate_tile_layout(mesh)
    nx, ny = mesh.block_size
    number_of_blocks = nblocks(mesh)
    if _compute_threads() == 1 || number_of_blocks * nx * ny < _PARALLEL_GRAIN
        return ny, 1
    end
    requested_tiles = max(1, cld(_BATCHES_PER_THREAD * _compute_threads(), number_of_blocks))
    columns_per_tile = max(1, cld(ny, requested_tiles))
    return columns_per_tile, cld(ny, columns_per_tile)
end

"""

const MAXIMUM_PASS = raw"""
    columns_per_tile = workspace.rate_columns_per_tile
    tiles_per_block = workspace.rate_tiles_per_block
    # Only ordinary IEEE binary floats take the reordered finite maximum.
    # Other supported number types retain the original serial reduction.
    fast_type = eltype(rates) <: Union{Float16, Float32, Float64} &&
                T <: Union{Float16, Float32, Float64}
    _parallel_block_columns(mesh) do b, first_j, last_j
        rates_block = @inbounds rates.blocks[b]
        volume = mesh.blocks[b].volume
        cell_offset = (b - 1) * cells_per_block
        tile_max = fast_type ? zero(T) : nothing
        exceptional = false
        @inbounds for j in first_j:last_j, i in 1:nx
            cell = cell_offset + (j - 1) * nx + i
            integrated = _cell_advection_sum(workspace, b, i, j, nx, ny, cell, false, T)
            rates_block[i + h, j + h] = integrated / volume
            if fast_type
                # Compare the rounded workspace value, as the original scan does.
                stored_rate = rates_block[i + h, j + h]
                tile_max = max(tile_max, stored_rate)
                exceptional |= !isfinite(stored_rate) ||
                               (iszero(stored_rate) && signbit(stored_rate))
            end
        end
        if fast_type
            tile = (b - 1) * tiles_per_block + div(first_j - 1, columns_per_tile) + 1
            @inbounds workspace.rate_tile_max[tile] = tile_max
            @inbounds workspace.rate_tile_exception[tile] = exceptional ? 0x01 : 0x00
        end
    end
    exceptional = !fast_type
    if !exceptional
        for flag in workspace.rate_tile_exception
            if !iszero(flag)
                exceptional = true
                break
            end
        end
    end
    maxrate = zero(T)
    if exceptional
        # Preserve the original full scan for exceptional values and types.
        for b in eachindex(mesh.blocks)
            rates_block = @inbounds rates.blocks[b]
            @inbounds for j in 1:ny, i in 1:nx
                maxrate = max(maxrate, rates_block[i + h, j + h])
            end
        end
    else
        # Tile IDs follow block order and increasing column ranges.
        for tile_max in workspace.rate_tile_max
            maxrate = max(maxrate, tile_max)
        end
    end
"""

function candidate_source(original)
    result = replace_once(original,
        "    face_kinds::Vector{UInt8}\n    generation::UInt\n",
        "    face_kinds::Vector{UInt8}\n" *
        "    rate_tile_max::Vector{T}\n" *
        "    rate_tile_exception::Vector{UInt8}\n" *
        "    rate_columns_per_tile::Int\n" *
        "    rate_tiles_per_block::Int\n" *
        "    generation::UInt\n")
    result = replace_once(result, "AdvectionWorkspace(field::CellField) =\n",
                          TILE_LAYOUT * "AdvectionWorkspace(field::CellField) =\n")
    result = replace_once(result,
        "    offsets, permutation = _grouped_output_schedule(targets, ncells(mesh))\n" *
        "    return AdvectionWorkspace(\n",
        "    offsets, permutation = _grouped_output_schedule(targets, ncells(mesh))\n" *
        "    columns_per_tile, tiles_per_block = _cell_advection_rate_tile_layout(mesh)\n" *
        "    rate_tile_max = zeros(T, nblocks(mesh) * tiles_per_block)\n" *
        "    rate_tile_exception = zeros(UInt8, length(rate_tile_max))\n" *
        "    return AdvectionWorkspace(\n")
    result = replace_once(result,
        "        face_kinds[permutation], mesh.topology_cache.generation,\n",
        "        face_kinds[permutation], rate_tile_max, rate_tile_exception,\n" *
        "        columns_per_tile, tiles_per_block, mesh.topology_cache.generation,\n")
    original_maximum = """
    _parallel_block_columns(mesh) do b, first_j, last_j
        rates_block = @inbounds rates.blocks[b]
        volume = mesh.blocks[b].volume
        cell_offset = (b - 1) * cells_per_block
        @inbounds for j in first_j:last_j, i in 1:nx
            cell = cell_offset + (j - 1) * nx + i
            integrated = _cell_advection_sum(workspace, b, i, j, nx, ny, cell, false, T)
            rates_block[i + h, j + h] = integrated / volume
        end
    end
    maxrate = zero(T)
    for b in eachindex(mesh.blocks)
        rates_block = @inbounds rates.blocks[b]
        @inbounds for j in 1:ny, i in 1:nx
            maxrate = max(maxrate, rates_block[i + h, j + h])
        end
    end
"""
    result = replace_once(result, original_maximum, MAXIMUM_PASS)
    return result
end

function make_probe(source, output)
    source, output = abspath(source), abspath(output)
    !ispath(output) || error("probe output already exists: $output")
    clean = joinpath(source, RELATIVE)
    sha256file(clean) == EXPECTED_CLEAN_SHA256 ||
        error("source is not the frozen narrow-ID implementation")
    source_hash(source) == EXPECTED_CLEAN_TREE_SHA256 ||
        error("frozen source tree differs")
    candidate = candidate_source(read(clean, String))
    mkpath(output)
    cp(joinpath(source, "src"), joinpath(output, "src"))
    target = joinpath(output, RELATIVE)
    write(target, candidate)
    diff = run_capture(["diff", "-u", "--label", RELATIVE,
                        "--label", RELATIVE * ".cfl-tile-max", clean, target])
    diff.exitcode == 1 || error("expected copied-source diff")
    patch = joinpath(output, "cfl-tile-max.patch")
    write(patch, diff.stdout)
    receipt = Dict{String,Any}(
        "schema" => 1, "source" => source, "output" => output,
        "clean_source_sha256" => sha256file(clean),
        "candidate_source_sha256" => sha256file(target),
        "clean_tree_sha256" => source_hash(source),
        "candidate_tree_sha256" => source_hash(output),
        "generator_sha256" => sha256file(@__FILE__),
        "patch_sha256" => sha256file(patch),
        "scope" => "copied general-AMR stable_timestep stored-rate tile maxima",
        "exceptional_policy" => "original full serial scan for nonfinite, negative-zero, or unsupported rate types",
    )
    writejson(joinpath(output, "cfl-tile-max-receipt.json"), receipt)
    return receipt
end

function main(args=ARGS)
    length(args) == 4 || error("usage: make_probe.jl --source SOURCE --output NEW_DIRECTORY")
    options = Dict(args[i] => args[i + 1] for i in 1:2:length(args))
    Set(keys(options)) == Set(("--source", "--output")) ||
        error("expected only --source and --output")
    receipt = make_probe(options["--source"], options["--output"])
    println("created copied CFL tile-max source: ", receipt["output"])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
