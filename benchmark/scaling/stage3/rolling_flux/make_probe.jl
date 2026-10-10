#!/usr/bin/env julia
"""Make a copied general-AMR increment-first rolling-flux advection probe."""
module RollingFluxProbe

include(joinpath(@__DIR__, "..", "..", "Support.jl"))
using .ScalingSupport: writejson, sha256file, source_hash, run_capture

const RELATIVE = joinpath("src", "operators", "advection", "cell_advection.jl")
const EXPECTED_CLEAN_SHA256 =
    "047457e224baa1c5b5dfa26376eff71acd61de02b225ca7e13360df6c3184541"

function replace_once(source, original, replacement)
    first = findfirst(original, source)
    first === nothing && error("source anchor missing: $original")
    findnext(original, source, last(first) + 1) === nothing ||
        error("source anchor repeated: $original")
    return replace(source, original => replacement; count=1)
end

const LAYOUT = raw"""
@inline function _cell_advection_rolling_layout(mesh)
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

const ROLLING_HELPERS = raw"""
# The first pass stores every regular face touching an edge cell, plus each
# internal tile-seam y face. A face belongs to one block and one lower-row tile.
# Nonstored faces are computed once into that tile's rolling scratch below.
@inline _cell_advection_stored_y_row(j, ny, columns_per_tile) =
    j == 1 || j == ny - 1 || iszero(rem(j, columns_per_tile))

@inline function _cell_advection_store_x_flux!(
        workspace, values, velocity, dt, b, i, j, row, nx, ny, area
    )
    slot = _cell_advection_x_slot(b, i, j, nx, ny)
    @inbounds face_id = workspace.x_face_ids[slot]
    @inbounds speed = velocity.interior[face_id]
    @inbounds state = speed >= 0 ? values[row + i] : values[row + i + 1]
    @inbounds workspace.x_flux[slot] =
        _cell_advection_regular_flux(dt, speed, state, area, face_id, nothing)
    return nothing
end

@inline function _cell_advection_store_y_flux!(
        workspace, values, velocity, dt, b, i, j, row, stride, nx, ny, area
    )
    slot = _cell_advection_y_slot(b, i, j, nx, ny)
    @inbounds face_id = workspace.y_face_ids[slot]
    @inbounds speed = velocity.interior[face_id]
    @inbounds state = speed >= 0 ? values[row + i] : values[row + i + stride]
    @inbounds workspace.y_flux[slot] =
        _cell_advection_regular_flux(dt, speed, state, area, face_id, nothing)
    return nothing
end

function _cell_advection_edge_seam_fluxes!(
        scalar::CellField, velocity::FaceVelocity, dt,
        workspace::AdvectionWorkspace
    )
    mesh = scalar.mesh
    nx, ny = mesh.block_size
    h = mesh.halo
    stride = nx + 2h
    columns_per_tile = workspace.rolling_columns_per_tile
    _parallel_block_columns(mesh) do b, first_j, last_j
        values = @inbounds scalar.blocks[b]
        x_area = @inbounds workspace.x_areas[b]
        y_area = @inbounds workspace.y_areas[b]
        @inbounds for j in first_j:last_j
            row = (j + h - 1) * stride + h
            _cell_advection_store_x_flux!(workspace, values, velocity, dt,
                                          b, 1, j, row, nx, ny, x_area)
            if nx > 2
                _cell_advection_store_x_flux!(workspace, values, velocity, dt,
                                              b, nx - 1, j, row, nx, ny, x_area)
            end
            if j == 1 || j == ny
                for i in 2:(nx - 2)
                    _cell_advection_store_x_flux!(workspace, values, velocity, dt,
                                                  b, i, j, row, nx, ny, x_area)
                end
            end
            if j < ny
                _cell_advection_store_y_flux!(workspace, values, velocity, dt,
                                              b, 1, j, row, stride, nx, ny, y_area)
                _cell_advection_store_y_flux!(workspace, values, velocity, dt,
                                              b, nx, j, row, stride, nx, ny, y_area)
                stored_row = _cell_advection_stored_y_row(j, ny, columns_per_tile)
                if stored_row
                    for i in 2:(nx - 1)
                        _cell_advection_store_y_flux!(workspace, values, velocity,
                            dt, b, i, j, row, stride, nx, ny, y_area)
                    end
                end
            end
        end
    end
    return nothing
end

@inline function _cell_advection_rolling_y_start(base, nx, row)
    return isodd(row) ? base + nx : base + 2nx
end

function _cell_advection_rolling_increments!(
        scalar::CellField{T}, velocity::FaceVelocity, dt,
        workspace::AdvectionWorkspace{T}
    ) where {T}
    mesh = scalar.mesh
    nx, ny = mesh.block_size
    h = mesh.halo
    stride = nx + 2h
    cells_per_block = nx * ny
    columns_per_tile = workspace.rolling_columns_per_tile
    tiles_per_block = workspace.rolling_tiles_per_block
    scratch = workspace.rolling_flux
    _parallel_block_columns(mesh) do b, first_j, last_j
        local_tile = div(first_j - 1, columns_per_tile)
        tile = (b - 1) * tiles_per_block + local_tile + 1
        base = (tile - 1) * (3nx - 1)
        values = @inbounds scalar.blocks[b]
        increment = @inbounds workspace.increment.blocks[b]
        x_area = @inbounds workspace.x_areas[b]
        y_area = @inbounds workspace.y_areas[b]
        cell_offset = (b - 1) * cells_per_block
        @inbounds for j in first_j:last_j
            row = (j + h - 1) * stride + h
            if 1 < j < ny
                for i in 2:(nx - 2)
                    slot = _cell_advection_x_slot(b, i, j, nx, ny)
                    face_id = workspace.x_face_ids[slot]
                    speed = velocity.interior[face_id]
                    state = speed >= 0 ? values[row + i] : values[row + i + 1]
                    scratch[base + i] =
                        _cell_advection_regular_flux(dt, speed, state, x_area, face_id, nothing)
                end
                if !_cell_advection_stored_y_row(j, ny, columns_per_tile)
                    y_start = _cell_advection_rolling_y_start(base, nx, j)
                    for i in 2:(nx - 1)
                        slot = _cell_advection_y_slot(b, i, j, nx, ny)
                        face_id = workspace.y_face_ids[slot]
                        speed = velocity.interior[face_id]
                        state = speed >= 0 ? values[row + i] : values[row + i + stride]
                        scratch[y_start + i - 1] =
                            _cell_advection_regular_flux(dt, speed, state, y_area, face_id, nothing)
                    end
                end
            end
            for i in 1:nx
                cell = cell_offset + (j - 1) * nx + i
                integrated = if 1 < i < nx && 1 < j < ny
                    left_slot = _cell_advection_x_slot(b, i - 1, j, nx, ny)
                    right_slot = _cell_advection_x_slot(b, i, j, nx, ny)
                    down_slot = _cell_advection_y_slot(b, i, j - 1, nx, ny)
                    up_slot = _cell_advection_y_slot(b, i, j, nx, ny)
                    left = i == 2 ? workspace.x_flux[left_slot] : scratch[base + i - 1]
                    right = i == nx - 1 ? workspace.x_flux[right_slot] : scratch[base + i]
                    down = _cell_advection_stored_y_row(j - 1, ny, columns_per_tile) ?
                        workspace.y_flux[down_slot] :
                        scratch[_cell_advection_rolling_y_start(base, nx, j - 1) + i - 1]
                    up = _cell_advection_stored_y_row(j, ny, columns_per_tile) ?
                        workspace.y_flux[up_slot] :
                        scratch[_cell_advection_rolling_y_start(base, nx, j) + i - 1]
                    _cell_advection_ordered_sum(left, -right, down, -up,
                                                workspace.interior_order[cell], T)
                else
                    _cell_advection_residual_sum(workspace, cell, true, T)
                end
                # Keep the T-valued integrated sum until every flux has read
                # the unchanged scalar. Division can promote beyond T.
                increment[i + h, j + h] = integrated
            end
        end
    end
    return nothing
end

function _cell_advection_apply_increments!(scalar::CellField, workspace::AdvectionWorkspace)
    mesh = scalar.mesh
    nx, ny = mesh.block_size
    h = mesh.halo
    _parallel_block_columns(mesh) do b, first_j, last_j
        values = @inbounds scalar.blocks[b]
        increment = @inbounds workspace.increment.blocks[b]
        volume = mesh.blocks[b].volume
        @inbounds for j in first_j:last_j, i in 1:nx
            integrated = increment[i + h, j + h]
            change = integrated / volume
            increment[i + h, j + h] = change
            values[i + h, j + h] += change
        end
    end
    return nothing
end

"""

const ADVECT = raw"""
function advect!(
        scalar::CellField{T}, velocity::FaceVelocity, dt,
        workspace::AdvectionWorkspace{T};
        boundary::AbstractBoundaryCondition = ZeroGradient(), time = 0
    ) where {T}
    scalar.mesh === velocity.mesh === workspace.increment.mesh ||
        throw(ArgumentError("scalar, velocity, and workspace must share a mesh"))
    _validate_advection_workspace_generation(workspace)
    mesh = scalar.mesh
    @_parallel for index in eachindex(workspace.interior_faces)
        face = workspace.interior_faces[index]
        @inbounds speed = velocity.interior[face.face_id]
        state = _cell_advection_upwind_state(scalar, face, speed)
        @inbounds workspace.interior_flux[index] =
            dt * speed * state * face.area
    end
    _cell_advection_edge_seam_fluxes!(scalar, velocity, dt, workspace)
    @_parallel for index in 1:length(mesh.boundary_faces)
        face = mesh.boundary_faces[index]
        @inbounds coordinate_speed = velocity.boundary[index]
        outward_speed = face.side * coordinate_speed
        state = outward_speed >= 0 ? _get(scalar, face.cell) :
            _inflow_value(boundary, scalar, face, time)
        @inbounds workspace.boundary_flux[index] =
            dt * outward_speed * state * face.area
    end
    _cell_advection_rolling_increments!(scalar, velocity, dt, workspace)
    _cell_advection_apply_increments!(scalar, workspace)
    return scalar
end
"""

function candidate_source(original)
    result = replace_once(original,
        "    face_kinds::Vector{UInt8}\n    generation::UInt\n",
        "    face_kinds::Vector{UInt8}\n" *
        "    rolling_flux::Vector{T}\n" *
        "    rolling_columns_per_tile::Int\n" *
        "    rolling_tiles_per_block::Int\n" *
        "    generation::UInt\n")
    result = replace_once(result, "AdvectionWorkspace(field::CellField) =\n", 
                          LAYOUT * "AdvectionWorkspace(field::CellField) =\n")
    result = replace_once(result,
        "    offsets, permutation = _grouped_output_schedule(targets, ncells(mesh))\n" *
        "    return AdvectionWorkspace(\n",
        "    offsets, permutation = _grouped_output_schedule(targets, ncells(mesh))\n" *
        "    columns_per_tile, tiles_per_block = _cell_advection_rolling_layout(mesh)\n" *
        "    rolling_flux = zeros(T, nblocks(mesh) * tiles_per_block * (3nx - 1))\n" *
        "    return AdvectionWorkspace(\n")
    result = replace_once(result,
        "        face_kinds[permutation], mesh.topology_cache.generation,\n",
        "        face_kinds[permutation], rolling_flux, columns_per_tile,\n" *
        "        tiles_per_block, mesh.topology_cache.generation,\n")
    result = replace_once(result,
        "\nfunction _cell_advection_regular_rates!(",
        "\n" * ROLLING_HELPERS * "function _cell_advection_regular_rates!(")
    advection_anchor = findfirst("function advect!(\n        scalar::CellField", result)
    advection_anchor === nothing && error("ordinary advection anchor missing")
    timestep_anchor = findnext("\n\"\"\"\n    stable_timestep(", result,
                               last(advection_anchor) + 1)
    timestep_anchor === nothing && error("stable-timestep anchor missing")
    before = result[firstindex(result):prevind(result, first(advection_anchor))]
    after = result[first(timestep_anchor):lastindex(result)]
    return before * ADVECT * after
end

function make_probe(source, output)
    source, output = abspath(source), abspath(output)
    !ispath(output) || error("probe output already exists: $output")
    clean = joinpath(source, RELATIVE)
    isfile(clean) && sha256file(clean) == EXPECTED_CLEAN_SHA256 ||
        error("source is not the frozen narrow-ID implementation")
    candidate = candidate_source(read(clean, String))
    mkpath(output)
    cp(joinpath(source, "src"), joinpath(output, "src"))
    target = joinpath(output, RELATIVE)
    write(target, candidate)
    diff = run_capture(["diff", "-u", "--label", RELATIVE,
                        "--label", RELATIVE * ".rolling-flux", clean, target])
    diff.exitcode == 1 || error("expected copied-source patch")
    patch = joinpath(output, "rolling-flux.patch")
    write(patch, diff.stdout)
    receipt = Dict{String,Any}(
        "schema" => 1, "source" => source, "output" => output,
        "clean_source_sha256" => sha256file(clean),
        "candidate_source_sha256" => sha256file(target),
        "clean_tree_sha256" => source_hash(source),
        "candidate_tree_sha256" => source_hash(output),
        "generator_sha256" => sha256file(@__FILE__),
        "patch_sha256" => sha256file(patch),
        "scope" => "ordinary upwind advection on all AMR leaves; cut-cell and CFL unchanged",
        "schedule" => "perimeter and tile-seam faces once; deterministic tile scratch; increment then update",
    )
    writejson(joinpath(output, "rolling-flux-receipt.json"), receipt)
    return receipt
end

function main(args=ARGS)
    if any(arg -> arg in ("-h", "--help"), args)
        println("usage: make_probe.jl --source FROZEN_SOURCE --output NEW_DIRECTORY")
        return nothing
    end
    length(args) == 4 || error("usage: --source SOURCE --output NEW_DIRECTORY")
    options = Dict(args[i] => args[i + 1] for i in 1:2:length(args))
    Set(keys(options)) == Set(("--source", "--output")) ||
        error("expected only --source and --output")
    receipt = make_probe(options["--source"], options["--output"])
    println("created copied rolling-flux probe: ", receipt["output"])
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end

end # module
