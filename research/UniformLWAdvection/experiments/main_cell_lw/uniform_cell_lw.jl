"""One periodic halo assignment, with positions in haloed block matrices."""
struct _UniformLWHaloCopy
    destination_block::Int
    destination_i::Int
    destination_j::Int
    source_block::Int
    source_i::Int
    source_j::Int
end

"""
    UniformCellLWWorkspace(field, velocity; tile_columns=8, threshold=1024)

Reusable workspace for Float64, uniform, periodic, unit-square cell averages.
The carrier is spatially constant and is registered from the dense owned-face
`velocity` arrays at construction. Refresh it with `set_uniform_lw_velocity!`.
Direct mutation of the velocity arrays is outside this workspace's contract.
A workspace belongs to one in-flight solve; separate workspaces may run together.
"""
mutable struct UniformCellLWWorkspace{M <: BlockTreeMesh}
    mesh::M
    velocity::FaceVelocity{Float64,M}
    first::CellField{Float64,M}
    second::CellField{Float64,M}
    xfaces::Vector{Matrix{Float64}}
    yfaces::Vector{Matrix{Float64}}
    xhalos::Vector{_UniformLWHaloCopy}
    yhalos::Vector{_UniformLWHaloCopy}
    tiles::Vector{NTuple{3,Int}}
    generation::UInt
    nx::Int
    ny::Int
    u::Float64
    v::Float64
    threshold::Int
    record_workers::Bool
    worker_ids::Vector{Int}
    halo_worker_ids::Vector{Int}
end

function _uniform_lw_constant_speeds(velocity::FaceVelocity)
    mesh = velocity.mesh
    length(velocity.interior) == length(mesh.faces) &&
        length(velocity.boundary) == length(mesh.boundary_faces) ||
        throw(DimensionMismatch("velocity and face schedule differ"))
    u = NaN
    v = NaN
    for (index, face) in pairs(mesh.faces)
        speed = velocity.interior[index]
        isfinite(speed) || throw(ArgumentError("LW carrier speed must be finite"))
        if face.axis == 1
            isnan(u) ? (u = speed) : speed == u ||
                throw(ArgumentError("LW carrier must be spatially constant"))
        else
            isnan(v) ? (v = speed) : speed == v ||
                throw(ArgumentError("LW carrier must be spatially constant"))
        end
    end
    return u, v
end

function UniformCellLWWorkspace(
        field::CellField{Float64,M}, velocity::FaceVelocity{Float64,M};
        tile_columns::Integer=8, threshold::Integer=1024
    ) where {M <: BlockTreeMesh}
    mesh = field.mesh
    velocity.mesh === mesh || throw(ArgumentError("field and velocity must share a mesh"))
    tile_columns > 0 || throw(ArgumentError("tile_columns must be positive"))
    threshold >= 0 || throw(ArgumentError("threshold must be nonnegative"))
    all(mesh.periodic) && isempty(mesh.boundary_faces) ||
        throw(ArgumentError("uniform LW requires periodic boundaries"))
    mesh.origin == (0.0, 0.0) && mesh.extent == (1.0, 1.0) ||
        throw(ArgumentError("uniform LW currently requires the unit square"))
    # Certify every cell has exactly one owned positive and negative face in
    # each direction, and reject mixed refinement levels.
    _, xplus, _, yplus = _uniform_periodic_cell_faces(mesh)
    level = first(mesh.blocks).key.level
    nx = mesh.base_mesh[1] * (1 << level)
    ny = mesh.base_mesh[2] * (1 << level)
    nx * ny == ncells(mesh) || throw(ArgumentError("uniform grid is incomplete"))
    u, v = _uniform_lw_constant_speeds(velocity)
    h = mesh.halo
    bx, by = mesh.block_size
    xfaces = [fill(u, bx + 2h, by + 2h) for _ in 1:nblocks(mesh)]
    yfaces = [fill(v, bx + 2h, by + 2h) for _ in 1:nblocks(mesh)]
    for b in 1:nblocks(mesh), j in 1:by, i in 1:bx
        cell = (b - 1) * bx * by + (j - 1) * bx + i
        xfaces[b][i + h, j + h] = velocity.interior[xplus[cell]]
        yfaces[b][i + h, j + h] = velocity.interior[yplus[cell]]
    end
    xhalos = _UniformLWHaloCopy[]
    yhalos = _UniformLWHaloCopy[]
    for entry in mesh.halo_entries
        destination = entry.destination
        i, j = destination.i, destination.j
        # The stencil reads side halos only; corner entries are unnecessary.
        axis = (i == 0 || i == bx + 1) && 1 <= j <= by ? 1 :
            ((j == 0 || j == by + 1) && 1 <= i <= bx ? 2 : 0)
        axis == 0 && continue
        entry.physical_boundary && throw(ArgumentError("nonperiodic halo"))
        entry.donor_count == 1 || throw(ArgumentError("nonuniform halo donor"))
        source = mesh.halo_donors[entry.first_donor]
        copy = _UniformLWHaloCopy(destination.block, i + mesh.halo, j + mesh.halo,
            source.block, source.i + mesh.halo, source.j + mesh.halo)
        push!(axis == 1 ? xhalos : yhalos, copy)
    end
    length(xhalos) == 2by * nblocks(mesh) &&
        length(yhalos) == 2bx * nblocks(mesh) ||
        throw(ArgumentError("uniform halo schedule is incomplete"))
    tiles = NTuple{3,Int}[]
    for b in 1:nblocks(mesh), j in 1:tile_columns:by
        push!(tiles, (b, j, min(by, j + tile_columns - 1)))
    end
    return UniformCellLWWorkspace(mesh, velocity, CellField(mesh, Float64),
        CellField(mesh, Float64), xfaces, yfaces, xhalos, yhalos, tiles,
        mesh.topology_cache.generation, nx, ny, u, v, Int(threshold),
        false, zeros(Int, 3length(tiles)),
        zeros(Int, 3max(length(xhalos), length(yhalos))))
end

"""Refresh the registered spatially constant carrier and its owned face arrays."""
function set_uniform_lw_velocity!(workspace::UniformCellLWWorkspace,
        velocity::FaceVelocity, u::Real, v::Real)
    velocity === workspace.velocity || throw(ArgumentError("carrier identity differs"))
    workspace.generation == workspace.mesh.topology_cache.generation ||
        throw(ArgumentError("uniform LW workspace has stale mesh topology"))
    a, b = Float64(u), Float64(v)
    isfinite(a) && isfinite(b) || throw(ArgumentError("carrier speed must be finite"))
    for (index, face) in pairs(workspace.mesh.faces)
        velocity.interior[index] = face.axis == 1 ? a : b
    end
    for block in workspace.xfaces
        fill!(block, a)
    end
    for block in workspace.yfaces
        fill!(block, b)
    end
    workspace.u, workspace.v = a, b
    return workspace
end

"""Enable optional untimed participation recording for the next LW step."""
function track_uniform_lw_workers!(workspace::UniformCellLWWorkspace, enabled::Bool=true)
    fill!(workspace.worker_ids, 0)
    fill!(workspace.halo_worker_ids, 0)
    workspace.record_workers = enabled
    return workspace
end

"""Julia default-pool thread IDs that executed tiles in the recorded LW step."""
active_uniform_lw_workers(workspace::UniformCellLWWorkspace) =
    sort!(unique(filter(!iszero, workspace.worker_ids)))

active_uniform_lw_halo_workers(workspace::UniformCellLWWorkspace) =
    sort!(unique(filter(!iszero, workspace.halo_worker_ids)))

@inline _uniform_lw_flux(left, right, speed, ratio) =
    speed * ((left + right) * 0.5) -
        ((speed * speed) * ratio * 0.5) * (right - left)

function _uniform_lw_halos!(field::CellField, copies::Vector{_UniformLWHaloCopy},
        parallel::Bool, workspace::UniformCellLWWorkspace, pass::Int)
    offset = (pass - 1) * max(length(workspace.xhalos), length(workspace.yhalos))
    if parallel
        Polyester.@batch per=thread minbatch=1 for index in eachindex(copies)
            copy = @inbounds copies[index]
            if workspace.record_workers
                @inbounds workspace.halo_worker_ids[offset + index] = Threads.threadid()
            end
            @inbounds field.blocks[copy.destination_block][copy.destination_i, copy.destination_j] =
                field.blocks[copy.source_block][copy.source_i, copy.source_j]
        end
    else
        for index in eachindex(copies)
            copy = @inbounds copies[index]
            if workspace.record_workers
                @inbounds workspace.halo_worker_ids[offset + index] = Threads.threadid()
            end
            @inbounds field.blocks[copy.destination_block][copy.destination_i, copy.destination_j] =
                field.blocks[copy.source_block][copy.source_i, copy.source_j]
        end
    end
    return nothing
end

function _uniform_lw_tile!(out::CellField, input::CellField, axis::Int,
        faces::Vector{Matrix{Float64}}, ratio::Float64, tile::NTuple{3,Int})
    b, first_j, last_j = tile
    q = @inbounds input.blocks[b]
    result = @inbounds out.blocks[b]
    carrier = @inbounds faces[b]
    h = input.mesh.halo
    bx = input.mesh.block_size[1]
    if axis == 1
        for j in first_j:last_j
            @inbounds @simd for i in 1:bx
                ii, jj = i + h, j + h
                qi = q[ii, jj]
                fr = _uniform_lw_flux(qi, q[ii + 1, jj], carrier[ii, jj], ratio)
                fl = _uniform_lw_flux(q[ii - 1, jj], qi, carrier[ii - 1, jj], ratio)
                result[ii, jj] = qi - ratio * (fr - fl)
            end
        end
    else
        for j in first_j:last_j
            @inbounds @simd for i in 1:bx
                ii, jj = i + h, j + h
                qi = q[ii, jj]
                ft = _uniform_lw_flux(qi, q[ii, jj + 1], carrier[ii, jj], ratio)
                fb = _uniform_lw_flux(q[ii, jj - 1], qi, carrier[ii, jj - 1], ratio)
                result[ii, jj] = qi - ratio * (ft - fb)
            end
        end
    end
    return nothing
end

function _uniform_lw_sweep!(out, input, axis, faces, ratio,
        workspace::UniformCellLWWorkspace, pass)
    parallel = Threads.nthreads(:default) > 1 &&
        workspace.nx * workspace.ny >= workspace.threshold
    _uniform_lw_halos!(input, axis == 1 ? workspace.xhalos : workspace.yhalos,
        parallel, workspace, pass)
    tiles = workspace.tiles
    if workspace.record_workers
        offset = (pass - 1) * length(tiles)
        if parallel
            Polyester.@batch per=thread minbatch=1 for k in eachindex(tiles)
                @inbounds workspace.worker_ids[offset + k] = Threads.threadid()
                _uniform_lw_tile!(out, input, axis, faces, ratio, @inbounds tiles[k])
            end
        else
            for k in eachindex(tiles)
                @inbounds workspace.worker_ids[offset + k] = Threads.threadid()
                _uniform_lw_tile!(out, input, axis, faces, ratio, @inbounds tiles[k])
            end
        end
    elseif parallel
        Polyester.@batch per=thread minbatch=1 for k in eachindex(tiles)
            _uniform_lw_tile!(out, input, axis, faces, ratio, @inbounds tiles[k])
        end
    else
        for k in eachindex(tiles)
            _uniform_lw_tile!(out, input, axis, faces, ratio, @inbounds tiles[k])
        end
    end
    return nothing
end

"""
    advect!(field, velocity, dt, workspace::UniformCellLWWorkspace)

Advance constant-carrier periodic cell averages with the scalar reference's
`x(dt/2), y(dt), x(dt/2)` Lax–Wendroff composition. The registered carrier
must be refreshed through `set_uniform_lw_velocity!` when it changes.
"""
function advect!(field::CellField{Float64}, velocity::FaceVelocity{Float64},
        dt::Real, workspace::UniformCellLWWorkspace)
    mesh = workspace.mesh
    field.mesh === mesh && velocity === workspace.velocity ||
        throw(ArgumentError("LW field, velocity, and workspace must share their mesh/carrier"))
    workspace.generation == mesh.topology_cache.generation ||
        throw(ArgumentError("uniform LW workspace has stale mesh topology"))
    length(field.blocks) == nblocks(mesh) &&
        length(workspace.first.blocks) == nblocks(mesh) &&
        length(workspace.second.blocks) == nblocks(mesh) ||
        throw(DimensionMismatch("LW field or workspace block count differs"))
    field === workspace.first || field === workspace.second ||
        workspace.first === workspace.second ?
        throw(ArgumentError("LW field and intermediate buffers must be distinct")) : nothing
    delta = Float64(dt)
    isfinite(delta) && delta >= 0 ||
        throw(ArgumentError("LW timestep must be finite and nonnegative"))
    abs(workspace.u) * delta * workspace.nx * 0.5 <= 1 &&
        abs(workspace.v) * delta * workspace.ny <= 1 ||
        throw(ArgumentError("directional LW Courant number exceeds one"))
    rx = delta * workspace.nx * 0.5
    ry = delta * workspace.ny
    _uniform_lw_sweep!(workspace.first, field, 1, workspace.xfaces, rx, workspace, 1)
    _uniform_lw_sweep!(workspace.second, workspace.first, 2, workspace.yfaces, ry,
        workspace, 2)
    _uniform_lw_sweep!(field, workspace.second, 1, workspace.xfaces, rx, workspace, 3)
    return field
end
