"""Solid of one embedded geometry, fixed for a topology snapshot."""
struct SolidSnapshot{T}
    # Cells without open fluid volume.
    full::BitVector
    # Partially solid cells (sorted cell indices) and their solid planes
    # `(n_x, n_y, d)` in local cell coordinates: solid where `n⋅ξ <= d`.
    cut::Vector{Int32}
    planes::Vector{NTuple{3, T}}
    revision::UInt
    # Physical, filled PLIC solid for vector overlays.
    polygons::PLICSolidGeometry{T}
end

"""
    TopologySnapshot

Immutable leaf-block table of one mesh topology, shared by all frames captured
while the leaf set is unchanged. Holds block keys, origins and spacings, the
domain, and per-axis face tiles: every leaf block owns an `(nx+1) × 2ny` x-face
tile and a `2nx × (ny+1)` y-face tile whose texels store the index of the
staggered value drawn there (`0` = none). A tile column is a face line of the
block; its two subrows per cell row hold the two fine faces of a coarse–fine
interface. Embedded solids are stored as sparse PLIC planes.
"""
struct TopologySnapshot{T}
    id::Int
    generation::UInt
    keys::Vector{BlockKey}
    origins::Vector{NTuple{2, T}}
    spacings::Vector{NTuple{2, T}}
    block_size::NTuple{2, Int}
    base_mesh::NTuple{2, Int}
    origin::NTuple{2, T}
    extent::NTuple{2, T}
    periodic::NTuple{2, Bool}
    face_counts::NTuple{2, Int}
    face_tiles::NTuple{2, Vector{Int32}}
    solids::Vector{SolidSnapshot{T}}
    block_index::Dict{BlockKey, Int32}
    max_level::Int
end

nblocks(topology::TopologySnapshot) = length(topology.keys)
ncells(topology::TopologySnapshot) = length(topology.keys) * prod(topology.block_size)

_tile_shape(nx, ny, axis) = axis == 1 ? (nx + 1, 2ny) : (2nx, ny + 1)

# Mark the tile texels of one cell side. `line` is the face line inside the
# block (0:nx or 0:ny); the face covers both tangential subcells of the cell
# unless it is one of the two fine faces against a coarser cell.
function _mark_face_tile!(tiles, mesh, cell::CellRef, face, axis, line, value)
    nx, ny = mesh.block_size
    columns, rows = _tile_shape(nx, ny, axis)
    tangent = 3 - axis
    spacing = mesh.blocks[cell.block].spacing[tangent]
    position = axis == 1 ? cell.j : cell.i
    first_sub = 2 * (position - 1)
    subs = if face.area < 0.75 * spacing
        center = cellcenter(mesh, cell.block, cell.i, cell.j)[tangent]
        face.center[tangent] < center ? (first_sub,) : (first_sub + 1,)
    else
        (first_sub, first_sub + 1)
    end
    base = (cell.block - 1) * columns * rows
    for sub in subs
        index = axis == 1 ? base + sub * columns + line + 1 :
            base + line * columns + sub + 1
        # Both cells of a face inside one block share its texels.
        tiles[index] in (0, value) || throw(
            ErrorException("overlapping staggered tiles in block $(cell.block)"),
        )
        tiles[index] = value
    end
    return tiles
end

function _face_tiles(mesh::BlockTreeMesh, axis::Int)
    nx, ny = mesh.block_size
    columns, rows = _tile_shape(nx, ny, axis)
    tiles = zeros(Int32, nblocks(mesh) * columns * rows)
    value = Int32(0)
    for face in mesh.faces
        face.axis == axis || continue
        value += Int32(1)
        minus_line = axis == 1 ? face.minus.i : face.minus.j
        plus_line = (axis == 1 ? face.plus.i : face.plus.j) - 1
        _mark_face_tile!(tiles, mesh, face.minus, face, axis, minus_line, value)
        _mark_face_tile!(tiles, mesh, face.plus, face, axis, plus_line, value)
    end
    for face in mesh.boundary_faces
        face.axis == axis || continue
        value += Int32(1)
        line = face.side < 0 ? 0 : (axis == 1 ? nx : ny)
        _mark_face_tile!(tiles, mesh, face.cell, face, axis, line, value)
    end
    return tiles, Int(value)
end

function _solid_snapshot(geometry::GeometryField{T}) where {T}
    mesh = geometry.mesh
    nx, ny = mesh.block_size
    full = falses(ncells(mesh))
    cut = Int32[]
    planes = NTuple{3, T}[]
    solid = geometry.solid
    for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        ref = CellRef(block, i, j)
        index = _cell_index(ref, nx, ny)
        fraction = _get(solid.fraction, ref)
        if fraction >= one(T) || iszero(_get(geometry.open_fraction, ref))
            full[index] = true
        elseif fraction > zero(T)
            push!(cut, Int32(index))
            push!(planes, (
                _get(solid.normal_x, ref), _get(solid.normal_y, ref),
                _get(solid.intercept, ref),
            ))
        end
    end
    return SolidSnapshot{T}(
        full, cut, planes, geometry.revision, plic_solid_geometry(geometry),
    )
end

function _topology_snapshot(id::Int, mesh::BlockTreeMesh{T}, geometries) where {T}
    keys = BlockKey[block.key for block in mesh.blocks]
    x_tiles, x_count = _face_tiles(mesh, 1)
    y_tiles, y_count = _face_tiles(mesh, 2)
    solids = SolidSnapshot{T}[_solid_snapshot(geometry) for geometry in geometries]
    index = Dict{BlockKey, Int32}(key => Int32(b) for (b, key) in pairs(keys))
    return TopologySnapshot{T}(
        id, mesh.topology_cache.generation, keys,
        NTuple{2, T}[block.origin for block in mesh.blocks],
        NTuple{2, T}[block.spacing for block in mesh.blocks],
        mesh.block_size, mesh.base_mesh, mesh.origin, mesh.extent,
        mesh.periodic, (x_count, y_count), (x_tiles, y_tiles), solids, index,
        maximum(key -> key.level, keys; init = 0),
    )
end

# A snapshot remains valid while the leaf set (the mesh topology-cache
# generation, which changes exactly on schedule rebuilds) and every solid
# reconstruction revision are unchanged.
function _topology_current(topology::TopologySnapshot, mesh, geometries)
    topology.generation == mesh.topology_cache.generation || return false
    length(topology.solids) == length(geometries) || return false
    for (solid, geometry) in zip(topology.solids, geometries)
        solid.revision == geometry.revision || return false
    end
    return true
end

function _topology_bytes(topology::TopologySnapshot)
    bytes = sizeof(topology.keys) + sizeof(topology.origins) +
        sizeof(topology.spacings) + sizeof(topology.face_tiles[1]) +
        sizeof(topology.face_tiles[2])
    for solid in topology.solids
        bytes += sizeof(solid.full.chunks) + sizeof(solid.cut) +
            sizeof(solid.planes) + sizeof(solid.polygons.points) +
            sizeof(solid.polygons.triangles)
    end
    return bytes
end

"""
    locate(topology_or_frame, (x, y))::Union{Nothing, CellRef}

Return the leaf cell containing a physical point, or `nothing` outside the
domain. Periodic directions wrap.
"""
function locate(topology::TopologySnapshot, point)
    x, y = Float64(point[1]), Float64(point[2])
    local_point = (x - topology.origin[1], y - topology.origin[2])
    wrapped = ntuple(2) do axis
        coordinate = local_point[axis]
        extent = Float64(topology.extent[axis])
        topology.periodic[axis] ? mod(coordinate, extent) : coordinate
    end
    all(axis -> 0 <= wrapped[axis] < topology.extent[axis], 1:2) || return nothing
    nx, ny = topology.block_size
    for level in 0:topology.max_level
        counts = (topology.base_mesh .÷ topology.block_size) .* (1 << level)
        i = floor(Int, wrapped[1] / topology.extent[1] * counts[1])
        j = floor(Int, wrapped[2] / topology.extent[2] * counts[2])
        block = get(topology.block_index, BlockKey(level, i, j), Int32(0))
        iszero(block) && continue
        origin = topology.origins[block]
        spacing = topology.spacings[block]
        ci = clamp(floor(Int, (wrapped[1] + topology.origin[1] - origin[1]) / spacing[1]), 0, nx - 1)
        cj = clamp(floor(Int, (wrapped[2] + topology.origin[2] - origin[2]) / spacing[2]), 0, ny - 1)
        return CellRef(Int(block), ci + 1, cj + 1)
    end
    return nothing
end
