"""Dyadic quadtree address of a block, using zero-based coordinates."""
struct BlockKey
    level::Int
    i::Int
    j::Int
end

struct TreeNodeData
    key::BlockKey
end

"""Geometry and cell metric data copied from one active tree leaf."""
struct LeafBlock{T}
    key::BlockKey
    origin::NTuple{2, T}
    extent::NTuple{2, T}
    spacing::NTuple{2, T}
    volume::T
end

"""Local `(block, i, j)` reference to an owned cell."""
struct CellRef
    block::Int
    i::Int
    j::Int
end

@inline function _cell_index(ref::CellRef, nx::Int, ny::Int)
    return (ref.block - 1) * nx * ny + (ref.j - 1) * nx + ref.i
end

struct InteriorFace{T}
    minus::CellRef
    plus::CellRef
    axis::Int
    center::NTuple{2, T}
    area::T
end

struct BoundaryFace{T}
    cell::CellRef
    axis::Int
    side::Int
    center::NTuple{2, T}
    area::T
end

struct HaloEntry{T}
    destination::CellRef
    first_donor::Int
    donor_count::Int
    center::NTuple{2, T}
    physical_boundary::Bool
end

"""
    MeshAdaptationOptions(; update_interval=1, coarsen_delay=4,
                          minimum_level=0)

Frequency and hysteresis controls for dynamic mesh adaptation.
`update_interval` is the minimum number of simulation steps between adaptation
attempts. A newly created fine block cannot be coarsened for `coarsen_delay`
steps. `minimum_level` protects the underlying base mesh from coarsening.
"""
struct MeshAdaptationOptions
    update_interval::Int
    coarsen_delay::Int
    minimum_level::Int

    function MeshAdaptationOptions(;
            update_interval = 1, coarsen_delay = 4,
            minimum_level = 0
        )
        update_interval >= 1 || throw(
            ArgumentError(
                "mesh adaptation update_interval must be positive"
            )
        )
        coarsen_delay >= 0 || throw(
            ArgumentError(
                "mesh adaptation coarsen_delay must be nonnegative"
            )
        )
        minimum_level >= 0 || throw(
            ArgumentError(
                "mesh adaptation minimum_level must be nonnegative"
            )
        )
        return new(Int(update_interval), Int(coarsen_delay), Int(minimum_level))
    end
end

mutable struct MeshAdaptationState
    options::MeshAdaptationOptions
    last_update_step::Union{Nothing, Int}
    last_refined_step::Dict{BlockKey, Int}
    completed_steps::Int
end

MeshAdaptationState(options::MeshAdaptationOptions) =
    MeshAdaptationState(options, nothing, Dict{BlockKey, Int}(), 0)

"""Rectangular collection of permanent level-zero quadtree roots."""
struct RootGrid{T, Tree}
    roots::Vector{Tree}
    layout::NTuple{2, Int}
    origin::NTuple{2, T}
    extent::NTuple{2, T}
    root_extent::NTuple{2, T}
end

"""Read-only cell geometry/connectivity for one face-schedule generation."""
struct MeshCellConnectivity{T}
    centers::Vector{NTuple{2, T}}
    offsets::Vector{Int}
    neighbours::Vector{Int}
end

struct ThreePointStencil{T}
    upstream::Int
    ratio_scale::T
end

struct DualFace{T}
    minus::Int
    plus::Int
    flux_a::Int
    flux_b::Int
    axis::Int
    area::T
    distance::T
    minus_stencil::ThreePointStencil{T}
    plus_stencil::ThreePointStencil{T}
end

abstract type AbstractDualAdvectionSchedule{T} end

struct UniformDualAdvectionSchedule{T} <: AbstractDualAdvectionSchedule{T}
    faces::Vector{DualFace{T}}
    volume::T
end

struct AMRDualFace{T}
    minus::Int
    plus::Int
    first_weight::Int
    weight_count::Int
    axis::Int
    refinement_interface::Bool
    minus_stencil::ThreePointStencil{T}
    plus_stencil::ThreePointStencil{T}
end

struct AMRDualAdvectionSchedule{T} <: AbstractDualAdvectionSchedule{T}
    faces::Vector{AMRDualFace{T}}
    flux_indices::Vector{Int}
    flux_weights::Vector{T}
    volumes::Vector{T}
end

struct BoundaryDualFace{T}
    degree_of_freedom::Int
    boundary_index::Int
    boundary_local_index::Int
    opposite_flux::Int
    wall_axis::Int
    side::Int
    measure::T
    normal_component::Bool
end

"""Mesh-only phase transport graph; numerical fluxes/BCs are not shared."""
struct PhaseTransportConnectivity{Axis, T, S <: AbstractDualAdvectionSchedule{T}}
    generation::UInt
    schedule::S
    dual_offsets::Vector{Int}
    dual_faces::Vector{Int}
    dual_signs::Vector{Int8}
    boundary_faces::Vector{BoundaryDualFace{T}}
    interior_indices::Vector{Int}
    boundary_indices::Vector{Int}
    centers::Vector{NTuple{2, T}}
end

# Native precision and the two supported flow precisions can share mesh-only
# graphs. An uncommon foreign numeric type is constructed without caching.
const _PhaseTransportCacheEntry{Axis, T} = Union{
    Nothing,
    PhaseTransportConnectivity{Axis, T, UniformDualAdvectionSchedule{T}},
    PhaseTransportConnectivity{Axis, T, AMRDualAdvectionSchedule{T}},
    PhaseTransportConnectivity{Axis, Float32, UniformDualAdvectionSchedule{Float32}},
    PhaseTransportConnectivity{Axis, Float32, AMRDualAdvectionSchedule{Float32}},
    PhaseTransportConnectivity{Axis, Float64, UniformDualAdvectionSchedule{Float64}},
    PhaseTransportConnectivity{Axis, Float64, AMRDualAdvectionSchedule{Float64}},
}

"""Lazy, mesh-only data shared by otherwise independent numerical workspaces.

The epoch changes when face/halo schedules change, not when the quadtree is
first edited: AMR still needs the old connectivity to stage its old values.
Phase transport entries have concrete schedule types, and their arrays are
read-only. Geometry-dependent and evolving phase data are not stored here.
"""
mutable struct MeshTopologyCache{T}
    identity::Base.RefValue{Nothing}
    generation::UInt
    cells::Union{Nothing, MeshCellConnectivity{T}}
    phase_transport::Tuple{
        _PhaseTransportCacheEntry{1, T}, _PhaseTransportCacheEntry{2, T},
    }
    # Indices of the physical-boundary halo entries, in schedule order. Halo
    # filling reads this instead of scanning the whole schedule.
    physical_halos::Union{Nothing, Vector{Int}}

    # Only the explicit type-parameter constructor is meaningful when the
    # optional connectivity entries are all empty.
    MeshTopologyCache{T}() where {T} =
        new{T}(
        Ref{Nothing}(nothing), zero(UInt), nothing, (nothing, nothing),
        nothing,
    )
end

MeshTopologyCache(::Type{T}) where {T} = MeshTopologyCache{T}()

struct BlockTreeMesh{T, Roots}
    root::Roots
    blocks::Vector{LeafBlock{T}}
    base_mesh::NTuple{2, Int}
    block_size::NTuple{2, Int}
    halo::Int
    origin::NTuple{2, T}
    extent::NTuple{2, T}
    periodic::NTuple{2, Bool}
    max_level::Int
    faces::Vector{InteriorFace{T}}
    # Mesh-only interior staggered control-volume measures, in global-face
    # order. Cut/open dual volumes belong to GeometryField instead.
    face_dual_volumes::Vector{T}
    boundary_faces::Vector{BoundaryFace{T}}
    halo_entries::Vector{HaloEntry{T}}
    halo_donors::Vector{CellRef}
    halo_weights::Vector{T}
    adaptation::MeshAdaptationState
    topology_cache::MeshTopologyCache{T}
end

function _invalidate_mesh_topology_cache!(mesh::BlockTreeMesh)
    cache = mesh.topology_cache
    cache.generation += one(UInt)
    cache.cells = nothing
    cache.phase_transport = (nothing, nothing)
    cache.physical_halos = nothing
    return mesh
end

"""Indices of the physical-boundary halo entries, in schedule order.

Precollected once per halo schedule so that `fill_halo!` need not scan the
whole schedule when no boundary callback asks for it. The returned vector is
shared with the mesh cache and must not be mutated.
"""
function _physical_halo_entries(mesh::BlockTreeMesh)
    cache = mesh.topology_cache
    entries = cache.physical_halos
    isnothing(entries) || return entries
    indices = Int[]
    for (index, entry) in pairs(mesh.halo_entries)
        entry.physical_boundary && push!(indices, index)
    end
    cache.physical_halos = indices
    return indices
end

function _mesh_cell_connectivity(mesh::BlockTreeMesh{T}) where {T}
    cache = mesh.topology_cache
    isnothing(cache.cells) || return cache.cells
    nx, ny = mesh.block_size
    count = ncells(mesh)
    centers = Vector{NTuple{2, T}}(undef, count)
    cursor = zeros(Int, count)
    index = 0
    for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        index += 1
        centers[index] = cellcenter(mesh, block, i, j)
    end
    for face in mesh.faces
        cursor[_cell_index(face.minus, nx, ny)] += 1
        cursor[_cell_index(face.plus, nx, ny)] += 1
    end
    offsets = _csr_offsets(cursor)
    copyto!(cursor, 1, offsets, 1, length(cursor))
    neighbours = Vector{Int}(undef, offsets[end] - 1)
    # Preserve the old face traversal order, including repeated neighbours
    # on small periodic meshes. This keeps extrapolation's donor ordering.
    for face in mesh.faces
        minus = _cell_index(face.minus, nx, ny)
        plus = _cell_index(face.plus, nx, ny)
        neighbours[cursor[minus]] = plus
        neighbours[cursor[plus]] = minus
        cursor[minus] += 1
        cursor[plus] += 1
    end
    cache.cells = MeshCellConnectivity(centers, offsets, neighbours)
    return cache.cells
end

"""Build uncut dual volumes without constructing an advection schedule."""
function _face_dual_volumes(
        blocks::Vector{LeafBlock{T}}, faces::Vector{InteriorFace{T}},
    ) where {T}
    volumes = Vector{T}(undef, length(faces))
    for (index, face) in pairs(faces)
        # Each side contributes half of its cell's thickness normal to the
        # face, multiplied by the (possibly AMR-split) face area.
        @inbounds volumes[index] = face.area * (
            blocks[face.minus.block].spacing[face.axis] +
                blocks[face.plus.block].spacing[face.axis]
        ) / 2
    end
    return volumes
end

struct RawSide
    lo::Int
    hi::Int
    cell::CellRef
end

# Face-transfer and multigrid schedules group collinear face segments in
# increasing tangential order.  Keep the searches on those topology indices
# logarithmic/local: a schedule query must not rescan every segment or every
# normal line in the mesh.
@inline _line_segment_lo(segment) = segment.lo
@inline _line_segment_hi(segment) = segment.hi
@inline _line_segment_lo(segment::Tuple) = segment[1]
@inline _line_segment_hi(segment::Tuple) = segment[2]

@inline function _first_line_segment_reaching(line, coordinate)
    lower = firstindex(line)
    upper = lastindex(line)
    result = upper + 1
    while lower <= upper
        middle = (lower + upper) >>> 1
        if _line_segment_hi(line[middle]) >= coordinate
            result = middle
            upper = middle - 1
        else
            lower = middle + 1
        end
    end
    return result
end

@inline function _first_overlapping_line_segment(line, lower_coordinate)
    index = _first_line_segment_reaching(line, lower_coordinate)
    while index <= lastindex(line) &&
            _line_segment_hi(line[index]) <= lower_coordinate
        index += 1
    end
    return index
end

@inline function _line_segment_at(line, coordinate)
    index = _first_line_segment_reaching(line, coordinate)
    return index <= lastindex(line) &&
        _line_segment_lo(line[index]) <= coordinate ? index : 0
end

@inline function _wrapped_lattice_delta(delta::Int, period::Int)
    return delta - div(delta, period, RoundNearest) * period
end

function _bracketing_line_segments(
        lines, coordinates::Vector{Int}, normal::Int, tangent,
        normal_period::Int, periodic::Bool,
    )
    lower_distance = typemin(Int)
    upper_distance = typemax(Int)
    lower_coordinate = lower_segment = 0
    upper_coordinate = upper_segment = 0
    isempty(coordinates) && return (
        lower_distance, lower_coordinate, lower_segment,
        upper_distance, upper_coordinate, upper_segment,
    )

    insertion = searchsortedfirst(coordinates, normal)
    number_of_lines = length(coordinates)
    for direction in (-1, 1)
        position = direction < 0 ? insertion - 1 : insertion
        for _ in 1:number_of_lines
            if periodic
                position < 1 && (position = number_of_lines)
                position > number_of_lines && (position = 1)
            elseif position < 1 || position > number_of_lines
                break
            end
            coordinate = coordinates[position]
            delta = coordinate - normal
            periodic && (delta = _wrapped_lattice_delta(delta, normal_period))
            if direction * delta > 0
                segment = _line_segment_at(lines[coordinate], tangent)
                if !iszero(segment)
                    if direction < 0
                        lower_distance = delta
                        lower_coordinate = coordinate
                        lower_segment = segment
                    else
                        upper_distance = delta
                        upper_coordinate = coordinate
                        upper_segment = segment
                    end
                    break
                end
            elseif periodic && !iszero(delta)
                # We have crossed the half-period at which the wrapped sign
                # changes.  Remaining lines belong to the opposite bracket.
                break
            end
            position += direction
        end
    end
    return (
        lower_distance, lower_coordinate, lower_segment,
        upper_distance, upper_coordinate, upper_segment,
    )
end

"""Return the number of active leaf blocks."""
@inline nblocks(mesh::BlockTreeMesh) = length(mesh.blocks)

"""Return the number of owned cells across all active blocks."""
@inline ncells(mesh::BlockTreeMesh) = prod(mesh.block_size) * nblocks(mesh)

"""Global number of Cartesian cells represented at refinement `level`."""
@inline _cells_at_level(mesh::BlockTreeMesh, level::Integer) =
    mesh.base_mesh .* (1 << level)

@inline _block_scale(mesh::BlockTreeMesh, level::Integer) =
    1 << (mesh.max_level - level)

@inline _spacing_at_level(mesh::BlockTreeMesh, level::Integer) =
    mesh.extent ./ _cells_at_level(mesh, level)

@inline _finest_lattice_size(mesh::BlockTreeMesh) =
    _cells_at_level(mesh, mesh.max_level)

# The cell references carry the topology; physical face coordinates are
# deliberately not rounded back into indices.
@inline function _cell_lattice_bounds(mesh, blocks, ref::CellRef, lattice_size)
    key = blocks[ref.block].key
    sx = lattice_size[1] ÷ (mesh.base_mesh[1] << key.level)
    sy = lattice_size[2] ÷ (mesh.base_mesh[2] << key.level)
    low = (
        (key.i * mesh.block_size[1] + ref.i - 1) * sx,
        (key.j * mesh.block_size[2] + ref.j - 1) * sy,
    )
    return low, (sx, sy)
end

@inline function _face_lattice_geometry(
        mesh::BlockTreeMesh, face::InteriorFace,
        lattice_size = _finest_lattice_size(mesh), blocks = mesh.blocks,
    )
    minus_low, minus_scale = _cell_lattice_bounds(mesh, blocks, face.minus, lattice_size)
    plus_low, plus_scale = _cell_lattice_bounds(mesh, blocks, face.plus, lattice_size)
    axis = face.axis
    tangent = 3 - axis
    normal = minus_low[axis] + minus_scale[axis]
    mesh.periodic[axis] && (normal = mod(normal, lattice_size[axis]))
    lo = max(minus_low[tangent], plus_low[tangent])
    hi = min(minus_low[tangent] + minus_scale[tangent], plus_low[tangent] + plus_scale[tangent])
    return normal, lo, hi
end

@inline function _face_lattice_geometry(
        mesh::BlockTreeMesh, face::BoundaryFace,
        lattice_size = _finest_lattice_size(mesh), blocks = mesh.blocks,
    )
    low, scale = _cell_lattice_bounds(mesh, blocks, face.cell, lattice_size)
    axis = face.axis
    tangent = 3 - axis
    normal = low[axis] + (face.side > 0 ? scale[axis] : 0)
    return normal, low[tangent], low[tangent] + scale[tangent]
end

function _child_data(parent, indices)
    key = parent.data.key
    ci, cj = indices
    return TreeNodeData(BlockKey(key.level + 1, 2key.i + ci - 1, 2key.j + cj - 1))
end

function _split!(node)
    RegionTrees.split!(node, _child_data)
    return node
end

function _collect_leaves!(leaves, node)
    if RegionTrees.isleaf(node)
        push!(leaves, node)
    else
        for child in RegionTrees.children(node)
            _collect_leaves!(leaves, child)
        end
    end
    return leaves
end

function _collect_leaves!(leaves, roots::RootGrid)
    for root in roots.roots
        _collect_leaves!(leaves, root)
    end
    return leaves
end

@inline _tree_node_type(root) = typeof(root)
@inline _tree_node_type(roots::RootGrid) = eltype(roots.roots)

"""Find the leaf containing an integer point on a common cell lattice."""
@inline function _findleaf_lattice(roots::RootGrid, point, lattice_size)
    root_width = lattice_size .÷ roots.layout
    i, j = point .÷ root_width
    node = roots.roots[j * roots.layout[1] + i + 1]
    while !RegionTrees.isleaf(node)
        level = node.data.key.level + 1
        child_width = root_width .÷ (1 << level)
        ci = mod(fld(point[1], child_width[1]), 2) + 1
        cj = mod(fld(point[2], child_width[2]), 2) + 1
        node = node[ci, cj]
    end
    return node
end

@inline function _cellref_at_lattice(roots, block_indices, block_size, lattice_size, point)
    node = _findleaf_lattice(roots, point, lattice_size)
    key = node.data.key
    scale = lattice_size .÷ ((roots.layout .* block_size) .* (1 << key.level))
    i = mod(fld(point[1], scale[1]), block_size[1]) + 1
    j = mod(fld(point[2], scale[2]), block_size[2]) + 1
    return CellRef(block_indices[key], i, j)
end

@inline function _tree_box_intersects(low, high, node)
    origin = node.boundary.origin
    upper = origin + node.boundary.widths
    return !(
        upper[1] < low[1] || high[1] < origin[1] ||
            upper[2] < low[2] || high[2] < origin[2]
    )
end

"""Collect active block indices intersecting a physical search box.

The recursive walk prunes disjoint quadtree branches.  Local stencil and
transfer schedule builders use this instead of testing the box against every
active block.  The marked overload deduplicates results from periodic images
without sorting them.
"""
function _collect_leaf_blocks_in_window!(
        blocks, node, block_indices, low, high,
    )
    _tree_box_intersects(low, high, node) || return blocks
    if RegionTrees.isleaf(node)
        block = get(block_indices, node.data.key, 0)
        iszero(block) && error("active tree leaf is missing from block index")
        push!(blocks, block)
        return blocks
    end
    for child in RegionTrees.children(node)
        _collect_leaf_blocks_in_window!(
            blocks, child, block_indices, low, high,
        )
    end
    return blocks
end

@inline function _root_window_indices(roots::RootGrid, low, high, axis)
    domain_low = roots.origin[axis]
    domain_high = domain_low + roots.extent[axis]
    high[axis] < domain_low && return 1:0
    low[axis] > domain_high && return 1:0
    # Search boxes are closed.  At a root boundary the root on the lower side
    # intersects the lower edge too, so include it in the candidate range.
    first_root = clamp(
        ceil(Int, (low[axis] - domain_low) / roots.root_extent[axis]) - 1,
        0, roots.layout[axis] - 1,
    )
    last_root = clamp(
        floor(Int, (high[axis] - domain_low) / roots.root_extent[axis]),
        0, roots.layout[axis] - 1,
    )
    return first_root:last_root
end

function _collect_leaf_blocks_in_window!(
        blocks, roots::RootGrid, block_indices, low, high,
    )
    columns = _root_window_indices(roots, low, high, 1)
    rows = _root_window_indices(roots, low, high, 2)
    for j in rows, i in columns
        root = roots.roots[j * roots.layout[1] + i + 1]
        _collect_leaf_blocks_in_window!(blocks, root, block_indices, low, high)
    end
    return blocks
end

function _collect_leaf_blocks_in_window!(
        blocks, marks, epoch, node, block_indices, low, high,
    )
    _tree_box_intersects(low, high, node) || return blocks
    if RegionTrees.isleaf(node)
        block = get(block_indices, node.data.key, 0)
        iszero(block) && error("active tree leaf is missing from block index")
        if marks[block] != epoch
            marks[block] = epoch
            push!(blocks, block)
        end
        return blocks
    end
    for child in RegionTrees.children(node)
        _collect_leaf_blocks_in_window!(
            blocks, marks, epoch, child, block_indices, low, high,
        )
    end
    return blocks
end

function _collect_leaf_blocks_in_window!(
        blocks, marks, epoch, roots::RootGrid, block_indices, low, high,
    )
    columns = _root_window_indices(roots, low, high, 1)
    rows = _root_window_indices(roots, low, high, 2)
    for j in rows, i in columns
        root = roots.roots[j * roots.layout[1] + i + 1]
        _collect_leaf_blocks_in_window!(
            blocks, marks, epoch, root, block_indices, low, high,
        )
    end
    return blocks
end

# A closed search box on the level-`level` lattice, whose level-zero root
# spans `unit` lattice units. Compare at the finer of the node and box levels
# so that both face and corner contacts remain exact.
@inline function _lattice_box_intersects(node, low, high, level, unit)
    key = node.data.key
    common_level = max(level, key.level)
    node_scale = 1 << (common_level - key.level)
    window_scale = 1 << (common_level - level)
    xlo = key.i * unit[1] * node_scale
    ylo = key.j * unit[2] * node_scale
    return !(
        xlo + unit[1] * node_scale < low[1] * window_scale ||
            high[1] * window_scale < xlo ||
            ylo + unit[2] * node_scale < low[2] * window_scale ||
            high[2] * window_scale < ylo
    )
end

function _collect_leaf_nodes_in_window!(leaves, node, low, high, level, unit)
    _lattice_box_intersects(node, low, high, level, unit) || return leaves
    if RegionTrees.isleaf(node)
        push!(leaves, node)
    else
        for child in RegionTrees.children(node)
            _collect_leaf_nodes_in_window!(leaves, child, low, high, level, unit)
        end
    end
    return leaves
end

@inline function _lattice_root_window(low, high, width, count)
    (high < 0 || low > count * width) && return 1:0
    first_root = clamp(cld(low, width) - 1, 0, count - 1)
    last_root = clamp(fld(high, width), 0, count - 1)
    return first_root:last_root
end

function _collect_leaf_nodes_in_window!(
        leaves, roots::RootGrid, low, high, level, unit,
    )
    columns = _lattice_root_window(low[1], high[1], unit[1] << level, roots.layout[1])
    rows = _lattice_root_window(low[2], high[2], unit[2] << level, roots.layout[2])
    for j in rows, i in columns
        root = roots.roots[j * roots.layout[1] + i + 1]
        _collect_leaf_nodes_in_window!(leaves, root, low, high, level, unit)
    end
    return leaves
end

@inline function _touching_intervals(alo, ahi, blo, bhi, period, periodic)
    return (alo <= bhi && blo <= ahi) ||
        (periodic && ((alo == 0 && bhi == period) || (blo == 0 && ahi == period)))
end

@inline function _touch(a::BlockKey, b::BlockKey, layout, periodic)
    level = max(a.level, b.level)
    ascale = 1 << (level - a.level)
    bscale = 1 << (level - b.level)
    return _touching_intervals(
        a.i * ascale, (a.i + 1) * ascale,
        b.i * bscale, (b.i + 1) * bscale, layout[1] << level, periodic[1],
    ) && _touching_intervals(
        a.j * ascale, (a.j + 1) * ascale,
        b.j * bscale, (b.j + 1) * bscale, layout[2] << level, periodic[2],
    )
end

function _balance!(root::RootGrid, periodic)
    leaves = _tree_node_type(root)[]
    neighbours = _tree_node_type(root)[]
    while true
        empty!(leaves)
        _collect_leaves!(leaves, root)
        changed = false
        for leaf in leaves
            empty!(neighbours)
            key = leaf.data.key
            level = key.level
            low, high = (key.i, key.j), (key.i + 1, key.j + 1)
            xperiod, yperiod = root.layout .<< level
            xshifts = periodic[1] ? (-xperiod, 0, xperiod) : (0,)
            yshifts = periodic[2] ? (-yperiod, 0, yperiod) : (0,)
            for yshift in yshifts, xshift in xshifts
                _collect_leaf_nodes_in_window!(
                    neighbours, root,
                    (low[1] + xshift, low[2] + yshift),
                    (high[1] + xshift, high[2] + yshift), level, (1, 1),
                )
            end
            for neighbour in neighbours
                leaf === neighbour && continue
                leaf_level = leaf.data.key.level
                neighbour_level = neighbour.data.key.level
                if leaf_level + 1 < neighbour_level
                    _split!(leaf)
                    changed = true
                    break
                elseif neighbour_level + 1 < leaf_level
                    _split!(neighbour)
                    changed = true
                    break
                end
            end
            changed && break
        end
        changed || return root
    end
    return
end

function _build_tree(origin, extent, max_level, refine, root_key)
    root = RegionTrees.Cell(SVector(origin), SVector(extent), TreeNodeData(root_key))
    function visit(node)
        key = node.data.key
        key.level >= max_level && return
        rect = node.boundary
        block_origin = Tuple(rect.origin)
        block_extent = Tuple(rect.widths)
        if refine(key, block_origin, block_extent)
            _split!(node)
            for child in RegionTrees.children(node)
                visit(child)
            end
        end
        return
    end
    visit(root)
    return root
end

function _build_root_grid(origin, extent, base_mesh, block_size, periodic, max_level, refine)
    layout = base_mesh .÷ block_size
    root_extent = extent ./ layout
    first_root = _build_tree(
        origin, root_extent, max_level, refine, BlockKey(0, 0, 0),
    )
    roots = Vector{typeof(first_root)}(undef, prod(layout))
    roots[1] = first_root
    for j in 0:(layout[2] - 1), i in 0:(layout[1] - 1)
        iszero(i) && iszero(j) && continue
        root_origin = (
            origin[1] + i * root_extent[1],
            origin[2] + j * root_extent[2],
        )
        roots[j * layout[1] + i + 1] = _build_tree(
            root_origin, root_extent, max_level, refine, BlockKey(0, i, j),
        )
    end
    grid = RootGrid(roots, layout, origin, extent, root_extent)
    _balance!(grid, periodic)
    return grid
end

function _leaf_blocks(root, block_size)
    nodes = _tree_node_type(root)[]
    _collect_leaves!(nodes, root)
    sort!(nodes; by = n -> (n.data.key.level, n.data.key.j, n.data.key.i))
    T = root isa RootGrid ? eltype(root.origin) : eltype(root.boundary.origin)
    blocks = LeafBlock{T}[]
    for node in nodes
        rect = node.boundary
        origin = Tuple(rect.origin)
        extent = Tuple(rect.widths)
        spacing = (extent[1] / block_size[1], extent[2] / block_size[2])
        push!(blocks, LeafBlock(node.data.key, origin, extent, spacing, spacing[1] * spacing[2]))
    end
    return nodes, blocks
end

@inline function _canonical_coordinate(coordinate, domain_size, periodic)
    return periodic ? mod(coordinate, domain_size) : coordinate
end

"""
    _lattice_sides(blocks, block_size, max_level, Val(axis), lines, periodic)

Cell sides on the lattice lines normal to `axis`, grouped by canonical line
coordinate `c in 0:lines` with a stable counting sort: the sides of the cells
below (left of) line `c` are `lower[lower_offsets[c + 1]:(lower_offsets[c + 2] - 1)]`
and those of the cells above it `upper[...]` likewise, each sorted by `lo`.
"""
function _lattice_sides(blocks, block_size, max_level, ::Val{Axis}, lines, periodic) where {Axis}
    nx, ny = block_size
    lower_offsets = zeros(Int, lines + 2)
    upper_offsets = zeros(Int, lines + 2)
    for block in blocks, j in 1:ny, i in 1:nx
        low, high, _, _ = _cell_side_lattice(block, nx, ny, max_level, i, j, Val(Axis))
        lower_offsets[_canonical_coordinate(high, lines, periodic) + 2] += 1
        upper_offsets[_canonical_coordinate(low, lines, periodic) + 2] += 1
    end
    lower_offsets[1] = upper_offsets[1] = 1
    for line in 1:(lines + 1)
        lower_offsets[line + 1] += lower_offsets[line]
        upper_offsets[line + 1] += upper_offsets[line]
    end
    lower_next = lower_offsets[1:(lines + 1)]
    upper_next = upper_offsets[1:(lines + 1)]
    lower = Vector{RawSide}(undef, lower_offsets[end] - 1)
    upper = Vector{RawSide}(undef, upper_offsets[end] - 1)
    for (b, block) in pairs(blocks), j in 1:ny, i in 1:nx
        low, high, tangent_low, tangent_high =
            _cell_side_lattice(block, nx, ny, max_level, i, j, Val(Axis))
        side = RawSide(tangent_low, tangent_high, CellRef(b, i, j))
        line = _canonical_coordinate(high, lines, periodic) + 1
        lower[lower_next[line]] = side
        lower_next[line] += 1
        line = _canonical_coordinate(low, lines, periodic) + 1
        upper[upper_next[line]] = side
        upper_next[line] += 1
    end
    # Tangential intervals on one side of a line are disjoint: `lo` is unique.
    for line in 1:(lines + 1)
        sort!(view(lower, lower_offsets[line]:(lower_offsets[line + 1] - 1)); by = s -> s.lo)
        sort!(view(upper, upper_offsets[line]:(upper_offsets[line + 1] - 1)); by = s -> s.lo)
    end
    return lower_offsets, lower, upper_offsets, upper
end

"""Normal (`low`, `high`) and tangential lattice bounds of cell `(i, j)` of
`block` for the lines normal to `Axis`, on the finest lattice of `max_level`."""
@inline function _cell_side_lattice(block, nx, ny, max_level, i, j, ::Val{Axis}) where {Axis}
    scale = 1 << (max_level - block.key.level)
    xlo = block.key.i * nx * scale + (i - 1) * scale
    ylo = block.key.j * ny * scale + (j - 1) * scale
    return Axis == 1 ? (xlo, xlo + scale, ylo, ylo + scale) :
        (ylo, ylo + scale, xlo, xlo + scale)
end

@inline function _valid_face_cell(ref::CellRef, nblocks, nx, ny)
    return 1 <= ref.block <= nblocks && 1 <= ref.i <= nx && 1 <= ref.j <= ny
end

"""Validate owned-cell references and axes before using a face schedule.

Both lists are axis-major: every axis-1 face precedes every axis-2 face, so
the faces of one axis are a contiguous range (see `_axis_face_offset`)."""
function _check_face_schedules(faces, boundaries, nblocks, block_size)
    nx, ny = block_size
    for (index, face) in pairs(faces)
        if !(face.axis == 1 || face.axis == 2) ||
                !_valid_face_cell(face.minus, nblocks, nx, ny) ||
                !_valid_face_cell(face.plus, nblocks, nx, ny)
            error(
                "invalid interior face $index: $face; expected axis 1 or 2 " *
                    "and cell indices in (1:$nblocks, 1:$nx, 1:$ny)",
            )
        end
    end
    for (index, face) in pairs(boundaries)
        if !(face.axis == 1 || face.axis == 2) ||
                !_valid_face_cell(face.cell, nblocks, nx, ny)
            error(
                "invalid boundary face $index: $face; expected axis 1 or 2 " *
                    "and cell indices in (1:$nblocks, 1:$nx, 1:$ny)",
            )
        end
    end
    issorted(faces; by = face -> face.axis) &&
        issorted(boundaries; by = face -> face.axis) ||
        error("face schedules must list every axis-1 face before every axis-2 face")
    return nothing
end

"""Global index of the face before the first face of `axis` in the axis-major
face list `faces` (see `_check_face_schedules`), which holds `count` faces of
`axis`: the face of local (per-axis field) index `k` is `faces[offset + k]`."""
@inline _axis_face_offset(faces, axis, count) = axis == 1 ? 0 : length(faces) - count

function _build_face_schedules(
        blocks, base_mesh, block_size, origin, extent, periodic,
    )
    # Use the finest active leaf, not the mesh's unused refinement ceiling.
    max_level = maximum(block.key.level for block in blocks)
    sx, sy = base_mesh .* (1 << max_level)
    T = eltype(blocks[1].origin)
    faces = InteriorFace{T}[]
    boundaries = BoundaryFace{T}[]
    _match_sides!(
        faces, boundaries,
        _lattice_sides(blocks, block_size, max_level, Val(1), sx, periodic[1]),
        1, origin, extent, (sx, sy),
    )
    _match_sides!(
        faces, boundaries,
        _lattice_sides(blocks, block_size, max_level, Val(2), sy, periodic[2]),
        2, origin, extent, (sx, sy),
    )
    _check_face_schedules(faces, boundaries, length(blocks), block_size)
    return faces, boundaries
end

function _match_sides!(faces, boundaries, sides, axis, origin, extent, lattice_size)
    lower_offsets, lower, upper_offsets, upper = sides
    tangential_axis = 3 - axis
    normal_scale = extent[axis] / lattice_size[axis]
    tangent_scale = extent[tangential_axis] / lattice_size[tangential_axis]
    # Lattice-coordinate order, each line's sides in increasing `lo`.
    for line in 1:(length(lower_offsets) - 1)
        coordinate = line - 1
        li, lower_stop = lower_offsets[line], lower_offsets[line + 1] - 1
        ui, upper_stop = upper_offsets[line], upper_offsets[line + 1] - 1
        while li <= lower_stop && ui <= upper_stop
            lside, uside = lower[li], upper[ui]
            lo, hi = max(lside.lo, uside.lo), min(lside.hi, uside.hi)
            if lo < hi
                normal = origin[axis] + coordinate * normal_scale
                tangent = origin[tangential_axis] + (lo + hi) * tangent_scale / 2
                center = axis == 1 ? (normal, tangent) : (tangent, normal)
                push!(faces, InteriorFace(lside.cell, uside.cell, axis, center, (hi - lo) * tangent_scale))
            end
            lside.hi <= uside.hi && (li += 1)
            uside.hi <= lside.hi && (ui += 1)
        end
        for side in @view(lower[li:lower_stop])
            _push_boundary!(boundaries, side, axis, 1, coordinate, origin, extent, lattice_size)
        end
        for side in @view(upper[ui:upper_stop])
            _push_boundary!(boundaries, side, axis, -1, coordinate, origin, extent, lattice_size)
        end
    end
    return faces
end

function _push_boundary!(boundaries, raw, axis, side, coordinate, origin, extent, lattice_size)
    tangent_axis = 3 - axis
    normal = origin[axis] + coordinate * extent[axis] / lattice_size[axis]
    tangent = origin[tangent_axis] + (raw.lo + raw.hi) * extent[tangent_axis] / (2lattice_size[tangent_axis])
    center = axis == 1 ? (normal, tangent) : (tangent, normal)
    area = (raw.hi - raw.lo) * extent[tangent_axis] / lattice_size[tangent_axis]
    push!(boundaries, BoundaryFace(raw.cell, axis, side, center, area))
    return boundaries
end

@inline function _wrap_or_outside(x::Int, period::Int, periodic::Bool)
    periodic && return mod(x, period), false
    return clamp(x, 0, period - 1), x < 0 || x >= period
end

function _build_halo_schedule(root, nodes, blocks, block_size, halo, periodic)
    T = eltype(blocks[1].origin)
    node_to_block = Dict(node.data.key => b for (b, node) in pairs(nodes))
    max_level = maximum(block.key.level for block in blocks)
    lattice_size = (root.layout .* block_size) .* (1 << max_level)
    quarter_period = 4 .* lattice_size
    nx, ny = block_size
    destinations = CellRef[]
    sizehint!(
        destinations,
        length(blocks) * ((nx + 2halo) * (ny + 2halo) - nx * ny),
    )
    for (b, block) in pairs(blocks), j in (1 - halo):(ny + halo), i in (1 - halo):(nx + halo)
        (1 <= i <= nx && 1 <= j <= ny) && continue
        push!(destinations, CellRef(b, i, j))
    end
    number_of_entries = length(destinations)
    centers = Vector{NTuple{2, T}}(undef, number_of_entries)
    physical = Vector{Bool}(undef, number_of_entries)
    donor_counts = Vector{Int}(undef, number_of_entries)
    candidate_donors = Vector{CellRef}(undef, 4number_of_entries)
    candidate_weights = Vector{T}(undef, 4number_of_entries)
    # Tree searches through `root` and the `node_to_block` dictionary: launched
    # as one closure rather than a `@batch` body over these objects. This runs
    # once per topology change, so the closure box per launch is irrelevant.
    _parallel_for(number_of_entries) do entry_index
        destination = destinations[entry_index]
        b, i, j = destination.block, destination.i, destination.j
        block = blocks[b]
        xlo = block.origin[1] + (i - 1) * block.spacing[1]
        ylo = block.origin[2] + (j - 1) * block.spacing[2]
        center = (xlo + block.spacing[1] / 2, ylo + block.spacing[2] / 2)
        scale = 1 << (max_level - block.key.level)
        xbase = 4 * (block.key.i * nx + i - 1) * scale
        ybase = 4 * (block.key.j * ny + j - 1) * scale
        wrapped_center_x, outside_x =
            _wrap_or_outside(xbase + 2scale, quarter_period[1], periodic[1])
        wrapped_center_y, outside_y =
            _wrap_or_outside(ybase + 2scale, quarter_period[2], periodic[2])
        outside = outside_x || outside_y
        centers[entry_index] = center
        physical[entry_index] = outside
        first_candidate = 4(entry_index - 1) + 1
        if outside
            nearest = _cellref_at_lattice(
                root, node_to_block, block_size, lattice_size,
                (wrapped_center_x ÷ 4, wrapped_center_y ÷ 4),
            )
            candidate_donors[first_candidate] = nearest
            candidate_weights[first_candidate] = one(T)
            donor_counts[entry_index] = 1
        else
            local_donors = ntuple(4) do sample
                qx = isodd(sample) ? 1 : 3
                qy = sample <= 2 ? 1 : 3
                x, _ = _wrap_or_outside(xbase + qx * scale, quarter_period[1], periodic[1])
                y, _ = _wrap_or_outside(ybase + qy * scale, quarter_period[2], periodic[2])
                _cellref_at_lattice(root, node_to_block, block_size, lattice_size, (x ÷ 4, y ÷ 4))
            end
            count = 0
            for donor_index in eachindex(local_donors)
                donor = local_donors[donor_index]
                any(local_donors[index] == donor for index in 1:(donor_index - 1)) &&
                    continue
                multiplicity = Base.count(==(donor), local_donors)
                count += 1
                candidate_donors[first_candidate + count - 1] = donor
                candidate_weights[first_candidate + count - 1] =
                    T(multiplicity) / 4
            end
            donor_counts[entry_index] = count
        end
    end
    offsets = _csr_offsets(donor_counts)
    donors = Vector{CellRef}(undef, offsets[end] - 1)
    weights = Vector{T}(undef, length(donors))
    entries = Vector{HaloEntry{T}}(undef, number_of_entries)
    @_parallel for entry_index in 1:number_of_entries
        first = offsets[entry_index]
        count = donor_counts[entry_index]
        first_candidate = 4(entry_index - 1) + 1
        for local_index in 1:count
            donors[first + local_index - 1] =
                candidate_donors[first_candidate + local_index - 1]
            weights[first + local_index - 1] =
                candidate_weights[first_candidate + local_index - 1]
        end
        entries[entry_index] = HaloEntry(
            destinations[entry_index], first, count,
            centers[entry_index], physical[entry_index],
        )
    end
    return entries, donors, weights
end

"""
    BlockTreeMesh(; base_mesh=block_size, block_size=(16, 16), halo=1,
                    origin=(0.0, 0.0),
                    extent=(1.0, 1.0), periodic=(true, true), max_level=0,
                    refine=(key, origin, extent) -> false,
                    adaptation=MeshAdaptationOptions())

Build a serial 2-D forest of fixed-size Cartesian quadtree leaf blocks.
`base_mesh` is the global level-zero cell count and must be an integer multiple
of `block_size` in each direction. The resulting permanent base roots may be
refined but never coarsened away. `refine` is called for every candidate block
up to `max_level`. The forest is automatically refined until it is fully 2:1
balanced (including corners). Neighbours and halo donors are determined from
integer block-key lattices, including periodic seams, independently of
floating-point domain coordinates. Execution and halo schedules are then
materialized, so numerical kernels never traverse the trees. `max_level`
remains the refinement ceiling for subsequent dynamic adaptation; its
frequency, base-level, and coarsening-hysteresis controls are stored in
`adaptation`.
"""
function BlockTreeMesh(;
        block_size = (16, 16), base_mesh = block_size, halo = 1,
        origin = (0.0, 0.0), extent = (1.0, 1.0),
        periodic = (true, true), max_level = 0,
        refine = (key, origin, extent) -> false,
        adaptation = MeshAdaptationOptions()
    )
    all(value -> value isa Integer && !(value isa Bool), base_mesh) || throw(
        ArgumentError("base mesh dimensions must be integers")
    )
    nx, ny = Int.(block_size)
    base_nx, base_ny = Int.(base_mesh)
    nx >= 2 && ny >= 2 || throw(ArgumentError("block dimensions must be at least two"))
    iseven(nx) && iseven(ny) || throw(ArgumentError("block dimensions must be even for dyadic transfer"))
    base_nx >= nx && base_ny >= ny || throw(
        ArgumentError("base mesh dimensions must be at least the block dimensions")
    )
    iszero(base_nx % nx) && iszero(base_ny % ny) || throw(
        ArgumentError("base mesh dimensions must be integer multiples of block dimensions")
    )
    halo >= 1 || throw(ArgumentError("halo width must be positive"))
    max_level >= 0 || throw(ArgumentError("max_level must be nonnegative"))
    adaptation isa MeshAdaptationOptions || throw(
        ArgumentError(
            "adaptation must be MeshAdaptationOptions"
        )
    )
    adaptation.minimum_level <= max_level || throw(
        ArgumentError(
            "adaptation minimum_level must not exceed max_level"
        )
    )
    all(>(0), extent) || throw(ArgumentError("domain extents must be positive"))
    T = float(
        promote_type(
            typeof(origin[1]), typeof(origin[2]),
            typeof(extent[1]), typeof(extent[2])
        )
    )
    mesh_origin = (T(origin[1]), T(origin[2]))
    mesh_extent = (T(extent[1]), T(extent[2]))
    mesh_periodic = (Bool(periodic[1]), Bool(periodic[2]))
    mesh_base = (base_nx, base_ny)
    root = _build_root_grid(
        mesh_origin, mesh_extent, mesh_base, (nx, ny), mesh_periodic,
        max_level, refine,
    )
    nodes, blocks = _leaf_blocks(root, (nx, ny))
    faces, boundaries = _build_face_schedules(
        blocks, mesh_base, (nx, ny), mesh_origin,
        mesh_extent, mesh_periodic
    )
    halo_entries, halo_donors, halo_weights = _build_halo_schedule(
        root, nodes, blocks, (nx, ny), halo, mesh_periodic,
    )
    return BlockTreeMesh(
        root, blocks, mesh_base, (nx, ny), halo, mesh_origin, mesh_extent,
        mesh_periodic, max_level, faces, _face_dual_volumes(blocks, faces),
        boundaries,
        halo_entries, halo_donors, halo_weights,
        MeshAdaptationState(adaptation), MeshTopologyCache(T)
    )
end

"""Return whether face- or corner-touching blocks satisfy 2:1 grading."""
function isbalanced(mesh::BlockTreeMesh)
    blocks = mesh.blocks
    layout = mesh.base_mesh .÷ mesh.block_size
    for i in eachindex(blocks), j in (i + 1):length(blocks)
        a, b = blocks[i].key, blocks[j].key
        abs(a.level - b.level) <= 1 && continue
        _touch(a, b, layout, mesh.periodic) && return false
    end
    return true
end

"""Return the physical centre of local cell `(i, j)` in an active block."""
@inline cellcenter(mesh::BlockTreeMesh, block_index::Integer, i::Integer, j::Integer) =
    _leaf_cell_center(mesh.blocks[block_index], i, j)

"""Centre of local cell `(i, j)` of one leaf block (the plain-data form of
`cellcenter` that threaded loops use)."""
@inline _leaf_cell_center(block, i, j) = (
    block.origin[1] + (i - 1 / 2) * block.spacing[1],
    block.origin[2] + (j - 1 / 2) * block.spacing[2],
)
