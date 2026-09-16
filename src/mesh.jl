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

struct BlockTreeMesh{T, Tree}
    root::Tree
    blocks::Vector{LeafBlock{T}}
    block_size::NTuple{2, Int}
    halo::Int
    origin::NTuple{2, T}
    extent::NTuple{2, T}
    periodic::NTuple{2, Bool}
    max_level::Int
    faces::Vector{InteriorFace{T}}
    boundary_faces::Vector{BoundaryFace{T}}
    halo_entries::Vector{HaloEntry{T}}
    halo_donors::Vector{CellRef}
    halo_weights::Vector{T}
    adaptation::MeshAdaptationState
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
    return delta - round(Int, delta / period) * period
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

@inline function _tree_box_intersects(low, high, node)
    origin = node.boundary.origin
    upper = origin + node.boundary.widths
    return !(upper[1] < low[1] || high[1] < origin[1] ||
        upper[2] < low[2] || high[2] < origin[2])
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

function _touch(a, b, extent, periodic)
    ao, aw = a.boundary.origin, a.boundary.widths
    bo, bw = b.boundary.origin, b.boundary.widths
    ax1, ay1 = ao
    ax2, ay2 = ao + aw
    tol = 16eps(promote_type(eltype(ao), eltype(bo))) * max(one(eltype(ao)), maximum(aw), maximum(bw))
    xshifts = periodic[1] ? (-extent[1], zero(extent[1]), extent[1]) : (zero(extent[1]),)
    yshifts = periodic[2] ? (-extent[2], zero(extent[2]), extent[2]) : (zero(extent[2]),)
    for yshift in yshifts, xshift in xshifts
        bx1, by1 = bo[1] + xshift, bo[2] + yshift
        bx2, by2 = bx1 + bw[1], by1 + bw[2]
        separated = ax2 < bx1 - tol || bx2 < ax1 - tol ||
            ay2 < by1 - tol || by2 < ay1 - tol
        separated || return true
    end
    return false
end

function _balance!(root, extent, periodic)
    leaves = typeof(root)[]
    while true
        empty!(leaves)
        _collect_leaves!(leaves, root)
        changed = false
        for aidx in eachindex(leaves), bidx in (aidx + 1):length(leaves)
            a, b = leaves[aidx], leaves[bidx]
            _touch(a, b, extent, periodic) || continue
            la, lb = a.data.key.level, b.data.key.level
            if la + 1 < lb
                _split!(a)
                changed = true
                break
            elseif lb + 1 < la
                _split!(b)
                changed = true
                break
            end
        end
        changed || return root
    end
    return
end

function _build_tree(origin, extent, periodic, max_level, refine)
    root = RegionTrees.Cell(SVector(origin), SVector(extent), TreeNodeData(BlockKey(0, 0, 0)))
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
    _balance!(root, extent, periodic)
    return root
end

function _leaf_blocks(root, block_size)
    nodes = typeof(root)[]
    _collect_leaves!(nodes, root)
    sort!(nodes; by = n -> (n.data.key.level, n.data.key.j, n.data.key.i))
    T = eltype(root.boundary.origin)
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

function _insert_side!(table, key, side, lo, hi, cell)
    lower, upper = get!(table, key) do
        (RawSide[], RawSide[])
    end
    push!(side > 0 ? lower : upper, RawSide(lo, hi, cell))
    return table
end

function _build_face_schedules(blocks, block_size, origin, extent, periodic, max_level)
    nx, ny = block_size
    sx = nx * (1 << max_level)
    sy = ny * (1 << max_level)
    vertical = Dict{Int, Tuple{Vector{RawSide}, Vector{RawSide}}}()
    horizontal = Dict{Int, Tuple{Vector{RawSide}, Vector{RawSide}}}()

    for (b, block) in pairs(blocks)
        scale = 1 << (max_level - block.key.level)
        block_x = block.key.i * nx * scale
        block_y = block.key.j * ny * scale
        for j in 1:ny, i in 1:nx
            cell = CellRef(b, i, j)
            xlo = block_x + (i - 1) * scale
            xhi = xlo + scale
            ylo = block_y + (j - 1) * scale
            yhi = ylo + scale
            _insert_side!(vertical, _canonical_coordinate(xlo, sx, periodic[1]), -1, ylo, yhi, cell)
            _insert_side!(vertical, _canonical_coordinate(xhi, sx, periodic[1]), 1, ylo, yhi, cell)
            _insert_side!(horizontal, _canonical_coordinate(ylo, sy, periodic[2]), -1, xlo, xhi, cell)
            _insert_side!(horizontal, _canonical_coordinate(yhi, sy, periodic[2]), 1, xlo, xhi, cell)
        end
    end

    T = eltype(blocks[1].origin)
    faces = InteriorFace{T}[]
    boundaries = BoundaryFace{T}[]
    _match_sides!(faces, boundaries, vertical, 1, origin, extent, (sx, sy))
    _match_sides!(faces, boundaries, horizontal, 2, origin, extent, (sx, sy))
    return faces, boundaries
end

function _match_sides!(faces, boundaries, table, axis, origin, extent, lattice_size)
    tangential_axis = 3 - axis
    normal_scale = extent[axis] / lattice_size[axis]
    tangent_scale = extent[tangential_axis] / lattice_size[tangential_axis]
    for (coordinate, (lower, upper)) in table
        sort!(lower; by = s -> s.lo)
        sort!(upper; by = s -> s.lo)
        li = ui = 1
        while li <= length(lower) && ui <= length(upper)
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
        for side in @view(lower[li:end])
            _push_boundary!(boundaries, side, axis, 1, coordinate, origin, extent, lattice_size)
        end
        for side in @view(upper[ui:end])
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

@inline function _wrap_or_outside(x, axis, origin, extent, periodic)
    low, high = origin[axis], origin[axis] + extent[axis]
    if periodic[axis]
        return low + mod(x - low, extent[axis]), false
    end
    return clamp(x, low, prevfloat(high)), x < low || x >= high
end

function _cellref_at(root, node_to_block, blocks, block_size, point)
    node = RegionTrees.findleaf(root, SVector(point))
    b = node_to_block[node]
    block = blocks[b]
    nx, ny = block_size
    i = clamp(floor(Int, (point[1] - block.origin[1]) / block.spacing[1]) + 1, 1, nx)
    j = clamp(floor(Int, (point[2] - block.origin[2]) / block.spacing[2]) + 1, 1, ny)
    return CellRef(b, i, j)
end

function _build_halo_schedule(root, nodes, blocks, block_size, halo, origin, extent, periodic)
    T = eltype(blocks[1].origin)
    node_to_block = IdDict(node => b for (b, node) in pairs(nodes))
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
    @_parallel_tasks for entry_index in 1:number_of_entries
        destination = destinations[entry_index]
        b, i, j = destination.block, destination.i, destination.j
        block = blocks[b]
        xlo = block.origin[1] + (i - 1) * block.spacing[1]
        ylo = block.origin[2] + (j - 1) * block.spacing[2]
        center = (xlo + block.spacing[1] / 2, ylo + block.spacing[2] / 2)
        wrapped_center_x, outside_x = _wrap_or_outside(center[1], 1, origin, extent, periodic)
        wrapped_center_y, outside_y = _wrap_or_outside(center[2], 2, origin, extent, periodic)
        outside = outside_x || outside_y
        centers[entry_index] = center
        physical[entry_index] = outside
        first_candidate = 4(entry_index - 1) + 1
        if outside
            nearest = _cellref_at(
                root, node_to_block, blocks, block_size,
                (wrapped_center_x, wrapped_center_y)
            )
            candidate_donors[first_candidate] = nearest
            candidate_weights[first_candidate] = one(T)
            donor_counts[entry_index] = 1
        else
            local_donors = ntuple(4) do sample
                qx = isodd(sample) ? T(0.25) : T(0.75)
                qy = sample <= 2 ? T(0.25) : T(0.75)
                x, _ = _wrap_or_outside(xlo + qx * block.spacing[1], 1, origin, extent, periodic)
                y, _ = _wrap_or_outside(ylo + qy * block.spacing[2], 2, origin, extent, periodic)
                _cellref_at(root, node_to_block, blocks, block_size, (x, y))
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
    offsets = Vector{Int}(undef, number_of_entries + 1)
    offsets[1] = 1
    for entry_index in 1:number_of_entries
        offsets[entry_index + 1] =
            offsets[entry_index] + donor_counts[entry_index]
    end
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
    BlockTreeMesh(; block_size=(16, 16), halo=1, origin=(0.0, 0.0),
                    extent=(1.0, 1.0), periodic=(true, true), max_level=0,
                    refine=(key, origin, extent) -> false,
                    adaptation=MeshAdaptationOptions())

Build a serial 2-D quadtree of fixed-size Cartesian leaf blocks. `refine` is
called for every candidate block up to `max_level`. The resulting tree is
automatically refined until it is fully 2:1 balanced (including corners).
Execution and halo schedules are then materialized, so numerical kernels never
traverse the tree. `max_level` remains the refinement ceiling for subsequent
dynamic adaptation; its frequency, base-level, and coarsening-hysteresis
controls are stored in `adaptation`.
"""
function BlockTreeMesh(;
        block_size = (16, 16), halo = 1,
        origin = (0.0, 0.0), extent = (1.0, 1.0),
        periodic = (true, true), max_level = 0,
        refine = (key, origin, extent) -> false,
        adaptation = MeshAdaptationOptions()
    )
    nx, ny = Int.(block_size)
    nx >= 2 && ny >= 2 || throw(ArgumentError("block dimensions must be at least two"))
    iseven(nx) && iseven(ny) || throw(ArgumentError("block dimensions must be even for dyadic transfer"))
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
    root = _build_tree(mesh_origin, mesh_extent, mesh_periodic, max_level, refine)
    nodes, blocks = _leaf_blocks(root, (nx, ny))
    faces, boundaries = _build_face_schedules(
        blocks, (nx, ny), mesh_origin,
        mesh_extent, mesh_periodic, max_level
    )
    halo_entries, halo_donors, halo_weights = _build_halo_schedule(
        root, nodes, blocks, (nx, ny), halo, mesh_origin, mesh_extent, mesh_periodic
    )
    return BlockTreeMesh(
        root, blocks, (nx, ny), halo, mesh_origin, mesh_extent,
        mesh_periodic, max_level, faces, boundaries,
        halo_entries, halo_donors, halo_weights,
        MeshAdaptationState(adaptation)
    )
end

"""Return whether touching active blocks satisfy the 2:1 grading rule."""
function isbalanced(mesh::BlockTreeMesh)
    blocks = mesh.blocks
    for i in eachindex(blocks), j in (i + 1):length(blocks)
        a, b = blocks[i], blocks[j]
        xshifts = mesh.periodic[1] ? (-mesh.extent[1], zero(eltype(mesh.extent)), mesh.extent[1]) :
            (zero(eltype(mesh.extent)),)
        yshifts = mesh.periodic[2] ? (-mesh.extent[2], zero(eltype(mesh.extent)), mesh.extent[2]) :
            (zero(eltype(mesh.extent)),)
        for yshift in yshifts, xshift in xshifts
            ax2, ay2 = a.origin[1] + a.extent[1], a.origin[2] + a.extent[2]
            bx1, by1 = b.origin[1] + xshift, b.origin[2] + yshift
            bx2, by2 = bx1 + b.extent[1], by1 + b.extent[2]
            separated = ax2 < bx1 || bx2 < a.origin[1] ||
                ay2 < by1 || by2 < a.origin[2]
            if !separated && abs(a.key.level - b.key.level) > 1
                return false
            end
        end
    end
    return true
end

"""Return the physical centre of local cell `(i, j)` in an active block."""
@inline function cellcenter(
        mesh::BlockTreeMesh, block_index::Integer,
        i::Integer, j::Integer
    )
    block = mesh.blocks[block_index]
    return (
        block.origin[1] + (i - 1 / 2) * block.spacing[1],
        block.origin[2] + (j - 1 / 2) * block.spacing[2],
    )
end
