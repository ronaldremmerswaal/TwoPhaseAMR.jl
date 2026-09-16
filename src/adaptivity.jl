"""Interface for block-wise AMR label estimators."""
abstract type AbstractRefinementIndicator end

"""
    InterpolationErrorIndicator(field; E_refine, E_coarsen,
                                relative_floor=sqrt(eps(T)),
                                boundary=ZeroGradient(), minimum_level=0,
                                maximum_level=field.mesh.max_level,
                                t_init=-Inf, t_end=Inf)

Block indicator for a cell-average field. At every owned cell, compare the
stored value with the centred interpolation of its four axial neighbours and
normalize by their local magnitude. A block requests `:refine` when its
maximum relative defect exceeds `E_refine`, `:coarsen` when it is below
`E_coarsen`, and `:keep` in between.

The two thresholds provide value hysteresis and must satisfy
`0 <= E_coarsen < E_refine`. `relative_floor` prevents a meaningless relative
error where the field and all its neighbours vanish. The criterion is active
for `t_init <= time <= t_end`. Its level bounds apply independently of the
error thresholds.
"""
struct InterpolationErrorIndicator{T, F <: CellField, B <: AbstractBoundaryCondition} <:
    AbstractRefinementIndicator
    field::F
    E_refine::T
    E_coarsen::T
    relative_floor::T
    boundary::B
    minimum_level::Int
    maximum_level::Int
    t_init::T
    t_end::T
end

@inline _indicator_field(indicator::InterpolationErrorIndicator) =
    indicator.field

function InterpolationErrorIndicator(
        field::CellField{T}; E_refine, E_coarsen,
        relative_floor = sqrt(eps(T)), boundary = ZeroGradient(),
        minimum_level = 0, maximum_level = field.mesh.max_level,
        t_init = -Inf, t_end = Inf
    ) where {T}
    refine = T(E_refine)
    coarsen = T(E_coarsen)
    floor = T(relative_floor)
    zero(T) <= coarsen < refine || throw(
        ArgumentError(
            "indicator thresholds must satisfy 0 <= E_coarsen < E_refine"
        )
    )
    floor > zero(T) || throw(
        ArgumentError(
            "indicator relative_floor must be strictly positive"
        )
    )
    lower_level, upper_level = _refinement_levels(
        field.mesh, minimum_level, maximum_level
    )
    start_time, end_time = _refinement_window(T, t_init, t_end)
    return InterpolationErrorIndicator(
        field, refine, coarsen, floor, boundary,
        lower_level, upper_level, start_time, end_time
    )
end

function _refinement_levels(mesh, minimum_level, maximum_level)
    minimum_level isa Integer && !(minimum_level isa Bool) ||
        throw(ArgumentError("minimum_level must be an integer"))
    maximum_level isa Integer && !(maximum_level isa Bool) ||
        throw(ArgumentError("maximum_level must be an integer"))
    0 <= minimum_level <= maximum_level <= mesh.max_level ||
        throw(
        ArgumentError(
            "minimum_level and maximum_level must lie within mesh limits"
        )
    )
    return Int(minimum_level), Int(maximum_level)
end

function _refinement_window(::Type{T}, t_init, t_end) where {T}
    start_time = T(t_init)
    end_time = T(t_end)
    !isnan(start_time) && !isnan(end_time) && start_time <= end_time ||
        throw(ArgumentError("require t_init <= t_end and neither may be NaN"))
    return start_time, end_time
end

function _indicator_active(indicator, time)
    time isa Real && !isnan(time) ||
        throw(ArgumentError("refinement time must be a real number and not NaN"))
    return indicator.t_init <= time <= indicator.t_end
end

@inline _indicator_can_coarsen(level, minimum_level) =
    iszero(level) || level > minimum_level

@inline _refinement_priority(action::Symbol) =
    action === :coarsen ? 0 : action === :keep ? 1 : action === :refine ? 2 :
    throw(ArgumentError("refinement action must be :refine, :keep, or :coarsen"))

@inline function _combine_refinement_actions(first::Symbol, second::Symbol)
    return _refinement_priority(first) >= _refinement_priority(second) ? first : second
end

"""
    refinement_labels!(labels, indicator; time=0)
    refinement_labels!(workspace, indicators; time=0)

Write one `:refine`, `:keep`, or `:coarsen` proposal per active block. Multiple
indicators are combined with `:refine > :keep > :coarsen` precedence. Built-in
indicators outside their inclusive activation window contribute the neutral
`:coarsen` proposal.
"""
function refinement_labels!(
        labels::AbstractVector{Symbol},
        indicator::InterpolationErrorIndicator;
        time = 0
    )
    field = indicator.field
    length(labels) == nblocks(field.mesh) || throw(
        DimensionMismatch(
            "one refinement label is required per active block"
        )
    )
    if !_indicator_active(indicator, time)
        fill!(labels, :coarsen)
        return labels
    end
    fill_halo!(field, indicator.boundary)
    nx, ny = field.mesh.block_size
    h = field.mesh.halo
    for block_index in eachindex(field.blocks)
        values = field.blocks[block_index]
        maximum_error = zero(typeof(indicator.E_refine))
        for j in 1:ny, i in 1:nx
            @inbounds begin
                value = values[i + h, j + h]
                left = values[i + h - 1, j + h]
                right = values[i + h + 1, j + h]
                below = values[i + h, j + h - 1]
                above = values[i + h, j + h + 1]
            end
            interpolated = (left + right + below + above) / 4
            scale = max(
                indicator.relative_floor, abs(value), abs(left),
                abs(right), abs(below), abs(above)
            )
            maximum_error = max(
                maximum_error, abs(value - interpolated) / scale
            )
        end
        level = field.mesh.blocks[block_index].key.level
        @inbounds labels[block_index] =
            level < indicator.minimum_level ? :refine :
            level > indicator.maximum_level ? :coarsen :
            maximum_error > indicator.E_refine &&
            level < indicator.maximum_level ? :refine :
            maximum_error < indicator.E_coarsen &&
            _indicator_can_coarsen(
                level, indicator.minimum_level
            ) ? :coarsen : :keep
    end
    return labels
end


_refinement_labels_at_time!(labels, indicator, time) =
    refinement_labels!(labels, indicator)

_refinement_labels_at_time!(
    labels, indicator::InterpolationErrorIndicator, time
) = refinement_labels!(labels, indicator; time)

abstract type AbstractAMRFieldBuffer end

mutable struct CellAMRFieldBuffer{T, F <: CellField{T}, B, P} <:
    AbstractAMRFieldBuffer
    field::F
    boundary::B
    prolongation::P
    staging::Vector{Matrix{T}}
    pool::Vector{Matrix{T}}
end

mutable struct FaceAMRFieldBuffer{T, F} <: AbstractAMRFieldBuffer
    field::F
    old_interior::Vector{T}
    old_boundary::Vector{T}
    piecewise_constant::Bool
    conserve_integral::Bool
end

struct FaceTransferSegment
    lo::Int
    hi::Int
    source_index::Int
    local_index::Int
    boundary::Bool
end

"""Reusable buffers and topology scratch for in-place mesh adaptation."""
mutable struct AMRWorkspace{T, M <: BlockTreeMesh{T}, N}
    mesh::M
    old_blocks::Vector{LeafBlock{T}}
    old_faces::Vector{InteriorFace{T}}
    old_boundary_faces::Vector{BoundaryFace{T}}
    leaf_nodes::Vector{N}
    nodes_by_key::Dict{BlockKey, N}
    old_indices::Dict{BlockKey, Int}
    new_indices::Dict{BlockKey, Int}
    labels_by_key::Dict{BlockKey, Symbol}
    labels::Vector{Symbol}
    indicator_labels::Vector{Symbol}
    reused_old_blocks::BitVector
    field_buffers::Vector{AbstractAMRFieldBuffer}
    face_lines::NTuple{2, Dict{Int, Vector{FaceTransferSegment}}}
    face_line_coordinates::NTuple{2, Vector{Int}}
    new_face_indices::Dict{NTuple{4, Int}, Int}
    old_cell_flux::Vector{T}
    new_cell_flux::Vector{T}
end

function AMRWorkspace(mesh::M, fields...) where {T, N, M <: BlockTreeMesh{T, N}}
    workspace = AMRWorkspace{T, M, N}(
        mesh, LeafBlock{T}[], InteriorFace{T}[], BoundaryFace{T}[], N[],
        Dict{BlockKey, N}(), Dict{BlockKey, Int}(), Dict{BlockKey, Int}(),
        Dict{BlockKey, Symbol}(), Symbol[], Symbol[], BitVector(),
        AbstractAMRFieldBuffer[],
        (
            Dict{Int, Vector{FaceTransferSegment}}(),
            Dict{Int, Vector{FaceTransferSegment}}(),
        ),
        (Int[], Int[]), Dict{NTuple{4, Int}, Int}(), T[], T[]
    )
    for field in fields
        register_adaptive_field!(workspace, field)
    end
    return workspace
end

function _find_field_buffer(workspace::AMRWorkspace, field)
    return findfirst(
        buffer -> getfield(buffer, :field) === field,
        workspace.field_buffers
    )
end

"""
    register_adaptive_field!(workspace, field; boundary=ZeroGradient())

Register a field for in-place transfer whenever `workspace.mesh` changes.
Repeated registration of the same object is a no-op. The boundary keyword is
used only for cell fields during reconstruction.
"""
function register_adaptive_field!(
        workspace::AMRWorkspace,
        field::CellField;
        boundary = ZeroGradient(), prolongation = _prolong_child!
    )
    field.mesh === workspace.mesh || throw(
        ArgumentError(
            "an adaptive field must use the workspace mesh"
        )
    )
    isnothing(_find_field_buffer(workspace, field)) || return workspace
    push!(
        workspace.field_buffers,
        CellAMRFieldBuffer(
            field, boundary, prolongation, Matrix{eltype(field)}[],
            Matrix{eltype(field)}[]
        )
    )
    return workspace
end

function register_adaptive_field!(
        workspace::AMRWorkspace,
        field::Union{FaceField, FaceVelocity};
        piecewise_constant = false, conserve_integral = true
    )
    field.mesh === workspace.mesh || throw(
        ArgumentError(
            "an adaptive field must use the workspace mesh"
        )
    )
    isnothing(_find_field_buffer(workspace, field)) || return workspace
    push!(
        workspace.field_buffers,
        FaceAMRFieldBuffer(
            field, eltype(field)[], eltype(field)[],
            piecewise_constant, conserve_integral
        )
    )
    return workspace
end

function _register_indicator_field!(
        workspace::AMRWorkspace,
        indicator::InterpolationErrorIndicator
    )
    return register_adaptive_field!(
        workspace, indicator.field; boundary = indicator.boundary
    )
end

function _register_indicator_field!(
        workspace::AMRWorkspace,
        indicator::AbstractRefinementIndicator
    )
    hasproperty(indicator, :field) || return workspace
    return register_adaptive_field!(workspace, getproperty(indicator, :field))
end

"""Summary of one requested mesh-adaptation attempt."""
struct AMRStats
    attempted::Bool
    changed::Bool
    old_blocks::Int
    new_blocks::Int
    refined_blocks::Int
    coarsened_families::Int
end

function refinement_labels!(workspace::AMRWorkspace, indicators; time = 0)
    number_of_blocks = nblocks(workspace.mesh)
    resize!(workspace.labels, number_of_blocks)
    resize!(workspace.indicator_labels, number_of_blocks)
    fill!(workspace.labels, :coarsen)
    for indicator in indicators
        _refinement_labels_at_time!(
            workspace.indicator_labels, indicator, time
        )
        for index in eachindex(workspace.labels)
            @inbounds workspace.labels[index] = _combine_refinement_actions(
                workspace.labels[index], workspace.indicator_labels[index]
            )
        end
    end
    isempty(indicators) && fill!(workspace.labels, :keep)
    return workspace.labels
end

refinement_labels!(
    workspace::AMRWorkspace,
    indicator::AbstractRefinementIndicator;
    time = 0
) =
    refinement_labels!(workspace, (indicator,); time)

@inline _parent_key(key::BlockKey) = BlockKey(
    key.level - 1, key.i ÷ 2, key.j ÷ 2
)

@inline function _child_keys(key::BlockKey)
    level = key.level + 1
    i, j = 2key.i, 2key.j
    return (
        BlockKey(level, i, j), BlockKey(level, i + 1, j),
        BlockKey(level, i, j + 1), BlockKey(level, i + 1, j + 1),
    )
end

function _collect_active_nodes!(workspace::AMRWorkspace)
    empty!(workspace.leaf_nodes)
    _collect_leaves!(workspace.leaf_nodes, workspace.mesh.root)
    sort!(
        workspace.leaf_nodes;
        by = node -> (node.data.key.level, node.data.key.j, node.data.key.i)
    )
    empty!(workspace.nodes_by_key)
    for node in workspace.leaf_nodes
        workspace.nodes_by_key[node.data.key] = node
    end
    return workspace.leaf_nodes
end

function _can_coarsen(parent, labels_by_key, state, step)
    RegionTrees.isleaf(parent) && return false
    children = RegionTrees.children(parent)
    all(RegionTrees.isleaf, children) || return false
    for child in children
        key = child.data.key
        get(labels_by_key, key, :keep) === :coarsen || return false
        last_refined = get(state.last_refined_step, key, typemin(Int))
        last_refined != typemin(Int) &&
            step - last_refined < state.options.coarsen_delay && return false
    end
    return true
end

function _apply_refinement_actions!(
        workspace::AMRWorkspace,
        labels::AbstractVector{Symbol}, step
    )
    mesh = workspace.mesh
    state = mesh.adaptation
    _collect_active_nodes!(workspace)
    length(labels) == length(workspace.leaf_nodes) || throw(
        DimensionMismatch(
            "one refinement label is required per active block"
        )
    )
    empty!(workspace.labels_by_key)
    for (index, node) in pairs(workspace.leaf_nodes)
        action = labels[index]
        _refinement_priority(action)
        workspace.labels_by_key[node.data.key] = action
    end

    for node in workspace.leaf_nodes
        key = node.data.key
        workspace.labels_by_key[key] === :refine || continue
        key.level < mesh.max_level || continue
        _split!(node)
    end

    parents = Dict{BlockKey, typeof(mesh.root)}()
    for node in workspace.leaf_nodes
        parent = RegionTrees.parent(node)
        isnothing(parent) && continue
        parents[parent.data.key] = parent
    end
    coarsened = 0
    for parent in values(parents)
        parent.data.key.level >= state.options.minimum_level || continue
        _can_coarsen(parent, workspace.labels_by_key, state, step) || continue
        parent.children = nothing
        coarsened += 1
    end
    _balance!(mesh.root, mesh.extent, mesh.periodic)
    return coarsened
end

function _copy_geometry_snapshot!(workspace::AMRWorkspace)
    empty!(workspace.old_blocks)
    append!(workspace.old_blocks, workspace.mesh.blocks)
    empty!(workspace.old_faces)
    append!(workspace.old_faces, workspace.mesh.faces)
    empty!(workspace.old_boundary_faces)
    append!(workspace.old_boundary_faces, workspace.mesh.boundary_faces)
    return workspace
end

function _rebuild_mesh_schedules!(workspace::AMRWorkspace)
    mesh = workspace.mesh
    nodes, blocks = _leaf_blocks(mesh.root, mesh.block_size)
    faces, boundaries = _build_face_schedules(
        blocks, mesh.block_size, mesh.origin, mesh.extent,
        mesh.periodic, mesh.max_level
    )
    halo_entries, halo_donors, halo_weights = _build_halo_schedule(
        mesh.root, nodes, blocks, mesh.block_size, mesh.halo,
        mesh.origin, mesh.extent, mesh.periodic
    )
    empty!(mesh.blocks)
    append!(mesh.blocks, blocks)
    empty!(mesh.faces)
    append!(mesh.faces, faces)
    empty!(mesh.boundary_faces)
    append!(mesh.boundary_faces, boundaries)
    empty!(mesh.halo_entries)
    append!(mesh.halo_entries, halo_entries)
    empty!(mesh.halo_donors)
    append!(mesh.halo_donors, halo_donors)
    empty!(mesh.halo_weights)
    append!(mesh.halo_weights, halo_weights)
    return mesh
end

function _acquire_cell_block!(buffer::CellAMRFieldBuffer)
    isempty(buffer.pool) || return pop!(buffer.pool)
    field = buffer.field
    nx, ny = field.mesh.block_size
    h = field.mesh.halo
    return Matrix{eltype(field)}(undef, nx + 2h, ny + 2h)
end

function _prolong_child!(child, parent, halo, child_key::BlockKey)
    nx = size(parent, 1) - 2halo
    ny = size(parent, 2) - 2halo
    child_x = mod(child_key.i, 2)
    child_y = mod(child_key.j, 2)
    for j in 1:ny, i in 1:nx
        fine_i = child_x * nx + i
        fine_j = child_y * ny + j
        parent_i = (fine_i + 1) ÷ 2
        parent_j = (fine_j + 1) ÷ 2
        @inbounds begin
            value = parent[parent_i + halo, parent_j + halo]
            slope_x = _mc_slope(
                value - parent[parent_i + halo - 1, parent_j + halo],
                parent[parent_i + halo + 1, parent_j + halo] - value
            )
            slope_y = _mc_slope(
                value - parent[parent_i + halo, parent_j + halo - 1],
                parent[parent_i + halo, parent_j + halo + 1] - value
            )
            offset_x = isodd(fine_i) ? -one(value) / 4 : one(value) / 4
            offset_y = isodd(fine_j) ? -one(value) / 4 : one(value) / 4
            child[i + halo, j + halo] =
                value + offset_x * slope_x + offset_y * slope_y
        end
    end
    return child
end

function _transfer_cell_field!(
        buffer::CellAMRFieldBuffer,
        workspace::AMRWorkspace
    )
    field = buffer.field
    empty!(workspace.old_indices)
    for (index, block) in pairs(workspace.old_blocks)
        workspace.old_indices[block.key] = index
    end
    empty!(workspace.new_indices)
    for (index, block) in pairs(workspace.mesh.blocks)
        workspace.new_indices[block.key] = index
    end
    resize!(workspace.reused_old_blocks, length(workspace.old_blocks))
    fill!(workspace.reused_old_blocks, false)

    for block in workspace.mesh.blocks
        old_index = get(workspace.old_indices, block.key, 0)
        if !iszero(old_index)
            push!(field.blocks, buffer.staging[old_index])
            workspace.reused_old_blocks[old_index] = true
        else
            push!(field.blocks, _acquire_cell_block!(buffer))
        end
    end

    h = workspace.mesh.halo
    for (new_index, block) in pairs(workspace.mesh.blocks)
        haskey(workspace.old_indices, block.key) && continue
        parent_key = _parent_key(block.key)
        old_parent = get(workspace.old_indices, parent_key, 0)
        if !iszero(old_parent)
            buffer.prolongation(
                field.blocks[new_index],
                buffer.staging[old_parent], h, block.key
            )
            continue
        end
        children = _child_keys(block.key)
        child_indices = ntuple(
            index -> get(workspace.old_indices, children[index], 0), 4
        )
        all(index -> !iszero(index), child_indices) || throw(
            ArgumentError(
                "mesh adaptation changed by more than one level in one update"
            )
        )
        _restrict_parent!(
            buffer.prolongation, field.blocks[new_index],
            ntuple(index -> buffer.staging[child_indices[index]], 4), h,
            block.key
        )
    end

    for index in eachindex(buffer.staging)
        workspace.reused_old_blocks[index] && continue
        push!(buffer.pool, buffer.staging[index])
    end
    empty!(buffer.staging)
    fill_halo!(field, buffer.boundary)
    return field
end

function _stage_field!(buffer::CellAMRFieldBuffer)
    _prepare_prolongation!(buffer.prolongation, buffer.field)
    fill_halo!(buffer.field, buffer.boundary)
    empty!(buffer.staging)
    append!(buffer.staging, buffer.field.blocks)
    empty!(buffer.field.blocks)
    return buffer
end

_prepare_prolongation!(prolongation, field) = nothing
_finish_prolongation!(prolongation) = nothing
_restrict_parent!(prolongation, parent, children, halo, key) =
    restrict!(parent, children, halo)

function _stage_field!(buffer::FaceAMRFieldBuffer)
    resize!(buffer.old_interior, length(buffer.field.interior))
    resize!(buffer.old_boundary, length(buffer.field.boundary))
    copyto!(buffer.old_interior, buffer.field.interior)
    copyto!(buffer.old_boundary, buffer.field.boundary)
    return buffer
end

@inline function _face_lattice_geometry(mesh::BlockTreeMesh, face)
    axis = face.axis
    tangent_axis = 3 - axis
    lattice_size = mesh.block_size .* (1 << mesh.max_level)
    normal_spacing = mesh.extent[axis] / lattice_size[axis]
    tangent_spacing = mesh.extent[tangent_axis] / lattice_size[tangent_axis]
    normal = round(
        Int, (face.center[axis] - mesh.origin[axis]) /
            normal_spacing
    )
    tangent_center = (face.center[tangent_axis] - mesh.origin[tangent_axis]) /
        tangent_spacing
    half_length = face.area / (2tangent_spacing)
    lo = round(Int, tangent_center - half_length)
    hi = round(Int, tangent_center + half_length)
    return normal, lo, hi
end

function _build_face_transfer_geometry!(workspace::AMRWorkspace)
    for axis in 1:2
        empty!(workspace.face_lines[axis])
        empty!(workspace.face_line_coordinates[axis])
    end
    local_indices = zeros(Int, 2)
    for (source_index, face) in pairs(workspace.old_faces)
        axis = face.axis
        local_indices[axis] += 1
        normal, lo, hi = _face_lattice_geometry(workspace.mesh, face)
        line = get!(workspace.face_lines[axis], normal) do
            FaceTransferSegment[]
        end
        push!(
            line, FaceTransferSegment(
                lo, hi, source_index, local_indices[axis], false
            )
        )
    end
    fill!(local_indices, 0)
    for (source_index, face) in pairs(workspace.old_boundary_faces)
        axis = face.axis
        local_indices[axis] += 1
        normal, lo, hi = _face_lattice_geometry(workspace.mesh, face)
        line = get!(workspace.face_lines[axis], normal) do
            FaceTransferSegment[]
        end
        push!(
            line, FaceTransferSegment(
                lo, hi, source_index, local_indices[axis], true
            )
        )
    end
    for axis in 1:2
        for line in values(workspace.face_lines[axis])
            sort!(line; by = segment -> segment.lo)
        end
        append!(
            workspace.face_line_coordinates[axis],
            keys(workspace.face_lines[axis])
        )
        sort!(workspace.face_line_coordinates[axis])
    end
    return workspace
end

@inline function _old_face_value(
        buffer::FaceAMRFieldBuffer,
        segment::FaceTransferSegment
    )
    field = buffer.field
    index = field isa FaceVelocity ? segment.source_index : segment.local_index
    return segment.boundary ? buffer.old_boundary[index] :
        buffer.old_interior[index]
end

@inline _segment_center(segment::FaceTransferSegment) =
    (segment.lo + segment.hi) / 2

function _tangential_slope(
        buffer::FaceAMRFieldBuffer, line,
        segment_index, tangent_period, periodic
    )
    segment = line[segment_index]
    value = _old_face_value(buffer, segment)
    centre = _segment_center(segment)
    left_index = segment_index > firstindex(line) ? segment_index - 1 :
        (periodic ? lastindex(line) : 0)
    right_index = segment_index < lastindex(line) ? segment_index + 1 :
        (periodic ? firstindex(line) : 0)

    left_gradient = nothing
    if !iszero(left_index)
        left = line[left_index]
        left_centre = _segment_center(left)
        left_centre >= centre && (left_centre -= tangent_period)
        left_gradient = (value - _old_face_value(buffer, left)) /
            (centre - left_centre)
    end
    right_gradient = nothing
    if !iszero(right_index)
        right = line[right_index]
        right_centre = _segment_center(right)
        right_centre <= centre && (right_centre += tangent_period)
        right_gradient = (_old_face_value(buffer, right) - value) /
            (right_centre - centre)
    end
    isnothing(left_gradient) && isnothing(right_gradient) && return zero(value)
    isnothing(left_gradient) && return right_gradient
    isnothing(right_gradient) && return left_gradient
    return _mc_slope(left_gradient, right_gradient)
end

function _line_value(
        buffer::FaceAMRFieldBuffer, line, lo, hi,
        tangent_period, periodic
    )
    covered = 0
    integral = zero(eltype(buffer.old_interior))
    containing = 0
    contributors = 0
    for (index, segment) in pairs(line)
        overlap = min(hi, segment.hi) - max(lo, segment.lo)
        overlap > 0 || continue
        covered += overlap
        integral += overlap * _old_face_value(buffer, segment)
        contributors += 1
        segment.lo <= lo && hi <= segment.hi && (containing = index)
    end
    covered == hi - lo || return nothing
    if contributors == 1 && !iszero(containing)
        segment = line[containing]
        value = _old_face_value(buffer, segment)
        if !buffer.piecewise_constant && (segment.lo != lo || segment.hi != hi)
            slope = _tangential_slope(
                buffer, line, containing, tangent_period, periodic
            )
            value += slope * ((lo + hi) / 2 - _segment_center(segment))
        end
        return value
    end
    return integral / covered
end

function _line_point_value(
        buffer::FaceAMRFieldBuffer, line, tangent,
        tangent_period, periodic
    )
    for (index, segment) in pairs(line)
        segment.lo <= tangent <= segment.hi || continue
        value = _old_face_value(buffer, segment)
        slope = buffer.piecewise_constant ? zero(value) : _tangential_slope(
                buffer, line, index, tangent_period, periodic
            )
        return value + slope * (tangent - _segment_center(segment))
    end
    return nothing
end

function _face_transfer_value(
        workspace::AMRWorkspace,
        buffer::FaceAMRFieldBuffer, face
    )
    mesh = workspace.mesh
    axis = face.axis
    tangent_axis = 3 - axis
    normal, lo, hi = _face_lattice_geometry(mesh, face)
    tangent_period = mesh.block_size[tangent_axis] << mesh.max_level
    lines = workspace.face_lines[axis]
    if haskey(lines, normal)
        value = _line_value(
            buffer, lines[normal], lo, hi,
            tangent_period, mesh.periodic[tangent_axis]
        )
        isnothing(value) || return value
    end

    tangent = (lo + hi) / 2
    normal_period = mesh.block_size[axis] << mesh.max_level
    lower_distance = typemin(Int)
    upper_distance = typemax(Int)
    lower_value = nothing
    upper_value = nothing
    for coordinate in workspace.face_line_coordinates[axis]
        delta = coordinate - normal
        if mesh.periodic[axis]
            delta -= round(Int, delta / normal_period) * normal_period
        end
        iszero(delta) && continue
        sampled = _line_point_value(
            buffer, lines[coordinate], tangent,
            tangent_period, mesh.periodic[tangent_axis]
        )
        isnothing(sampled) && continue
        if delta < 0 && delta > lower_distance
            lower_distance = delta
            lower_value = sampled
        elseif delta > 0 && delta < upper_distance
            upper_distance = delta
            upper_value = sampled
        end
    end
    if !isnothing(lower_value) && !isnothing(upper_value)
        fraction = -lower_distance / (upper_distance - lower_distance)
        return lower_value + fraction * (upper_value - lower_value)
    elseif !isnothing(lower_value)
        return lower_value
    elseif !isnothing(upper_value)
        return upper_value
    end
    throw(ArgumentError("could not locate old face data for a new face"))
end

@inline function _dual_face_volume(blocks, face)
    tangent_axis = 3 - face.axis
    result = zero(typeof(face.area))
    for cell in (face.minus, face.plus)
        block = blocks[cell.block]
        fraction = face.area / block.spacing[tangent_axis]
        result += fraction * block.volume / 2
    end
    return result
end

function _old_face_momentum(
        workspace::AMRWorkspace,
        buffer::FaceAMRFieldBuffer, axis
    )
    result = zero(eltype(buffer.old_interior))
    local_index = 0
    for (global_index, face) in pairs(workspace.old_faces)
        face.axis == axis || continue
        local_index += 1
        index = buffer.field isa FaceVelocity ? global_index : local_index
        result += buffer.old_interior[index] *
            _dual_face_volume(workspace.old_blocks, face)
    end
    return result
end

@inline _field_axes(::FaceVelocity) = (1, 2)
@inline _field_axes(::FaceField{Axis}) where {Axis} = (Axis,)

function _resize_face_field!(field::FaceVelocity, mesh)
    resize!(field.interior, length(mesh.faces))
    resize!(field.boundary, length(mesh.boundary_faces))
    return field
end

function _resize_face_field!(field::FaceField{Axis}, mesh) where {Axis}
    resize!(field.interior, _nfaces(mesh, Axis))
    resize!(field.boundary, _nboundary_faces(mesh, Axis))
    return field
end

function _set_transferred_interior!(
        field::FaceVelocity, global_index,
        local_index, value
    )
    return @inbounds field.interior[global_index] = value
end

function _set_transferred_interior!(
        field::FaceField, global_index,
        local_index, value
    )
    return @inbounds field.interior[local_index] = value
end

function _set_transferred_boundary!(
        field::FaceVelocity, global_index,
        local_index, value
    )
    return @inbounds field.boundary[global_index] = value
end

function _set_transferred_boundary!(
        field::FaceField, global_index,
        local_index, value
    )
    return @inbounds field.boundary[local_index] = value
end

function _conserve_face_momentum!(field, axis, target, mesh)
    current = zero(typeof(target))
    total_volume = zero(typeof(target))
    local_index = 0
    for (global_index, face) in pairs(mesh.faces)
        face.axis == axis || continue
        local_index += 1
        index = field isa FaceVelocity ? global_index : local_index
        volume = _dual_face_volume(mesh.blocks, face)
        @inbounds current += field.interior[index] * volume
        total_volume += volume
    end
    correction = (target - current) / total_volume
    local_index = 0
    for (global_index, face) in pairs(mesh.faces)
        face.axis == axis || continue
        local_index += 1
        index = field isa FaceVelocity ? global_index : local_index
        @inbounds field.interior[index] += correction
    end
    return field
end

function _transfer_face_field!(
        buffer::FaceAMRFieldBuffer,
        workspace::AMRWorkspace
    )
    field = buffer.field
    mesh = workspace.mesh
    old_momenta = map(
        axis -> _old_face_momentum(workspace, buffer, axis),
        _field_axes(field)
    )
    _resize_face_field!(field, mesh)

    local_indices = zeros(Int, 2)
    for (global_index, face) in pairs(mesh.faces)
        local_indices[face.axis] += 1
        face.axis in _field_axes(field) || continue
        value = _face_transfer_value(workspace, buffer, face)
        _set_transferred_interior!(
            field, global_index, local_indices[face.axis], value
        )
    end
    fill!(local_indices, 0)
    for (global_index, face) in pairs(mesh.boundary_faces)
        local_indices[face.axis] += 1
        face.axis in _field_axes(field) || continue
        value = _face_transfer_value(workspace, buffer, face)
        _set_transferred_boundary!(
            field, global_index, local_indices[face.axis], value
        )
    end
    field isa FaceVelocity &&
        _preserve_refined_divergence!(field, buffer, workspace)
    for (index, axis) in pairs(_field_axes(field))
        # A constant correction is divergence-free only when the component's
        # normal direction is periodic. At a physical wall its missing
        # boundary half-control-volumes make the interior-only staggered
        # integral resolution dependent; forcing that integral would spoil
        # affine exactness and the boundary-cell divergence.
        buffer.conserve_integral && mesh.periodic[axis] && _conserve_face_momentum!(
            field, axis, old_momenta[index], mesh
        )
    end
    return field
end

@inline function _cell_linear_index(ref::CellRef, nx, ny)
    return (ref.block - 1) * nx * ny + (ref.j - 1) * nx + ref.i
end

function _integrated_cell_fluxes!(
        fluxes, blocks, faces, boundary_faces,
        interior_values, boundary_values,
        block_size
    )
    nx, ny = block_size
    resize!(fluxes, length(blocks) * nx * ny)
    fill!(fluxes, zero(eltype(fluxes)))
    for (index, face) in pairs(faces)
        @inbounds flux = interior_values[index] * face.area
        minus = _cell_linear_index(face.minus, nx, ny)
        plus = _cell_linear_index(face.plus, nx, ny)
        @inbounds begin
            fluxes[minus] += flux
            fluxes[plus] -= flux
        end
    end
    for (index, face) in pairs(boundary_faces)
        flux = face.side * boundary_values[index] * face.area
        cell = _cell_linear_index(face.cell, nx, ny)
        @inbounds fluxes[cell] += flux
    end
    return fluxes
end

function _build_new_face_indices!(workspace::AMRWorkspace)
    empty!(workspace.new_face_indices)
    for (index, face) in pairs(workspace.mesh.faces)
        normal, lo, hi = _face_lattice_geometry(workspace.mesh, face)
        workspace.new_face_indices[(face.axis, normal, lo, hi)] = index
    end
    return workspace.new_face_indices
end

function _current_cellref(workspace::AMRWorkspace, x, y)
    mesh = workspace.mesh
    point = SVector(
        mesh.origin[1] + x * mesh.extent[1] /
            (mesh.block_size[1] << mesh.max_level),
        mesh.origin[2] + y * mesh.extent[2] /
            (mesh.block_size[2] << mesh.max_level)
    )
    node = RegionTrees.findleaf(mesh.root, point)
    block_index = workspace.new_indices[node.data.key]
    block = mesh.blocks[block_index]
    i = clamp(
        floor(Int, (point[1] - block.origin[1]) / block.spacing[1]) + 1,
        1, mesh.block_size[1]
    )
    j = clamp(
        floor(Int, (point[2] - block.origin[2]) / block.spacing[2]) + 1,
        1, mesh.block_size[2]
    )
    return CellRef(block_index, i, j)
end

function _correct_internal_face!(
        velocity::FaceVelocity, workspace::AMRWorkspace,
        key, integrated_correction
    )
    index = workspace.new_face_indices[key]
    face = workspace.mesh.faces[index]
    @inbounds velocity.interior[index] += integrated_correction / face.area
    return nothing
end

function _preserve_refined_divergence!(
        velocity::FaceVelocity,
        buffer::FaceAMRFieldBuffer,
        workspace::AMRWorkspace
    )
    mesh = workspace.mesh
    nx, ny = mesh.block_size
    empty!(workspace.old_indices)
    for (index, block) in pairs(workspace.old_blocks)
        workspace.old_indices[block.key] = index
    end
    empty!(workspace.new_indices)
    for (index, block) in pairs(mesh.blocks)
        workspace.new_indices[block.key] = index
    end
    _build_new_face_indices!(workspace)
    _integrated_cell_fluxes!(
        workspace.old_cell_flux, workspace.old_blocks,
        workspace.old_faces, workspace.old_boundary_faces,
        buffer.old_interior, buffer.old_boundary, mesh.block_size
    )
    _integrated_cell_fluxes!(
        workspace.new_cell_flux, mesh.blocks, mesh.faces, mesh.boundary_faces,
        velocity.interior, velocity.boundary, mesh.block_size
    )

    for (old_block_index, block) in pairs(workspace.old_blocks)
        haskey(workspace.new_indices, block.key) && continue
        children = _child_keys(block.key)
        all(haskey(workspace.new_indices, child) for child in children) || continue
        scale = 1 << (mesh.max_level - block.key.level)
        fine_scale = scale ÷ 2
        block_x = block.key.i * nx * scale
        block_y = block.key.j * ny * scale
        for j in 1:ny, i in 1:nx
            xlo = block_x + (i - 1) * scale
            ylo = block_y + (j - 1) * scale
            xmid, ymid = xlo + fine_scale, ylo + fine_scale
            xhi, yhi = xlo + scale, ylo + scale
            fine_cells = (
                _current_cellref(
                    workspace, xlo + fine_scale / 2,
                    ylo + fine_scale / 2
                ),
                _current_cellref(
                    workspace, xmid + fine_scale / 2,
                    ylo + fine_scale / 2
                ),
                _current_cellref(
                    workspace, xlo + fine_scale / 2,
                    ymid + fine_scale / 2
                ),
                _current_cellref(
                    workspace, xmid + fine_scale / 2,
                    ymid + fine_scale / 2
                ),
            )
            old_ref = CellRef(old_block_index, i, j)
            target = workspace.old_cell_flux[
                _cell_linear_index(old_ref, nx, ny),
            ] / 4
            residuals = ntuple(
                index -> target - workspace.new_cell_flux[
                    _cell_linear_index(fine_cells[index], nx, ny),
                ], 4
            )

            # Incidence-minimum-norm correction for the two vertical and two
            # horizontal internal fine faces. The fourth equation follows
            # from conservation of the transferred parent boundary flux.
            vertical_lower =
                (2residuals[1] + residuals[3] - residuals[2]) / 4
            vertical_upper = residuals[3] + residuals[1] - vertical_lower
            horizontal_left = residuals[1] - vertical_lower
            horizontal_right = residuals[2] + vertical_lower
            _correct_internal_face!(
                velocity, workspace, (1, xmid, ylo, ymid), vertical_lower
            )
            _correct_internal_face!(
                velocity, workspace, (1, xmid, ymid, yhi), vertical_upper
            )
            _correct_internal_face!(
                velocity, workspace, (2, ymid, xlo, xmid), horizontal_left
            )
            _correct_internal_face!(
                velocity, workspace, (2, ymid, xmid, xhi), horizontal_right
            )
        end
    end
    return velocity
end

function _mesh_topology_changed(workspace::AMRWorkspace)
    length(workspace.leaf_nodes) == length(workspace.old_blocks) || return true
    for (node, block) in zip(workspace.leaf_nodes, workspace.old_blocks)
        node.data.key == block.key || return true
    end
    return false
end

function _actual_adaptation_counts!(workspace::AMRWorkspace, step)
    old_keys = workspace.old_indices
    new_keys = workspace.new_indices
    empty!(old_keys)
    for (index, block) in pairs(workspace.old_blocks)
        old_keys[block.key] = index
    end
    empty!(new_keys)
    for (index, block) in pairs(workspace.mesh.blocks)
        new_keys[block.key] = index
    end
    refined_parents = Set{BlockKey}()
    coarsened_parents = 0
    for block in workspace.mesh.blocks
        haskey(old_keys, block.key) && continue
        parent = _parent_key(block.key)
        if haskey(old_keys, parent)
            push!(refined_parents, parent)
            workspace.mesh.adaptation.last_refined_step[block.key] = step
        elseif all(haskey(old_keys, child) for child in _child_keys(block.key))
            coarsened_parents += 1
        end
    end
    return length(refined_parents), coarsened_parents
end

function _adapt_with_labels!(
        workspace::AMRWorkspace,
        labels::AbstractVector{Symbol}; step, force = false
    )
    mesh = workspace.mesh
    state = mesh.adaptation
    step = Int(step)
    step >= 0 || throw(ArgumentError("adaptation step must be nonnegative"))
    if !force && !isnothing(state.last_update_step) &&
            step - state.last_update_step < state.options.update_interval
        return AMRStats(false, false, nblocks(mesh), nblocks(mesh), 0, 0)
    end
    state.last_update_step = step
    old_count = nblocks(mesh)
    _copy_geometry_snapshot!(workspace)
    _apply_refinement_actions!(workspace, labels, step)
    _collect_active_nodes!(workspace)
    if !_mesh_topology_changed(workspace)
        return AMRStats(true, false, old_count, old_count, 0, 0)
    end

    for buffer in workspace.field_buffers
        _stage_field!(buffer)
    end
    _build_face_transfer_geometry!(workspace)
    _rebuild_mesh_schedules!(workspace)
    for buffer in workspace.field_buffers
        if buffer isa CellAMRFieldBuffer
            _transfer_cell_field!(buffer, workspace)
        else
            _transfer_face_field!(buffer, workspace)
        end
    end
    for buffer in workspace.field_buffers
        buffer isa CellAMRFieldBuffer && _finish_prolongation!(buffer.prolongation)
    end
    refined, coarsened = _actual_adaptation_counts!(workspace, step)
    return AMRStats(true, true, old_count, nblocks(mesh), refined, coarsened)
end

"""
    adapt!(workspace, indicators; step, time=0, force=false)
    adapt!(workspace, labels::Vector{Symbol}; step, force=false)

Adapt the workspace mesh in place and transfer every registered field.
Indicator proposals are combined with `:refine > :keep > :coarsen` priority.
Coarsening requires all four active siblings to request it, respects the mesh
cooldown and minimum level, and is followed by refinement-only 2:1 balancing.
The mesh's update interval is enforced unless `force=true`.
"""
function adapt!(
        workspace::AMRWorkspace, indicators;
        step, time = 0, force = false
    )
    for indicator in indicators
        indicator isa AbstractRefinementIndicator || throw(
            ArgumentError(
                "adapt! expects refinement indicators or a Symbol label vector"
            )
        )
        _register_indicator_field!(workspace, indicator)
    end
    labels = refinement_labels!(workspace, indicators; time)
    return _adapt_with_labels!(workspace, labels; step, force)
end

adapt!(
    workspace::AMRWorkspace, indicator::AbstractRefinementIndicator;
    step, time = 0, force = false
) = adapt!(workspace, (indicator,); step, time, force)

function adapt!(
        workspace::AMRWorkspace, labels::AbstractVector{Symbol};
        step, force = false
    )
    return _adapt_with_labels!(workspace, labels; step, force)
end
