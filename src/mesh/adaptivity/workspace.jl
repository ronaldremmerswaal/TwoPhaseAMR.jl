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

"""Finite AMR extension scratch; never aliases owned intrinsic velocities."""
struct PhaseTransferScratch{T}
    values::Vector{T}
    original::Vector{T}
    next::Vector{T}
    known::BitVector
    candidates::Vector{Int}
end

PhaseTransferScratch(::Type{T}) where {T} =
    PhaseTransferScratch(T[], T[], T[], BitVector(), Int[])

mutable struct CenteredVelocityWorkspace{T}
    weighted::Matrix{T}
    measures::Matrix{T}
    x::Vector{T}
    y::Vector{T}
end

CenteredVelocityWorkspace(::Type{T}) where {T} =
    CenteredVelocityWorkspace(zeros(T, 4, 0), zeros(T, 4, 0), T[], T[])

"""Reusable curl fit data; the original valid-donor mask is frozen per call."""
struct VorticityWorkspace{T}
    centered::CenteredVelocityWorkspace{T}
    known::BitVector
    fit::NTuple{7, Vector{T}}
    values::Vector{T}
    candidates::Vector{Int}
end

VorticityWorkspace(::Type{T}) where {T} = VorticityWorkspace(
    CenteredVelocityWorkspace(T), BitVector(), ntuple(_ -> T[], 7), T[], Int[])

"""Reusable buffers and topology scratch for in-place mesh adaptation."""
mutable struct AMRWorkspace{T, M <: BlockTreeMesh{T}, N}
    mesh::M
    old_blocks::Vector{LeafBlock{T}}
    old_faces::Vector{InteriorFace{T}}
    old_boundary_faces::Vector{BoundaryFace{T}}
    leaf_nodes::Vector{N}
    nodes_by_key::Dict{BlockKey, N}
    parents_by_key::Dict{BlockKey, N}
    old_indices::Dict{BlockKey, Int}
    new_indices::Dict{BlockKey, Int}
    labels_by_key::Dict{BlockKey, Symbol}
    labels::Vector{Symbol}
    indicator_labels::Vector{Symbol}
    reused_old_blocks::BitVector
    field_buffers::Vector{AbstractAMRFieldBuffer}
    face_lines::NTuple{2, Dict{Int, Vector{FaceTransferSegment}}}
    face_line_pool::NTuple{2, Vector{Vector{FaceTransferSegment}}}
    face_line_coordinates::NTuple{2, Vector{Int}}
    new_face_indices::Dict{NTuple{4, Int}, Int}
    old_cell_flux::Vector{T}
    new_cell_flux::Vector{T}
    interior_face_local_indices::Vector{Int}
    boundary_face_local_indices::Vector{Int}
    refined_parents::Set{BlockKey}
    phase_velocities::Vector{Any}
    phase_transfer::PhaseTransferScratch{T}
    vorticity::VorticityWorkspace{T}
end

function AMRWorkspace(mesh::M, fields...) where {T, M <: BlockTreeMesh{T}}
    N = _tree_node_type(mesh.root)
    workspace = AMRWorkspace{T, M, N}(
        mesh, LeafBlock{T}[], InteriorFace{T}[], BoundaryFace{T}[], N[],
        Dict{BlockKey, N}(), Dict{BlockKey, N}(), Dict{BlockKey, Int}(),
        Dict{BlockKey, Int}(),
        Dict{BlockKey, Symbol}(), Symbol[], Symbol[], BitVector(),
        AbstractAMRFieldBuffer[],
        (
            Dict{Int, Vector{FaceTransferSegment}}(),
            Dict{Int, Vector{FaceTransferSegment}}(),
        ),
        (
            Vector{FaceTransferSegment}[],
            Vector{FaceTransferSegment}[],
        ),
        (Int[], Int[]), Dict{NTuple{4, Int}, Int}(), T[], T[], Int[], Int[],
        Set{BlockKey}(), Any[], PhaseTransferScratch(T), VorticityWorkspace(T)
    )
    for field in fields
        register_adaptive_field!(workspace, field)
    end
    _associate_adaptive_phase_velocities!(workspace, fields)
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
        _adaptive_refinement_labels!(workspace, indicator, time)
        @_parallel for index in 1:length(workspace.labels)
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

@inline function _parent_key(key::BlockKey)
    key.level > 0 || throw(ArgumentError("a base-root key has no parent"))
    return BlockKey(key.level - 1, key.i ÷ 2, key.j ÷ 2)
end

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

    parents = workspace.parents_by_key
    empty!(parents)
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
