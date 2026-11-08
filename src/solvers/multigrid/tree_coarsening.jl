@inline function _ancestor_key(key::BlockKey, level::Int)
    shift = key.level - level
    shift >= 0 || throw(ArgumentError("ancestor level exceeds block level"))
    return BlockKey(level, key.i >> shift, key.j >> shift)
end

function _capped_tree_mesh(mesh::BlockTreeMesh, maximum_active_level::Int)
    maximum_active_level >= 0 || throw(
        ArgumentError("tree hierarchy level must be nonnegative"),
    )
    internal = Set{BlockKey}()
    for block in mesh.blocks
        upper = min(block.key.level - 1, maximum_active_level - 1)
        for level in 0:upper
            push!(internal, _ancestor_key(block.key, level))
        end
    end
    return BlockTreeMesh(
        base_mesh = mesh.base_mesh,
        block_size = mesh.block_size,
        halo = mesh.halo,
        origin = mesh.origin,
        extent = mesh.extent,
        periodic = mesh.periodic,
        max_level = maximum_active_level,
        refine = (key, origin, extent) -> key in internal,
    )
end

function _all_leaf_pruned_mesh(mesh::BlockTreeMesh)
    internal = Set{BlockKey}()
    function retain_internal(node)
        RegionTrees.isleaf(node) && return
        children = RegionTrees.children(node)
        all(RegionTrees.isleaf, children) && return
        push!(internal, node.data.key)
        for child in children
            retain_internal(child)
        end
        return
    end
    for root in mesh.root.roots
        retain_internal(root)
    end
    maximum_active_level = isempty(internal) ? 0 :
        maximum(key.level for key in internal) + 1
    return BlockTreeMesh(
        base_mesh = mesh.base_mesh,
        block_size = mesh.block_size,
        halo = mesh.halo,
        origin = mesh.origin,
        extent = mesh.extent,
        periodic = mesh.periodic,
        max_level = maximum_active_level,
        refine = (key, origin, extent) -> key in internal,
    )
end

# Once the adaptive tree is fully pruned, merge 2×2 root blocks without
# changing the block layout within a leaf. Both axes must remain divisible by
# the block size; there is no anisotropic coarsening.
function _coarsened_base_mesh(mesh::BlockTreeMesh)
    all(iseven, mesh.root.layout) || return nothing
    coarse_base = mesh.base_mesh .÷ 2
    # With only two cells on a periodic axis, opposite cell centres are
    # exactly half a period apart. The current gradient metric's wrapped
    # displacement cannot disambiguate the face orientation (issue #36).
    for axis in 1:2
        mesh.periodic[axis] && coarse_base[axis] < 4 && return nothing
    end
    return BlockTreeMesh(
        base_mesh = coarse_base,
        block_size = mesh.block_size,
        halo = mesh.halo,
        origin = mesh.origin,
        extent = mesh.extent,
        periodic = mesh.periodic,
        max_level = 0,
    )
end

function _same_tree_topology(first::BlockTreeMesh, second::BlockTreeMesh)
    eltype(first.origin) === eltype(second.origin) || return false
    first.base_mesh == second.base_mesh || return false
    first.block_size == second.block_size || return false
    first.halo == second.halo || return false
    first.origin == second.origin || return false
    first.extent == second.extent || return false
    first.periodic == second.periodic || return false
    length(first.blocks) == length(second.blocks) || return false
    length(first.faces) == length(second.faces) || return false
    length(first.boundary_faces) == length(second.boundary_faces) || return false

    for index in eachindex(first.blocks)
        a = @inbounds first.blocks[index]
        b = @inbounds second.blocks[index]
        a.key == b.key && isequal(a.origin, b.origin) &&
            isequal(a.extent, b.extent) && isequal(a.spacing, b.spacing) &&
            isequal(a.volume, b.volume) || return false
    end

    # Face-indexed geometry and ordering are part of the operator schedule,
    # so compare them even when the active leaf keys agree.
    for index in eachindex(first.faces)
        a = @inbounds first.faces[index]
        b = @inbounds second.faces[index]
        a.minus == b.minus && a.plus == b.plus && a.axis == b.axis &&
            isequal(a.center, b.center) && isequal(a.area, b.area) ||
            return false
    end
    first.face_dual_volumes == second.face_dual_volumes || return false
    for index in eachindex(first.boundary_faces)
        a = @inbounds first.boundary_faces[index]
        b = @inbounds second.boundary_faces[index]
        a.cell == b.cell && a.axis == b.axis && a.side == b.side &&
            isequal(a.center, b.center) && isequal(a.area, b.area) ||
            return false
    end
    return true
end

function _matching_tree_level(operators, mesh, first_index = 2)
    for index in first_index:length(operators)
        # The finest stored operator shares the live mesh, which has already
        # been mutated when this search runs after adaptation. Its matrix still
        # represents the old topology and must not be mistaken for a match.
        operators[index].mesh === mesh && continue
        _same_tree_topology(operators[index].mesh, mesh) && return index
    end
    return 0
end

@inline function _validate_tree_pruning(pruning)
    pruning in (:deepest, :all) || throw(
        ArgumentError("tree pruning must be :deepest or :all"),
    )
    return pruning
end

# This plan carries only topology decisions. The builders retain their own
# level data and request a suffix match at the point where that data is ready
# to be constructed or refreshed.
mutable struct _TreeCoarseningPlan{Mode, P, S, L}
    pruning::P
    target_size::S
    max_levels::L
    active_level::Int
    reused_new_start::Int
    reused_old_start::Int
end

function _TreeCoarseningPlan(
        mesh, pruning, target_size, max_levels, ::Val{Mode},
    ) where {Mode}
    _validate_tree_pruning(pruning)
    return _TreeCoarseningPlan{
        Mode, typeof(pruning), typeof(target_size), typeof(max_levels),
    }(
        pruning, target_size, max_levels,
        maximum(block.key.level for block in mesh.blocks), 0, 0,
    )
end

@inline _tree_can_reuse(plan::_TreeCoarseningPlan, reuse) =
    !isnothing(reuse) && reuse.pruning === plan.pruning

@inline _tree_progress_size(::_TreeCoarseningPlan{:pressure}, mesh) =
    ncells(mesh)
@inline _tree_progress_size(::_TreeCoarseningPlan, mesh) = length(mesh.faces)

# Tree-level stopping retains each solver's unknown-count convention. Below
# the root, the target describes base-mesh cells for all three hierarchies;
# the coupled face system passes twice that target to the plan.
@inline _tree_base_target_size(plan::_TreeCoarseningPlan) = plan.target_size
@inline _tree_base_target_size(plan::_TreeCoarseningPlan{:variable}) =
    plan.target_size ÷ 2
@inline _tree_above_target(plan::_TreeCoarseningPlan, mesh, current_size) =
    iszero(plan.active_level) ?
    ncells(mesh) > _tree_base_target_size(plan) : current_size > plan.target_size

@inline _tree_stop_on_no_progress(::_TreeCoarseningPlan{:staggered}) = true
@inline _tree_stop_on_no_progress(plan::_TreeCoarseningPlan) =
    plan.pruning === :all

function _tree_next_coarse_mesh!(
        plan::_TreeCoarseningPlan, fine_mesh, current_mesh,
        current_size, number_of_levels,
    )
    while iszero(plan.reused_new_start) &&
            _tree_above_target(plan, current_mesh, current_size) &&
            number_of_levels < plan.max_levels
        coarse_mesh = if iszero(plan.active_level)
            _coarsened_base_mesh(current_mesh)
        elseif plan.pruning === :deepest
            plan.active_level -= 1
            _capped_tree_mesh(fine_mesh, plan.active_level)
        else
            _all_leaf_pruned_mesh(current_mesh)
        end
        isnothing(coarse_mesh) && break
        if _tree_progress_size(plan, coarse_mesh) >=
                _tree_progress_size(plan, current_mesh)
            _tree_stop_on_no_progress(plan) && break
            continue
        end
        plan.active_level = maximum(block.key.level for block in coarse_mesh.blocks)
        return coarse_mesh
    end
    return nothing
end

function _tree_reuse_suffix!(
        plan::_TreeCoarseningPlan, old_operators, mesh, preceding_levels,
        eligible = true,
    )
    eligible && !isnothing(old_operators) || return 0
    matching_level = _matching_tree_level(old_operators, mesh, 1)
    iszero(matching_level) && return 0
    suffix_length = length(old_operators) - matching_level + 1
    preceding_levels + suffix_length <= plan.max_levels || return 0
    plan.reused_new_start = preceding_levels + 1
    plan.reused_old_start = matching_level
    return matching_level
end

function _tree_cell_prolongation(
        fine::BlockTreeMesh, coarse::BlockTreeMesh, ::Type{T},
    ) where {T}
    coarse_blocks = Dict(
        block.key => index for (index, block) in pairs(coarse.blocks)
    )
    nx, ny = fine.block_size
    rows = collect(1:ncells(fine))
    columns = Vector{Int}(undef, length(rows))
    for (fine_block, block) in pairs(fine.blocks), j in 1:ny, i in 1:nx
        point = SVector(cellcenter(fine, fine_block, i, j))
        node = _findleaf(coarse.root, point)
        coarse_block = coarse_blocks[node.data.key]
        target = coarse.blocks[coarse_block]
        coarse_i = clamp(
            floor(Int, (point[1] - target.origin[1]) / target.spacing[1]) + 1,
            1, nx,
        )
        coarse_j = clamp(
            floor(Int, (point[2] - target.origin[2]) / target.spacing[2]) + 1,
            1, ny,
        )
        fine_index = (fine_block - 1) * nx * ny + (j - 1) * nx + i
        columns[fine_index] =
            (coarse_block - 1) * nx * ny + (coarse_j - 1) * nx + coarse_i
    end
    return sparse(
        rows, columns, ones(T, length(rows)), ncells(fine), ncells(coarse),
    )
end

"""Volume-average a cell coefficient between two nested tree meshes."""
struct CellCoefficientRestriction{T}
    row_pointers::Vector{Int}
    fine_indices::Vector{Int}
    weights::Vector{T}
end

function _cell_coefficient_restriction(
        coarse::BlockTreeMesh, fine::BlockTreeMesh, ::Type{T},
    ) where {T}
    prolongation = _tree_cell_prolongation(fine, coarse, T)
    fine_to_coarse = zeros(Int, size(prolongation, 1))
    rows = rowvals(prolongation)
    for column in axes(prolongation, 2), pointer in nzrange(prolongation, column)
        row = rows[pointer]
        iszero(fine_to_coarse[row]) || throw(
            ArgumentError("tree cell restriction has multiple owners"),
        )
        fine_to_coarse[row] = column
    end
    all(!iszero, fine_to_coarse) || throw(
        ArgumentError("tree cell restriction leaves a cell unowned"),
    )

    counts = zeros(Int, ncells(coarse))
    for owner in fine_to_coarse
        @inbounds counts[owner] += 1
    end
    row_pointers = Vector{Int}(undef, ncells(coarse) + 1)
    row_pointers[1] = 1
    for row in eachindex(counts)
        @inbounds row_pointers[row + 1] = row_pointers[row] + counts[row]
    end
    fine_indices = Vector{Int}(undef, ncells(fine))
    weights = Vector{T}(undef, ncells(fine))
    positions = copy(@view(row_pointers[1:(end - 1)]))
    nx, ny = fine.block_size
    coarse_nx, coarse_ny = coarse.block_size
    for (fine_block, block) in pairs(fine.blocks), j in 1:ny, i in 1:nx
        fine_index = (fine_block - 1) * nx * ny + (j - 1) * nx + i
        owner = fine_to_coarse[fine_index]
        coarse_block = fld(owner - 1, coarse_nx * coarse_ny) + 1
        position = positions[owner]
        @inbounds fine_indices[position] = fine_index
        @inbounds weights[position] = T(block.volume) /
            T(coarse.blocks[coarse_block].volume)
        @inbounds positions[owner] += 1
    end
    for row in 1:ncells(coarse)
        total = sum(
            @view(weights[row_pointers[row]:(row_pointers[row + 1] - 1)]),
        )
        isapprox(total, one(T); rtol = 128eps(T), atol = 128eps(T)) ||
            error("tree cell restriction does not partition a coarse cell")
    end
    return CellCoefficientRestriction(row_pointers, fine_indices, weights)
end

function _restrict_cell_coefficient!(
        coarse_coefficient::Vector{T}, fine_coefficient::Vector{T},
        restriction::CellCoefficientRestriction{T},
    ) where {T}
    length(coarse_coefficient) + 1 == length(restriction.row_pointers) ||
        throw(DimensionMismatch("coarse cell restriction size differs"))
    @_parallel for coarse_index in 1:length(coarse_coefficient)
        value = zero(T)
        for pointer in restriction.row_pointers[coarse_index]:(
                restriction.row_pointers[coarse_index + 1] - 1
            )
            @inbounds value += restriction.weights[pointer] *
                fine_coefficient[restriction.fine_indices[pointer]]
        end
        @inbounds coarse_coefficient[coarse_index] = value
    end
    return coarse_coefficient
end

@inline function _face_lattice_geometry(
        mesh::BlockTreeMesh, face, lattice_size,
    )
    axis = face.axis
    tangent_axis = 3 - axis
    normal_spacing = mesh.extent[axis] / lattice_size[axis]
    tangent_spacing = mesh.extent[tangent_axis] / lattice_size[tangent_axis]
    normal = round(
        Int, (face.center[axis] - mesh.origin[axis]) / normal_spacing,
    )
    tangent_center =
        (face.center[tangent_axis] - mesh.origin[tangent_axis]) /
        tangent_spacing
    half_length = face.area / (2tangent_spacing)
    return normal,
        round(Int, tangent_center - half_length),
        round(Int, tangent_center + half_length)
end

# Face coordinates for nested meshes must be expressed on the same physical
# lattice. Below the root level their base_mesh values differ, even though
# both meshes have max_level == 0.
@inline function _tree_common_lattice_size(fine, coarse)
    fine_size = _cells_at_level(fine, fine.max_level)
    coarse_size = _cells_at_level(coarse, coarse.max_level)
    return ntuple(axis -> max(fine_size[axis], coarse_size[axis]), 2)
end

struct FaceCoefficientRestriction{T}
    row_pointers::Vector{Int}
    fine_indices::Vector{Int}
    weights::Vector{T}
end

function _face_coefficient_restriction(
        coarse::BlockTreeMesh, fine::BlockTreeMesh, ::Type{T},
    ) where {T}
    lattice_size = _tree_common_lattice_size(fine, coarse)
    lines = Dict{Tuple{Int, Int}, Vector{NTuple{3, Int}}}()
    for (index, face) in pairs(fine.faces)
        normal, lo, hi = _face_lattice_geometry(
            fine, face, lattice_size,
        )
        line = get!(lines, (face.axis, normal)) do
            NTuple{3, Int}[]
        end
        push!(line, (lo, hi, index))
    end
    for segments in values(lines)
        sort!(segments; by = first)
    end
    row_pointers = Vector{Int}(undef, length(coarse.faces) + 1)
    fine_indices = Int[]
    weights = T[]
    empty_line = NTuple{3, Int}[]
    row_pointers[1] = 1
    for (index, face) in pairs(coarse.faces)
        normal, lo, hi = _face_lattice_geometry(
            coarse, face, lattice_size,
        )
        covered = 0
        line = get(lines, (face.axis, normal), empty_line)
        pointer = _first_overlapping_line_segment(line, lo)
        while pointer <= lastindex(line)
            fine_lo, fine_hi, fine_index = line[pointer]
            fine_lo >= hi && break
            overlap = min(hi, fine_hi) - max(lo, fine_lo)
            if overlap > 0
                push!(fine_indices, fine_index)
                push!(weights, T(overlap) / T(hi - lo))
                covered += overlap
            end
            pointer += 1
        end
        covered == hi - lo || error(
            "tree hierarchy could not cover a coarse face with fine faces",
        )
        row_pointers[index + 1] = length(fine_indices) + 1
    end
    return FaceCoefficientRestriction(
        row_pointers, fine_indices, weights,
    )
end

struct TreeFaceSegment
    lo::Int
    hi::Int
    index::Int
end

function _tree_face_lines(mesh::BlockTreeMesh, lattice_size)
    lines = ntuple(
        axis -> Dict{Int, Vector{TreeFaceSegment}}(), 2,
    )
    for (index, face) in pairs(mesh.faces)
        normal, lo, hi = _face_lattice_geometry(
            mesh, face, lattice_size,
        )
        line = get!(lines[face.axis], normal) do
            TreeFaceSegment[]
        end
        push!(line, TreeFaceSegment(lo, hi, index))
    end
    for axis in 1:2, line in values(lines[axis])
        sort!(line; by = segment -> segment.lo)
    end
    coordinates = ntuple(axis -> sort!(collect(keys(lines[axis]))), 2)
    return lines, coordinates
end

function _append_tree_line_average_weights!(
        rows, columns, values, row, line, lo, hi, ::Type{T},
    ) where {T}
    first_entry = length(rows) + 1
    covered = 0
    pointer = _first_overlapping_line_segment(line, lo)
    while pointer <= lastindex(line)
        segment = line[pointer]
        segment.lo >= hi && break
        overlap = min(hi, segment.hi) - max(lo, segment.lo)
        if overlap > 0
            covered += overlap
            push!(rows, row)
            push!(columns, segment.index)
            push!(values, T(overlap) / T(hi - lo))
        end
        pointer += 1
    end
    if covered == hi - lo
        return true
    end
    resize!(rows, first_entry - 1)
    resize!(columns, first_entry - 1)
    resize!(values, first_entry - 1)
    return false
end

"""Linear staggered interpolation from a pruned tree to its finer tree.

Coincident faces use area-overlap interpolation in the tangential direction.
New faces inside a coarse dual cell interpolate between the two neighbouring
coarse face planes. At a physical wall the missing normal velocity is the
homogeneous boundary correction. Periodic distances wrap on the common
integer lattice. The construction is topology-only and preserves constant
velocity exactly away from a physical normal wall.
"""
function _tree_face_prolongation(
        fine::BlockTreeMesh, coarse::BlockTreeMesh, ::Type{T},
    ) where {T}
    lattice_size = _tree_common_lattice_size(fine, coarse)
    lines, coordinates = _tree_face_lines(coarse, lattice_size)
    rows = Int[]
    columns = Int[]
    values = T[]
    sizehint!(rows, 2length(fine.faces))
    sizehint!(columns, 2length(fine.faces))
    sizehint!(values, 2length(fine.faces))

    for (row, face) in pairs(fine.faces)
        axis = face.axis
        normal, lo, hi = _face_lattice_geometry(
            fine, face, lattice_size,
        )
        exact = get(lines[axis], normal, nothing)
        if !isnothing(exact)
            if _append_tree_line_average_weights!(
                    rows, columns, values, row, exact, lo, hi, T,
                )
                continue
            end
        end

        tangent = T(lo + hi) / 2
        normal_period = lattice_size[axis]
        lower_distance, lower_coordinate, lower_segment,
            upper_distance, upper_coordinate, upper_segment =
            _bracketing_line_segments(
            lines[axis], coordinates[axis], normal, tangent,
            normal_period, coarse.periodic[axis],
        )
        lower_index = iszero(lower_segment) ? 0 :
            lines[axis][lower_coordinate][lower_segment].index
        upper_index = iszero(upper_segment) ? 0 :
            lines[axis][upper_coordinate][upper_segment].index

        if !coarse.periodic[axis]
            if iszero(lower_index)
                lower_distance = -normal
            end
            if iszero(upper_index)
                upper_distance = normal_period - normal
            end
        end
        if lower_distance != typemin(Int) &&
                upper_distance != typemax(Int)
            fraction = T(-lower_distance) /
                T(upper_distance - lower_distance)
            if !iszero(lower_index) && fraction < one(T)
                push!(rows, row)
                push!(columns, lower_index)
                push!(values, one(T) - fraction)
            end
            if !iszero(upper_index) && fraction > zero(T)
                push!(rows, row)
                push!(columns, upper_index)
                push!(values, fraction)
            end
        elseif !iszero(lower_index)
            push!(rows, row); push!(columns, lower_index); push!(values, one(T))
        elseif !iszero(upper_index)
            push!(rows, row); push!(columns, upper_index); push!(values, one(T))
        else
            error("tree hierarchy could not interpolate a fine face")
        end
    end
    prolongation = sparse(
        rows, columns, values, length(fine.faces), length(coarse.faces),
    )
    dropzeros!(prolongation)
    return prolongation
end

function _restrict_face_coefficients!(
        coarse_coefficient::Vector{T}, fine_coefficient::Vector{T},
        restriction::FaceCoefficientRestriction{T},
    ) where {T}
    length(coarse_coefficient) + 1 == length(restriction.row_pointers) ||
        throw(DimensionMismatch("coarse face restriction size differs"))
    @_parallel for coarse_index in 1:length(coarse_coefficient)
        value = zero(T)
        for pointer in restriction.row_pointers[coarse_index]:(
                restriction.row_pointers[coarse_index + 1] - 1
            )
            @inbounds value += restriction.weights[pointer] *
                fine_coefficient[restriction.fine_indices[pointer]]
        end
        @inbounds coarse_coefficient[coarse_index] = value
    end
    return coarse_coefficient
end

function _restrict_geometry_material!(
        coarse_material::Vector{T}, fine::EllipticOperator{T},
        fine_geometry, coarse_geometry,
        restriction::FaceCoefficientRestriction{T},
    ) where {T}
    fallback_sum = zero(T)
    fallback_weight = zero(T)
    for index in eachindex(fine.face_coefficient)
        aperture = fine_geometry.aperture[index]
        aperture > zero(T) || continue
        material = fine.face_coefficient[index] / aperture^2
        fallback_sum += aperture * material
        fallback_weight += aperture
    end
    fallback = fallback_weight > zero(T) ?
        fallback_sum / fallback_weight : one(T)

    @_parallel for coarse_index in eachindex(coarse_material)
        numerator = zero(T)
        denominator = zero(T)
        for pointer in restriction.row_pointers[coarse_index]:(
                restriction.row_pointers[coarse_index + 1] - 1
            )
            fine_index = @inbounds restriction.fine_indices[pointer]
            aperture = @inbounds fine_geometry.aperture[fine_index]
            aperture > zero(T) || continue
            weight = @inbounds restriction.weights[pointer] * aperture
            numerator += weight *
                fine.face_coefficient[fine_index] / aperture^2
            denominator += weight
        end
        @inbounds coarse_material[coarse_index] =
            coarse_geometry.aperture[coarse_index] > zero(T) ?
            (denominator > zero(T) ? numerator / denominator : fallback) :
            zero(T)
    end
    return coarse_material
end
