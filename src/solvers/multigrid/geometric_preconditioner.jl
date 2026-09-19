
"""Convergence statistics from the latest external Krylov solve."""
mutable struct LinearSolverStats{T}
    cycles::Int
    converged::Bool
    initial_l2::T
    initial_linf::T
    final_l2::T
    final_linf::T
    target_linf::T
    residual_scale::T
    removed_rhs_mean::T
    method::Symbol
    preconditioner::Symbol
end

# Preserve the public name used by earlier releases.
"""Compatibility alias for [`LinearSolverStats`](@ref)."""
const MultigridStats = LinearSolverStats

LinearSolverStats(::Type{T}; method = :none, preconditioner = :none) where {T} =
    LinearSolverStats(
    0, false, T(Inf), T(Inf), T(Inf), T(Inf), T(Inf), one(T), zero(T),
    method, preconditioner
)

function Base.show(io::IO, stats::LinearSolverStats)
    return print(
        io,
        "LinearSolverStats(",
        "cycles=", stats.cycles,
        ", converged=", stats.converged,
        ", initial_l2=", stats.initial_l2,
        ", initial_linf=", stats.initial_linf,
        ", final_l2=", stats.final_l2,
        ", final_linf=", stats.final_linf,
        ", target_linf=", stats.target_linf,
        ", residual_scale=", stats.residual_scale,
        ", removed_rhs_mean=", stats.removed_rhs_mean,
        ", method=:", stats.method,
        ", preconditioner=:", stats.preconditioner,
        ")",
    )
end

function _dimensionless_tolerance(::Type{T}, tolerance) where {T}
    value = T(tolerance)
    isfinite(value) && zero(T) < value < one(T) || throw(
        ArgumentError("tolerance must be finite and lie in (0, 1)"),
    )
    return value
end

function _strong_maximum(values, weights)
    T = promote_type(eltype(values), eltype(weights))
    result = zero(T)
    for index in eachindex(values, weights)
        weight = @inbounds weights[index]
        weight > zero(weight) || continue
        result = max(result, abs(@inbounds(values[index])) / weight)
    end
    return result
end

function _dimensionless_residual(stats::LinearSolverStats{T}) where {T}
    scale = stats.residual_scale
    scale > zero(T) && return stats.final_linf / scale
    return iszero(stats.final_linf) ? zero(T) : T(Inf)
end

# Composite linear solvers expose component statistics through `x` and `y`.
# Model-specific projection statistics extend this function in their own domain.
_dimensionless_residual(stats) =
    map(_dimensionless_residual, (stats.x, stats.y))

function _solver_failure_message(
        problem, stats, tolerance; context = nothing, details = nothing
    )
    prefix = isnothing(context) ? string(problem) :
        "$(context) $(problem)"
    achieved = _dimensionless_residual(stats)
    detail_text = isnothing(details) ? "" : ", $(details)"
    return "$(prefix) failed to converge " *
        "(tolerance=$(tolerance), " *
        "dimensionless_residual=$(achieved)$(detail_text), " *
        "stats=$(repr(stats)))"
end

abstract type AbstractGeometricMultigridPreconditioner{T} end

"""
    TreeGeometricMultigridPreconditioner(operator; kwargs...)

Rediscretized geometric multigrid on successively capped block trees.

By default, each coarse mesh simultaneously replaces every complete terminal
sibling family by its parent; `pruning=:deepest` instead caps only the current
deepest generation. The tree constructor restores 2:1 grading. The same
compatible elliptic discretization is constructed on every composite mesh,
so no Galerkin `P' * A * P` product is used. A coefficient-dependent smoothed
cell transfer preserves the active-domain pressure null vector, and its
adjoint restricts integrated residuals. Once the root block is reached, its
sparse operator is factorized directly even when it contains more than
`coarse_size` cells.
"""
mutable struct TreeGeometricMultigridPreconditioner{
        T, O <: EllipticOperator{T}, F, L, R,
    } <: AbstractGeometricMultigridPreconditioner{T}
    operator::O
    operators::Vector{EllipticOperator{T}}
    geometry_levels::Vector{Any}
    levels::Vector{L}
    transfer_refreshes::Vector{Any}
    coefficient_restrictions::Vector{R}
    coarse_factorization::F
    coarse_augmented::Bool
    coarse_rhs::Vector{T}
    coarse_solution::Vector{T}
    pre_sweeps::Int
    post_sweeps::Int
    pruning::Symbol
    reused_levels::Int
end

_preconditioner_name(::TreeGeometricMultigridPreconditioner) =
    :tree_geometric_multigrid

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

function _same_tree_topology(first::BlockTreeMesh, second::BlockTreeMesh)
    first.base_mesh == second.base_mesh || return false
    first.block_size == second.block_size || return false
    first.halo == second.halo || return false
    first.origin == second.origin || return false
    first.extent == second.extent || return false
    first.periodic == second.periodic || return false
    length(first.blocks) == length(second.blocks) || return false
    return all(
        first.blocks[index].key == second.blocks[index].key
            for index in eachindex(first.blocks)
    )
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

@inline function _face_lattice_geometry_at_level(
        mesh::BlockTreeMesh, face, level::Int,
    )
    axis = face.axis
    tangent_axis = 3 - axis
    lattice_size = _cells_at_level(mesh, level)
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

struct FaceCoefficientRestriction{T}
    row_pointers::Vector{Int}
    fine_indices::Vector{Int}
    weights::Vector{T}
end

function _face_coefficient_restriction(
        coarse::BlockTreeMesh, fine::BlockTreeMesh, ::Type{T},
    ) where {T}
    reference_level = max(fine.max_level, coarse.max_level)
    lines = Dict{Tuple{Int, Int}, Vector{NTuple{3, Int}}}()
    for (index, face) in pairs(fine.faces)
        normal, lo, hi = _face_lattice_geometry_at_level(
            fine, face, reference_level,
        )
        push!(
            get!(lines, (face.axis, normal), NTuple{3, Int}[]),
            (lo, hi, index),
        )
    end
    for segments in values(lines)
        sort!(segments; by = first)
    end
    row_pointers = Vector{Int}(undef, length(coarse.faces) + 1)
    fine_indices = Int[]
    weights = T[]
    row_pointers[1] = 1
    for (index, face) in pairs(coarse.faces)
        normal, lo, hi = _face_lattice_geometry_at_level(
            coarse, face, reference_level,
        )
        covered = 0
        line = get(lines, (face.axis, normal), NTuple{3, Int}[])
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

function _tree_face_lines(mesh::BlockTreeMesh, reference_level::Int)
    lines = ntuple(
        axis -> Dict{Int, Vector{TreeFaceSegment}}(), 2,
    )
    for (index, face) in pairs(mesh.faces)
        normal, lo, hi = _face_lattice_geometry_at_level(
            mesh, face, reference_level,
        )
        push!(
            get!(lines[face.axis], normal, TreeFaceSegment[]),
            TreeFaceSegment(lo, hi, index),
        )
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
    reference_level = max(fine.max_level, coarse.max_level)
    lines, coordinates = _tree_face_lines(coarse, reference_level)
    rows = Int[]
    columns = Int[]
    values = T[]
    sizehint!(rows, 2length(fine.faces))
    sizehint!(columns, 2length(fine.faces))
    sizehint!(values, 2length(fine.faces))
    lattice_size = _cells_at_level(coarse, reference_level)

    for (row, face) in pairs(fine.faces)
        axis = face.axis
        normal, lo, hi = _face_lattice_geometry_at_level(
            fine, face, reference_level,
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

# A cut-cell fine operator retains covered background cells as constrained
# algebraic unknowns.  A rediscretized coarse approximation has only the
# restricted aperture coefficient, so recover the same harmless extension:
# cells with no positive connection are assigned zero volume and a unit
# constraint.  This prevents additional null modes when a solid spans several
# cells on an intermediate tree level.
function _coarse_cut_cell_activity(mesh, coefficients, ::Type{T}) where {T}
    active = falses(ncells(mesh))
    nx, ny = mesh.block_size
    for (index, face) in pairs(mesh.faces)
        @inbounds coefficients[index] > zero(T) || continue
        active[_elliptic_cell_index(face.minus, nx, ny)] = true
        active[_elliptic_cell_index(face.plus, nx, ny)] = true
    end
    volumes = _cell_volumes(mesh, T)
    for index in eachindex(volumes)
        @inbounds active[index] || (volumes[index] = zero(T))
    end
    return active, volumes
end

function TreeGeometricMultigridPreconditioner(
        operator::O; coarse_size = 64,
        pre_sweeps = 2, post_sweeps = 2,
        max_levels = 32, pruning = :all, reuse = nothing,
        geometry = nothing,
    ) where {T, O <: EllipticOperator{T}}
    coarse_size >= 1 || throw(ArgumentError("coarse_size must be positive"))
    pre_sweeps >= 0 || throw(ArgumentError("pre_sweeps must be nonnegative"))
    post_sweeps >= 0 || throw(ArgumentError("post_sweeps must be nonnegative"))
    max_levels >= 1 || throw(ArgumentError("max_levels must be positive"))
    pruning in (:deepest, :all) || throw(
        ArgumentError("tree pruning must be :deepest or :all"),
    )
    isnothing(reuse) ||
        reuse isa TreeGeometricMultigridPreconditioner{T} || throw(
        ArgumentError("reused pressure hierarchy has the wrong type"),
    )
    isnothing(geometry) || geometry.mesh === operator.mesh || throw(
        ArgumentError("multigrid geometry must share the operator mesh"),
    )
    isnothing(geometry) || iszero(operator.reaction) || throw(
        ArgumentError(
            "geometry-aware tree multigrid currently supports pressure operators",
        ),
    )
    can_reuse = isnothing(geometry) && !isnothing(reuse) &&
        reuse.pruning === pruning &&
        iszero(reuse.operator.reaction - operator.reaction) &&
        iszero(reuse.operator.diffusion - operator.diffusion)

    operators = EllipticOperator{T}[operator]
    geometry_levels = Any[]
    isnothing(geometry) || push!(geometry_levels, geometry)
    prolongations = SparseMatrixCSC{T, Int}[]
    transfer_refreshes = Any[]
    coefficient_restrictions = FaceCoefficientRestriction{T}[]
    current_mesh = operator.mesh
    active_level = maximum(block.key.level for block in current_mesh.blocks)
    reused_new_start = 0
    reused_old_start = 0
    matrix_free_fine = size(operator.matrix) !=
        (length(operator.volumes), length(operator.volumes))
    initial_matching_level = can_reuse && !matrix_free_fine ?
        _matching_tree_level(reuse.operators, current_mesh, 1) : 0
    initial_suffix_length = iszero(initial_matching_level) ? 0 :
        length(reuse.operators) - initial_matching_level + 1
    if !iszero(initial_matching_level) &&
            initial_suffix_length <= max_levels
        reused_new_start = 1
        reused_old_start = initial_matching_level
        for old_index in (initial_matching_level + 1):length(reuse.operators)
            push!(operators, reuse.operators[old_index])
        end
        for old_index in initial_matching_level:length(
                reuse.transfer_refreshes
            )
            push!(transfer_refreshes, reuse.transfer_refreshes[old_index])
            push!(
                coefficient_restrictions,
                reuse.coefficient_restrictions[old_index],
            )
            push!(
                prolongations,
                reuse.transfer_refreshes[old_index].prolongation,
            )
        end
    end
    while iszero(reused_new_start) &&
            ncells(current_mesh) > coarse_size && active_level > 0 &&
            length(operators) < max_levels
        coarse_mesh = if pruning === :deepest
            active_level -= 1
            _capped_tree_mesh(operator.mesh, active_level)
        else
            _all_leaf_pruned_mesh(current_mesh)
        end
        if ncells(coarse_mesh) >= ncells(current_mesh)
            pruning === :all && break
            continue
        end
        tentative = _tree_cell_prolongation(
            current_mesh, coarse_mesh, T,
        )
        transfer_refresh = if matrix_free_fine && length(operators) == 1
            stencil = _factorized_elliptic_stencil(last(operators))
            _factorized_smoothed_transfer_refresh(
                last(operators), tentative, stencil,
            )
        else
            _smoothed_transfer_refresh(last(operators).stiffness, tentative)
        end
        prolongation = transfer_refresh.prolongation
        coefficient_restriction = _face_coefficient_restriction(
            coarse_mesh, current_mesh, T,
        )
        coarse_geometry = if isnothing(geometry)
            nothing
        else
            cell_restriction = _cell_coefficient_restriction(
                coarse_mesh, current_mesh, T,
            )
            _restrict_geometry_to_mesh(
                last(geometry_levels), coarse_mesh, cell_restriction,
            )
        end
        isnothing(coarse_geometry) ||
            _restrict_transfer_to_active_domain!(
            prolongation, last(geometry_levels).active_cells,
            coarse_geometry.active_cells,
            transfer_refresh isa FactorizedSmoothedTransferRefresh ?
                transfer_refresh.stencil.row_sums :
                transfer_refresh.row_sums,
        )
        coarse_coefficient = zeros(T, length(coarse_mesh.faces))
        if isnothing(coarse_geometry)
            _restrict_face_coefficients!(
                coarse_coefficient, last(operators).face_coefficient,
                coefficient_restriction,
            )
        else
            _restrict_geometry_material!(
                coarse_coefficient, last(operators),
                last(geometry_levels), coarse_geometry,
                coefficient_restriction,
            )
        end
        matching_level = can_reuse ?
            _matching_tree_level(reuse.operators, coarse_mesh, 1) : 0
        suffix_length = iszero(matching_level) ? 0 :
            length(reuse.operators) - matching_level + 1
        can_append_suffix = !iszero(matching_level) &&
            length(operators) + suffix_length <= max_levels
        coarse_operator = if !isnothing(coarse_geometry)
            PoissonOperator(
                coarse_geometry; face_coefficient = coarse_coefficient,
                construction_cache =
                    operator.gradient_operator.construction_cache,
            )
        elseif can_append_suffix
            candidate = reuse.operators[matching_level]
            copyto!(candidate.face_coefficient, coarse_coefficient)
            _update_elliptic_face_coefficient!(
                candidate, candidate.face_coefficient,
            )
            candidate
        else
            cut_extension = any(!iszero, operator.constraints)
            active, volumes = cut_extension ?
                _coarse_cut_cell_activity(
                    coarse_mesh, coarse_coefficient, T,
                ) : (nothing, nothing)
            EllipticOperator(
                coarse_mesh;
                reaction = operator.reaction,
                diffusion = operator.diffusion,
                face_coefficient = coarse_coefficient,
                construction_cache =
                    operator.gradient_operator.construction_cache,
                active_cells = active, volumes, T,
            )
        end
        push!(prolongations, prolongation)
        push!(transfer_refreshes, transfer_refresh)
        push!(coefficient_restrictions, coefficient_restriction)
        push!(operators, coarse_operator)
        isnothing(coarse_geometry) ||
            push!(geometry_levels, coarse_geometry)
        current_mesh = coarse_mesh
        if can_append_suffix
            reused_new_start = length(operators)
            reused_old_start = matching_level
            for old_index in (matching_level + 1):length(reuse.operators)
                push!(operators, reuse.operators[old_index])
            end
            for old_index in matching_level:length(reuse.transfer_refreshes)
                push!(transfer_refreshes, reuse.transfer_refreshes[old_index])
                push!(
                    coefficient_restrictions,
                    reuse.coefficient_restrictions[old_index],
                )
                push!(
                    prolongations,
                    reuse.transfer_refreshes[old_index].prolongation,
                )
            end
            break
        end
    end

    levels = _multigrid_levels(T, matrix_free_fine)
    for index in eachindex(operators)
        prolongation = index < length(operators) ?
            prolongations[index] :
            spzeros(T, length(operators[index].volumes), 0)
        if matrix_free_fine && index == 1
            isempty(transfer_refreshes) && throw(
                ArgumentError(
                    "a matrix-free finest level requires a coarser tree level",
                ),
            )
            push!(
                levels,
                _factorized_elliptic_level(
                    operators[index], prolongation,
                    transfer_refreshes[1].stencil,
                ),
            )
        elseif !iszero(reused_new_start) && index >= reused_new_start
            old_index = reused_old_start + index - reused_new_start
            level = reuse.levels[old_index]
            level.matrix = operators[index].matrix
            level.volumes = operators[index].volumes
            level.prolongation = prolongation
            push!(levels, level)
        else
            push!(
                levels,
                _multigrid_level(
                    operators[index].matrix,
                    operators[index].volumes,
                    prolongation,
                ),
            )
        end
    end
    coarse = last(levels)
    singular = operator.singular
    if !iszero(reused_new_start)
        factorization = reuse.coarse_factorization
        coarse_rhs = reuse.coarse_rhs
        coarse_solution = reuse.coarse_solution
    elseif singular
        number_of_cells = length(coarse.volumes)
        augmented = sparse(
            [
                coarse.matrix sparse(coarse.volumes);
                sparse(transpose(coarse.volumes)) spzeros(T, 1, 1)
            ],
        )
        factorization = lu(augmented)
        coarse_rhs = zeros(T, number_of_cells + 1)
        coarse_solution = similar(coarse_rhs)
    else
        factorization = lu(coarse.matrix)
        coarse_rhs = zeros(T, length(coarse.volumes))
        coarse_solution = similar(coarse_rhs)
    end
    preconditioner = TreeGeometricMultigridPreconditioner(
        operator, operators, geometry_levels, levels, transfer_refreshes,
        coefficient_restrictions,
        factorization, singular,
        coarse_rhs, coarse_solution, pre_sweeps, post_sweeps, pruning,
        iszero(reused_new_start) ? 0 :
            length(operators) - reused_new_start + 1,
    )
    !iszero(reused_new_start) &&
        _refresh_geometric_multigrid!(preconditioner, operator)
    return preconditioner
end

function _refresh_multigrid_level!(
        level::MultigridLevel{T}, matrix, volumes, prolongation
    ) where {T}
    level.matrix = matrix
    matrix_transpose = level.matrix_transpose
    matrix_transpose isa SparseMatrixCSC || throw(
        ArgumentError("assembled multigrid level has no sparse row form"),
    )
    size(matrix_transpose) == reverse(size(matrix)) || throw(
        ArgumentError("multigrid row-form size changed on fixed topology"),
    )
    transpose!(matrix_transpose, matrix)
    level.volumes = volumes
    level.prolongation = prolongation
    length(level.inverse_diagonal) == size(matrix, 1) ||
        throw(ArgumentError("multigrid level size changed on fixed topology"))
    column_pointers = matrix_transpose.colptr
    rows = rowvals(matrix_transpose)
    matrix_values = nonzeros(matrix_transpose)
    row_sums = level.temporary
    inverse_diagonal = level.inverse_diagonal
    @_parallel for column in 1:size(matrix_transpose, 2)
        row_sum = zero(eltype(row_sums))
        diagonal = zero(eltype(inverse_diagonal))
        for pointer in column_pointers[column]:(column_pointers[column + 1] - 1)
            value = @inbounds matrix_values[pointer]
            row_sum += abs(value)
            @inbounds rows[pointer] == column && (diagonal = value)
        end
        @inbounds row_sums[column] = row_sum
        @inbounds inverse_diagonal[column] = diagonal
    end
    spectral_bound = zero(T)
    for index in eachindex(level.inverse_diagonal)
        diagonal = @inbounds level.inverse_diagonal[index]
        diagonal > zero(T) || throw(
            ArgumentError("elliptic matrix has a non-positive diagonal"),
        )
        @inbounds spectral_bound = max(
            spectral_bound, level.temporary[index] / diagonal,
        )
        @inbounds level.inverse_diagonal[index] = inv(diagonal)
    end
    level.relaxation_weight = iszero(spectral_bound) ? one(T) :
        min(one(T), T(4) / (T(3) * spectral_bound))
    return level
end

function _refresh_geometric_multigrid!(
        preconditioner::TreeGeometricMultigridPreconditioner{T},
        operator::EllipticOperator{T},
    ) where {T}
    preconditioner.operator = operator
    preconditioner.operators[1] = operator
    for index in 2:length(preconditioner.operators)
        fine = preconditioner.operators[index - 1]
        coarse = preconditioner.operators[index]
        restriction = preconditioner.coefficient_restrictions[index - 1]
        if isempty(preconditioner.geometry_levels)
            _restrict_face_coefficients!(
                coarse.face_coefficient, fine.face_coefficient,
                restriction,
            )
        else
            fine_geometry = preconditioner.geometry_levels[index - 1]
            coarse_geometry = preconditioner.geometry_levels[index]
            material = coarse.square_root_face_coefficient
            _restrict_geometry_material!(
                material, fine, fine_geometry, coarse_geometry,
                restriction,
            )
            @. coarse.face_coefficient =
                material * coarse_geometry.aperture^2
        end
        _update_elliptic_face_coefficient!(
            coarse, coarse.face_coefficient,
        )
    end
    for index in eachindex(preconditioner.transfer_refreshes)
        refresh = preconditioner.transfer_refreshes[index]
        prolongation = refresh isa FactorizedSmoothedTransferRefresh ?
            _refresh_smoothed_transfer!(
                refresh, preconditioner.operators[index],
            ) : _refresh_smoothed_transfer!(
                refresh, preconditioner.operators[index].stiffness,
            )
        isempty(preconditioner.geometry_levels) ||
            _restrict_transfer_to_active_domain!(
            prolongation,
            preconditioner.geometry_levels[index].active_cells,
            preconditioner.geometry_levels[index + 1].active_cells,
            refresh isa FactorizedSmoothedTransferRefresh ?
                refresh.stencil.row_sums : refresh.row_sums,
        )
        preconditioner.levels[index].prolongation = prolongation
    end
    for index in eachindex(preconditioner.levels)
        level = preconditioner.levels[index]
        level_operator = preconditioner.operators[index]
        if level.matrix isa FactorizedEllipticLevelAction
            stencil = _refresh_factorized_elliptic_stencil!(
                level.matrix.stencil, level_operator,
            )
            level.matrix.action.operator = level_operator
            level.volumes = level_operator.volumes
            copyto!(level.inverse_diagonal, stencil.inverse_diagonal)
            level.relaxation_weight = stencil.relaxation_weight
        else
            _refresh_multigrid_level!(
                level, level_operator.matrix, level_operator.volumes,
                level.prolongation,
            )
        end
    end
    coarse = last(preconditioner.levels)
    if preconditioner.coarse_augmented
        augmented = sparse(
            [
                coarse.matrix sparse(coarse.volumes);
                sparse(transpose(coarse.volumes)) spzeros(T, 1, 1)
            ],
        )
        preconditioner.coarse_factorization = lu(augmented)
    else
        preconditioner.coarse_factorization = lu(coarse.matrix)
    end
    return preconditioner
end
