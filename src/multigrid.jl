function _jacobi_spectral_bound(matrix::SparseMatrixCSC{T}) where {T}
    diagonal = collect(diag(matrix))
    row_sums = zeros(T, size(matrix, 1))
    rows = rowvals(matrix)
    values = nonzeros(matrix)
    for column in axes(matrix, 2), pointer in nzrange(matrix, column)
        @inbounds row_sums[rows[pointer]] += abs(values[pointer])
    end
    bound = zero(T)
    for index in eachindex(diagonal)
        @inbounds diagonal[index] > zero(T) || continue
        @inbounds bound = max(bound, row_sums[index] / diagonal[index])
    end
    return bound
end

function _prolongate!(destination, prolongation::SparseMatrixCSC, source)
    mul!(destination, prolongation, source)
    return destination
end

function _restrict!(destination, prolongation::SparseMatrixCSC, source)
    mul!(destination, transpose(prolongation), source)
    return destination
end

mutable struct MultigridLevel{T, P}
    matrix::SparseMatrixCSC{T, Int}
    volumes::Vector{T}
    prolongation::P
    inverse_diagonal::Vector{T}
    relaxation_weight::T
    solution::Vector{T}
    right_hand_side::Vector{T}
    residual::Vector{T}
    temporary::Vector{T}
end

"""Fixed sparsity for one Jacobi-smoothed geometric transfer.

`stiffness_to_transfer[p]` identifies the interpolation entry receiving the
contribution of stiffness entry `p`.  Only the values are refreshed when the
two-phase coefficient changes; all graph searches belong to topology setup.
"""
struct SmoothedTransferRefresh{T}
    prolongation::SparseMatrixCSC{T, Int}
    tentative_pointers::Vector{Int}
    stiffness_to_transfer::Vector{Int}
    inverse_diagonal::Vector{T}
    row_sums::Vector{T}
end

function _sparse_pointer_lookup(matrix::SparseMatrixCSC)
    lookup = Dict{Tuple{Int, Int}, Int}()
    rows = rowvals(matrix)
    sizehint!(lookup, nnz(matrix))
    for column in axes(matrix, 2), pointer in nzrange(matrix, column)
        lookup[(rows[pointer], column)] = pointer
    end
    return lookup
end

function _smoothed_transfer_refresh(
        stiffness::SparseMatrixCSC{T, Int},
        tentative::SparseMatrixCSC{T, Int},
    ) where {T}
    number_of_fine, number_of_coarse = size(tentative)
    owners = zeros(Int, number_of_fine)
    tentative_rows = rowvals(tentative)
    for column in axes(tentative, 2), pointer in nzrange(tentative, column)
        row = tentative_rows[pointer]
        iszero(owners[row]) || throw(
            ArgumentError("tentative pressure transfer must have one owner"),
        )
        owners[row] = column
    end
    all(!iszero, owners) || throw(
        ArgumentError("tentative pressure transfer leaves an unknown unowned"),
    )

    rows = Int[]
    columns = Int[]
    sizehint!(rows, nnz(stiffness) + number_of_fine)
    sizehint!(columns, nnz(stiffness) + number_of_fine)
    for row in 1:number_of_fine
        push!(rows, row)
        push!(columns, owners[row])
    end
    stiffness_rows = rowvals(stiffness)
    for column in axes(stiffness, 2), pointer in nzrange(stiffness, column)
        push!(rows, stiffness_rows[pointer])
        push!(columns, owners[column])
    end
    prolongation = sparse(
        rows, columns, ones(T, length(rows)),
        number_of_fine, number_of_coarse,
    )
    fill!(nonzeros(prolongation), zero(T))
    lookup = _sparse_pointer_lookup(prolongation)
    tentative_pointers = [
        lookup[(row, owners[row])] for row in 1:number_of_fine
    ]
    stiffness_to_transfer = Vector{Int}(undef, nnz(stiffness))
    for column in axes(stiffness, 2), pointer in nzrange(stiffness, column)
        @inbounds stiffness_to_transfer[pointer] =
            lookup[(stiffness_rows[pointer], owners[column])]
    end
    refresh = SmoothedTransferRefresh(
        prolongation, tentative_pointers, stiffness_to_transfer,
        zeros(T, number_of_fine), zeros(T, number_of_fine),
    )
    _refresh_smoothed_transfer!(refresh, stiffness)
    return refresh
end

function _refresh_smoothed_transfer!(
        refresh::SmoothedTransferRefresh{T},
        stiffness::SparseMatrixCSC{T, Int},
    ) where {T}
    length(refresh.stiffness_to_transfer) == nnz(stiffness) || throw(
        ArgumentError("pressure stiffness sparsity changed on fixed topology"),
    )
    inverse_diagonal = refresh.inverse_diagonal
    row_sums = refresh.row_sums
    fill!(inverse_diagonal, zero(T))
    fill!(row_sums, zero(T))
    rows = rowvals(stiffness)
    values = nonzeros(stiffness)
    for column in axes(stiffness, 2), pointer in nzrange(stiffness, column)
        row = rows[pointer]
        value = values[pointer]
        @inbounds row_sums[row] += abs(value)
        row == column && (@inbounds inverse_diagonal[row] = value)
    end
    spectral_bound = zero(T)
    for index in eachindex(inverse_diagonal)
        diagonal = @inbounds inverse_diagonal[index]
        diagonal > zero(T) || continue
        @inbounds spectral_bound = max(
            spectral_bound, row_sums[index] / diagonal,
        )
        @inbounds inverse_diagonal[index] = inv(diagonal)
    end
    weight = iszero(spectral_bound) ? zero(T) :
        T(4) / (T(3) * spectral_bound)

    prolongation_values = nonzeros(refresh.prolongation)
    fill!(prolongation_values, zero(T))
    for pointer in refresh.tentative_pointers
        @inbounds prolongation_values[pointer] += one(T)
    end
    for stiffness_pointer in eachindex(values)
        transfer_pointer =
            @inbounds refresh.stiffness_to_transfer[stiffness_pointer]
        row = @inbounds rows[stiffness_pointer]
        @inbounds prolongation_values[transfer_pointer] -=
            weight * inverse_diagonal[row] * values[stiffness_pointer]
    end
    return refresh.prolongation
end

function _multigrid_level(
        matrix::SparseMatrixCSC{T, Int},
        volumes::Vector{T},
        prolongation
    ) where {T}
    diagonal = collect(diag(matrix))
    all(>(zero(T)), diagonal) ||
        throw(ArgumentError("elliptic matrix has a non-positive diagonal"))
    inverse_diagonal = inv.(diagonal)
    spectral_bound = _jacobi_spectral_bound(matrix)
    relaxation_weight = iszero(spectral_bound) ? one(T) :
        min(one(T), T(4) / (T(3) * spectral_bound))
    number_of_cells = size(matrix, 1)
    return MultigridLevel(
        matrix, volumes, prolongation,
        inverse_diagonal, relaxation_weight,
        zeros(T, number_of_cells), zeros(T, number_of_cells),
        zeros(T, number_of_cells), zeros(T, number_of_cells),
    )
end

"""Convergence statistics from the latest external Krylov solve."""
mutable struct LinearSolverStats{T}
    cycles::Int
    converged::Bool
    initial_l2::T
    final_l2::T
    final_linf::T
    removed_rhs_mean::T
    method::Symbol
    preconditioner::Symbol
end

# Preserve the public name used by earlier releases.
"""Compatibility alias for [`LinearSolverStats`](@ref)."""
const MultigridStats = LinearSolverStats

LinearSolverStats(::Type{T}; method = :none, preconditioner = :none) where {T} =
    LinearSolverStats(
    0, false, T(Inf), T(Inf), T(Inf), zero(T),
    method, preconditioner
)

abstract type AbstractGeometricMultigridPreconditioner{T} end

"""
    TreeGeometricMultigridPreconditioner(operator; kwargs...)

Rediscretized geometric multigrid on successively capped block trees.

By default, each coarse mesh simultaneously replaces every complete terminal
sibling family by its parent; `pruning=:deepest` instead caps only the current
deepest generation. The tree constructor restores 2:1 grading. The same
compatible elliptic discretization is constructed on every composite mesh,
so no Galerkin `P' * A * P` product is used. A coefficient-dependent smoothed
cell transfer preserves constants, and its adjoint restricts integrated
residuals. Once the root block is reached, its sparse operator is factorized
directly even when it contains more than `coarse_size` cells.
"""
mutable struct TreeGeometricMultigridPreconditioner{
        T, O <: EllipticOperator{T}, F, L, R,
    } <: AbstractGeometricMultigridPreconditioner{T}
    operator::O
    operators::Vector{EllipticOperator{T}}
    levels::Vector{L}
    transfer_refreshes::Vector{SmoothedTransferRefresh{T}}
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
    retain_internal(mesh.root)
    maximum_active_level = isempty(internal) ? 0 :
        maximum(key.level for key in internal) + 1
    return BlockTreeMesh(
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
        node = RegionTrees.findleaf(coarse.root, point)
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
    positions = copy(@view(row_pointers[1:end - 1]))
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
    for coarse_index in eachindex(coarse_coefficient)
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
    lattice_size = mesh.block_size .* (1 << level)
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
        for (fine_lo, fine_hi, fine_index) in
                get(lines, (face.axis, normal), NTuple{3, Int}[])
            overlap = min(hi, fine_hi) - max(lo, fine_lo)
            overlap > 0 || continue
            push!(fine_indices, fine_index)
            push!(weights, T(overlap) / T(hi - lo))
            covered += overlap
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

function _tree_line_average_weights(line, lo, hi, ::Type{T}) where {T}
    indices = Int[]
    weights = T[]
    covered = 0
    for segment in line
        overlap = min(hi, segment.hi) - max(lo, segment.lo)
        overlap > 0 || continue
        covered += overlap
        push!(indices, segment.index)
        push!(weights, T(overlap) / T(hi - lo))
    end
    return covered == hi - lo ? (indices, weights) : nothing
end

function _tree_line_point_index(line, tangent)
    for segment in line
        segment.lo <= tangent <= segment.hi && return segment.index
    end
    return 0
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
    lattice_size = coarse.block_size .* (1 << reference_level)

    for (row, face) in pairs(fine.faces)
        axis = face.axis
        normal, lo, hi = _face_lattice_geometry_at_level(
            fine, face, reference_level,
        )
        exact = get(lines[axis], normal, nothing)
        if !isnothing(exact)
            average = _tree_line_average_weights(exact, lo, hi, T)
            if !isnothing(average)
                indices, weights = average
                for index in eachindex(indices)
                    push!(rows, row)
                    push!(columns, indices[index])
                    push!(values, weights[index])
                end
                continue
            end
        end

        tangent = T(lo + hi) / 2
        normal_period = lattice_size[axis]
        lower_distance = typemin(Int)
        upper_distance = typemax(Int)
        lower_index = 0
        upper_index = 0
        for coordinate in coordinates[axis]
            delta = coordinate - normal
            if coarse.periodic[axis]
                delta -= round(Int, delta / normal_period) * normal_period
            end
            iszero(delta) && continue
            index = _tree_line_point_index(lines[axis][coordinate], tangent)
            iszero(index) && continue
            if delta < 0 && delta > lower_distance
                lower_distance = delta
                lower_index = index
            elseif delta > 0 && delta < upper_distance
                upper_distance = delta
                upper_index = index
            end
        end

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
    for coarse_index in eachindex(coarse_coefficient)
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

function TreeGeometricMultigridPreconditioner(
        operator::O; coarse_size = 64,
        pre_sweeps = 2, post_sweeps = 2,
        max_levels = 32, pruning = :all, reuse = nothing,
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
    can_reuse = !isnothing(reuse) && reuse.pruning === pruning &&
        iszero(reuse.operator.reaction - operator.reaction) &&
        iszero(reuse.operator.diffusion - operator.diffusion)

    operators = EllipticOperator{T}[operator]
    prolongations = SparseMatrixCSC{T, Int}[]
    transfer_refreshes = SmoothedTransferRefresh{T}[]
    coefficient_restrictions = FaceCoefficientRestriction{T}[]
    current_mesh = operator.mesh
    active_level = maximum(block.key.level for block in current_mesh.blocks)
    reused_new_start = 0
    reused_old_start = 0
    initial_matching_level = can_reuse ?
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
                reuse.transfer_refreshes)
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
        transfer_refresh = _smoothed_transfer_refresh(
            last(operators).stiffness, tentative,
        )
        prolongation = transfer_refresh.prolongation
        coarse_coefficient = zeros(T, length(coarse_mesh.faces))
        coefficient_restriction = _face_coefficient_restriction(
            coarse_mesh, current_mesh, T,
        )
        _restrict_face_coefficients!(
            coarse_coefficient, last(operators).face_coefficient,
            coefficient_restriction,
        )
        matching_level = can_reuse ?
            _matching_tree_level(reuse.operators, coarse_mesh, 1) : 0
        suffix_length = iszero(matching_level) ? 0 :
            length(reuse.operators) - matching_level + 1
        can_append_suffix = !iszero(matching_level) &&
            length(operators) + suffix_length <= max_levels
        coarse_operator = if can_append_suffix
            candidate = reuse.operators[matching_level]
            copyto!(candidate.face_coefficient, coarse_coefficient)
            _update_elliptic_face_coefficient!(
                candidate, candidate.face_coefficient,
            )
            candidate
        else
            EllipticOperator(
                coarse_mesh;
                reaction = operator.reaction,
                diffusion = operator.diffusion,
                face_coefficient = coarse_coefficient,
                T,
            )
        end
        push!(prolongations, prolongation)
        push!(transfer_refreshes, transfer_refresh)
        push!(coefficient_restrictions, coefficient_restriction)
        push!(operators, coarse_operator)
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

    levels = MultigridLevel{T, SparseMatrixCSC{T, Int}}[]
    for index in eachindex(operators)
        prolongation = index < length(operators) ?
            prolongations[index] :
            spzeros(T, length(operators[index].volumes), 0)
        if !iszero(reused_new_start) && index >= reused_new_start
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
    singular = iszero(operator.reaction)
    if !iszero(reused_new_start)
        factorization = reuse.coarse_factorization
        coarse_rhs = reuse.coarse_rhs
        coarse_solution = reuse.coarse_solution
    elseif singular
        number_of_cells = length(coarse.volumes)
        augmented = sparse(
            [coarse.matrix sparse(coarse.volumes);
             sparse(transpose(coarse.volumes)) spzeros(T, 1, 1)],
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
        operator, operators, levels, transfer_refreshes,
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
    level.volumes = volumes
    level.prolongation = prolongation
    length(level.inverse_diagonal) == size(matrix, 1) ||
        throw(ArgumentError("multigrid level size changed on fixed topology"))
    fill!(level.inverse_diagonal, zero(T))
    fill!(level.temporary, zero(T))
    rows = rowvals(matrix)
    values = nonzeros(matrix)
    for column in axes(matrix, 2), pointer in nzrange(matrix, column)
        row = rows[pointer]
        value = values[pointer]
        @inbounds level.temporary[row] += abs(value)
        row == column && (@inbounds level.inverse_diagonal[row] = value)
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
        _restrict_face_coefficients!(
            coarse.face_coefficient, fine.face_coefficient,
            preconditioner.coefficient_restrictions[index - 1],
        )
        _update_elliptic_face_coefficient!(
            coarse, coarse.face_coefficient,
        )
    end
    for index in eachindex(preconditioner.transfer_refreshes)
        prolongation = _refresh_smoothed_transfer!(
            preconditioner.transfer_refreshes[index],
            preconditioner.operators[index].stiffness,
        )
        preconditioner.levels[index].prolongation = prolongation
    end
    for index in eachindex(preconditioner.levels)
        level = preconditioner.levels[index]
        level_operator = preconditioner.operators[index]
        _refresh_multigrid_level!(
            level, level_operator.matrix, level_operator.volumes,
            level.prolongation,
        )
    end
    coarse = last(preconditioner.levels)
    if preconditioner.coarse_augmented
        augmented = sparse(
            [coarse.matrix sparse(coarse.volumes);
             sparse(transpose(coarse.volumes)) spzeros(T, 1, 1)],
        )
        preconditioner.coarse_factorization = lu(augmented)
    else
        preconditioner.coarse_factorization = lu(coarse.matrix)
    end
    return preconditioner
end

# Compatibility names now resolve to the tree-rediscretized implementation.
const GeometricMultigridPreconditioner = TreeGeometricMultigridPreconditioner
const GeometricMultigridSolver = TreeGeometricMultigridPreconditioner

function _remove_volume_mean!(values, volumes)
    mean_value = dot(values, volumes) / sum(volumes)
    for index in eachindex(values)
        @inbounds values[index] -= mean_value
    end
    return mean_value
end

function _project_integrated_rhs!(right_hand_side, volumes)
    mean_value = sum(right_hand_side) / sum(volumes)
    for index in eachindex(right_hand_side)
        @inbounds right_hand_side[index] -= mean_value * volumes[index]
    end
    return mean_value
end

function _smooth!(level::MultigridLevel, sweeps, singular)
    for _ in 1:sweeps
        mul!(level.temporary, level.matrix, level.solution)
        for index in eachindex(level.solution)
            @inbounds level.solution[index] += level.relaxation_weight *
                level.inverse_diagonal[index] *
                (level.right_hand_side[index] - level.temporary[index])
        end
        singular && _remove_volume_mean!(level.solution, level.volumes)
    end
    return level.solution
end

function _coarse_solve!(solver::AbstractGeometricMultigridPreconditioner)
    coarse = last(solver.levels)
    number_of_cells = length(coarse.solution)
    if solver.coarse_augmented
        copyto!(solver.coarse_rhs, 1, coarse.right_hand_side, 1, number_of_cells)
        solver.coarse_rhs[end] = zero(eltype(solver.coarse_rhs))
        ldiv!(
            solver.coarse_solution, solver.coarse_factorization,
            solver.coarse_rhs
        )
        copyto!(coarse.solution, 1, solver.coarse_solution, 1, number_of_cells)
    else
        ldiv!(
            coarse.solution, solver.coarse_factorization,
            coarse.right_hand_side
        )
    end
    return coarse.solution
end

function _vcycle!(solver::AbstractGeometricMultigridPreconditioner, level_index::Int)
    level = solver.levels[level_index]
    if level_index == length(solver.levels)
        return _coarse_solve!(solver)
    end

    singular = solver.coarse_augmented
    _smooth!(level, solver.pre_sweeps, singular)
    mul!(level.temporary, level.matrix, level.solution)
    for index in eachindex(level.residual)
        @inbounds level.residual[index] =
            level.right_hand_side[index] - level.temporary[index]
    end

    coarse = solver.levels[level_index + 1]
    _restrict!(coarse.right_hand_side, level.prolongation, level.residual)
    singular && _project_integrated_rhs!(coarse.right_hand_side, coarse.volumes)
    fill!(coarse.solution, zero(eltype(coarse.solution)))
    _vcycle!(solver, level_index + 1)
    _prolongate!(level.temporary, level.prolongation, coarse.solution)
    for index in eachindex(level.solution)
        @inbounds level.solution[index] += level.temporary[index]
    end
    _smooth!(level, solver.post_sweeps, singular)
    return level.solution
end

function _residual_norms!(level::MultigridLevel)
    mul!(level.temporary, level.matrix, level.solution)
    squared = zero(eltype(level.solution))
    maximum_norm = zero(eltype(level.solution))
    total_volume = sum(level.volumes)
    for index in eachindex(level.residual)
        @inbounds integrated = level.right_hand_side[index] - level.temporary[index]
        @inbounds level.residual[index] = integrated
        @inbounds strong = integrated / level.volumes[index]
        @inbounds squared += level.volumes[index] * strong^2
        maximum_norm = max(maximum_norm, abs(strong))
    end
    return sqrt(squared / total_volume), maximum_norm
end

function _roundoff_residual_floor!(
        matrix::SparseMatrixCSC{T}, solution,
        right_hand_side, volumes,
        temporary
    ) where {T}
    fill!(temporary, zero(T))
    rows = rowvals(matrix)
    values = nonzeros(matrix)
    for column in axes(matrix, 2), pointer in nzrange(matrix, column)
        row = rows[pointer]
        @inbounds temporary[row] +=
            abs(values[pointer]) * abs(solution[column])
    end
    scale = one(T)
    for index in eachindex(temporary)
        @inbounds scale = max(
            scale,
            temporary[index] / volumes[index],
            abs(right_hand_side[index] / volumes[index])
        )
    end
    return T(128) * eps(T) * scale
end

_roundoff_residual_floor!(level::MultigridLevel) =
    _roundoff_residual_floor!(
    level.matrix, level.solution, level.right_hand_side,
    level.volumes, level.temporary
)

function _integrated_residual_norms(residual, volumes)
    T = eltype(residual)
    squared = zero(T)
    maximum_norm = zero(T)
    total_volume = zero(T)
    for index in eachindex(residual)
        @inbounds strong = residual[index] / volumes[index]
        @inbounds squared += volumes[index] * strong^2
        maximum_norm = max(maximum_norm, abs(strong))
        @inbounds total_volume += volumes[index]
    end
    return sqrt(squared / total_volume), maximum_norm
end

function _apply_vcycle_preconditioner!(
        destination,
        solver::AbstractGeometricMultigridPreconditioner,
        right_hand_side
    )
    fine = first(solver.levels)
    copyto!(fine.right_hand_side, right_hand_side)
    fill!(fine.solution, zero(eltype(fine.solution)))
    _vcycle!(solver, 1)
    copyto!(destination, fine.solution)
    return destination
end

function ldiv!(
        destination::AbstractVector,
        solver::AbstractGeometricMultigridPreconditioner,
        right_hand_side::AbstractVector
    )
    return _apply_vcycle_preconditioner!(
        destination, solver, right_hand_side
    )
end

Base.size(solver::AbstractGeometricMultigridPreconditioner) =
    size(first(solver.levels).matrix)
Base.eltype(::AbstractGeometricMultigridPreconditioner{T}) where {T} = T

"""A named wrapper around an external preconditioner implementing `ldiv!`."""
struct ExternalPreconditioner{T, P}
    name::Symbol
    value::P
    dimensions::Tuple{Int, Int}
end

ExternalPreconditioner(name, value, matrix) =
    ExternalPreconditioner{eltype(matrix), typeof(value)}(
    name, value, size(matrix)
)

function ldiv!(
        destination::AbstractVector,
        preconditioner::ExternalPreconditioner,
        right_hand_side::AbstractVector
    )
    return ldiv!(destination, preconditioner.value, right_hand_side)
end

Base.size(preconditioner::ExternalPreconditioner) = preconditioner.dimensions
Base.eltype(::ExternalPreconditioner{T}) where {T} = T

_preconditioner_name(::Nothing) = :none
_preconditioner_name(::AbstractGeometricMultigridPreconditioner) = :geometric_multigrid
_preconditioner_name(preconditioner::ExternalPreconditioner) = preconditioner.name
_preconditioner_name(preconditioner) = Symbol(nameof(typeof(preconditioner)))

function _optional_extension(name::Symbol, package::String)
    extension = Base.get_extension(@__MODULE__, name)
    isnothing(extension) && throw(
        ArgumentError(
            "$package is an optional dependency; run `import $package` before " *
                "constructing this preconditioner"
        )
    )
    return extension
end

"""
    algebraic_multigrid_preconditioner(matrix; symmetric=true,
                                       singular=false, kwargs...)

Construct an AlgebraicMultigrid.jl preconditioner. Import
`AlgebraicMultigrid` before calling this optional integration.
"""
function algebraic_multigrid_preconditioner(matrix; kwargs...)
    extension = _optional_extension(
        :TwoPhaseAMRAlgebraicMultigridExt, "AlgebraicMultigrid"
    )
    return ExternalPreconditioner(
        :algebraic_multigrid,
        extension.build_algebraic_multigrid_preconditioner(matrix; kwargs...),
        matrix
    )
end

"""
    incomplete_lu_preconditioner(matrix; drop_tolerance=1e-3)

Construct an IncompleteLU.jl Crout ILU preconditioner. Import `IncompleteLU`
before calling this optional integration.
"""
function incomplete_lu_preconditioner(matrix; kwargs...)
    extension = _optional_extension(:TwoPhaseAMRIncompleteLUExt, "IncompleteLU")
    return ExternalPreconditioner(
        :ilu, extension.build_incomplete_lu_preconditioner(matrix; kwargs...),
        matrix
    )
end

"""
    incomplete_cholesky_preconditioner(matrix; kwargs...)

Construct a LimitedLDLFactorizations.jl limited-memory incomplete LDLᵀ
preconditioner. For a positive (semi-)definite matrix this is the symmetric
incomplete-Cholesky option. Import `LimitedLDLFactorizations` before calling it.
"""
function incomplete_cholesky_preconditioner(matrix; kwargs...)
    extension = _optional_extension(
        :TwoPhaseAMRLimitedLDLFactorizationsExt,
        "LimitedLDLFactorizations"
    )
    return ExternalPreconditioner(
        :incomplete_cholesky,
        extension.build_incomplete_cholesky_preconditioner(matrix; kwargs...),
        matrix
    )
end

function _construct_krylov(krylov, matrix, right_hand_side)
    workspace = applicable(krylov, matrix, right_hand_side) ?
        krylov(matrix, right_hand_side) : krylov
    size(matrix) == (workspace.m, workspace.n) ||
        throw(
        DimensionMismatch(
            "Krylov workspace and elliptic matrix sizes differ"
        )
    )
    return workspace
end

function _construct_matrix_preconditioner(
        preconditioner, matrix;
        symmetric, singular,
        options = (;)
    )
    preconditioner === nothing && return nothing
    if preconditioner === :amg
        return algebraic_multigrid_preconditioner(
            matrix; symmetric, singular, options...
        )
    elseif preconditioner === :ilu
        return incomplete_lu_preconditioner(matrix; options...)
    elseif preconditioner === :ichol
        symmetric || throw(
            ArgumentError(
                "incomplete Cholesky requires a symmetric operator"
            )
        )
        return incomplete_cholesky_preconditioner(matrix; options...)
    elseif preconditioner isa Symbol
        throw(
            ArgumentError(
                "preconditioner must be :geometric, :amg, :ilu, :ichol, " *
                    ":tree_geometric (for tree-aware operators), nothing, " *
                    "a factory, or a pre-built preconditioner"
            )
        )
    elseif applicable(preconditioner, matrix)
        isempty(options) || throw(
            ArgumentError(
                "preconditioner_options cannot accompany a preconditioner factory"
            )
        )
        return preconditioner(matrix)
    end
    isempty(options) || throw(
        ArgumentError(
            "preconditioner_options cannot accompany a pre-built preconditioner"
        )
    )
    return preconditioner
end

function _construct_pressure_preconditioner(
        preconditioner,
        operator::EllipticOperator;
        coarse_size, pre_sweeps,
        post_sweeps, max_levels,
        options = (;)
    )
    if preconditioner in (:geometric, :tree_geometric)
        settings = merge(
            (; coarse_size, pre_sweeps, post_sweeps, max_levels),
            options,
        )
        return TreeGeometricMultigridPreconditioner(operator; settings...)
    end
    return _construct_matrix_preconditioner(
        preconditioner, operator.matrix;
        symmetric = true, singular = iszero(operator.reaction), options
    )
end

"""
    EllipticSolver(operator, krylov, preconditioner)
    EllipticSolver(operator; krylov=Krylov.CgWorkspace,
                   krylov_options=(;),
                   preconditioner=:tree_geometric, kwargs...)

Couple an elliptic operator to an external Krylov.jl workspace and an
independent preconditioner. The default is Krylov.jl CG with this package's
geometric multigrid V-cycle. A pre-built external Krylov workspace and any
preconditioner implementing three-argument `ldiv!` may be passed directly.
CG, BiCGSTAB, GMRES, and FGMRES workspaces are supported. `krylov_options`
forwards method settings while convergence tolerances, iteration limits, and
preconditioner placement remain controlled by `solve!` and this composition.
"""
mutable struct EllipticSolver{T, O, A, K, P, Q}
    operator::O
    action::A
    krylov::K
    preconditioner::P
    krylov_options::Q
    solution::Vector{T}
    right_hand_side::Vector{T}
    residual::Vector{T}
    product::Vector{T}
    stats::LinearSolverStats{T}
end

function EllipticSolver(
        operator::O, krylov, preconditioner;
        krylov_options = (;)
    ) where
    {T, O <: EllipticOperator{T}}
    number_of_unknowns = length(operator.volumes)
    (krylov.m, krylov.n) == (number_of_unknowns, number_of_unknowns) ||
        throw(
        DimensionMismatch(
            "Krylov workspace and elliptic operator sizes differ"
        )
    )
    method = _krylov_method(krylov)
    return EllipticSolver(
        operator, FactorizedEllipticOperator(operator),
        krylov, preconditioner, krylov_options,
        zeros(T, number_of_unknowns), zeros(T, number_of_unknowns),
        zeros(T, number_of_unknowns), zeros(T, number_of_unknowns),
        LinearSolverStats(
            T; method,
            preconditioner = _preconditioner_name(preconditioner)
        )
    )
end

function EllipticSolver(
        operator::StaggeredEllipticComponent{Axis, T},
        krylov, preconditioner;
        krylov_options = (;)
    ) where {Axis, T}
    number_of_unknowns = length(operator.volumes)
    (krylov.m, krylov.n) == (number_of_unknowns, number_of_unknowns) ||
        throw(
        DimensionMismatch(
            "Krylov workspace and elliptic operator sizes differ"
        )
    )
    method = _krylov_method(krylov)
    return EllipticSolver(
        operator, operator.matrix,
        krylov, preconditioner, krylov_options,
        zeros(T, number_of_unknowns), zeros(T, number_of_unknowns),
        zeros(T, number_of_unknowns), zeros(T, number_of_unknowns),
        LinearSolverStats(
            T; method,
            preconditioner = _preconditioner_name(preconditioner)
        )
    )
end

function EllipticSolver(
        operator::EllipticOperator;
        krylov = Krylov.CgWorkspace,
        krylov_options = (;),
        preconditioner = :tree_geometric,
        preconditioner_options = (;),
        coarse_size = 64, pre_sweeps = 2, post_sweeps = 2,
        max_levels = 32
    )
    right_hand_side = zeros(eltype(operator.volumes), length(operator.volumes))
    action = FactorizedEllipticOperator(operator)
    workspace = _construct_krylov(krylov, action, right_hand_side)
    built_preconditioner = _construct_pressure_preconditioner(
        preconditioner, operator; coarse_size, pre_sweeps, post_sweeps,
        max_levels, options = preconditioner_options
    )
    return EllipticSolver(
        operator, workspace, built_preconditioner; krylov_options
    )
end

"""Retarget pressure-solver storage after an AMR topology change.

The sparse operator and external Krylov workspace necessarily change size.
The package-owned solution/residual vectors and factorized face scratch grow
or shrink in place. Returns `nothing` when a concrete field type changed and a
new solver object is required.
"""
function _reconfigure_elliptic_solver!(
        solver::EllipticSolver{T}, operator::EllipticOperator{T},
        krylov_factory, preconditioner,
    ) where {T}
    typeof(operator) === typeof(solver.operator) || return nothing
    typeof(preconditioner) === typeof(solver.preconditioner) || return nothing
    action = solver.action
    action isa FactorizedEllipticOperator || return nothing
    typeof(action.operator) === typeof(operator) || return nothing

    action.operator = operator
    resize!(action.face_input, length(operator.face_coefficient))
    resize!(action.face_output, length(operator.face_coefficient))
    fill!(action.face_input, zero(T))
    fill!(action.face_output, zero(T))
    number_of_unknowns = length(operator.volumes)
    for values in (
            solver.solution, solver.right_hand_side,
            solver.residual, solver.product,
        )
        resize!(values, number_of_unknowns)
        fill!(values, zero(T))
    end
    krylov = _construct_krylov(
        krylov_factory, action, solver.right_hand_side,
    )
    typeof(krylov) === typeof(solver.krylov) || return nothing
    solver.operator = operator
    solver.krylov = krylov
    solver.preconditioner = preconditioner
    solver.stats = LinearSolverStats(
        T; method = _krylov_method(krylov),
        preconditioner = _preconditioner_name(preconditioner),
    )
    return solver
end

_krylov_method(::Krylov.CgWorkspace) = :cg
_krylov_method(::Krylov.BicgstabWorkspace) = :bicgstab
_krylov_method(::Krylov.GmresWorkspace) = :gmres
_krylov_method(::Krylov.FgmresWorkspace) = :fgmres
_krylov_method(workspace) = Symbol(nameof(typeof(workspace)))

struct StrongResidualCallback{T, V}
    target::T
    volumes::V
end

function (callback::StrongResidualCallback)(workspace)
    _, maximum_norm = _integrated_residual_norms(
        workspace.r, callback.volumes
    )
    return maximum_norm <= callback.target
end

function _run_krylov!(
        workspace::Krylov.CgWorkspace, matrix,
        right_hand_side, initial_solution, preconditioner,
        target, volumes, max_iterations, options
    )
    callback = StrongResidualCallback(target, volumes)
    settings = merge(
        options, (;
            atol = zero(target), rtol = zero(target),
            itmax = max_iterations, callback,
        )
    )
    if isnothing(preconditioner)
        Krylov.cg!(
            workspace, matrix, right_hand_side, initial_solution;
            settings...
        )
    else
        settings = merge(settings, (; M = preconditioner, ldiv = true))
        Krylov.cg!(
            workspace, matrix, right_hand_side, initial_solution;
            settings...
        )
    end
    return workspace
end

function _run_krylov!(
        workspace::Krylov.BicgstabWorkspace, matrix,
        right_hand_side, initial_solution, preconditioner,
        target, volumes, max_iterations, options
    )
    callback = StrongResidualCallback(target, volumes)
    settings = merge(
        options, (;
            atol = zero(target), rtol = zero(target),
            itmax = max_iterations, callback,
        )
    )
    if isnothing(preconditioner)
        Krylov.bicgstab!(
            workspace, matrix, right_hand_side, initial_solution;
            settings...
        )
    else
        preconditioned_settings = merge(
            settings, (; N = preconditioner, ldiv = true)
        )
        Krylov.bicgstab!(
            workspace, matrix, right_hand_side, initial_solution;
            preconditioned_settings...
        )
    end
    # Krylov.jl's BiCGSTAB currently evaluates omega before testing the
    # intermediate residual.  An (almost) exact preconditioner can therefore
    # produce 0/0 for a right-hand side contained in one eigenspace.  Retry the
    # same external method without the preconditioner only for that breakdown;
    # `initial_solution` has not been mutated by `bicgstab!`.
    if !all(isfinite, Krylov.solution(workspace)) &&
            !isnothing(preconditioner)
        Krylov.bicgstab!(
            workspace, matrix, right_hand_side, initial_solution;
            settings...
        )
    end
    return workspace
end

# GMRES does not expose the current true residual to callbacks until the end
# of an Arnoldi pass. Since the Krylov residual is the integrated residual for
# right preconditioning, this Euclidean tolerance is a conservative guarantee
# for the package's strong L-infinity residual contract:
# norm(r, 2) <= target*minimum(volumes) implies abs(r[i])/volumes[i] <= target.
function _gmres_absolute_tolerance(target, volumes)
    return target * minimum(volumes)
end

function _run_krylov!(
        workspace::Krylov.GmresWorkspace, matrix,
        right_hand_side, initial_solution, preconditioner,
        target, volumes, max_iterations, options
    )
    atol = _gmres_absolute_tolerance(target, volumes)
    settings = merge(
        options, (;
            atol, rtol = zero(target), itmax = max_iterations,
        )
    )
    if isnothing(preconditioner)
        Krylov.gmres!(
            workspace, matrix, right_hand_side, initial_solution;
            settings...
        )
    else
        settings = merge(settings, (; N = preconditioner, ldiv = true))
        Krylov.gmres!(
            workspace, matrix, right_hand_side, initial_solution;
            settings...
        )
    end
    return workspace
end

function _run_krylov!(
        workspace::Krylov.FgmresWorkspace, matrix,
        right_hand_side, initial_solution, preconditioner,
        target, volumes, max_iterations, options
    )
    atol = _gmres_absolute_tolerance(target, volumes)
    settings = merge(
        options, (;
            atol, rtol = zero(target), itmax = max_iterations,
        )
    )
    if isnothing(preconditioner)
        Krylov.fgmres!(
            workspace, matrix, right_hand_side, initial_solution;
            settings...
        )
    else
        settings = merge(settings, (; N = preconditioner, ldiv = true))
        Krylov.fgmres!(
            workspace, matrix, right_hand_side, initial_solution;
            settings...
        )
    end
    return workspace
end

function _run_krylov!(
        workspace, matrix, right_hand_side, initial_solution,
        preconditioner, target, volumes, max_iterations, options
    )
    throw(
        ArgumentError(
            "unsupported Krylov workspace $(typeof(workspace)); extend " *
                "TwoPhaseAMR._run_krylov! for this external solver"
        )
    )
end

function _true_residual!(solver::EllipticSolver)
    if solver.action isa FactorizedEllipticOperator
        solver.action.operator = solver.operator
    end
    mul!(solver.product, solver.action, solver.solution)
    for index in eachindex(solver.residual)
        @inbounds solver.residual[index] =
            solver.right_hand_side[index] - solver.product[index]
    end
    return solver.residual
end

"""
    solve!(solution, solver, rhs; rtol=1e-10, atol=0,
           max_cycles=100, compatibility=:project)

Solve the strong-form equation `L(solution) = rhs` with the external Krylov
workspace and independent preconditioner stored in `solver`.
For periodic or all-Neumann Poisson problems the right-hand side must have zero
volume mean.
The default `compatibility=:project` removes and reports its discrete mean;
use `:error` to reject an incompatible right-hand side.  The returned pressure
always has zero volume-weighted mean, which fixes the constant nullspace.
"""
function solve!(
        solution::CellField{T}, solver::EllipticSolver{T},
        right_hand_side::CellField{T}; rtol = 1.0e-10, atol = 0,
        max_cycles = 100, compatibility = :project
    ) where {T}
    solution.mesh === right_hand_side.mesh === solver.operator.mesh ||
        throw(ArgumentError("fields and multigrid solver must share a mesh"))
    zero(T) <= rtol < one(T) || throw(ArgumentError("rtol must lie in [0, 1)"))
    atol >= zero(T) || throw(ArgumentError("atol must be nonnegative"))
    max_cycles >= 0 || throw(ArgumentError("max_cycles must be nonnegative"))
    compatibility in (:project, :error) ||
        throw(ArgumentError("compatibility must be :project or :error"))

    _copy_from_field!(solver.solution, solution, solver.operator.references)
    _copy_from_field!(
        solver.right_hand_side, right_hand_side,
        solver.operator.references
    )
    volumes = solver.operator.volumes
    for index in eachindex(solver.right_hand_side)
        @inbounds solver.right_hand_side[index] *= volumes[index]
    end

    removed_mean = zero(T)
    singular = iszero(solver.operator.reaction)
    if singular
        integrated_sum = sum(solver.right_hand_side)
        scale = sum(abs, solver.right_hand_side)
        tolerance = T(256) * eps(T) * max(one(T), scale)
        if compatibility === :error && abs(integrated_sum) > tolerance
            throw(
                ArgumentError(
                    "periodic Poisson right-hand side has nonzero volume mean"
                )
            )
        end
        removed_mean = _project_integrated_rhs!(solver.right_hand_side, volumes)
        _remove_volume_mean!(solver.solution, volumes)
    end

    _true_residual!(solver)
    initial_l2, initial_linf =
        _integrated_residual_norms(solver.residual, volumes)
    roundoff_floor = _roundoff_residual_floor!(
        solver.operator.matrix, solver.solution, solver.right_hand_side,
        volumes, solver.product
    )
    target = max(
        T(atol), T(rtol) * max(initial_linf, eps(T)),
        roundoff_floor
    )
    if initial_linf > target
        _run_krylov!(
            solver.krylov, solver.action,
            solver.right_hand_side, solver.solution, solver.preconditioner,
            target, volumes, max_cycles, solver.krylov_options
        )
        copyto!(solver.solution, Krylov.solution(solver.krylov))
    end
    singular && _remove_volume_mean!(solver.solution, volumes)
    _true_residual!(solver)
    final_l2, final_linf =
        _integrated_residual_norms(solver.residual, volumes)
    final_target = max(
        target,
        _roundoff_residual_floor!(
            solver.operator.matrix, solver.solution,
            solver.right_hand_side, volumes, solver.product
        )
    )
    _copy_to_field!(solution, solver.solution, solver.operator.references)

    stats = solver.stats
    stats.cycles = initial_linf <= target ? 0 :
        Krylov.iteration_count(solver.krylov)
    stats.converged = final_linf <= final_target
    stats.initial_l2 = initial_l2
    stats.final_l2 = final_l2
    stats.final_linf = final_linf
    stats.removed_rhs_mean = removed_mean
    stats.method = _krylov_method(solver.krylov)
    stats.preconditioner = _preconditioner_name(solver.preconditioner)
    return stats
end

"""Return `(L2, LInf)` strong residual norms from the most recent solve."""
residual_norms(solver::EllipticSolver) =
    (solver.stats.final_l2, solver.stats.final_linf)

"""Reusable divergence, gradient, and elliptic scratch for `project!`."""
struct ProjectionWorkspace{T, M <: BlockTreeMesh}
    divergence::CellField{T, M}
    weighted_gradient::FaceVelocity{T, M}
    elliptic::EllipticWorkspace{T}
end

function ProjectionWorkspace(solver::EllipticSolver{T}) where {T}
    mesh = solver.operator.mesh
    divergence = CellField(mesh, T)
    weighted_gradient = FaceVelocity(
        mesh, (x, y, time, axis) -> zero(T); T
    )
    return ProjectionWorkspace(
        divergence, weighted_gradient,
        EllipticWorkspace(solver.operator)
    )
end

"""
    project!(velocity, pressure, solver, dt, workspace; solve_options...)

Apply the pressure projection using exactly the weighted gradient in the
solver's Poisson operator. Physical face velocities are first set to zero; the
corresponding pressure correction has homogeneous Neumann data. The pressure is overwritten by the mean-zero
solution of `-diffusion*D*G_kappa*p = -D*u/dt`, then
`u <- u - dt*diffusion*G_kappa*p`.
"""
function project!(
        velocity::FaceVelocity{T}, pressure::CellField{T},
        solver::EllipticSolver{T}, dt,
        workspace::ProjectionWorkspace{T}; kwargs...
    ) where {T}
    velocity.mesh === pressure.mesh === solver.operator.mesh ===
        workspace.divergence.mesh ||
        throw(ArgumentError("projection arguments must share a mesh"))
    iszero(solver.operator.reaction) ||
        throw(ArgumentError("pressure projection requires a zero-reaction operator"))
    dt > zero(dt) || throw(ArgumentError("dt must be strictly positive"))

    enforce_no_permeation!(velocity)
    divergence!(workspace.divergence, velocity)
    nx, ny = velocity.mesh.block_size
    for block in eachindex(velocity.mesh.blocks), j in 1:ny, i in 1:nx
        reference = CellRef(block, i, j)
        _set!(
            workspace.divergence,
            -_get(workspace.divergence, reference) / dt, reference
        )
    end
    stats = solve!(pressure, solver, workspace.divergence; kwargs...)
    weighted_gradient!(
        workspace.weighted_gradient, pressure,
        solver.operator, workspace.elliptic
    )
    scale = T(dt) * solver.operator.diffusion
    for index in eachindex(velocity.interior)
        @inbounds velocity.interior[index] -=
            scale * workspace.weighted_gradient.interior[index]
    end
    for index in eachindex(velocity.boundary)
        @inbounds velocity.boundary[index] -=
            scale * workspace.weighted_gradient.boundary[index]
    end
    enforce_no_permeation!(velocity)
    return stats
end
