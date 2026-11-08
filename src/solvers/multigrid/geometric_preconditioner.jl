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
        active[_cell_index(face.minus, nx, ny)] = true
        active[_cell_index(face.plus, nx, ny)] = true
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
