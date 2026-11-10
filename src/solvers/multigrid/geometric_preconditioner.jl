using LinearAlgebra: lu!

# UMFPACK owns a zero-based copy of the pattern used for its symbolic
# factorization. Compare the complete CSC structure, including explicit zeros,
# before reusing it; component constraints change with activity/connectivity.
function _refactorize_coarse!(factorization, matrix::SparseMatrixCSC)
    columns = matrix.colptr
    rows = rowvals(matrix)
    size(factorization) == size(matrix) &&
        length(factorization.colptr) == length(columns) &&
        length(factorization.rowval) == length(rows) || return lu(matrix)
    for index in eachindex(columns)
        @inbounds factorization.colptr[index] == columns[index] - 1 ||
            return lu(matrix)
    end
    for index in eachindex(rows)
        @inbounds factorization.rowval[index] == rows[index] - 1 ||
            return lu(matrix)
    end
    # The pattern and workspace are already current. Copy only values so
    # Float32 input does not convert the CSC indices through Float32.
    copyto!(factorization.nzval, nonzeros(matrix))
    return lu!(factorization)
end

abstract type AbstractGeometricMultigridPreconditioner{T} end

"""
    TreeGeometricMultigridPreconditioner(operator; kwargs...)

Rediscretized geometric multigrid on successively capped block trees and
coarser base meshes.

By default, each coarse mesh simultaneously replaces every complete terminal
sibling family by its parent; `pruning=:deepest` instead caps only the current
deepest generation. The tree constructor restores 2:1 grading. Below the
root level, 2×2 root-block agglomeration continues while both root counts are
even and each periodic axis retains at least four cells, subject to
`coarse_size` and `max_levels`. When further coarsening is unavailable, the
last level is factorized directly, even if it exceeds `coarse_size`. The same
compatible elliptic discretization is constructed on
every mesh, without a Galerkin `P' * A * P` product. A coefficient-dependent
smoothed cell transfer preserves the active-domain pressure null vector, and
its adjoint restricts integrated residuals. Each level projects and gauges
only its unanchored connected components, with one coarse constraint per
unanchored coarse component; coarsening can change that set. Set
`multigrid=MultigridOptions(...)` to configure the hierarchy and smoothing;
individual keywords override its corresponding fields.
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
    boundary_restrictions::Vector{R}
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

# Retain the existing connection-based coarse activity extension for
# nonsingular cut-cell operators.
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

# The tentative owner, rather than a positive coarse face, determines whether
# a coarse cell contains fluid.  In particular an isolated active cell is a
# legitimate singleton nullspace, not a covered-cell constraint.
function _coarse_cut_cell_activity(
        mesh, tentative::SparseMatrixCSC, fine_volumes, ::Type{T},
    ) where {T}
    active = falses(ncells(mesh))
    rows = rowvals(tentative)
    for column in axes(tentative, 2), pointer in nzrange(tentative, column)
        @inbounds fine_volumes[rows[pointer]] > zero(T) && (active[column] = true)
    end
    volumes = _cell_volumes(mesh, T)
    for cell in eachindex(volumes)
        @inbounds active[cell] || (volumes[cell] = zero(T))
    end
    return active, volumes
end

function _refresh_coarse_cut_cell_activity!(
        coarse::EllipticOperator{T}, fine::EllipticOperator{T}, refresh,
    ) where {T}
    prolongation = refresh.prolongation
    rows = rowvals(prolongation)
    cells_per_block = prod(coarse.mesh.block_size)
    for column in axes(prolongation, 2)
        active = false
        for pointer in nzrange(prolongation, column)
            if (@inbounds refresh.tentative_entries[pointer]) &&
                    (@inbounds fine.volumes[rows[pointer]]) > zero(T)
                active = true
                break
            end
        end
        @inbounds begin
            coarse.constraints[column] = active ? zero(T) : one(T)
            block = (column - 1) ÷ cells_per_block + 1
            coarse.volumes[column] = active ?
                T(coarse.mesh.blocks[block].volume) : zero(T)
        end
    end
    return coarse
end

function _multigrid_coarse_matrix(level::MultigridLevel)
    matrix = level.matrix
    matrix isa FactorizedEllipticLevelAction || return matrix
    stencil = matrix.stencil
    number_of_cells = length(level.volumes)
    return sparse(
        stencil.entry_rows, stencil.entry_columns, stencil.values,
        number_of_cells, number_of_cells,
    )
end

function _augmented_multigrid_coarse_matrix(level::MultigridLevel{T}) where {T}
    number_of_cells = length(level.volumes)
    number_of_components = length(level.component_volumes)
    rows = Int[]
    columns = Int[]
    values = T[]
    sizehint!(rows, number_of_cells)
    sizehint!(columns, number_of_cells)
    sizehint!(values, number_of_cells)
    for cell in 1:number_of_cells
        component = @inbounds level.component_indices[cell]
        iszero(component) && continue
        push!(rows, cell)
        push!(columns, component)
        push!(values, @inbounds level.volumes[cell])
    end
    constraints = sparse(
        rows, columns, values, number_of_cells, number_of_components,
    )
    return sparse(
        [
            _multigrid_coarse_matrix(level) constraints;
            transpose(constraints) spzeros(T, number_of_components, number_of_components)
        ],
    )
end

function TreeGeometricMultigridPreconditioner(
        operator::O; multigrid::MultigridOptions = MultigridOptions(),
        coarse_size = multigrid.coarse_size,
        pre_sweeps = multigrid.pre_sweeps, post_sweeps = multigrid.post_sweeps,
        max_levels = multigrid.max_levels, pruning = :all, reuse = nothing,
        geometry = nothing,
    ) where {T, O <: EllipticOperator{T}}
    multigrid = MultigridOptions(; coarse_size, pre_sweeps, post_sweeps, max_levels)
    (; coarse_size, pre_sweeps, post_sweeps, max_levels) = multigrid
    plan = _TreeCoarseningPlan(
        operator.mesh, pruning, coarse_size, max_levels, Val(:pressure),
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
    isnothing(geometry) || operator.inactive_condition === :decoupled || throw(
        ArgumentError(
            "geometry-aware tree multigrid requires decoupled inactive cells",
        ),
    )
    can_reuse = isnothing(geometry) && _tree_can_reuse(plan, reuse) &&
        iszero(reuse.operator.reaction - operator.reaction) &&
        iszero(reuse.operator.diffusion - operator.diffusion) &&
        reuse.operator.boundary_sides == operator.boundary_sides &&
        reuse.operator.inactive_condition === operator.inactive_condition

    operators = EllipticOperator{T}[operator]
    geometry_levels = Any[]
    isnothing(geometry) || push!(geometry_levels, geometry)
    prolongations = SparseMatrixCSC{T, Int}[]
    transfer_refreshes = Any[]
    coefficient_restrictions = FaceCoefficientRestriction{T}[]
    boundary_restrictions = FaceCoefficientRestriction{T}[]
    current_mesh = operator.mesh
    matrix_free_fine = isnothing(operator.matrix)
    reuse_operators = can_reuse ? reuse.operators : nothing
    _tree_reuse_suffix!(
        plan, reuse_operators, current_mesh, 0, !matrix_free_fine,
    )
    while true
        coarse_mesh = _tree_next_coarse_mesh!(
            plan, operator.mesh, current_mesh, ncells(current_mesh),
            length(operators),
        )
        isnothing(coarse_mesh) && break
        # Geometry-dependent coefficients and activity must be recomputed after
        # AMR, but an unchanged coarse topology itself can still be shared.
        if !isnothing(geometry) && !isnothing(reuse) &&
                reuse.pruning === plan.pruning && !isempty(reuse.geometry_levels)
            matching_mesh = _matching_tree_level(reuse.operators, coarse_mesh)
            iszero(matching_mesh) ||
                (coarse_mesh = reuse.operators[matching_mesh].mesh)
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
        boundary_restriction = _face_coefficient_restriction(
            coarse_mesh, current_mesh, T; boundary = true,
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
        coarse_boundary = zeros(T, length(coarse_mesh.boundary_faces))
        _restrict_face_coefficients!(
            coarse_boundary, last(operators).boundary_face_coefficient,
            boundary_restriction,
        )
        matching_level = _tree_reuse_suffix!(
            plan, reuse_operators, coarse_mesh, length(operators),
        )
        can_append_suffix = !iszero(matching_level)
        coarse_operator = if !isnothing(coarse_geometry)
            PoissonOperator(
                coarse_geometry; face_coefficient = coarse_coefficient,
                velocity_boundary = operator.boundary_sides,
                boundary_face_coefficient = coarse_boundary,
                construction_cache =
                    operator.gradient_operator.construction_cache,
            )
        elseif can_append_suffix
            candidate = reuse.operators[matching_level]
            if iszero(operator.reaction)
                _refresh_coarse_cut_cell_activity!(
                    candidate, last(operators), transfer_refresh,
                )
            end
            _update_elliptic_boundary_coefficient!(candidate, coarse_boundary)
            copyto!(candidate.face_coefficient, coarse_coefficient)
            _update_elliptic_face_coefficient!(
                candidate, candidate.face_coefficient,
            )
            candidate
        else
            cut_extension = any(!iszero, operator.constraints)
            active, volumes = if !cut_extension
                (nothing, nothing)
            elseif iszero(operator.reaction)
                _coarse_cut_cell_activity(
                    coarse_mesh, tentative, last(operators).volumes, T,
                )
            else
                _coarse_cut_cell_activity(coarse_mesh, coarse_coefficient, T)
            end
            EllipticOperator(
                coarse_mesh;
                reaction = operator.reaction,
                diffusion = operator.diffusion,
                face_coefficient = coarse_coefficient,
                velocity_boundary = operator.boundary_sides,
                boundary_face_coefficient = coarse_boundary,
                construction_cache =
                    operator.gradient_operator.construction_cache,
                active_cells = active, volumes,
                inactive_condition = operator.inactive_condition, T,
            )
        end
        if isnothing(coarse_geometry) && iszero(operator.reaction) &&
                (
                any(!iszero, last(operators).constraints) ||
                    any(!iszero, coarse_operator.constraints)
            )
            _restrict_transfer_to_active_domain!(
                prolongation, last(operators).volumes, coarse_operator.volumes,
                transfer_refresh isa FactorizedSmoothedTransferRefresh ?
                    transfer_refresh.stencil.row_sums : transfer_refresh.row_sums,
            )
        end
        push!(prolongations, prolongation)
        push!(transfer_refreshes, transfer_refresh)
        push!(coefficient_restrictions, coefficient_restriction)
        push!(boundary_restrictions, boundary_restriction)
        push!(operators, coarse_operator)
        isnothing(coarse_geometry) ||
            push!(geometry_levels, coarse_geometry)
        current_mesh = coarse_mesh
    end

    reused_new_start = plan.reused_new_start
    reused_old_start = plan.reused_old_start
    if !iszero(reused_new_start)
        for old_index in (reused_old_start + 1):length(reuse.operators)
            push!(operators, reuse.operators[old_index])
        end
        for old_index in reused_old_start:length(reuse.transfer_refreshes)
            push!(transfer_refreshes, reuse.transfer_refreshes[old_index])
            push!(
                coefficient_restrictions,
                reuse.coefficient_restrictions[old_index],
            )
            push!(
                boundary_restrictions, reuse.boundary_restrictions[old_index],
            )
            push!(
                prolongations,
                reuse.transfer_refreshes[old_index].prolongation,
            )
        end
    end

    levels = _multigrid_levels(T, matrix_free_fine)
    for index in eachindex(operators)
        prolongation = index < length(operators) ?
            prolongations[index] :
            spzeros(T, length(operators[index].volumes), 0)
        if matrix_free_fine && index == 1
            stencil = isempty(transfer_refreshes) ?
                _factorized_elliptic_stencil(operators[index]) :
                transfer_refreshes[1].stencil
            push!(
                levels,
                _factorized_elliptic_level(
                    operators[index], prolongation, stencil,
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
                    prolongation; singular = operators[index].singular,
                ),
            )
        end
        _refresh_multigrid_components!(last(levels), operators[index])
    end
    coarse = last(levels)
    singular = !isempty(coarse.component_volumes)
    if !iszero(reused_new_start)
        factorization = reuse.coarse_factorization
        coarse_rhs = reuse.coarse_rhs
        coarse_solution = reuse.coarse_solution
    elseif singular
        factorization = lu(_augmented_multigrid_coarse_matrix(coarse))
        coarse_rhs = zeros(
            T, length(coarse.volumes) + length(coarse.component_volumes),
        )
        coarse_solution = similar(coarse_rhs)
    else
        factorization = lu(_multigrid_coarse_matrix(coarse))
        coarse_rhs = zeros(T, length(coarse.volumes))
        coarse_solution = similar(coarse_rhs)
    end
    preconditioner = TreeGeometricMultigridPreconditioner(
        operator, operators, geometry_levels, levels, transfer_refreshes,
        coefficient_restrictions, boundary_restrictions,
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
        level::MultigridLevel{T}, matrix, volumes, prolongation;
        singular = false,
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
    level.relaxation_weight = _jacobi_weight!(
        inverse_diagonal, row_sums; allow_zero_diagonal = singular,
    )
    return level
end

function _refresh_geometric_multigrid!(
        preconditioner::TreeGeometricMultigridPreconditioner{T},
        operator::EllipticOperator{T},
    ) where {T}
    iszero(preconditioner.operator.reaction) == iszero(operator.reaction) || throw(
        ArgumentError("changing between pressure and shifted operators requires rebuilding the multigrid hierarchy"),
    )
    preconditioner.operator.inactive_condition === operator.inactive_condition || throw(
        ArgumentError("changing the inactive-cell condition requires rebuilding the multigrid hierarchy"),
    )
    preconditioner.operator.boundary_sides == operator.boundary_sides || throw(
        ArgumentError("changing open pressure boundary sides requires rebuilding the multigrid hierarchy"),
    )
    preconditioner.operator = operator
    preconditioner.operators[1] = operator
    for index in 2:length(preconditioner.operators)
        fine = preconditioner.operators[index - 1]
        coarse = preconditioner.operators[index]
        restriction = preconditioner.coefficient_restrictions[index - 1]
        if isempty(preconditioner.geometry_levels) && iszero(operator.reaction)
            _refresh_coarse_cut_cell_activity!(
                coarse, fine, preconditioner.transfer_refreshes[index - 1],
            )
        end
        _restrict_face_coefficients!(
            coarse.boundary_face_coefficient, fine.boundary_face_coefficient,
            preconditioner.boundary_restrictions[index - 1],
        )
        _update_elliptic_boundary_coefficient!(
            coarse, coarse.boundary_face_coefficient,
        )
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
        if !isempty(preconditioner.geometry_levels)
            _restrict_transfer_to_active_domain!(
                prolongation,
                preconditioner.geometry_levels[index].active_cells,
                preconditioner.geometry_levels[index + 1].active_cells,
                refresh isa FactorizedSmoothedTransferRefresh ?
                    refresh.stencil.row_sums : refresh.row_sums,
            )
        elseif iszero(operator.reaction)
            fine = preconditioner.operators[index]
            coarse = preconditioner.operators[index + 1]
            if any(!iszero, fine.constraints) || any(!iszero, coarse.constraints)
                _restrict_transfer_to_active_domain!(
                    prolongation, fine.volumes, coarse.volumes,
                    refresh isa FactorizedSmoothedTransferRefresh ?
                        refresh.stencil.row_sums : refresh.row_sums,
                )
            end
        end
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
                level.prolongation; singular = level_operator.singular,
            )
        end
        _refresh_multigrid_components!(level, level_operator)
    end
    coarse = last(preconditioner.levels)
    preconditioner.coarse_augmented = !isempty(coarse.component_volumes)
    coarse_matrix = preconditioner.coarse_augmented ?
        _augmented_multigrid_coarse_matrix(coarse) :
        _multigrid_coarse_matrix(coarse)
    number_of_unknowns = length(coarse.volumes) + length(coarse.component_volumes)
    resize!(preconditioner.coarse_rhs, number_of_unknowns)
    resize!(preconditioner.coarse_solution, number_of_unknowns)
    preconditioner.coarse_factorization = _refactorize_coarse!(
        preconditioner.coarse_factorization, coarse_matrix,
    )
    return preconditioner
end
