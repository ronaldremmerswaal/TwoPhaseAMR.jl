"""Rediscretized full-vector diffusion multigrid on pruned block trees.

Each coarse level has its own staggered strain/stress discretization. Cell
log-viscosity is volume averaged, staggered density is area averaged on
coincident faces, and the shifted mass matrix is then assembled directly on
the coarse mesh. Thus no level forms `P' * A * P`. The face transfer is a
topology-only staggered interpolation between nested meshes. Below the root
level, 2×2 root-block agglomeration continues until the target size is reached,
either root count is odd, or a periodic axis would have fewer than four cells;
the last operator is factorized directly.

With an embedded solid, `options = (; geometry)` (the `GeometryField` of the
fine operator; the `NavierStokesSystem` passes it itself) rediscretizes every
coarse level on the geometrically restricted solid, so the coarse operators
keep the walls, the face transfers skip completely covered faces, as in
`TreeStaggeredGeometricMultigridPreconditioner`, and the
Jacobi smoother uses per-row weights (`_local_jacobi_relaxation!`).
Every coarse operator carries the fine operator's cut-cell closure, and a
cached hierarchy built for another closure is not reused.
"""

mutable struct TreeVariableDiffusionPreconditioner{T, F, L, C} <:
    AbstractGeometricMultigridPreconditioner{T}
    operators::Vector{VariableStaggeredDiffusionOperator{T}}
    levels::Vector{L}
    cell_restrictions::Vector{CellCoefficientRestriction{T}}
    density_restrictions::Vector{FaceCoefficientRestriction{T}}
    densities::Vector{Vector{T}}
    construction_cache::C
    coarse_factorization::F
    coarse_augmented::Bool
    coarse_rhs::Vector{T}
    coarse_solution::Vector{T}
    pre_sweeps::Int
    post_sweeps::Int
    pruning::Symbol
    reused_levels::Int
    # Per-row Jacobi weights (`_local_jacobi_relaxation!`), with a solid.
    local_relaxation::Bool
end

_preconditioner_name(::TreeVariableDiffusionPreconditioner) =
    :tree_variable_diffusion_multigrid

function _cell_viscosity_from_log!(field::CellField{T}, values) where {T}
    length(values) == ncells(field.mesh) || throw(
        DimensionMismatch("one log-viscosity value is required per cell"),
    )
    mesh = field.mesh
    nx, ny = mesh.block_size
    h = mesh.halo
    cells_per_block = nx * ny
    _parallel_block_columns(mesh) do b, first_j, last_j
        field_values = @inbounds field.blocks[b]
        offset = (b - 1) * cells_per_block
        @inbounds for j in first_j:last_j, i in 1:nx
            index = offset + (j - 1) * nx + i
            field_values[i + h, j + h] = exp(T(values[index]))
        end
    end
    return field
end

# A cut cell makes a few rows of the stress operator much less diagonally
# dominant than the bulk (row sum over diagonal up to ≈ 3.9 against 2.4–2.7),
# and the global Gershgorin weight 4/(3 maxᵢ sᵢ/dᵢ) of `_jacobi_weight!` then
# damps every row for their sake, so the smoother weakens as the mesh resolves
# more cut cells. Per-row weights ωᵢ = min(1, 4dᵢ/(3sᵢ)) are folded into the
# inverse diagonal (the level weight becomes one). The smoother stays
# symmetric and convergent: 2 diag(d/ω) − A ⪰ diag(s)/2 ≻ 0. Each row is
# independent, so the result does not depend on the thread count.
function _local_jacobi_relaxation!(level::MultigridLevel{T}) where {T}
    matrix_transpose = level.matrix_transpose
    matrix_transpose isa SparseMatrixCSC || throw(
        ArgumentError("assembled multigrid level has no sparse row form"),
    )
    column_pointers = matrix_transpose.colptr
    rows = rowvals(matrix_transpose)
    matrix_values = nonzeros(matrix_transpose)
    inverse_diagonal = level.inverse_diagonal
    @_parallel for column in 1:size(matrix_transpose, 2)
        row_sum = zero(T)
        diagonal = zero(T)
        for pointer in column_pointers[column]:(column_pointers[column + 1] - 1)
            value = @inbounds matrix_values[pointer]
            row_sum += abs(value)
            @inbounds rows[pointer] == column && (diagonal = value)
        end
        @inbounds inverse_diagonal[column] =
            min(one(T), T(4) * diagonal / (T(3) * row_sum)) / diagonal
    end
    level.relaxation_weight = one(T)
    return level
end

function _tree_variable_diffusion_preconditioner(
        operator::VariableStaggeredDiffusionOperator{T}, matrix,
        inertia, dt;
        multigrid::MultigridOptions, options = (;), reuse = nothing,
    ) where {T}
    all(key -> key in (:pruning, :geometry), keys(options)) || throw(
        ArgumentError(
            "tree-geometric diffusion options support only pruning and geometry",
        ),
    )
    pruning = get(options, :pruning, :all)
    geometry = get(options, :geometry, nothing)
    isnothing(geometry) || geometry.mesh === operator.mesh || throw(
        ArgumentError("multigrid geometry must share the operator mesh"),
    )
    (; coarse_size, pre_sweeps, post_sweeps, max_levels) = multigrid
    plan = _TreeCoarseningPlan(
        operator.mesh, pruning, 2coarse_size, max_levels, Val(:variable),
    )
    isnothing(reuse) ||
        reuse isa TreeVariableDiffusionPreconditioner{T} || throw(
        ArgumentError("reused diffusion hierarchy has the wrong type"),
    )
    # Geometry-dependent coarse operators are rebuilt on every construction.
    can_reuse = isnothing(geometry) && _tree_can_reuse(plan, reuse) &&
        typeof(reuse.operators[1].boundary) === typeof(operator.boundary) &&
        typeof(reuse.operators[1].closure) === typeof(operator.closure)
    fine_geometry = geometry

    operators = VariableStaggeredDiffusionOperator{T}[operator]
    matrices = SparseMatrixCSC{T, Int}[matrix]
    inertias = Vector{T}[inertia]
    fine_density = isnothing(reuse) ? T[] : reuse.densities[1]
    resize!(fine_density, length(inertia))
    for index in eachindex(fine_density)
        @inbounds fine_density[index] = inertia[index] / operator.volumes[index]
    end
    densities = Vector{T}[fine_density]
    prolongations = SparseMatrixCSC{T, Int}[]
    cell_restrictions = CellCoefficientRestriction{T}[]
    density_restrictions = FaceCoefficientRestriction{T}[]
    current_mesh = operator.mesh
    construction_cache = isnothing(reuse) ?
        VariableDiffusionConstructionCache(T) : reuse.construction_cache
    reuse_operators = can_reuse ? reuse.operators : nothing
    _tree_reuse_suffix!(plan, reuse_operators, current_mesh, 0)
    if !iszero(plan.reused_new_start)
        initial_matching_level = plan.reused_old_start
        fine_density = reuse.densities[initial_matching_level]
        resize!(fine_density, length(inertia))
        for index in eachindex(fine_density)
            @inbounds fine_density[index] =
                inertia[index] / operator.volumes[index]
        end
        densities[1] = fine_density
    end

    while true
        coarse_mesh = _tree_next_coarse_mesh!(
            plan, operator.mesh, current_mesh, length(current_mesh.faces),
            length(operators),
        )
        isnothing(coarse_mesh) && break

        prolongation = _tree_face_prolongation(
            current_mesh, coarse_mesh, T,
        )
        cell_restriction = _cell_coefficient_restriction(
            coarse_mesh, current_mesh, T,
        )
        density_restriction = _face_coefficient_restriction(
            coarse_mesh, current_mesh, T,
        )
        matching_level = _tree_reuse_suffix!(
            plan, reuse_operators, coarse_mesh, length(operators),
        )
        can_append_suffix = !iszero(matching_level)
        if can_append_suffix
            coarse_operator = reuse.operators[matching_level]
            coarse_density = reuse.densities[matching_level]
            coarse_matrix = reuse.levels[matching_level].matrix
            coarse_inertia = reuse.levels[matching_level].volumes
        else
            coarse_log_viscosity = zeros(T, ncells(coarse_mesh))
            _restrict_cell_coefficient!(
                coarse_log_viscosity, last(operators).cell_work,
                cell_restriction,
            )
            coarse_viscosity = CellField(coarse_mesh, T)
            _cell_viscosity_from_log!(
                coarse_viscosity, coarse_log_viscosity,
            )
            coarse_geometry = isnothing(geometry) ? nothing :
                _restrict_geometry_to_mesh(
                    fine_geometry, coarse_mesh, cell_restriction,
                )
            coarse_operator = _staggered_diffusion_operator(
                coarse_mesh; viscosity = coarse_viscosity,
                boundary = operator.boundary, T, construction_cache,
                averaging = operator.averaging, geometry = coarse_geometry,
                closure = operator.closure,
            )
            if !isnothing(geometry)
                _remove_covered_staggered_transfer!(
                    prolongation, fine_geometry, eachindex(current_mesh.faces),
                    coarse_geometry, eachindex(coarse_mesh.faces),
                )
                fine_geometry = coarse_geometry
            end
            coarse_density = zeros(T, length(coarse_mesh.faces))
            _restrict_face_coefficients!(
                coarse_density, last(densities), density_restriction,
            )
            coarse_matrix, coarse_inertia = _variable_diffusion_matrix(
                coarse_operator, coarse_density, dt,
            )
        end
        push!(prolongations, prolongation)
        push!(cell_restrictions, cell_restriction)
        push!(density_restrictions, density_restriction)
        push!(operators, coarse_operator)
        push!(matrices, coarse_matrix)
        push!(inertias, coarse_inertia)
        push!(densities, coarse_density)
        current_mesh = coarse_mesh
    end

    reused_new_start = plan.reused_new_start
    reused_old_start = plan.reused_old_start
    if !iszero(reused_new_start)
        for old_index in (reused_old_start + 1):length(reuse.operators)
            push!(operators, reuse.operators[old_index])
            push!(matrices, reuse.levels[old_index].matrix)
            push!(inertias, reuse.levels[old_index].volumes)
            push!(densities, reuse.densities[old_index])
        end
        for old_index in reused_old_start:length(reuse.cell_restrictions)
            push!(cell_restrictions, reuse.cell_restrictions[old_index])
            push!(
                density_restrictions,
                reuse.density_restrictions[old_index],
            )
            push!(prolongations, reuse.levels[old_index].prolongation)
        end
    end

    levels = _multigrid_levels(T, false)
    for index in eachindex(matrices)
        prolongation = index < length(matrices) ?
            prolongations[index] :
            spzeros(T, length(inertias[index]), 0)
        if !iszero(reused_new_start) && index >= reused_new_start
            old_index = reused_old_start + index - reused_new_start
            level = reuse.levels[old_index]
            level.matrix = matrices[index]
            level.volumes = inertias[index]
            level.prolongation = prolongation
            push!(levels, level)
        else
            push!(
                levels,
                _multigrid_level(
                    matrices[index], inertias[index], prolongation,
                ),
            )
        end
    end
    local_relaxation = !isnothing(geometry)
    local_relaxation && foreach(_local_jacobi_relaxation!, levels)
    coarse = last(levels)
    if !iszero(reused_new_start)
        factorization = reuse.coarse_factorization
        coarse_rhs = reuse.coarse_rhs
        coarse_solution = reuse.coarse_solution
    else
        factorization = lu(coarse.matrix)
        coarse_rhs = zeros(T, length(coarse.volumes))
        coarse_solution = similar(coarse_rhs)
    end
    preconditioner = TreeVariableDiffusionPreconditioner(
        operators, levels, cell_restrictions, density_restrictions,
        densities, construction_cache,
        factorization, false, coarse_rhs, coarse_solution,
        pre_sweeps, post_sweeps, pruning,
        iszero(reused_new_start) ? 0 :
            length(operators) - reused_new_start + 1,
        local_relaxation,
    )
    !iszero(reused_new_start) &&
        _refresh_tree_variable_diffusion_preconditioner!(
        preconditioner, operator, matrix, inertia, dt,
    )
    return preconditioner
end

"""Refresh numeric hierarchy data, reusing coarse LU symbolic analysis when its sparsity is unchanged."""
function _refresh_tree_variable_diffusion_preconditioner!(
        preconditioner::TreeVariableDiffusionPreconditioner{T},
        operator::VariableStaggeredDiffusionOperator{T}, matrix,
        inertia, dt,
    ) where {T}
    typeof(preconditioner.operators[1].closure) === typeof(operator.closure) ||
        throw(ArgumentError("refreshed diffusion operator changed its cut-cell closure"))
    preconditioner.operators[1] = operator
    fine_density = preconditioner.densities[1]
    for index in eachindex(fine_density)
        @inbounds fine_density[index] =
            inertia[index] / operator.volumes[index]
    end
    preconditioner.levels[1].matrix = matrix
    preconditioner.levels[1].volumes = inertia

    for index in 2:length(preconditioner.operators)
        fine_operator = preconditioner.operators[index - 1]
        coarse_operator = preconditioner.operators[index]
        _restrict_cell_coefficient!(
            coarse_operator.cell_work, fine_operator.cell_work,
            preconditioner.cell_restrictions[index - 1],
        )
        _update_log_dynamic_viscosity!(
            coarse_operator, coarse_operator.cell_work,
        )
        _restrict_face_coefficients!(
            preconditioner.densities[index],
            preconditioner.densities[index - 1],
            preconditioner.density_restrictions[index - 1],
        )
        coarse_inertia = preconditioner.levels[index].volumes
        coarse_density = preconditioner.densities[index]
        for face in eachindex(coarse_inertia)
            @inbounds coarse_inertia[face] =
                coarse_density[face] * coarse_operator.volumes[face]
        end
        _refresh_variable_diffusion_matrix_from_inertia!(
            preconditioner.levels[index].matrix,
            coarse_operator, coarse_inertia, dt,
        )
    end

    for index in eachindex(preconditioner.levels)
        level = preconditioner.levels[index]
        _refresh_multigrid_level!(
            level, level.matrix, level.volumes, level.prolongation,
        )
        preconditioner.local_relaxation && _local_jacobi_relaxation!(level)
    end
    coarse = last(preconditioner.levels)
    preconditioner.coarse_factorization = _refactorize_coarse!(
        preconditioner.coarse_factorization, coarse.matrix,
    )
    return preconditioner
end

function _variable_diffusion_preconditioner(
        specification, operator, matrix, inertia, dt;
        multigrid::MultigridOptions, options, reuse = nothing,
    )
    if specification in (:geometric, :tree_geometric)
        return _tree_variable_diffusion_preconditioner(
            operator, matrix, inertia, dt;
            multigrid, options, reuse,
        )
    end
    isnothing(reuse) || throw(
        ArgumentError("hierarchy reuse requires a tree-geometric preconditioner"),
    )
    return _construct_matrix_preconditioner(
        specification, matrix;
        symmetric = true, singular = false, options
    )
end
