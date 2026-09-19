
"""Rediscretized full-vector diffusion multigrid on pruned block trees.

Each coarse level has its own staggered strain/stress discretization. Cell
log-viscosity is volume averaged, staggered density is area averaged on
coincident faces, and the shifted mass matrix is then assembled directly on
the coarse mesh. Thus no level forms `P' * A * P`. The face transfer is a
topology-only staggered interpolation between the nested tree meshes. Once
the root block is reached, its sparse coupled operator is factorized directly.
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

function _tree_variable_diffusion_preconditioner(
        operator::VariableStaggeredDiffusionOperator{T}, matrix,
        inertia, dt;
        coarse_size, pre_sweeps, post_sweeps, max_levels,
        options = (;), reuse = nothing,
    ) where {T}
    all(key -> key === :pruning, keys(options)) || throw(
        ArgumentError(
            "tree-geometric diffusion options currently support only pruning",
        ),
    )
    pruning = get(options, :pruning, :all)
    pruning in (:deepest, :all) || throw(
        ArgumentError("tree pruning must be :deepest or :all"),
    )
    coarse_size >= 1 || throw(ArgumentError("coarse_size must be positive"))
    pre_sweeps >= 0 || throw(ArgumentError("pre_sweeps must be nonnegative"))
    post_sweeps >= 0 || throw(ArgumentError("post_sweeps must be nonnegative"))
    max_levels >= 1 || throw(ArgumentError("max_levels must be positive"))
    isnothing(reuse) ||
        reuse isa TreeVariableDiffusionPreconditioner{T} || throw(
        ArgumentError("reused diffusion hierarchy has the wrong type"),
    )
    can_reuse = !isnothing(reuse) && reuse.pruning === pruning &&
        typeof(reuse.operators[1].boundary) === typeof(operator.boundary)

    operators = VariableStaggeredDiffusionOperator{T}[operator]
    matrices = Any[matrix]
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
    active_level = maximum(block.key.level for block in current_mesh.blocks)
    construction_cache = isnothing(reuse) ?
        VariableDiffusionConstructionCache(T) : reuse.construction_cache
    target_size = 2coarse_size
    reused_new_start = 0
    reused_old_start = 0
    matrix_free_fine = isnothing(matrix)
    initial_matching_level = can_reuse && !matrix_free_fine ?
        _matching_tree_level(reuse.operators, current_mesh, 1) : 0
    initial_suffix_length = iszero(initial_matching_level) ? 0 :
        length(reuse.operators) - initial_matching_level + 1
    if !iszero(initial_matching_level) &&
            initial_suffix_length <= max_levels
        reused_new_start = 1
        reused_old_start = initial_matching_level
        fine_density = reuse.densities[initial_matching_level]
        resize!(fine_density, length(inertia))
        for index in eachindex(fine_density)
            @inbounds fine_density[index] =
                inertia[index] / operator.volumes[index]
        end
        densities[1] = fine_density
        for old_index in (initial_matching_level + 1):length(reuse.operators)
            push!(operators, reuse.operators[old_index])
            push!(matrices, reuse.levels[old_index].matrix)
            push!(inertias, reuse.levels[old_index].volumes)
            push!(densities, reuse.densities[old_index])
        end
        for old_index in initial_matching_level:length(
                reuse.cell_restrictions
            )
            push!(cell_restrictions, reuse.cell_restrictions[old_index])
            push!(
                density_restrictions,
                reuse.density_restrictions[old_index],
            )
            push!(prolongations, reuse.levels[old_index].prolongation)
        end
    end

    while iszero(reused_new_start) &&
            length(current_mesh.faces) > target_size && active_level > 0 &&
            length(operators) < max_levels
        coarse_mesh = if pruning === :deepest
            active_level -= 1
            _capped_tree_mesh(operator.mesh, active_level)
        else
            _all_leaf_pruned_mesh(current_mesh)
        end
        if length(coarse_mesh.faces) >= length(current_mesh.faces)
            pruning === :all && break
            continue
        end

        prolongation = _tree_face_prolongation(
            current_mesh, coarse_mesh, T,
        )
        cell_restriction = _cell_coefficient_restriction(
            coarse_mesh, current_mesh, T,
        )
        density_restriction = _face_coefficient_restriction(
            coarse_mesh, current_mesh, T,
        )
        matching_level = can_reuse ?
            _matching_tree_level(reuse.operators, coarse_mesh, 1) : 0
        suffix_length = iszero(matching_level) ? 0 :
            length(reuse.operators) - matching_level + 1
        can_append_suffix = !iszero(matching_level) &&
            length(operators) + suffix_length <= max_levels
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
            coarse_operator = _staggered_diffusion_operator(
                coarse_mesh; viscosity = coarse_viscosity,
                boundary = operator.boundary,
                construction_cache,
                cache_layer = length(operators),
            )
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
        if can_append_suffix
            reused_new_start = length(operators)
            reused_old_start = matching_level
            for old_index in (matching_level + 1):length(reuse.operators)
                push!(operators, reuse.operators[old_index])
                push!(matrices, reuse.levels[old_index].matrix)
                push!(inertias, reuse.levels[old_index].volumes)
                push!(densities, reuse.densities[old_index])
            end
            for old_index in matching_level:length(reuse.cell_restrictions)
                push!(
                    cell_restrictions,
                    reuse.cell_restrictions[old_index],
                )
                push!(
                    density_restrictions,
                    reuse.density_restrictions[old_index],
                )
                push!(
                    prolongations,
                    reuse.levels[old_index].prolongation,
                )
            end
            break
        end
    end

    levels = _multigrid_levels(T, matrix_free_fine)
    for index in eachindex(matrices)
        prolongation = index < length(matrices) ?
            prolongations[index] :
            spzeros(T, length(inertias[index]), 0)
        if matrix_free_fine && index == 1
            length(matrices) > 1 || throw(
                ArgumentError(
                    "a matrix-free finest diffusion level requires a coarser tree level",
                ),
            )
            push!(
                levels,
                _factorized_variable_diffusion_level(
                    operators[index], inertias[index], dt, prolongation,
                ),
            )
        elseif !iszero(reused_new_start) && index >= reused_new_start
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
    )
    !iszero(reused_new_start) &&
        _refresh_tree_variable_diffusion_preconditioner!(
        preconditioner, operator, matrix, inertia, dt,
    )
    return preconditioner
end

function _refresh_tree_variable_diffusion_preconditioner!(
        preconditioner::TreeVariableDiffusionPreconditioner{T},
        operator::VariableStaggeredDiffusionOperator{T}, matrix,
        inertia, dt,
    ) where {T}
    preconditioner.operators[1] = operator
    fine_density = preconditioner.densities[1]
    for index in eachindex(fine_density)
        @inbounds fine_density[index] =
            inertia[index] / operator.volumes[index]
    end
    if preconditioner.levels[1].matrix isa
            FactorizedVariableDiffusionOperator
        _refresh_factorized_variable_diffusion_level!(
            preconditioner.levels[1], operator, inertia, dt,
        )
    else
        preconditioner.levels[1].matrix = matrix
        preconditioner.levels[1].volumes = inertia
    end

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
        level.matrix isa FactorizedVariableDiffusionOperator && continue
        _refresh_multigrid_level!(
            level, level.matrix, level.volumes, level.prolongation,
        )
    end
    coarse = last(preconditioner.levels)
    preconditioner.coarse_factorization = lu(coarse.matrix)
    return preconditioner
end

function _variable_diffusion_preconditioner(
        specification, operator, matrix, inertia, dt;
        coarse_size, pre_sweeps, post_sweeps, max_levels,
        options, reuse = nothing,
    )
    if specification in (:geometric, :tree_geometric)
        return _tree_variable_diffusion_preconditioner(
            operator, matrix, inertia, dt;
            coarse_size, pre_sweeps, post_sweeps, max_levels, options, reuse,
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

