mutable struct VariableStaggeredDiffusionSolver{
        T, K, KO, P, PO, A, KW, PC,
    }
    krylov_factory::K
    krylov_options::KO
    preconditioner_specification::P
    preconditioner_options::PO
    multigrid::MultigridOptions
    matrix::SparseMatrixCSC{T, Int}
    action::A
    krylov::KW
    preconditioner::PC
    inertia::Vector{T}
    solution::Vector{T}
    right_hand_side::Vector{T}
    residual::Vector{T}
    product::Vector{T}
    stats::LinearSolverStats{T}
end

function _variable_diffusion_matrix(operator, density, dt)
    T = eltype(operator.volumes)
    length(density) == length(operator.volumes) ||
        throw(DimensionMismatch("one density per staggered volume is required"))
    all(rho -> isfinite(rho) && rho > zero(T), density) ||
        throw(ArgumentError("density must be finite and positive"))
    inertia = T.(density) .* operator.volumes
    matrix = copy(operator.stiffness)
    matrix.nzval .*= T(dt) / 2
    rows = rowvals(matrix)
    for column in axes(matrix, 2)
        diagonal = 0
        for pointer in nzrange(matrix, column)
            if rows[pointer] == column
                diagonal = pointer
                break
            end
        end
        iszero(diagonal) && error(
            "variable diffusion stiffness has no diagonal at index $column",
        )
        @inbounds matrix.nzval[diagonal] += inertia[column]
    end
    return matrix, inertia
end

function _refresh_variable_diffusion_matrix_from_inertia!(
        matrix::SparseMatrixCSC{T, Int},
        operator::VariableStaggeredDiffusionOperator{T}, inertia, dt,
    ) where {T}
    length(inertia) == length(operator.volumes) || throw(
        DimensionMismatch("one inertia value per staggered volume is required"),
    )
    matrix.colptr == operator.stiffness.colptr &&
        rowvals(matrix) == rowvals(operator.stiffness) || throw(
        ArgumentError("diffusion matrix and stiffness sparsity differ"),
    )
    matrix_values = nonzeros(matrix)
    stiffness_values = nonzeros(operator.stiffness)
    half_step = T(dt) / 2
    @_parallel for index in 1:length(matrix_values)
        @inbounds matrix_values[index] = half_step * stiffness_values[index]
    end
    rows = rowvals(matrix)
    @_parallel for column in 1:size(matrix, 2)
        diagonal = 0
        for pointer in nzrange(matrix, column)
            if rows[pointer] == column
                diagonal = pointer
                break
            end
        end
        iszero(diagonal) && error(
            "variable diffusion stiffness has no diagonal at index $column",
        )
        @inbounds matrix_values[diagonal] += inertia[column]
    end
    return matrix
end

function _refresh_variable_diffusion_matrix!(
        solver::VariableStaggeredDiffusionSolver{T}, operator,
        density, dt
    ) where {T}
    length(density) == length(operator.volumes) || throw(
        DimensionMismatch("one density per staggered volume is required"),
    )
    all(rho -> isfinite(rho) && rho > zero(T), density) || throw(
        ArgumentError("density must be finite and positive"),
    )
    @_parallel for column in 1:length(solver.inertia)
        @inbounds solver.inertia[column] =
            T(density[column]) * operator.volumes[column]
    end
    return _refresh_variable_diffusion_matrix_from_inertia!(
        solver.matrix, operator, solver.inertia, dt,
    )
end

function VariableStaggeredDiffusionSolver(
        operator::VariableStaggeredDiffusionOperator{T}, density, dt;
        krylov = Krylov.CgWorkspace, krylov_options = (;),
        preconditioner = :tree_geometric, preconditioner_options = (;),
        multigrid::MultigridOptions = MultigridOptions(),
        reuse_preconditioner = nothing,
        reuse_solver = nothing,
    ) where {T}
    matrix, inertia = _variable_diffusion_matrix(operator, density, dt)
    right_hand_side = zeros(T, length(inertia))
    action = FactorizedVariableDiffusionOperator(operator, inertia, dt)
    krylov_workspace = _construct_krylov(
        krylov, action, right_hand_side
    )
    built_preconditioner = _variable_diffusion_preconditioner(
        preconditioner, operator, matrix, inertia, dt;
        multigrid,
        options = preconditioner_options,
        reuse = reuse_preconditioner,
    )
    if !isnothing(reuse_solver)
        reconfigured = _reconfigure_variable_diffusion_solver!(
            reuse_solver, operator, matrix, inertia, dt,
            built_preconditioner,
        )
        isnothing(reconfigured) || return reconfigured
    end
    return VariableStaggeredDiffusionSolver{
        T, typeof(krylov), typeof(krylov_options),
        typeof(preconditioner), typeof(preconditioner_options),
        typeof(action),
        typeof(krylov_workspace), typeof(built_preconditioner),
    }(
        krylov, krylov_options, preconditioner,
        preconditioner_options, multigrid, matrix, action, krylov_workspace,
        built_preconditioner, inertia, zeros(T, length(inertia)),
        right_hand_side, zeros(T, length(inertia)),
        zeros(T, length(inertia)),
        LinearSolverStats(
            T; method = _krylov_method(krylov_workspace),
            preconditioner = _preconditioner_name(built_preconditioner)
        )
    )
end

"""Retarget package-owned diffusion solver storage after an AMR regrid."""
function _reconfigure_variable_diffusion_solver!(
        solver::VariableStaggeredDiffusionSolver{T},
        operator::VariableStaggeredDiffusionOperator{T}, matrix, inertia, dt,
        preconditioner,
    ) where {T}
    typeof(operator) === typeof(solver.action.operator) || return nothing
    typeof(preconditioner) === typeof(solver.preconditioner) || return nothing
    solver.matrix = matrix
    solver.inertia = inertia
    action = solver.action
    action.operator = operator
    action.inertia = inertia
    action.half_step = T(dt) / 2
    resize!(action.stress_work, length(operator.stress_weights))
    fill!(action.stress_work, zero(T))
    number_of_unknowns = length(inertia)
    for values in (
            solver.solution, solver.right_hand_side,
            solver.residual, solver.product,
        )
        resize!(values, number_of_unknowns)
        fill!(values, zero(T))
    end
    krylov = _resize_krylov_workspace!(solver.krylov, number_of_unknowns)
    isnothing(krylov) && (
        krylov = _construct_krylov(
            solver.krylov_factory, action, solver.right_hand_side,
        )
    )
    typeof(krylov) === typeof(solver.krylov) || return nothing
    solver.krylov = krylov
    solver.preconditioner = preconditioner
    solver.stats = LinearSolverStats(
        T; method = _krylov_method(krylov),
        preconditioner = _preconditioner_name(preconditioner),
    )
    return solver
end

function _update_variable_diffusion_solver!(
        solver::VariableStaggeredDiffusionSolver,
        operator, density, dt
    )
    matrix = _refresh_variable_diffusion_matrix!(solver, operator, density, dt)
    solver.action.operator = operator
    solver.action.half_step = eltype(solver.inertia)(dt) / 2
    if solver.preconditioner_specification in (:geometric, :tree_geometric) &&
            solver.preconditioner isa TreeVariableDiffusionPreconditioner
        _refresh_tree_variable_diffusion_preconditioner!(
            solver.preconditioner, operator, matrix, solver.inertia, dt,
        )
    else
        solver.preconditioner = _variable_diffusion_preconditioner(
            solver.preconditioner_specification, operator, matrix,
            solver.inertia, dt;
            multigrid = solver.multigrid,
            options = solver.preconditioner_options
        )
    end
    return solver
end

function _solve_variable_diffusion!(
        destination, solver::VariableStaggeredDiffusionSolver,
        operator; tolerance, max_cycles
    )
    T = eltype(solver.solution)
    dimensionless_tolerance = _dimensionless_tolerance(T, tolerance)
    max_cycles >= 0 ||
        throw(ArgumentError("max_cycles must be nonnegative"))
    _zero_variable_diffusion_constraints!(solver.right_hand_side, operator)
    prepare! = () -> begin
        copyto!(solver.solution, destination.interior)
        _zero_variable_diffusion_constraints!(solver.solution, operator)
        return zero(T)
    end
    # Dividing the integrated momentum residual by staggered inertia gives
    # a velocity defect and matches the single-phase strong residual.
    return _solve_krylov_with_stats!(
        prepare!, solver, solver.action, solver.inertia, dimensionless_tolerance,
        max_cycles; relative_to_rhs = true,
    ) do values
        _zero_variable_diffusion_constraints!(values, operator)
        copyto!(destination.interior, values)
        fill!(destination.boundary, zero(T))
    end
end

"""Implicit midpoint of the coupled stress operator, with an optional positive
staggered density. `old_velocity` supplies the explicit half of the stress;
it defaults to `source` for a standalone diffusion substep. The sparse solve
is rebuilt here when coefficients change."""
function implicit_midpoint_diffusion!(
        destination::FaceVelocity, source::FaceVelocity,
        operator::VariableStaggeredDiffusionOperator, dt;
        old_velocity = source,
        density = ones(eltype(source), length(source.interior))
    )
    destination.mesh === source.mesh === old_velocity.mesh === operator.mesh || throw(ArgumentError("diffusion arguments must share a mesh"))
    isfinite(dt)&&dt > 0 || throw(ArgumentError("dt must be finite and positive"))
    length(density) == length(operator.volumes) || throw(DimensionMismatch("one density per staggered volume is required"))
    all(rho -> isfinite(rho)&&rho > 0, density) || throw(ArgumentError("density must be finite and positive"))
    inertia = density .* operator.volumes
    matrix = spdiagm(0 => inertia) + (dt / 2) * operator.stiffness
    rhs = inertia .* source.interior -
        (dt / 2) * (operator.stiffness * old_velocity.interior)
    _zero_variable_diffusion_constraints!(rhs, operator)
    destination.interior .= matrix \ rhs
    _zero_variable_diffusion_constraints!(destination.interior, operator)
    fill!(destination.boundary, zero(eltype(destination)))
    return destination
end

function implicit_midpoint_diffusion!(
        destination::FaceVelocity{T}, source::FaceVelocity{T},
        operator::VariableStaggeredDiffusionOperator{T}, dt,
        solver::VariableStaggeredDiffusionSolver{T};
        old_velocity = source,
        density = ones(T, length(source.interior)),
        tolerance = sqrt(eps(T)), max_cycles = 500
    ) where {T}
    destination.mesh === source.mesh === old_velocity.mesh ===
        operator.mesh ||
        throw(ArgumentError("diffusion arguments must share a mesh"))
    isfinite(dt) && dt > zero(T) ||
        throw(ArgumentError("dt must be finite and positive"))
    _update_variable_diffusion_solver!(
        solver, operator, density, T(dt)
    )
    _factorized_variable_stiffness_mul!(
        solver.product, operator, old_velocity.interior,
        solver.action.stress_work,
    )
    for index in eachindex(solver.right_hand_side)
        @inbounds solver.right_hand_side[index] =
            solver.inertia[index] * source.interior[index] -
            T(dt) * solver.product[index] / 2
    end
    stats = _solve_variable_diffusion!(
        destination, solver, operator; tolerance, max_cycles
    )
    stats.converged ||
        throw(
        ErrorException(
            _solver_failure_message(
                "variable staggered diffusion solve", stats, tolerance;
                details = "dt=$(dt), max_cycles=$(max_cycles)",
            )
        )
    )
    return destination
end
