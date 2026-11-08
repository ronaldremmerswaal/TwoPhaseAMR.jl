function _remove_volume_mean!(field_values, volumes)
    mean_value = dot(field_values, volumes) / sum(volumes)
    @_parallel for index in 1:length(field_values)
        @inbounds volumes[index] > zero(eltype(volumes)) &&
            (field_values[index] -= mean_value)
    end
    return mean_value
end

function _project_integrated_rhs!(right_hand_side, volumes)
    mean_value = sum(right_hand_side) / sum(volumes)
    @_parallel for index in 1:length(right_hand_side)
        @inbounds right_hand_side[index] -= mean_value * volumes[index]
    end
    return mean_value
end

@inline _multigrid_mul!(destination, level::MultigridLevel, source) =
    isnothing(level.matrix_transpose) ?
    mul!(destination, level.matrix, source) :
    _parallel_csr_row_mul!(
        destination, level.matrix_transpose, source,
    )

function _smooth!(level::MultigridLevel, sweeps, singular)
    solution = level.solution
    right_hand_side = level.right_hand_side
    temporary = level.temporary
    inverse_diagonal = level.inverse_diagonal
    relaxation_weight = level.relaxation_weight
    for _ in 1:sweeps
        _multigrid_mul!(temporary, level, solution)
        @_parallel for index in 1:length(solution)
            @inbounds solution[index] += relaxation_weight *
                inverse_diagonal[index] *
                (right_hand_side[index] - temporary[index])
        end
        singular && _remove_volume_mean!(solution, level.volumes)
    end
    return solution
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
    _multigrid_mul!(level.temporary, level, level.solution)
    residual = level.residual
    right_hand_side = level.right_hand_side
    temporary = level.temporary
    @_parallel for index in 1:length(residual)
        @inbounds residual[index] = right_hand_side[index] - temporary[index]
    end

    coarse = solver.levels[level_index + 1]
    _restrict!(coarse.right_hand_side, level.prolongation, level.residual)
    singular && _project_integrated_rhs!(coarse.right_hand_side, coarse.volumes)
    fill!(coarse.solution, zero(eltype(coarse.solution)))
    _vcycle!(solver, level_index + 1)
    _prolongate!(level.temporary, level.prolongation, coarse.solution)
    solution = level.solution
    @_parallel for index in 1:length(solution)
        @inbounds solution[index] += temporary[index]
    end
    _smooth!(level, solver.post_sweeps, singular)
    return level.solution
end


function _roundoff_residual_floor!(
        matrix::SparseMatrixCSC{T}, solution,
        right_hand_side, volumes,
        temporary
    ) where {T}
    fill!(temporary, zero(T))
    rows = rowvals(matrix)
    matrix_values = nonzeros(matrix)
    for column in axes(matrix, 2), pointer in nzrange(matrix, column)
        row = @inbounds rows[pointer]
        @inbounds temporary[row] +=
            abs(matrix_values[pointer]) * abs(solution[column])
    end
    scale = one(T)
    for index in eachindex(temporary)
        @inbounds volumes[index] > zero(T) || continue
        @inbounds scale = max(
            scale,
            temporary[index] / volumes[index],
            abs(right_hand_side[index] / volumes[index])
        )
    end
    return T(128) * eps(T) * scale
end

function _roundoff_residual_floor!(
        action::FactorizedEllipticOperator{T}, solution,
        right_hand_side, volumes, temporary,
    ) where {T}
    operator = action.operator
    face_input = action.face_input
    face_output = action.face_output
    square_root = operator.square_root_face_coefficient
    nx, ny = operator.mesh.block_size
    for (face_index, face) in pairs(operator.mesh.faces)
        minus = _cell_index(face.minus, nx, ny)
        plus = _cell_index(face.plus, nx, ny)
        @inbounds face_input[face_index] = square_root[face_index] * face.area *
            (abs(solution[minus]) + abs(solution[plus]))
    end
    metric = operator.gradient_operator.inverse_metric
    if metric isa DiagonalInverseFaceMetric
        for index in eachindex(face_output)
            @inbounds face_output[index] =
                abs(metric.diagonal[index]) * face_input[index]
        end
    else
        fill!(face_output, zero(T))
        rows = rowvals(metric.matrix)
        values = nonzeros(metric.matrix)
        for column in axes(metric.matrix, 2), pointer in nzrange(metric.matrix, column)
            @inbounds face_output[rows[pointer]] +=
                abs(values[pointer]) * face_input[column]
        end
    end
    for index in eachindex(temporary)
        @inbounds temporary[index] =
            (
            abs(operator.reaction) * volumes[index] +
                operator.constraints[index]
        ) * abs(solution[index])
    end
    for (face_index, face) in pairs(operator.mesh.faces)
        bound = abs(operator.diffusion) * face.area *
            square_root[face_index] * face_output[face_index]
        minus = _cell_index(face.minus, nx, ny)
        plus = _cell_index(face.plus, nx, ny)
        @inbounds temporary[minus] += bound
        @inbounds temporary[plus] += bound
    end
    scale = one(T)
    for index in eachindex(temporary)
        @inbounds volumes[index] > zero(T) || continue
        @inbounds scale = max(
            scale, temporary[index] / volumes[index],
            abs(right_hand_side[index] / volumes[index]),
        )
    end
    return T(128) * eps(T) * scale
end

function _integrated_residual_norms(residual, volumes)
    T = eltype(residual)
    squared = zero(T)
    maximum_norm = zero(T)
    total_volume = zero(T)
    for index in eachindex(residual)
        @inbounds volumes[index] > zero(T) || continue
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
    if applicable(
            ldiv!, destination, preconditioner.value, right_hand_side,
        )
        return ldiv!(destination, preconditioner.value, right_hand_side)
    end
    copyto!(destination, right_hand_side)
    return ldiv!(preconditioner.value, destination)
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

Construct an IncompleteLU.jl Crout ILU preconditioner.
"""
function incomplete_lu_preconditioner(matrix; drop_tolerance = 1.0e-3)
    drop_tolerance >= 0 ||
        throw(ArgumentError("drop_tolerance must be nonnegative"))
    return ExternalPreconditioner(
        :ilu, IncompleteLU.ilu(matrix; τ = drop_tolerance), matrix,
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
    krylov isa Symbol && throw(ArgumentError(
        "krylov must be a workspace or constructor (CG, BiCGSTAB, GMRES, FGMRES); symbols are not supported",
    ))
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

@inline function _resize_optional_krylov_vector!(values, number_of_unknowns)
    isempty(values) || resize!(values, number_of_unknowns)
    return values
end

"""Retarget a Krylov.jl workspace while retaining grow-only vector capacity.

Krylov.jl workspaces are mutable, but their public constructors allocate every
method vector again. The four methods supported by this package all use plain
`Vector` storage, so an AMR dimension change can retain those buffers. Empty
optional vectors stay empty and Krylov.jl will materialise them if a later solve
needs the corresponding feature.
"""
function _resize_krylov_workspace!(
        workspace::Krylov.CgWorkspace, number_of_unknowns::Int,
    )
    workspace.m = workspace.n = number_of_unknowns
    for values in (workspace.x, workspace.r, workspace.p, workspace.Ap)
        resize!(values, number_of_unknowns)
    end
    for values in (workspace.Δx, workspace.npc_dir, workspace.z)
        _resize_optional_krylov_vector!(values, number_of_unknowns)
    end
    workspace.warm_start = false
    return workspace
end

function _resize_krylov_workspace!(
        workspace::Krylov.BicgstabWorkspace, number_of_unknowns::Int,
    )
    workspace.m = workspace.n = number_of_unknowns
    for values in (
            workspace.x, workspace.r, workspace.p, workspace.v,
            workspace.s, workspace.qd,
        )
        resize!(values, number_of_unknowns)
    end
    for values in (workspace.Δx, workspace.yz, workspace.t)
        _resize_optional_krylov_vector!(values, number_of_unknowns)
    end
    workspace.warm_start = false
    return workspace
end


function _resize_krylov_workspace!(
        workspace::Krylov.GmresWorkspace, number_of_unknowns::Int,
    )
    # Krylov caps the basis length by the problem dimension. Reusing a basis
    # that is longer than a newly tiny problem would violate that invariant.
    length(workspace.V) <= number_of_unknowns || return nothing
    workspace.m = workspace.n = number_of_unknowns
    for values in (workspace.x, workspace.w)
        resize!(values, number_of_unknowns)
    end
    for values in (workspace.Δx, workspace.p, workspace.q)
        _resize_optional_krylov_vector!(values, number_of_unknowns)
    end
    for values in workspace.V
        resize!(values, number_of_unknowns)
    end
    workspace.warm_start = false
    workspace.inner_iter = 0
    return workspace
end

function _resize_krylov_workspace!(
        workspace::Krylov.FgmresWorkspace, number_of_unknowns::Int,
    )
    length(workspace.V) <= number_of_unknowns || return nothing
    workspace.m = workspace.n = number_of_unknowns
    for values in (workspace.x, workspace.w)
        resize!(values, number_of_unknowns)
    end
    for values in (workspace.Δx, workspace.q)
        _resize_optional_krylov_vector!(values, number_of_unknowns)
    end
    for basis in (workspace.V, workspace.Z), values in basis
        resize!(values, number_of_unknowns)
    end
    workspace.warm_start = false
    workspace.inner_iter = 0
    return workspace
end

_resize_krylov_workspace!(workspace, number_of_unknowns::Int) = nothing

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
                "preconditioner must be :auto (in NavierStokes options), :geometric, :amg, :ilu, :ichol, " *
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
        symmetric = true, singular = operator.singular, options
    )
end
