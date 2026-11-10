using LinearAlgebra: lu!

# Retaining a small dense LU avoids UMFPACK numeric allocations on each shift.
# The size cap bounds persistent quadratic storage; larger systems use sparse LU.
const _STAGGERED_DENSE_COARSE_SIZE = 256

mutable struct DenseStaggeredCoarseLU{F}
    factorization::F
end

function _refactorize_coarse!(
        coarse::DenseStaggeredCoarseLU, matrix::SparseMatrixCSC,
    )
    dense = coarse.factorization.factors
    size(dense) == size(matrix) || throw(
        DimensionMismatch("staggered coarse matrix size changed on fixed topology"),
    )
    fill!(dense, zero(eltype(dense)))
    rows = rowvals(matrix)
    values = nonzeros(matrix)
    for column in axes(matrix, 2), pointer in nzrange(matrix, column)
        @inbounds dense[rows[pointer], column] = values[pointer]
    end
    coarse.factorization = lu!(dense)
    return coarse
end
Base.size(coarse::DenseStaggeredCoarseLU) = size(coarse.factorization)
Base.size(coarse::DenseStaggeredCoarseLU, dim::Integer) =
    size(coarse.factorization, dim)
Base.eltype(coarse::DenseStaggeredCoarseLU) = eltype(coarse.factorization)

ldiv!(destination, coarse::DenseStaggeredCoarseLU, source) =
    ldiv!(destination, coarse.factorization, source)
Base.:\(coarse::DenseStaggeredCoarseLU, source) = coarse.factorization \ source

"""One rediscretized single-phase staggered hierarchy on pruned block trees."""
mutable struct TreeStaggeredMultigridComponent{T, F, L, D, O} <:
    AbstractGeometricMultigridPreconditioner{T}
    diffusion_operators::D
    operators::O
    levels::Vector{L}
    face_prolongations::Vector{SparseMatrixCSC{T, Int}}
    coarse_factorization::F
    coarse_augmented::Bool
    coarse_rhs::Vector{T}
    coarse_solution::Vector{T}
    symmetric_coarse::Bool
    pre_sweeps::Int
    post_sweeps::Int
    pruning::Symbol
    reused_levels::Int
end

_preconditioner_name(::TreeStaggeredMultigridComponent) =
    :tree_staggered_geometric_multigrid

@inline _staggered_component(operator, ::Val{1}) = operator.x
@inline _staggered_component(operator, ::Val{2}) = operator.y
# For conservative rows L = -V^-1 K, with K symmetric positive
# semidefinite by construction. Checking that relation in the same arithmetic
# used to assemble L distinguishes roundoff from genuinely nonsymmetric rows,
# without assuming adjacent dual volumes are equal.
function _staggered_conservative_laplacian(component::StaggeredDiffusionComponent)
    stiffness = component.stiffness
    laplacian = component.laplacian
    issymmetric(stiffness) || return false
    stiffness_rows, stiffness_values = rowvals(stiffness), nonzeros(stiffness)
    laplacian_rows, laplacian_values = rowvals(laplacian), nonzeros(laplacian)
    volumes = component.volumes
    for column in axes(laplacian, 2)
        stiffness_pointer = stiffness.colptr[column]
        stiffness_end = stiffness.colptr[column + 1]
        laplacian_pointer = laplacian.colptr[column]
        laplacian_end = laplacian.colptr[column + 1]
        while stiffness_pointer < stiffness_end || laplacian_pointer < laplacian_end
            stiffness_row = stiffness_pointer < stiffness_end ?
                @inbounds(stiffness_rows[stiffness_pointer]) : typemax(Int)
            laplacian_row = laplacian_pointer < laplacian_end ?
                @inbounds(laplacian_rows[laplacian_pointer]) : typemax(Int)
            row = min(stiffness_row, laplacian_row)
            stiffness_value = stiffness_row == row ?
                @inbounds(stiffness_values[stiffness_pointer]) : zero(eltype(stiffness))
            laplacian_value = laplacian_row == row ?
                @inbounds(laplacian_values[laplacian_pointer]) : zero(eltype(laplacian))
            isfinite(stiffness_value) && isfinite(laplacian_value) || return false
            @inbounds laplacian_value == -inv(volumes[row]) * stiffness_value ||
                return false
            stiffness_pointer += stiffness_row == row
            laplacian_pointer += laplacian_row == row
        end
    end
    return true
end

# Check all shifts, not just the present one: a zero shift can hide
# asymmetry. The conservative relation certifies symmetry despite rounding in
# V*L; otherwise accept only entrywise-exact weighted symmetry.
function _staggered_shift_is_symmetric(component::StaggeredDiffusionComponent)
    volumes = component.volumes
    all(volume -> isfinite(volume) && volume > zero(volume), volumes) || return false
    _staggered_conservative_laplacian(component) && return true
    laplacian = component.laplacian
    rows, values = rowvals(laplacian), nonzeros(laplacian)
    for column in axes(laplacian, 2), pointer in nzrange(laplacian, column)
        row = @inbounds rows[pointer]
        @inbounds isfinite(values[pointer]) || return false
        row == column && continue
        @inbounds weighted = volumes[row] * values[pointer]
        isfinite(weighted) && @inbounds(
            weighted ==
                volumes[column] * laplacian[column, row]
        ) || return false
    end
    return true
end

# Verify that an externally supplied or modified matrix is still the shifted
# matrix represented by its diffusion component. Sparse zeros can disappear at
# zero diffusion or by cancellation, so compare the union of both patterns.
function _staggered_shift_matches(component, diffusion_component, reaction, diffusion)
    matrix, laplacian = component.matrix, diffusion_component.laplacian
    size(matrix) == size(laplacian) || return false
    matrix_rows, matrix_values = rowvals(matrix), nonzeros(matrix)
    laplacian_rows, laplacian_values = rowvals(laplacian), nonzeros(laplacian)
    volumes = diffusion_component.volumes
    T = eltype(volumes)
    for column in axes(matrix, 2)
        matrix_pointer = matrix.colptr[column]
        matrix_end = matrix.colptr[column + 1]
        laplacian_pointer = laplacian.colptr[column]
        laplacian_end = laplacian.colptr[column + 1]
        diagonal_pending = true
        while matrix_pointer < matrix_end || laplacian_pointer < laplacian_end ||
                diagonal_pending
            matrix_row = matrix_pointer < matrix_end ?
                @inbounds(matrix_rows[matrix_pointer]) : typemax(Int)
            laplacian_row = laplacian_pointer < laplacian_end ?
                @inbounds(laplacian_rows[laplacian_pointer]) : typemax(Int)
            row = min(matrix_row, laplacian_row, diagonal_pending ? column : typemax(Int))
            actual = matrix_row == row ?
                @inbounds(matrix_values[matrix_pointer]) : zero(T)
            expected = row == column ? reaction : zero(T)
            if laplacian_row == row
                @inbounds expected -= diffusion * laplacian_values[laplacian_pointer]
                laplacian_pointer += 1
            end
            @inbounds expected *= volumes[row]
            isfinite(expected) && actual == expected || return false
            matrix_pointer += matrix_row == row
            diagonal_pending &= row != column
        end
    end
    return true
end

function _staggered_has_shift_pattern(matrix, laplacian)
    rows = rowvals(matrix)
    laplacian_rows = rowvals(laplacian)
    for column in axes(matrix, 2)
        pointer = matrix.colptr[column]
        last_pointer = matrix.colptr[column + 1] - 1
        diagonal_found = false
        for laplacian_pointer in nzrange(laplacian, column)
            row = @inbounds laplacian_rows[laplacian_pointer]
            while pointer <= last_pointer && @inbounds(rows[pointer] < row)
                @inbounds diagonal_found |= rows[pointer] == column
                pointer += 1
            end
            (pointer <= last_pointer && @inbounds(rows[pointer] == row)) ||
                return false
            diagonal_found |= row == column
            pointer += 1
        end
        while pointer <= last_pointer
            @inbounds diagonal_found |= rows[pointer] == column
            pointer += 1
        end
        diagonal_found || return false
    end
    return true
end

# A zero initial shift, underflow, or cancellation can omit entries from the
# fresh matrix. Expand to the union of the operator and diagonal once, keeping
# explicit zeros thereafter so later shifts retain all matrix storage.
function _staggered_shift_pattern(laplacian::SparseMatrixCSC{T, Int}) where {T}
    laplacian_rows = rowvals(laplacian)
    n = size(laplacian, 2)
    columns = Vector{Int}(undef, n + 1)
    rows = Vector{Int}(undef, nnz(laplacian) + n)
    values = Vector{T}(undef, length(rows))
    next = 1
    for column in 1:n
        columns[column] = next
        diagonal_found = false
        for pointer in nzrange(laplacian, column)
            row = @inbounds laplacian_rows[pointer]
            if !diagonal_found && row > column
                @inbounds rows[next] = column
                next += 1
                diagonal_found = true
            end
            @inbounds rows[next] = row
            next += 1
            diagonal_found |= row == column
        end
        if !diagonal_found
            @inbounds rows[next] = column
            next += 1
        end
    end
    columns[n + 1] = next
    resize!(rows, next - 1)
    resize!(values, next - 1)
    return SparseMatrixCSC(n, n, columns, rows, values)
end

function _refresh_staggered_component!(
        component::StaggeredEllipticComponent{Axis, T},
        diffusion_component::StaggeredDiffusionComponent{Axis, T},
        reaction::T, diffusion::T; symmetric_coarse = false,
    ) where {Axis, T}
    laplacian = symmetric_coarse ?
        diffusion_component.stiffness : diffusion_component.laplacian
    matrix = component.matrix
    if !_staggered_has_shift_pattern(matrix, laplacian)
        matrix = _staggered_shift_pattern(laplacian)
        component = StaggeredEllipticComponent{Axis, T}(
            matrix, diffusion_component.volumes,
            diffusion_component.local_to_global,
        )
    end
    rows, values = rowvals(matrix), nonzeros(matrix)
    laplacian_rows, laplacian_values = rowvals(laplacian), nonzeros(laplacian)
    volumes = diffusion_component.volumes
    for column in axes(matrix, 2)
        laplacian_pointer = laplacian.colptr[column]
        laplacian_end = laplacian.colptr[column + 1] - 1
        for pointer in nzrange(matrix, column)
            row = @inbounds rows[pointer]
            while laplacian_pointer <= laplacian_end &&
                    @inbounds(laplacian_rows[laplacian_pointer] < row)
                laplacian_pointer += 1
            end
            has_laplacian_entry = laplacian_pointer <= laplacian_end &&
                @inbounds(laplacian_rows[laplacian_pointer] == row)
            if symmetric_coarse
                # Match diffusion * stiffness + reaction * mass.
                value = has_laplacian_entry ?
                    @inbounds(diffusion * laplacian_values[laplacian_pointer]) : zero(T)
                row == column && (@inbounds value += reaction * volumes[row])
                @inbounds values[pointer] = value
            else
                # Match mass * (reaction * I - diffusion * laplacian).
                value = row == column ? reaction : zero(T)
                if has_laplacian_entry
                    @inbounds value -= diffusion * laplacian_values[laplacian_pointer]
                end
                @inbounds values[pointer] = volumes[row] * value
            end
            laplacian_pointer += has_laplacian_entry
        end
    end
    return component
end


function _tree_staggered_component_prolongation(
        fine, coarse, prolongation, ::Val{Axis}, ::Type{T},
    ) where {Axis, T}
    fine_indices = _staggered_component(fine, Val(Axis)).local_to_global
    coarse_indices = _staggered_component(coarse, Val(Axis)).local_to_global
    return sparse(prolongation[fine_indices, coarse_indices])
end

function _tree_staggered_component(
        diffusion_operators, elliptic_operators, prolongations,
        ::Val{Axis}; pre_sweeps, post_sweeps, pruning, symmetric_coarse,
        reuse = nothing, reused_new_start = 0, reused_old_start = 0,
    ) where {Axis}
    components = [
        _staggered_component(operator, Val(Axis))
            for operator in elliptic_operators
    ]
    T = eltype(first(components).volumes)
    levels = _multigrid_levels(T, false)
    for index in eachindex(components)
        component = components[index]
        prolongation = index < length(components) ?
            _tree_staggered_component_prolongation(
                diffusion_operators[index], diffusion_operators[index + 1],
                prolongations[index], Val(Axis), T,
            ) : spzeros(T, length(component.volumes), 0)
        if !iszero(reused_new_start) && index >= reused_new_start
            old_index = reused_old_start + index - reused_new_start
            level = reuse.levels[old_index]
            _refresh_multigrid_level!(
                level, component.matrix, component.volumes, prolongation,
            )
            push!(levels, level)
        else
            push!(
                levels,
                _multigrid_level(
                    component.matrix, component.volumes, prolongation,
                ),
            )
        end
    end
    coarse = last(levels)
    if iszero(reused_new_start)
        factorization = size(coarse.matrix, 1) <= _STAGGERED_DENSE_COARSE_SIZE ?
            DenseStaggeredCoarseLU(lu!(Matrix(coarse.matrix))) :
            lu(coarse.matrix)
        coarse_rhs = zeros(T, length(coarse.volumes))
        coarse_solution = similar(coarse_rhs)
    else
        factorization = reuse.coarse_factorization
        if reuse.symmetric_coarse != symmetric_coarse
            factorization = _refactorize_coarse!(factorization, coarse.matrix)
        end
        coarse_rhs = reuse.coarse_rhs
        coarse_solution = reuse.coarse_solution
    end
    return TreeStaggeredMultigridComponent(
        diffusion_operators, components, levels, prolongations,
        factorization, false,
        coarse_rhs, coarse_solution, symmetric_coarse, pre_sweeps, post_sweeps, pruning,
        iszero(reused_new_start) ? 0 :
            length(components) - reused_new_start + 1,
    )
end

"""
    TreeStaggeredGeometricMultigridPreconditioner(operator; kwargs...)

Build the two independent single-phase viscous V-cycles by successively
pruning the block tree, then agglomerating 2×2 root blocks where possible, and
rediscretizing the staggered Laplacian on every resulting mesh. For a
volume-symmetric fine operator, coarse levels use the symmetric conservative
stiffness, so equal-sweep V-cycles remain symmetric. No `P' * A * P`
coarse matrix is formed. Bottom component
systems with at most 256 unknowns use retained dense LU; larger systems use
sparse LU. Configure the hierarchy with `multigrid=MultigridOptions(...)` or
individual keywords.
"""
struct TreeStaggeredGeometricMultigridPreconditioner{T, O, X, Y}
    operator::O
    x::X
    y::Y
end

TreeStaggeredGeometricMultigridPreconditioner(
    operator::StaggeredEllipticOperator{T}, x, y,
) where {T} = TreeStaggeredGeometricMultigridPreconditioner{
    T, typeof(operator), typeof(x), typeof(y),
}(operator, x, y)

# For a symmetric fine system, use the symmetric energy stiffness on every
# coarse level so that the V-cycle remains suitable for CG.
function _staggered_coarse_operator(
        diffusion_operator::StaggeredDiffusionOperator{T}, reaction, diffusion,
        symmetric_coarse::Bool,
    ) where {T}
    symmetric_coarse || return StaggeredEllipticOperator(
        diffusion_operator; reaction, diffusion,
    )
    make_component(component::StaggeredDiffusionComponent{Axis, T}) where {Axis} =
        StaggeredEllipticComponent{Axis, T}(
        _staggered_stable_matrix(component, reaction, diffusion),
        component.volumes, component.local_to_global,
    )
    x = make_component(diffusion_operator.x)
    y = make_component(diffusion_operator.y)
    return StaggeredEllipticOperator{
        T, typeof(diffusion_operator.mesh), typeof(x), typeof(y),
        typeof(diffusion_operator),
    }(diffusion_operator.mesh, x, y, diffusion_operator, reaction, diffusion)
end

function TreeStaggeredGeometricMultigridPreconditioner(
        operator::StaggeredEllipticOperator{T};
        multigrid::MultigridOptions = MultigridOptions(),
        coarse_size = multigrid.coarse_size,
        pre_sweeps = multigrid.pre_sweeps, post_sweeps = multigrid.post_sweeps,
        max_levels = multigrid.max_levels, pruning = :all, reuse = nothing,
    ) where {T}
    multigrid = MultigridOptions(; coarse_size, pre_sweeps, post_sweeps, max_levels)
    (; coarse_size, pre_sweeps, post_sweeps, max_levels) = multigrid
    symmetric_coarse = _staggered_shift_is_symmetric(operator.diffusion_operator.x) &&
        _staggered_shift_is_symmetric(operator.diffusion_operator.y)
    plan = _TreeCoarseningPlan(
        operator.mesh, pruning, coarse_size, max_levels, Val(:staggered),
    )
    isnothing(reuse) ||
        reuse isa TreeStaggeredGeometricMultigridPreconditioner{T} || throw(
        ArgumentError("reused staggered hierarchy has the wrong type"),
    )
    can_reuse = _tree_can_reuse(plan, isnothing(reuse) ? nothing : reuse.x) &&
        typeof(reuse.operator.diffusion_operator.boundary) ===
        typeof(operator.diffusion_operator.boundary) &&
        reuse.operator.reaction == operator.reaction &&
        reuse.operator.diffusion == operator.diffusion

    diffusion_operators = [operator.diffusion_operator]
    elliptic_operators = [operator]
    prolongations = SparseMatrixCSC{T, Int}[]
    current_mesh = operator.mesh
    reuse_operators = can_reuse ? reuse.x.diffusion_operators : nothing
    _tree_reuse_suffix!(plan, reuse_operators, current_mesh, 0)
    while true
        coarse_mesh = _tree_next_coarse_mesh!(
            plan, operator.mesh, current_mesh,
            maximum(
                (
                    length(last(elliptic_operators).x.volumes),
                    length(last(elliptic_operators).y.volumes),
                )
            ),
            length(elliptic_operators),
        )
        isnothing(coarse_mesh) && break
        push!(
            prolongations,
            _tree_face_prolongation(current_mesh, coarse_mesh, T),
        )
        matching_level = _tree_reuse_suffix!(
            plan, reuse_operators, coarse_mesh, length(diffusion_operators),
        )
        can_append_suffix = !iszero(matching_level)
        coarse_diffusion = can_append_suffix ?
            reuse.x.diffusion_operators[matching_level] :
            StaggeredDiffusionOperator(
                coarse_mesh;
                viscosity = operator.diffusion_operator.viscosity,
                boundary = operator.diffusion_operator.boundary, T,
            )
        push!(diffusion_operators, coarse_diffusion)
        push!(
            elliptic_operators,
            _staggered_coarse_operator(
                coarse_diffusion, operator.reaction, operator.diffusion,
                symmetric_coarse,
            ),
        )
        current_mesh = coarse_mesh
    end
    reused_new_start = plan.reused_new_start
    reused_old_start = plan.reused_old_start
    if !iszero(reused_new_start)
        for old_index in (reused_old_start + 1):length(
                reuse.x.diffusion_operators
            )
            old_diffusion = reuse.x.diffusion_operators[old_index]
            push!(diffusion_operators, old_diffusion)
            push!(
                elliptic_operators,
                _staggered_coarse_operator(
                    old_diffusion, operator.reaction, operator.diffusion,
                    symmetric_coarse,
                ),
            )
        end
        for old_index in reused_old_start:length(reuse.x.face_prolongations)
            push!(prolongations, reuse.x.face_prolongations[old_index])
        end
    end
    x = _tree_staggered_component(
        diffusion_operators, elliptic_operators, prolongations, Val(1);
        pre_sweeps, post_sweeps, pruning, symmetric_coarse,
        reuse = isnothing(reuse) ? nothing : reuse.x,
        reused_new_start, reused_old_start,
    )
    y = _tree_staggered_component(
        diffusion_operators, elliptic_operators, prolongations, Val(2);
        pre_sweeps, post_sweeps, pruning, symmetric_coarse,
        reuse = isnothing(reuse) ? nothing : reuse.y,
        reused_new_start, reused_old_start,
    )
    return TreeStaggeredGeometricMultigridPreconditioner{
        T, typeof(operator), typeof(x), typeof(y),
    }(operator, x, y)
end

function _construct_staggered_preconditioners(
        preconditioner, operator::StaggeredEllipticOperator;
        multigrid::MultigridOptions, options = (;),
        reuse = nothing,
    )
    if preconditioner in (:geometric, :tree_geometric)
        hierarchy = TreeStaggeredGeometricMultigridPreconditioner(
            operator; multigrid, options..., reuse,
        )
        return hierarchy.x, hierarchy.y
    elseif preconditioner isa NamedTuple
        keys(preconditioner) == (:x, :y) || throw(
            ArgumentError(
                "a staggered preconditioner tuple must have x and y entries"
            )
        )
        isempty(options) || throw(
            ArgumentError(
                "preconditioner_options cannot accompany pre-built preconditioners"
            )
        )
        return preconditioner.x, preconditioner.y
    end
    isnothing(reuse) || throw(
        ArgumentError("hierarchy reuse requires a tree-geometric preconditioner"),
    )
    if preconditioner isa Symbol
        x_matrix = _staggered_stable_matrix(
            operator.diffusion_operator.x,
            operator.reaction, operator.diffusion
        )
        y_matrix = _staggered_stable_matrix(
            operator.diffusion_operator.y,
            operator.reaction, operator.diffusion
        )
        x = _construct_matrix_preconditioner(
            preconditioner, x_matrix;
            symmetric = true, singular = false, options
        )
        y = _construct_matrix_preconditioner(
            preconditioner, y_matrix;
            symmetric = true, singular = false, options
        )
        return x, y
    end
    x = _construct_matrix_preconditioner(
        preconditioner, operator.x.matrix;
        symmetric = false, singular = false, options
    )
    y = _construct_matrix_preconditioner(
        preconditioner, operator.y.matrix;
        symmetric = false, singular = false, options
    )
    return x, y
end

"""
    StaggeredEllipticSolver(operator;
        krylov=nothing,
        krylov_options=(;),
        preconditioner=:tree_geometric, kwargs...)

Couple both staggered component matrices to external Krylov.jl workspaces and
independent preconditioners. Automatic selection uses PCG when symmetry is
guaranteed for subsequent diffusion shifts and BiCGSTAB otherwise. `krylov`
may explicitly supply a workspace factory accepting `(matrix, rhs)`.
`krylov_options` forwards method settings while the solve controls convergence
and preconditioner placement.
The solver owns its component matrices; numerical timestep updates do not
mutate the operator supplied by the caller.
"""
mutable struct StaggeredEllipticSolver{T, O, X, Y, K, KO, P, Q}
    operator::O
    x::X
    y::Y
    krylov_factory::K
    krylov_options::KO
    preconditioner_specification::P
    preconditioner_options::Q
    coarse_size::Int
    pre_sweeps::Int
    post_sweeps::Int
    max_levels::Int
end

# A user factory may return different concrete preconditioner types for
# different shifts. Widen only those fields; the default tree solver keeps its
# fully concrete, allocation-free solve path.
function _widen_staggered_preconditioner(
        solver::EllipticSolver{T, O, A, K, P, Q},
    ) where {T, O, A, K, P, Q}
    return EllipticSolver{T, O, A, K, Any, Q}(
        solver.operator, solver.action, solver.krylov, solver.preconditioner,
        solver.krylov_options, solver.solution, solver.right_hand_side,
        solver.residual, solver.product, solver.components, solver.stats,
    )
end

function _owned_staggered_component(
        component::StaggeredEllipticComponent{Axis, T},
    ) where {Axis, T}
    return StaggeredEllipticComponent{Axis, T}(
        copy(component.matrix), component.volumes, component.local_to_global,
    )
end

function _owned_staggered_operator(operator::StaggeredEllipticOperator)
    return typeof(operator)(
        operator.mesh, _owned_staggered_component(operator.x),
        _owned_staggered_component(operator.y), operator.diffusion_operator,
        operator.reaction, operator.diffusion,
    )
end

function StaggeredEllipticSolver(
        operator::StaggeredEllipticOperator{T};
        krylov = nothing,
        krylov_options = (;),
        preconditioner = :tree_geometric, preconditioner_options = (;),
        multigrid::MultigridOptions = MultigridOptions(),
        coarse_size = multigrid.coarse_size,
        pre_sweeps = multigrid.pre_sweeps, post_sweeps = multigrid.post_sweeps,
        max_levels = multigrid.max_levels, reuse_preconditioner = nothing,
    ) where {T}
    multigrid = MultigridOptions(; coarse_size, pre_sweeps, post_sweeps, max_levels)
    (; coarse_size, pre_sweeps, post_sweeps, max_levels) = multigrid
    operator = _owned_staggered_operator(operator)
    selected_krylov = if isnothing(krylov)
        diffusion_operator = operator.diffusion_operator
        symmetric = _staggered_shift_is_symmetric(diffusion_operator.x) &&
            _staggered_shift_is_symmetric(diffusion_operator.y) &&
            _staggered_shift_matches(
            operator.x, diffusion_operator.x, operator.reaction, operator.diffusion,
        ) && _staggered_shift_matches(
            operator.y, diffusion_operator.y, operator.reaction, operator.diffusion,
        )
        symmetric ? Krylov.CgWorkspace : Krylov.BicgstabWorkspace
    else
        krylov
    end
    x_preconditioner, y_preconditioner =
        _construct_staggered_preconditioners(
        preconditioner, operator; multigrid, options = preconditioner_options,
        reuse = reuse_preconditioner,
    )
    x_rhs = zeros(T, length(operator.x.volumes))
    y_rhs = zeros(T, length(operator.y.volumes))
    x_krylov = _construct_krylov(
        selected_krylov, operator.x.matrix, x_rhs,
    )
    y_krylov = _construct_krylov(
        selected_krylov, operator.y.matrix, y_rhs,
    )
    x = EllipticSolver(
        operator.x, x_krylov, x_preconditioner; krylov_options
    )
    y = EllipticSolver(
        operator.y, y_krylov, y_preconditioner; krylov_options
    )
    if !(preconditioner isa Union{Nothing, Symbol, NamedTuple}) &&
            applicable(preconditioner, operator.x.matrix)
        x = _widen_staggered_preconditioner(x)
        y = _widen_staggered_preconditioner(y)
    end
    return StaggeredEllipticSolver{
        T, typeof(operator), typeof(x), typeof(y), typeof(selected_krylov),
        typeof(krylov_options),
        typeof(preconditioner), typeof(preconditioner_options),
    }(
        operator, x, y, selected_krylov, krylov_options,
        preconditioner, preconditioner_options,
        coarse_size, pre_sweeps, post_sweeps, max_levels
    )
end

function _refresh_tree_staggered_component!(
        preconditioner::TreeStaggeredMultigridComponent{T},
        operator::StaggeredEllipticOperator{T}, ::Val{Axis},
    ) where {T, Axis}
    preconditioner.diffusion_operators[1] = operator.diffusion_operator
    preconditioner.operators[1] = _staggered_component(operator, Val(Axis))
    for index in 2:length(preconditioner.operators)
        diffusion_component = _staggered_component(
            preconditioner.diffusion_operators[index], Val(Axis),
        )
        preconditioner.operators[index] = _refresh_staggered_component!(
            preconditioner.operators[index], diffusion_component,
            operator.reaction, operator.diffusion;
            symmetric_coarse = preconditioner.symmetric_coarse,
        )
    end
    for index in eachindex(preconditioner.levels)
        level = preconditioner.levels[index]
        component = preconditioner.operators[index]
        _refresh_multigrid_level!(
            level, component.matrix, component.volumes,
            level.prolongation,
        )
    end
    coarse = last(preconditioner.levels)
    preconditioner.coarse_factorization = _refactorize_coarse!(
        preconditioner.coarse_factorization, coarse.matrix,
    )
    return preconditioner
end


function _update_staggered_elliptic!(
        solver::StaggeredEllipticSolver{T}, diffusion
    ) where {T}
    value = T(diffusion)
    value == solver.operator.diffusion && return solver
    value >= zero(T) ||
        throw(ArgumentError("staggered elliptic diffusion must be nonnegative"))
    old = solver.operator
    x = _refresh_staggered_component!(
        old.x, old.diffusion_operator.x, old.reaction, value,
    )
    y = _refresh_staggered_component!(
        old.y, old.diffusion_operator.y, old.reaction, value,
    )
    operator = typeof(old)(
        old.mesh, x, y, old.diffusion_operator, old.reaction, value,
    )
    if solver.preconditioner_specification in (:geometric, :tree_geometric) &&
            solver.x.preconditioner isa TreeStaggeredMultigridComponent &&
            solver.y.preconditioner isa TreeStaggeredMultigridComponent
        _refresh_tree_staggered_component!(
            solver.x.preconditioner, operator, Val(1),
        )
        _refresh_tree_staggered_component!(
            solver.y.preconditioner, operator, Val(2),
        )
    else
        x_preconditioner, y_preconditioner =
            _construct_staggered_preconditioners(
            solver.preconditioner_specification, operator;
            multigrid = _multigrid_options(solver),
            options = solver.preconditioner_options,
        )
        solver.x.preconditioner = x_preconditioner
        solver.y.preconditioner = y_preconditioner
    end
    solver.operator = operator
    solver.x.operator = x
    solver.x.action = x.matrix
    solver.y.operator = y
    solver.y.action = y.matrix
    return solver
end

function _solve_staggered_component!(
        destination, right_hand_side,
        component::StaggeredEllipticComponent, solver::EllipticSolver;
        tolerance, max_cycles
    )
    T = eltype(component.volumes)
    dimensionless_tolerance = _dimensionless_tolerance(T, tolerance)
    max_cycles >= 0 ||
        throw(ArgumentError("max_cycles must be nonnegative"))

    prepare! = () -> begin
        for local_index in eachindex(component.local_to_global)
            global_index = component.local_to_global[local_index]
            @inbounds solver.solution[local_index] = destination[global_index]
            @inbounds solver.right_hand_side[local_index] =
                component.volumes[local_index] * right_hand_side[global_index]
        end
        return zero(T)
    end

    return _solve_krylov_with_stats!(
        prepare!, solver, component.matrix, component.volumes,
        dimensionless_tolerance, max_cycles; relative_to_rhs = true,
    ) do values
        for local_index in eachindex(component.local_to_global)
            global_index = component.local_to_global[local_index]
            @inbounds destination[global_index] = values[local_index]
        end
    end
end

"""
    solve!(solution, solver, rhs; tolerance=sqrt(eps(T)), max_cycles=100)

Solve both positive-reaction staggered elliptic components with the external
Krylov workspaces and independent preconditioners stored in `solver`.
`tolerance` bounds the strong residual relative to the strong right-hand-side
velocity scale, independently of the initial guess.
"""
function solve!(
        solution::FaceVelocity{T},
        solver::StaggeredEllipticSolver{T},
        right_hand_side::FaceVelocity{T};
        tolerance = sqrt(eps(T)), max_cycles = 100
    ) where {T}
    solution.mesh === right_hand_side.mesh === solver.operator.mesh ||
        throw(
        ArgumentError(
            "staggered elliptic arguments must share a mesh"
        )
    )
    solution === right_hand_side &&
        throw(
        ArgumentError(
            "staggered elliptic solution must not alias its right-hand side"
        )
    )
    for global_index in solver.operator.diffusion_operator.x.dirichlet_faces
        @inbounds solution.interior[global_index] = zero(T)
    end
    for global_index in solver.operator.diffusion_operator.y.dirichlet_faces
        @inbounds solution.interior[global_index] = zero(T)
    end
    x_stats = _solve_staggered_component!(
        solution.interior, right_hand_side.interior, solver.operator.x,
        solver.x; tolerance, max_cycles
    )
    y_stats = _solve_staggered_component!(
        solution.interior, right_hand_side.interior, solver.operator.y,
        solver.y; tolerance, max_cycles
    )
    fill!(solution.boundary, zero(T))
    return (
        x = x_stats, y = y_stats,
        converged = x_stats.converged && y_stats.converged,
    )
end

"""Return component-wise `(L2, LInf)` residuals from the latest solve."""
residual_norms(solver::StaggeredEllipticSolver) =
    (
    x = (solver.x.stats.final_l2, solver.x.stats.final_linf),
    y = (solver.y.stats.final_l2, solver.y.stats.final_linf),
)

function _add_wall_source!(
        destination, component, diffusion_operator,
        scale, time, axis
    )
    boundary = diffusion_operator.boundary
    T = eltype(destination)
    for connection in component.wall_connections
        _include_wall_connection(boundary, connection) || continue
        wall_value = _wall_value(boundary, connection, time, axis, T)
        global_index = component.local_to_global[connection.degree_of_freedom]
        @inbounds destination[global_index] += scale *
            connection.conductance * wall_value /
            component.volumes[connection.degree_of_freedom]
    end
    return destination
end

"""
    implicit_midpoint_diffusion!(destination, convected, old_velocity,
                                 diffusion_operator, dt, solver, work;
                                 time=0, tolerance=sqrt(eps(T)),
                                 max_cycles=100)

Apply the Crank--Nicolson/implicit-midpoint viscous substep

`(I - dt*nu*L/2) u_new = u_convected + dt*nu*L*u_old/2 + dt*nu*b_wall`.

`work` is a `FaceVelocity` scratch field. Every actual coefficient change
refreshes the matrices at the requested value on the existing topology while
retaining Krylov workspaces. A missing shift pattern (for example, an initially
zero coefficient) is expanded once; coarse LU reuses its symbolic factorization
when the sparsity matches and falls back to a fresh factorization otherwise.
"""
function implicit_midpoint_diffusion!(
        destination::FaceVelocity{T}, convected::FaceVelocity{T},
        old_velocity::FaceVelocity{T},
        diffusion_operator::StaggeredDiffusionOperator{T}, dt,
        solver::StaggeredEllipticSolver{T},
        work::FaceVelocity{T}; time = zero(T),
        tolerance = sqrt(eps(T)),
        max_cycles = 100
    ) where {T}
    destination.mesh === convected.mesh === old_velocity.mesh ===
        diffusion_operator.mesh === work.mesh ||
        throw(
        ArgumentError(
            "implicit diffusion arguments must share a mesh"
        )
    )
    dt > zero(T) || throw(ArgumentError("dt must be strictly positive"))
    midpoint = T(time) + T(dt) / 2
    half_strength = T(dt) * diffusion_operator.viscosity / 2
    _update_staggered_elliptic!(solver, half_strength)

    diffusion!(work, old_velocity, diffusion_operator; time = midpoint)
    for index in eachindex(work.interior)
        @inbounds work.interior[index] =
            convected.interior[index] + T(dt) * work.interior[index] / 2
    end
    # `diffusion!` included half of the midpoint wall source through the old
    # state. Add the other half so the affine boundary term is centred in time.
    _add_wall_source!(
        work.interior, diffusion_operator.x,
        diffusion_operator, half_strength, midpoint, 1
    )
    _add_wall_source!(
        work.interior, diffusion_operator.y,
        diffusion_operator, half_strength, midpoint, 2
    )
    fill!(work.boundary, zero(T))
    stats = solve!(destination, solver, work; tolerance, max_cycles)
    stats.converged ||
        throw(
        ErrorException(
            _solver_failure_message(
                "staggered diffusion solve", stats, tolerance;
                details = "dt=$(dt), time=$(time), " *
                    "max_cycles=$(max_cycles)",
            )
        )
    )
    return destination
end
