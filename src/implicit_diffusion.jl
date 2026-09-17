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
    pre_sweeps::Int
    post_sweeps::Int
    pruning::Symbol
    reused_levels::Int
end

_preconditioner_name(::TreeStaggeredMultigridComponent) =
    :tree_staggered_geometric_multigrid

@inline _staggered_component(operator, ::Val{1}) = operator.x
@inline _staggered_component(operator, ::Val{2}) = operator.y

function _tree_staggered_component_prolongation(
        fine, coarse, prolongation, ::Val{Axis}, ::Type{T},
    ) where {Axis, T}
    fine_indices = _staggered_component(fine, Val(Axis)).local_to_global
    coarse_indices = _staggered_component(coarse, Val(Axis)).local_to_global
    return sparse(prolongation[fine_indices, coarse_indices])
end

function _tree_staggered_component(
        diffusion_operators, elliptic_operators, prolongations,
        ::Val{Axis}; pre_sweeps, post_sweeps, pruning,
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
        factorization = lu(coarse.matrix)
        coarse_rhs = zeros(T, length(coarse.volumes))
        coarse_solution = similar(coarse_rhs)
    else
        factorization = reuse.coarse_factorization
        coarse_rhs = reuse.coarse_rhs
        coarse_solution = reuse.coarse_solution
    end
    return TreeStaggeredMultigridComponent(
        diffusion_operators, components, levels, prolongations,
        factorization, false,
        coarse_rhs, coarse_solution, pre_sweeps, post_sweeps, pruning,
        iszero(reused_new_start) ? 0 :
            length(components) - reused_new_start + 1,
    )
end

"""
    TreeStaggeredGeometricMultigridPreconditioner(operator; kwargs...)

Build the two independent single-phase viscous V-cycles by successively
pruning the block tree and rediscretizing the staggered Laplacian on every
resulting composite mesh. No `P' * A * P` coarse matrix is formed.
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

function TreeStaggeredGeometricMultigridPreconditioner(
        operator::StaggeredEllipticOperator{T}; coarse_size = 64,
        pre_sweeps = 2, post_sweeps = 2, max_levels = 32,
        pruning = :all, reuse = nothing,
    ) where {T}
    coarse_size >= 1 || throw(ArgumentError("coarse_size must be positive"))
    pre_sweeps >= 0 || throw(ArgumentError("pre_sweeps must be nonnegative"))
    post_sweeps >= 0 || throw(ArgumentError("post_sweeps must be nonnegative"))
    max_levels >= 1 || throw(ArgumentError("max_levels must be positive"))
    pruning in (:deepest, :all) || throw(
        ArgumentError("tree pruning must be :deepest or :all"),
    )
    isnothing(reuse) ||
        reuse isa TreeStaggeredGeometricMultigridPreconditioner{T} || throw(
        ArgumentError("reused staggered hierarchy has the wrong type"),
    )
    can_reuse = !isnothing(reuse) &&
        reuse.x.pruning === pruning &&
        typeof(reuse.operator.diffusion_operator.boundary) ===
            typeof(operator.diffusion_operator.boundary) &&
        reuse.operator.reaction == operator.reaction &&
        reuse.operator.diffusion == operator.diffusion

    diffusion_operators = [operator.diffusion_operator]
    elliptic_operators = [operator]
    prolongations = SparseMatrixCSC{T, Int}[]
    current_mesh = operator.mesh
    active_level = maximum(block.key.level for block in current_mesh.blocks)
    reused_new_start = 0
    reused_old_start = 0
    initial_matching_level = can_reuse ? _matching_tree_level(
        reuse.x.diffusion_operators, current_mesh, 1,
    ) : 0
    initial_suffix_length = iszero(initial_matching_level) ? 0 :
        length(reuse.x.diffusion_operators) - initial_matching_level + 1
    if !iszero(initial_matching_level) &&
            initial_suffix_length <= max_levels
        reused_new_start = 1
        reused_old_start = initial_matching_level
        for old_index in (initial_matching_level + 1):length(
                reuse.x.diffusion_operators)
            old_diffusion = reuse.x.diffusion_operators[old_index]
            push!(diffusion_operators, old_diffusion)
            push!(
                elliptic_operators,
                StaggeredEllipticOperator(
                    old_diffusion; reaction = operator.reaction,
                    diffusion = operator.diffusion,
                ),
            )
        end
        append!(
            prolongations,
            reuse.x.face_prolongations[
                initial_matching_level:length(reuse.x.face_prolongations)
            ],
        )
    end
    while iszero(reused_new_start) &&
            maximum((length(last(elliptic_operators).x.volumes),
                   length(last(elliptic_operators).y.volumes))) > coarse_size &&
            active_level > 0 && length(elliptic_operators) < max_levels
        coarse_mesh = if pruning === :deepest
            active_level -= 1
            _capped_tree_mesh(operator.mesh, active_level)
        else
            _all_leaf_pruned_mesh(current_mesh)
        end
        length(coarse_mesh.faces) < length(current_mesh.faces) || break
        push!(
            prolongations,
            _tree_face_prolongation(current_mesh, coarse_mesh, T),
        )
        matching_level = can_reuse ? _matching_tree_level(
            reuse.x.diffusion_operators, coarse_mesh, 1,
        ) : 0
        suffix_length = iszero(matching_level) ? 0 :
            length(reuse.x.diffusion_operators) - matching_level + 1
        can_append_suffix = !iszero(matching_level) &&
            length(diffusion_operators) + suffix_length <= max_levels
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
            StaggeredEllipticOperator(
                coarse_diffusion; reaction = operator.reaction,
                diffusion = operator.diffusion,
            ),
        )
        current_mesh = coarse_mesh
        if can_append_suffix
            reused_new_start = length(diffusion_operators)
            reused_old_start = matching_level
            for old_index in (matching_level + 1):length(
                    reuse.x.diffusion_operators)
                old_diffusion = reuse.x.diffusion_operators[old_index]
                push!(diffusion_operators, old_diffusion)
                push!(
                    elliptic_operators,
                    StaggeredEllipticOperator(
                        old_diffusion; reaction = operator.reaction,
                        diffusion = operator.diffusion,
                    ),
                )
            end
            append!(
                prolongations,
                reuse.x.face_prolongations[
                    matching_level:length(reuse.x.face_prolongations)
                ],
            )
            break
        end
    end
    x = _tree_staggered_component(
        diffusion_operators, elliptic_operators, prolongations, Val(1);
        pre_sweeps, post_sweeps, pruning,
        reuse = isnothing(reuse) ? nothing : reuse.x,
        reused_new_start, reused_old_start,
    )
    y = _tree_staggered_component(
        diffusion_operators, elliptic_operators, prolongations, Val(2);
        pre_sweeps, post_sweeps, pruning,
        reuse = isnothing(reuse) ? nothing : reuse.y,
        reused_new_start, reused_old_start,
    )
    return TreeStaggeredGeometricMultigridPreconditioner{
        T, typeof(operator), typeof(x), typeof(y),
    }(operator, x, y)
end

# Compatibility names now resolve to the tree-rediscretized implementation.
const StaggeredGeometricMultigridPreconditioner =
    TreeStaggeredGeometricMultigridPreconditioner
const StaggeredGeometricMultigridSolver =
    TreeStaggeredGeometricMultigridPreconditioner

function _construct_staggered_preconditioners(
        preconditioner, operator::StaggeredEllipticOperator;
        coarse_size, pre_sweeps, post_sweeps, max_levels, options = (;),
        reuse = nothing,
    )
    if preconditioner in (:geometric, :tree_geometric)
        settings = merge(
            (; coarse_size, pre_sweeps, post_sweeps, max_levels),
            options
        )
        multigrid = TreeStaggeredGeometricMultigridPreconditioner(
            operator; settings..., reuse,
        )
        return multigrid.x, multigrid.y
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
independent preconditioners. Automatic selection uses PCG when both component
matrices are symmetric and BiCGSTAB otherwise. `krylov` may explicitly supply
a workspace factory accepting `(matrix, rhs)`; changed implicit time steps
rebuild the component systems. `krylov_options` forwards method settings while
the solve controls convergence and preconditioner placement.
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

function StaggeredEllipticSolver(
        operator::StaggeredEllipticOperator{T};
        krylov = nothing,
        krylov_options = (;),
        preconditioner = :tree_geometric, preconditioner_options = (;),
        coarse_size = 64, pre_sweeps = 2, post_sweeps = 2,
        max_levels = 32, reuse_preconditioner = nothing,
    ) where {T}
    selected_krylov = if isnothing(krylov)
        issymmetric(operator.x.matrix) && issymmetric(operator.y.matrix) ?
            Krylov.CgWorkspace : Krylov.BicgstabWorkspace
    else
        krylov
    end
    x_preconditioner, y_preconditioner =
        _construct_staggered_preconditioners(
            preconditioner, operator; coarse_size, pre_sweeps, post_sweeps,
            max_levels, options = preconditioner_options,
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
    fine_component = _staggered_component(operator, Val(Axis))
    preconditioner.diffusion_operators[1] = operator.diffusion_operator
    preconditioner.operators[1] = fine_component
    for index in 2:length(preconditioner.operators)
        diffusion_component = _staggered_component(
            preconditioner.diffusion_operators[index], Val(Axis),
        )
        preconditioner.operators[index] = _staggered_elliptic_component(
            diffusion_component, operator.reaction, operator.diffusion,
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
    preconditioner.coarse_factorization = lu(coarse.matrix)
    return preconditioner
end

function _staggered_component_solver(
        operator, krylov_factory, preconditioner, krylov_options,
    )
    right_hand_side = zeros(eltype(operator.volumes), length(operator.volumes))
    krylov = _construct_krylov(
        krylov_factory, operator.matrix, right_hand_side,
    )
    return EllipticSolver(
        operator, krylov, preconditioner; krylov_options,
    )
end

function _update_staggered_elliptic!(
        solver::StaggeredEllipticSolver{T}, diffusion
    ) where {T}
    value = T(diffusion)
    value == solver.operator.diffusion && return solver
    operator = StaggeredEllipticOperator(
        solver.operator.diffusion_operator;
        reaction = solver.operator.reaction, diffusion = value
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
        solver.operator = operator
        solver.x = _staggered_component_solver(
            operator.x, solver.krylov_factory,
            solver.x.preconditioner, solver.krylov_options,
        )
        solver.y = _staggered_component_solver(
            operator.y, solver.krylov_factory,
            solver.y.preconditioner, solver.krylov_options,
        )
        return solver
    end
    replacement = StaggeredEllipticSolver(
        operator; krylov = solver.krylov_factory,
        krylov_options = solver.krylov_options,
        preconditioner = solver.preconditioner_specification,
        preconditioner_options = solver.preconditioner_options,
        coarse_size = solver.coarse_size, pre_sweeps = solver.pre_sweeps,
        post_sweeps = solver.post_sweeps, max_levels = solver.max_levels
    )
    solver.operator = replacement.operator
    solver.x = replacement.x
    solver.y = replacement.y
    return solver
end

function _solve_staggered_component!(
        destination, right_hand_side,
        component::StaggeredEllipticComponent, solver::EllipticSolver;
        rtol, atol, max_cycles
    )
    T = eltype(component.volumes)
    zero(T) <= rtol < one(T) ||
        throw(ArgumentError("rtol must lie in [0, 1)"))
    atol >= zero(T) || throw(ArgumentError("atol must be nonnegative"))
    max_cycles >= 0 ||
        throw(ArgumentError("max_cycles must be nonnegative"))

    for local_index in eachindex(component.local_to_global)
        global_index = component.local_to_global[local_index]
        @inbounds solver.solution[local_index] = destination[global_index]
        @inbounds solver.right_hand_side[local_index] =
            component.volumes[local_index] * right_hand_side[global_index]
    end

    _true_residual!(solver)
    initial_l2, initial_linf = _integrated_residual_norms(
        solver.residual, component.volumes
    )
    roundoff_floor = _roundoff_residual_floor!(
        component.matrix, solver.solution, solver.right_hand_side,
        component.volumes, solver.product
    )
    target = max(
        T(atol), T(rtol) * max(initial_linf, eps(T)),
        roundoff_floor
    )
    iterations = 0
    if initial_linf > target
        iterations = _run_krylov_reliably!(
            solver.krylov, component.matrix, solver.right_hand_side,
            solver.solution, solver.preconditioner, target,
            component.volumes, max_cycles, solver.krylov_options,
            solver.residual, solver.product,
        )
    end
    _true_residual!(solver)
    final_l2, final_linf = _integrated_residual_norms(
        solver.residual, component.volumes
    )
    final_target = max(
        target,
        _roundoff_residual_floor!(
            component.matrix, solver.solution,
            solver.right_hand_side, component.volumes,
            solver.product
        )
    )

    for local_index in eachindex(component.local_to_global)
        global_index = component.local_to_global[local_index]
        @inbounds destination[global_index] = solver.solution[local_index]
    end

    stats = solver.stats
    stats.cycles = iterations
    stats.converged = final_linf <= final_target
    stats.initial_l2 = initial_l2
    stats.final_l2 = final_l2
    stats.final_linf = final_linf
    stats.removed_rhs_mean = zero(T)
    stats.method = _krylov_method(solver.krylov)
    stats.preconditioner = _preconditioner_name(solver.preconditioner)
    return stats
end

"""
    solve!(solution, solver, rhs; rtol=64eps(T), atol=0, max_cycles=100)

Solve both positive-reaction staggered elliptic components with the external
Krylov workspaces and independent preconditioners stored in `solver`.
"""
function solve!(
        solution::FaceVelocity{T},
        solver::StaggeredEllipticSolver{T},
        right_hand_side::FaceVelocity{T};
        rtol = 64eps(T), atol = 0, max_cycles = 100
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
    x_stats = _solve_staggered_component!(
        solution.interior, right_hand_side.interior, solver.operator.x,
        solver.x; rtol, atol, max_cycles
    )
    y_stats = _solve_staggered_component!(
        solution.interior, right_hand_side.interior, solver.operator.y,
        solver.y; rtol, atol, max_cycles
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
                                 time=0, rtol=64eps(T), atol=0,
                                 max_cycles=100)

Apply the Crank--Nicolson/implicit-midpoint viscous substep

`(I - dt*nu*L/2) u_new = u_convected + dt*nu*L*u_old/2 + dt*nu*b_wall`.

`work` is a `FaceVelocity` scratch field. The preconditioner is reused when
`dt` is unchanged and rebuilt only when the shifted elliptic coefficient
changes.
"""
function implicit_midpoint_diffusion!(
        destination::FaceVelocity{T}, convected::FaceVelocity{T},
        old_velocity::FaceVelocity{T},
        diffusion_operator::StaggeredDiffusionOperator{T}, dt,
        solver::StaggeredEllipticSolver{T},
        work::FaceVelocity{T}; time = zero(T), rtol = 64eps(T), atol = 0,
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
    stats = solve!(destination, solver, work; rtol, atol, max_cycles)
    stats.converged ||
        throw(ErrorException("staggered diffusion solve failed to converge"))
    return destination
end
