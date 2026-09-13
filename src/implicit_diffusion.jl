"""
    StaggeredGeometricMultigridPreconditioner(operator; kwargs...)

Build the two independent geometric/Galerkin V-cycle preconditioners for a
staggered elliptic operator. This object owns no Krylov method or Krylov
workspace; its `x` and `y` components only implement the preconditioner action
through `ldiv!`.
"""
struct StaggeredGeometricMultigridPreconditioner{T,O,X,Y}
    operator::O
    x::X
    y::Y
end

function StaggeredGeometricMultigridPreconditioner(
        operator::StaggeredEllipticOperator{T}; coarse_size=64,
        pre_sweeps=2, post_sweeps=2, max_levels=32,
        max_transfer_entries=3) where {T}
    max_transfer_entries >= 2 || throw(ArgumentError(
        "max_transfer_entries must be at least two"))
    x = _staggered_multigrid_component(
        operator.mesh, operator.diffusion_operator.x, operator.x;
        reaction=operator.reaction, diffusion=operator.diffusion,
        coarse_size, pre_sweeps, post_sweeps, max_levels,
        max_transfer_entries)
    y = _staggered_multigrid_component(
        operator.mesh, operator.diffusion_operator.y, operator.y;
        reaction=operator.reaction, diffusion=operator.diffusion,
        coarse_size, pre_sweeps, post_sweeps, max_levels,
        max_transfer_entries)
    return StaggeredGeometricMultigridPreconditioner{
        T,typeof(operator),typeof(x),typeof(y)}(operator, x, y)
end

# Compatibility name from before Krylov iteration and multigrid were split.
const StaggeredGeometricMultigridSolver =
    StaggeredGeometricMultigridPreconditioner

function _construct_staggered_preconditioners(
        preconditioner, operator::StaggeredEllipticOperator;
        coarse_size, pre_sweeps, post_sweeps, max_levels, options=(;))
    if preconditioner === :geometric
        settings = merge((; coarse_size, pre_sweeps, post_sweeps, max_levels),
                         options)
        multigrid = StaggeredGeometricMultigridPreconditioner(
            operator; settings...)
        return multigrid.x, multigrid.y
    elseif preconditioner isa NamedTuple
        keys(preconditioner) == (:x, :y) || throw(ArgumentError(
            "a staggered preconditioner tuple must have x and y entries"))
        isempty(options) || throw(ArgumentError(
            "preconditioner_options cannot accompany pre-built preconditioners"))
        return preconditioner.x, preconditioner.y
    end
    if preconditioner isa Symbol
        x_matrix = _staggered_stable_matrix(
            operator.diffusion_operator.x,
            operator.reaction, operator.diffusion)
        y_matrix = _staggered_stable_matrix(
            operator.diffusion_operator.y,
            operator.reaction, operator.diffusion)
        x = _construct_matrix_preconditioner(
            preconditioner, x_matrix;
            symmetric=true, singular=false, options)
        y = _construct_matrix_preconditioner(
            preconditioner, y_matrix;
            symmetric=true, singular=false, options)
        return x, y
    end
    x = _construct_matrix_preconditioner(
        preconditioner, operator.x.matrix;
        symmetric=false, singular=false, options)
    y = _construct_matrix_preconditioner(
        preconditioner, operator.y.matrix;
        symmetric=false, singular=false, options)
    return x, y
end

"""
    StaggeredEllipticSolver(operator;
        krylov=Krylov.BicgstabWorkspace,
        preconditioner=:geometric, kwargs...)

Couple both staggered component matrices to external Krylov.jl workspaces and
independent preconditioners. The default remains right-preconditioned
BiCGSTAB with this package's geometric multigrid V-cycle. `krylov` must be a
workspace factory accepting `(matrix, rhs)` because changed implicit time
steps rebuild the component systems.
"""
mutable struct StaggeredEllipticSolver{T,O,X,Y,K,KO,P,Q}
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
        krylov=Krylov.BicgstabWorkspace,
        krylov_options=(;),
        preconditioner=:geometric, preconditioner_options=(;),
        coarse_size=64, pre_sweeps=2, post_sweeps=2,
        max_levels=32) where {T}
    x_preconditioner, y_preconditioner =
        _construct_staggered_preconditioners(
            preconditioner, operator; coarse_size, pre_sweeps, post_sweeps,
            max_levels, options=preconditioner_options)
    x_rhs = zeros(T, length(operator.x.volumes))
    y_rhs = zeros(T, length(operator.y.volumes))
    x_krylov = _construct_krylov(krylov, operator.x.matrix, x_rhs)
    y_krylov = _construct_krylov(krylov, operator.y.matrix, y_rhs)
    x = EllipticSolver(
        operator.x, x_krylov, x_preconditioner; krylov_options)
    y = EllipticSolver(
        operator.y, y_krylov, y_preconditioner; krylov_options)
    return StaggeredEllipticSolver{
        T,typeof(operator),typeof(x),typeof(y),typeof(krylov),
        typeof(krylov_options),
        typeof(preconditioner),typeof(preconditioner_options)}(
        operator, x, y, krylov, krylov_options,
        preconditioner, preconditioner_options,
        coarse_size, pre_sweeps, post_sweeps, max_levels)
end

function _update_staggered_elliptic!(
        solver::StaggeredEllipticSolver{T}, diffusion) where {T}
    value = T(diffusion)
    value == solver.operator.diffusion && return solver
    operator = StaggeredEllipticOperator(
        solver.operator.diffusion_operator;
        reaction=solver.operator.reaction, diffusion=value)
    replacement = StaggeredEllipticSolver(
        operator; krylov=solver.krylov_factory,
        krylov_options=solver.krylov_options,
        preconditioner=solver.preconditioner_specification,
        preconditioner_options=solver.preconditioner_options,
        coarse_size=solver.coarse_size, pre_sweeps=solver.pre_sweeps,
        post_sweeps=solver.post_sweeps, max_levels=solver.max_levels)
    solver.operator = replacement.operator
    solver.x = replacement.x
    solver.y = replacement.y
    return solver
end

function _solve_staggered_component!(
        destination, right_hand_side,
        component::StaggeredEllipticComponent, solver::EllipticSolver;
        rtol, atol, max_cycles)
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
        solver.residual, component.volumes)
    roundoff_floor = _roundoff_residual_floor!(
        component.matrix, solver.solution, solver.right_hand_side,
        component.volumes, solver.product)
    target = max(T(atol), T(rtol) * max(initial_linf, eps(T)),
                 roundoff_floor)
    if initial_linf > target
        _run_krylov!(
            solver.krylov, component.matrix, solver.right_hand_side,
            solver.solution, solver.preconditioner, target,
            component.volumes, max_cycles, solver.krylov_options)
        copyto!(solver.solution, Krylov.solution(solver.krylov))
    end
    _true_residual!(solver)
    final_l2, final_linf = _integrated_residual_norms(
        solver.residual, component.volumes)

    for local_index in eachindex(component.local_to_global)
        global_index = component.local_to_global[local_index]
        @inbounds destination[global_index] = solver.solution[local_index]
    end

    stats = solver.stats
    stats.cycles = initial_linf <= target ? 0 :
                   Krylov.iteration_count(solver.krylov)
    stats.converged = final_linf <= target
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
function solve!(solution::FaceVelocity{T},
                solver::StaggeredEllipticSolver{T},
                right_hand_side::FaceVelocity{T};
                rtol=64eps(T), atol=0, max_cycles=100) where {T}
    solution.mesh === right_hand_side.mesh === solver.operator.mesh ||
        throw(ArgumentError(
            "staggered elliptic arguments must share a mesh"))
    solution === right_hand_side &&
        throw(ArgumentError(
            "staggered elliptic solution must not alias its right-hand side"))
    x_stats = _solve_staggered_component!(
        solution.interior, right_hand_side.interior, solver.operator.x,
        solver.x; rtol, atol, max_cycles)
    y_stats = _solve_staggered_component!(
        solution.interior, right_hand_side.interior, solver.operator.y,
        solver.y; rtol, atol, max_cycles)
    fill!(solution.boundary, zero(T))
    return (x=x_stats, y=y_stats,
            converged=x_stats.converged && y_stats.converged)
end

"""Return component-wise `(L2, LInf)` residuals from the latest solve."""
residual_norms(solver::StaggeredEllipticSolver) =
    (x=(solver.x.stats.final_l2, solver.x.stats.final_linf),
     y=(solver.y.stats.final_l2, solver.y.stats.final_linf))

function _add_wall_source!(destination, component, diffusion_operator,
                           scale, time, axis)
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
        work::FaceVelocity{T}; time=zero(T), rtol=64eps(T), atol=0,
        max_cycles=100) where {T}
    destination.mesh === convected.mesh === old_velocity.mesh ===
        diffusion_operator.mesh === work.mesh ||
        throw(ArgumentError(
            "implicit diffusion arguments must share a mesh"))
    dt > zero(T) || throw(ArgumentError("dt must be strictly positive"))
    midpoint = T(time) + T(dt) / 2
    half_strength = T(dt) * diffusion_operator.viscosity / 2
    _update_staggered_elliptic!(solver, half_strength)

    diffusion!(work, old_velocity, diffusion_operator; time=midpoint)
    for index in eachindex(work.interior)
        @inbounds work.interior[index] =
            convected.interior[index] + T(dt) * work.interior[index] / 2
    end
    # `diffusion!` included half of the midpoint wall source through the old
    # state. Add the other half so the affine boundary term is centred in time.
    _add_wall_source!(work.interior, diffusion_operator.x,
                      diffusion_operator, half_strength, midpoint, 1)
    _add_wall_source!(work.interior, diffusion_operator.y,
                      diffusion_operator, half_strength, midpoint, 2)
    fill!(work.boundary, zero(T))
    stats = solve!(destination, solver, work; rtol, atol, max_cycles)
    stats.converged ||
        throw(ErrorException("staggered diffusion solve failed to converge"))
    return destination
end
