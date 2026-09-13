"""
    StaggeredGeometricMultigridSolver(operator; kwargs...)

Geometric/Galerkin multigrid solver for the two component-wise staggered
elliptic systems. The fine matrices retain the widened, mildly nonsymmetric
AMR diffusion stencil. Transfers are constructed geometrically on each dual
face grid and smoothed with the symmetric stable diffusion core. A direct LU
factorization is used only on each coarsest level.
"""
mutable struct StaggeredGeometricMultigridSolver{T,O,X,Y}
    operator::O
    x::X
    y::Y
    coarse_size::Int
    pre_sweeps::Int
    post_sweeps::Int
    max_levels::Int
end

function StaggeredGeometricMultigridSolver(
        operator::StaggeredEllipticOperator{T}; coarse_size=64,
        pre_sweeps=2, post_sweeps=2, max_levels=32) where {T}
    x = _staggered_multigrid_component(
        operator.mesh, operator.diffusion_operator.x, operator.x;
        coarse_size, pre_sweeps, post_sweeps, max_levels)
    y = _staggered_multigrid_component(
        operator.mesh, operator.diffusion_operator.y, operator.y;
        coarse_size, pre_sweeps, post_sweeps, max_levels)
    return StaggeredGeometricMultigridSolver{
        T,typeof(operator),typeof(x),typeof(y)}(
        operator, x, y, coarse_size, pre_sweeps, post_sweeps, max_levels)
end

# Kept as a descriptive compatibility name: it is now a geometric multigrid
# solver, not a fine-grid sparse factorization.
const StaggeredEllipticSolver = StaggeredGeometricMultigridSolver

function _update_staggered_elliptic!(
        solver::StaggeredGeometricMultigridSolver{T}, diffusion) where {T}
    value = T(diffusion)
    value == solver.operator.diffusion && return solver
    operator = StaggeredEllipticOperator(
        solver.operator.diffusion_operator;
        reaction=solver.operator.reaction, diffusion=value)
    solver.operator = operator
    solver.x = _staggered_multigrid_component(
        operator.mesh, operator.diffusion_operator.x, operator.x;
        coarse_size=solver.coarse_size, pre_sweeps=solver.pre_sweeps,
        post_sweeps=solver.post_sweeps, max_levels=solver.max_levels)
    solver.y = _staggered_multigrid_component(
        operator.mesh, operator.diffusion_operator.y, operator.y;
        coarse_size=solver.coarse_size, pre_sweeps=solver.pre_sweeps,
        post_sweeps=solver.post_sweeps, max_levels=solver.max_levels)
    return solver
end

function _solve_staggered_component!(
        destination, right_hand_side, component, solver;
        rtol, atol, max_cycles)
    T = eltype(component.volumes)
    zero(T) <= rtol < one(T) ||
        throw(ArgumentError("rtol must lie in [0, 1)"))
    atol >= zero(T) || throw(ArgumentError("atol must be nonnegative"))
    max_cycles >= 0 ||
        throw(ArgumentError("max_cycles must be nonnegative"))

    fine = first(solver.levels)
    for local_index in eachindex(component.local_to_global)
        global_index = component.local_to_global[local_index]
        @inbounds fine.solution[local_index] = destination[global_index]
        @inbounds fine.right_hand_side[local_index] =
            component.volumes[local_index] * right_hand_side[global_index]
    end

    initial_l2, initial_linf = _residual_norms!(fine)
    target = max(T(atol), T(rtol) * max(initial_linf, eps(T)),
                 _roundoff_residual_floor!(fine))
    cycles = 0
    final_l2, final_linf = initial_l2, initial_linf
    while final_linf > target && cycles < max_cycles
        _vcycle!(solver, 1)
        cycles += 1
        final_l2, final_linf = _residual_norms!(fine)
    end
    for local_index in eachindex(component.local_to_global)
        global_index = component.local_to_global[local_index]
        @inbounds destination[global_index] = fine.solution[local_index]
    end

    stats = solver.stats
    stats.cycles = cycles
    stats.converged = final_linf <= target
    stats.initial_l2 = initial_l2
    stats.final_l2 = final_l2
    stats.final_linf = final_linf
    stats.removed_rhs_mean = zero(T)
    return stats
end

"""
    solve!(solution, solver, rhs; rtol=64eps(T), atol=0, max_cycles=100)

Solve both positive-reaction staggered elliptic components with geometric
multigrid. The returned named tuple contains the component statistics and a
combined `converged` flag.
"""
function solve!(solution::FaceVelocity{T},
                solver::StaggeredGeometricMultigridSolver{T},
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
residual_norms(solver::StaggeredGeometricMultigridSolver) =
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

`work` is a `FaceVelocity` scratch field. The multigrid hierarchy is reused
when `dt` is unchanged and rebuilt only when the shifted elliptic coefficient
changes.
"""
function implicit_midpoint_diffusion!(
        destination::FaceVelocity{T}, convected::FaceVelocity{T},
        old_velocity::FaceVelocity{T},
        diffusion_operator::StaggeredDiffusionOperator{T}, dt,
        solver::StaggeredGeometricMultigridSolver{T},
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
        throw(ErrorException("staggered diffusion multigrid failed to converge"))
    return destination
end
