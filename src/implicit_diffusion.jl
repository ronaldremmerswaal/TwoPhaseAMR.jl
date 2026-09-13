"""
    StaggeredGeometricMultigridSolver(operator; method=:auto, kwargs...)

Geometric/Galerkin multigrid solver for the two component-wise staggered
elliptic systems. The fine matrices retain the widened, mildly nonsymmetric
AMR diffusion stencil. Transfers are constructed geometrically on each dual
face grid and smoothed with the symmetric stable diffusion core. A direct LU
factorization is used only on each coarsest level. `method=:multigrid` applies
standalone V-cycles and `method=:bicgstab` uses a V-cycle as a right
preconditioner. `method=:auto` selects BiCGStab for a multilevel system whose
diffusion-to-reaction stiffness is at least 4 and standalone multigrid
otherwise.
"""
mutable struct StaggeredGeometricMultigridSolver{T,O,X,Y}
    operator::O
    x::X
    y::Y
    coarse_size::Int
    pre_sweeps::Int
    post_sweeps::Int
    max_levels::Int
    method::Symbol
end

function StaggeredGeometricMultigridSolver(
        operator::StaggeredEllipticOperator{T}; coarse_size=64,
        pre_sweeps=2, post_sweeps=2, max_levels=32,
        method=:auto) where {T}
    method in (:multigrid, :bicgstab, :auto) ||
        throw(ArgumentError(
            "method must be :multigrid, :bicgstab, or :auto"))
    x = _staggered_multigrid_component(
        operator.mesh, operator.diffusion_operator.x, operator.x;
        coarse_size, pre_sweeps, post_sweeps, max_levels)
    y = _staggered_multigrid_component(
        operator.mesh, operator.diffusion_operator.y, operator.y;
        coarse_size, pre_sweeps, post_sweeps, max_levels)
    return StaggeredGeometricMultigridSolver{
        T,typeof(operator),typeof(x),typeof(y)}(
        operator, x, y, coarse_size, pre_sweeps, post_sweeps, max_levels,
        method)
end

function _true_bicgstab_residual!(workspace::BiCGStabWorkspace, matrix)
    mul!(workspace.direction_product, matrix, workspace.solution)
    for index in eachindex(workspace.residual)
        @inbounds workspace.residual[index] =
            workspace.right_hand_side[index] -
            workspace.direction_product[index]
    end
    return workspace.residual
end

function _restart_bicgstab!(workspace::BiCGStabWorkspace)
    copyto!(workspace.shadow_residual, workspace.residual)
    fill!(workspace.direction, zero(eltype(workspace.direction)))
    fill!(workspace.direction_product,
          zero(eltype(workspace.direction_product)))
    return workspace
end

function _bicgstab_breakdown(value, first, second)
    T = typeof(value)
    scale = sqrt(abs(dot(first, first) * dot(second, second)))
    return abs(value) <= eps(T) * max(scale, floatmin(T))
end

function _bicgstab!(solver::StaggeredMultigridComponent{T}, target,
                    max_iterations) where {T}
    fine = first(solver.levels)
    matrix = fine.matrix
    workspace = solver.krylov
    solution = workspace.solution
    residual = workspace.residual
    shadow = workspace.shadow_residual
    direction = workspace.direction
    preconditioned_direction = workspace.preconditioned_direction
    direction_product = workspace.direction_product
    intermediate = workspace.intermediate_residual
    preconditioned_intermediate = workspace.preconditioned_intermediate
    intermediate_product = workspace.intermediate_product

    copyto!(solution, fine.solution)
    copyto!(workspace.right_hand_side, fine.right_hand_side)
    copyto!(residual, fine.residual)
    _restart_bicgstab!(workspace)
    final_l2, final_linf = _integrated_residual_norms(
        residual, fine.volumes)
    iterations = 0
    previous_rho = one(T)
    alpha = one(T)
    omega = one(T)
    restarted = true

    while final_linf > target && iterations < max_iterations
        rho = dot(shadow, residual)
        _bicgstab_breakdown(rho, shadow, residual) &&
            throw(ErrorException("BiCGStab shadow-residual breakdown"))
        if restarted
            copyto!(direction, residual)
            restarted = false
        else
            beta = (rho / previous_rho) * (alpha / omega)
            for index in eachindex(direction)
                @inbounds direction[index] = residual[index] + beta *
                    (direction[index] - omega * direction_product[index])
            end
        end

        _apply_vcycle_preconditioner!(
            preconditioned_direction, solver, direction)
        mul!(direction_product, matrix, preconditioned_direction)
        denominator = dot(shadow, direction_product)
        _bicgstab_breakdown(denominator, shadow, direction_product) &&
            throw(ErrorException("BiCGStab alpha breakdown"))
        alpha = rho / denominator
        for index in eachindex(intermediate)
            @inbounds intermediate[index] =
                residual[index] - alpha * direction_product[index]
        end
        iterations += 1
        final_l2, final_linf = _integrated_residual_norms(
            intermediate, fine.volumes)
        if final_linf <= target
            for index in eachindex(solution)
                @inbounds solution[index] +=
                    alpha * preconditioned_direction[index]
            end
            _true_bicgstab_residual!(workspace, matrix)
            final_l2, final_linf = _integrated_residual_norms(
                residual, fine.volumes)
            final_linf <= target && break
            _restart_bicgstab!(workspace)
            previous_rho = alpha = omega = one(T)
            restarted = true
            continue
        end

        _apply_vcycle_preconditioner!(
            preconditioned_intermediate, solver, intermediate)
        mul!(intermediate_product, matrix, preconditioned_intermediate)
        product_norm_squared = dot(intermediate_product,
                                   intermediate_product)
        product_norm_squared > floatmin(T) ||
            throw(ErrorException("BiCGStab intermediate-product breakdown"))
        omega_numerator = dot(intermediate_product, intermediate)
        _bicgstab_breakdown(
            omega_numerator, intermediate_product, intermediate) &&
            throw(ErrorException("BiCGStab omega breakdown"))
        omega = omega_numerator / product_norm_squared
        for index in eachindex(solution)
            @inbounds solution[index] +=
                alpha * preconditioned_direction[index] +
                omega * preconditioned_intermediate[index]
            @inbounds residual[index] =
                intermediate[index] - omega * intermediate_product[index]
        end
        final_l2, final_linf = _integrated_residual_norms(
            residual, fine.volumes)
        if final_linf <= target
            _true_bicgstab_residual!(workspace, matrix)
            final_l2, final_linf = _integrated_residual_norms(
                residual, fine.volumes)
            final_linf <= target && break
            _restart_bicgstab!(workspace)
            previous_rho = alpha = omega = one(T)
            restarted = true
            continue
        end
        previous_rho = rho
    end

    _true_bicgstab_residual!(workspace, matrix)
    final_l2, final_linf = _integrated_residual_norms(
        residual, fine.volumes)
    copyto!(fine.solution, solution)
    return iterations, final_l2, final_linf
end

function _staggered_solve_method(solver::StaggeredGeometricMultigridSolver,
                                 method)
    method in (:multigrid, :bicgstab, :auto) ||
        throw(ArgumentError(
            "method must be :multigrid, :bicgstab, or :auto"))
    method === :auto || return method
    multilevel = length(solver.x.levels) > 1 && length(solver.y.levels) > 1
    diffusion_operator = solver.operator.diffusion_operator
    maximum_rate = max(diffusion_operator.x.maximum_row_sum,
                       diffusion_operator.y.maximum_row_sum)
    relative_stiffness = solver.operator.diffusion * maximum_rate /
                         solver.operator.reaction
    # Below this crossover, a standalone V-cycle already converges in very few
    # traversals and the second preconditioner application in a full BiCGStab
    # iteration is not recovered. Graded-mesh timings put the crossover near
    # 1--4; use the conservative end so `:auto` only selects a measured win.
    return multilevel && relative_stiffness >= 4 ? :bicgstab : :multigrid
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
        rtol, atol, max_cycles, method)
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
    if method === :bicgstab
        cycles, final_l2, final_linf = _bicgstab!(
            solver, target, max_cycles)
    else
        cycles = 0
        final_l2, final_linf = initial_l2, initial_linf
        while final_linf > target && cycles < max_cycles
            _vcycle!(solver, 1)
            cycles += 1
            final_l2, final_linf = _residual_norms!(fine)
        end
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
    stats.method = method
    return stats
end

"""
    solve!(solution, solver, rhs; rtol=64eps(T), atol=0, max_cycles=100,
           method=solver.method)

Solve both positive-reaction staggered elliptic components with standalone
geometric multigrid or V-cycle-preconditioned BiCGStab. The returned named
tuple contains the component statistics and a combined `converged` flag.
"""
function solve!(solution::FaceVelocity{T},
                solver::StaggeredGeometricMultigridSolver{T},
                right_hand_side::FaceVelocity{T};
                rtol=64eps(T), atol=0, max_cycles=100,
                method=solver.method) where {T}
    solution.mesh === right_hand_side.mesh === solver.operator.mesh ||
        throw(ArgumentError(
            "staggered elliptic arguments must share a mesh"))
    solution === right_hand_side &&
        throw(ArgumentError(
            "staggered elliptic solution must not alias its right-hand side"))
    selected_method = _staggered_solve_method(solver, method)
    x_stats = _solve_staggered_component!(
        solution.interior, right_hand_side.interior, solver.operator.x,
        solver.x; rtol, atol, max_cycles, method=selected_method)
    y_stats = _solve_staggered_component!(
        solution.interior, right_hand_side.interior, solver.operator.y,
        solver.y; rtol, atol, max_cycles, method=selected_method)
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
                                 max_cycles=100, method=solver.method)

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
        max_cycles=100, method=solver.method) where {T}
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
    stats = solve!(
        destination, solver, work; rtol, atol, max_cycles, method)
    stats.converged ||
        throw(ErrorException("staggered diffusion solve failed to converge"))
    return destination
end
