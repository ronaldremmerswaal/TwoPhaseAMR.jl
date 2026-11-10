"""Reusable open-face connectivity and component reductions for an elliptic solve."""
mutable struct EllipticComponentWorkspace{T}
    incidence::SparseMatrixCSC{T, Int}
    counts::Vector{Int}
    labels::Vector{Int}
    stack::Vector{Int}
    active::BitVector
    open_faces::BitVector
    anchors::Vector{Int}
    volumes::Vector{T}
    sums::Vector{T}
    scales::Vector{T}
end

function EllipticComponentWorkspace(operator::EllipticOperator{T}) where {T}
    mesh = operator.mesh
    components = EllipticComponentWorkspace(
        spzeros(T, length(mesh.faces), ncells(mesh)), Int[], Int[], Int[],
        falses(0), falses(0), Int[], T[], T[], T[],
    )
    return _rebuild_elliptic_components!(components, operator)
end

function _rebuild_elliptic_components!(components, operator)
    mesh = operator.mesh
    components.incidence = _projection_incidence!(
        components.incidence, components.counts, mesh,
    )
    resize!(components.active, length(operator.volumes))
    resize!(components.open_faces, length(operator.face_coefficient))
    for index in eachindex(components.active)
        @inbounds components.active[index] = operator.volumes[index] > 0
    end
    for index in eachindex(components.open_faces)
        @inbounds components.open_faces[index] =
            !iszero(operator.face_coefficient[index])
    end
    _projection_component_labels!(
        components.labels, components.stack, components.incidence, mesh,
        components.active, components.open_faces,
    )
    empty!(components.anchors)
    for index in eachindex(components.labels)
        @inbounds components.labels[index] == index && push!(components.anchors, index)
    end
    for values in (components.volumes, components.sums, components.scales)
        resize!(values, length(operator.volumes))
    end
    return components
end

function _refresh_elliptic_components!(components, operator)
    changed = length(components.active) != length(operator.volumes) ||
        length(components.open_faces) != length(operator.face_coefficient)
    if !changed
        for index in eachindex(components.active)
            if @inbounds(components.active[index] != (operator.volumes[index] > 0))
                changed = true
                break
            end
        end
    end
    if !changed
        for index in eachindex(components.open_faces)
            if @inbounds(components.open_faces[index] != !iszero(operator.face_coefficient[index]))
                changed = true
                break
            end
        end
    end
    changed && _rebuild_elliptic_components!(components, operator)
    fill!(components.volumes, zero(eltype(components.volumes)))
    for index in eachindex(components.labels)
        root = @inbounds components.labels[index]
        iszero(root) && continue
        @inbounds components.volumes[root] += operator.volumes[index]
    end
    return components
end

"""Remove an independent volume-weighted constant from each active component."""
function _remove_component_volume_means!(values, volumes, components)
    fill!(components.sums, zero(eltype(components.sums)))
    for index in eachindex(values)
        root = @inbounds components.labels[index]
        iszero(root) && continue
        @inbounds components.sums[root] += volumes[index] * values[index]
    end
    for root in components.anchors
        @inbounds components.sums[root] /= components.volumes[root]
    end
    for index in eachindex(values)
        root = @inbounds components.labels[index]
        @inbounds values[index] = iszero(root) ? zero(eltype(values)) :
            values[index] - components.sums[root]
    end
    return values
end

function _project_component_rhs!(solver, compatibility)
    components = solver.components
    volumes = solver.operator.volumes
    rhs = solver.right_hand_side
    T = eltype(rhs)
    fill!(components.sums, zero(T))
    fill!(components.scales, zero(T))
    for index in eachindex(rhs)
        root = @inbounds components.labels[index]
        iszero(root) && continue
        value = @inbounds rhs[index]
        @inbounds components.sums[root] += value
        @inbounds components.scales[root] += abs(value)
    end
    total_sum = zero(T)
    total_volume = zero(T)
    for root in components.anchors
        sum_value = @inbounds components.sums[root]
        tolerance = T(256) * eps(T) * max(one(T), @inbounds(components.scales[root]))
        compatibility === :error && abs(sum_value) > tolerance && throw(
            ArgumentError("Poisson right-hand side has nonzero volume mean on component rooted at cell $root"),
        )
        total_sum += sum_value
        @inbounds total_volume += components.volumes[root]
    end
    stats = solver.stats
    copyto!(resize!(stats.removed_rhs_component_anchors, length(components.anchors)), components.anchors)
    resize!(stats.removed_rhs_component_means, length(components.anchors))
    for (component, root) in pairs(components.anchors)
        @inbounds components.sums[root] /= components.volumes[root]
        @inbounds stats.removed_rhs_component_means[component] = components.sums[root]
    end
    for index in eachindex(rhs)
        root = @inbounds components.labels[index]
        iszero(root) && continue
        @inbounds rhs[index] -= components.sums[root] * volumes[index]
    end
    return iszero(total_volume) ? zero(T) : total_sum / total_volume
end

"""
    EllipticSolver(operator, krylov, preconditioner)
    EllipticSolver(operator; krylov=Krylov.CgWorkspace,
                   krylov_options=(;),
                   preconditioner=:tree_geometric,
                   multigrid=MultigridOptions(), kwargs...)

Couple an elliptic operator to an external Krylov.jl workspace and an
independent preconditioner. The default is Krylov.jl CG with this package's
geometric multigrid V-cycle. A pre-built external Krylov workspace and any
preconditioner implementing three-argument `ldiv!` may be passed directly.
CG, BiCGSTAB, GMRES, and FGMRES workspaces are supported. `krylov_options`
forwards method settings while convergence tolerances, iteration limits, and
preconditioner placement remain controlled by `solve!` and this composition.
The multigrid object configures tree hierarchies; explicit individual hierarchy
keywords override its corresponding fields.
"""
mutable struct EllipticSolver{T, O, A, K, P, Q, C}
    operator::O
    action::A
    krylov::K
    preconditioner::P
    krylov_options::Q
    solution::Vector{T}
    right_hand_side::Vector{T}
    residual::Vector{T}
    product::Vector{T}
    components::C
    stats::LinearSolverStats{T}
end

function EllipticSolver(
        operator::O, krylov, preconditioner;
        krylov_options = (;)
    ) where
    {T, O <: EllipticOperator{T}}
    number_of_unknowns = length(operator.volumes)
    (krylov.m, krylov.n) == (number_of_unknowns, number_of_unknowns) ||
        throw(
        DimensionMismatch(
            "Krylov workspace and elliptic operator sizes differ"
        )
    )
    method = _krylov_method(krylov)
    return EllipticSolver(
        operator, FactorizedEllipticOperator(operator),
        krylov, preconditioner, krylov_options,
        zeros(T, number_of_unknowns), zeros(T, number_of_unknowns),
        zeros(T, number_of_unknowns), zeros(T, number_of_unknowns),
        operator.singular ? EllipticComponentWorkspace(operator) : nothing,
        LinearSolverStats(
            T; method,
            preconditioner = _preconditioner_name(preconditioner)
        )
    )
end

function EllipticSolver(
        operator::StaggeredEllipticComponent{Axis, T},
        krylov, preconditioner;
        krylov_options = (;)
    ) where {Axis, T}
    number_of_unknowns = length(operator.volumes)
    (krylov.m, krylov.n) == (number_of_unknowns, number_of_unknowns) ||
        throw(
        DimensionMismatch(
            "Krylov workspace and elliptic operator sizes differ"
        )
    )
    method = _krylov_method(krylov)
    return EllipticSolver(
        operator, operator.matrix,
        krylov, preconditioner, krylov_options,
        zeros(T, number_of_unknowns), zeros(T, number_of_unknowns),
        zeros(T, number_of_unknowns), zeros(T, number_of_unknowns),
        nothing,
        LinearSolverStats(
            T; method,
            preconditioner = _preconditioner_name(preconditioner)
        )
    )
end

function EllipticSolver(
        operator::EllipticOperator;
        krylov = Krylov.CgWorkspace,
        krylov_options = (;),
        preconditioner = :tree_geometric,
        preconditioner_options = (;),
        multigrid::MultigridOptions = MultigridOptions(),
        coarse_size = multigrid.coarse_size,
        pre_sweeps = multigrid.pre_sweeps, post_sweeps = multigrid.post_sweeps,
        max_levels = multigrid.max_levels,
    )
    multigrid = MultigridOptions(; coarse_size, pre_sweeps, post_sweeps, max_levels)
    right_hand_side = zeros(eltype(operator.volumes), length(operator.volumes))
    action = FactorizedEllipticOperator(operator)
    workspace = _construct_krylov(krylov, action, right_hand_side)
    built_preconditioner = _construct_pressure_preconditioner(
        preconditioner, operator; multigrid, options = preconditioner_options,
    )
    return EllipticSolver(
        operator, workspace, built_preconditioner; krylov_options
    )
end

"""Retarget pressure-solver storage after an AMR topology change.

The sparse operator and external Krylov workspace necessarily change size.
The package-owned solution/residual vectors and factorized face scratch grow
or shrink in place. Returns `nothing` when a concrete field type changed and a
new solver object is required.
"""
function _reconfigure_elliptic_solver!(
        solver::EllipticSolver{T}, operator::EllipticOperator{T},
        krylov_factory, preconditioner,
    ) where {T}
    typeof(operator) === typeof(solver.operator) || return nothing
    typeof(preconditioner) === typeof(solver.preconditioner) || return nothing
    isnothing(solver.components) == !operator.singular || return nothing
    action = solver.action
    action isa FactorizedEllipticOperator || return nothing
    typeof(action.operator) === typeof(operator) || return nothing

    action.operator = operator
    resize!(action.face_input, length(operator.face_coefficient))
    resize!(action.face_output, length(operator.face_coefficient))
    fill!(action.face_input, zero(T))
    fill!(action.face_output, zero(T))
    offsets, faces, signs = _cell_face_incidence(operator.mesh)
    action.cell_face_offsets = offsets
    action.cell_faces = faces
    action.cell_face_signs = signs
    number_of_unknowns = length(operator.volumes)
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
            krylov_factory, action, solver.right_hand_side,
        )
    )
    typeof(krylov) === typeof(solver.krylov) || return nothing
    solver.operator = operator
    isnothing(solver.components) ||
        _rebuild_elliptic_components!(solver.components, operator)
    solver.krylov = krylov
    solver.preconditioner = preconditioner
    solver.stats = LinearSolverStats(
        T; method = _krylov_method(krylov),
        preconditioner = _preconditioner_name(preconditioner),
    )
    return solver
end

function _krylov_method(workspace)
    traits = _krylov_traits(workspace)
    return isnothing(traits) ? Symbol(nameof(typeof(workspace))) : traits.method
end

struct StrongResidualCallback{T, V}
    target::T
    volumes::V
end

function (callback::StrongResidualCallback)(workspace)
    _, maximum_norm = _integrated_residual_norms(
        workspace.r, callback.volumes
    )
    return maximum_norm <= callback.target
end

# GMRES does not expose the current true residual to callbacks until the end
# of an Arnoldi pass. Since the Krylov residual is the integrated residual for
# right preconditioning, this Euclidean tolerance is a conservative guarantee
# for the package's strong L-infinity residual contract:
# norm(r, 2) <= target*minimum(volumes) implies abs(r[i])/volumes[i] <= target.
function _gmres_absolute_tolerance(target, volumes)
    minimum_positive = minimum(
        volume for volume in volumes if volume > zero(eltype(volumes));
        init = one(eltype(volumes)),
    )
    return target * minimum_positive
end

function _run_krylov!(
        workspace, matrix, right_hand_side, initial_solution,
        preconditioner, target, volumes, max_iterations, options
    )
    traits = _krylov_traits(workspace)
    isnothing(traits) && throw(
        ArgumentError(
            "unsupported Krylov workspace $(typeof(workspace)); extend " *
                "TwoPhaseAMR._run_krylov! for this external solver"
        )
    )
    settings = if traits.callback
        callback = StrongResidualCallback(target, volumes)
        merge(
            options, (;
                atol = zero(target), rtol = zero(target),
                itmax = max_iterations, callback,
            )
        )
    else
        atol = _gmres_absolute_tolerance(target, volumes)
        merge(
            options, (;
                atol, rtol = zero(target), itmax = max_iterations,
            )
        )
    end
    if isnothing(preconditioner)
        traits.solve(
            workspace, matrix, right_hand_side, initial_solution;
            settings...
        )
    else
        preconditioned_settings = if traits.preconditioner === :M
            merge(settings, (; M = preconditioner, ldiv = true))
        else
            merge(settings, (; N = preconditioner, ldiv = true))
        end
        traits.solve(
            workspace, matrix, right_hand_side, initial_solution;
            preconditioned_settings...
        )
    end
    # BiCGSTAB can evaluate 0/0 before checking an almost-exact intermediate
    # residual. Retry without our preconditioner only for this breakdown.
    if traits.retry_unpreconditioned &&
            !all(isfinite, Krylov.solution(workspace)) &&
            !isnothing(preconditioner)
        traits.solve(
            workspace, matrix, right_hand_side, initial_solution;
            settings...
        )
    end
    return workspace
end

function _run_krylov_reliably!(
        workspace, matrix, right_hand_side, solution, preconditioner,
        target, volumes, max_iterations, options, residual, product,
    )
    total_iterations = 0
    final_linf = oftype(target, Inf)
    while total_iterations < max_iterations
        _run_krylov!(
            workspace, matrix, right_hand_side, solution,
            preconditioner, target, volumes,
            max_iterations - total_iterations, options,
        )
        iterations = Krylov.iteration_count(workspace)
        total_iterations += iterations
        copyto!(solution, Krylov.solution(workspace))

        # Krylov recurrences update a recursively computed residual. In a cut
        # cell the strong residual divides that integrated residual by a very
        # small physical volume, so roundoff drift can satisfy the recurrence
        # callback while the true residual still violates the requested
        # tolerance. Restarting from the current iterate replaces the
        # recurrence residual by b-A*x without changing the discretization.
        mul!(product, matrix, solution)
        for index in eachindex(residual)
            @inbounds residual[index] =
                right_hand_side[index] - product[index]
        end
        _, final_linf = _integrated_residual_norms(residual, volumes)
        final_linf <= target && break
        (!isfinite(final_linf) || iszero(iterations)) && break
    end
    return total_iterations
end

function _true_residual!(solver)
    if solver isa EllipticSolver && solver.action isa FactorizedEllipticOperator
        solver.action.operator = solver.operator
    end
    mul!(solver.product, solver.action, solver.solution)
    for index in eachindex(solver.residual)
        @inbounds solver.residual[index] =
            solver.right_hand_side[index] - solver.product[index]
    end
    return solver.residual
end

"""Run the common Krylov/statistics sequence with problem-specific copies.

`prepare!` gathers the initial guess and RHS and returns the removed RHS mean.
`copy_solution!` runs after the final residual, but before stats are
updated. The roundoff floor is fixed from the initial state, never the iterates.
Elliptic solves use the initial residual as the relative scale; shifted
diffusion solves use the strong right-hand-side scale instead.
`matrix` is the solve/floor operator, while the true residual uses `solver.action`.
"""
function _solve_krylov_with_stats!(
        copy_solution!, prepare!, solver, matrix, volumes, rtol, max_cycles;
        atol = zero(eltype(solver.solution)), relative_to_rhs = false,
        components = nothing,
    )
    T = eltype(solver.solution)
    removed_mean = prepare!()
    _true_residual!(solver)
    initial_l2, initial_linf =
        _integrated_residual_norms(solver.residual, volumes)
    roundoff_floor = _roundoff_residual_floor!(
        matrix, solver.solution, solver.right_hand_side,
        volumes, solver.product,
    )
    if relative_to_rhs
        residual_scale = _strong_maximum(solver.right_hand_side, volumes)
        target = max(rtol * residual_scale, roundoff_floor)
    else
        target = max(
            T(atol), T(rtol) * max(initial_linf, eps(T)),
            roundoff_floor,
        )
    end
    iterations = 0
    if initial_linf > target
        iterations = _run_krylov_reliably!(
            solver.krylov, matrix,
            solver.right_hand_side, solver.solution, solver.preconditioner,
            target, volumes, max_cycles, solver.krylov_options,
            solver.residual, solver.product,
        )
    end
    isnothing(components) ||
        _remove_component_volume_means!(solver.solution, volumes, components)
    _true_residual!(solver)
    final_l2, final_linf =
        _integrated_residual_norms(solver.residual, volumes)
    copy_solution!(solver.solution)

    stats = solver.stats
    stats.cycles = iterations
    stats.converged = isfinite(final_linf) && isfinite(target) && final_linf <= target
    stats.initial_l2 = initial_l2
    stats.initial_linf = initial_linf
    stats.final_l2 = final_l2
    stats.final_linf = final_linf
    stats.target_linf = target
    stats.residual_scale =
        relative_to_rhs ? residual_scale : max(initial_linf, eps(T))
    stats.removed_rhs_mean = removed_mean
    stats.method = _krylov_method(solver.krylov)
    stats.preconditioner = _preconditioner_name(solver.preconditioner)
    return stats
end

"""
    solve!(solution, solver, rhs; rtol=1e-10, atol=0,
           max_cycles=100, compatibility=:project)

Solve the strong-form equation `L(solution) = rhs` with the external Krylov
workspace and independent preconditioner stored in `solver`.
For periodic or all-Neumann Poisson problems the right-hand side must have zero
volume mean on each open-face connected component. `compatibility=:project`
removes and reports each component mean; `:error` rejects an incompatible
component. For these singular problems, the returned pressure has zero
volume-weighted mean separately on each component and covered cells are
set to zero. Covered-cell RHS values are ignored.
"""
function solve!(
        solution::CellField{T}, solver::EllipticSolver{T},
        right_hand_side::CellField{T}; rtol = 1.0e-10, atol = 0,
        max_cycles = 100, compatibility = :project
    ) where {T}
    solution.mesh === right_hand_side.mesh === solver.operator.mesh ||
        throw(ArgumentError("fields and multigrid solver must share a mesh"))
    zero(T) <= rtol < one(T) || throw(ArgumentError("rtol must lie in [0, 1)"))
    atol >= zero(T) || throw(ArgumentError("atol must be nonnegative"))
    max_cycles >= 0 || throw(ArgumentError("max_cycles must be nonnegative"))
    compatibility in (:project, :error) ||
        throw(ArgumentError("compatibility must be :project or :error"))

    volumes = solver.operator.volumes
    singular = solver.operator.singular
    prepare! = () -> begin
        _copy_from_field!(solver.solution, solution, solver.operator.references)
        _copy_from_field!(
            solver.right_hand_side, right_hand_side,
            solver.operator.references,
        )
        for index in eachindex(solver.right_hand_side)
            @inbounds solver.right_hand_side[index] = iszero(volumes[index]) ?
                zero(T) : solver.right_hand_side[index] * volumes[index]
        end

        empty!(solver.stats.removed_rhs_component_means)
        empty!(solver.stats.removed_rhs_component_anchors)
        removed_mean = zero(T)
        if singular
            _refresh_elliptic_components!(solver.components, solver.operator)
            removed_mean = _project_component_rhs!(solver, compatibility)
            _remove_component_volume_means!(solver.solution, volumes, solver.components)
        end
        return removed_mean
    end

    return _solve_krylov_with_stats!(
        prepare!, solver, solver.action, volumes, rtol, max_cycles;
        atol, components = solver.components,
    ) do values
        _copy_to_field!(solution, values, solver.operator.references)
    end
end

"""Return `(L2, LInf)` strong residual norms from the most recent solve."""
residual_norms(solver::EllipticSolver) =
    (solver.stats.final_l2, solver.stats.final_linf)
