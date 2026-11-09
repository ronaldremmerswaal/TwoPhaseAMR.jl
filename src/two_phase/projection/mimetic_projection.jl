# Jointly fitted mimetic pressure projection. The only layouts are the
# smoothed aperture-based cell rows (default) and CMDGFM-selected face rows.

"""
    MimeticProjectionSolver(interface; density=(1, 1), variant=:joint_smoothed, ...)

Configure the jointly fitted mimetic two-velocity pressure projection for
`interface`. The `:joint_smoothed` variant uses aperture-based cell rows;
`:joint_face` uses CMDGFM-selected face rows. `density` gives the liquid and
gas densities, and the remaining keywords control the preconditioner and fit.
"""
struct MimeticProjectionSolver{T, V <: _MimeticVariant, I, P, PO}
    interface::I
    density::NTuple{2, T}
    variant::V
    preconditioner::P
    preconditioner_options::PO
    active_mass_fraction::T
    rank_tolerance::T
    patch_radius::Int
    pressure_weight::T
    commutator_weight::T
    correction_cap::T
    fit_sweeps::Int
end

function MimeticProjectionSolver(
        interface::InterfaceField{T}; density = (one(T), one(T)),
        variant = :joint_smoothed,
        preconditioner = :cholesky, preconditioner_options = (;),
        active_mass_fraction = 1.0e-6, rank_tolerance = nothing,
        patch_radius = 2, pressure_weight = 1000,
        commutator_weight = 1000, correction_cap = 1,
        fit_sweeps = 3,
    ) where {T}
    return _mimetic_projection_solver(
        interface, _mimetic_variant(variant);
        density, preconditioner, preconditioner_options,
        active_mass_fraction, rank_tolerance, patch_radius,
        pressure_weight, commutator_weight, correction_cap, fit_sweeps,
    )
end

function _mimetic_projection_solver(
        interface::InterfaceField{T}, variant::V;
        density, preconditioner, preconditioner_options,
        active_mass_fraction, rank_tolerance, patch_radius,
        pressure_weight, commutator_weight, correction_cap, fit_sweeps,
    ) where {T, V <: _MimeticVariant}
    rho = (T(density[1]), T(density[2]))
    all(x -> isfinite(x) && x > zero(T), rho) || throw(
        ArgumentError(
            "phase densities must be finite and positive",
        )
    )
    mass_floor = isnothing(active_mass_fraction) ?
        sqrt(interface.fraction_floor) : T(active_mass_fraction)
    isfinite(mass_floor) && zero(T) < mass_floor < T(0.5) || throw(
        ArgumentError("active_mass_fraction must lie in (0, 0.5)"),
    )
    row_tolerance = isnothing(rank_tolerance) ?
        sqrt(eps(T)) : T(rank_tolerance)
    isfinite(row_tolerance) && zero(T) < row_tolerance < one(T) ||
        throw(ArgumentError("rank_tolerance must lie in (0, 1)"))
    radius = Int(patch_radius)
    radius >= 0 || throw(
        ArgumentError(
            "patch_radius must be nonnegative",
        )
    )
    weights = (
        T(pressure_weight), T(commutator_weight),
        T(correction_cap),
    )
    all(x -> isfinite(x) && x > zero(T), weights) || throw(
        ArgumentError(
            "fit weights and correction cap must be finite and positive",
        )
    )
    sweeps = Int(fit_sweeps)
    sweeps >= 1 || throw(ArgumentError("fit_sweeps must be positive"))
    return MimeticProjectionSolver{
        T, V, typeof(interface), typeof(preconditioner),
        typeof(preconditioner_options),
    }(
        interface, rho, variant, preconditioner, preconditioner_options,
        mass_floor, row_tolerance, radius, weights..., sweeps,
    )
end

function _local_block_preconditioner(
        pressure_block, relative_diagonal, constraint, ::Type{T},
    ) where {T}
    np = size(pressure_block, 1)
    nx = length(relative_diagonal)
    nl = size(constraint, 1)
    zero_px = spzeros(T, np, nx)
    zero_pl = spzeros(T, np, nl)
    zero_xl = spzeros(T, nx, nl)
    jump_block = spdiagm(0 => relative_diagonal)
    multiplier_block = if nl == 0
        spzeros(T, 0, 0)
    else
        schur = constraint * spdiagm(0 => inv.(relative_diagonal)) *
            transpose(constraint)
        # Algebraically independent local rows can still be almost parallel.
        # A roundoff-sized shift belongs only to the preconditioner; the KKT
        # operator and its exact constraints are left unchanged.
        scale = maximum(abs, diag(schur); init = one(T))
        shift = T(256) * eps(T) * scale
        schur + spdiagm(0 => fill(shift, nl))
    end
    return [
        pressure_block zero_px zero_pl
        transpose(zero_px) jump_block zero_xl
        transpose(zero_pl) transpose(zero_xl) multiplier_block
    ]
end

"""Exact inverse of the three independent SPD preconditioner blocks.

The block matrix is diagonal in pressure, jump, and multiplier variables.
Factorizing those pieces separately is algebraically identical to a single
Cholesky of the assembled block matrix. In particular, the jump entries use
the *same* physical `K` diagonal. Dedicated right-hand-side vectors for the
two sparse solves avoid passing a potentially aliased Krylov workspace to a
SuiteSparse factorization; this matters when the blocks change every step.
"""
struct MimeticBlockCholeskyPreconditioner{T, P, L}
    pressure::P
    inverse_jump::Vector{T}
    multiplier::L
    counts::NTuple{3, Int}
    pressure_rhs::Vector{T}
    multiplier_rhs::Vector{T}
end

Base.size(preconditioner::MimeticBlockCholeskyPreconditioner) =
    (sum(preconditioner.counts), sum(preconditioner.counts))
Base.eltype(::MimeticBlockCholeskyPreconditioner{T}) where {T} = T
_preconditioner_name(::MimeticBlockCholeskyPreconditioner) = :cholesky

# Julia 1.11's CHOLMOD factorization has only the allocating `\` API. Julia
# 1.12 and newer provide a three-argument in-place sparse solve.
@static if VERSION >= v"1.12"
    _mimetic_cholesky_ldiv!(destination, factor, rhs) =
        ldiv!(destination, factor, rhs)
else
    _mimetic_cholesky_ldiv!(destination, factor, rhs) =
        copyto!(destination, factor \ rhs)
end

function ldiv!(
        destination::AbstractVector{T},
        preconditioner::MimeticBlockCholeskyPreconditioner{T},
        right_hand_side::AbstractVector{T},
    ) where {T}
    np, nx, nl = preconditioner.counts
    length(destination) == length(right_hand_side) == np + nx + nl ||
        throw(DimensionMismatch("mimetic preconditioner vector size"))
    # Krylov may pass the same vector as input and output. Save both sparse
    # right-hand sides before writing any block, then solve into destination
    # views; CHOLMOD's three-argument ldiv! rejects aliased input/output.
    np > 0 && copyto!(preconditioner.pressure_rhs, 1, right_hand_side, 1, np)
    nl > 0 && copyto!(preconditioner.multiplier_rhs, 1, right_hand_side, np + nx + 1, nl)
    if np > 0
        _mimetic_cholesky_ldiv!(
            view(destination, 1:np), preconditioner.pressure,
            preconditioner.pressure_rhs
        )
    end
    for index in 1:nx
        @inbounds destination[np + index] =
            preconditioner.inverse_jump[index] * right_hand_side[np + index]
    end
    if nl > 0
        _mimetic_cholesky_ldiv!(
            view(destination, (np + nx + 1):(np + nx + nl)),
            preconditioner.multiplier, preconditioner.multiplier_rhs
        )
    end
    return destination
end

"""Reusable fields and the most recently assembled local KKT system.

Arrays and matrices exposed by this workspace are scratch owned by the
workspace and may be overwritten or resized by the next projection or
gradient reconstruction. Copy values that must outlive the next call.
"""
mutable struct MimeticProjectionWorkspace{T, M <: BlockTreeMesh}
    pressure_gradient::FaceVelocity{T, M}
    base_gradient::FaceVelocity{T, M}
    liquid_gradient::FaceVelocity{T, M}
    gas_gradient::FaceVelocity{T, M}
    jump::Vector{T}
    liquid_fraction_x::FaceField{1, T, M}
    liquid_fraction_y::FaceField{2, T, M}
    matrix::SparseMatrixCSC{T, Int}
    block_preconditioner::SparseMatrixCSC{T, Int}
    constraint_matrix::SparseMatrixCSC{T, Int}
    right_hand_side::Vector{T}
    solution::Vector{T}
    pressure_indices::Vector{Int}
    active_faces::Vector{Int}
    mixed_faces::Vector{Int}
    constraint_locations::Vector{Int}
    constraint_scales::Vector{T}
    cache::MimeticProjectionCache{T, M}
end

function MimeticProjectionWorkspace(
        solver::MimeticProjectionSolver{T},
    ) where {T}
    mesh = solver.interface.mesh
    return MimeticProjectionWorkspace(
        FaceVelocity(mesh, T), FaceVelocity(mesh, T),
        FaceVelocity(mesh, T), FaceVelocity(mesh, T),
        zeros(T, length(mesh.faces)), FaceField{1}(mesh, T),
        FaceField{2}(mesh, T), spzeros(T, 0, 0), spzeros(T, 0, 0),
        spzeros(T, 0, 0), T[], T[], Int[], Int[], Int[], Int[], T[],
        MimeticProjectionCache(T, typeof(mesh)),
    )
end

"""
    MimeticProjectionStats(; ...)

Diagnostics for one local, energy-compatible projection. `initial_residual`
and `final_residual` are the algebraic residual 2-norms. The convergence
decision compares `target_defect` with `mixture_volume_defect`,
`normal_velocity_defect`, and `tangential_defect`. The normal defect measures
the largest scaled local normal-jump violation, including inactive faces.
`independent_constraint_rows` is the number of multiplier unknowns.
`matrix_symmetry_error` is `nothing` unless `project!` is called with
`check_symmetry = true`.
"""
Base.@kwdef struct MimeticProjectionStats{T}
    converged::Bool
    iterations::Int
    initial_residual::T
    final_residual::T
    matrix_symmetry_error::Union{Nothing, T}
    mixture_divergence::T
    mixture_volume_defect::T
    normal_velocity_defect::T
    tangential_defect::T
    target_defect::T
    momentum_error::NTuple{2, T}
    momentum_identity_error::T
    energy_identity_error::T
    orthogonality_error::T
    pressure_unknowns::Int
    jump_unknowns::Int
    candidate_constraint_rows::Int
    independent_constraint_rows::Int
    dependent_constraint_rows::Int
    underresolved_constraint_rows::Int
    commuting_constraint_rows::Int
    fallback_constraint_rows::Int
    commuting_residual::T
    pressure_commuting_constraint_rows::Int
    pressure_fallback_constraint_rows::Int
    pressure_commuting_residual::T
    mass_deactivated_faces::Int
    embedded_deactivated_faces::Int
    preconditioner::Symbol
    variant::Symbol
end

_dimensionless_residual(stats::MimeticProjectionStats) = max(
    stats.mixture_volume_defect,
    stats.normal_velocity_defect,
    stats.tangential_defect,
)

function Base.show(io::IO, stats::MimeticProjectionStats)
    print(io, "MimeticProjectionStats(")
    for (index, name) in enumerate(fieldnames(typeof(stats)))
        index > 1 && print(io, ", ")
        print(io, name, '=')
        show(io, getfield(stats, name))
    end
    return print(io, ')')
end

function _local_embedded_cells(interface::InterfaceField{T}) where {T}
    geometry = interface.geometry
    result = falses(ncells(interface.mesh))
    isnothing(geometry) && return result
    references = _cell_references(interface.mesh)
    threshold = interface.fraction_floor
    for (index, reference) in pairs(references)
        open_fraction = _get(geometry.open_fraction, reference)
        @inbounds result[index] = open_fraction < one(T) - threshold
    end
    return result
end

function _local_active_faces(
        solver, data, active = Int[],
        embedded_cells = _local_embedded_cells(solver.interface)
    )
    interface = solver.interface
    mesh = interface.mesh
    nx, ny = mesh.block_size
    empty!(active)
    mass_deactivated = 0
    embedded_deactivated = 0
    for face_index in data.mixed
        face = mesh.faces[face_index]
        minus = _cell_index(face.minus, nx, ny)
        plus = _cell_index(face.plus, nx, ny)
        if embedded_cells[minus] || embedded_cells[plus]
            embedded_deactivated += 1
            continue
        end
        total = @inbounds data.total_mass[face_index]
        relative_fraction = @inbounds min(
            data.liquid_mass[face_index], data.gas_mass[face_index],
        ) / total
        open_aperture = @inbounds data.liquid_aperture[face_index] +
            data.gas_aperture[face_index]
        if relative_fraction < solver.active_mass_fraction ||
                open_aperture <= interface.fraction_floor
            mass_deactivated += 1
            continue
        end
        push!(active, face_index)
    end
    return active, mass_deactivated, embedded_deactivated
end

"""Cell-integrated `D(a_l n)` vectors, including physical-boundary faces."""
function _local_integrated_aperture_vectors(interface, data)
    T = eltype(interface)
    mesh = interface.mesh
    nx, ny = mesh.block_size
    x = zeros(T, ncells(mesh))
    y = zeros(T, ncells(mesh))
    for (index, face) in pairs(mesh.faces)
        minus = _cell_index(face.minus, nx, ny)
        plus = _cell_index(face.plus, nx, ny)
        contribution = T(face.area) * data.liquid_aperture[index]
        component = face.axis == 1 ? x : y
        @inbounds begin
            component[minus] += contribution
            component[plus] -= contribution
        end
    end
    for (index, face) in pairs(mesh.boundary_faces)
        liquid_aperture, = _phase_boundary_face_apertures(interface, index)
        cell = _cell_index(face.cell, nx, ny)
        component = face.axis == 1 ? x : y
        @inbounds component[cell] +=
            T(face.side) * T(face.area) * liquid_aperture
    end
    return x, y
end

function _local_independent_rows(
        matrix, relative_diagonal, tolerance, ::Type{T},
    ) where {T}
    size(matrix, 1) == 0 && return Int[]
    # Rank must be measured in the multiplier Schur metric, not merely in H.
    # This catches rows which are independent algebraically but indistinguish-
    # able after the phase kinetic Riesz map becomes strongly nonuniform.
    weighted = matrix * spdiagm(0 => inv.(sqrt.(relative_diagonal)))
    row_norms = sqrt.(diag(weighted * transpose(weighted)))
    normalised = spdiagm(0 => inv.(row_norms)) * weighted
    factorisation = qr(transpose(normalised); tol = tolerance)
    numerical_rank = rank(factorisation)
    numerical_rank == 0 && return Int[]
    return sort!(factorisation.pcol[1:numerical_rank])
end

"""Kinetic minimum-change projection onto local linear constraints.

The objective metric is `inv(K_xi)`, hence its inverse in the projection
formula is the diagonal `K_xi`.  Constraint rows are first normalised in
that metric.  Rank-deficient but compatible rows are harmless; incompatibility,
poor nonzero conditioning, or excessive coefficient amplification causes a
certification failure and triggers patch growth or fallback.
"""
function _local_commuting_fit(
        baseline, constraints, target, kinetic;
        rank_tolerance, amplification_limit,
    )
    T = eltype(baseline)
    retained = Int[]
    scales = T[]
    for row in axes(constraints, 1)
        scale = sqrt(
            sum(
                abs2(constraints[row, column]) * kinetic[column]
                    for column in axes(constraints, 2)
            )
        )
        if scale > T(64) * eps(T)
            push!(retained, row)
            push!(scales, scale)
        elseif abs(target[row]) > T(64) * eps(T)
            return (;
                success = false, values = baseline,
                residual = T(Inf), condition = T(Inf),
                amplification = T(Inf),
            )
        end
    end
    isempty(retained) && return (;
        success = true, values = baseline, residual = zero(T),
        condition = one(T), amplification = one(T),
    )
    A = Matrix(constraints[retained, :])
    d = target[retained]
    for row in axes(A, 1)
        A[row, :] ./= scales[row]
        d[row] /= scales[row]
    end
    gram = (A .* transpose(kinetic)) * transpose(A)
    decomposition = eigen(Symmetric(gram))
    largest = maximum(decomposition.values; init = zero(T))
    cutoff = T(rank_tolerance) * max(largest, one(T))
    retained_modes = findall(>(cutoff), decomposition.values)
    isempty(retained_modes) && return (;
        success = false, values = baseline, residual = T(Inf),
        condition = T(Inf), amplification = T(Inf),
    )
    vectors = decomposition.vectors[:, retained_modes]
    values = decomposition.values[retained_modes]
    correction_rhs = A * baseline - d
    multiplier = vectors * (
        (transpose(vectors) * correction_rhs) ./ values
    )
    result = baseline .- kinetic .* (transpose(A) * multiplier)
    residual = norm(A * result - d, Inf)
    condition = largest / minimum(values)
    baseline_norm = sqrt(
        sum(
            abs2(baseline[index]) / kinetic[index]
                for index in eachindex(baseline)
        )
    )
    correction_norm = sqrt(
        sum(
            abs2(result[index] - baseline[index]) / kinetic[index]
                for index in eachindex(result)
        )
    )
    amplification = correction_norm / max(baseline_norm, sqrt(eps(T)))
    residual_limit = max(T(128) * eps(T), T(10) * T(rank_tolerance))
    condition_limit = inv(sqrt(T(rank_tolerance)))
    success = residual <= residual_limit && condition <= condition_limit &&
        amplification <= T(amplification_limit)
    return (; success, values = result, residual, condition, amplification)
end

"""Assemble the aperture-divergence / joint-normal three-block KKT system.

The physical jump space is the full active-face space. The normal observation
`N` is the only velocity-jump constraint, and the *same* `N'` occurs in jump
stationarity. Thus fitting the rows changes accuracy but cannot break the
momentum or kinetic-energy adjoint identities of this projection.
"""
function _assemble_mimetic_system!(
        solver::MimeticProjectionSolver{T},
        workspace::MimeticProjectionWorkspace{T}, dt,
        velocity::TwoPhaseVelocity,
    ) where {T}
    interface = solver.interface
    mesh = interface.mesh
    cache = workspace.cache
    pressure_cache = cache.pressure
    data = _projection_face_data(solver, workspace; storage = pressure_cache.face_data)
    pressure_cache.face_data = data
    _mimetic_pressure_cache!(cache, solver, data)
    pressure_operator = pressure_cache.operator
    incidence = pressure_cache.incidence
    pressure_indices = pressure_cache.pressure_indices
    active_faces, mass_deactivated, embedded_deactivated =
        _local_active_faces(solver, data, cache.active_faces, cache.embedded_cells)
    fill!(cache.active_mask, false)
    for face in active_faces
        @inbounds cache.active_mask[face] = true
    end
    inactive_faces = empty!(cache.inactive_faces)
    for face in data.mixed
        @inbounds cache.active_mask[face] || push!(inactive_faces, face)
    end
    _mimetic_face_slices!(cache)
    pressure_positive = pressure_cache.reduced_pressure
    incidence_pressure = cache.incidence_pressure
    pressure_jump = -transpose(cache.active_incidence) *
        spdiagm(0 => data.coupling[active_faces])
    face_dual_volumes = cache.face_dual_volumes[active_faces]
    relative_diagonal =
        data.relative_mass[active_faces] .* face_dual_volumes
    all(>(zero(T)), relative_diagonal) || throw(
        ArgumentError(
            "active jump faces require positive phase masses",
        )
    )
    local_operator = _local_joint_constraint_operator(
        solver, data, active_faces, relative_diagonal, cache.fit,
    )
    constraint = local_operator.constraint
    multiplier_count = size(constraint, 1)
    zero_pl = spzeros(T, length(pressure_indices), multiplier_count)
    matrix = [
        -pressure_positive pressure_jump zero_pl
        transpose(pressure_jump) spdiagm(0 => relative_diagonal) transpose(constraint)
        transpose(zero_pl) constraint spzeros(T, multiplier_count, multiplier_count)
    ]
    block_preconditioner = _local_block_preconditioner(
        pressure_positive, relative_diagonal, constraint, T,
    )

    mixture = _projection_phase_sum.(
        data.liquid_aperture, velocity[1].interior,
        data.gas_aperture, velocity[2].interior
    )
    jump_velocity = zeros(T, length(mesh.faces))
    jump_velocity[data.mixed] .=
        velocity[2].interior[data.mixed] .- velocity[1].interior[data.mixed]
    # Faces outside the active jump space carry a known homogenisation.
    # This maintains the homogeneous feasible velocity space required by the
    # kinetic-energy projection identity; it is not a fitted joint row.
    inactive_jump_gradient = jump_velocity[inactive_faces] ./ T(dt)
    inactive_pressure_jump = isempty(inactive_faces) ?
        spzeros(T, length(pressure_indices), 0) :
        -transpose(cache.inactive_incidence) *
        spdiagm(0 => data.coupling[inactive_faces])
    boundary_mixture, _ = _mimetic_boundary_constraints(
        interface, velocity,
    )
    pressure_rhs = (
        -transpose(incidence_pressure) * mixture .+
            boundary_mixture[pressure_indices]
    ) ./ T(dt)
    pressure_rhs .-= inactive_pressure_jump * inactive_jump_gradient
    constraint_rhs =
        constraint * jump_velocity[active_faces] ./ T(dt)
    right_hand_side = workspace.right_hand_side
    np, nx = length(pressure_indices), length(active_faces)
    resize!(right_hand_side, np + nx + length(constraint_rhs))
    copyto!(right_hand_side, 1, pressure_rhs, 1, np)
    fill!(view(right_hand_side, (np + 1):(np + nx)), zero(T))
    copyto!(right_hand_side, np + nx + 1, constraint_rhs, 1, length(constraint_rhs))
    jump_residual_scales = [
        data.total_mass[face] * face_dual_volumes[local_face] *
            sqrt(face_dual_volumes[local_face])
            for (local_face, face) in pairs(active_faces)
    ]
    return (;
        data, pressure_operator, incidence, matrix, block_preconditioner,
        right_hand_side, pressure_indices, active_faces, constraint,
        candidate_constraint = local_operator.candidate_constraint,
        candidate_boundary = zeros(T, local_operator.candidate_count),
        candidate_scales = local_operator.candidate_scales,
        inactive_faces, inactive_jump_gradient,
        constraint_locations = local_operator.constraint_locations,
        constraint_scales = local_operator.constraint_scales,
        candidate_count = local_operator.candidate_count,
        dependent_count = local_operator.dependent_count,
        underresolved = local_operator.underresolved,
        commuting_count = local_operator.commuting_count,
        fallback_count = local_operator.fallback_count,
        commuting_residual = local_operator.commuting_residual,
        pressure_commuting_count =
            local_operator.pressure_commuting_count,
        pressure_fallback_count =
            local_operator.pressure_fallback_count,
        pressure_commuting_residual =
            local_operator.pressure_commuting_residual,
        mass_deactivated, embedded_deactivated, relative_diagonal,
        face_dual_volumes, jump_residual_scales,
    )
end

function _local_mimetic_residual_defects(residual, assembled, dt)
    T = eltype(residual)
    np = length(assembled.pressure_indices)
    nx = length(assembled.active_faces)
    nl = size(assembled.constraint, 1)
    volumes = assembled.pressure_operator.volumes
    pressure = _projection_pressure_residual_maximum(
        residual, assembled.pressure_indices, volumes, Val(false)
    )
    tangent = zero(T)
    for local_index in 1:nx
        scale = @inbounds assembled.jump_residual_scales[local_index]
        @inbounds tangent = max(
            tangent, abs(residual[np + local_index]) / scale,
        )
    end
    normal = zero(T)
    for local_index in 1:nl
        scale = @inbounds assembled.constraint_scales[local_index]
        @inbounds normal = max(
            normal, abs(residual[np + nx + local_index]) / scale,
        )
    end
    return (;
        pressure = T(dt)^2 * pressure,
        tangent = T(dt)^2 * tangent,
        normal = T(dt) * normal,
    )
end

function _local_mimetic_preconditioner(
        solver::MimeticProjectionSolver, assembled, cache = nothing,
    )
    specification = solver.preconditioner
    specification === nothing && return nothing
    block_preconditioner = assembled.block_preconditioner
    if specification === :cholesky
        isempty(solver.preconditioner_options) || throw(
            ArgumentError(
                "preconditioner_options are not used by exact Cholesky",
            ),
        )
        T = eltype(block_preconditioner)
        np = length(assembled.pressure_indices)
        nx = length(assembled.relative_diagonal)
        nl = size(assembled.constraint, 1)
        pressure = np == 0 ? nothing : cholesky(
                Symmetric(
                    block_preconditioner[1:np, 1:np], :L,
                )
            )
        multiplier_range = (np + nx + 1):(np + nx + nl)
        multiplier = nl == 0 ? nothing : cholesky(
                Symmetric(
                    block_preconditioner[multiplier_range, multiplier_range], :L,
                )
            )
        pressure_rhs = isnothing(cache) ? Vector{T}(undef, np) :
            resize!(cache.pressure_rhs, np)
        multiplier_rhs = isnothing(cache) ? Vector{T}(undef, nl) :
            resize!(cache.multiplier_rhs, nl)
        return MimeticBlockCholeskyPreconditioner(
            pressure, inv.(assembled.relative_diagonal), multiplier,
            (np, nx, nl), pressure_rhs, multiplier_rhs,
        )
    end
    return _construct_matrix_preconditioner(
        specification, block_preconditioner;
        symmetric = true, singular = false,
        options = solver.preconditioner_options,
    )
end

"""Reconstruct both phase pressure gradients from the latest local solve."""
function mimetic_gradient!(
        destination::TwoPhaseVelocity, pressure::CellField,
        solver::MimeticProjectionSolver{T},
        workspace::MimeticProjectionWorkspace{T}, pressure_operator,
        data,
    ) where {T}
    _validate_two_phase_velocity(destination, solver.interface.mesh, T)
    elliptic = workspace.cache.pressure.elliptic
    resize!(elliptic.input, ncells(solver.interface.mesh))
    resize!(elliptic.output, ncells(solver.interface.mesh))
    resize!(elliptic.face_input, length(solver.interface.mesh.faces))
    resize!(elliptic.face_output, length(solver.interface.mesh.faces))
    return _projection_gradient_core!(
        destination, pressure, solver, workspace, pressure_operator, data, elliptic,
    )
end

function mimetic_gradient!(
        destination::TwoPhaseVelocity, pressure::CellField,
        solver::MimeticProjectionSolver,
        workspace::MimeticProjectionWorkspace,
    )
    cache = workspace.cache
    data = _projection_face_data(solver, workspace; storage = cache.pressure.face_data)
    cache.pressure.face_data = data
    _mimetic_pressure_cache!(cache, solver, data)
    operator = cache.pressure.operator
    return mimetic_gradient!(
        destination, pressure, solver, workspace, operator, data,
    )
end

"""
    project!(velocity, pressure, solver::MimeticProjectionSolver, dt,
             workspace; tolerance, max_iterations, check_symmetry=false)

Project onto mixture incompressibility and the selected joint normal-jump
condition. Independent constraint rows and a phase-mass active set make the
three-block KKT system nonsingular; its block preconditioner contains the
multiplier Schur approximation `H*inv(K_xi)*H'`. Set `check_symmetry=true`
to measure the assembled matrix asymmetry for debugging.
"""
function project!(
        velocity::TwoPhaseVelocity, pressure::CellField{T},
        solver::MimeticProjectionSolver{T}, dt,
        workspace::MimeticProjectionWorkspace{T};
        tolerance = sqrt(eps(T)), max_iterations = 500, check_symmetry = false,
    ) where {T}
    interface = solver.interface
    _validate_two_phase_velocity(velocity, interface.mesh, T)
    pressure.mesh === interface.mesh || throw(
        ArgumentError("mimetic projection arguments must share a mesh"),
    )
    dt > zero(dt) || throw(ArgumentError("dt must be strictly positive"))
    tolerance > zero(tolerance) || throw(
        ArgumentError("tolerance must be strictly positive"),
    )
    max_iterations > 0 || throw(
        ArgumentError("max_iterations must be strictly positive"),
    )
    _mark_inactive_phase_velocities!(velocity, interface)
    assembled = _assemble_mimetic_system!(
        solver, workspace, dt, velocity,
    )
    matrix = assembled.matrix
    right_hand_side = assembled.right_hand_side
    symmetry_error = check_symmetry ?
        (isempty(matrix) ? zero(T) : maximum(abs, matrix - transpose(matrix))) :
        nothing
    initial_residual = norm(right_hand_side)
    solution = workspace.solution
    resize!(solution, length(right_hand_side))
    fill!(solution, zero(T))
    iterations = 0
    preconditioner = nothing
    if !isempty(solution) && initial_residual > zero(T)
        preconditioner = _local_mimetic_preconditioner(
            solver, assembled, workspace.cache,
        )
        correction_rhs = workspace.cache.correction_rhs
        resize!(correction_rhs, length(right_hand_side))
        copyto!(correction_rhs, right_hand_side)
        for refinement in 1:8
            remaining = max_iterations - iterations
            remaining > 0 || break
            defects = _local_mimetic_residual_defects(
                correction_rhs, assembled, dt,
            )
            maximum(defects) <= T(tolerance) && break
            krylov = _mimetic_krylov_workspace!(workspace.cache, matrix, correction_rhs)
            requested = min(
                T(tolerance), T(tolerance) / max(maximum(defects), eps(T)),
            )
            if isnothing(preconditioner)
                Krylov.minres!(
                    krylov, matrix, correction_rhs;
                    atol = zero(T), rtol = requested,
                    etol = zero(T), conlim = zero(T), itmax = remaining,
                )
            else
                Krylov.minres!(
                    krylov, matrix, correction_rhs;
                    M = preconditioner, ldiv = true,
                    atol = zero(T), rtol = requested,
                    etol = zero(T), conlim = zero(T), itmax = remaining,
                )
            end
            solution .+= Krylov.solution(krylov)
            iterations += Krylov.iteration_count(krylov)
            mul!(correction_rhs, matrix, solution)
            correction_rhs .= right_hand_side .- correction_rhs
        end
    end
    residual = workspace.cache.residual
    resize!(residual, length(right_hand_side))
    mul!(residual, matrix, solution)
    residual .-= right_hand_side
    final_residual = norm(residual)
    residual_defects = _local_mimetic_residual_defects(
        residual, assembled, dt,
    )

    np = length(assembled.pressure_indices)
    nx = length(assembled.active_faces)
    nl = size(assembled.constraint, 1)
    pressure_values = view(solution, 1:np)
    jump_values = view(solution, (np + 1):(np + nx))
    _copy_projection_cell_solution!(
        pressure, zip(assembled.pressure_indices, pressure_values),
    )
    fill!(workspace.jump, zero(T))
    workspace.jump[assembled.inactive_faces] .=
        assembled.inactive_jump_gradient
    workspace.jump[assembled.active_faces] .= jump_values

    gradients = (workspace.liquid_gradient, workspace.gas_gradient)
    mimetic_gradient!(
        gradients, pressure, solver, workspace,
        assembled.pressure_operator, assembled.data,
    )
    _projection_correct_phase_velocities!(velocity, gradients, dt, T)
    _mark_inactive_phase_velocities!(velocity, interface)
    final_liquid = velocity[1].interior
    final_gas = velocity[2].interior
    orthogonality = zero(T)
    pressure_momentum = zeros(T, 2)
    identity_error = zero(T)
    geometry = interface.geometry
    pressure_jump =
        assembled.pressure_operator.gradient_operator.pressure_jump
    for (index, face) in pairs(interface.mesh.faces)
        axis = face.axis
        !isnothing(geometry) && iszero(geometry.face_dual_volumes[index]) && continue
        @inbounds begin
            liquid_mass = assembled.data.liquid_mass[index]
            gas_mass = assembled.data.gas_mass[index]
            liquid_gradient = gradients[1].interior[index]
            gas_gradient = gradients[2].interior[index]
            final_common = _projection_phase_sum(
                liquid_mass, final_liquid[index], gas_mass, final_gas[index],
            ) / (liquid_mass + gas_mass)
            open_aperture = isnothing(geometry) ? one(T) :
                geometry.aperture[index]
            orthogonality +=
                final_common * open_aperture * pressure_jump[index]
            pressure_momentum[axis] -= T(dt) * pressure_jump[index]
            identity_error = max(
                identity_error, _projection_momentum_identity_error(
                    liquid_mass, gas_mass, liquid_gradient, gas_gradient,
                    workspace.pressure_gradient.interior[index]
                )
            )
        end
    end
    all_face_dual_volumes = workspace.cache.face_dual_volumes
    for face_index in assembled.data.mixed
        @inbounds orthogonality +=
            (final_gas[face_index] - final_liquid[face_index]) *
            assembled.data.relative_mass[face_index] *
            all_face_dual_volumes[face_index] *
            workspace.jump[face_index]
    end
    energy_error = abs(2T(dt) * orthogonality)
    mixture_constraint, = _mimetic_integrated_constraints(
        assembled.incidence, assembled.data, interface, velocity,
        assembled.active_faces,
    )
    volumes = assembled.pressure_operator.volumes
    mixture_divergence = _projection_strong_maximum(
        mixture_constraint, volumes,
    )
    mixture_volume_defect = T(dt) * mixture_divergence
    final_jump = final_gas[assembled.active_faces] .-
        final_liquid[assembled.active_faces]
    # Assess every resolved local row, including rows omitted from the solve
    # because rank revelation classified them as dependent.  This prevents
    # the rank filter from hiding a loss of local consistency.
    local_residual = assembled.candidate_constraint * final_jump .-
        assembled.candidate_boundary
    normal_velocity_defect = zero(T)
    for index in eachindex(local_residual)
        @inbounds normal_velocity_defect = max(
            normal_velocity_defect,
            abs(local_residual[index]) / assembled.candidate_scales[index],
        )
    end
    for face_index in assembled.inactive_faces
        @inbounds normal_velocity_defect = max(
            normal_velocity_defect,
            abs(final_gas[face_index] - final_liquid[face_index]),
        )
    end
    converged = mixture_volume_defect <= T(tolerance) &&
        normal_velocity_defect <= T(tolerance) &&
        residual_defects.tangent <= T(tolerance)
    preconditioner_name = isnothing(preconditioner) ? :none :
        _preconditioner_name(preconditioner)

    workspace.matrix = matrix
    workspace.block_preconditioner = assembled.block_preconditioner
    workspace.constraint_matrix = assembled.constraint
    workspace.right_hand_side = right_hand_side
    workspace.solution = solution
    workspace.pressure_indices = assembled.pressure_indices
    workspace.active_faces = assembled.active_faces
    workspace.mixed_faces = assembled.active_faces
    workspace.constraint_locations = assembled.constraint_locations
    workspace.constraint_scales = assembled.constraint_scales
    return MimeticProjectionStats(;
        converged, iterations, initial_residual, final_residual,
        matrix_symmetry_error = symmetry_error,
        mixture_divergence, mixture_volume_defect, normal_velocity_defect,
        tangential_defect = residual_defects.tangent,
        target_defect = T(tolerance),
        momentum_error = (pressure_momentum[1], pressure_momentum[2]),
        momentum_identity_error = identity_error,
        energy_identity_error = energy_error,
        orthogonality_error = abs(orthogonality),
        pressure_unknowns = np, jump_unknowns = nx,
        candidate_constraint_rows = assembled.candidate_count,
        independent_constraint_rows = nl,
        dependent_constraint_rows = assembled.dependent_count,
        underresolved_constraint_rows = assembled.underresolved,
        commuting_constraint_rows = assembled.commuting_count,
        fallback_constraint_rows = assembled.fallback_count,
        commuting_residual = assembled.commuting_residual,
        pressure_commuting_constraint_rows = assembled.pressure_commuting_count,
        pressure_fallback_constraint_rows = assembled.pressure_fallback_count,
        pressure_commuting_residual = assembled.pressure_commuting_residual,
        mass_deactivated_faces = assembled.mass_deactivated,
        embedded_deactivated_faces = assembled.embedded_deactivated,
        preconditioner = preconditioner_name,
        variant = _mimetic_variant_symbol(solver.variant),
    )
end
