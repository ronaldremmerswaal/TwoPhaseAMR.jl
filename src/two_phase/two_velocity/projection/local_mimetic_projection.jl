"""Configuration for the locally consistent, energy-compatible projection.

The jump condition on an active face `f` is

`q_f ⋅ J¹_f(u_g - u_l) = 0`,  `q_f = I D(a_l n)_f`.

`q_f` is deliberately not normalised: every equation retains its natural
interface-measure scaling.  The transpose of this local operator appears in
the jump stationarity equation, so the resulting KKT projection is symmetric
and orthogonal in the phase kinetic-energy inner product.
"""
struct LocalMimeticProjectionSolver{T, I, P, PO}
    interface::I
    density::NTuple{2, T}
    preconditioner::P
    preconditioner_options::PO
    active_mass_fraction::T
    rank_tolerance::T
end

function LocalMimeticProjectionSolver(
        interface::InterfaceField{T}; density = (one(T), one(T)),
        preconditioner = :cholesky, preconditioner_options = (;),
        active_mass_fraction = nothing, rank_tolerance = nothing,
    ) where {T}
    rho = (T(density[1]), T(density[2]))
    all(value -> isfinite(value) && value > zero(T), rho) || throw(
        ArgumentError("phase densities must be finite and positive"),
    )
    mass_floor = isnothing(active_mass_fraction) ?
        sqrt(interface.fraction_floor) : T(active_mass_fraction)
    isfinite(mass_floor) && zero(T) < mass_floor < T(0.5) || throw(
        ArgumentError("active_mass_fraction must be finite and in (0, 0.5)"),
    )
    row_tolerance = isnothing(rank_tolerance) ?
        sqrt(sqrt(eps(T))) : T(rank_tolerance)
    isfinite(row_tolerance) && zero(T) < row_tolerance < one(T) || throw(
        ArgumentError("rank_tolerance must be finite and in (0, 1)"),
    )
    return LocalMimeticProjectionSolver{
        T, typeof(interface), typeof(preconditioner),
        typeof(preconditioner_options),
    }(
        interface, rho, preconditioner, preconditioner_options,
        mass_floor, row_tolerance,
    )
end

"""Reusable fields and the most recently assembled local KKT system."""
mutable struct LocalMimeticProjectionWorkspace{T, M <: BlockTreeMesh}
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
    constraint_faces::Vector{Int}
    constraint_scales::Vector{T}
end

function LocalMimeticProjectionWorkspace(
        solver::LocalMimeticProjectionSolver{T},
    ) where {T}
    mesh = solver.interface.mesh
    return LocalMimeticProjectionWorkspace(
        FaceVelocity(mesh, T), FaceVelocity(mesh, T),
        FaceVelocity(mesh, T), FaceVelocity(mesh, T),
        zeros(T, length(mesh.faces)), FaceField{1}(mesh, T),
        FaceField{2}(mesh, T), spzeros(T, 0, 0), spzeros(T, 0, 0),
        spzeros(T, 0, 0), T[], T[], Int[], Int[], Int[], T[],
    )
end

"""Diagnostics for one local, energy-compatible projection."""
struct LocalMimeticProjectionStats{T}
    converged::Bool
    iterations::Int
    initial_residual::T
    final_residual::T
    matrix_symmetry_error::T
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
    mass_deactivated_faces::Int
    embedded_deactivated_faces::Int
    preconditioner::Symbol
end

_dimensionless_residual(stats::LocalMimeticProjectionStats) = max(
    stats.mixture_volume_defect,
    stats.normal_velocity_defect,
    stats.tangential_defect,
)

function Base.show(io::IO, stats::LocalMimeticProjectionStats)
    return print(
        io, "LocalMimeticProjectionStats(converged=", stats.converged,
        ", iterations=", stats.iterations,
        ", initial_residual=", stats.initial_residual,
        ", final_residual=", stats.final_residual,
        ", matrix_symmetry_error=", stats.matrix_symmetry_error,
        ", mixture_divergence=", stats.mixture_divergence,
        ", mixture_volume_defect=", stats.mixture_volume_defect,
        ", normal_velocity_defect=", stats.normal_velocity_defect,
        ", tangential_defect=", stats.tangential_defect,
        ", target_defect=", stats.target_defect,
        ", momentum_error=", stats.momentum_error,
        ", momentum_identity_error=", stats.momentum_identity_error,
        ", energy_identity_error=", stats.energy_identity_error,
        ", orthogonality_error=", stats.orthogonality_error,
        ", unknowns=(`p`=", stats.pressure_unknowns,
        ", `xi`=", stats.jump_unknowns,
        ", `lambda`=", stats.independent_constraint_rows, ")",
        ", constraint_rows=(candidate=", stats.candidate_constraint_rows,
        ", dependent=", stats.dependent_constraint_rows,
        ", underresolved=", stats.underresolved_constraint_rows, ")",
        ", deactivated=(mass=", stats.mass_deactivated_faces,
        ", embedded=", stats.embedded_deactivated_faces, ")",
        ", preconditioner=", stats.preconditioner, ")",
    )
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

function _local_active_faces(solver, data)
    interface = solver.interface
    mesh = interface.mesh
    nx, ny = mesh.block_size
    embedded_cells = _local_embedded_cells(interface)
    active = Int[]
    mass_deactivated = 0
    embedded_deactivated = 0
    for face_index in data.mixed
        face = mesh.faces[face_index]
        minus = _elliptic_cell_index(face.minus, nx, ny)
        plus = _elliptic_cell_index(face.plus, nx, ny)
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
        minus = _elliptic_cell_index(face.minus, nx, ny)
        plus = _elliptic_cell_index(face.plus, nx, ny)
        contribution = T(face.area) * data.liquid_aperture[index]
        component = face.axis == 1 ? x : y
        @inbounds begin
            component[minus] += contribution
            component[plus] -= contribution
        end
    end
    for (index, face) in pairs(mesh.boundary_faces)
        liquid_aperture, = _phase_boundary_face_apertures(interface, index)
        cell = _elliptic_cell_index(face.cell, nx, ny)
        component = face.axis == 1 ? x : y
        @inbounds component[cell] +=
            T(face.side) * T(face.area) * liquid_aperture
    end
    return x, y
end

@inline function _local_face_aperture_vector(
        mesh::BlockTreeMesh, face::InteriorFace, x, y,
    )
    nx, ny = mesh.block_size
    minus = _elliptic_cell_index(face.minus, nx, ny)
    plus = _elliptic_cell_index(face.plus, nx, ny)
    hm = mesh.blocks[face.minus.block].spacing[face.axis]
    hp = mesh.blocks[face.plus.block].spacing[face.axis]
    denominator = hm + hp
    return (
        (hm * x[minus] + hp * x[plus]) / denominator,
        (hm * y[minus] + hp * y[plus]) / denominator,
    )
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

"""Assemble independent rows of `I D(a_l n) ⋅ J¹` on the active jump space."""
function _local_constraint_operator(
        solver::LocalMimeticProjectionSolver{T}, data, active_faces,
        relative_diagonal,
    ) where {T}
    interface = solver.interface
    mesh = interface.mesh
    active_lookup = zeros(Int, length(mesh.faces))
    for (local_index, face_index) in pairs(active_faces)
        active_lookup[face_index] = local_index
    end
    vertex_patches = _vertex_patches(mesh, T)
    aperture_x, aperture_y = _local_integrated_aperture_vectors(
        interface, data,
    )
    rows = Int[]
    columns = Int[]
    values = T[]
    candidate_faces = Int[]
    candidate_scales = T[]
    underresolved = 0
    coefficient_tolerance = T(64) * eps(T)
    for face_index in active_faces
        face = mesh.faces[face_index]
        q = _local_face_aperture_vector(
            mesh, face, aperture_x, aperture_y,
        )
        qnorm = hypot(q[1], q[2])
        if qnorm <= coefficient_tolerance * max(T(face.area), one(T))
            underresolved += 1
            continue
        end
        tangent = 3 - face.axis
        transverse = Int[]
        weights = T[]
        if abs(q[tangent]) > coefficient_tolerance * qnorm
            transverse = _cmdgfm_vertex_candidates(
                vertex_patches, mesh, face, face_index, tangent,
            )
            filter!(index -> active_lookup[index] > 0, transverse)
            raw_weights = map(transverse) do index
                fraction = data.liquid_fraction[index]
                fraction * (one(T) - fraction)
            end
            weight_sum = sum(raw_weights)
            touches_boundary = any(
                endpoint -> _physical_vertex(
                    mesh, _vertex_key(mesh, face, endpoint),
                ),
                1:2,
            )
            crosses_level = any(
                index -> mesh.faces[index].area != face.area, transverse,
            )
            required = touches_boundary || crosses_level ? 1 : 2
            if length(transverse) < required || weight_sum <= zero(T)
                underresolved += 1
                continue
            end
            weights = raw_weights ./ weight_sum
        end

        local_columns = Int[]
        local_values = T[]
        normal_coefficient = q[face.axis]
        if abs(normal_coefficient) > coefficient_tolerance * qnorm
            push!(local_columns, active_lookup[face_index])
            push!(local_values, normal_coefficient)
        end
        for (candidate, weight) in zip(transverse, weights)
            coefficient = q[tangent] * weight
            abs(coefficient) > coefficient_tolerance * qnorm || continue
            push!(local_columns, active_lookup[candidate])
            push!(local_values, coefficient)
        end
        if isempty(local_columns)
            underresolved += 1
            continue
        end
        row = length(candidate_faces) + 1
        append!(rows, fill(row, length(local_columns)))
        append!(columns, local_columns)
        append!(values, local_values)
        push!(candidate_faces, face_index)
        push!(candidate_scales, qnorm)
    end
    candidate = sparse(
        rows, columns, values, length(candidate_faces),
        length(active_faces),
    )
    retained = _local_independent_rows(
        candidate, relative_diagonal, solver.rank_tolerance, T,
    )
    constraint = candidate[retained, :]
    return (;
        constraint, candidate_constraint = candidate,
        constraint_faces = candidate_faces[retained],
        constraint_scales = candidate_scales[retained],
        candidate_scales,
        candidate_count = length(candidate_faces),
        dependent_count = length(candidate_faces) - length(retained),
        underresolved,
    )
end

function _assemble_local_mimetic_system!(
        solver::LocalMimeticProjectionSolver{T},
        workspace::LocalMimeticProjectionWorkspace{T}, dt,
        velocity::TwoPhaseVelocity,
    ) where {T}
    interface = solver.interface
    mesh = interface.mesh
    data = _mimetic_face_data(solver, workspace)
    pressure_operator = _mimetic_pressure_operator(solver)
    incidence = _mimetic_incidence(mesh, T)
    active_cells = _mimetic_active_cells(interface)
    open_weights = isnothing(interface.geometry) ?
        ones(T, length(mesh.faces)) : copy(interface.geometry.aperture)
    pressure_indices = _gauge_free_component_rows(
        mesh, active_cells, open_weights,
    )
    active_faces, mass_deactivated, embedded_deactivated =
        _local_active_faces(solver, data)
    active_set = Set(active_faces)
    inactive_faces = filter(
        face_index -> !(face_index in active_set), data.mixed,
    )
    pressure_positive = pressure_operator.stiffness[
        pressure_indices, pressure_indices,
    ]
    incidence_pressure = incidence[:, pressure_indices]
    coupling = data.coupling[active_faces]
    pressure_jump = -transpose(incidence_pressure[active_faces, :]) *
        spdiagm(0 => coupling)
    all_face_dual_volumes = isnothing(interface.geometry) ?
        _uncut_face_dual_volumes(mesh, T) :
        _geometry_face_dual_volumes(interface.geometry)
    face_dual_volumes = all_face_dual_volumes[active_faces]
    relative_diagonal =
        data.relative_mass[active_faces] .* face_dual_volumes
    all(>(zero(T)), relative_diagonal) || throw(
        ArgumentError("active jump faces require positive phase masses"),
    )
    local_operator = _local_constraint_operator(
        solver, data, active_faces, relative_diagonal,
    )
    constraint = local_operator.constraint
    multiplier_count = size(constraint, 1)
    zero_pl = spzeros(T, length(pressure_indices), multiplier_count)
    matrix = [
        -pressure_positive pressure_jump zero_pl
        transpose(pressure_jump) spdiagm(0 => relative_diagonal) transpose(constraint)
        transpose(zero_pl) constraint spzeros(T, multiplier_count, multiplier_count)
    ]
    block_preconditioner = _mimetic_block_preconditioner(
        pressure_positive, relative_diagonal, constraint, T,
    )

    mixture = data.liquid_aperture .* velocity[1].interior .+
        data.gas_aperture .* velocity[2].interior
    jump_velocity = velocity[2].interior .- velocity[1].interior
    # On a face removed by the mass/geometry active set, constrain the final
    # jump to zero.  Treating its correction as known keeps the feasible
    # velocity space homogeneous, which is essential for the pressure-work
    # (and hence energy) identity.  Merely setting xi=0 would leave an affine
    # divergence constraint and destroy orthogonality.
    inactive_jump_gradient = jump_velocity[inactive_faces] ./ T(dt)
    inactive_pressure_jump = isempty(inactive_faces) ?
        spzeros(T, length(pressure_indices), 0) :
        -transpose(incidence_pressure[inactive_faces, :]) *
            spdiagm(0 => data.coupling[inactive_faces])
    boundary_mixture, = _mimetic_boundary_constraints(interface, velocity)
    pressure_rhs = (
        -transpose(incidence_pressure) * mixture .+
            boundary_mixture[pressure_indices]
    ) ./ T(dt)
    pressure_rhs .-= inactive_pressure_jump * inactive_jump_gradient
    constraint_rhs = constraint * jump_velocity[active_faces] ./ T(dt)
    right_hand_side = vcat(
        pressure_rhs, zeros(T, length(active_faces)), constraint_rhs,
    )
    return (;
        data, pressure_operator, incidence, matrix, block_preconditioner,
        right_hand_side, pressure_indices, active_faces, constraint,
        candidate_constraint = local_operator.candidate_constraint,
        candidate_scales = local_operator.candidate_scales,
        inactive_faces, inactive_jump_gradient,
        constraint_faces = local_operator.constraint_faces,
        constraint_scales = local_operator.constraint_scales,
        candidate_count = local_operator.candidate_count,
        dependent_count = local_operator.dependent_count,
        underresolved = local_operator.underresolved,
        mass_deactivated, embedded_deactivated,
        relative_diagonal, face_dual_volumes,
    )
end

function _local_mimetic_residual_defects(residual, assembled, dt)
    T = eltype(residual)
    np = length(assembled.pressure_indices)
    nx = length(assembled.active_faces)
    nl = size(assembled.constraint, 1)
    volumes = assembled.pressure_operator.volumes
    pressure = zero(T)
    for local_index in 1:np
        cell = @inbounds assembled.pressure_indices[local_index]
        @inbounds pressure = max(
            pressure, abs(residual[local_index]) / volumes[cell],
        )
    end
    tangent = zero(T)
    for local_index in 1:nx
        face = @inbounds assembled.active_faces[local_index]
        dual_volume = @inbounds assembled.face_dual_volumes[local_index]
        scale = @inbounds assembled.data.total_mass[face] * dual_volume *
            sqrt(dual_volume)
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
        solver::LocalMimeticProjectionSolver, block_preconditioner,
    )
    specification = solver.preconditioner
    specification === nothing && return nothing
    if specification === :cholesky
        isempty(solver.preconditioner_options) || throw(
            ArgumentError(
                "preconditioner_options are not used by exact Cholesky",
            ),
        )
        return ExternalPreconditioner(
            :cholesky,
            cholesky(Symmetric(block_preconditioner, :L)),
            block_preconditioner,
        )
    end
    return _construct_matrix_preconditioner(
        specification, block_preconditioner;
        symmetric = true, singular = false,
        options = solver.preconditioner_options,
    )
end

"""Reconstruct both phase pressure gradients from the latest local solve."""
function local_mimetic_gradient!(
        destination::TwoPhaseVelocity, pressure::CellField,
        solver::LocalMimeticProjectionSolver{T},
        workspace::LocalMimeticProjectionWorkspace{T}, pressure_operator,
        data,
    ) where {T}
    _validate_two_phase_velocity(destination, solver.interface.mesh, T)
    elliptic = EllipticWorkspace(pressure_operator)
    _mimetic_weighted_gradient!(
        workspace.pressure_gradient, pressure, pressure_operator, elliptic,
        solver.interface.geometry,
    )
    gradient!(
        workspace.base_gradient, pressure,
        pressure_operator.gradient_operator,
    )
    for index in eachindex(solver.interface.mesh.faces)
        mass = @inbounds data.total_mass[index]
        jump = @inbounds workspace.jump[index]
        common = @inbounds workspace.pressure_gradient.interior[index]
        @inbounds begin
            destination[1].interior[index] = common +
                data.zeta_liquid[index] / mass * jump
            destination[2].interior[index] = common +
                data.zeta_gas[index] / mass * jump
        end
    end
    fill!(destination[1].boundary, zero(T))
    fill!(destination[2].boundary, zero(T))
    return destination
end

function local_mimetic_gradient!(
        destination::TwoPhaseVelocity, pressure::CellField,
        solver::LocalMimeticProjectionSolver,
        workspace::LocalMimeticProjectionWorkspace,
    )
    operator = _mimetic_pressure_operator(solver)
    data = _mimetic_face_data(solver, workspace)
    return local_mimetic_gradient!(
        destination, pressure, solver, workspace, operator, data,
    )
end

"""
    project!(velocity, pressure, solver::LocalMimeticProjectionSolver, dt,
             workspace; tolerance, max_iterations)

Project onto mixture incompressibility and the face-local unnormalised
aperture-normal condition.  Independent constraint rows and a phase-mass
active set make the three-block KKT system nonsingular; its block
preconditioner contains the multiplier Schur approximation
`H*inv(K_xi)*H'`.
"""
function project!(
        velocity::TwoPhaseVelocity, pressure::CellField{T},
        solver::LocalMimeticProjectionSolver{T}, dt,
        workspace::LocalMimeticProjectionWorkspace{T};
        tolerance = sqrt(eps(T)), max_iterations = 500,
    ) where {T}
    interface = solver.interface
    _validate_two_phase_velocity(velocity, interface.mesh, T)
    pressure.mesh === interface.mesh || throw(
        ArgumentError("local mimetic projection arguments must share a mesh"),
    )
    dt > zero(dt) || throw(ArgumentError("dt must be strictly positive"))
    tolerance > zero(tolerance) || throw(
        ArgumentError("tolerance must be strictly positive"),
    )
    max_iterations > 0 || throw(
        ArgumentError("max_iterations must be strictly positive"),
    )
    assembled = _assemble_local_mimetic_system!(
        solver, workspace, dt, velocity,
    )
    matrix = assembled.matrix
    right_hand_side = assembled.right_hand_side
    symmetry_error = isempty(matrix) ? zero(T) :
        maximum(abs, matrix - transpose(matrix))
    initial_residual = norm(right_hand_side)
    solution = zeros(T, length(right_hand_side))
    iterations = 0
    preconditioner = nothing
    if !isempty(solution) && initial_residual > zero(T)
        preconditioner = _local_mimetic_preconditioner(
            solver, assembled.block_preconditioner,
        )
        correction_rhs = copy(right_hand_side)
        for refinement in 1:8
            remaining = max_iterations - iterations
            remaining > 0 || break
            defects = _local_mimetic_residual_defects(
                correction_rhs, assembled, dt,
            )
            maximum(defects) <= T(tolerance) && break
            krylov = Krylov.MinresWorkspace(matrix, correction_rhs)
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
    residual = matrix * solution - right_hand_side
    final_residual = norm(residual)
    residual_defects = _local_mimetic_residual_defects(
        residual, assembled, dt,
    )

    np = length(assembled.pressure_indices)
    nx = length(assembled.active_faces)
    nl = size(assembled.constraint, 1)
    pressure_values = view(solution, 1:np)
    jump_values = view(solution, (np + 1):(np + nx))
    _copy_mimetic_cell_solution!(
        pressure, zip(assembled.pressure_indices, pressure_values),
    )
    fill!(workspace.jump, zero(T))
    workspace.jump[assembled.inactive_faces] .=
        assembled.inactive_jump_gradient
    workspace.jump[assembled.active_faces] .= jump_values

    gradients = (workspace.liquid_gradient, workspace.gas_gradient)
    local_mimetic_gradient!(
        gradients, pressure, solver, workspace,
        assembled.pressure_operator, assembled.data,
    )
    for phase in 1:2
        for index in eachindex(velocity[phase].interior)
            @inbounds velocity[phase].interior[index] -=
                T(dt) * gradients[phase].interior[index]
        end
    end

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
        @inbounds begin
            liquid_mass = assembled.data.liquid_mass[index]
            gas_mass = assembled.data.gas_mass[index]
            liquid_gradient = gradients[1].interior[index]
            gas_gradient = gradients[2].interior[index]
            final_common = (
                liquid_mass * final_liquid[index] +
                    gas_mass * final_gas[index]
            ) / (liquid_mass + gas_mass)
            open_aperture = isnothing(geometry) ? one(T) :
                geometry.aperture[index]
            orthogonality +=
                final_common * open_aperture * pressure_jump[index]
            weighted_gradient =
                liquid_mass * liquid_gradient + gas_mass * gas_gradient
            pressure_momentum[axis] -= T(dt) * pressure_jump[index]
            identity_error = max(
                identity_error, abs(
                    weighted_gradient -
                        (liquid_mass + gas_mass) *
                        workspace.pressure_gradient.interior[index]
                ),
            )
        end
    end
    all_face_dual_volumes = isnothing(interface.geometry) ?
        _uncut_face_dual_volumes(interface.mesh, T) :
        _geometry_face_dual_volumes(interface.geometry)
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
    mixture_divergence = _mimetic_strong_maximum(
        mixture_constraint, volumes,
    )
    mixture_volume_defect = T(dt) * mixture_divergence
    final_jump = final_gas[assembled.active_faces] .-
        final_liquid[assembled.active_faces]
    # Assess every resolved local row, including rows omitted from the solve
    # because rank revelation classified them as dependent.  This prevents
    # the rank filter from hiding a loss of local consistency.
    local_residual = assembled.candidate_constraint * final_jump
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
    workspace.constraint_faces = assembled.constraint_faces
    workspace.constraint_scales = assembled.constraint_scales
    return LocalMimeticProjectionStats(
        converged, iterations, initial_residual, final_residual,
        symmetry_error, mixture_divergence, mixture_volume_defect,
        normal_velocity_defect, residual_defects.tangent, T(tolerance),
        (pressure_momentum[1], pressure_momentum[2]), identity_error,
        energy_error, abs(orthogonality), np, nx,
        assembled.candidate_count, nl, assembled.dependent_count,
        assembled.underresolved, assembled.mass_deactivated,
        assembled.embedded_deactivated, preconditioner_name,
    )
end
