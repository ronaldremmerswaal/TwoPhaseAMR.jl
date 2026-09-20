"""Configuration for the three-block two-velocity pressure projection.

The unknowns are the cell pressure, the face-normal pressure-gradient jump on
mixed faces, and the cell multiplier for the aperture-normal constraint.  The
linear system is the symmetric weak form documented in the two-velocity
projection section of the manual. A direct literature reference for this
package-specific formulation is tracked in the manual's literature-coverage
TODOs.
"""
struct MimeticProjectionSolver{T, I, P, PO}
    interface::I
    density::NTuple{2, T}
    preconditioner::P
    preconditioner_options::PO
end

function MimeticProjectionSolver(
        interface::InterfaceField{T}; density = (one(T), one(T)),
        preconditioner = :cholesky,
        preconditioner_options = (;),
    ) where {T}
    rho = (T(density[1]), T(density[2]))
    all(value -> isfinite(value) && value > zero(T), rho) || throw(
        ArgumentError("phase densities must be finite and positive"),
    )
    return MimeticProjectionSolver{
        T, typeof(interface), typeof(preconditioner),
        typeof(preconditioner_options),
    }(interface, rho, preconditioner, preconditioner_options)
end

"""Reusable fields and the most recently assembled three-block system."""
mutable struct MimeticProjectionWorkspace{T, M <: BlockTreeMesh}
    pressure_gradient::FaceVelocity{T, M}
    base_gradient::FaceVelocity{T, M}
    liquid_gradient::FaceVelocity{T, M}
    gas_gradient::FaceVelocity{T, M}
    lambda::CellField{T, M}
    jump::Vector{T}
    liquid_fraction_x::FaceField{1, T, M}
    liquid_fraction_y::FaceField{2, T, M}
    matrix::SparseMatrixCSC{T, Int}
    block_preconditioner::SparseMatrixCSC{T, Int}
    right_hand_side::Vector{T}
    solution::Vector{T}
    pressure_indices::Vector{Int}
    mixed_faces::Vector{Int}
    multiplier_indices::Vector{Int}
end

function MimeticProjectionWorkspace(
        solver::MimeticProjectionSolver{T},
    ) where {T}
    mesh = solver.interface.mesh
    return MimeticProjectionWorkspace(
        FaceVelocity(mesh, T), FaceVelocity(mesh, T),
        FaceVelocity(mesh, T), FaceVelocity(mesh, T), CellField(mesh, T),
        zeros(T, length(mesh.faces)), FaceField{1}(mesh, T),
        FaceField{2}(mesh, T), spzeros(T, 0, 0), spzeros(T, 0, 0),
        T[], T[], Int[], Int[], Int[],
    )
end

"""Diagnostics for one mimetic projection."""
struct MimeticProjectionStats{T}
    converged::Bool
    iterations::Int
    initial_residual::T
    final_residual::T
    matrix_symmetry_error::T
    mixture_divergence::T
    normal_jump_divergence::T
    mixture_volume_defect::T
    normal_volume_defect::T
    tangential_defect::T
    target_defect::T
    momentum_error::NTuple{2, T}
    momentum_identity_error::T
    energy_identity_error::T
    orthogonality_error::T
    pressure_unknowns::Int
    jump_unknowns::Int
    multiplier_unknowns::Int
    preconditioner::Symbol
end

_dimensionless_residual(stats::MimeticProjectionStats) = max(
    stats.mixture_volume_defect,
    stats.normal_volume_defect,
    stats.tangential_defect,
)

function Base.show(io::IO, stats::MimeticProjectionStats)
    return print(
        io, "MimeticProjectionStats(converged=", stats.converged,
        ", iterations=", stats.iterations,
        ", initial_residual=", stats.initial_residual,
        ", final_residual=", stats.final_residual,
        ", matrix_symmetry_error=", stats.matrix_symmetry_error,
        ", mixture_divergence=", stats.mixture_divergence,
        ", normal_jump_divergence=", stats.normal_jump_divergence,
        ", mixture_volume_defect=", stats.mixture_volume_defect,
        ", normal_volume_defect=", stats.normal_volume_defect,
        ", tangential_defect=", stats.tangential_defect,
        ", target_defect=", stats.target_defect,
        ", momentum_error=", stats.momentum_error,
        ", momentum_identity_error=", stats.momentum_identity_error,
        ", energy_identity_error=", stats.energy_identity_error,
        ", orthogonality_error=", stats.orthogonality_error,
        ", unknowns=(`p`=", stats.pressure_unknowns,
        ", `xi`=", stats.jump_unknowns,
        ", `lambda`=", stats.multiplier_unknowns, ")",
        ", preconditioner=", stats.preconditioner, ")",
    )
end

function _mimetic_incidence(mesh::BlockTreeMesh, ::Type{T}) where {T}
    nx, ny = mesh.block_size
    rows = Int[]
    columns = Int[]
    values = T[]
    for (face_index, face) in pairs(mesh.faces)
        minus = _elliptic_cell_index(face.minus, nx, ny)
        plus = _elliptic_cell_index(face.plus, nx, ny)
        push!(rows, face_index, face_index)
        push!(columns, minus, plus)
        push!(values, -T(face.area), T(face.area))
    end
    return sparse(
        rows, columns, values, length(mesh.faces), ncells(mesh),
    )
end

function _mimetic_global_face_fractions!(
        result, workspace,
        interface::InterfaceField,
    )
    _phase_face_average!(
        workspace.liquid_fraction_x, interface.conditional_fraction,
        interface.geometry,
    )
    _phase_face_average!(
        workspace.liquid_fraction_y, interface.conditional_fraction,
        interface.geometry,
    )
    local_x = local_y = 0
    for (index, face) in pairs(interface.mesh.faces)
        if face.axis == 1
            local_x += 1
            @inbounds result[index] =
                workspace.liquid_fraction_x.interior[local_x]
        else
            local_y += 1
            @inbounds result[index] =
                workspace.liquid_fraction_y.interior[local_y]
        end
    end
    return result
end

function _mimetic_active_cells(interface::InterfaceField)
    geometry = interface.geometry
    return isnothing(geometry) ? trues(ncells(interface.mesh)) :
        copy(geometry.active_cells)
end

function _gauge_free_component_partition(
        mesh::BlockTreeMesh, eligible::AbstractVector{Bool}, face_weights,
    )
    nx, ny = mesh.block_size
    adjacency = [Int[] for _ in 1:ncells(mesh)]
    for (index, face) in pairs(mesh.faces)
        iszero(@inbounds face_weights[index]) && continue
        minus = _elliptic_cell_index(face.minus, nx, ny)
        plus = _elliptic_cell_index(face.plus, nx, ny)
        eligible[minus] && eligible[plus] || continue
        push!(adjacency[minus], plus)
        push!(adjacency[plus], minus)
    end
    retained = Int[]
    anchor_for_retained = Int[]
    visited = falses(length(eligible))
    stack = Int[]
    for root in eachindex(eligible)
        eligible[root] && !visited[root] || continue
        empty!(stack)
        push!(stack, root)
        visited[root] = true
        component = Int[]
        while !isempty(stack)
            cell = pop!(stack)
            push!(component, cell)
            for neighbour in adjacency[cell]
                visited[neighbour] && continue
                visited[neighbour] = true
                push!(stack, neighbour)
            end
        end
        length(component) > 1 || continue
        anchor = minimum(component)
        for cell in component
            cell == anchor && continue
            push!(retained, cell)
            push!(anchor_for_retained, anchor)
        end
    end
    permutation = sortperm(retained)
    return retained[permutation], anchor_for_retained[permutation]
end

function _gauge_free_component_rows(
        mesh::BlockTreeMesh, eligible::AbstractVector{Bool}, face_weights,
    )
    retained, = _gauge_free_component_partition(
        mesh, eligible, face_weights,
    )
    return retained
end

function _mimetic_face_data(
        solver, workspace,
    )
    T = eltype(solver.interface)
    interface = solver.interface
    reconstruct!(interface)
    number_of_faces = length(interface.mesh.faces)
    liquid_fraction = zeros(T, number_of_faces)
    _mimetic_global_face_fractions!(liquid_fraction, workspace, interface)
    liquid_mass = solver.density[1] .* liquid_fraction
    gas_mass = solver.density[2] .* (one(T) .- liquid_fraction)
    total_mass = liquid_mass .+ gas_mass
    liquid_aperture = zeros(T, number_of_faces)
    gas_aperture = zeros(T, number_of_faces)
    for index in 1:number_of_faces
        liquid, gas = _phase_face_apertures(
            interface, interface.geometry, index,
        )
        @inbounds begin
            liquid_aperture[index] = liquid
            gas_aperture[index] = gas
        end
    end
    zeta_liquid = .-gas_mass
    zeta_gas = liquid_mass
    aperture_zeta = liquid_aperture .* zeta_liquid .+
        gas_aperture .* zeta_gas
    coupling = aperture_zeta ./ total_mass
    relative_mass = liquid_mass .* gas_mass ./ total_mass
    aperture_jump = gas_aperture .- liquid_aperture
    floor = interface.fraction_floor
    # For an exactly half-filled open face the normal-constraint coefficient
    # vanishes.  PLIC clipping can leave roundoff-sized remnants here.  If
    # retained, those remnants create false graph connections when choosing
    # the multiplier gauge and make the KKT system numerically singular.
    # Remove roundoff only; this threshold is deliberately independent of the
    # physical volume-fraction floor.
    for index in eachindex(aperture_jump)
        open_aperture = liquid_aperture[index] + gas_aperture[index]
        threshold = T(64) * eps(T) * max(open_aperture, one(T))
        abs(aperture_jump[index]) <= threshold &&
            (aperture_jump[index] = zero(T))
    end
    mixed = findall(eachindex(total_mass)) do index
        liquid_fraction[index] > floor &&
            liquid_fraction[index] < one(T) - floor &&
            liquid_aperture[index] + gas_aperture[index] > zero(T)
    end
    return (;
        liquid_fraction, liquid_mass, gas_mass, total_mass,
        liquid_aperture, gas_aperture, zeta_liquid, zeta_gas,
        coupling, relative_mass, aperture_jump, mixed,
    )
end

function _mimetic_pressure_operator(
        solver,
    )
    interface = solver.interface
    isnothing(interface.geometry) &&
        return PoissonOperator(interface; density = solver.density)

    # The two-velocity projection minimizes the correction in the phase
    # kinetic norm of the *open staggered dual volume*.  Its common pressure
    # correction therefore has the coefficient A/rho, where A is the total
    # open aperture.  The one-velocity symmetry-preserving cut-cell operator
    # uses A^2/rho and is intentionally a different discretization.
    geometry = interface.geometry
    density = mixture_density(interface, solver.density, geometry)
    coefficient = geometry.aperture ./ density
    T = eltype(interface)
    return EllipticOperator(
        interface.mesh; reaction = 0, diffusion = 1,
        face_coefficient = coefficient,
        gradient_operator = GradientOperator(geometry),
        volumes = _geometry_cell_volumes(geometry),
        active_cells = geometry.active_cells, T,
    )
end

function _mimetic_boundary_constraints(
        interface::InterfaceField{T}, velocity::TwoPhaseVelocity,
    ) where {T}
    mesh = interface.mesh
    nx, ny = mesh.block_size
    mixture = zeros(T, ncells(mesh))
    normal_jump = zeros(T, ncells(mesh))
    for (index, face) in pairs(mesh.boundary_faces)
        liquid_aperture, gas_aperture =
            _phase_boundary_face_apertures(interface, index)
        cell = _elliptic_cell_index(face.cell, nx, ny)
        factor = T(face.side) * T(face.area)
        liquid = velocity[1].boundary[index]
        gas = velocity[2].boundary[index]
        @inbounds begin
            mixture[cell] += factor * (
                liquid_aperture * liquid + gas_aperture * gas
            )
            normal_jump[cell] += factor *
                (gas_aperture - liquid_aperture) * (gas - liquid)
        end
    end
    return mixture, normal_jump
end

function _mimetic_block_preconditioner(
        pressure_block, relative_diagonal, normal_block, ::Type{T},
    ) where {T}
    np = size(pressure_block, 1)
    nx = length(relative_diagonal)
    nl = size(normal_block, 1)
    zero_px = spzeros(T, np, nx)
    zero_pl = spzeros(T, np, nl)
    zero_xl = spzeros(T, nx, nl)
    jump_block = spdiagm(0 => relative_diagonal)
    multiplier_block = nl == 0 ? spzeros(T, 0, 0) :
        normal_block * spdiagm(0 => inv.(relative_diagonal)) *
        transpose(normal_block)
    return [
        pressure_block zero_px zero_pl
        transpose(zero_px) jump_block zero_xl
        transpose(zero_pl) transpose(zero_xl) multiplier_block
    ]
end

function _assemble_mimetic_system!(
        solver::MimeticProjectionSolver{T},
        workspace::MimeticProjectionWorkspace{T}, dt,
        velocity::TwoPhaseVelocity,
    ) where {T}
    interface = solver.interface
    mesh = interface.mesh
    data = _mimetic_face_data(solver, workspace)
    pressure_operator = _mimetic_pressure_operator(solver)
    incidence = _mimetic_incidence(mesh, T)
    active = _mimetic_active_cells(interface)
    open_weights = isnothing(interface.geometry) ? ones(T, length(mesh.faces)) :
        copy(interface.geometry.aperture)
    pressure_indices = _gauge_free_component_rows(
        mesh, active, open_weights,
    )
    mixed_faces = data.mixed
    mixed_count = length(mixed_faces)

    jump_weights = zeros(T, length(mesh.faces))
    jump_weights[mixed_faces] .= data.aperture_jump[mixed_faces]
    multiplier_eligible = falses(ncells(mesh))
    nx, ny = mesh.block_size
    for face_index in mixed_faces
        iszero(jump_weights[face_index]) && continue
        face = mesh.faces[face_index]
        minus = _elliptic_cell_index(face.minus, nx, ny)
        plus = _elliptic_cell_index(face.plus, nx, ny)
        multiplier_eligible[minus] = active[minus]
        multiplier_eligible[plus] = active[plus]
    end
    multiplier_indices = _gauge_free_component_rows(
        mesh, multiplier_eligible, jump_weights,
    )

    pressure_positive = pressure_operator.stiffness[
        pressure_indices, pressure_indices,
    ]
    incidence_pressure = incidence[:, pressure_indices]
    coupling = data.coupling[mixed_faces]
    pressure_jump = -transpose(incidence_pressure[mixed_faces, :]) *
        spdiagm(0 => coupling)
    all_face_dual_volumes = isnothing(interface.geometry) ?
        _uncut_face_dual_volumes(mesh, T) :
        _geometry_face_dual_volumes(interface.geometry)
    face_dual_volumes = all_face_dual_volumes[mixed_faces]
    relative_diagonal =
        data.relative_mass[mixed_faces] .* face_dual_volumes
    all(>(zero(T)), relative_diagonal) || throw(
        ArgumentError("retained jump faces require positive phase masses"),
    )
    normal_block = isempty(multiplier_indices) ?
        spzeros(T, 0, mixed_count) :
        transpose(incidence[:, multiplier_indices][mixed_faces, :]) *
        spdiagm(0 => data.aperture_jump[mixed_faces])

    zero_pl = spzeros(T, length(pressure_indices), length(multiplier_indices))
    matrix = [
        -pressure_positive pressure_jump zero_pl
        transpose(pressure_jump) spdiagm(0 => relative_diagonal) transpose(normal_block)
        transpose(zero_pl) normal_block spzeros(T, length(multiplier_indices), length(multiplier_indices))
    ]
    block_preconditioner = _mimetic_block_preconditioner(
        pressure_positive, relative_diagonal, normal_block, T,
    )

    mixture = data.liquid_aperture .* velocity[1].interior .+
        data.gas_aperture .* velocity[2].interior
    jump_velocity = velocity[2].interior .- velocity[1].interior
    boundary_mixture, boundary_normal =
        _mimetic_boundary_constraints(interface, velocity)
    pressure_rhs = (
        -transpose(incidence_pressure) * mixture .+
            boundary_mixture[pressure_indices]
    ) ./ T(dt)
    multiplier_rhs = isempty(multiplier_indices) ? T[] : (
            normal_block * jump_velocity[mixed_faces] .-
            boundary_normal[multiplier_indices]
        ) ./ T(dt)
    right_hand_side = vcat(
        pressure_rhs, zeros(T, mixed_count), multiplier_rhs,
    )
    return (;
        data, pressure_operator, incidence, matrix, block_preconditioner,
        right_hand_side, pressure_indices, mixed_faces,
        multiplier_indices, normal_block, relative_diagonal,
        face_dual_volumes,
    )
end

function _mimetic_residual_defects(residual, assembled, dt)
    T = eltype(residual)
    np = length(assembled.pressure_indices)
    nx = length(assembled.mixed_faces)
    nl = length(assembled.multiplier_indices)
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
        row = np + local_index
        face = @inbounds assembled.mixed_faces[local_index]
        dual_volume = @inbounds assembled.face_dual_volumes[local_index]
        # The face equation is a kinetic weak balance.  Scale its integrated
        # residual by the total staggered mass, as in the pressure correction
        # norm, rather than by the relative phase mass K_xi.  K_xi vanishes
        # with a disappearing phase; dividing by it turns roundoff in an
        # otherwise converged KKT solve into an artificial solver failure.
        # Accuracy of the weakly controlled phase velocity is assessed
        # separately by the rotation refinement test.
        scale = @inbounds assembled.data.total_mass[face] * dual_volume *
            sqrt(dual_volume)
        @inbounds tangent = max(tangent, abs(residual[row]) / scale)
    end
    normal = zero(T)
    for local_index in 1:nl
        row = np + nx + local_index
        cell = @inbounds assembled.multiplier_indices[local_index]
        @inbounds normal = max(
            normal, abs(residual[row]) / volumes[cell],
        )
    end
    scale = T(dt)^2
    return (
        pressure = scale * pressure,
        tangent = scale * tangent,
        normal = scale * normal,
    )
end

function _mimetic_preconditioner(
        solver::MimeticProjectionSolver, block_preconditioner,
    )
    specification = solver.preconditioner
    specification === nothing && return nothing
    if specification === :cholesky
        isempty(solver.preconditioner_options) || throw(
            ArgumentError(
                "preconditioner_options are not used by exact Cholesky",
            )
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

function _copy_mimetic_cell_solution!(field::CellField{T}, values) where {T}
    for block in eachindex(field.blocks)
        fill!(interior(field, block), zero(T))
    end
    references = _cell_references(field.mesh)
    for (index, value) in values
        _set!(field, T(value), references[index])
    end
    fill_halo!(field)
    return field
end

function _mimetic_weighted_gradient!(
        destination::FaceVelocity, pressure::CellField,
        operator::EllipticOperator, work::EllipticWorkspace,
        geometry,
    )
    if isnothing(geometry)
        weighted_gradient!(destination, pressure, operator, work)
    else
        _cut_cell_weighted_gradient!(
            destination, pressure, operator, work, geometry,
        )
    end
    return destination
end

"""Reconstruct both phase pressure gradients from the latest solve."""
function mimetic_gradient!(
        destination::TwoPhaseVelocity, pressure::CellField,
        solver::MimeticProjectionSolver{T},
        workspace::MimeticProjectionWorkspace{T}, pressure_operator,
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

function mimetic_gradient!(
        destination::TwoPhaseVelocity, pressure::CellField,
        solver::MimeticProjectionSolver,
        workspace::MimeticProjectionWorkspace,
    )
    operator = _mimetic_pressure_operator(solver)
    data = _mimetic_face_data(solver, workspace)
    return mimetic_gradient!(
        destination, pressure, solver, workspace, operator, data,
    )
end

function _mimetic_integrated_constraints(
        incidence, data, interface, velocity, mixed_faces,
    )
    liquid = velocity[1].interior
    gas = velocity[2].interior
    mixture = data.liquid_aperture .* liquid .+
        data.gas_aperture .* gas
    # The face jump is a degree of freedom only where both staggered phase
    # masses are present.  As in the derivation, extend it by zero elsewhere;
    # inactive-phase storage must never enter the physical constraint.
    jump_flux = zeros(eltype(liquid), length(liquid))
    jump_flux[mixed_faces] .= data.aperture_jump[mixed_faces] .*
        (gas[mixed_faces] .- liquid[mixed_faces])
    boundary_mixture, boundary_normal =
        _mimetic_boundary_constraints(interface, velocity)
    return -transpose(incidence) * mixture .+ boundary_mixture,
        -transpose(incidence) * jump_flux .+ boundary_normal
end

function _mimetic_strong_maximum(values, volumes)
    result = zero(eltype(values))
    for index in eachindex(values)
        volumes[index] > 0 || continue
        @inbounds result = max(result, abs(values[index]) / volumes[index])
    end
    return result
end

"""
    project!(velocity, pressure, solver, dt, workspace; tolerance, max_iterations)

Apply the symmetric three-block mimetic projection to separate liquid and gas
velocities. The method enforces mixture incompressibility and the centred
aperture approximation of normal-velocity continuity while retaining
tangential slip.
"""
function project!(
        velocity::TwoPhaseVelocity, pressure::CellField{T},
        solver::MimeticProjectionSolver{T}, dt,
        workspace::MimeticProjectionWorkspace{T};
        tolerance = sqrt(eps(T)), max_iterations = 500,
    ) where {T}
    interface = solver.interface
    _validate_two_phase_velocity(velocity, interface.mesh, T)
    pressure.mesh === interface.mesh === workspace.lambda.mesh || throw(
        ArgumentError("mimetic projection arguments must share a mesh"),
    )
    dt > zero(dt) || throw(ArgumentError("dt must be strictly positive"))
    tolerance > zero(tolerance) || throw(
        ArgumentError("tolerance must be strictly positive"),
    )
    max_iterations > 0 || throw(
        ArgumentError("max_iterations must be strictly positive"),
    )
    assembled = _assemble_mimetic_system!(solver, workspace, dt, velocity)
    matrix = assembled.matrix
    right_hand_side = assembled.right_hand_side
    symmetry_error = isempty(matrix) ? zero(T) :
        maximum(abs, matrix - transpose(matrix))
    initial_residual = norm(right_hand_side)
    solution = zeros(T, length(right_hand_side))
    iterations = 0
    preconditioner = nothing
    if !isempty(solution) && initial_residual > zero(T)
        preconditioner = _mimetic_preconditioner(
            solver, assembled.block_preconditioner,
        )
        correction_rhs = copy(right_hand_side)
        # MINRES stops on a recurrence-based estimate.  With a block
        # preconditioner that estimate can be optimistic for the unscaled KKT
        # residual, in particular at large density ratio.  A few residual
        # correction solves make the user tolerance refer to the assembled
        # physical equations rather than the internal estimate.
        for refinement in 1:8
            remaining = max_iterations - iterations
            remaining > 0 || break
            defects = _mimetic_residual_defects(
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
    residual_defects = _mimetic_residual_defects(residual, assembled, dt)

    np = length(assembled.pressure_indices)
    nx = length(assembled.mixed_faces)
    nl = length(assembled.multiplier_indices)
    pressure_values = view(solution, 1:np)
    jump_values = view(solution, (np + 1):(np + nx))
    lambda_values = view(solution, (np + nx + 1):(np + nx + nl))
    _copy_mimetic_cell_solution!(
        pressure, zip(assembled.pressure_indices, pressure_values),
    )
    _copy_mimetic_cell_solution!(
        workspace.lambda,
        zip(assembled.multiplier_indices, lambda_values),
    )
    fill!(workspace.jump, zero(T))
    workspace.jump[assembled.mixed_faces] .= jump_values

    gradients = (workspace.liquid_gradient, workspace.gas_gradient)
    mimetic_gradient!(
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
            # This is the exact face-Riesz pairing.  In particular, it does
            # not replace the AMR Riesz map by a diagonal dual-volume norm.
            orthogonality +=
                final_common * open_aperture * pressure_jump[index]
            weighted_gradient =
                liquid_mass * liquid_gradient + gas_mass * gas_gradient
            pressure_momentum[axis] -=
                T(dt) * pressure_jump[index]
            identity_error = max(
                identity_error, abs(
                    weighted_gradient -
                        (liquid_mass + gas_mass) *
                        workspace.pressure_gradient.interior[index]
                ),
            )
        end
    end
    for (local_index, face_index) in pairs(assembled.mixed_faces)
        @inbounds orthogonality +=
            (final_gas[face_index] - final_liquid[face_index]) *
            assembled.relative_diagonal[local_index] *
            workspace.jump[face_index]
    end
    # If u*** = u + dt*g, the Pythagorean defect is exactly
    # 2dt*<u,g>.  The pairing above is the same Riesz product used to form
    # the symmetric three-block equations.
    energy_error = abs(2T(dt) * orthogonality)
    mixture_constraint, normal_constraint = _mimetic_integrated_constraints(
        assembled.incidence, assembled.data, interface, velocity,
        assembled.mixed_faces,
    )
    volumes = assembled.pressure_operator.volumes
    mixture_divergence = _mimetic_strong_maximum(
        mixture_constraint, volumes,
    )
    normal_jump_divergence = _mimetic_strong_maximum(
        normal_constraint, volumes,
    )
    mixture_volume_defect = T(dt) * mixture_divergence
    normal_volume_defect = T(dt) * normal_jump_divergence
    converged = mixture_volume_defect <= T(tolerance) &&
        normal_volume_defect <= T(tolerance) &&
        residual_defects.tangent <= T(tolerance)
    preconditioner_name = isnothing(preconditioner) ? :none :
        _preconditioner_name(preconditioner)

    workspace.matrix = matrix
    workspace.block_preconditioner = assembled.block_preconditioner
    workspace.right_hand_side = right_hand_side
    workspace.solution = solution
    workspace.pressure_indices = assembled.pressure_indices
    workspace.mixed_faces = assembled.mixed_faces
    workspace.multiplier_indices = assembled.multiplier_indices
    return MimeticProjectionStats(
        converged, iterations, initial_residual, final_residual,
        symmetry_error, mixture_divergence, normal_jump_divergence,
        mixture_volume_defect, normal_volume_defect,
        residual_defects.tangent, T(tolerance),
        (pressure_momentum[1], pressure_momentum[2]),
        identity_error,
        energy_error,
        abs(orthogonality), np, nx, nl, preconditioner_name,
    )
end
