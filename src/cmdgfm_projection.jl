"""Pressure-projection policy for the symmetric three-block formulation."""
abstract type AbstractTwoVelocityPressureProjection end

struct MimeticProjection <: AbstractTwoVelocityPressureProjection end

"""Pressure-projection policy for the conservative multidimensional GFM."""
struct CMDGFMProjection{P, O} <: AbstractTwoVelocityPressureProjection
    preconditioner::P
    preconditioner_options::O
end

CMDGFMProjection(;
    preconditioner = :block, preconditioner_options = (;),
) = CMDGFMProjection(preconditioner, preconditioner_options)

"""One local CMDGFM normal-jump equation associated with a mixed face."""
struct CMDGFMJumpStencil{T}
    face::Int
    transverse_faces::Vector{Int}
    transverse_weights::Vector{T}
    ratio::T
    gas_inference_face::Int
    liquid_inference_face::Int
    beta::T
    resolved::Bool
    full_remove::Bool
end

"""Configuration for the nonsymmetric two-block CMDGFM projection."""
struct CMDGFMProjectionSolver{T, I, P, PO}
    interface::I
    density::NTuple{2, T}
    preconditioner::P
    preconditioner_options::PO
end

function CMDGFMProjectionSolver(
        interface::InterfaceField{T}; density = (one(T), one(T)),
        preconditioner = :block,
        preconditioner_options = (;),
    ) where {T}
    rho = (T(density[1]), T(density[2]))
    all(value -> isfinite(value) && value > zero(T), rho) || throw(
        ArgumentError("phase densities must be finite and positive"),
    )
    preconditioner in (:block, :ilu, :lu, nothing) || throw(
        ArgumentError(
            "CMDGFM preconditioner must be :block, :ilu, :lu, or nothing",
        ),
    )
    return CMDGFMProjectionSolver{
        T, typeof(interface), typeof(preconditioner),
        typeof(preconditioner_options),
    }(interface, rho, preconditioner, preconditioner_options)
end

"""Reusable fields and the most recently assembled CMDGFM system."""
mutable struct CMDGFMProjectionWorkspace{T, M <: BlockTreeMesh}
    pressure_gradient::FaceVelocity{T, M}
    base_gradient::FaceVelocity{T, M}
    liquid_gradient::FaceVelocity{T, M}
    gas_gradient::FaceVelocity{T, M}
    jump::Vector{T}
    liquid_fraction_x::FaceField{1, T, M}
    liquid_fraction_y::FaceField{2, T, M}
    matrix::SparseMatrixCSC{T, Int}
    jump_block::SparseMatrixCSC{T, Int}
    right_hand_side::Vector{T}
    solution::Vector{T}
    pressure_indices::Vector{Int}
    mixed_faces::Vector{Int}
    stencils::Vector{CMDGFMJumpStencil{T}}
end

function CMDGFMProjectionWorkspace(
        solver::CMDGFMProjectionSolver{T},
    ) where {T}
    mesh = solver.interface.mesh
    return CMDGFMProjectionWorkspace(
        FaceVelocity(mesh, T), FaceVelocity(mesh, T),
        FaceVelocity(mesh, T), FaceVelocity(mesh, T),
        zeros(T, length(mesh.faces)), FaceField{1}(mesh, T),
        FaceField{2}(mesh, T), spzeros(T, 0, 0), spzeros(T, 0, 0),
        T[], T[], Int[], Int[], CMDGFMJumpStencil{T}[],
    )
end

"""Diagnostics for one conservative multidimensional GFM projection."""
struct CMDGFMProjectionStats{T}
    converged::Bool
    iterations::Int
    initial_residual::T
    final_residual::T
    matrix_asymmetry::T
    mixture_divergence::T
    mixture_volume_defect::T
    normal_velocity_defect::T
    target_defect::T
    momentum_error::NTuple{2, T}
    momentum_identity_error::T
    minimum_jump_diagonal_margin::T
    pressure_unknowns::Int
    jump_unknowns::Int
    underresolved_jump_rows::Int
    preconditioner::Symbol
end

function Base.show(io::IO, stats::CMDGFMProjectionStats)
    return print(
        io, "CMDGFMProjectionStats(converged=", stats.converged,
        ", iterations=", stats.iterations,
        ", initial_residual=", stats.initial_residual,
        ", final_residual=", stats.final_residual,
        ", matrix_asymmetry=", stats.matrix_asymmetry,
        ", mixture_divergence=", stats.mixture_divergence,
        ", mixture_volume_defect=", stats.mixture_volume_defect,
        ", normal_velocity_defect=", stats.normal_velocity_defect,
        ", target_defect=", stats.target_defect,
        ", momentum_error=", stats.momentum_error,
        ", momentum_identity_error=", stats.momentum_identity_error,
        ", minimum_jump_diagonal_margin=",
        stats.minimum_jump_diagonal_margin,
        ", unknowns=(`p`=", stats.pressure_unknowns,
        ", `xi`=", stats.jump_unknowns, ")",
        ", underresolved_jump_rows=", stats.underresolved_jump_rows,
        ", preconditioner=", stats.preconditioner, ")",
    )
end

"""Lower block preconditioner with exact pressure and jump diagonal blocks."""
struct CMDGFMBlockPreconditioner{T, F, J, C}
    scalar_type::Type{T}
    pressure_factor::F
    jump_factor::J
    jump_pressure::C
    pressure_size::Int
    dimensions::Tuple{Int, Int}
end

Base.size(preconditioner::CMDGFMBlockPreconditioner) = preconditioner.dimensions
Base.eltype(::CMDGFMBlockPreconditioner{T}) where {T} = T
_preconditioner_name(::CMDGFMBlockPreconditioner) = :cmdgfm_block

function ldiv!(
        destination::AbstractVector{T},
        preconditioner::CMDGFMBlockPreconditioner{T},
        right_hand_side::AbstractVector{T},
    ) where {T}
    np = preconditioner.pressure_size
    if np > 0
        pressure_destination = view(destination, 1:np)
        copyto!(
            pressure_destination,
            preconditioner.pressure_factor \ view(right_hand_side, 1:np),
        )
    end
    nx = length(right_hand_side) - np
    if nx > 0
        jump_rhs = collect(view(right_hand_side, (np + 1):(np + nx)))
        if np > 0
            mul!(
                jump_rhs, preconditioner.jump_pressure,
                view(destination, 1:np), -one(T), one(T),
            )
        end
        copyto!(
            view(destination, (np + 1):(np + nx)),
            preconditioner.jump_factor \ jump_rhs,
        )
    end
    return destination
end

function _cmdgfm_incident_faces(mesh::BlockTreeMesh)
    nx, ny = mesh.block_size
    result = [Int[] for _ in 1:ncells(mesh)]
    for (index, face) in pairs(mesh.faces)
        minus = _elliptic_cell_index(face.minus, nx, ny)
        plus = _elliptic_cell_index(face.plus, nx, ny)
        push!(result[minus], index)
        push!(result[plus], index)
    end
    return result
end

@inline function _cmdgfm_physical_cell_normal(
        interface::InterfaceField{T}, reference::CellRef,
    ) where {T}
    spacing = interface.mesh.blocks[reference.block].spacing
    nx = _get(interface.normal_x, reference) / spacing[1]
    ny = _get(interface.normal_y, reference) / spacing[2]
    magnitude = hypot(nx, ny)
    magnitude > sqrt(eps(T)) || return (zero(T), zero(T))
    return (nx / magnitude, ny / magnitude)
end

function _cmdgfm_face_normal(
        interface::InterfaceField{T}, face::InteriorFace,
    ) where {T}
    minus = _cmdgfm_physical_cell_normal(interface, face.minus)
    plus = _cmdgfm_physical_cell_normal(interface, face.plus)
    nx = minus[1] + plus[1]
    ny = minus[2] + plus[2]
    magnitude = hypot(nx, ny)
    if magnitude <= sqrt(eps(T))
        # An interface aligned with this face can have pure cells on both
        # sides. The fraction difference then supplies its oriented normal.
        difference = _conditional_fraction(interface, face.minus) -
            _conditional_fraction(interface, face.plus)
        nx = face.axis == 1 ? difference : zero(T)
        ny = face.axis == 2 ? difference : zero(T)
        magnitude = hypot(nx, ny)
    end
    magnitude > sqrt(eps(T)) || return nothing
    return (nx / magnitude, ny / magnitude)
end

function _cmdgfm_unique_incident_candidates(
        incident, face::InteriorFace, face_index, candidate_axis, axis_of,
    )
    result = Int[]
    for cell in (face.minus, face.plus)
        # `_elliptic_cell_index` needs the block dimensions, but the incident
        # table is indexed by global cell number. Recover it from any face is
        # deliberately avoided; callers pass an incident accessor closure.
        for candidate in incident(cell)
            candidate == face_index && continue
            axis_of(candidate) == candidate_axis || continue
            candidate in result || push!(result, candidate)
        end
    end
    return result
end

function _cmdgfm_vertex_candidates(
        patches, mesh::BlockTreeMesh, face::InteriorFace,
        face_index::Int, candidate_axis::Int,
    )
    result = Int[]
    for endpoint in 1:2
        patch = get(patches, _vertex_key(mesh, face, endpoint), nothing)
        isnothing(patch) && continue
        for candidate in patch
            candidate.index == face_index && continue
            candidate.axis == candidate_axis || continue
            candidate.index in result || push!(result, candidate.index)
        end
    end
    sort!(result)
    return result
end

@inline function _cmdgfm_aperture_mixed(data, index, floor)
    return data.liquid_aperture[index] > floor &&
        data.gas_aperture[index] > floor
end

function _cmdgfm_jump_stencils(
        solver::CMDGFMProjectionSolver{T}, data,
    ) where {T}
    interface = solver.interface
    mesh = interface.mesh
    nx, ny = mesh.block_size
    incident_faces = _cmdgfm_incident_faces(mesh)
    vertex_patches = _vertex_patches(mesh, T)
    incident(reference) = incident_faces[
        _elliptic_cell_index(reference, nx, ny),
    ]
    face_axis(index) = mesh.faces[index].axis
    floor = interface.fraction_floor
    stencils = CMDGFMJumpStencil{T}[]
    for face_index in data.mixed
        face = mesh.faces[face_index]
        open_aperture = data.liquid_aperture[face_index] +
            data.gas_aperture[face_index]
        if open_aperture <= floor
            push!(
                stencils,
                CMDGFMJumpStencil(
                    face_index, Int[], T[], zero(T), 0, 0, one(T),
                    false, false,
                ),
            )
            continue
        end
        if !_cmdgfm_aperture_mixed(data, face_index, floor)
            normal_neighbours = _cmdgfm_unique_incident_candidates(
                incident, face, face_index, face.axis, face_axis,
            )
            if any(
                    index -> _cmdgfm_aperture_mixed(data, index, floor),
                    normal_neighbours,
                )
                # The Fortran reference suppresses a fraction-mixed face when
                # its phase aperture is closed but a neighbouring pressure
                # gradient jump is active in the divergence operator.
                push!(
                    stencils,
                    CMDGFMJumpStencil(
                        face_index, Int[], T[], zero(T), 0, 0, one(T),
                        false, false,
                    ),
                )
                continue
            end
        end
        normal = _cmdgfm_face_normal(interface, face)
        if isnothing(normal) || abs(normal[face.axis]) <= sqrt(eps(T))
            push!(
                stencils,
                CMDGFMJumpStencil(
                    face_index, Int[], T[], zero(T), 0, 0, one(T),
                    false, false,
                ),
            )
            continue
        end
        tangent = 3 - face.axis
        ratio = normal[tangent] / normal[face.axis]
        if abs(normal[face.axis]) >= one(T) - T(64) * eps(T)
            # With an axis-aligned normal the Fortran implementation bypasses
            # interpolation and removes the face-normal velocity jump exactly.
            push!(
                stencils,
                CMDGFMJumpStencil(
                    face_index, Int[], T[], ratio, 0, 0, one(T), true, true,
                ),
            )
            continue
        end
        transverse = _cmdgfm_vertex_candidates(
            vertex_patches, mesh, face, face_index, tangent,
        )
        filter!(transverse) do index
            fraction = data.liquid_fraction[index]
            fraction > floor && fraction < one(T) - floor &&
                data.liquid_aperture[index] + data.gas_aperture[index] > floor
        end
        raw_weights = [
            data.liquid_fraction[index] *
                (one(T) - data.liquid_fraction[index])
                for index in transverse
        ]
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
        required_transverse = touches_boundary || crosses_level ? 1 : 2
        if length(transverse) < required_transverse ||
                weight_sum <= zero(T)
            push!(
                stencils,
                CMDGFMJumpStencil(
                    face_index, Int[], T[], zero(T), 0, 0, one(T),
                    false, false,
                ),
            )
            continue
        end
        weights = raw_weights ./ weight_sum
        ratio_sum = abs(ratio)
        beta = one(T)
        gas_face = liquid_face = 0
        if ratio_sum > one(T)
            parallel = _cmdgfm_vertex_candidates(
                vertex_patches, mesh, face, face_index, face.axis,
            )
            if length(parallel) < 2
                push!(
                    stencils,
                    CMDGFMJumpStencil(
                        face_index, Int[], T[], zero(T), 0, 0, one(T),
                        false, false,
                    ),
                )
                continue
            end
            sort!(
                parallel; by = index -> (
                    data.liquid_mass[index] / data.total_mass[index], index,
                )
            )
            gas_face = first(parallel)
            liquid_face = last(parallel)
            if data.gas_mass[gas_face] < data.liquid_mass[gas_face] ||
                    data.liquid_mass[liquid_face] < data.gas_mass[liquid_face]
                # `include_infer_contribution` in the reference requires the
                # gas inference face to be gas dominated and the liquid face
                # to be liquid dominated. Without both, the row is explicitly
                # classified as underresolved and xi is extended by zero.
                push!(
                    stencils,
                    CMDGFMJumpStencil(
                        face_index, Int[], T[], zero(T), 0, 0, one(T),
                        false, false,
                    ),
                )
                continue
            end
            gamma_liquid_gas =
                data.liquid_mass[gas_face] / data.total_mass[gas_face]
            gamma_gas_liquid =
                data.gas_mass[liquid_face] / data.total_mass[liquid_face]
            inferred_sum = gamma_liquid_gas + gamma_gas_liquid
            denominator = ratio_sum - inferred_sum
            denominator > zero(T) || error(
                "CMDGFM beta denominator is not positive on face $face_index",
            )
            beta = clamp(
                (one(T) - inferred_sum) / denominator, zero(T), one(T),
            )
        end
        push!(
            stencils,
            CMDGFMJumpStencil(
                face_index, transverse, weights, ratio,
                gas_face, liquid_face, beta, true, false,
            ),
        )
    end
    return stencils
end

function _cmdgfm_metric_matrix(metric::DiagonalInverseFaceMetric{T}) where {T}
    return spdiagm(0 => metric.diagonal)
end

_cmdgfm_metric_matrix(metric::SparseInverseFaceMetric) = metric.matrix

function _cmdgfm_common_gradient_matrix(
        operator::EllipticOperator{T}, incidence,
        geometry,
    ) where {T}
    scale = operator.square_root_face_coefficient
    matrix = spdiagm(0 => scale) *
        _cmdgfm_metric_matrix(operator.gradient_operator.inverse_metric) *
        spdiagm(0 => scale) * incidence
    if !isnothing(geometry)
        inverse_aperture = map(geometry.aperture) do aperture
            aperture > zero(T) ? inv(aperture) : zero(T)
        end
        matrix = spdiagm(0 => inverse_aperture) * matrix
    end
    dropzeros!(matrix)
    return matrix
end

function _cmdgfm_jump_blocks(
        stencils, mixed_faces, data, common_gradient, pressure_indices,
        velocity::TwoPhaseVelocity, dt, ::Type{T},
    ) where {T}
    jump_index = Dict(face => index for (index, face) in pairs(mixed_faces))
    rows = Int[]
    columns = Int[]
    values = T[]
    pressure_rows = Int[]
    pressure_columns = Int[]
    pressure_values = T[]
    right_hand_side = zeros(T, length(stencils))
    underresolved = 0
    liquid = velocity[1].interior
    gas = velocity[2].interior
    pressure_lookup = Dict(
        cell => local_index for (local_index, cell) in pairs(pressure_indices)
    )
    gradient_transpose = copy(transpose(common_gradient))
    gradient_cells = rowvals(gradient_transpose)
    for (row, stencil) in pairs(stencils)
        face = stencil.face
        push!(rows, row)
        push!(columns, jump_index[face])
        push!(values, one(T))
        if !stencil.resolved
            underresolved += 1
            @inbounds right_hand_side[row] =
                (gas[face] - liquid[face]) / T(dt)
            continue
        end
        rhs = @inbounds(gas[face] - liquid[face]) / T(dt)
        transverse_factor = stencil.beta * stencil.ratio
        for (candidate, weight) in zip(
                stencil.transverse_faces, stencil.transverse_weights,
            )
            coefficient = transverse_factor * weight
            local_index = get(jump_index, candidate, 0)
            iszero(local_index) || begin
                push!(rows, row)
                push!(columns, local_index)
                push!(values, coefficient)
            end
            @inbounds rhs += coefficient *
                (gas[candidate] - liquid[candidate]) / T(dt)
        end
        if stencil.gas_inference_face > 0
            gas_face = stencil.gas_inference_face
            liquid_face = stencil.liquid_inference_face
            inference_factor = stencil.beta - one(T)
            gamma_liquid_gas =
                data.liquid_mass[gas_face] / data.total_mass[gas_face]
            gamma_gas_liquid =
                data.gas_mass[liquid_face] / data.total_mass[liquid_face]
            for (candidate, gamma) in (
                    (gas_face, gamma_liquid_gas),
                    (liquid_face, gamma_gas_liquid),
                )
                local_index = get(jump_index, candidate, 0)
                iszero(local_index) || begin
                    push!(rows, row)
                    push!(columns, local_index)
                    push!(values, inference_factor * gamma)
                end
            end
            # Add (beta-1)*(Gbar_hg-Gbar_hl). Sparse matrices are CSC,
            # hence extracting these two rows through their transpose is cheap.
            for (gradient_face, sign) in (
                    (gas_face, one(T)),
                    (liquid_face, -one(T)),
                )
                for pointer in nzrange(gradient_transpose, gradient_face)
                    cell = gradient_cells[pointer]
                    local_pressure = get(pressure_lookup, cell, 0)
                    iszero(local_pressure) && continue
                    push!(pressure_rows, row)
                    push!(pressure_columns, local_pressure)
                    push!(
                        pressure_values,
                        inference_factor * sign *
                            nonzeros(gradient_transpose)[pointer],
                    )
                end
            end
            @inbounds rhs += inference_factor *
                (gas[gas_face] - liquid[liquid_face]) / T(dt)
        end
        @inbounds right_hand_side[row] = rhs
    end
    jump_block = sparse(
        rows, columns, values, length(stencils), length(mixed_faces),
    )
    pressure_block = sparse(
        pressure_rows, pressure_columns, pressure_values,
        length(stencils), length(pressure_indices),
    )
    return pressure_block, jump_block, right_hand_side, underresolved
end

function _assemble_cmdgfm_system!(
        solver::CMDGFMProjectionSolver{T},
        workspace::CMDGFMProjectionWorkspace{T}, dt,
        velocity::TwoPhaseVelocity,
    ) where {T}
    interface = solver.interface
    mesh = interface.mesh
    data = _mimetic_face_data(solver, workspace)
    pressure_operator = _mimetic_pressure_operator(solver)
    incidence = _mimetic_incidence(mesh, T)
    active = _mimetic_active_cells(interface)
    open_weights = isnothing(interface.geometry) ?
        ones(T, length(mesh.faces)) : copy(interface.geometry.aperture)
    pressure_indices = _gauge_free_component_rows(mesh, active, open_weights)
    all_stencils = _cmdgfm_jump_stencils(solver, data)
    underresolved = count(stencil -> !stencil.resolved, all_stencils)
    # The CMDGFM note explicitly says not to impose a jump condition when the
    # transverse interface stencil is underresolved. Such a face therefore has
    # no retained xi unknown; its gradient jump is extended by zero, exactly as
    # on every other non-retained face.
    stencils = filter(stencil -> stencil.resolved, all_stencils)
    mixed_faces = [stencil.face for stencil in stencils]
    pressure_positive = pressure_operator.stiffness[
        pressure_indices, pressure_indices,
    ]
    incidence_pressure = incidence[:, pressure_indices]
    pressure_jump = transpose(incidence_pressure[mixed_faces, :]) *
        spdiagm(0 => data.coupling[mixed_faces])
    common_gradient = _cmdgfm_common_gradient_matrix(
        pressure_operator, incidence, interface.geometry,
    )
    jump_pressure, jump_block, jump_rhs, retained_underresolved =
        _cmdgfm_jump_blocks(
        stencils, mixed_faces, data, common_gradient,
        pressure_indices, velocity, dt, T,
    )
    iszero(retained_underresolved) || error(
        "an underresolved CMDGFM row was unexpectedly retained",
    )
    matrix = [pressure_positive pressure_jump; jump_pressure jump_block]
    mixture = data.liquid_aperture .* velocity[1].interior .+
        data.gas_aperture .* velocity[2].interior
    boundary_mixture, = _mimetic_boundary_constraints(interface, velocity)
    pressure_rhs = (
        transpose(incidence_pressure) * mixture .-
            boundary_mixture[pressure_indices]
    ) ./ T(dt)
    right_hand_side = vcat(pressure_rhs, jump_rhs)
    return (;
        data, pressure_operator, incidence, matrix, jump_block, jump_pressure,
        right_hand_side, pressure_indices, mixed_faces, stencils,
        pressure_positive, underresolved,
    )
end

function _cmdgfm_block_preconditioner(assembled, ::Type{T}) where {T}
    pressure = assembled.pressure_positive
    pressure_factor = cholesky(Symmetric(pressure, :L))
    jump_factor = isempty(assembled.jump_block) ? nothing :
        lu(assembled.jump_block)
    dimensions = size(assembled.matrix)
    return CMDGFMBlockPreconditioner(
        T, pressure_factor, jump_factor, assembled.jump_pressure,
        size(pressure, 1), dimensions,
    )
end

function _cmdgfm_preconditioner(
        solver::CMDGFMProjectionSolver{T}, assembled,
    ) where {T}
    specification = solver.preconditioner
    specification === nothing && return nothing
    if specification === :block
        isempty(solver.preconditioner_options) || throw(
            ArgumentError("preconditioner_options are not used by :block"),
        )
        return _cmdgfm_block_preconditioner(assembled, T)
    elseif specification === :lu
        isempty(solver.preconditioner_options) || throw(
            ArgumentError("preconditioner_options are not used by exact LU"),
        )
        return ExternalPreconditioner(
            :lu, lu(assembled.matrix), assembled.matrix,
        )
    end
    return _construct_matrix_preconditioner(
        specification, assembled.matrix;
        symmetric = false, singular = false,
        options = solver.preconditioner_options,
    )
end

function _cmdgfm_residual_defects(residual, assembled, dt)
    T = eltype(residual)
    np = length(assembled.pressure_indices)
    pressure = zero(T)
    for local_index in 1:np
        cell = @inbounds assembled.pressure_indices[local_index]
        volume = @inbounds assembled.pressure_operator.volumes[cell]
        volume > zero(T) || continue
        @inbounds pressure = max(
            pressure, abs(residual[local_index]) / volume,
        )
    end
    jump = isempty(assembled.mixed_faces) ? zero(T) :
        maximum(abs, view(residual, (np + 1):length(residual)))
    return (pressure = T(dt)^2 * pressure, jump = T(dt) * jump)
end

function _cmdgfm_jump_diagonal_margin(jump_block)
    T = eltype(jump_block)
    isempty(jump_block) && return T(Inf)
    diagonal = zeros(T, size(jump_block, 1))
    off_diagonal = zeros(T, size(jump_block, 1))
    rows = rowvals(jump_block)
    for column in axes(jump_block, 2)
        for pointer in nzrange(jump_block, column)
            row = rows[pointer]
            value = abs(nonzeros(jump_block)[pointer])
            if row == column
                diagonal[row] += value
            else
                off_diagonal[row] += value
            end
        end
    end
    return minimum(diagonal .- off_diagonal)
end

"""Reconstruct both phase pressure gradients from a CMDGFM solution."""
function cmdgfm_gradient!(
        destination::TwoPhaseVelocity, pressure::CellField,
        solver::CMDGFMProjectionSolver{T},
        workspace::CMDGFMProjectionWorkspace{T}, pressure_operator, data,
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

function cmdgfm_gradient!(
        destination::TwoPhaseVelocity, pressure::CellField,
        solver::CMDGFMProjectionSolver,
        workspace::CMDGFMProjectionWorkspace,
    )
    operator = _mimetic_pressure_operator(solver)
    data = _mimetic_face_data(solver, workspace)
    return cmdgfm_gradient!(
        destination, pressure, solver, workspace, operator, data,
    )
end

"""Evaluate the vector jump interpolant represented by one CMDGFM stencil."""
function cmdgfm_jump_interpolant(
        velocity::TwoPhaseVelocity,
        stencil::CMDGFMJumpStencil{T},
    ) where {T}
    stencil.resolved || throw(
        ArgumentError("CMDGFM jump stencil is underresolved"),
    )
    beta = stencil.beta
    beta > sqrt(eps(T)) || throw(
        ArgumentError("the unscaled jump interpolant is singular for beta=0"),
    )
    liquid, gas = velocity
    face = stencil.face
    normal_jump = @inbounds gas.interior[face] - liquid.interior[face]
    if stencil.gas_inference_face > 0
        inferred = @inbounds(
            gas.interior[stencil.gas_inference_face] -
                liquid.interior[stencil.liquid_inference_face]
        )
        normal_jump = normal_jump / beta + (one(T) - inv(beta)) * inferred
    end
    tangential_jump = zero(T)
    for (candidate, weight) in zip(
            stencil.transverse_faces, stencil.transverse_weights,
        )
        @inbounds tangential_jump += weight *
            (gas.interior[candidate] - liquid.interior[candidate])
    end
    axis = velocity[1].mesh.faces[face].axis
    return axis == 1 ? (normal_jump, tangential_jump) :
        (tangential_jump, normal_jump)
end

function _cmdgfm_normal_velocity_defect(
        velocity::TwoPhaseVelocity, stencils,
    )
    T = eltype(velocity[1])
    result = zero(T)
    liquid, gas = velocity
    for stencil in stencils
        face = stencil.face
        value = @inbounds gas.interior[face] - liquid.interior[face]
        stencil.resolved || begin
            result = max(result, abs(value))
            continue
        end
        factor = stencil.beta * stencil.ratio
        for (candidate, weight) in zip(
                stencil.transverse_faces, stencil.transverse_weights,
            )
            @inbounds value += factor * weight *
                (gas.interior[candidate] - liquid.interior[candidate])
        end
        if stencil.gas_inference_face > 0
            @inbounds value += (stencil.beta - one(T)) * (
                gas.interior[stencil.gas_inference_face] -
                    liquid.interior[stencil.liquid_inference_face]
            )
        end
        result = max(result, abs(value))
    end
    return result
end

"""
    project!(velocity, pressure, solver::CMDGFMProjectionSolver, dt, workspace;
             tolerance, max_iterations)

Apply the conservative multidimensional ghost-fluid projection. The cell
pressure and mixed-face gradient jump form one nonsymmetric two-block system;
BiCGSTAB is used with either a pressure/jump block preconditioner or ILU.
"""
function project!(
        velocity::TwoPhaseVelocity, pressure::CellField{T},
        solver::CMDGFMProjectionSolver{T}, dt,
        workspace::CMDGFMProjectionWorkspace{T};
        tolerance = sqrt(eps(T)), max_iterations = 500,
    ) where {T}
    interface = solver.interface
    _validate_two_phase_velocity(velocity, interface.mesh, T)
    pressure.mesh === interface.mesh === workspace.pressure_gradient.mesh ||
        throw(ArgumentError("CMDGFM projection arguments must share a mesh"))
    dt > zero(dt) || throw(ArgumentError("dt must be strictly positive"))
    tolerance > zero(tolerance) || throw(
        ArgumentError("tolerance must be strictly positive"),
    )
    max_iterations > 0 || throw(
        ArgumentError("max_iterations must be strictly positive"),
    )
    assembled = _assemble_cmdgfm_system!(solver, workspace, dt, velocity)
    matrix = assembled.matrix
    right_hand_side = assembled.right_hand_side
    asymmetry = isempty(matrix) ? zero(T) :
        maximum(abs, matrix - transpose(matrix))
    initial_residual = norm(right_hand_side)
    solution = zeros(T, length(right_hand_side))
    iterations = 0
    preconditioner = nothing
    if !isempty(solution) && initial_residual > zero(T)
        preconditioner = _cmdgfm_preconditioner(solver, assembled)
        correction_rhs = copy(right_hand_side)
        for _ in 1:8
            remaining = max_iterations - iterations
            remaining > 0 || break
            defects = _cmdgfm_residual_defects(
                correction_rhs, assembled, dt,
            )
            maximum(defects) <= T(tolerance) && break
            krylov = Krylov.BicgstabWorkspace(matrix, correction_rhs)
            requested = min(
                T(tolerance), T(tolerance) / max(maximum(defects), eps(T)),
            )
            settings = (;
                atol = zero(T), rtol = requested, itmax = remaining,
            )
            if isnothing(preconditioner)
                Krylov.bicgstab!(
                    krylov, matrix, correction_rhs; settings...,
                )
            else
                Krylov.bicgstab!(
                    krylov, matrix, correction_rhs;
                    N = preconditioner, ldiv = true, settings...,
                )
            end
            increment = Krylov.solution(krylov)
            if !all(isfinite, increment) && !isnothing(preconditioner)
                krylov = Krylov.BicgstabWorkspace(matrix, correction_rhs)
                Krylov.bicgstab!(
                    krylov, matrix, correction_rhs; settings...,
                )
                increment = Krylov.solution(krylov)
            end
            solution .+= increment
            iterations += Krylov.iteration_count(krylov)
            mul!(correction_rhs, matrix, solution)
            correction_rhs .= right_hand_side .- correction_rhs
        end
    end
    residual = matrix * solution - right_hand_side
    final_residual = norm(residual)
    residual_defects = _cmdgfm_residual_defects(residual, assembled, dt)

    np = length(assembled.pressure_indices)
    nx = length(assembled.mixed_faces)
    _copy_mimetic_cell_solution!(
        pressure,
        zip(assembled.pressure_indices, view(solution, 1:np)),
    )
    fill!(workspace.jump, zero(T))
    workspace.jump[assembled.mixed_faces] .=
        view(solution, (np + 1):(np + nx))
    gradients = (workspace.liquid_gradient, workspace.gas_gradient)
    cmdgfm_gradient!(
        gradients, pressure, solver, workspace,
        assembled.pressure_operator, assembled.data,
    )
    for phase in 1:2
        for index in eachindex(velocity[phase].interior)
            @inbounds velocity[phase].interior[index] -=
                T(dt) * gradients[phase].interior[index]
        end
    end

    mixture_constraint, = _mimetic_integrated_constraints(
        assembled.incidence, assembled.data, interface, velocity,
        assembled.mixed_faces,
    )
    mixture_divergence = _mimetic_strong_maximum(
        mixture_constraint, assembled.pressure_operator.volumes,
    )
    mixture_volume_defect = T(dt) * mixture_divergence
    normal_velocity_defect = _cmdgfm_normal_velocity_defect(
        velocity, assembled.stencils,
    )
    momentum = zeros(T, 2)
    identity_error = zero(T)
    pressure_jump = assembled.pressure_operator.gradient_operator.pressure_jump
    for (index, face) in pairs(interface.mesh.faces)
        @inbounds begin
            liquid_mass = assembled.data.liquid_mass[index]
            gas_mass = assembled.data.gas_mass[index]
            weighted = liquid_mass * gradients[1].interior[index] +
                gas_mass * gradients[2].interior[index]
            expected = (liquid_mass + gas_mass) *
                workspace.pressure_gradient.interior[index]
            identity_error = max(identity_error, abs(weighted - expected))
            momentum[face.axis] -= T(dt) * pressure_jump[index]
        end
    end
    converged = residual_defects.pressure <= T(tolerance) &&
        residual_defects.jump <= T(tolerance) &&
        mixture_volume_defect <= T(tolerance) &&
        normal_velocity_defect <= T(tolerance)
    workspace.matrix = matrix
    workspace.jump_block = assembled.jump_block
    workspace.right_hand_side = right_hand_side
    workspace.solution = solution
    workspace.pressure_indices = assembled.pressure_indices
    workspace.mixed_faces = assembled.mixed_faces
    workspace.stencils = assembled.stencils
    preconditioner_name = isnothing(preconditioner) ? :none :
        _preconditioner_name(preconditioner)
    return CMDGFMProjectionStats(
        converged, iterations, initial_residual, final_residual, asymmetry,
        mixture_divergence, mixture_volume_defect, normal_velocity_defect,
        T(tolerance), (momentum[1], momentum[2]), identity_error,
        _cmdgfm_jump_diagonal_margin(assembled.jump_block), np, nx,
        assembled.underresolved, preconditioner_name,
    )
end
