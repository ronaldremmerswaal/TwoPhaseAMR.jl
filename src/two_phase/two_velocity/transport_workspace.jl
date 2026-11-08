"""
    MomentumAdvectionWorkspace(interface, velocity::TwoPhaseVelocity)

Construct the compatible staggered workspace for separate liquid and gas
momentum transport. The returned object contains intrinsic velocity fields;
it does not retain phase-weighted momentum.
"""
function MomentumAdvectionWorkspace(
        interface::InterfaceField{T, M}, velocity::TwoPhaseVelocity,
    ) where {T, M}
    _validate_two_phase_velocity(velocity, interface.mesh, T)
    return TwoVelocityMomentumAdvectionWorkspace(
        interface, PhaseFaceScalar{1}(interface),
        PhaseFaceScalar{2}(interface),
    )
end

mutable struct ProjectedLiquidAdvectionWorkspace{T, M, O, S, E, FX, FY, SX, SY}
    projected::FaceVelocity{T, M}
    extended_gas::FaceVelocity{T, M}
    correction::FaceVelocity{T, M}
    pressure::CellField{T, M}
    divergence::CellField{T, M}
    operator::O
    solver::S
    elliptic::E
    fraction_x::FX
    fraction_y::FY
    schedule_x::SX
    schedule_y::SY
    global_to_local_x::Vector{Int}
    global_to_local_y::Vector{Int}
    values_x::Vector{T}
    values_y::Vector{T}
    known_x::BitVector
    known_y::BitVector
    next_known_x::BitVector
    next_known_y::BitVector
    sums_x::Vector{T}
    sums_y::Vector{T}
    counts_x::Vector{Int}
    counts_y::Vector{Int}
    face_coefficient::Vector{T}
    active_cells::BitVector
    cell_volumes::Vector{T}
end

_interface_transport_workspace(field, ::FrozenInterfaceFluxes) = nothing

function _projected_cell_activity(field::InterfaceField{T}) where {T}
    mesh = field.mesh
    nx, ny = mesh.block_size
    halo = mesh.halo
    active = falses(ncells(mesh))
    volumes = zeros(T, ncells(mesh))
    for (block, leaf) in pairs(mesh.blocks)
        fraction = @inbounds field.conditional_fraction.blocks[block]
        open = isnothing(field.geometry) ? nothing :
            @inbounds(field.geometry.open_fraction.blocks[block])
        offset = (block - 1) * nx * ny
        @inbounds for j in 1:ny, i in 1:nx
            index = offset + (j - 1) * nx + i
            is_active = fraction[i + halo, j + halo] > zero(T)
            active[index] = is_active
            open_fraction = isnothing(open) ? one(T) : open[i + halo, j + halo]
            volumes[index] = is_active ? leaf.volume * open_fraction : zero(T)
        end
    end
    return active, volumes
end

function _projected_face_data!(
        coefficients, fraction_x, fraction_y, field::InterfaceField{T},
        global_to_local_x, global_to_local_y,
    ) where {T}
    _phase_face_average!(
        fraction_x, field.conditional_fraction, field.geometry,
    )
    _phase_face_average!(
        fraction_y, field.conditional_fraction, field.geometry,
    )
    geometry = field.geometry
    for (index, face) in pairs(field.mesh.faces)
        local_index = face.axis == 1 ? global_to_local_x[index] :
            global_to_local_y[index]
        fraction = face.axis == 1 ? fraction_x.interior[local_index] :
            fraction_y.interior[local_index]
        active = fraction > zero(T)
        aperture_squared = isnothing(geometry) ? one(T) :
            geometry.aperture[index]^2
        @inbounds coefficients[index] = active ? aperture_squared : zero(T)
    end
    return coefficients
end

function _projected_operator(
        field::InterfaceField{T}, coefficients, active, volumes,
    ) where {T}
    gradient_operator = isnothing(field.geometry) ?
        GradientOperator(field.mesh, T) : GradientOperator(field.geometry)
    return EllipticOperator(
        field.mesh; reaction = 0, diffusion = 1,
        face_coefficient = coefficients, gradient_operator,
        active_cells = active, volumes,
        inactive_condition = :dirichlet, T,
    )
end

function _interface_transport_workspace(
        field::InterfaceField{T, M}, ::ProjectedLiquidVelocity,
    ) where {T, M}
    reconstruct!(field)
    mesh = field.mesh
    fraction_x = _phase_face_average!(
        FaceField{1}(mesh, T), field.conditional_fraction, field.geometry,
    )
    fraction_y = _phase_face_average!(
        FaceField{2}(mesh, T), field.conditional_fraction, field.geometry,
    )
    global_to_local_x, count_x = _component_face_indices(mesh, 1)
    global_to_local_y, count_y = _component_face_indices(mesh, 2)
    coefficients = zeros(T, length(mesh.faces))
    _projected_face_data!(
        coefficients, fraction_x, fraction_y, field,
        global_to_local_x, global_to_local_y,
    )
    active, volumes = _projected_cell_activity(field)
    operator = _projected_operator(field, coefficients, active, volumes)
    solver = EllipticSolver(operator; preconditioner = nothing)
    schedule_x = _dual_advection_schedule(mesh, Val(1), T, false)
    schedule_y = _dual_advection_schedule(mesh, Val(2), T, false)
    return ProjectedLiquidAdvectionWorkspace(
        FaceVelocity(mesh, T), FaceVelocity(mesh, T), FaceVelocity(mesh, T),
        CellField(mesh, T), CellField(mesh, T), operator, solver,
        EllipticWorkspace(operator), fraction_x, fraction_y,
        schedule_x, schedule_y, global_to_local_x, global_to_local_y,
        zeros(T, count_x), zeros(T, count_y),
        falses(count_x), falses(count_y),
        falses(count_x), falses(count_y),
        zeros(T, count_x), zeros(T, count_y),
        zeros(Int, count_x), zeros(Int, count_y),
        coefficients, active, volumes,
    )
end

function _refresh_projected_operator!(
        state::ProjectedLiquidAdvectionWorkspace{T},
        field::InterfaceField{T},
    ) where {T}
    _projected_face_data!(
        state.face_coefficient, state.fraction_x, state.fraction_y, field,
        state.global_to_local_x, state.global_to_local_y,
    )
    active, volumes = _projected_cell_activity(field)
    singular = !any(!, active)
    singular == state.operator.singular || return false
    copyto!(state.active_cells, active)
    copyto!(state.cell_volumes, volumes)
    _update_elliptic_activity!(state.operator, active, volumes)
    _update_elliptic_face_coefficient!(
        state.operator, state.face_coefficient,
    )
    return true
end

function _extend_projected_component!(
        destination::FaceVelocity{T}, source::FaceVelocity{T}, axis,
        fraction, schedule, global_to_local, values, known, next_known,
        sums, counts, layers; phase = :liquid, geometry = nothing,
    ) where {T}
    fill!(values, zero(T))
    fill!(known, false)
    for (face_index, face) in pairs(source.mesh.faces)
        face.axis == axis || continue
        local_index = global_to_local[face_index]
        active = (isnothing(geometry) || geometry.face_dual_volumes[face_index] > zero(T)) &&
            (
            phase == :liquid ? fraction.interior[local_index] > zero(T) :
                fraction.interior[local_index] < one(T)
        )
        @inbounds begin
            known[local_index] = active
            values[local_index] = active ? source.interior[face_index] : zero(T)
        end
    end
    for _ in 1:layers
        copyto!(next_known, known)
        fill!(sums, zero(T))
        fill!(counts, 0)
        for connection in schedule.faces
            minus, plus = connection.minus, connection.plus
            if known[minus] && !known[plus]
                @inbounds begin
                    sums[plus] += values[minus]
                    counts[plus] += 1
                end
            elseif known[plus] && !known[minus]
                @inbounds begin
                    sums[minus] += values[plus]
                    counts[minus] += 1
                end
            end
        end
        changed = false
        for index in eachindex(values)
            if !known[index] && counts[index] > 0
                @inbounds begin
                    values[index] = sums[index] / counts[index]
                    next_known[index] = true
                end
                changed = true
            end
        end
        copyto!(known, next_known)
        changed || break
    end
    for (face_index, face) in pairs(source.mesh.faces)
        face.axis == axis || continue
        @inbounds destination.interior[face_index] =
            values[global_to_local[face_index]]
    end
    return destination
end

function _extend_projected_gas_velocity!(work::WYAdvectionWorkspace, velocity)
    state = work.method_workspace[]
    for (axis, fraction, schedule, indices, values, known, next_known, sums, counts) in (
            (
                1, state.fraction_x, state.schedule_x, state.global_to_local_x,
                state.values_x, state.known_x, state.next_known_x, state.sums_x, state.counts_x,
            ),
            (
                2, state.fraction_y, state.schedule_y, state.global_to_local_y,
                state.values_y, state.known_y, state.next_known_y, state.sums_y, state.counts_y,
            ),
        )
        _extend_projected_component!(
            state.extended_gas, velocity[2], axis,
            fraction, schedule, indices, values, known, next_known, sums, counts,
            work.interface_transport.extension_layers;
            phase = :gas, geometry = work.interface.geometry
        )
    end
    fill!(state.extended_gas.boundary, zero(eltype(state.extended_gas)))
    return state.extended_gas
end

function _project_liquid_velocity!(
        work::WYAdvectionWorkspace{T}, velocity::TwoPhaseVelocity,
    ) where {T}
    throw(
        ArgumentError(
            "the WY workspace does not use projected liquid advection",
        )
    )
end

function _project_liquid_velocity!(
        work::WYAdvectionWorkspace{T, M, <:ProjectedLiquidVelocity},
        velocity::TwoPhaseVelocity,
    ) where {T, M <: BlockTreeMesh}
    field = work.interface
    method = work.interface_transport
    state = work.method_workspace[]
    reconstruct!(field)
    if !_refresh_projected_operator!(state, field)
        state = _interface_transport_workspace(field, method)
        work.method_workspace[] = state
    end
    liquid = velocity[1]
    _extend_projected_component!(
        state.projected, liquid, 1, state.fraction_x, state.schedule_x,
        state.global_to_local_x, state.values_x, state.known_x,
        state.next_known_x, state.sums_x, state.counts_x,
        method.extension_layers; geometry = field.geometry,
    )
    _extend_projected_component!(
        state.projected, liquid, 2, state.fraction_y, state.schedule_y,
        state.global_to_local_y, state.values_y, state.known_y,
        state.next_known_y, state.sums_y, state.counts_y,
        method.extension_layers; geometry = field.geometry,
    )
    fill!(state.projected.boundary, zero(T))
    if !any(state.active_cells)
        fill!(state.correction.interior, zero(T))
        fill!(state.correction.boundary, zero(T))
        diagnostics = work.diagnostics
        diagnostics.projection_initial_divergence = zero(T)
        diagnostics.projection_final_divergence = zero(T)
        diagnostics.projection_iterations = 0
        return state.projected
    end
    isnothing(field.geometry) ?
        divergence!(state.divergence, state.projected) :
        divergence!(state.divergence, state.projected, field.geometry)
    initial = zero(T)
    for (index, ref) in pairs(state.operator.references)
        state.active_cells[index] || continue
        initial = max(initial, abs(_get(state.divergence, ref)))
        _set!(state.divergence, -_get(state.divergence, ref), ref)
    end
    _zero_interior!(state.pressure)
    stats = solve!(
        state.pressure, state.solver, state.divergence;
        rtol = zero(T), atol = T(method.tolerance),
        max_cycles = method.max_cycles,
        compatibility = state.operator.singular ? :project : :error,
    )
    stats.converged || error(
        "projected liquid-velocity extension failed: cycles=$(stats.cycles), " *
            "initial_linf=$(stats.initial_linf), " *
            "final_linf=$(stats.final_linf), target=$(stats.target_linf)",
    )
    if isnothing(field.geometry)
        weighted_gradient!(
            state.correction, state.pressure,
            state.operator, state.elliptic,
        )
    else
        _cut_cell_weighted_gradient!(
            state.correction, state.pressure,
            state.operator, state.elliptic, field.geometry,
        )
    end
    @_parallel for index in eachindex(state.projected.interior)
        @inbounds state.projected.interior[index] -=
            state.correction.interior[index]
    end
    fill!(state.projected.boundary, zero(T))
    isnothing(field.geometry) ?
        divergence!(state.divergence, state.projected) :
        divergence!(state.divergence, state.projected, field.geometry)
    final = zero(T)
    for (index, ref) in pairs(state.operator.references)
        state.active_cells[index] || continue
        final = max(final, abs(_get(state.divergence, ref)))
    end
    diagnostics = work.diagnostics
    diagnostics.projection_initial_divergence = initial
    diagnostics.projection_final_divergence = final
    diagnostics.projection_iterations = stats.cycles
    return state.projected
end
