function _check_staggered_arguments(
        destination::FaceField{Axis},
        advected::FaceField{Axis},
        velocity::FaceVelocity,
        workspace::StaggeredAdvectionWorkspace{Axis}
    ) where {Axis}
    mesh = advected.mesh
    destination.mesh === mesh === velocity.mesh === workspace.tendency.mesh ||
        throw(ArgumentError("all staggered-advection arguments must share a mesh"))
    destination === advected && throw(
        ArgumentError(
            "convection! destination must not alias its advected field"
        )
    )
    return nothing
end

@inline function _amr_dual_flux(
        face::AMRDualFace,
        schedule::AMRDualAdvectionSchedule,
        velocity::FaceVelocity
    )
    flux = zero(promote_type(eltype(schedule.flux_weights), eltype(velocity)))
    stop = face.first_weight + face.weight_count - 1
    for index in face.first_weight:stop
        @inbounds begin
            primal_face = schedule.flux_indices[index]
            flux += schedule.flux_weights[index] *
                velocity.interior[primal_face] *
                velocity.mesh.faces[primal_face].area
        end
    end
    return flux
end

@inline function _configured_side_boundary(boundary, axis, side)
    return boundary isa VelocityBoundaryConditions ?
        _side_boundary(boundary, axis, side) : boundary
end

@inline function _tangential_boundary_state(
        condition, point, time, axis, interior_state, ::Type{T},
    ) where {T}
    if condition isa Inflow
        return T(condition.velocity(point[1], point[2], time, axis))
    elseif condition isa NoSlip
        return T(condition.velocity(point[1], point[2], time, axis))
    end
    return interior_state
end

function _add_boundary_convection!(
        destination, advected, advecting,
        workspace::StaggeredAdvectionWorkspace{Axis, T},
        scheme, sweep_axis,
    ) where {Axis, T}
    configuration = workspace.boundary_conditions[]
    isnothing(configuration) && return destination
    boundary, time = configuration
    mesh = advected.mesh
    for connection in workspace.boundary_faces
        (iszero(sweep_axis) || connection.wall_axis == sweep_axis) || continue
        boundary_face = mesh.boundary_faces[connection.boundary_index]
        condition = _configured_side_boundary(
            boundary, connection.wall_axis, connection.side,
        )
        q = @inbounds advected.interior[connection.degree_of_freedom]
        flux = if connection.normal_component
            @inbounds connection.measure * (
                advecting.boundary[connection.boundary_index] +
                    advecting.interior[connection.opposite_flux]
            ) / 2
        else
            @inbounds connection.measure *
                advecting.boundary[connection.boundary_index]
        end
        state = if connection.normal_component
            qb = @inbounds advected.boundary[connection.boundary_local_index]
            if _is_upwind_scheme(scheme)
                connection.side * flux < zero(T) ? qb : q
            else
                (q + qb) / 2
            end
        else
            _tangential_boundary_state(
                condition, boundary_face.center, time, Axis, q, T,
            )
        end
        volume = _dual_volumes(
            workspace.schedule, connection.degree_of_freedom,
        )
        @inbounds destination.interior[connection.degree_of_freedom] +=
            connection.side * flux * state / volume
    end
    return destination
end

@inline _amr_dual_state(
    qminus, qplus, flux, divergence, dt, face,
    minus_volume, plus_volume,
    ::Upwind, values
) =
    ifelse(flux >= zero(flux), qminus, qplus)

@inline _amr_dual_state(
    qminus, qplus, flux, divergence, dt, face,
    minus_volume, plus_volume,
    ::Centred, values
) = (qminus + qplus) / 2

@inline function _amr_dual_state(
        qminus, qplus, flux, divergence, dt, face,
        minus_volume, plus_volume, ::LaxWendroff,
        values
    )
    # On a nonconforming schedule there is no unique transverse donor. Use the
    # one-dimensional LW state on each algebraic connection. The smaller
    # adjacent dual volume gives a conservative local length scale and is
    # covered by `stable_timestep`. `Upwind` remains the robust AMR option.
    courant = abs(dt * flux) / min(minus_volume, plus_volume)
    upwind, downwind = flux >= zero(flux) ? (qminus, qplus) : (qplus, qminus)
    return (
        (one(courant) - courant) * downwind +
            (one(courant) + courant) * upwind
    ) / 2 -
        dt * (
        qminus * divergence[face.minus] +
            qplus * divergence[face.plus]
    ) / 4
end

@inline function _amr_dual_state(
        qminus, qplus, flux, divergence, dt, face,
        minus_volume, plus_volume,
        scheme::AbstractThreePointFluxScheme, values
    )
    positive_flux = flux >= zero(flux)
    upwind, downwind = positive_flux ? (qminus, qplus) : (qplus, qminus)
    stencil = positive_flux ? face.minus_stencil : face.plus_stencil
    iszero(stencil.upstream) && return upwind
    high_order = _amr_dual_state(
        qminus, qplus, flux, divergence, dt, face,
        minus_volume, plus_volume, LaxWendroff(), values
    )
    upstream = @inbounds values[stencil.upstream]
    return _three_point_state(
        upwind, downwind, upstream,
        stencil.ratio_scale, high_order, scheme
    )
end

function _directional_flux_divergence!(
        divergence, velocity,
        workspace::StaggeredAdvectionWorkspace{
            Axis, T, M, <:UniformDualAdvectionSchedule,
        },
        sweep_axis::Int
    ) where {Axis, T, M}
    schedule = workspace.schedule
    inverse_volume = inv(schedule.volume)
    @_parallel for face_index in 1:length(schedule.faces)
        face = schedule.faces[face_index]
        @inbounds begin
            workspace.dual_flux[face_index] =
                (iszero(sweep_axis) || face.axis == sweep_axis) ? (
                    velocity.interior[face.flux_a] +
                    velocity.interior[face.flux_b]
                ) * face.area / 2 : zero(T)
        end
    end
    @_parallel for output in 1:length(divergence)
        value = zero(T)
        for pointer in workspace.dual_offsets[output]:(
                workspace.dual_offsets[output + 1] - 1
            )
            face_index = @inbounds workspace.dual_faces[pointer]
            @inbounds value += workspace.dual_signs[pointer] *
                workspace.dual_flux[face_index]
        end
        @inbounds divergence[output] = value * inverse_volume
    end
    return divergence
end

function _directional_flux_divergence!(
        divergence, velocity,
        workspace::StaggeredAdvectionWorkspace{
            Axis, T, M, <:AMRDualAdvectionSchedule,
        },
        sweep_axis::Int
    ) where {Axis, T, M}
    schedule = workspace.schedule
    @_parallel for face_index in 1:length(schedule.faces)
        face = schedule.faces[face_index]
        @inbounds workspace.dual_flux[face_index] =
            (iszero(sweep_axis) || face.axis == sweep_axis) ?
            _amr_dual_flux(face, schedule, velocity) : zero(T)
    end
    @_parallel for output in 1:length(divergence)
        value = zero(T)
        for pointer in workspace.dual_offsets[output]:(
                workspace.dual_offsets[output + 1] - 1
            )
            face_index = @inbounds workspace.dual_faces[pointer]
            @inbounds value += workspace.dual_signs[pointer] *
                workspace.dual_flux[face_index]
        end
        @inbounds divergence[output] = value / schedule.volumes[output]
    end
    return divergence
end

function _convection_schedule!(
        destination, advected, velocity, dt,
        workspace::StaggeredAdvectionWorkspace{
            Axis, T, M, <:UniformDualAdvectionSchedule,
        },
        directional_divergence, scheme,
        amr_interface_scheme,
        sweep_axis::Int = 0
    ) where {Axis, T, M}
    schedule = workspace.schedule
    inverse_volume = inv(schedule.volume)
    @_parallel for face_index in 1:length(schedule.faces)
        face = schedule.faces[face_index]
        if !(iszero(sweep_axis) || face.axis == sweep_axis)
            @inbounds workspace.dual_flux[face_index] = zero(T)
            continue
        end
        @inbounds begin
            flux = (
                velocity.interior[face.flux_a] +
                    velocity.interior[face.flux_b]
            ) * face.area / 2
            qminus = advected.interior[face.minus]
            qplus = advected.interior[face.plus]
            state = _dual_state(
                qminus, qplus, flux,
                directional_divergence, dt, face, scheme,
                advected.interior
            )
            workspace.dual_flux[face_index] = flux * state
        end
    end
    @_parallel for output in 1:length(destination.interior)
        value = zero(T)
        for pointer in workspace.dual_offsets[output]:(
                workspace.dual_offsets[output + 1] - 1
            )
            face_index = @inbounds workspace.dual_faces[pointer]
            @inbounds value += workspace.dual_signs[pointer] *
                workspace.dual_flux[face_index]
        end
        @inbounds destination.interior[output] = value * inverse_volume
    end
    return _add_boundary_convection!(
        destination, advected, velocity, workspace, scheme, sweep_axis,
    )
end

function _convection_schedule!(
        destination, advected, velocity, dt,
        workspace::StaggeredAdvectionWorkspace{
            Axis, T, M, <:AMRDualAdvectionSchedule,
        },
        directional_divergence, scheme,
        amr_interface_scheme,
        sweep_axis::Int = 0
    ) where {Axis, T, M}
    schedule = workspace.schedule
    @_parallel for face_index in 1:length(schedule.faces)
        face = schedule.faces[face_index]
        if !(iszero(sweep_axis) || face.axis == sweep_axis)
            @inbounds workspace.dual_flux[face_index] = zero(T)
            continue
        end
        @inbounds begin
            flux = _amr_dual_flux(face, schedule, velocity)
            qminus = advected.interior[face.minus]
            qplus = advected.interior[face.plus]
            minus_volume = schedule.volumes[face.minus]
            plus_volume = schedule.volumes[face.plus]
            state = if face.refinement_interface
                _amr_dual_state(
                    qminus, qplus, flux,
                    directional_divergence, dt, face,
                    minus_volume, plus_volume,
                    amr_interface_scheme, advected.interior
                )
            else
                _amr_dual_state(
                    qminus, qplus, flux,
                    directional_divergence, dt, face,
                    minus_volume, plus_volume, scheme,
                    advected.interior
                )
            end
            workspace.dual_flux[face_index] = flux * state
        end
    end
    @_parallel for output in 1:length(destination.interior)
        value = zero(T)
        for pointer in workspace.dual_offsets[output]:(
                workspace.dual_offsets[output + 1] - 1
            )
            face_index = @inbounds workspace.dual_faces[pointer]
            @inbounds value += workspace.dual_signs[pointer] *
                workspace.dual_flux[face_index]
        end
        @inbounds destination.interior[output] =
            value / schedule.volumes[output]
    end
    return _add_boundary_convection!(
        destination, advected, velocity, workspace, scheme, sweep_axis,
    )
end

function _convection!(
        destination::FaceField{Axis, T},
        advected::FaceField{Axis}, velocity::FaceVelocity, dt,
        workspace::StaggeredAdvectionWorkspace{Axis},
        scheme, amr_interface_scheme,
        sweep_axis::Int = 0
    ) where {Axis, T}
    _check_staggered_arguments(destination, advected, velocity, workspace)
    fill!(destination.interior, zero(T))
    fill!(destination.boundary, zero(T))
    if !iszero(dt) && _directionally_split_scheme(
            scheme, amr_interface_scheme, workspace.schedule
        )
        _directional_flux_divergence!(
            workspace.rates.interior, velocity, workspace, sweep_axis
        )
    else
        fill!(workspace.rates.interior, zero(T))
    end
    return _convection_schedule!(
        destination, advected, velocity, dt, workspace,
        workspace.rates.interior, scheme, amr_interface_scheme, sweep_axis
    )
end

function _dispatch_interface_convection!(
        destination, advected, velocity, dt,
        workspace, scheme,
        amr_interface_scheme,
        sweep_axis::Int = 0
    )
    if amr_interface_scheme isa Upwind || amr_interface_scheme === :upwind
        return _convection!(
            destination, advected, velocity, dt, workspace,
            scheme, Upwind(), sweep_axis
        )
    elseif amr_interface_scheme isa Centred ||
            amr_interface_scheme === :centred ||
            amr_interface_scheme === :centered ||
            amr_interface_scheme === :central
        return _convection!(
            destination, advected, velocity, dt, workspace,
            scheme, Centred(), sweep_axis
        )
    elseif amr_interface_scheme isa LaxWendroff ||
            amr_interface_scheme === :lw ||
            amr_interface_scheme === :lax_wendroff
        return _convection!(
            destination, advected, velocity, dt, workspace,
            scheme, LaxWendroff(), sweep_axis
        )
    elseif amr_interface_scheme isa Fromm ||
            amr_interface_scheme === :fromm
        return _convection!(
            destination, advected, velocity, dt, workspace,
            scheme, Fromm(), sweep_axis
        )
    elseif amr_interface_scheme isa Minmod ||
            amr_interface_scheme === :minmod
        return _convection!(
            destination, advected, velocity, dt, workspace,
            scheme, Minmod(), sweep_axis
        )
    elseif amr_interface_scheme isa MonotonizedCentral ||
            amr_interface_scheme === :mc ||
            amr_interface_scheme === :monotonized_central ||
            amr_interface_scheme === :monotonised_central
        return _convection!(
            destination, advected, velocity, dt, workspace,
            scheme, MonotonizedCentral(), sweep_axis
        )
    elseif amr_interface_scheme isa VanLeer ||
            amr_interface_scheme === :van_leer ||
            amr_interface_scheme === :vanleer
        return _convection!(
            destination, advected, velocity, dt, workspace,
            scheme, VanLeer(), sweep_axis
        )
    elseif amr_interface_scheme isa Superbee ||
            amr_interface_scheme === :superbee
        return _convection!(
            destination, advected, velocity, dt, workspace,
            scheme, Superbee(), sweep_axis
        )
    end
    throw(
        ArgumentError(
            "unsupported AMR-interface advection scheme: $amr_interface_scheme"
        )
    )
end

function _dispatch_convection!(
        destination, advected, velocity, dt, workspace,
        scheme, amr_interface_scheme,
        sweep_axis::Int = 0
    )
    if scheme isa Upwind || scheme === :upwind
        return _dispatch_interface_convection!(
            destination, advected, velocity, dt, workspace,
            Upwind(), amr_interface_scheme, sweep_axis
        )
    elseif scheme isa Centred || scheme === :centred ||
            scheme === :centered || scheme === :central
        return _dispatch_interface_convection!(
            destination, advected, velocity, dt, workspace,
            Centred(), amr_interface_scheme, sweep_axis
        )
    elseif scheme isa LaxWendroff || scheme === :lw || scheme === :lax_wendroff
        return _dispatch_interface_convection!(
            destination, advected, velocity, dt, workspace,
            LaxWendroff(), amr_interface_scheme, sweep_axis
        )
    elseif scheme isa Fromm || scheme === :fromm
        return _dispatch_interface_convection!(
            destination, advected, velocity, dt, workspace,
            Fromm(), amr_interface_scheme, sweep_axis
        )
    elseif scheme isa Minmod || scheme === :minmod
        return _dispatch_interface_convection!(
            destination, advected, velocity, dt, workspace,
            Minmod(), amr_interface_scheme, sweep_axis
        )
    elseif scheme isa MonotonizedCentral || scheme === :mc ||
            scheme === :monotonized_central ||
            scheme === :monotonised_central
        return _dispatch_interface_convection!(
            destination, advected, velocity, dt, workspace,
            MonotonizedCentral(), amr_interface_scheme, sweep_axis
        )
    elseif scheme isa VanLeer || scheme === :van_leer || scheme === :vanleer
        return _dispatch_interface_convection!(
            destination, advected, velocity, dt, workspace,
            VanLeer(), amr_interface_scheme, sweep_axis
        )
    elseif scheme isa Superbee || scheme === :superbee
        return _dispatch_interface_convection!(
            destination, advected, velocity, dt, workspace,
            Superbee(), amr_interface_scheme, sweep_axis
        )
    end
    throw(ArgumentError("unsupported staggered advection scheme: $scheme"))
end

@inline _directionally_split_scheme(::LaxWendroff) = true
@inline _directionally_split_scheme(::AbstractThreePointFluxScheme) = true
@inline _directionally_split_scheme(scheme::Symbol) =
    scheme === :lw || scheme === :lax_wendroff || scheme === :fromm ||
    scheme === :minmod || scheme === :mc ||
    scheme === :monotonized_central || scheme === :monotonised_central ||
    scheme === :van_leer || scheme === :vanleer || scheme === :superbee
@inline _directionally_split_scheme(scheme) = false
@inline _directionally_split_scheme(scheme, amr_interface_scheme) =
    _directionally_split_scheme(scheme) ||
    _directionally_split_scheme(amr_interface_scheme)
@inline _directionally_split_scheme(
    scheme, amr_interface_scheme,
    ::UniformDualAdvectionSchedule
) =
    _directionally_split_scheme(scheme)
@inline _directionally_split_scheme(
    scheme, amr_interface_scheme,
    ::AMRDualAdvectionSchedule
) =
    _directionally_split_scheme(scheme, amr_interface_scheme)

function _split_convection_rate!(
        destination::FaceField{Axis, T},
        advected::FaceField{Axis},
        velocity::FaceVelocity, dt,
        workspace::StaggeredAdvectionWorkspace{Axis},
        scheme, amr_interface_scheme
    ) where {Axis, T}
    destination === advected && throw(
        ArgumentError(
            "convection! destination must not alias its advected field"
        )
    )
    (destination === workspace.tendency || destination === workspace.rates) &&
        throw(
        ArgumentError(
            "split convection destination must be separate from its workspace"
        )
    )
    copyto!(destination.interior, advected.interior)
    copyto!(destination.boundary, advected.boundary)
    scratch = destination === workspace.tendency ?
        workspace.rates : workspace.tendency
    sweep_axes = workspace.sweep_forward[] ? (1, 2) : (2, 1)
    for sweep_axis in sweep_axes
        sweep_dt = dt
        _dispatch_convection!(
            scratch, destination, velocity, sweep_dt, workspace,
            scheme, amr_interface_scheme, sweep_axis
        )
        for index in eachindex(destination.interior)
            @inbounds destination.interior[index] -=
                sweep_dt * scratch.interior[index]
        end
    end
    for index in eachindex(destination.interior)
        @inbounds destination.interior[index] =
            (advected.interior[index] - destination.interior[index]) / dt
    end
    fill!(destination.boundary, zero(T))
    return destination
end
