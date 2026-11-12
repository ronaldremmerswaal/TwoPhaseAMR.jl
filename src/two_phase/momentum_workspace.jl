struct MomentumAdvectionWorkspace{T, M <: BlockTreeMesh, WX, WY, F <: InterfaceField{T, M}}
    interface::F
    x::MomentumComponentWorkspace{1, T, M, WX}
    y::MomentumComponentWorkspace{2, T, M, WY}
end

# The one-velocity momentum shares the phase workspace of the staggered phase
# scalars, open-strip factorization included (`_open_strip_phase_schedule`), so
# its divergence commutes with the staggered density of
# `_phase_face_average!`.
function _momentum_advection_workspace(
        tendency::FaceField, rates::FaceField, geometry, forward;
        reuse_flux = nothing,
    )
    return _phase_advection_workspace(
        tendency, rates, geometry, forward; reuse_flux,
    )
end

function _momentum_advection_workspace(field::FaceField{Axis, T}, geometry) where {Axis, T}
    return _momentum_advection_workspace(
        FaceField{Axis}(field.mesh, T), FaceField{Axis}(field.mesh, T),
        geometry, Ref(true),
    )
end

function _momentum_component_workspace(
        interface::InterfaceField{T, M},
        ::Val{Axis}
    ) where {T, M, Axis}
    mesh = interface.mesh
    value = FaceField{Axis}(mesh, T)
    advection = _momentum_advection_workspace(value, interface.geometry)
    face_count = length(value.interior) + length(value.boundary)
    return MomentumComponentWorkspace(
        value, FaceField{Axis}(mesh, T),
        FaceField{Axis}(mesh, T), FaceField{Axis}(mesh, T), FaceField{Axis}(mesh, T),
        advection, ntuple(_ -> zeros(T, face_count), 6)...,
        PhaseDrainLimiter(T, face_count, advection),
        PhaseDrainLimiter(T, face_count, advection),
    )
end

function MomentumAdvectionWorkspace(interface::InterfaceField{T, M}) where {T, M}
    x = _momentum_component_workspace(interface, Val(1))
    y = _momentum_component_workspace(interface, Val(2))
    return MomentumAdvectionWorkspace{T, M, typeof(x.advection), typeof(y.advection), typeof(interface)}(
        interface, x, y
    )
end

function _resize_zero_face_field!(field::FaceField{Axis, T}) where {Axis, T}
    resize!(field.interior, _nfaces(field.mesh, Axis))
    resize!(field.boundary, _nboundary_faces(field.mesh, Axis))
    fill!(field.interior, zero(T))
    fill!(field.boundary, zero(T))
    return field
end

function _reconfigure_momentum_component!(
        component::MomentumComponentWorkspace{Axis, T},
        interface::InterfaceField{T},
    ) where {Axis, T}
    mesh = interface.mesh
    for field in (
            component.value, component.source, component.fraction,
            component.predicted_fraction, component.increment,
            component.advection.tendency, component.advection.rates,
        )
        _resize_zero_face_field!(field)
    end
    advection = _momentum_advection_workspace(
        component.advection.tendency, component.advection.rates,
        interface.geometry, component.advection.sweep_forward;
        reuse_flux = component.advection.dual_flux,
    )
    face_count = length(component.value.interior) +
        length(component.value.boundary)
    for values in (
            component.old_fraction, component.new_fraction,
            component.values, component.sources, component.delta,
            component.rates,
        )
        resize!(values, face_count)
        fill!(values, zero(T))
    end
    _resize_drain_limiter!(component.liquid_limiter, face_count, advection)
    _resize_drain_limiter!(component.gas_limiter, face_count, advection)
    return MomentumComponentWorkspace(
        component.value, component.source, component.fraction,
        component.predicted_fraction, component.increment, advection,
        component.old_fraction, component.new_fraction,
        component.values, component.sources, component.delta, component.rates,
        component.liquid_limiter, component.gas_limiter,
    )
end

"""Rebuild only momentum connectivity and retain all field/vector capacity."""
function _reconfigure_momentum_workspace!(
        workspace::MomentumAdvectionWorkspace{T},
        interface::InterfaceField{T, M},
    ) where {T, M}
    workspace.interface === interface || return MomentumAdvectionWorkspace(interface)
    x = _reconfigure_momentum_component!(workspace.x, interface)
    y = _reconfigure_momentum_component!(workspace.y, interface)
    return MomentumAdvectionWorkspace{T, M, typeof(x.advection), typeof(y.advection), typeof(interface)}(
        interface, x, y,
    )
end

function _prepare_momentum_advection!(
        work::MomentumAdvectionWorkspace,
        velocity::FaceVelocity
    )
    copyto!(work.x.value, velocity);copyto!(work.y.value, velocity)
    copyto!(work.x.source, velocity);copyto!(work.y.source, velocity)
    return work
end

function _validate_phase_momentum!(
        momentum::FaceVelocity, source::FaceVelocity,
        workspace::MomentumAdvectionWorkspace,
        field::InterfaceField, ::Type,
    )
    momentum.mesh === field.mesh || throw(
        ArgumentError("momentum destination must use the interface mesh"),
    )
    workspace.interface === field || throw(
        ArgumentError("momentum workspace belongs to a different interface"),
    )
    _prepare_momentum_advection!(workspace, source)
    return workspace
end

_phase_momentum_sources(::MomentumAdvectionWorkspace) = nothing

function _advect_phase_momentum!(
        workspace::MomentumAdvectionWorkspace, work, field, axis, sources,
        density, scheme, amr_interface_scheme,
    )
    _advect_momentum_component!(
        workspace.x, work, field, axis, density, work.ctu_threshold,
        scheme, amr_interface_scheme,
    )
    _advect_momentum_component!(
        workspace.y, work, field, axis, density, work.ctu_threshold,
        scheme, amr_interface_scheme,
    )
    return workspace
end

function _finish_phase_momentum!(
        destination::FaceVelocity, workspace::MomentumAdvectionWorkspace,
    )
    copyto!(destination, workspace.x.value)
    copyto!(destination, workspace.y.value)
    return destination
end


function _advect_momentum_component!(
        component::MomentumComponentWorkspace,
        work::WYAdvectionWorkspace, field::InterfaceField,
        axis, density, ctu_threshold, scheme,
        amr_interface_scheme
    )
    q = component.value
    mesh = q.mesh
    _phase_face_average!(
        component.fraction, field.conditional_fraction, field.geometry,
    )
    nx, _ = mesh.block_size
    halo = mesh.halo
    _parallel_block_columns(mesh) do b, first_j, last_j
        fraction = @inbounds field.conditional_fraction.blocks[b]
        increment = @inbounds work.increment.blocks[b]
        incoming = @inbounds work.incoming.blocks[b]
        @inbounds for j in first_j:last_j, i in 1:nx
            ii, jj = i + halo, j + halo
            incoming[ii, jj] = clamp(
                fraction[ii, jj] + increment[ii, jj],
                zero(eltype(q)), one(eltype(q)),
            )
        end
    end
    _phase_face_average!(component.predicted_fraction, work.incoming, field.geometry)
    oldc = _pack_phase_faces!(component.old_fraction, component.fraction)
    newc = _pack_phase_faces!(
        component.new_fraction, component.predicted_fraction
    )
    values = _pack_phase_faces!(component.values, q)
    sources = _pack_phase_faces!(component.sources, component.source)
    delta = component.delta
    schedule = component.advection.schedule
    for (flux, liquid, limiter) in (
            (work.liquid_flux, true, component.liquid_limiter),
            (work.gas_flux, false, component.gas_limiter),
        )
        _phase_drain_limit!(
            limiter.excess, limiter.flush, limiter.suspect,
            component.advection, schedule, flux, values, oldc, liquid, axis,
        )
    end
    @_parallel for output in 1:length(values)
        value = zero(eltype(delta))
        for (rho, flux, liquid, limiter) in (
                (density[1], work.liquid_flux, true, component.liquid_limiter),
                (density[2], work.gas_flux, false, component.gas_limiter),
            )
            for pointer in component.advection.dual_offsets[output]:(
                    component.advection.dual_offsets[output + 1] - 1
                )
                face_index = @inbounds component.advection.dual_faces[pointer]
                face = schedule.faces[face_index]
                face.axis == axis || continue
                m, p = face.minus, face.plus
                cm, cp = liquid ? (oldc[m], oldc[p]) :
                    (1 - oldc[m], 1 - oldc[p])
                nm, np = liquid ? (newc[m], newc[p]) :
                    (1 - newc[m], 1 - newc[p])
                _, minus_difference, plus_difference = _phase_transport(
                    face, schedule, flux, mesh, values,
                    cm, cp, nm, np, ctu_threshold,
                    scheme, amr_interface_scheme,
                    limiter.excess, limiter.flush,
                )
                sign = @inbounds component.advection.dual_signs[pointer]
                value += rho * sign * (
                    sign > 0 ? minus_difference : plus_difference
                ) / _phase_dual_volume(schedule, output)
            end
        end
        @inbounds delta[output] = value
    end
    # WY's directional dilation term, with the phase density applied before
    # the centred-to-staggered interpolation as required by compatibility.
    _parallel_block_columns(mesh) do b, first_j, last_j
        indicator = @inbounds work.indicator.blocks[b]
        outgoing = @inbounds work.outgoing.blocks[b]
        incoming = @inbounds work.incoming.blocks[b]
        @inbounds for j in first_j:last_j, i in 1:nx
            ii, jj = i + halo, j + halo
            chi = indicator[ii, jj]
            rhochi = density[1] * chi + density[2] * (1 - chi)
            incoming[ii, jj] = rhochi * outgoing[ii, jj]
        end
    end
    _phase_face_average!(component.advection.rates, work.incoming, field.geometry)
    rates = _pack_phase_faces!(component.rates, component.advection.rates)
    @_parallel for i in 1:length(values)
        rhonew = density[1] * newc[i] + density[2] * (1 - newc[i])
        values[i] += (delta[i] + rates[i] * (sources[i] - values[i])) / rhonew
    end
    _unpack_phase_faces!(q, values)
    return component
end
