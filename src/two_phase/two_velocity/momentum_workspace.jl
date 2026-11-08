# Two-velocity extensions to the shared phase-momentum workspace protocol.
"""Reusable intrinsic-velocity storage for two-velocity momentum transport.

Each component is represented by the existing `PhaseFaceScalar`, so its two
phases share one dual-flux schedule. The fields store velocities throughout;
phase-weighted momentum is formed only in the local conservative update.
"""
struct TwoVelocityMomentumAdvectionWorkspace{
        T, M <: BlockTreeMesh, X, Y, G,
    }
    interface::InterfaceField{T, M, G}
    x::X
    y::Y
end

function _reconfigure_momentum_workspace!(
        workspace::TwoVelocityMomentumAdvectionWorkspace{T},
        interface::InterfaceField{T, M},
    ) where {T, M}
    mesh = interface.mesh
    first_level = first(mesh.blocks).key.level
    uniform = isnothing(interface.geometry) && all(mesh.periodic) &&
        all(block -> block.key.level == first_level, mesh.blocks)
    old_uniform = workspace.x.workspace.schedule isa UniformDualAdvectionSchedule
    # The schedule type is intentionally concrete in the transport hot loop.
    # If AMR changes it, recreate transport scratch only; the caller can still
    # retain the pressure/Krylov workspace and its geometric templates.
    (workspace.interface === interface && uniform == old_uniform) || return (
        TwoVelocityMomentumAdvectionWorkspace(
            interface, PhaseFaceScalar{1}(interface),
            PhaseFaceScalar{2}(interface),
        )
    )
    for component in (workspace.x, workspace.y)
        _resize_face_field!(component.liquid, interface.mesh)
        _resize_face_field!(component.gas, interface.mesh)
        _finish_phase_face_transfer!(component)
    end
    return TwoVelocityMomentumAdvectionWorkspace(
        interface, workspace.x, workspace.y,
    )
end

function _prepare_momentum_advection!(
        work::TwoVelocityMomentumAdvectionWorkspace,
        velocity::Tuple{<:FaceVelocity, <:FaceVelocity},
    )
    for (field, source) in (
            (work.x.liquid, velocity[1]),
            (work.y.liquid, velocity[1]),
            (work.x.gas, velocity[2]),
            (work.y.gas, velocity[2]),
        )
        copyto!(field, source)
    end
    for component in (work.x, work.y)
        _prepare_phase_momentum_values!(component, work.interface)
    end
    return work
end

function _prepare_phase_momentum_values!(component, interface::InterfaceField)
    geometry = interface.geometry
    _phase_face_average!(component.fraction, interface.conditional_fraction, geometry)
    fractions = _pack_phase_faces!(component.scratch.liquid_fraction, component.fraction)
    for index in eachindex(fractions)
        fractions[index] = clamp(fractions[index], zero(eltype(fractions)), one(eltype(fractions)))
    end
    interior_count = length(component.fraction.interior)
    for (field, liquid) in ((component.liquid, true), (component.gas, false))
        values = _pack_phase_faces!(component.scratch.values, field)
        for index in eachindex(values)
            boundary = index > interior_count
            local_index = boundary ? index - interior_count : index
            # Schedule volumes replace a closed volume by one to protect
            # scratch divisions. Only the *physical* volume decides activity.
            open = _staggered_open_volume(
                interface, geometry, component.fraction,
                local_index; boundary
            ) > 0
            present = open && (liquid ? fractions[index] > 0 : fractions[index] < 1)
            if present && !isfinite(values[index])
                throw(ArgumentError("non-finite active phase velocity in momentum transport"))
            elseif !present
                values[index] = eltype(values)(NaN)
            end
        end
        _extend_phase_face_values!(values, component.scratch.increment, component.workspace)
        _unpack_phase_faces!(field, values)
    end
    return component
end

function _copy_momentum_velocity!(
        destination::Tuple{<:FaceVelocity, <:FaceVelocity},
        work::TwoVelocityMomentumAdvectionWorkspace,
    )
    copyto!(destination[1], work.x.liquid)
    copyto!(destination[1], work.y.liquid)
    copyto!(destination[2], work.x.gas)
    copyto!(destination[2], work.y.gas)
    return destination
end

function _validate_phase_momentum!(
        momentum::TwoPhaseVelocity, source::TwoPhaseVelocity,
        workspace::TwoVelocityMomentumAdvectionWorkspace,
        field::InterfaceField, ::Type{T},
    ) where {T}
    _validate_two_phase_velocity(momentum, field.mesh, T)
    _validate_two_phase_velocity(source, field.mesh, T)
    workspace.interface === field || throw(
        ArgumentError("momentum workspace belongs to a different interface"),
    )
    _prepare_momentum_advection!(workspace, source)
    return workspace
end

_phase_momentum_sources(workspace::TwoVelocityMomentumAdvectionWorkspace) = (
    _phase_face_sources(workspace.x),
    _phase_face_sources(workspace.y),
)

function _advect_phase_momentum!(
        workspace::TwoVelocityMomentumAdvectionWorkspace, work, field, axis,
        sources, density, scheme, amr_interface_scheme; auxiliary_gas,
    )
    _advect_phase_faces!(
        workspace.x, work, axis, sources[1], work.ctu_threshold, scheme,
        amr_interface_scheme; auxiliary_gas,
        gas_momentum_component = 1, gas_density = density[2],
    )
    _advect_phase_faces!(
        workspace.y, work, axis, sources[2], work.ctu_threshold, scheme,
        amr_interface_scheme; auxiliary_gas,
        gas_momentum_component = 2, gas_density = density[2],
    )
    return workspace
end

function _finish_phase_momentum!(
        destination::TwoPhaseVelocity,
        workspace::TwoVelocityMomentumAdvectionWorkspace,
    )
    _copy_momentum_velocity!(destination, workspace)
    return _mark_inactive_phase_velocities!(destination, workspace.interface)
end
