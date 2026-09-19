# Two-velocity extensions to the shared phase-momentum workspace protocol.
"""Reusable intrinsic-velocity storage for two-velocity momentum transport.

Each component is represented by the existing `PhaseFaceScalar`, so its two
phases share one dual-flux schedule. The fields store velocities throughout;
phase-weighted momentum is formed only in the local conservative update.
"""
struct TwoVelocityMomentumAdvectionWorkspace{
        T, M <: BlockTreeMesh, X, Y,
    }
    interface::InterfaceField{T, M}
    x::X
    y::Y
end

function _reconfigure_momentum_workspace!(
        workspace::TwoVelocityMomentumAdvectionWorkspace{T},
        interface::InterfaceField{T, M},
    ) where {T, M}
    workspace.interface === interface || return (
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
    return work
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

_finish_phase_momentum!(
    destination::TwoPhaseVelocity,
    workspace::TwoVelocityMomentumAdvectionWorkspace,
) = _copy_momentum_velocity!(destination, workspace)
