# Shared stage utilities. Model domains extend the `_..._substep!` protocol
# after the equation-system and workspace types have been defined.
function _copy_velocity!(destination::FaceVelocity, source::FaceVelocity)
    destination.mesh === source.mesh ||
        throw(ArgumentError("velocities must share a mesh"))
    copyto!(destination.interior, source.interior)
    copyto!(destination.boundary, source.boundary)
    return destination
end

@inline _body_force(::Nothing, face, time) = zero(typeof(face.area))
@inline _body_force(force, face, time) =
    force(face.center[1], face.center[2], time, face.axis)

"""Add the external acceleration to the transported predictor.

The intrinsic workspace is cached through `Ref{Any}` because its transport
type can change at AMR. Pass the actual velocity fields into this function
once: do not access that dynamic cache inside a face loop. Explicit `F`
specialization also keeps forwarded callable forces concrete. `nothing` is
a true no-op, rather than a full traversal adding zero to every face.
"""
_body_force_substep!(velocity, dt, midpoint, ::Nothing) = velocity

function _body_force_substep!(velocity, dt, midpoint, force::F) where {F}
    return _add_body_force!(velocity, dt, midpoint, force)
end

function _add_body_force!(velocity::FaceVelocity, dt, midpoint, force::F) where {F}
    return _add_body_force!(velocity, dt, midpoint, force, Val(false))
end

function _add_body_force!(velocity::TwoPhaseVelocity, dt, midpoint, force::F) where {F}
    # Retain phase-major/face-major order: a user callback can have state.
    # Inactive intrinsic values stay NaN and never evaluate the callback.
    for phase in velocity
        _add_body_force!(phase, dt, midpoint, force, Val(true))
    end
    return velocity
end

function _add_body_force!(velocity::FaceVelocity, dt, midpoint, force::F,
        ::Val{AllowAbsent}) where {F, AllowAbsent}
    values = velocity.interior
    for (index, face) in pairs(velocity.mesh.faces)
        @inbounds begin
            AllowAbsent && isnan(values[index]) && continue
            values[index] += dt * _body_force(force, face, midpoint)
        end
    end
    # Boundary velocities are still set by the established boundary stage.
    return velocity
end


function _navier_stokes_cycle_limit(solver, max_cycles)
    isnothing(max_cycles) && return isnothing(solver.two_phase) ? 100 : 500
    max_cycles isa Integer || throw(
        ArgumentError("max_cycles must be an integer or nothing"),
    )
    max_cycles >= 0 || throw(
        ArgumentError("max_cycles must be nonnegative"),
    )
    return Int(max_cycles)
end
