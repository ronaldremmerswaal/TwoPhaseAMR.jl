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
