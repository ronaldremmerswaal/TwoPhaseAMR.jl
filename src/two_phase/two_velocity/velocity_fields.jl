
"""Validate the storage contract shared by all two-velocity entry points."""

function _validate_two_phase_velocity(
        velocity::TwoPhaseVelocity, mesh::BlockTreeMesh, ::Type{T},
    ) where {T}
    for (phase, component) in zip((:liquid, :gas), velocity)
        component.mesh === mesh || throw(
            ArgumentError(
                "$phase velocity and the two-phase field must share a mesh",
            )
        )
        eltype(component) === T || throw(
            ArgumentError(
                "$phase velocity has element type $(eltype(component)); expected $T",
            )
        )
        length(component.interior) == length(mesh.faces) || throw(
            DimensionMismatch("$phase velocity has the wrong interior-face layout"),
        )
        length(component.boundary) == length(mesh.boundary_faces) || throw(
            DimensionMismatch("$phase velocity has the wrong boundary-face layout"),
        )
    end
    return velocity
end

"""Set both phase velocities from `(liquid_function, gas_function)`."""
function set_velocities!(
        velocity::TwoPhaseVelocity, functions::Tuple; time = 0,
    )
    length(functions) == 2 || throw(
        ArgumentError(
            "two phase-velocity initializers are required",
        )
    )
    liquid, gas = velocity
    _validate_two_phase_velocity(velocity, liquid.mesh, eltype(liquid))
    set_velocities!(liquid, functions[1]; time)
    set_velocities!(gas, functions[2]; time)
    return velocity
end

"""Apply the impermeable-domain condition to both phase velocities."""
function enforce_no_permeation!(velocity::TwoPhaseVelocity)
    foreach(enforce_no_permeation!, velocity)
    return velocity
end

"""Apply the fixed-solid no-slip mask to both phase velocities."""
function enforce_geometry_no_slip!(
        velocity::TwoPhaseVelocity, geometry::GeometryField,
    )
    _validate_two_phase_velocity(
        velocity, geometry.mesh, eltype(geometry),
    )
    foreach(component -> enforce_geometry_no_slip!(component, geometry), velocity)
    return velocity
end

"""
Register both intrinsic velocity fields for AMR transfer.

The existing face interpolation and divergence-preserving refinement are used
for each phase.  The global face-integral correction is disabled: these fields
store velocity, not phase momentum.
"""
function register_adaptive_field!(
        workspace::AMRWorkspace{T}, velocity::TwoPhaseVelocity,
    ) where {T}
    liquid, gas = velocity
    _validate_two_phase_velocity(
        velocity, workspace.mesh, T,
    )
    for component in (liquid, gas)
        register_adaptive_field!(
            workspace, component;
            piecewise_constant = false, conserve_integral = false,
        )
    end
    return workspace
end
