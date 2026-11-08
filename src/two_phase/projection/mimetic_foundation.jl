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
        cell = _cell_index(face.cell, nx, ny)
        factor = T(face.side) * T(face.area)
        liquid = velocity[1].boundary[index]
        gas = velocity[2].boundary[index]
        @inbounds begin
            mixture[cell] += factor * _projection_phase_sum(
                liquid_aperture, liquid, gas_aperture, gas,
            )
            # A jump exists only when both intrinsic boundary values exist.
            if isfinite(liquid) && isfinite(gas)
                normal_jump[cell] += factor *
                    (gas_aperture - liquid_aperture) * (gas - liquid)
            end
        end
    end
    return mixture, normal_jump
end


"""Compute the integrated mixture and normal-jump constraints."""
function _mimetic_integrated_constraints(
        incidence, data, interface, velocity, mixed_faces,
    )
    liquid = velocity[1].interior
    gas = velocity[2].interior
    mixture = _projection_phase_sum.(
        data.liquid_aperture, liquid, data.gas_aperture, gas,
    )
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

