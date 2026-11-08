# Physical phase values and absent storage have different contracts. Numerical
# comparisons use only active faces; the excluded entries must remain NaN.
if !isdefined(@__MODULE__, :_phase_interior_indices)
    function _phase_interior_indices(interface, phase)
        name = phase == 1 ? :liquid : :gas
        return findall(eachindex(interface.mesh.faces)) do index
            TwoPhaseAMR._phase_face_is_active(interface, index, interface.mesh.faces[index], name)
        end
    end

    function _phase_boundary_indices(interface, phase)
        name = phase == 1 ? :liquid : :gas
        return findall(
            face -> TwoPhaseAMR._phase_cell_is_active(interface, face.cell, name),
            interface.mesh.boundary_faces
        )
    end

    function _phase_storage_contract(velocity, interface)
        for (phase, name) in ((1, :liquid), (2, :gas))
            for (index, face) in pairs(interface.mesh.faces)
                active = TwoPhaseAMR._phase_face_is_active(interface, index, face, name)
                value = velocity[phase].interior[index]
                (active ? isfinite(value) : isnan(value)) || return false
            end
            for (index, face) in pairs(interface.mesh.boundary_faces)
                active = TwoPhaseAMR._phase_cell_is_active(interface, face.cell, name)
                value = velocity[phase].boundary[index]
                (active ? isfinite(value) : isnan(value)) || return false
            end
        end
        return true
    end
end
