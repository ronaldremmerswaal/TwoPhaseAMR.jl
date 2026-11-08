using TwoPhaseAMR

function _component_volumes_by_global_face(
        mesh::TwoPhaseAMR.BlockTreeMesh, component_volumes, ::Type{T};
        covered_value = one(T),
    ) where {T}
    result = Vector{T}(undef, length(mesh.faces))
    for axis in 1:2
        global_to_local, _ = TwoPhaseAMR._component_face_indices(mesh, axis)
        local_volumes = component_volumes[axis]
        for global_index in eachindex(mesh.faces)
            mesh.faces[global_index].axis == axis || continue
            local_index = global_to_local[global_index]
            value = local_volumes[local_index]
            @inbounds result[global_index] = value > zero(T) ?
                value : covered_value
        end
    end
    return result
end
