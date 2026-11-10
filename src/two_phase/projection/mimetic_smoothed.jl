# Smoothed-aperture cell-row layout for the jointly fitted projection.

function _local_joint_reconstruction(
        solver::MimeticProjectionSolver{T, _MimeticJointSmoothed},
        data, active_faces, relative_diagonal,
    ) where {T}
    return _local_aperture_normal_tangent_reconstruction(
        solver, data, active_faces,
    )
end

function _local_joint_tangent_moments(
        ::MimeticProjectionSolver{T, _MimeticJointSmoothed},
        reconstruction, normal_moments, constant_samples,
    ) where {T}
    return reconstruction.tangent * constant_samples
end

@inline function _local_joint_row_geometry(
        ::MimeticProjectionSolver{T, _MimeticJointSmoothed},
        mesh, references, centres, location,
    ) where {T}
    reference = references[location]
    return centres[location], mesh.blocks[reference.block].spacing
end

@inline function _local_joint_row_block(
        ::MimeticProjectionSolver{T, _MimeticJointSmoothed},
        mesh, references, location,
    ) where {T}
    return references[location].block
end

"""One-ring, orientation-preserving sum of cut-cell aperture normals.

The cell vectors orient the default normal and tangential observations.
Only mixed neighbours whose raw integrated normals have a positive
dot product are added. This prevents cancellation across a folded sheet and
retains the interface-length weighting already present in `D(a_l n)`.
"""
function _local_one_ring_aperture_vectors(
        mesh::BlockTreeMesh, ax, ay, references, mixed_cells,
    )
    T = eltype(ax)
    smooth_ax = copy(ax)
    smooth_ay = copy(ay)
    neighbours = [Int[] for _ in eachindex(references)]
    nx, ny = mesh.block_size
    for face in mesh.faces
        minus = _cell_index(face.minus, nx, ny)
        plus = _cell_index(face.plus, nx, ny)
        push!(neighbours[minus], plus)
        push!(neighbours[plus], minus)
    end
    for cell in eachindex(references)
        mixed_cells[cell] || continue
        for adjacent in neighbours[cell]
            mixed_cells[adjacent] || continue
            ax[cell] * ax[adjacent] + ay[cell] * ay[adjacent] > zero(T) ||
                continue
            smooth_ax[cell] += ax[adjacent]
            smooth_ay[cell] += ay[adjacent]
        end
    end
    return smooth_ax, smooth_ay
end

"""Build aperture-based normal and tangential jump measurements.

For a mixed cell ``c``, let ``eta_c = D(a_l n)_c`` be its integrated
aperture Gauss vector and let ``J¹_c`` reconstruct the two Cartesian jump
components from active staggered faces.  This routine assembles

    N_a j = eta_x J¹_x j + eta_y J¹_y j,
    T   j = -eta_y J¹_x j + eta_x J¹_y j.

Both observations retain the scale `|eta|`, which is the interface length for
a planar cut. `T` is used only to fit the normal rows;
it is not an additional equation in the pressure solve.

`J¹_c` is a purely geometric, constant-reproducing average of the active
face-normal jump samples of each Cartesian component. In particular this
construction does not use volume-fraction-product weights. Its only interface
geometry is the aperture Gauss vector `eta = D(a_l n)`; no PLIC normal or
interface endpoint enters the operator.
"""
function _local_aperture_normal_tangent_reconstruction(
        solver::MimeticProjectionSolver{T, _MimeticJointSmoothed},
        data, active_faces,
    ) where {T}
    interface = solver.interface
    mesh = interface.mesh
    active_lookup = zeros(Int, length(mesh.faces))
    for (local_index, face_index) in pairs(active_faces)
        active_lookup[face_index] = local_index
    end
    aperture_x, aperture_y = _local_integrated_aperture_vectors(
        interface, data,
    )
    incident_faces = _cmdgfm_incident_faces(mesh)
    references = _cell_references(mesh)
    mixed = [
        begin
            fraction = _conditional_fraction(interface, reference)
            interface.fraction_floor < fraction <
                one(T) - interface.fraction_floor
        end for reference in references
    ]
    smooth_x, smooth_y = _local_one_ring_aperture_vectors(
        mesh, aperture_x, aperture_y, references, mixed,
    )
    embedded_cells = _local_embedded_cells(interface)
    normal_rows = Int[]
    normal_columns = Int[]
    normal_values = T[]
    tangent_rows = Int[]
    tangent_columns = Int[]
    tangent_values = T[]
    candidate_cells = Int[]
    candidate_scales = T[]
    tangent_resolved = Bool[]
    underresolved = 0
    coefficient_tolerance = T(64) * eps(T)
    floor = interface.fraction_floor

    for (cell_index, reference) in pairs(references)
        fraction = _conditional_fraction(interface, reference)
        floor < fraction < one(T) - floor || continue
        embedded_cells[cell_index] && continue
        raw_eta = (aperture_x[cell_index], aperture_y[cell_index])
        measure = hypot(raw_eta...)
        smooth_measure = hypot(
            smooth_x[cell_index], smooth_y[cell_index],
        )
        eta = smooth_measure > zero(T) ? (
                measure * smooth_x[cell_index] / smooth_measure,
                measure * smooth_y[cell_index] / smooth_measure,
            ) : raw_eta
        cell_scale = maximum(mesh.blocks[reference.block].spacing)
        if measure <= coefficient_tolerance * T(cell_scale)
            underresolved += 1
            continue
        end

        axis_columns = (Int[], Int[])
        axis_weights = (T[], T[])
        axis_resolved = trues(2)
        for axis in 1:2
            incident = filter(incident_faces[cell_index]) do face_index
                mesh.faces[face_index].axis == axis
            end
            # The jump space contains only active mixed faces. Normalising
            # their geometric measures makes J¹ reproduce a constant vector
            # exactly on that space, including one-sided/AMR configurations.
            # No volume-fraction-dependent c(1-c) factor is present.
            available = filter(
                face_index -> active_lookup[face_index] > 0, incident,
            )
            weight_sum = sum(
                face_index -> T(mesh.faces[face_index].area), available;
                init = zero(T),
            )
            if weight_sum <= zero(T)
                axis_resolved[axis] = false
                continue
            end
            for face_index in available
                push!(axis_columns[axis], active_lookup[face_index])
                push!(
                    axis_weights[axis],
                    T(mesh.faces[face_index].area) / weight_sum,
                )
            end
        end

        normal_needed = (
            abs(eta[1]) > coefficient_tolerance * measure,
            abs(eta[2]) > coefficient_tolerance * measure,
        )
        tangent_coefficients = (-eta[2], eta[1])
        tangent_needed = (
            abs(tangent_coefficients[1]) > coefficient_tolerance * measure,
            abs(tangent_coefficients[2]) > coefficient_tolerance * measure,
        )
        normal_ok = all(
            !normal_needed[axis] || axis_resolved[axis] for axis in 1:2
        )
        if !normal_ok
            underresolved += 1
            continue
        end

        new_row = length(candidate_cells) + 1
        normal_nonzero = false
        for axis in 1:2
            for (column, weight) in zip(
                    axis_columns[axis], axis_weights[axis],
                )
                normal_value = eta[axis] * weight
                if abs(normal_value) > coefficient_tolerance * measure
                    push!(normal_rows, new_row)
                    push!(normal_columns, column)
                    push!(normal_values, normal_value)
                    normal_nonzero = true
                end
                tangent_value = tangent_coefficients[axis] * weight
                if abs(tangent_value) > coefficient_tolerance * measure
                    push!(tangent_rows, new_row)
                    push!(tangent_columns, column)
                    push!(tangent_values, tangent_value)
                end
            end
        end
        # If every contributing jump belongs to the active-set complement,
        # the homogeneous zero extension already supplies this equation.
        normal_nonzero || continue
        push!(candidate_cells, cell_index)
        push!(candidate_scales, measure)
        push!(
            tangent_resolved,
            all(
                !tangent_needed[axis] || axis_resolved[axis]
                    for axis in 1:2
            ),
        )
    end

    row_count = length(candidate_cells)
    face_count = length(active_faces)
    return (;
        normal = sparse(
            normal_rows, normal_columns, normal_values,
            row_count, face_count,
        ),
        tangent = sparse(
            tangent_rows, tangent_columns, tangent_values,
            row_count, face_count,
        ),
        candidate_cells, candidate_scales, tangent_resolved,
        underresolved,
    )
end
