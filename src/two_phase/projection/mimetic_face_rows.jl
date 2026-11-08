# CMDGFM-selected face-row layout for the jointly fitted projection.

function _local_joint_reconstruction(
        solver::MimeticProjectionSolver{T, _MimeticJointFace},
        data, active_faces, relative_diagonal,
    ) where {T}
    # CMDGFM's beta only chooses *where* normal continuity is useful.
    # The fitted row remains a physical normal observation and enters the
    # same symmetric KKT system as the cell-row variant.
    face_rows = _local_joint_face_rows(
        solver, data, active_faces, relative_diagonal,
    )
    return (;
        normal = face_rows.candidate_constraint,
        tangent = spzeros(T, face_rows.candidate_count, length(active_faces)),
        candidate_cells = face_rows.candidate_locations,
        candidate_scales = face_rows.candidate_scales,
        underresolved = face_rows.underresolved,
    )
end

function _local_joint_tangent_moments(
        ::MimeticProjectionSolver{T, _MimeticJointFace},
        reconstruction, normal_moments, constant_samples,
    ) where {T}
    return hcat(-normal_moments[:, 2], normal_moments[:, 1])
end

@inline function _local_joint_row_geometry(
        ::MimeticProjectionSolver{T, _MimeticJointFace},
        mesh, references, centres, location,
    ) where {T}
    face_location = mesh.faces[location]
    return face_location.center, mesh.blocks[face_location.minus.block].spacing
end

@inline function _local_joint_row_block(
        ::MimeticProjectionSolver{T, _MimeticJointFace},
        mesh, references, location,
    ) where {T}
    return mesh.faces[location].minus.block
end

"""Select a face whose axis sees most of the local interface normal.

This selects where a normal row is useful; the row itself must still use
*both* components of the interface normal, even when one is O(h).
"""
@inline function _local_face_row_axis_aligned(mesh, face, q)
    T = eltype(q)
    magnitude = hypot(q[1], q[2])
    magnitude > zero(T) || return false
    spacing = mesh.blocks[face.minus.block].spacing
    h = max(spacing...)
    length_scale = min(mesh.extent...)
    angular_tolerance = min(one(T), h / length_scale)
    # Setting a single Cartesian jump to zero represents n dot j = 0 only
    # when the omitted normal component is O(h). This criterion makes that
    # fallback a first-order normal equation on smooth, resolved interfaces.
    return abs(q[3 - face.axis]) <= angular_tolerance * magnitude
end

"""One interface-normal jump row on physical staggered faces.

The transverse component is interpolated from faces meeting the endpoints
of `face`, with the same continuous c(1-c) weights used by the resolved
face stencils. A Cartesian face-only row is returned only when the actual
interface normal is axis-aligned to roundoff. Otherwise missing transverse
support means no condition can be imposed at this location.
"""
function _local_physical_normal_row(
        mesh, face_index, q, active_lookup, vertex_patches, data,
    )
    T = eltype(q)
    face = mesh.faces[face_index]
    qnorm = hypot(q...)
    qnorm > zero(T) || return nothing
    axial = q[face.axis] / qnorm
    transverse = q[3 - face.axis] / qnorm
    abs(axial) > sqrt(eps(T)) || return nothing
    result = Tuple{Int, T}[(active_lookup[face_index], axial)]
    abs(transverse) <= T(64) * eps(T) && return result
    neighbours = _cmdgfm_vertex_candidates(
        vertex_patches, mesh, face, face_index, 3 - face.axis,
    )
    filter!(index -> active_lookup[index] > 0, neighbours)
    isempty(neighbours) && return nothing
    weights = [
        data.liquid_fraction[index] *
            (one(T) - data.liquid_fraction[index])
            for index in neighbours
    ]
    total = sum(weights)
    total > zero(T) || return nothing
    for (index, weight) in zip(neighbours, weights)
        push!(result, (active_lookup[index], transverse * weight / total))
    end
    return result
end

"""Build the `:joint_face` seed rows from CMDGFM's beta classification.

Beta selects normal-dominant locations; the row still contains the complete
interface-normal jump. Tangent-dominant faces are left to KKT stationarity.
The fallback uses the aperture-interpolated normal when a stencil is missing.
"""
function _local_joint_face_rows(
        solver::MimeticProjectionSolver{T, _MimeticJointFace},
        data, active_faces,
        relative_diagonal,
    ) where {T}
    mesh = solver.interface.mesh
    cmd_solver = CMDGFMProjectionSolver(
        solver.interface; density = solver.density,
    )
    ax, ay = _local_integrated_aperture_vectors(solver.interface, data)
    stencils = Dict(
        stencil.face => stencil for stencil in
            _cmdgfm_jump_stencils(cmd_solver, data)
    )
    lookup = zeros(Int, length(mesh.faces))
    vertex_patches = _vertex_patches(mesh, T)
    for (local_index, face) in pairs(active_faces)
        lookup[face] = local_index
    end
    rows = Int[]
    columns = Int[]
    values = T[]
    locations = Int[]
    scales = T[]
    underresolved = 0
    for face_index in active_faces
        face = mesh.faces[face_index]
        stencil = get(stencils, face_index, nothing)
        if isnothing(stencil) || !stencil.resolved
            underresolved += 1
            q = _local_face_aperture_vector(
                mesh, face, ax, ay,
            )
            _local_face_row_axis_aligned(mesh, face, q) || continue
            physical_row = _local_physical_normal_row(
                mesh, face_index, q, lookup, vertex_patches, data,
            )
            isnothing(physical_row) && continue
            row = length(locations) + 1
            for (local_index, value) in physical_row
                push!(rows, row)
                push!(columns, local_index)
                push!(values, value)
            end
            push!(locations, face_index)
            push!(scales, one(T))
            continue
        end
        stencil.beta >= T(0.5) || continue
        available = all(
            candidate -> lookup[candidate] > 0,
            stencil.transverse_faces,
        )
        if !available
            underresolved += 1
            continue
        end
        row = length(locations) + 1
        push!(rows, row)
        push!(columns, lookup[face_index])
        push!(values, one(T))
        for (candidate, weight) in zip(
                stencil.transverse_faces, stencil.transverse_weights,
            )
            push!(rows, row)
            push!(columns, lookup[candidate])
            push!(values, stencil.ratio * weight)
        end
        push!(locations, face_index)
        push!(scales, hypot(one(T), stencil.ratio))
    end
    candidate = sparse(
        rows, columns, values, length(locations), length(active_faces),
    )
    retained = _local_independent_rows(
        candidate, relative_diagonal, solver.rank_tolerance, T,
    )
    return (;
        constraint = candidate[retained, :],
        candidate_constraint = candidate,
        candidate_locations = locations,
        constraint_locations = locations[retained],
        constraint_scales = scales[retained],
        candidate_scales = scales,
        candidate_count = length(locations),
        dependent_count = length(locations) - length(retained),
        underresolved,
    )
end

@inline function _local_face_aperture_vector(
        mesh::BlockTreeMesh, face::InteriorFace, x, y,
    )
    nx, ny = mesh.block_size
    minus = _cell_index(face.minus, nx, ny)
    plus = _cell_index(face.plus, nx, ny)
    hm = mesh.blocks[face.minus.block].spacing[face.axis]
    hp = mesh.blocks[face.plus.block].spacing[face.axis]
    denominator = hm + hp
    return (
        (hm * x[minus] + hp * x[plus]) / denominator,
        (hm * y[minus] + hp * y[plus]) / denominator,
    )
end
