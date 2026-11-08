function _copy_geometry_snapshot!(workspace::AMRWorkspace)
    empty!(workspace.old_blocks)
    append!(workspace.old_blocks, workspace.mesh.blocks)
    empty!(workspace.old_faces)
    append!(workspace.old_faces, workspace.mesh.faces)
    empty!(workspace.old_boundary_faces)
    append!(workspace.old_boundary_faces, workspace.mesh.boundary_faces)
    return workspace
end

function _rebuild_mesh_schedules!(workspace::AMRWorkspace)
    mesh = workspace.mesh
    nodes, blocks = _leaf_blocks(mesh.root, mesh.block_size)
    faces, boundaries = _build_face_schedules(
        blocks, mesh.base_mesh, mesh.block_size, mesh.origin, mesh.extent,
        mesh.periodic
    )
    halo_entries, halo_donors, halo_weights = _build_halo_schedule(
        mesh.root, nodes, blocks, mesh.block_size, mesh.halo,
        mesh.origin, mesh.extent, mesh.periodic
    )
    empty!(mesh.blocks)
    append!(mesh.blocks, blocks)
    empty!(mesh.faces)
    append!(mesh.faces, faces)
    # Face order can change even when its length does not. Refresh this
    # topology-dependent cache on every schedule rebuild, not by a size test.
    empty!(mesh.face_dual_volumes)
    append!(mesh.face_dual_volumes, _face_dual_volumes(blocks, faces))
    empty!(mesh.boundary_faces)
    append!(mesh.boundary_faces, boundaries)
    empty!(mesh.halo_entries)
    append!(mesh.halo_entries, halo_entries)
    empty!(mesh.halo_donors)
    append!(mesh.halo_donors, halo_donors)
    empty!(mesh.halo_weights)
    append!(mesh.halo_weights, halo_weights)
    _invalidate_mesh_topology_cache!(mesh)
    return mesh
end

function _acquire_cell_block!(buffer::CellAMRFieldBuffer)
    isempty(buffer.pool) || return pop!(buffer.pool)
    field = buffer.field
    nx, ny = field.mesh.block_size
    h = field.mesh.halo
    return Matrix{eltype(field)}(undef, nx + 2h, ny + 2h)
end

function _prolong_child!(child, parent, halo, child_key::BlockKey)
    nx = size(parent, 1) - 2halo
    ny = size(parent, 2) - 2halo
    child_x = mod(child_key.i, 2)
    child_y = mod(child_key.j, 2)
    for j in 1:ny, i in 1:nx
        fine_i = child_x * nx + i
        fine_j = child_y * ny + j
        parent_i = (fine_i + 1) ÷ 2
        parent_j = (fine_j + 1) ÷ 2
        @inbounds begin
            value = parent[parent_i + halo, parent_j + halo]
            slope_x = _mc_slope(
                value - parent[parent_i + halo - 1, parent_j + halo],
                parent[parent_i + halo + 1, parent_j + halo] - value
            )
            slope_y = _mc_slope(
                value - parent[parent_i + halo, parent_j + halo - 1],
                parent[parent_i + halo, parent_j + halo + 1] - value
            )
            offset_x = isodd(fine_i) ? -one(value) / 4 : one(value) / 4
            offset_y = isodd(fine_j) ? -one(value) / 4 : one(value) / 4
            child[i + halo, j + halo] =
                value + offset_x * slope_x + offset_y * slope_y
        end
    end
    return child
end

function _transfer_cell_field!(
        buffer::CellAMRFieldBuffer,
        workspace::AMRWorkspace
    )
    field = buffer.field
    empty!(workspace.old_indices)
    for (index, block) in pairs(workspace.old_blocks)
        workspace.old_indices[block.key] = index
    end
    empty!(workspace.new_indices)
    for (index, block) in pairs(workspace.mesh.blocks)
        workspace.new_indices[block.key] = index
    end
    resize!(workspace.reused_old_blocks, length(workspace.old_blocks))
    fill!(workspace.reused_old_blocks, false)

    for block in workspace.mesh.blocks
        old_index = get(workspace.old_indices, block.key, 0)
        if !iszero(old_index)
            push!(field.blocks, buffer.staging[old_index])
            workspace.reused_old_blocks[old_index] = true
        else
            push!(field.blocks, _acquire_cell_block!(buffer))
        end
    end

    h = workspace.mesh.halo
    _prepare_parallel_prolongation!(buffer.prolongation, workspace)
    _parallel_for(nblocks(workspace.mesh); grain = 4) do new_index
        block = workspace.mesh.blocks[new_index]
        haskey(workspace.old_indices, block.key) && return nothing
        if block.key.level > 0
            parent_key = _parent_key(block.key)
            old_parent = get(workspace.old_indices, parent_key, 0)
            if !iszero(old_parent)
                buffer.prolongation(
                    field.blocks[new_index],
                    buffer.staging[old_parent], h, block.key
                )
                return nothing
            end
        end
        children = _child_keys(block.key)
        child_indices = ntuple(
            index -> get(workspace.old_indices, children[index], 0), 4
        )
        all(index -> !iszero(index), child_indices) || throw(
            ArgumentError(
                "mesh adaptation changed by more than one level in one update"
            )
        )
        _restrict_parent!(
            buffer.prolongation, field.blocks[new_index],
            ntuple(index -> buffer.staging[child_indices[index]], 4), h,
            block.key
        )
    end

    for index in eachindex(buffer.staging)
        workspace.reused_old_blocks[index] && continue
        push!(buffer.pool, buffer.staging[index])
    end
    empty!(buffer.staging)
    fill_halo!(field, buffer.boundary)
    return field
end

function _stage_field!(
        buffer::CellAMRFieldBuffer, workspace::AMRWorkspace
    )
    _prepare_prolongation!(buffer.prolongation, buffer.field, workspace)
    fill_halo!(buffer.field, buffer.boundary)
    empty!(buffer.staging)
    append!(buffer.staging, buffer.field.blocks)
    empty!(buffer.field.blocks)
    return buffer
end

_prepare_prolongation!(prolongation, field, workspace) = nothing
_prepare_parallel_prolongation!(prolongation, workspace) = nothing
_finish_prolongation!(prolongation) = nothing
_restrict_parent!(prolongation, parent, children, halo, key) =
    restrict!(parent, children, halo)

function _stage_field!(buffer::FaceAMRFieldBuffer, workspace::AMRWorkspace)
    # Two-phase source buffers were prepared from valid phase data before
    # cell staging. Their finite AMR extension must not overwrite the owned
    # NaN-marked velocity, nor be overwritten by a fresh raw-field copy here.
    for (liquid, gas, _) in workspace.phase_velocities
        (buffer.field === liquid || buffer.field === gas) && return buffer
    end
    resize!(buffer.old_interior, length(buffer.field.interior))
    resize!(buffer.old_boundary, length(buffer.field.boundary))
    copyto!(buffer.old_interior, buffer.field.interior)
    copyto!(buffer.old_boundary, buffer.field.boundary)
    return buffer
end

@inline function _face_lattice_geometry(mesh::BlockTreeMesh, face)
    axis = face.axis
    tangent_axis = 3 - axis
    lattice_size = _finest_lattice_size(mesh)
    normal_spacing = mesh.extent[axis] / lattice_size[axis]
    tangent_spacing = mesh.extent[tangent_axis] / lattice_size[tangent_axis]
    normal = round(
        Int, (face.center[axis] - mesh.origin[axis]) /
            normal_spacing
    )
    tangent_center = (face.center[tangent_axis] - mesh.origin[tangent_axis]) /
        tangent_spacing
    half_length = face.area / (2tangent_spacing)
    lo = round(Int, tangent_center - half_length)
    hi = round(Int, tangent_center + half_length)
    return normal, lo, hi
end

function _build_face_transfer_geometry!(workspace::AMRWorkspace)
    for axis in 1:2
        for line in values(workspace.face_lines[axis])
            empty!(line)
            push!(workspace.face_line_pool[axis], line)
        end
        empty!(workspace.face_lines[axis])
        empty!(workspace.face_line_coordinates[axis])
    end
    local_indices = zeros(Int, 2)
    for (source_index, face) in pairs(workspace.old_faces)
        axis = face.axis
        local_indices[axis] += 1
        normal, lo, hi = _face_lattice_geometry(workspace.mesh, face)
        line = get!(workspace.face_lines[axis], normal) do
            isempty(workspace.face_line_pool[axis]) ?
                FaceTransferSegment[] : pop!(workspace.face_line_pool[axis])
        end
        push!(
            line, FaceTransferSegment(
                lo, hi, source_index, local_indices[axis], false
            )
        )
    end
    fill!(local_indices, 0)
    for (source_index, face) in pairs(workspace.old_boundary_faces)
        axis = face.axis
        local_indices[axis] += 1
        normal, lo, hi = _face_lattice_geometry(workspace.mesh, face)
        line = get!(workspace.face_lines[axis], normal) do
            isempty(workspace.face_line_pool[axis]) ?
                FaceTransferSegment[] : pop!(workspace.face_line_pool[axis])
        end
        push!(
            line, FaceTransferSegment(
                lo, hi, source_index, local_indices[axis], true
            )
        )
    end
    for axis in 1:2
        for line in values(workspace.face_lines[axis])
            sort!(line; by = segment -> segment.lo)
        end
        append!(
            workspace.face_line_coordinates[axis],
            keys(workspace.face_lines[axis])
        )
        sort!(workspace.face_line_coordinates[axis])
    end
    return workspace
end

@inline function _old_face_value(
        buffer::FaceAMRFieldBuffer,
        segment::FaceTransferSegment
    )
    field = buffer.field
    index = field isa FaceVelocity ? segment.source_index : segment.local_index
    return segment.boundary ? buffer.old_boundary[index] :
        buffer.old_interior[index]
end

@inline _segment_center(segment::FaceTransferSegment) =
    (segment.lo + segment.hi) / 2

function _tangential_slope(
        buffer::FaceAMRFieldBuffer, line,
        segment_index, tangent_period, periodic
    )
    segment = line[segment_index]
    value = _old_face_value(buffer, segment)
    centre = _segment_center(segment)
    left_index = segment_index > firstindex(line) ? segment_index - 1 :
        (periodic ? lastindex(line) : 0)
    right_index = segment_index < lastindex(line) ? segment_index + 1 :
        (periodic ? firstindex(line) : 0)

    left_gradient = nothing
    if !iszero(left_index)
        left = line[left_index]
        left_centre = _segment_center(left)
        left_centre >= centre && (left_centre -= tangent_period)
        left_gradient = (value - _old_face_value(buffer, left)) /
            (centre - left_centre)
    end
    right_gradient = nothing
    if !iszero(right_index)
        right = line[right_index]
        right_centre = _segment_center(right)
        right_centre <= centre && (right_centre += tangent_period)
        right_gradient = (_old_face_value(buffer, right) - value) /
            (right_centre - centre)
    end
    isnothing(left_gradient) && isnothing(right_gradient) && return zero(value)
    isnothing(left_gradient) && return right_gradient
    isnothing(right_gradient) && return left_gradient
    return _mc_slope(left_gradient, right_gradient)
end

function _line_value(
        buffer::FaceAMRFieldBuffer, line, lo, hi,
        tangent_period, periodic
    )
    covered = 0
    integral = zero(eltype(buffer.old_interior))
    containing = 0
    contributors = 0
    index = _first_overlapping_line_segment(line, lo)
    while index <= lastindex(line)
        segment = line[index]
        segment.lo >= hi && break
        overlap = min(hi, segment.hi) - max(lo, segment.lo)
        if overlap > 0
            covered += overlap
            integral += overlap * _old_face_value(buffer, segment)
            contributors += 1
            segment.lo <= lo && hi <= segment.hi && (containing = index)
        end
        index += 1
    end
    covered == hi - lo || return nothing
    if contributors == 1 && !iszero(containing)
        segment = line[containing]
        value = _old_face_value(buffer, segment)
        if !buffer.piecewise_constant && (segment.lo != lo || segment.hi != hi)
            slope = _tangential_slope(
                buffer, line, containing, tangent_period, periodic
            )
            value += slope * ((lo + hi) / 2 - _segment_center(segment))
        end
        return value
    end
    return integral / covered
end

function _line_point_value(
        buffer::FaceAMRFieldBuffer, line, tangent,
        tangent_period, periodic
    )
    index = _line_segment_at(line, tangent)
    iszero(index) && return nothing
    segment = line[index]
    value = _old_face_value(buffer, segment)
    slope = buffer.piecewise_constant ? zero(value) : _tangential_slope(
            buffer, line, index, tangent_period, periodic
        )
    return value + slope * (tangent - _segment_center(segment))
end

function _face_transfer_value(
        workspace::AMRWorkspace,
        buffer::FaceAMRFieldBuffer, face
    )
    mesh = workspace.mesh
    axis = face.axis
    tangent_axis = 3 - axis
    normal, lo, hi = _face_lattice_geometry(mesh, face)
    tangent_period = _finest_lattice_size(mesh)[tangent_axis]
    lines = workspace.face_lines[axis]
    if haskey(lines, normal)
        value = _line_value(
            buffer, lines[normal], lo, hi,
            tangent_period, mesh.periodic[tangent_axis]
        )
        isnothing(value) || return value
    end

    tangent = (lo + hi) / 2
    normal_period = _finest_lattice_size(mesh)[axis]
    lower_distance, lower_coordinate, lower_segment,
        upper_distance, upper_coordinate, upper_segment =
        _bracketing_line_segments(
        lines, workspace.face_line_coordinates[axis], normal, tangent,
        normal_period, mesh.periodic[axis],
    )
    lower_value = iszero(lower_segment) ? nothing : _line_point_value(
            buffer, lines[lower_coordinate], tangent,
            tangent_period, mesh.periodic[tangent_axis],
        )
    upper_value = iszero(upper_segment) ? nothing : _line_point_value(
            buffer, lines[upper_coordinate], tangent,
            tangent_period, mesh.periodic[tangent_axis],
        )
    if !isnothing(lower_value) && !isnothing(upper_value)
        fraction = -lower_distance / (upper_distance - lower_distance)
        return lower_value + fraction * (upper_value - lower_value)
    elseif !isnothing(lower_value)
        return lower_value
    elseif !isnothing(upper_value)
        return upper_value
    end
    throw(ArgumentError("could not locate old face data for a new face"))
end

@inline function _dual_face_volume(blocks, face)
    tangent_axis = 3 - face.axis
    result = zero(typeof(face.area))
    for cell in (face.minus, face.plus)
        block = blocks[cell.block]
        fraction = face.area / block.spacing[tangent_axis]
        result += fraction * block.volume / 2
    end
    return result
end

function _old_face_momentum(
        workspace::AMRWorkspace,
        buffer::FaceAMRFieldBuffer, axis
    )
    result = zero(eltype(buffer.old_interior))
    local_index = 0
    for (global_index, face) in pairs(workspace.old_faces)
        face.axis == axis || continue
        local_index += 1
        index = buffer.field isa FaceVelocity ? global_index : local_index
        result += buffer.old_interior[index] *
            _dual_face_volume(workspace.old_blocks, face)
    end
    return result
end

@inline _field_axes(::FaceVelocity) = (1, 2)
@inline _field_axes(::FaceField{Axis}) where {Axis} = (Axis,)

function _resize_face_field!(field::FaceVelocity, mesh)
    resize!(field.interior, length(mesh.faces))
    resize!(field.boundary, length(mesh.boundary_faces))
    return field
end

function _resize_face_field!(field::FaceField{Axis}, mesh) where {Axis}
    resize!(field.interior, _nfaces(mesh, Axis))
    resize!(field.boundary, _nboundary_faces(mesh, Axis))
    return field
end

function _set_transferred_interior!(
        field::FaceVelocity, global_index,
        local_index, value
    )
    return @inbounds field.interior[global_index] = value
end

function _set_transferred_interior!(
        field::FaceField, global_index,
        local_index, value
    )
    return @inbounds field.interior[local_index] = value
end

function _set_transferred_boundary!(
        field::FaceVelocity, global_index,
        local_index, value
    )
    return @inbounds field.boundary[global_index] = value
end

function _set_transferred_boundary!(
        field::FaceField, global_index,
        local_index, value
    )
    return @inbounds field.boundary[local_index] = value
end

function _conserve_face_momentum!(field, axis, target, mesh)
    current = zero(typeof(target))
    total_volume = zero(typeof(target))
    local_index = 0
    for (global_index, face) in pairs(mesh.faces)
        face.axis == axis || continue
        local_index += 1
        index = field isa FaceVelocity ? global_index : local_index
        volume = _dual_face_volume(mesh.blocks, face)
        @inbounds current += field.interior[index] * volume
        total_volume += volume
    end
    correction = (target - current) / total_volume
    local_index = 0
    for (global_index, face) in pairs(mesh.faces)
        face.axis == axis || continue
        local_index += 1
        index = field isa FaceVelocity ? global_index : local_index
        @inbounds field.interior[index] += correction
    end
    return field
end

function _transfer_face_field!(
        buffer::FaceAMRFieldBuffer,
        workspace::AMRWorkspace
    )
    field = buffer.field
    mesh = workspace.mesh
    old_momenta = map(
        axis -> _old_face_momentum(workspace, buffer, axis),
        _field_axes(field)
    )
    _resize_face_field!(field, mesh)

    resize!(workspace.interior_face_local_indices, length(mesh.faces))
    local_indices = (0, 0)
    for (global_index, face) in pairs(mesh.faces)
        axis = face.axis
        local_indices = axis == 1 ?
            (local_indices[1] + 1, local_indices[2]) :
            (local_indices[1], local_indices[2] + 1)
        workspace.interior_face_local_indices[global_index] =
            local_indices[axis]
    end
    _parallel_for(length(mesh.faces); grain = 512) do global_index
        face = mesh.faces[global_index]
        face.axis in _field_axes(field) || return nothing
        value = _face_transfer_value(workspace, buffer, face)
        _set_transferred_interior!(
            field, global_index,
            workspace.interior_face_local_indices[global_index], value
        )
    end
    resize!(workspace.boundary_face_local_indices, length(mesh.boundary_faces))
    local_indices = (0, 0)
    for (global_index, face) in pairs(mesh.boundary_faces)
        axis = face.axis
        local_indices = axis == 1 ?
            (local_indices[1] + 1, local_indices[2]) :
            (local_indices[1], local_indices[2] + 1)
        workspace.boundary_face_local_indices[global_index] =
            local_indices[axis]
    end
    _parallel_for(length(mesh.boundary_faces); grain = 512) do global_index
        face = mesh.boundary_faces[global_index]
        face.axis in _field_axes(field) || return nothing
        value = _face_transfer_value(workspace, buffer, face)
        _set_transferred_boundary!(
            field, global_index,
            workspace.boundary_face_local_indices[global_index], value
        )
    end
    field isa FaceVelocity &&
        _preserve_refined_divergence!(field, buffer, workspace)
    for (index, axis) in pairs(_field_axes(field))
        # A constant correction is divergence-free only when the component's
        # normal direction is periodic. At a physical wall its missing
        # boundary half-control-volumes make the interior-only staggered
        # integral resolution dependent; forcing that integral would spoil
        # affine exactness and the boundary-cell divergence.
        buffer.conserve_integral && mesh.periodic[axis] && _conserve_face_momentum!(
            field, axis, old_momenta[index], mesh
        )
    end
    return field
end


function _integrated_cell_fluxes!(
        fluxes, blocks, faces, boundary_faces,
        interior_values, boundary_values,
        block_size
    )
    nx, ny = block_size
    resize!(fluxes, length(blocks) * nx * ny)
    fill!(fluxes, zero(eltype(fluxes)))
    for (index, face) in pairs(faces)
        @inbounds flux = interior_values[index] * face.area
        minus = _cell_index(face.minus, nx, ny)
        plus = _cell_index(face.plus, nx, ny)
        @inbounds begin
            fluxes[minus] += flux
            fluxes[plus] -= flux
        end
    end
    for (index, face) in pairs(boundary_faces)
        flux = face.side * boundary_values[index] * face.area
        cell = _cell_index(face.cell, nx, ny)
        @inbounds fluxes[cell] += flux
    end
    return fluxes
end

function _build_new_face_indices!(workspace::AMRWorkspace)
    empty!(workspace.new_face_indices)
    for (index, face) in pairs(workspace.mesh.faces)
        normal, lo, hi = _face_lattice_geometry(workspace.mesh, face)
        workspace.new_face_indices[(face.axis, normal, lo, hi)] = index
    end
    return workspace.new_face_indices
end

function _current_cellref(workspace::AMRWorkspace, x, y)
    mesh = workspace.mesh
    lattice = _finest_lattice_size(mesh)
    point = SVector(
        mesh.origin[1] + x * mesh.extent[1] / lattice[1],
        mesh.origin[2] + y * mesh.extent[2] / lattice[2]
    )
    node = _findleaf(mesh.root, point)
    block_index = workspace.new_indices[node.data.key]
    block = mesh.blocks[block_index]
    i = clamp(
        floor(Int, (point[1] - block.origin[1]) / block.spacing[1]) + 1,
        1, mesh.block_size[1]
    )
    j = clamp(
        floor(Int, (point[2] - block.origin[2]) / block.spacing[2]) + 1,
        1, mesh.block_size[2]
    )
    return CellRef(block_index, i, j)
end

function _correct_internal_face!(
        velocity::FaceVelocity, workspace::AMRWorkspace,
        key, integrated_correction
    )
    index = workspace.new_face_indices[key]
    face = workspace.mesh.faces[index]
    @inbounds velocity.interior[index] += integrated_correction / face.area
    return nothing
end

function _preserve_refined_divergence!(
        velocity::FaceVelocity,
        buffer::FaceAMRFieldBuffer,
        workspace::AMRWorkspace
    )
    mesh = workspace.mesh
    nx, ny = mesh.block_size
    empty!(workspace.old_indices)
    for (index, block) in pairs(workspace.old_blocks)
        workspace.old_indices[block.key] = index
    end
    empty!(workspace.new_indices)
    for (index, block) in pairs(mesh.blocks)
        workspace.new_indices[block.key] = index
    end
    _build_new_face_indices!(workspace)
    _integrated_cell_fluxes!(
        workspace.old_cell_flux, workspace.old_blocks,
        workspace.old_faces, workspace.old_boundary_faces,
        buffer.old_interior, buffer.old_boundary, mesh.block_size
    )
    _integrated_cell_fluxes!(
        workspace.new_cell_flux, mesh.blocks, mesh.faces, mesh.boundary_faces,
        velocity.interior, velocity.boundary, mesh.block_size
    )

    for (old_block_index, block) in pairs(workspace.old_blocks)
        haskey(workspace.new_indices, block.key) && continue
        children = _child_keys(block.key)
        all(haskey(workspace.new_indices, child) for child in children) || continue
        scale = _block_scale(mesh, block.key.level)
        fine_scale = scale ÷ 2
        block_x = block.key.i * nx * scale
        block_y = block.key.j * ny * scale
        for j in 1:ny, i in 1:nx
            xlo = block_x + (i - 1) * scale
            ylo = block_y + (j - 1) * scale
            xmid, ymid = xlo + fine_scale, ylo + fine_scale
            xhi, yhi = xlo + scale, ylo + scale
            fine_cells = (
                _current_cellref(
                    workspace, xlo + fine_scale / 2,
                    ylo + fine_scale / 2
                ),
                _current_cellref(
                    workspace, xmid + fine_scale / 2,
                    ylo + fine_scale / 2
                ),
                _current_cellref(
                    workspace, xlo + fine_scale / 2,
                    ymid + fine_scale / 2
                ),
                _current_cellref(
                    workspace, xmid + fine_scale / 2,
                    ymid + fine_scale / 2
                ),
            )
            old_ref = CellRef(old_block_index, i, j)
            target = workspace.old_cell_flux[
                _cell_index(old_ref, nx, ny),
            ] / 4
            residuals = ntuple(
                index -> target - workspace.new_cell_flux[
                    _cell_index(fine_cells[index], nx, ny),
                ], 4
            )

            # Incidence-minimum-norm correction for the two vertical and two
            # horizontal internal fine faces. The fourth equation follows
            # from conservation of the transferred parent boundary flux.
            vertical_lower =
                (2residuals[1] + residuals[3] - residuals[2]) / 4
            vertical_upper = residuals[3] + residuals[1] - vertical_lower
            horizontal_left = residuals[1] - vertical_lower
            horizontal_right = residuals[2] + vertical_lower
            _correct_internal_face!(
                velocity, workspace, (1, xmid, ylo, ymid), vertical_lower
            )
            _correct_internal_face!(
                velocity, workspace, (1, xmid, ymid, yhi), vertical_upper
            )
            _correct_internal_face!(
                velocity, workspace, (2, ymid, xlo, xmid), horizontal_left
            )
            _correct_internal_face!(
                velocity, workspace, (2, ymid, xmid, xhi), horizontal_right
            )
        end
    end
    return velocity
end
