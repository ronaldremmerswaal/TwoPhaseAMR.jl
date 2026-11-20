# LS-STAG closure of the single-phase viscous operator next to embedded solids
# (Cheny and Botella 2010, `LSStagClosure`). It replaces, for one velocity
# component, every contact of the faces of cut cells, of their neighbours on
# the same grid line and of fine faces whose line ends at a coarse cell with
# such a face; every other contact keeps the default closure. On uncut cells
# of one level both closures give the same two-point contacts, so the seam is
# consistent.
#
#   value    at the midpoint of the open part of the face
#   normal   |c| (G_c u)^2 for every cell with a replaced face,
#            G_c u = (1/|c|) sum_f o_{c,f} a_f |f| u_f (wall value zero)
#   shear    along the grid line of the face: links between open midpoints,
#            a linear-exact three-point derivative where a fine line ends at
#            a coarse cell, and the one-sided quotient to the wall or the
#            domain boundary with what the links leave of each half control
#            volume's projected extent, so every half control volume closes
#
# A face owns half of the open length of each edge of its strip in each
# adjacent cell (measure bisection), as the open staggered volumes do.

struct _LSStagFace{T}
    # Lattice coordinates: the grid line (along the component axis), the
    # tangential bounds, and the finer adjacent cell's width along the line
    # normal.
    line::Int
    lo::Int
    hi::Int
    width::Int
    lower::T
    upper::T
    open_lower::T
    open_upper::T
    open_at_lo::Bool
    open_at_hi::Bool
    active::Bool
end

@inline _ls_stag_midpoint(face::_LSStagFace) = (face.open_lower + face.open_upper) / 2
@inline _ls_stag_half_open(face::_LSStagFace) = (face.open_upper - face.open_lower) / 2
@inline _ls_stag_open_at(face::_LSStagFace, which) =
    which == 1 ? face.open_at_lo : face.open_at_hi

# The open part of a partially open face lies at the end that the solid planes
# of its cut cells leave open: the first cut cell whose own trace of the face is
# partial decides, otherwise any cut cell with a tangential normal component,
# otherwise the upper end. The cell-centred PLIC normal points into the fluid.
function _ls_stag_open_part(geometry::GeometryField{T}, face, aperture) where {T}
    solid = geometry.solid
    tangent = 3 - face.axis
    lower = T(face.center[tangent]) - T(face.area) / 2
    upper = T(face.center[tangent]) + T(face.area) / 2
    one(T) - aperture <= 256eps(T) && return lower, upper, lower, upper, true, true
    open_length = aperture * T(face.area)
    decision = 0
    fallback = 0
    for (ref, side) in ((face.minus, 1), (face.plus, -1))
        fraction = _get(solid.fraction, ref)
        zero(T) < fraction < one(T) || continue
        tangential = tangent == 1 ? _get(solid.normal_x, ref) : _get(solid.normal_y, ref)
        abs(tangential) > 64eps(T) || continue
        trace = _cell_face_open_fraction(solid, ref, face, side)
        if zero(T) < trace < one(T)
            decision = tangential > zero(T) ? 1 : -1
            break
        end
        iszero(fallback) && (fallback = tangential > zero(T) ? 1 : -1)
    end
    iszero(decision) && (decision = iszero(fallback) ? 1 : fallback)
    return decision > 0 ?
        (lower, upper, upper - open_length, upper, false, true) :
        (lower, upper, lower, lower + open_length, true, false)
end

# Length, along the line normal, of the polygon edges on `x[tangent] == value`.
function _ls_stag_edge_length(polygon, tangent, value::T) where {T}
    length(polygon) >= 3 || return zero(T)
    axis = 3 - tangent
    tolerance = 256eps(T)
    total = zero(T)
    previous = last(polygon)
    for point in polygon
        if abs(previous[tangent] - value) <= tolerance &&
                abs(point[tangent] - value) <= tolerance
            total += abs(point[axis] - previous[axis])
        end
        previous = point
    end
    return total
end

# Owned open lengths of the lower and upper edge of the strip of `face` in
# cell `ref`: half of the open length of each edge.
function _ls_stag_strip_edges(
        geometry::GeometryField{T}, ::Val{Axis}, ref, face,
    ) where {T, Axis}
    mesh = geometry.mesh
    fraction = _get(geometry.solid.fraction, ref)
    fraction >= one(T) && return zero(T), zero(T)
    half_width = T(mesh.blocks[ref.block].spacing[Axis]) / 2
    fraction <= zero(T) && return half_width, half_width
    lo, hi = _face_local_tangent_interval(mesh, ref, face)
    lo, hi = T(lo), T(hi)
    half = T(0.5)
    polygon = _open_cell_polygon(geometry, ref)
    strip = Axis == 1 ? _clip_to_rectangle(polygon, -half, half, lo, hi) :
        _clip_to_rectangle(polygon, lo, hi, -half, half)
    tangent = 3 - Axis
    return half_width * _ls_stag_edge_length(strip, tangent, lo),
        half_width * _ls_stag_edge_length(strip, tangent, hi)
end

# Distance from `point` along `direction` times the tangent axis to the PLIC
# plane of cut cell `ref`; zero if the cell is not cut or the plane is not
# ahead.
function _ls_stag_plane_distance(
        geometry::GeometryField{T}, ref, point, tangent, direction,
    ) where {T}
    solid = geometry.solid
    fraction = _get(solid.fraction, ref)
    zero(T) < fraction < one(T) || return zero(T)
    mesh = geometry.mesh
    spacing = mesh.blocks[ref.block].spacing
    center = cellcenter(mesh, ref.block, ref.i, ref.j)
    normal_x = _get(solid.normal_x, ref)
    normal_y = _get(solid.normal_y, ref)
    dx = T(_wrapped_delta(point[1] - center[1], mesh.extent[1], mesh.periodic[1])) /
        T(spacing[1])
    dy = T(_wrapped_delta(point[2] - center[2], mesh.extent[2], mesh.periodic[2])) /
        T(spacing[2])
    level = normal_x * dx + normal_y * dy - _get(solid.intercept, ref)
    slope = direction * (tangent == 1 ? normal_x / T(spacing[1]) : normal_y / T(spacing[2]))
    abs(slope) > 64eps(T) * hypot(normal_x / T(spacing[1]), normal_y / T(spacing[2])) ||
        return zero(T)
    distance = -level / slope
    return distance > zero(T) ? distance : zero(T)
end

# Weights on (node, first, second) of the derivative along the tangent that is
# exact for affine functions; the points are (normal, tangent) offsets from the
# node. `nothing` for (nearly) collinear points.
@inline function _ls_stag_affine_derivative(first::NTuple{2, T}, second::NTuple{2, T}) where {T}
    determinant = first[1] * second[2] - second[1] * first[2]
    scale = max(abs(first[1]), abs(second[1])) * max(abs(first[2]), abs(second[2]))
    abs(determinant) > sqrt(eps(T)) * scale || return nothing
    first_weight = -second[1] / determinant
    second_weight = first[1] / determinant
    return (-(first_weight + second_weight), first_weight, second_weight)
end

# A point on the solid next to a closed face: the middle of the PLIC segment
# of an adjacent cut cell, or the face centre.
function _ls_stag_wall_point(geometry::GeometryField{T}, face) where {T}
    mesh = geometry.mesh
    for ref in (face.minus, face.plus)
        segment = _local_plic_segment(geometry.solid, ref)
        isnothing(segment) && continue
        center = cellcenter(mesh, ref.block, ref.i, ref.j)
        spacing = mesh.blocks[ref.block].spacing
        first = _local_to_physical(T, center, spacing, segment[1])
        last = _local_to_physical(T, center, spacing, segment[2])
        return ((first[1] + last[1]) / 2, (first[2] + last[2]) / 2)
    end
    return (T(face.center[1]), T(face.center[2]))
end

function _ls_stag_push_square!(rows, columns, values, indices, weights, scale)
    for a in eachindex(indices), b in eachindex(indices)
        (iszero(indices[a]) || iszero(indices[b])) && continue
        push!(rows, indices[a])
        push!(columns, indices[b])
        push!(values, scale * (weights[a] * weights[b]))
    end
    return nothing
end

"""
    _ls_stag_closure(geometry, Val(Axis), boundary, incidences, global_to_local,
                     local_to_global, connections, wall_connections)

LS-STAG closure of one component (see `LSStagClosure`). Returns the retained
default contacts and wall connections (with the LS-STAG wall quotients, whose
conductances enter the stiffness), the LS-STAG stiffness, the boundary sources
of the Green rows and hanging derivatives (source only: their energy is in the
LS-STAG stiffness) and the wall reaction coefficient of every face.
"""
function _ls_stag_closure(
        geometry::GeometryField{T}, ::Val{Axis}, boundary, incidences,
        global_to_local, local_to_global, connections, wall_connections,
    ) where {T, Axis}
    mesh = geometry.mesh
    solid = geometry.solid
    nx, ny = mesh.block_size
    tangent = 3 - Axis
    lattice = _finest_lattice_size(mesh)
    period = lattice[tangent]
    periodic = mesh.periodic[tangent]
    axis_period = lattice[Axis]
    axis_periodic = mesh.periodic[Axis]
    wrap(coordinate) = periodic ? mod(coordinate, period) : coordinate
    tangent_delta(value) = T(_wrapped_delta(value, mesh.extent[tangent], periodic))
    count = length(local_to_global)
    open_volumes = geometry.dual_volumes[Axis]

    # Faces, their open parts and their neighbours along the grid lines.
    faces = Vector{_LSStagFace{T}}(undef, count)
    starts = Dict{Tuple{Int, Int}, Int}()
    ends = Dict{Tuple{Int, Int}, Int}()
    sizehint!(starts, count)
    sizehint!(ends, count)
    for (local_index, global_index) in pairs(local_to_global)
        face = mesh.faces[global_index]
        line, lo, hi = _face_lattice_geometry(mesh, face, lattice)
        _, minus_scale = _cell_lattice_bounds(mesh, mesh.blocks, face.minus, lattice)
        _, plus_scale = _cell_lattice_bounds(mesh, mesh.blocks, face.plus, lattice)
        aperture = geometry.aperture[global_index]
        active = aperture > T(1.0e-10) && open_volumes[local_index] > zero(T)
        lower, upper, open_lower, open_upper, at_lo, at_hi = active ?
            _ls_stag_open_part(geometry, face, aperture) :
            (
                T(face.center[tangent]) - T(face.area) / 2,
                T(face.center[tangent]) + T(face.area) / 2,
                zero(T), zero(T), false, false,
            )
        faces[local_index] = _LSStagFace{T}(
            line, lo, hi, min(minus_scale[Axis], plus_scale[Axis]),
            lower, upper, open_lower, open_upper, at_lo, at_hi, active,
        )
        starts[(line, lo)] = local_index
        ends[(line, wrap(hi))] = local_index
    end
    below(face) = get(ends, (face.line, face.lo), 0)
    above(face) = get(starts, (face.line, wrap(face.hi)), 0)
    beyond(face, which) = which == 1 ? below(face) : above(face)
    on_boundary(face, which) = !periodic && (which == 1 ? face.lo == 0 : face.hi == period)
    # The faces of the coarse cell beyond a fine line's end, on the lines one
    # fine width to either side; `in_domain` is false at a domain boundary.
    function partner(face, which, offset)
        line = face.line + offset
        if axis_periodic
            line = mod(line, axis_period)
        elseif !(0 < line < axis_period)
            return 0, false
        end
        index = which == 1 ? get(ends, (line, face.lo), 0) :
            get(starts, (line, wrap(face.hi)), 0)
        return index, true
    end
    hanging(face, which) = iszero(beyond(face, which)) && !on_boundary(face, which)

    # Replaced faces: the faces of cut cells, their neighbours on the same
    # grid line, and fine faces whose line ends at a coarse cell with a
    # replaced face.
    replaced = falses(count)
    cell_refs = [
        CellRef(
            (index - 1) ÷ (nx * ny) + 1, (index - 1) % (nx * ny) % nx + 1,
            (index - 1) % (nx * ny) ÷ nx + 1
        )
            for index in eachindex(incidences)
    ]
    for (cell_index, cell_faces) in pairs(incidences)
        zero(T) < _get(solid.fraction, cell_refs[cell_index]) < one(T) || continue
        for incidence in cell_faces
            replaced[incidence.local_index] = true
        end
    end
    for local_index in findall(replaced)
        face = faces[local_index]
        for which in 1:2
            neighbour = beyond(face, which)
            iszero(neighbour) || (replaced[neighbour] = true)
        end
    end
    changed = true
    while changed
        changed = false
        for local_index in 1:count
            replaced[local_index] && continue
            face = faces[local_index]
            for which in 1:2
                hanging(face, which) || continue
                for offset in (-face.width, face.width)
                    index, _ = partner(face, which, offset)
                    if !iszero(index) && replaced[index]
                        replaced[local_index] = true
                        changed = true
                    end
                end
            end
        end
    end

    # Green cells: every cell with a replaced face.
    green = falses(length(incidences))
    for (cell_index, cell_faces) in pairs(incidences)
        green[cell_index] = any(incidence -> replaced[incidence.local_index], cell_faces)
    end
    green_pairs = Set{Tuple{Int, Int}}()
    for (cell_index, cell_faces) in pairs(incidences)
        green[cell_index] || continue
        for first in cell_faces, second in cell_faces
            first.side < 0 < second.side || continue
            push!(green_pairs, minmax(first.local_index, second.local_index))
        end
    end
    boundary_faces = (zeros(Int, length(incidences)), zeros(Int, length(incidences)))
    green_boundary_rows = Set{Tuple{Int, Int8}}()
    for (index, boundary_face) in pairs(mesh.boundary_faces)
        boundary_face.axis == Axis || continue
        cell_index = _cell_index(boundary_face.cell, nx, ny)
        green[cell_index] || continue
        boundary_faces[boundary_face.side < 0 ? 1 : 2][cell_index] = index
        for incidence in incidences[cell_index]
            incidence.side == -boundary_face.side &&
                push!(green_boundary_rows, (incidence.local_index, Int8(boundary_face.side)))
        end
    end

    retained = DualDiffusionConnection{T}[]
    for connection in connections
        if connection.axis == tangent
            (replaced[connection.minus] || replaced[connection.plus]) && continue
        else
            (connection.minus, connection.plus) in green_pairs && continue
        end
        push!(retained, connection)
    end
    retained_walls = WallDiffusionConnection{T}[]
    for wall in wall_connections
        index = wall.degree_of_freedom
        if iszero(wall.boundary_axis)
            # Fully covered faces keep their algebraic constraint.
            replaced[index] && wall.tangential && continue
        elseif wall.boundary_axis == tangent
            replaced[index] && continue
        else
            (index, wall.boundary_side) in green_boundary_rows && continue
        end
        push!(retained_walls, wall)
    end

    rows, columns, values = Int[], Int[], T[]
    reactions = zeros(T, count)
    sources = WallDiffusionConnection{T}[]
    indices = Int[]
    weights = T[]

    # Normal part: the Green gradient of every green cell.
    for (cell_index, cell_faces) in pairs(incidences)
        green[cell_index] || continue
        ref = cell_refs[cell_index]
        spacing = mesh.blocks[ref.block].spacing
        volume = _polygon_area(_open_cell_polygon(geometry, ref)) *
            T(spacing[1]) * T(spacing[2])
        volume > zero(T) || continue
        # A side without a prescribed normal velocity (outflow) leaves the
        # cell without a normal stress, as the default closure does.
        natural = false
        boundary_weight = zero(T)
        for position in 1:2
            index = boundary_faces[position][cell_index]
            iszero(index) && continue
            boundary_face = mesh.boundary_faces[index]
            probe = WallDiffusionConnection(
                1, one(T), T.(boundary_face.center), false,
                Int8(Axis), Int8(boundary_face.side),
            )
            _include_wall_connection(boundary, probe) || (natural = true)
            boundary_weight += T(boundary_face.side) *
                geometry.boundary_aperture[index] * T(boundary_face.area)
        end
        natural && continue
        empty!(indices)
        empty!(weights)
        total = zero(T)
        for incidence in cell_faces
            faces[incidence.local_index].active || continue
            weight = T(incidence.side) * geometry.aperture[incidence.global_index] *
                T(mesh.faces[incidence.global_index].area)
            push!(indices, incidence.local_index)
            push!(weights, weight)
            total += weight
        end
        isempty(indices) && continue
        _ls_stag_push_square!(rows, columns, values, indices, weights, inv(volume))
        # The wall carries what the faces and the boundary leave of the
        # closure sum, so the gradient of a constant is zero.
        for position in eachindex(indices)
            reactions[indices[position]] +=
                weights[position] * (total + boundary_weight) / volume
        end
        for position in 1:2
            index = boundary_faces[position][cell_index]
            iszero(index) && continue
            boundary_face = mesh.boundary_faces[index]
            face_weight = T(boundary_face.side) *
                geometry.boundary_aperture[index] * T(boundary_face.area)
            for entry in eachindex(indices)
                push!(
                    sources, WallDiffusionConnection(
                        indices[entry], -weights[entry] * face_weight / volume,
                        T.(boundary_face.center), false,
                        Int8(Axis), Int8(boundary_face.side),
                    ),
                )
            end
        end
    end

    # Owned open lengths of the strip edges, (lower, upper) per side.
    edges = zeros(T, 4, count)
    edges_ready = falses(count)
    function strip_edges(local_index)
        if !edges_ready[local_index]
            face = mesh.faces[local_to_global[local_index]]
            lower, upper = _ls_stag_strip_edges(geometry, Val(Axis), face.minus, face)
            edges[1, local_index], edges[2, local_index] = lower, upper
            lower, upper = _ls_stag_strip_edges(geometry, Val(Axis), face.plus, face)
            edges[3, local_index], edges[4, local_index] = lower, upper
            edges_ready[local_index] = true
        end
        return local_index
    end
    edge(local_index, side, which) = edges[2 * (side - 1) + which, strip_edges(local_index)]
    node(local_index) = (
        T(mesh.faces[local_to_global[local_index]].center[Axis]),
        _ls_stag_midpoint(faces[local_index]),
    )
    point(line, tangential) = Axis == 1 ? (line, tangential) : (tangential, line)

    # Shear part, pass 1: links and hanging derivatives. `covered[w, f, s]`
    # is the part of the projected extent at end `w` of the half control
    # volume of `f` on side `s` that they carry; `status` is 0 for an end
    # closed against a wall or the boundary, 1 for a link, 2 for a hanging end.
    covered = zeros(T, 2, 2, count)
    status = zeros(Int8, 2, count)
    linked = zeros(Int, 2, count)
    for local_index in 1:count
        replaced[local_index] || continue
        face = faces[local_index]
        face.active || continue
        mesh_face = mesh.faces[local_to_global[local_index]]
        line, midpoint = node(local_index)
        for which in 1:2
            _ls_stag_open_at(face, which) || continue
            neighbour = beyond(face, which)
            if !iszero(neighbour)
                other = faces[neighbour]
                (other.active && _ls_stag_open_at(other, 3 - which)) || continue
                status[which, local_index] = 1
                linked[which, local_index] = neighbour
                # Each link once: from below, or from the replaced face.
                (which == 2 || !replaced[neighbour]) || continue
                area = zero(T)
                for side in 1:2
                    shared = min(edge(local_index, side, which), edge(neighbour, side, 3 - which))
                    covered[which, side, local_index] += shared
                    covered[3 - which, side, neighbour] += shared
                    area += shared
                end
                distance = abs(tangent_delta(_ls_stag_midpoint(other) - midpoint))
                area > zero(T) && _ls_stag_push_square!(
                    rows, columns, values, (local_index, neighbour),
                    (-inv(distance), inv(distance)), area * distance,
                )
                continue
            end
            on_boundary(face, which) && continue
            status[which, local_index] = 2
            area = zero(T)
            for side in 1:2
                share = edge(local_index, side, which)
                covered[which, side, local_index] += share
                area += share
            end
            area > zero(T) || continue
            direction = which == 1 ? -1 : 1
            end_coordinate = which == 1 ? face.lower : face.upper
            fine_width = T(
                min(
                    mesh.blocks[mesh_face.minus.block].spacing[Axis],
                    mesh.blocks[mesh_face.plus.block].spacing[Axis],
                ),
            )
            partners = (0, 0)
            kinds = (0, 0)              # 1 value, 2 wall, 3 boundary
            offsets = ((zero(T), zero(T)), (zero(T), zero(T)))
            boundary_points = ((zero(T), zero(T)), (zero(T), zero(T)))
            for (position, offset) in ((1, -face.width), (2, face.width))
                index, in_domain = partner(face, which, offset)
                normal_offset = position == 1 ? -fine_width : fine_width
                kind, partner_offset, wall_point = 0, (zero(T), zero(T)), (zero(T), zero(T))
                if !iszero(index) && faces[index].active
                    kind = 1
                    partner_offset = (normal_offset, tangent_delta(_ls_stag_midpoint(faces[index]) - midpoint))
                elseif !iszero(index)
                    # A closed face of the coarse cell: the wall value at a
                    # point of the solid next to it.
                    kind = 2
                    wall = _ls_stag_wall_point(geometry, mesh.faces[local_to_global[index]])
                    partner_offset = (
                        T(_wrapped_delta(wall[Axis] - line, mesh.extent[Axis], axis_periodic)),
                        tangent_delta(wall[tangent] - midpoint),
                    )
                elseif !in_domain
                    # The coarse cell's boundary face: its normal velocity is
                    # prescribed unless the side is an outflow.
                    wall_tangential = end_coordinate + direction * T(mesh_face.area)
                    wall_point = point(line + normal_offset, wall_tangential)
                    probe = WallDiffusionConnection(
                        local_index, one(T), wall_point, false, Int8(Axis), Int8(position == 1 ? -1 : 1),
                    )
                    if _include_wall_connection(boundary, probe)
                        kind = 3
                        partner_offset = (normal_offset, tangent_delta(wall_tangential - midpoint))
                    end
                end
                partners = Base.setindex(partners, kind == 1 ? index : 0, position)
                kinds = Base.setindex(kinds, kind, position)
                offsets = Base.setindex(offsets, partner_offset, position)
                boundary_points = Base.setindex(boundary_points, wall_point, position)
            end
            derivative = all(!iszero, kinds) ?
                _ls_stag_affine_derivative(offsets[1], offsets[2]) : nothing
            if isnothing(derivative)
                # No usable pair: difference with an open partner, otherwise
                # the end is a wall.
                position = kinds[1] == 1 ? 1 : (kinds[2] == 1 ? 2 : 0)
                if iszero(position)
                    end_point = point(line, end_coordinate)
                    push!(
                        retained_walls, WallDiffusionConnection(
                            local_index, area / _ls_stag_half_open(face), end_point, true,
                        ),
                    )
                    continue
                end
                distance = abs(offsets[position][2])
                _ls_stag_push_square!(
                    rows, columns, values, (local_index, partners[position]),
                    (-inv(distance), inv(distance)), area * distance,
                )
                partners[position] != 0 && replaced[partners[position]] &&
                    (covered[3 - which, 3 - position, partners[position]] += area)
                continue
            end
            scale = area / abs(derivative[1])
            stencil = (local_index, partners[1], partners[2])
            _ls_stag_push_square!(rows, columns, values, stencil, derivative, scale)
            for position in 1:2
                kind = kinds[position]
                weight = derivative[position + 1]
                if kind == 1
                    replaced[partners[position]] &&
                        (covered[3 - which, 3 - position, partners[position]] += area * abs(weight / derivative[1]))
                    continue
                end
                for entry in 1:3
                    index = stencil[entry]
                    iszero(index) && continue
                    coefficient = scale * (derivative[entry] * weight)
                    if kind == 2
                        reactions[index] -= coefficient
                    else
                        push!(
                            sources, WallDiffusionConnection(
                                index, -coefficient, boundary_points[position], false,
                                Int8(Axis), Int8(position == 1 ? -1 : 1),
                            ),
                        )
                    end
                end
            end
        end
    end

    # Pass 2: the wall (or the domain boundary) carries what the links leave
    # of the projected extent of every half control volume, the longer of its
    # two owned strip edges, at the distance along the grid line to the wall.
    for local_index in 1:count
        replaced[local_index] || continue
        face = faces[local_index]
        global_index = local_to_global[local_index]
        if !face.active
            # A face without an open part next to a cut cell: its value is
            # the wall value. Keep the row finite at the scale of the uncut
            # Laplacian of its open volume.
            volume = open_volumes[local_index]
            volume > zero(T) || continue
            mesh_face = mesh.faces[global_index]
            spacing_minus = mesh.blocks[mesh_face.minus.block].spacing
            spacing_plus = mesh.blocks[mesh_face.plus.block].spacing
            h_axis = T(min(spacing_minus[Axis], spacing_plus[Axis]))
            h_tangent = T(min(spacing_minus[tangent], spacing_plus[tangent]))
            push!(
                retained_walls, WallDiffusionConnection(
                    local_index, volume * (2 / h_axis^2 + 2 / h_tangent^2),
                    T.(mesh_face.center), false,
                ),
            )
            continue
        end
        mesh_face = mesh.faces[global_index]
        line, midpoint = node(local_index)
        half_open = _ls_stag_half_open(face)
        for which in 1:2
            direction = which == 1 ? -1 : 1
            end_open = _ls_stag_open_at(face, which)
            at_boundary = iszero(status[which, local_index]) && end_open &&
                on_boundary(face, which)
            for (side, ref) in ((1, mesh_face.minus), (2, mesh_face.plus))
                deficit = max(edge(local_index, side, 1), edge(local_index, side, 2)) -
                    covered[which, side, local_index]
                deficit > T(1.0e-12) * T(mesh_face.area) || continue
                if iszero(status[which, local_index])
                    distance = half_open
                else
                    # An open end: the wall lies beyond it, on the plane of
                    # this cell or of the cell across the end.
                    distance = _ls_stag_plane_distance(
                        geometry, ref, point(line, midpoint), tangent, direction,
                    )
                    neighbour = linked[which, local_index]
                    if iszero(distance) && !iszero(neighbour)
                        neighbour_face = mesh.faces[local_to_global[neighbour]]
                        distance = _ls_stag_plane_distance(
                            geometry, side == 1 ? neighbour_face.minus : neighbour_face.plus,
                            point(line, midpoint), tangent, direction,
                        )
                    end
                    iszero(distance) && continue
                    distance = max(distance, half_open)
                end
                if at_boundary
                    end_coordinate = which == 1 ? face.lower : face.upper
                    push!(
                        retained_walls, WallDiffusionConnection(
                            local_index, deficit / distance, point(line, end_coordinate), true,
                            Int8(tangent), Int8(direction),
                        ),
                    )
                else
                    push!(
                        retained_walls, WallDiffusionConnection(
                            local_index, deficit / distance,
                            point(line, midpoint + direction * distance), true,
                        ),
                    )
                end
            end
        end
    end

    stiffness = sparse(rows, columns, values, count, count)
    stiffness = sparse((stiffness + transpose(stiffness)) / 2)
    dropzeros!(stiffness)
    return (;
        connections = retained, wall_connections = retained_walls,
        stiffness, sources, reactions,
    )
end
