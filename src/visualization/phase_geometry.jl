_is_discontinuous_source(source) = source isa _DiscontinuousVisualizationSource
_source_interface(source::PhaseVisualizationSource) = source.quantity.interface
_source_interface(
    source::Union{
        PhaseVelocityComponentVisualizationSource,
        PhaseVelocityDerivedVisualizationSource,
    }
) = source.interface

function _local_phase_polygons(interface, ref, replacements)
    T = eltype(interface)
    open_polygon = isnothing(interface.geometry) ? NTuple{2, T}[
            (-T(0.5), -T(0.5)), (T(0.5), -T(0.5)),
            (T(0.5), T(0.5)), (-T(0.5), T(0.5)),
        ] : _open_cell_polygon(interface.geometry, ref)
    isempty(open_polygon) && return (open_polygon, copy(open_polygon))

    fraction_field = _visualization_replacement(
        replacements, interface.conditional_fraction,
    )
    fraction = clamp(_get(fraction_field, ref), zero(T), one(T))
    floor = interface.fraction_floor
    fraction <= floor && return (NTuple{2, T}[], open_polygon)
    fraction >= one(T) - floor &&
        return (open_polygon, NTuple{2, T}[])

    normal_x = _visualization_replacement(replacements, interface.normal_x)
    normal_y = _visualization_replacement(replacements, interface.normal_y)
    intercept = _visualization_replacement(replacements, interface.intercept)
    nx = _get(normal_x, ref)
    ny = _get(normal_y, ref)
    plane = _get(intercept, ref)
    return (
        _clip_polygon(open_polygon, nx, ny, plane; keep_less = true),
        _clip_polygon(open_polygon, nx, ny, plane; keep_less = false),
    )
end

function _physical_polygon(
        interface, ref::CellRef, polygon, target_center = nothing,
    )
    block = interface.mesh.blocks[ref.block]
    center = cellcenter(interface.mesh, ref.block, ref.i, ref.j)
    shifted_center = if isnothing(target_center)
        center
    else
        ntuple(2) do axis
            shift = if interface.mesh.periodic[axis]
                round(
                    (target_center[axis] - center[axis]) /
                        interface.mesh.extent[axis],
                ) * interface.mesh.extent[axis]
            else
                zero(eltype(interface))
            end
            center[axis] + shift
        end
    end
    return [
        (
            shifted_center[1] + point[1] * block.spacing[1],
            shifted_center[2] + point[2] * block.spacing[2],
        ) for point in polygon
    ]
end

function _append_phase_polygon!(
        points, triangles, owners, value_indices, phases,
        polygon, owner, value_index, liquid,
    )
    length(polygon) >= 3 || return
    first_vertex = length(points) + 1
    append!(points, polygon)
    append!(owners, fill(owner, length(polygon)))
    append!(value_indices, fill(value_index, length(polygon)))
    append!(phases, fill(liquid, length(polygon)))
    for vertex in 2:(length(polygon) - 1)
        push!(
            triangles,
            (first_vertex, first_vertex + vertex - 1, first_vertex + vertex),
        )
    end
    return
end

function _cell_phase_geometry(interface, topology, replacements)
    T = eltype(interface)
    points = NTuple{2, T}[]
    triangles = NTuple{3, Int}[]
    owners = CellRef[]
    value_indices = Int[]
    phases = BitVector()
    for (value_index, ref) in pairs(topology.cell_geometry.cells)
        liquid, gas = _local_phase_polygons(interface, ref, replacements)
        _append_phase_polygon!(
            points, triangles, owners, value_indices, phases,
            _physical_polygon(interface, ref, liquid), ref, value_index, true,
        )
        _append_phase_polygon!(
            points, triangles, owners, value_indices, phases,
            _physical_polygon(interface, ref, gas), ref, value_index, false,
        )
    end
    return PLICVisualizationGeometry(
        points, triangles, owners, value_indices, phases,
    )
end

function _clip_to_rectangle(polygon, xlower, xupper, ylower, yupper)
    clipped = _clip_polygon(
        polygon, -one(xlower), zero(xlower), -xlower; keep_less = true,
    )
    clipped = _clip_polygon(
        clipped, one(xlower), zero(xlower), xupper; keep_less = true,
    )
    clipped = _clip_polygon(
        clipped, zero(xlower), -one(xlower), -ylower; keep_less = true,
    )
    return _clip_polygon(
        clipped, zero(xlower), one(xlower), yupper; keep_less = true,
    )
end

function _face_cells(topology, reference::FaceRef)
    face = reference.boundary ? topology.boundary_faces[reference.index] :
        topology.interior_faces[reference.index]
    return reference.boundary ? (face.cell,) : (face.minus, face.plus)
end

function _face_phase_geometry(
        interface, topology, base_geometry::AMRFaceGeometry{Axis}, replacements,
    ) where {Axis}
    T = eltype(interface)
    points = NTuple{2, T}[]
    triangles = NTuple{3, Int}[]
    owners = FaceRef[]
    value_indices = Int[]
    phases = BitVector()
    for tile in eachindex(base_geometry.faces)
        first_vertex = 4tile - 3
        rectangle = @view base_geometry.points[first_vertex:(first_vertex + 3)]
        xlower = minimum(first, rectangle)
        xupper = maximum(first, rectangle)
        ylower = minimum(last, rectangle)
        yupper = maximum(last, rectangle)
        target_center = ((xlower + xupper) / 2, (ylower + yupper) / 2)
        owner = base_geometry.faces[tile]
        value_index = base_geometry.value_indices[tile]
        for ref in _face_cells(topology, owner)
            liquid, gas = _local_phase_polygons(interface, ref, replacements)
            for (polygon, is_liquid) in ((liquid, true), (gas, false))
                physical = _physical_polygon(
                    interface, ref, polygon, target_center,
                )
                clipped = _clip_to_rectangle(
                    physical, xlower, xupper, ylower, yupper,
                )
                _append_phase_polygon!(
                    points, triangles, owners, value_indices, phases,
                    clipped, owner, value_index, is_liquid,
                )
            end
        end
    end
    return PLICVisualizationGeometry(
        points, triangles, owners, value_indices, phases,
    )
end

function _phase_visualization_geometry(
        source::_DiscontinuousVisualizationSource, topology,
        replacements = IdDict{Any, Any}(),
    )
    interface = _source_interface(source)
    placement = _source_placement(source)
    placement isa Cell &&
        return _cell_phase_geometry(interface, topology, replacements)
    axis = placement isa Face{1} ? 1 : 2
    return _face_phase_geometry(
        interface, topology, topology.face_geometries[axis], replacements,
    )
end

function _phase_visualization_vertex_values(
        values::TwoPhaseVisualizationValues,
        geometry::PLICVisualizationGeometry,
    )
    length(values.liquid) == length(values.gas) || throw(
        DimensionMismatch("liquid and gas visualization fields differ in size"),
    )
    maximum(geometry.value_indices; init = 0) <= length(values.liquid) || throw(
        DimensionMismatch("phase field size differs from PLIC geometry"),
    )
    result = Vector{
        promote_type(
            eltype(values.liquid), eltype(values.gas),
        ),
    }(undef, length(geometry.points))
    for vertex in eachindex(result)
        index = geometry.value_indices[vertex]
        @inbounds result[vertex] = geometry.liquid[vertex] ?
            values.liquid[index] : values.gas[index]
    end
    return result
end

_phase_visualization_colorrange_values(values::TwoPhaseVisualizationValues) =
    Iterators.flatten((values.liquid, values.gas))

"""
    visualization_geometry(mesh)

Materialize plotting geometry from the compact leaf-block records in `mesh`.
The ordering is block-major, then `j`, then `i`, matching
`visualization_values`.
"""
function visualization_geometry(mesh::BlockTreeMesh{T}) where {T}
    nx, ny = mesh.block_size
    count = ncells(mesh)
    points = Vector{NTuple{2, T}}(undef, 4count)
    triangles = Vector{NTuple{3, Int}}(undef, 2count)
    cells = Vector{CellRef}(undef, count)
    block_segments = Vector{NTuple{2, T}}(undef, 8nblocks(mesh))

    cell_index = 0
    for (block_index, block) in pairs(mesh.blocks), j in 1:ny, i in 1:nx
        cell_index += 1
        xlo = block.origin[1] + (i - 1) * block.spacing[1]
        xhi = xlo + block.spacing[1]
        ylo = block.origin[2] + (j - 1) * block.spacing[2]
        yhi = ylo + block.spacing[2]
        first_vertex = 4cell_index - 3
        points[first_vertex] = (xlo, ylo)
        points[first_vertex + 1] = (xhi, ylo)
        points[first_vertex + 2] = (xhi, yhi)
        points[first_vertex + 3] = (xlo, yhi)
        first_triangle = 2cell_index - 1
        triangles[first_triangle] =
            (first_vertex, first_vertex + 1, first_vertex + 2)
        triangles[first_triangle + 1] =
            (first_vertex, first_vertex + 2, first_vertex + 3)
        cells[cell_index] = CellRef(block_index, i, j)
    end

    for (block_index, block) in pairs(mesh.blocks)
        xlo, ylo = block.origin
        xhi = xlo + block.extent[1]
        yhi = ylo + block.extent[2]
        first_point = 8block_index - 7
        block_segments[first_point] = (xlo, ylo)
        block_segments[first_point + 1] = (xhi, ylo)
        block_segments[first_point + 2] = (xhi, ylo)
        block_segments[first_point + 3] = (xhi, yhi)
        block_segments[first_point + 4] = (xhi, yhi)
        block_segments[first_point + 5] = (xlo, yhi)
        block_segments[first_point + 6] = (xlo, yhi)
        block_segments[first_point + 7] = (xlo, ylo)
    end
    return AMRCellGeometry(
        points, triangles, cells, block_segments,
        mesh.block_size
    )
end

function _face_normal_intervals(
        mesh::BlockTreeMesh{T}, face::InteriorFace,
        axis::Int
    ) where {T}
    minus_width = mesh.blocks[face.minus.block].spacing[axis] / 2
    plus_width = mesh.blocks[face.plus.block].spacing[axis] / 2
    lower = face.center[axis] - minus_width
    upper = face.center[axis] + plus_width
    domain_lower = mesh.origin[axis]
    domain_upper = domain_lower + mesh.extent[axis]
    if !mesh.periodic[axis]
        return [(max(lower, domain_lower), min(upper, domain_upper))]
    elseif lower < domain_lower
        return [
            (domain_lower, min(upper, domain_upper)),
            (domain_upper - (domain_lower - lower), domain_upper),
        ]
    elseif upper > domain_upper
        return [
            (max(lower, domain_lower), domain_upper),
            (domain_lower, domain_lower + (upper - domain_upper)),
        ]
    end
    return [(lower, upper)]
end

function _face_normal_intervals(
        mesh::BlockTreeMesh{T}, face::BoundaryFace,
        axis::Int
    ) where {T}
    half_width = mesh.blocks[face.cell.block].spacing[axis] / 2
    lower = face.side < 0 ? face.center[axis] :
        face.center[axis] - half_width
    upper = face.side < 0 ? face.center[axis] + half_width :
        face.center[axis]
    return [(lower, upper)]
end

function _push_face_tile!(
        geometry_points, geometry_triangles,
        geometry_faces, value_indices, reference,
        value_index, axis, normal_lower, normal_upper,
        tangent_lower, tangent_upper
    )
    normal_upper > normal_lower || return
    tangent_upper > tangent_lower || return
    xlower, xupper, ylower, yupper = if axis == 1
        (normal_lower, normal_upper, tangent_lower, tangent_upper)
    else
        (tangent_lower, tangent_upper, normal_lower, normal_upper)
    end
    first_vertex = length(geometry_points) + 1
    push!(
        geometry_points,
        (xlower, ylower), (xupper, ylower),
        (xupper, yupper), (xlower, yupper)
    )
    push!(
        geometry_triangles,
        (first_vertex, first_vertex + 1, first_vertex + 2),
        (first_vertex, first_vertex + 2, first_vertex + 3)
    )
    push!(geometry_faces, reference)
    push!(value_indices, value_index)
    return
end

"""Construct filled staggered dual-cell geometry for one face direction."""
function visualization_geometry(mesh::BlockTreeMesh{T}, ::Face{Axis}) where {T, Axis}
    points = NTuple{2, T}[]
    triangles = NTuple{3, Int}[]
    faces = FaceRef[]
    value_indices = Int[]
    value_index = 0
    tangent_axis = 3 - Axis
    tangent_domain_lower = mesh.origin[tangent_axis]
    tangent_domain_upper =
        tangent_domain_lower + mesh.extent[tangent_axis]

    for (global_index, face) in pairs(mesh.faces)
        face.axis == Axis || continue
        value_index += 1
        tangent_lower = max(
            tangent_domain_lower, face.center[tangent_axis] - face.area / 2
        )
        tangent_upper = min(
            tangent_domain_upper, face.center[tangent_axis] + face.area / 2
        )
        for (normal_lower, normal_upper) in
            _face_normal_intervals(mesh, face, Axis)
            _push_face_tile!(
                points, triangles, faces, value_indices,
                FaceRef(false, global_index), value_index, Axis,
                normal_lower, normal_upper, tangent_lower, tangent_upper
            )
        end
    end
    for (global_index, face) in pairs(mesh.boundary_faces)
        face.axis == Axis || continue
        value_index += 1
        tangent_lower = max(
            tangent_domain_lower, face.center[tangent_axis] - face.area / 2
        )
        tangent_upper = min(
            tangent_domain_upper, face.center[tangent_axis] + face.area / 2
        )
        for (normal_lower, normal_upper) in
            _face_normal_intervals(mesh, face, Axis)
            _push_face_tile!(
                points, triangles, faces, value_indices,
                FaceRef(true, global_index), value_index, Axis,
                normal_lower, normal_upper, tangent_lower, tangent_upper
            )
        end
    end
    return AMRFaceGeometry{Axis, T}(
        points, triangles, faces, value_indices
    )
end
