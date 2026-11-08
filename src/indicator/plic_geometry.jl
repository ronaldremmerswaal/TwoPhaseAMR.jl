"""Return physical PLIC line segments of a reconstructed field."""
function plic_segments(field::InterfaceField{T}) where {T}
    segments = NTuple{2, NTuple{2, T}}[]
    nx, ny = field.mesh.block_size
    for block in eachindex(field.mesh.blocks), j in 1:ny, i in 1:nx
        ref = CellRef(block, i, j)
        fraction = _conditional_fraction(field, ref)
        zero(T) < fraction < one(T) || continue
        segment = _local_plic_segment(field, ref)
        isnothing(segment) && continue
        points = segment
        leaf = field.mesh.blocks[block]
        center = cellcenter(field.mesh, block, i, j)
        push!(segments, (
            _local_to_physical(T, center, leaf.spacing, points[1]),
            _local_to_physical(T, center, leaf.spacing, points[2]),
        ))
    end
    return segments
end

plic_segments(geometry::GeometryField) = plic_segments(geometry.solid)

"""
    PLICSolidGeometry

Triangulated physical-space solid reconstructed from the cell-local PLIC
planes of a [`GeometryField`](@ref).  Vertices are private to each cut cell so
the object remains valid across later mesh adaptation.
"""
struct PLICSolidGeometry{T}
    points::Vector{NTuple{2, T}}
    triangles::Vector{NTuple{3, Int}}
end

"""Return a filled PLIC reconstruction of the solid for visualization."""
function plic_solid_geometry(geometry::GeometryField{T}) where {T}
    points = NTuple{2, T}[]
    triangles = NTuple{3, Int}[]
    square = _unit_cell_polygon(T)
    nx, ny = geometry.mesh.block_size
    for block in eachindex(geometry.mesh.blocks), j in 1:ny, i in 1:nx
        ref = CellRef(block, i, j)
        fraction = _get(geometry.solid.fraction, ref)
        fraction <= zero(T) && continue
        polygon = if fraction >= one(T)
            square
        else
            _clip_polygon(
                square,
                _get(geometry.solid.normal_x, ref),
                _get(geometry.solid.normal_y, ref),
                _get(geometry.solid.intercept, ref); keep_less = true,
            )
        end
        length(polygon) >= 3 || continue
        leaf = geometry.mesh.blocks[block]
        center = cellcenter(geometry.mesh, block, i, j)
        first_vertex = length(points) + 1
        for point in polygon
            push!(points, _local_to_physical(T, center, leaf.spacing, point))
        end
        for vertex in 2:(length(polygon) - 1)
            push!(
                triangles, (
                    first_vertex, first_vertex + vertex - 1,
                    first_vertex + vertex,
                )
            )
        end
    end
    return PLICSolidGeometry(points, triangles)
end
