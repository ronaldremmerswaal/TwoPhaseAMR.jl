"""
    visualization_values(field)

Copy the owned (non-halo) values of a cell field into the same block-major,
`j`, `i` ordering as `visualization_geometry(mesh).cells`.
"""
function visualization_values(field::CellField{T}) where {T}
    nx, ny = field.mesh.block_size
    values = Vector{T}(undef, ncells(field.mesh))
    index = 0
    for block in eachindex(field.blocks)
        owned = interior(field, block)
        for j in 1:ny, i in 1:nx
            index += 1
            @inbounds values[index] = owned[i, j]
        end
    end
    return values
end

"""Copy owned directional face values in the order used by face geometry."""
function visualization_values(field::FaceField{Axis, T}) where {Axis, T}
    values = Vector{T}(undef, length(field.interior) + length(field.boundary))
    copyto!(values, 1, field.interior, 1, length(field.interior))
    copyto!(
        values, length(field.interior) + 1,
        field.boundary, 1, length(field.boundary)
    )
    return values
end

function visualization_values(
        velocity::FaceVelocity{T},
        ::Face{Axis}
    ) where {T, Axis}
    count = _nfaces(velocity.mesh, Axis) +
        _nboundary_faces(velocity.mesh, Axis)
    values = Vector{T}(undef, count)
    local_index = 0
    for (global_index, face) in pairs(velocity.mesh.faces)
        face.axis == Axis || continue
        local_index += 1
        @inbounds values[local_index] = velocity.interior[global_index]
    end
    for (global_index, face) in pairs(velocity.mesh.boundary_faces)
        face.axis == Axis || continue
        local_index += 1
        @inbounds values[local_index] = velocity.boundary[global_index]
    end
    return values
end

@inline function _visualization_cell_index(mesh::BlockTreeMesh, cell::CellRef)
    nx, ny = mesh.block_size
    return (cell.block - 1) * nx * ny + (cell.j - 1) * nx + cell.i
end

@inline _visualization_side_slot(axis::Int, side::Int) =
    2axis - (side < 0 ? 1 : 0)

function _cell_centered_velocity(velocity::FaceVelocity{T}) where {T}
    mesh = velocity.mesh
    count = ncells(mesh)
    weighted = zeros(T, 4, count)
    measures = zeros(T, 4, count)
    for (face_index, face) in pairs(mesh.faces)
        value = @inbounds velocity.interior[face_index]
        for (cell, side) in ((face.minus, 1), (face.plus, -1))
            index = _visualization_cell_index(mesh, cell)
            slot = _visualization_side_slot(face.axis, side)
            @inbounds weighted[slot, index] += value * face.area
            @inbounds measures[slot, index] += face.area
        end
    end
    for (face_index, face) in pairs(mesh.boundary_faces)
        index = _visualization_cell_index(mesh, face.cell)
        slot = _visualization_side_slot(face.axis, face.side)
        value = @inbounds velocity.boundary[face_index]
        @inbounds weighted[slot, index] += value * face.area
        @inbounds measures[slot, index] += face.area
    end

    x = Vector{T}(undef, count)
    y = Vector{T}(undef, count)
    for index in 1:count
        all(@view(measures[:, index]) .> zero(T)) ||
            throw(ArgumentError("incomplete cell-face schedule"))
        @inbounds x[index] = (
            weighted[1, index] / measures[1, index] +
                weighted[2, index] / measures[2, index]
        ) / 2
        @inbounds y[index] = (
            weighted[3, index] / measures[3, index] +
                weighted[4, index] / measures[4, index]
        ) / 2
    end
    return x, y
end

function visualization_values(
        velocity::FaceVelocity{T},
        ::CenteredMagnitude
    ) where {T}
    x, y = _cell_centered_velocity(velocity)
    return hypot.(x, y)
end

function visualization_values(
        fields::Tuple{X, Y},
        derived::Union{CenteredMagnitude, Vorticity}
    ) where
    {X <: FaceField{1}, Y <: FaceField{2}}
    fields[1].mesh === fields[2].mesh ||
        throw(ArgumentError("face-field components must share one mesh"))
    return visualization_values(FaceVelocity(fields...), derived)
end

function _cell_centers(mesh::BlockTreeMesh{T}) where {T}
    nx, ny = mesh.block_size
    centers = Vector{NTuple{2, T}}(undef, ncells(mesh))
    index = 0
    for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        index += 1
        centers[index] = cellcenter(mesh, block, i, j)
    end
    return centers
end

function visualization_values(velocity::FaceVelocity{T}, ::Vorticity) where {T}
    mesh = velocity.mesh
    xvelocity, yvelocity = _cell_centered_velocity(velocity)
    centers = _cell_centers(mesh)
    count = length(centers)
    a11 = zeros(T, count)
    a12 = zeros(T, count)
    a22 = zeros(T, count)
    bx1 = zeros(T, count)
    bx2 = zeros(T, count)
    by1 = zeros(T, count)
    by2 = zeros(T, count)

    for face in mesh.faces
        minus = _visualization_cell_index(mesh, face.minus)
        plus = _visualization_cell_index(mesh, face.plus)
        dx = T(
            _wrapped_delta(
                centers[plus][1] - centers[minus][1],
                mesh.extent[1], mesh.periodic[1]
            )
        )
        dy = T(
            _wrapped_delta(
                centers[plus][2] - centers[minus][2],
                mesh.extent[2], mesh.periodic[2]
            )
        )
        distance_squared = dx^2 + dy^2
        distance_squared > zero(T) || continue
        weight = T(face.area) / distance_squared
        du = xvelocity[plus] - xvelocity[minus]
        dv = yvelocity[plus] - yvelocity[minus]
        for index in (minus, plus)
            @inbounds a11[index] += weight * dx^2
            @inbounds a12[index] += weight * dx * dy
            @inbounds a22[index] += weight * dy^2
            @inbounds bx1[index] += weight * du * dx
            @inbounds bx2[index] += weight * du * dy
            @inbounds by1[index] += weight * dv * dx
            @inbounds by2[index] += weight * dv * dy
        end
    end

    result = Vector{T}(undef, count)
    for index in 1:count
        determinant = @inbounds a11[index] * a22[index] - a12[index]^2
        scale = @inbounds max(abs(a11[index] * a22[index]), a12[index]^2)
        determinant > 128eps(T) * scale ||
            throw(ArgumentError("cell-neighbor geometry is rank deficient"))
        dvdx = @inbounds (
            a22[index] * by1[index] -
                a12[index] * by2[index]
        ) / determinant
        dudy = @inbounds (
            a11[index] * bx2[index] -
                a12[index] * bx1[index]
        ) / determinant
        @inbounds result[index] = dvdx - dudy
    end
    return result
end

function _visualization_vertex_values(field::CellField)
    cell_values = visualization_values(field)
    values = Vector{eltype(cell_values)}(undef, 4length(cell_values))
    for index in eachindex(cell_values)
        value = @inbounds cell_values[index]
        first_vertex = 4index - 3
        @inbounds for vertex in first_vertex:(first_vertex + 3)
            values[vertex] = value
        end
    end
    return values
end

function _visualization_vertex_values(values::AbstractVector)
    result = Vector{eltype(values)}(undef, 4length(values))
    for index in eachindex(values)
        first_vertex = 4index - 3
        @inbounds result[first_vertex] = values[index]
        @inbounds result[first_vertex + 1] = values[index]
        @inbounds result[first_vertex + 2] = values[index]
        @inbounds result[first_vertex + 3] = values[index]
    end
    return result
end

function _visualization_face_vertex_values(
        values::AbstractVector, geometry::AMRFaceGeometry
    )
    result = Vector{eltype(values)}(undef, 4length(geometry.faces))
    for tile in eachindex(geometry.faces)
        value = values[geometry.value_indices[tile]]
        first_vertex = 4tile - 3
        @inbounds result[first_vertex] = value
        @inbounds result[first_vertex + 1] = value
        @inbounds result[first_vertex + 2] = value
        @inbounds result[first_vertex + 3] = value
    end
    return result
end

"""
    cell_for_vertex(geometry, vertex_index)

Return the cell owning a picked mesh vertex, or `nothing` for an invalid
index. Makie reports the largest vertex index of a picked mesh triangle, which
is compatible with this four-private-vertices-per-cell mapping.
"""
function cell_for_vertex(geometry::AMRCellGeometry, vertex_index::Integer)
    1 <= vertex_index <= length(geometry.points) || return nothing
    return @inbounds geometry.cells[cld(vertex_index, 4)]
end

"""Return the face owning a picked dual-cell vertex, or `nothing`."""
function face_for_vertex(geometry::AMRFaceGeometry, vertex_index::Integer)
    1 <= vertex_index <= length(geometry.points) || return nothing
    return @inbounds geometry.faces[cld(vertex_index, 4)]
end

function cell_for_vertex(
        geometry::PLICVisualizationGeometry{CellRef}, vertex_index::Integer,
    )
    1 <= vertex_index <= length(geometry.points) || return nothing
    return @inbounds geometry.owners[vertex_index]
end

function face_for_vertex(
        geometry::PLICVisualizationGeometry{FaceRef}, vertex_index::Integer,
    )
    1 <= vertex_index <= length(geometry.points) || return nothing
    return @inbounds geometry.owners[vertex_index]
end

function _face_outline(
        geometry::AMRFaceGeometry,
        vertex_index::Integer
    )
    1 <= vertex_index <= length(geometry.points) ||
        return eltype(geometry.points)[]
    first_vertex = 4cld(vertex_index, 4) - 3
    southwest = geometry.points[first_vertex]
    southeast = geometry.points[first_vertex + 1]
    northeast = geometry.points[first_vertex + 2]
    northwest = geometry.points[first_vertex + 3]
    return [
        southwest, southeast, southeast, northeast,
        northeast, northwest, northwest, southwest,
    ]
end

function _face_outline(geometry::AMRFaceGeometry, reference::FaceRef)
    tile = findfirst(==(reference), geometry.faces)
    isnothing(tile) && return eltype(geometry.points)[]
    return _face_outline(geometry, 4tile)
end

function _cell_outline(geometry::AMRCellGeometry, cell::CellRef)
    nx, ny = geometry.block_size
    1 <= cell.block || return eltype(geometry.points)[]
    1 <= cell.i <= nx || return eltype(geometry.points)[]
    1 <= cell.j <= ny || return eltype(geometry.points)[]
    cell_index = (cell.block - 1) * nx * ny + (cell.j - 1) * nx + cell.i
    cell_index <= length(geometry.cells) || return eltype(geometry.points)[]
    first_vertex = 4cell_index - 3
    southwest = geometry.points[first_vertex]
    southeast = geometry.points[first_vertex + 1]
    northeast = geometry.points[first_vertex + 2]
    northwest = geometry.points[first_vertex + 3]
    return [
        southwest, southeast, southeast, northeast,
        northeast, northwest, northwest, southwest,
    ]
end
