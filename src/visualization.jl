"""
    AbstractVisualization

Supertype for optional simulation observers and interactive viewers. Numerical
code can dispatch on this interface without depending on a graphics package.
"""
abstract type AbstractVisualization end

"""A zero-cost visualization target for code paths that do not need output."""
struct NoVisualization <: AbstractVisualization end

"""
    AMRCellGeometry

Backend-independent triangulation of the owned cells in a `BlockTreeMesh`.
Every cell has four private vertices and two triangles. Repeating vertices is
intentional: it permits a discontinuous, piecewise-constant color per cell and
makes a picked mesh vertex map back to a cell by integer arithmetic.

`block_segments` contains pairs of endpoints for drawing leaf-block outlines.
No halo cells or `RegionTrees` objects are included.
"""
struct AMRCellGeometry{T}
    points::Vector{NTuple{2,T}}
    triangles::Vector{NTuple{3,Int}}
    cells::Vector{CellRef}
    block_segments::Vector{NTuple{2,T}}
    block_size::NTuple{2,Int}
end

"""
    visualization_geometry(mesh)

Materialize plotting geometry from the compact leaf-block records in `mesh`.
The ordering is block-major, then `j`, then `i`, matching
`visualization_values`.
"""
function visualization_geometry(mesh::BlockTreeMesh{T}) where {T}
    nx, ny = mesh.block_size
    count = ncells(mesh)
    points = Vector{NTuple{2,T}}(undef, 4count)
    triangles = Vector{NTuple{3,Int}}(undef, 2count)
    cells = Vector{CellRef}(undef, count)
    block_segments = Vector{NTuple{2,T}}(undef, 8nblocks(mesh))

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
    return AMRCellGeometry(points, triangles, cells, block_segments,
                           mesh.block_size)
end

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
    return [southwest, southeast, southeast, northeast,
            northeast, northwest, northwest, southwest]
end

"""
    visualize(field::CellField; kwargs...)

Create an interactive AMR field viewer. This method is supplied by the optional
Makie extension; load GLMakie or WGLMakie before calling it.
"""
function visualize end

"""Replace the values displayed by an existing visualization."""
function update_visualization! end

"""Return the cell selected in a visualization, or `nothing`."""
selected_cell(::NoVisualization) = nothing

"""Return the backend-specific figure owned by a visualization."""
function visualization_figure end

update_visualization!(visualization::NoVisualization, args...; kwargs...) =
    visualization
