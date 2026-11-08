"""
    PolygonIndicator(vertices)

The indicator of the interior of a simple polygon. Vertices may be clockwise
or counter-clockwise, and a repeated closing vertex is optional. Polygon cell
fractions are obtained by exact polygon clipping, so this representation is
appropriate for piecewise-linear boundaries that should not be sent through
smooth level-set quadrature.
"""
struct PolygonIndicator{T}
    vertices::Vector{NTuple{2, T}}

    function PolygonIndicator{T}(
            vertices::Vector{NTuple{2, T}}, ::Val{:validated},
        ) where {T}
        return new{T}(vertices)
    end
end

Base.eltype(::PolygonIndicator{T}) where {T} = T

@inline function _polygon_orientation(a, b, c)
    return (b[1] - a[1]) * (c[2] - a[2]) -
        (b[2] - a[2]) * (c[1] - a[1])
end

@inline function _polygon_between(value, first, second)
    lo, hi = minmax(first, second)
    return lo <= value <= hi
end

function _polygon_segments_intersect(a, b, c, d)
    ab_c = _polygon_orientation(a, b, c)
    ab_d = _polygon_orientation(a, b, d)
    cd_a = _polygon_orientation(c, d, a)
    cd_b = _polygon_orientation(c, d, b)
    if iszero(ab_c) && _polygon_between(c[1], a[1], b[1]) &&
            _polygon_between(c[2], a[2], b[2])
        return true
    elseif iszero(ab_d) && _polygon_between(d[1], a[1], b[1]) &&
            _polygon_between(d[2], a[2], b[2])
        return true
    elseif iszero(cd_a) && _polygon_between(a[1], c[1], d[1]) &&
            _polygon_between(a[2], c[2], d[2])
        return true
    elseif iszero(cd_b) && _polygon_between(b[1], c[1], d[1]) &&
            _polygon_between(b[2], c[2], d[2])
        return true
    end
    crosses_ab = (ab_c > 0 && ab_d < 0) || (ab_c < 0 && ab_d > 0)
    crosses_cd = (cd_a > 0 && cd_b < 0) || (cd_a < 0 && cd_b > 0)
    return crosses_ab && crosses_cd
end

function _validate_simple_polygon(vertices)
    count = length(vertices)
    count >= 3 || throw(
        ArgumentError("a polygon indicator requires at least three vertices"),
    )
    for index in eachindex(vertices)
        vertices[index] != vertices[mod1(index + 1, count)] || throw(
            ArgumentError("consecutive polygon vertices must be distinct"),
        )
    end
    _polygon_area(vertices) > zero(eltype(first(vertices))) || throw(
        ArgumentError("polygon vertices must enclose a nonzero area"),
    )
    for first_edge in 1:count
        a = vertices[first_edge]
        b = vertices[mod1(first_edge + 1, count)]
        for second_edge in (first_edge + 1):count
            adjacent = second_edge == first_edge + 1 ||
                (first_edge == 1 && second_edge == count)
            adjacent && continue
            c = vertices[second_edge]
            d = vertices[mod1(second_edge + 1, count)]
            _polygon_segments_intersect(a, b, c, d) && throw(
                ArgumentError("polygon boundary must not intersect itself"),
            )
        end
    end
    return vertices
end

function PolygonIndicator(vertices::AbstractVector)
    isempty(vertices) && throw(
        ArgumentError("a polygon indicator requires at least three vertices"),
    )
    first_vertex = first(vertices)
    valid_first_vertex = try
        length(first_vertex) == 2
    catch
        false
    end
    valid_first_vertex || throw(
        ArgumentError("polygon vertices must contain two coordinates"),
    )
    first_vertex[1] isa Real && !(first_vertex[1] isa Bool) &&
        first_vertex[2] isa Real && !(first_vertex[2] isa Bool) || throw(
        ArgumentError("polygon coordinates must be real numbers"),
    )
    T = promote_type(
        typeof(float(first_vertex[1])), typeof(float(first_vertex[2])),
    )
    for vertex in vertices
        valid_vertex = try
            length(vertex) == 2
        catch
            false
        end
        valid_vertex || throw(
            ArgumentError("polygon vertices must contain two coordinates"),
        )
        for coordinate in vertex
            coordinate isa Real && !(coordinate isa Bool) || throw(
                ArgumentError("polygon coordinates must be real numbers"),
            )
            T = promote_type(T, typeof(float(coordinate)))
        end
    end
    points = NTuple{2, T}[
        (T(vertex[1]), T(vertex[2])) for vertex in vertices
    ]
    if length(points) > 1 && points[begin] == points[end]
        pop!(points)
    end
    all(point -> all(isfinite, point), points) || throw(
        ArgumentError("polygon coordinates must be finite"),
    )
    return PolygonIndicator{T}(
        _validate_simple_polygon(points), Val(:validated),
    )
end

"""
    InterfaceField(mesh, polygon::PolygonIndicator; kwargs...)
    initialize!(interface, polygon::PolygonIndicator)

Initialize liquid volume fractions from a polygon using geometric clipping.
With embedded geometry, the clipped polygon is intersected with each cell's
open PLIC polygon. No numerical quadrature tolerance is involved.
"""
function InterfaceField(
        mesh::BlockTreeMesh, polygon::PolygonIndicator;
        T = eltype(mesh.origin),
        contact_angle = (pi / 2, pi / 2, pi / 2, pi / 2),
        geometry = nothing, embedded_contact_angle = missing,
        fraction_floor = sqrt(eps(T)), reconstruction_method = :lhf,
    )
    field = InterfaceField(
        mesh; T, contact_angle, geometry, embedded_contact_angle,
        fraction_floor, reconstruction_method,
    )
    return initialize!(field, polygon)
end

function initialize!(
        field::InterfaceField{T}, polygon::PolygonIndicator,
    ) where {T}
    mesh = field.mesh
    nx, ny = mesh.block_size
    halo = mesh.halo
    vertices = NTuple{2, T}[(T(x), T(y)) for (x, y) in polygon.vertices]
    xcoordinates = first.(vertices)
    ycoordinates = last.(vertices)
    polygon_xlower, polygon_xupper = extrema(xcoordinates)
    polygon_ylower, polygon_yupper = extrema(ycoordinates)
    result = CellField(mesh, T)

    _parallel_block_columns(mesh) do block_index, first_j, last_j
        block = @inbounds mesh.blocks[block_index]
        values = @inbounds result.blocks[block_index]
        dx, dy = block.spacing
        @inbounds for j in first_j:last_j, i in 1:nx
            xlower = block.origin[1] + (i - 1) * dx
            xupper = xlower + dx
            ylower = block.origin[2] + (j - 1) * dy
            yupper = ylower + dy
            disjoint = xupper <= polygon_xlower || xlower >= polygon_xupper ||
                yupper <= polygon_ylower || ylower >= polygon_yupper
            fraction = if disjoint
                zero(T)
            else
                clipped = _clip_to_rectangle(
                    vertices, xlower, xupper, ylower, yupper,
                )
                if isnothing(field.geometry)
                    _polygon_area(clipped) / block.volume
                else
                    center_x = (xlower + xupper) / 2
                    center_y = (ylower + yupper) / 2
                    local_polygon = NTuple{2, T}[
                        (
                            (point[1] - center_x) / dx,
                            (point[2] - center_y) / dy,
                        ) for point in clipped
                    ]
                    ref = CellRef(block_index, i, j)
                    open_polygon = _clip_polygon(
                        local_polygon,
                        _get(field.geometry.solid.normal_x, ref),
                        _get(field.geometry.solid.normal_y, ref),
                        _get(field.geometry.solid.intercept, ref);
                        keep_less = false,
                    )
                    _polygon_area(open_polygon)
                end
            end
            available = isnothing(field.geometry) ? one(T) :
                _get(field.geometry.open_fraction, CellRef(block_index, i, j))
            values[i + halo, j + halo] = clamp(
                fraction, zero(T), available,
            )
        end
    end
    _parallel_for(nblocks(mesh); grain = 4) do block
        copyto!(field.fraction.blocks[block], result.blocks[block])
    end
    return reconstruct!(field)
end
