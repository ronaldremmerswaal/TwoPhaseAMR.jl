_unit_cell_polygon(::Type{T}) where {T} = NTuple{2, T}[
    (-T(0.5), -T(0.5)), (T(0.5), -T(0.5)),
    (T(0.5), T(0.5)), (-T(0.5), T(0.5)),
]

@inline _local_to_physical(center, spacing, point) = (
    center[1] + point[1] * spacing[1],
    center[2] + point[2] * spacing[2],
)
@inline _local_to_physical(::Type{T}, center, spacing, point) where {T} = (
    T(center[1] + point[1] * spacing[1]),
    T(center[2] + point[2] * spacing[2]),
)

@inline function _polygon_crossing(first, second, value_first, value_second)
    fraction = value_first / (value_first - value_second)
    return (
        first[1] + fraction * (second[1] - first[1]),
        first[2] + fraction * (second[2] - first[2]),
    )
end

function _clip_polygon(poly, nx, ny, intercept; keep_less = true)
    isempty(poly) && return copy(poly)
    result = eltype(poly)[]
    previous = last(poly)
    previous_value = nx * previous[1] + ny * previous[2] - intercept
    previous_inside = keep_less ? previous_value <= 0 : previous_value >= 0
    for current in poly
        current_value = nx * current[1] + ny * current[2] - intercept
        current_inside = keep_less ? current_value <= 0 : current_value >= 0
        if current_inside != previous_inside
            push!(
                result,
                _polygon_crossing(
                    previous, current, previous_value, current_value,
                ),
            )
        end
        current_inside && push!(result, current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    end
    return result
end

function _polygon_area(poly::AbstractVector{<:NTuple{2, T}}) where {T}
    length(poly) >= 3 || return zero(T)
    area = zero(T)
    previous = last(poly)
    for point in poly
        area += previous[1] * point[2] - previous[2] * point[1]
        previous = point
    end
    return abs(area) / 2
end

"""Return polygon area and its two first moments in the same coordinates."""
function _polygon_area_moments(poly::AbstractVector{<:NTuple{2, T}}) where {T}
    length(poly) >= 3 || return (zero(T), zero(T), zero(T))
    twice_area = zero(T)
    first_x = zero(T)
    first_y = zero(T)
    previous = last(poly)
    for point in poly
        cross = previous[1] * point[2] - point[1] * previous[2]
        twice_area += cross
        first_x += (previous[1] + point[1]) * cross
        first_y += (previous[2] + point[2]) * cross
        previous = point
    end
    orientation = ifelse(twice_area < zero(T), -one(T), one(T))
    return (
        orientation * twice_area / 2,
        orientation * first_x / 6,
        orientation * first_y / 6,
    )
end

function _clip_to_rectangle(polygon, xlower, xupper, ylower, yupper)
    clipped = _clip_polygon(
        polygon, -one(xlower), zero(xlower), -xlower; keep_less = true,
    )
    clipped = _clip_polygon(
        clipped, one(xupper), zero(xupper), xupper; keep_less = true,
    )
    clipped = _clip_polygon(
        clipped, zero(ylower), -one(ylower), -ylower; keep_less = true,
    )
    return _clip_polygon(
        clipped, zero(yupper), one(yupper), yupper; keep_less = true,
    )
end

function _local_plic_segment(field::InterfaceField{T}, ref::CellRef) where {T}
    fraction = _conditional_fraction(field, ref)
    zero(T) < fraction < one(T) || return nothing
    nx = _get(field.normal_x, ref)
    ny = _get(field.normal_y, ref)
    intercept = _get(field.intercept, ref)
    polygon = isnothing(field.geometry) ?
        _unit_cell_polygon(T) : _open_cell_polygon(field.geometry, ref)
    length(polygon) >= 3 || return nothing
    tolerance = T(64) * eps(T)
    points = NTuple{2, T}[]
    previous = last(polygon)
    previous_value = nx * previous[1] + ny * previous[2] - intercept
    for current in polygon
        current_value = nx * current[1] + ny * current[2] - intercept
        abs(previous_value) <= tolerance && push!(points, previous)
        if (previous_value < -tolerance && current_value > tolerance) ||
                (previous_value > tolerance && current_value < -tolerance)
            push!(
                points,
                _polygon_crossing(
                    previous, current, previous_value, current_value,
                ),
            )
        end
        previous = current
        previous_value = current_value
    end
    unique_points = NTuple{2, T}[]
    for point in points
        any(
            other -> hypot(point[1] - other[1], point[2] - other[2]) <=
                tolerance, unique_points,
        ) || push!(unique_points, point)
    end
    length(unique_points) >= 2 || return nothing
    best = (unique_points[1], unique_points[2])
    best_distance = -one(T)
    for first in 1:(length(unique_points) - 1),
            second in (first + 1):length(unique_points)
        distance = (unique_points[first][1] - unique_points[second][1])^2 +
            (unique_points[first][2] - unique_points[second][2])^2
        if distance > best_distance
            best = (unique_points[first], unique_points[second])
            best_distance = distance
        end
    end
    return best_distance > tolerance^2 ? best : nothing
end
