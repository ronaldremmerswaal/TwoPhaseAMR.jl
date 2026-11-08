"""Interface for block-wise AMR label estimators."""
abstract type AbstractRefinementIndicator end

function _refinement_region(::Type{T}, xmin, xmax, ymin, ymax) where {T}
    all(x -> x isa Real && !isnan(x), (xmin, xmax, ymin, ymax)) ||
        throw(ArgumentError("region bounds must be real and not NaN"))
    bounds = (T(xmin), T(xmax), T(ymin), T(ymax))
    bounds[1] < bounds[2] && bounds[3] < bounds[4] ||
        throw(ArgumentError("region bounds must satisfy xmin < xmax and ymin < ymax"))
    return bounds
end

function _apply_refinement_region!(labels, indicator)
    xmin, xmax, ymin, ymax = indicator.region
    xmin == -Inf && xmax == Inf && ymin == -Inf && ymax == Inf && return labels
    for (b, block) in pairs(_indicator_field(indicator).mesh.blocks)
        x, y = block.origin
        dx, dy = block.extent
        if x > xmax || x + dx < xmin || y > ymax || y + dy < ymin
            labels[b] = :coarsen
        end
    end
    return labels
end

"""
    InterpolationErrorIndicator(field; E_refine, E_coarsen,
                                relative_floor=sqrt(eps(T)),
                                boundary=ZeroGradient(), minimum_level=0,
                                maximum_level=field.mesh.max_level,
                                t_init=-Inf, t_end=Inf,
                                xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf)

Block indicator for a cell-average field. At every owned cell, compare the
stored value with the centred interpolation of its four axial neighbours and
normalize by their local magnitude. A block requests `:refine` when its
maximum relative defect exceeds `E_refine`, `:coarsen` when it is below
`E_coarsen`, and `:keep` in between.

The two thresholds provide value hysteresis and must satisfy
`0 <= E_coarsen < E_refine`. `relative_floor` prevents a meaningless relative
error where the field and all its neighbours vanish. The criterion is active
for `t_init <= time <= t_end`. Its level bounds apply independently of the
error thresholds.
The spatial bounds select blocks whose closed rectangles intersect the region;
outside it the indicator requests `:coarsen` and does not enforce its level floor.
"""
struct InterpolationErrorIndicator{T, F <: CellField, B <: AbstractBoundaryCondition} <:
    AbstractRefinementIndicator
    field::F
    E_refine::T
    E_coarsen::T
    relative_floor::T
    boundary::B
    minimum_level::Int
    maximum_level::Int
    t_init::T
    t_end::T
    region::NTuple{4, T}
end

@inline _indicator_field(indicator::InterpolationErrorIndicator) =
    indicator.field

function InterpolationErrorIndicator(
        field::CellField{T}; E_refine, E_coarsen,
        relative_floor = sqrt(eps(T)), boundary = ZeroGradient(),
        minimum_level = 0, maximum_level = field.mesh.max_level,
        t_init = -Inf, t_end = Inf,
        xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf
    ) where {T}
    refine = T(E_refine)
    coarsen = T(E_coarsen)
    floor = T(relative_floor)
    zero(T) <= coarsen < refine || throw(
        ArgumentError(
            "indicator thresholds must satisfy 0 <= E_coarsen < E_refine"
        )
    )
    floor > zero(T) || throw(
        ArgumentError(
            "indicator relative_floor must be strictly positive"
        )
    )
    lower_level, upper_level = _refinement_levels(
        field.mesh, minimum_level, maximum_level
    )
    start_time, end_time = _refinement_window(T, t_init, t_end)
    return InterpolationErrorIndicator(
        field, refine, coarsen, floor, boundary,
        lower_level, upper_level, start_time, end_time,
        _refinement_region(T, xmin, xmax, ymin, ymax)
    )
end

function _refinement_levels(mesh, minimum_level, maximum_level)
    minimum_level isa Integer && !(minimum_level isa Bool) ||
        throw(ArgumentError("minimum_level must be an integer"))
    maximum_level isa Integer && !(maximum_level isa Bool) ||
        throw(ArgumentError("maximum_level must be an integer"))
    0 <= minimum_level <= maximum_level <= mesh.max_level ||
        throw(
        ArgumentError(
            "minimum_level and maximum_level must lie within mesh limits"
        )
    )
    return Int(minimum_level), Int(maximum_level)
end

function _refinement_window(::Type{T}, t_init, t_end) where {T}
    start_time = T(t_init)
    end_time = T(t_end)
    !isnan(start_time) && !isnan(end_time) && start_time <= end_time ||
        throw(ArgumentError("require t_init <= t_end and neither may be NaN"))
    return start_time, end_time
end

function _indicator_active(indicator, time)
    time isa Real && !isnan(time) ||
        throw(ArgumentError("refinement time must be a real number and not NaN"))
    return indicator.t_init <= time <= indicator.t_end
end

@inline _indicator_can_coarsen(level, minimum_level) =
    iszero(level) || level > minimum_level

@inline _refinement_priority(action::Symbol) =
    action === :coarsen ? 0 : action === :keep ? 1 : action === :refine ? 2 :
    throw(ArgumentError("refinement action must be :refine, :keep, or :coarsen"))

@inline function _combine_refinement_actions(first::Symbol, second::Symbol)
    return _refinement_priority(first) >= _refinement_priority(second) ? first : second
end

"""
    refinement_labels!(labels, indicator; time=0)
    refinement_labels!(workspace, indicators; time=0)

Write one `:refine`, `:keep`, or `:coarsen` proposal per active block. Multiple
indicators are combined with `:refine > :keep > :coarsen` precedence. Built-in
indicators outside their inclusive activation window contribute the neutral
`:coarsen` proposal.
"""
function refinement_labels!(
        labels::AbstractVector{Symbol},
        indicator::InterpolationErrorIndicator;
        time = 0
    )
    field = indicator.field
    length(labels) == nblocks(field.mesh) || throw(
        DimensionMismatch(
            "one refinement label is required per active block"
        )
    )
    if !_indicator_active(indicator, time)
        fill!(labels, :coarsen)
        return labels
    end
    fill_halo!(field, indicator.boundary)
    nx, ny = field.mesh.block_size
    h = field.mesh.halo
    _parallel_for(length(field.blocks); grain = 4) do block_index
        values = field.blocks[block_index]
        maximum_error = zero(typeof(indicator.E_refine))
        for j in 1:ny, i in 1:nx
            @inbounds begin
                value = values[i + h, j + h]
                left = values[i + h - 1, j + h]
                right = values[i + h + 1, j + h]
                below = values[i + h, j + h - 1]
                above = values[i + h, j + h + 1]
            end
            interpolated = (left + right + below + above) / 4
            scale = max(
                indicator.relative_floor, abs(value), abs(left),
                abs(right), abs(below), abs(above)
            )
            maximum_error = max(
                maximum_error, abs(value - interpolated) / scale
            )
        end
        level = field.mesh.blocks[block_index].key.level
        @inbounds labels[block_index] =
            level < indicator.minimum_level ? :refine :
            level > indicator.maximum_level ? :coarsen :
            maximum_error > indicator.E_refine &&
            level < indicator.maximum_level ? :refine :
            maximum_error < indicator.E_coarsen &&
            _indicator_can_coarsen(
                level, indicator.minimum_level
            ) ? :coarsen : :keep
    end
    return _apply_refinement_region!(labels, indicator)
end


_refinement_labels_at_time!(labels, indicator, time) =
    refinement_labels!(labels, indicator)

_refinement_labels_at_time!(
    labels, indicator::InterpolationErrorIndicator, time
) = refinement_labels!(labels, indicator; time)
