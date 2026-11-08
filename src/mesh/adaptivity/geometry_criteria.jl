"""
    GeometryRefinementIndicator(geometry; level, buffer_cells=1,
                                t_init=-Inf, t_end=Inf,
                                xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf)

Refine blocks intersecting an embedded solid boundary, or its cell-halo
neighbourhood, to a fixed level. Blocks away from the boundary request
coarsening, including blocks wholly inside the fixed solid. A boundary aligned
with cell or block faces is detected from halo jumps in the solid fraction.
The criterion is active for `t_init <= time <= t_end`.
Only blocks intersecting the closed spatial region are eligible; outside it
the indicator requests `:coarsen`.
"""
struct GeometryRefinementIndicator{G <: GeometryField, T} <:
    AbstractRefinementIndicator
    field::G
    level::Int
    buffer_cells::Int
    t_init::T
    t_end::T
    region::NTuple{4, T}
end

function GeometryRefinementIndicator(
        field::GeometryField{T}; level,
        buffer_cells = 1, t_init = -Inf, t_end = Inf,
        xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf
    ) where {T}
    interface_indicator = InterfaceRefinementIndicator(
        field.solid; level, buffer_cells, t_init, t_end, xmin, xmax, ymin, ymax
    )
    return GeometryRefinementIndicator(
        field, interface_indicator.level, interface_indicator.buffer_cells,
        interface_indicator.t_init, interface_indicator.t_end, interface_indicator.region
    )
end

_indicator_field(indicator::GeometryRefinementIndicator) = indicator.field

function refinement_labels!(
        labels::AbstractVector{Symbol},
        indicator::GeometryRefinementIndicator;
        time = 0
    )
    length(labels) == nblocks(indicator.field.mesh) || throw(
        DimensionMismatch("one label is required per block")
    )
    if !_indicator_active(indicator, time)
        fill!(labels, :coarsen)
        return labels
    end
    _fixed_interface_refinement_labels!(
        labels, indicator.field.solid, indicator.level, indicator.buffer_cells
    )
    return _apply_refinement_region!(labels, indicator)
end

_refinement_labels_at_time!(
    labels, indicator::GeometryRefinementIndicator, time
) = refinement_labels!(labels, indicator; time)
