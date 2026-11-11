module TwoPhaseAMRMakieExt

import Makie
import TwoPhaseAMR
import TwoPhaseAMR: amrfield, amrfield!
using TwoPhaseAMR: BlockTreeMesh, ViewCamera, DiagnosticCollection, DiagnosticGroup,
    DiagnosticInfo, FieldStats, FlowSimulation, Frame, FrameInfo, PhysicalDimension, Recorder,
    SnapshotStore, TopologySnapshot, VisualizationFieldRegistry,
    diagnostic_groups, diagnostic_history, diagnostic_info, diagnostic_names,
    frame_at, keyframes, newest_frame, rasterize, snapshot_store, video_frame_schedule,
    LiveVideoSampler, push_live_frame!, flush_live_frames!, visible_rect

const _automatic = Makie.automatic

_frame(frame::Frame) = frame
_frame(registry::VisualizationFieldRegistry) = TwoPhaseAMR._registry_frame(registry)

_store(store::SnapshotStore) = store
_store(recorder::Recorder) = snapshot_store(recorder)

_domain_rect(domain) = TwoPhaseAMR._domain_rect(domain)

# ---------------------------------------------------------------------------
# Colour policy

function _field_title(entry)
    isempty(entry.units) && return entry.label
    return "$(entry.label) [$(entry.units)]"
end

function _automatic_colorrange(stats::FieldStats, symmetric::Bool)
    stats.finite == 0 && return symmetric ? (-1.0f0, 1.0f0) : (0.0f0, 1.0f0)
    if symmetric
        largest = Float32(stats.absmax)
        iszero(largest) && return (-0.5f0, 0.5f0)
        return (-largest, largest)
    end
    lower, upper = Float32(stats.minimum), Float32(stats.maximum)
    if lower == upper
        padding = iszero(lower) ? 0.5f0 : max(0.01f0 * abs(lower), eps(lower))
        return (lower - padding, upper + padding)
    end
    return (lower, upper)
end

# cmocean's diverging maps carry no "diverging" note in ColorSchemes.
const _DIVERGING_CMOCEAN = (:balance, :curl, :delta, :diff, :tarn)

_diverging(colormap::Makie.Reverse) = _diverging(colormap.data)
function _diverging(colormap::Union{Symbol, AbstractString})
    name = Symbol(colormap)
    name in _DIVERGING_CMOCEAN && return true
    scheme = get(Makie.ColorSchemes.colorschemes, name, nothing)
    return !isnothing(scheme) && occursin("diverging", lowercase(scheme.notes))
end
_diverging(colormap) = false

_requested(value) = !(isnothing(value) || value isa Makie.Automatic)

_resolve_colormap(entry, requested) = _requested(requested) ? requested :
    isnothing(entry.colormap) ? :viridis : entry.colormap

_symmetric(entry, colormap) = entry.symmetric || _diverging(colormap)

# A requested range wins, then the registered one, then the statistics of the
# shown frames.
function _resolve_colorrange(entry, stats::FieldStats, requested, colormap)
    _requested(requested) && return _checked_range(requested)
    entry.colorrange isa Tuple && return _checked_range(entry.colorrange)
    return _automatic_colorrange(stats, _symmetric(entry, colormap))
end

function _checked_range(range)
    lower, upper = Float32(range[1]), Float32(range[2])
    isfinite(lower) && isfinite(upper) && lower < upper || throw(
        ArgumentError("a colour range must be finite and increasing, got $range"),
    )
    return (lower, upper)
end

# ---------------------------------------------------------------------------
# Overlays

function _block_segments(topology::TopologySnapshot)
    nx, ny = topology.block_size
    points = Makie.Point2f[]
    sizehint!(points, 8 * length(topology.origins))
    for (origin, spacing) in zip(topology.origins, topology.spacings)
        x0, y0 = origin
        x1, y1 = x0 + nx * spacing[1], y0 + ny * spacing[2]
        append!(
            points, Makie.Point2f[
                (x0, y0), (x1, y0), (x1, y0), (x1, y1),
                (x1, y1), (x0, y1), (x0, y1), (x0, y0),
            ]
        )
    end
    return points
end

function _interface_segments(frame::Frame)
    points = Makie.Point2f[]
    for phase in frame.phases, segment in phase.segments
        push!(points, Makie.Point2f(segment[1]...), Makie.Point2f(segment[2]...))
    end
    return points
end

function _solid_mesh(topology::TopologySnapshot)
    points = Makie.Point2f[]
    faces = Makie.GLTriangleFace[]
    for solid in topology.solids
        offset = length(points)
        append!(points, Makie.Point2f.(solid.polygons.points))
        for triangle in solid.polygons.triangles
            push!(faces, Makie.GLTriangleFace((offset .+ triangle)...))
        end
    end
    return Makie.GeometryBasics.Mesh(points, faces)
end

_has_solids(frame::Frame) =
    any(solid -> !isempty(solid.polygons.triangles), frame.topology.solids)

# ---------------------------------------------------------------------------
# amrfield recipe

"""
    amrfield(source, field; kwargs...)

Draw an AMR field; see `TwoPhaseAMR.amrfield!`.
"""
Makie.@recipe AMRField (frame, field) begin
    "Phase display of phase fields: `:cell` (majority phase) or `:plic` (sharp)."
    phase_rendering = :cell
    "Colormap; `automatic` uses the registered colormap."
    colormap = Makie.automatic
    "Colour range; `automatic` uses the registered range or the shown frame's range."
    colorrange = Makie.automatic
    "Colour of points without a value."
    nan_color = :transparent
    "Draw the PLIC interfaces."
    interface = true
    "Colour of the interface lines."
    interfacecolor = :black
    "Width of the interface lines."
    interfacewidth = 1.5
    "Fill embedded solids."
    geometry = true
    "Fill colour of embedded solids."
    geometrycolor = :gray35
    "Draw leaf-block outlines."
    blocks = false
    "Colour of the block outlines."
    blockcolor = (:gray40, 0.8)
    "Width of the block outlines."
    blockwidth = 0.75
    "Raster samples per screen pixel along each axis."
    oversample = 2
    Makie.mixin_generic_plot_attributes()...
end

Makie.convert_arguments(::Type{<:AMRField}, frame::Frame, field::AbstractString) =
    (frame, Symbol(field))
Makie.convert_arguments(::Type{<:AMRField}, registry::VisualizationFieldRegistry, field) =
    (_frame(registry), Symbol(field))

Makie.preferred_axis_type(::AMRField) = Makie.Axis
Makie.preferred_axis_attributes(::Type{Makie.Axis}, ::AMRField) = (
    aspect = nothing, autolimitaspect = 1,
    xautolimitmargin = (0.0, 0.0), yautolimitmargin = (0.0, 0.0),
)

# Rasterize the visible part of the domain at the panel's resolution.
function _visible_raster(frame::Frame, name::Symbol, limits, area, mode, oversample)
    domain = _domain_rect(frame)
    (lx0, ly0), (lx1, ly1) = extrema(limits)
    x0, x1 = max(Float64(lx0), domain[1]), min(Float64(lx1), domain[2])
    y0, y1 = max(Float64(ly0), domain[3]), min(Float64(ly1), domain[4])
    width, height = Makie.widths(area)
    density = Float64(oversample)
    if !(x1 > x0 && y1 > y0 && width > 0 && height > 0 && density > 0)
        # Nothing of the domain is visible: one transparent texel.
        return (domain[1], domain[2]), (domain[3], domain[4]), fill(NaN32, 1, 1)
    end
    w = max(1, round(Int, (x1 - x0) / (lx1 - lx0) * width * density))
    h = max(1, round(Int, (y1 - y0) / (ly1 - ly0) * height * density))
    image = rasterize(frame, name, (x0, x1, y0, y1), (w, h); phase_rendering = mode)
    return (x0, x1), (y0, y1), image
end

function Makie.plot!(plot::AMRField)
    # Fail before the compute graph would wrap the error.
    TwoPhaseAMR._frame_entry(plot.frame[], Symbol(plot.field[]))
    scene = Makie.parent_scene(plot)
    Makie.add_axis_limits!(plot)
    Makie.add_input!(plot.attributes, :pixel_area, scene.viewport)
    Makie.map!(plot.attributes, [:frame, :field], :entry) do frame, field
        TwoPhaseAMR._frame_entry(frame, Symbol(field))
    end
    Makie.map!(_resolve_colormap, plot.attributes, [:entry, :colormap], :field_colormap)
    Makie.map!(
        plot.attributes, [:frame, :entry, :colorrange, :field_colormap],
        :field_colorrange,
    ) do frame, entry, requested, colormap
        _resolve_colorrange(entry, frame.stats[entry.name], requested, colormap)
    end
    Makie.map!(
        plot.attributes,
        [:frame, :entry, :axis_limits, :pixel_area, :phase_rendering, :oversample],
        [:raster_x, :raster_y, :raster],
    ) do frame, entry, limits, area, mode, oversample
        _visible_raster(frame, entry.name, limits, area, mode, oversample)
    end
    Makie.image!(
        plot, plot.raster_x, plot.raster_y, plot.raster;
        colormap = plot.field_colormap, colorrange = plot.field_colorrange,
        nan_color = plot.nan_color, interpolate = false, visible = plot.visible,
        inspectable = plot.inspectable,
    )

    if _has_solids(plot.frame[])
        Makie.map!(frame -> _solid_mesh(frame.topology), plot.attributes, :frame, :solid_mesh)
        Makie.map!(&, plot.attributes, [:visible, :geometry], :solid_visible)
        Makie.mesh!(
            plot, plot.solid_mesh; color = plot.geometrycolor,
            shading = Makie.NoShading, visible = plot.solid_visible,
            inspectable = false,
        )
    end
    Makie.map!(frame -> _block_segments(frame.topology), plot.attributes, :frame, :block_segments)
    Makie.map!(&, plot.attributes, [:visible, :blocks], :blocks_visible)
    Makie.linesegments!(
        plot, plot.block_segments; color = plot.blockcolor,
        linewidth = plot.blockwidth, visible = plot.blocks_visible, inspectable = false,
    )
    Makie.map!(_interface_segments, plot.attributes, :frame, :interface_segments)
    Makie.map!(&, plot.attributes, [:visible, :interface], :interface_visible)
    Makie.linesegments!(
        plot, plot.interface_segments; color = plot.interfacecolor,
        linewidth = plot.interfacewidth, visible = plot.interface_visible,
        inspectable = false,
    )
    return plot
end

function Makie.data_limits(plot::AMRField)
    xmin, xmax, ymin, ymax = _domain_rect(plot.frame[])
    return Makie.Rect3d(
        Makie.Point3d(xmin, ymin, 0), Makie.Vec3d(xmax - xmin, ymax - ymin, 0),
    )
end
Makie.boundingbox(plot::AMRField, space::Symbol = :data) =
    Makie.apply_transform_and_model(plot, Makie.data_limits(plot))

Makie.extract_colormap(plot::AMRField) = Makie.extract_colormap(first(plot.plots))

# ---------------------------------------------------------------------------
# fieldaxis

const _GridPosition = Union{Makie.GridPosition, Makie.GridSubposition}
const _DECORATIONS = (:minimal, :full, :none)

function _checked_rect(rect)
    xmin, xmax, ymin, ymax = Float64.(Tuple(rect))
    all(isfinite, (xmin, xmax, ymin, ymax)) && xmax > xmin && ymax > ymin || throw(
        ArgumentError("a camera rectangle must be finite with positive extent, got $rect"),
    )
    return (xmin, xmax, ymin, ymax)
end

_camera_rect(rect::NTuple{4, Real}) = _checked_rect(rect)
_camera_rect(domain::Union{Frame, TopologySnapshot, BlockTreeMesh}) = _domain_rect(domain)

_apply_camera!(axis, camera::Symbol) = camera === :fit ? axis : throw(
        ArgumentError("camera must be :fit, a ViewCamera, a rectangle, or a domain"),
    )
function _apply_camera!(axis, camera::Union{NTuple{4, Real}, Frame, TopologySnapshot, BlockTreeMesh})
    Makie.limits!(axis, _camera_rect(camera)...)
    return axis
end
function _apply_camera!(axis, camera::ViewCamera)
    Makie.on(axis.blockscene, axis.scene.viewport; update = true) do area
        size = Makie.widths(area)
        all(>(0), size) && Makie.limits!(axis, visible_rect(camera, size)...)
    end
    return axis
end
_apply_camera!(axis, camera) = throw(
    ArgumentError("camera must be :fit, a ViewCamera, a rectangle, or a domain"),
)

function _nice_length(target)
    exponent = floor(log10(target))
    base = 10.0^exponent
    for factor in (5, 2, 1)
        factor * base <= target && return factor * base
    end
    return base
end

function _format_length(length)
    text = string(round(length; sigdigits = 3))
    return endswith(text, ".0") ? text[1:(end - 2)] : text
end

# Scale bar in the lower-left corner, in pixel space so that it never affects
# the data limits.
function _scalebar!(axis)
    scene = axis.scene
    geometry = Makie.lift(axis.finallimits, scene.viewport) do limits, area
        world = Makie.widths(limits)[1]
        pixels = Makie.widths(area)[1]
        if !(world > 0 && pixels > 0)
            return (Makie.Point2f[(0, 0), (0, 0)], "", Makie.Point2f(0, 0), Makie.Rect2f(0, 0, 0, 0))
        end
        length = _nice_length(0.2 * world)
        bar = Float32(length / world * pixels)
        x0, y0 = 14.0f0, 12.0f0
        return (
            Makie.Point2f[(x0, y0), (x0 + bar, y0)], _format_length(length),
            Makie.Point2f(x0 + bar / 2, y0 + 4), Makie.Rect2f(x0 - 6, y0 - 6, bar + 12, 26),
        )
    end
    # Bar and label take the theme's text colour on a contrasting backdrop.
    ink = Makie.to_color(axis.titlecolor[])
    light = (Makie.Colors.red(ink) + Makie.Colors.green(ink) + Makie.Colors.blue(ink)) / 3 > 0.5
    backdrop = Makie.poly!(
        scene, Makie.lift(g -> g[4], geometry); space = :pixel,
        color = light ? (:black, 0.55) : (:white, 0.75),
        strokewidth = 0, inspectable = false,
    )
    bar = Makie.linesegments!(
        scene, Makie.lift(first, geometry); space = :pixel, color = ink,
        linewidth = 2.5, inspectable = false,
    )
    label = Makie.text!(
        scene, Makie.lift(g -> g[3], geometry); text = Makie.lift(g -> g[2], geometry),
        space = :pixel, align = (:center, :bottom), fontsize = 11, color = ink,
        inspectable = false,
    )
    # Drawn above the field image, which is added later.
    for overlay in (backdrop, bar, label)
        Makie.translate!(overlay, 0, 0, 100)
    end
    return axis
end

function TwoPhaseAMR.fieldaxis(
        position::_GridPosition; camera = :fit, decorations::Symbol = :minimal,
        scalebar::Bool = decorations === :minimal, kwargs...,
    )
    decorations in _DECORATIONS ||
        throw(ArgumentError("decorations must be one of $(_DECORATIONS)"))
    labels = decorations === :full ? (; xlabel = "x", ylabel = "y") : (;)
    axis = Makie.Axis(
        position; aspect = nothing, autolimitaspect = 1,
        xautolimitmargin = (0.0, 0.0), yautolimitmargin = (0.0, 0.0),
        xgridvisible = false, ygridvisible = false, labels..., kwargs...,
    )
    if decorations !== :full
        Makie.hidedecorations!(axis)
        decorations === :none && Makie.hidespines!(axis)
    end
    _apply_camera!(axis, camera)
    scalebar && _scalebar!(axis)
    return axis
end

# ---------------------------------------------------------------------------
# diagnosticsgrid!

struct _DiagnosticSeries
    name::Symbol
    info::DiagnosticInfo
    times::Vector{Float64}
    components::Vector{String}
    values::Matrix{Float64}
end

function _diagnostic_series(store::SnapshotStore)
    return map(diagnostic_names(store)) do name
        history = diagnostic_history(store, name)
        _DiagnosticSeries(
            name, history.info, history.times, history.components, history.values,
        )
    end
end
_diagnostic_series(recorder::Recorder) = _diagnostic_series(snapshot_store(recorder))
_diagnostic_series(flow::FlowSimulation) = _diagnostic_series(flow.diagnostics)

# Split vector samples into components like the store's diagnostic mirror;
# samples whose component count differs from the first sample are skipped.
function _diagnostic_series(collection::DiagnosticCollection)
    return map(collection.series) do series
        name = TwoPhaseAMR.diagnostic_name(series.diagnostic)
        info = diagnostic_info(series.diagnostic)
        counts = [TwoPhaseAMR._value_components(value) for value in series.values]
        first_sample = findfirst(>(0), counts)
        components = isnothing(first_sample) ? 0 : counts[first_sample]
        samples = [k for k in eachindex(counts) if components > 0 && counts[k] == components]
        values = Matrix{Float64}(undef, length(samples), components)
        for (row, sample) in pairs(samples), component in 1:components
            values[row, component] =
                TwoPhaseAMR._component_value(series.values[sample], component)
        end
        _DiagnosticSeries(
            name, info, Float64[series.times[k] for k in samples],
            TwoPhaseAMR._component_names(info, components), values,
        )
    end
end

_name_list(names::Symbol) = names === :all ? nothing : [names]
_name_list(names) = Symbol[Symbol(name) for name in names]

function _reference_pairs(references)
    pairs = Pair{Symbol, Any}[]
    for (name, reference) in references
        reference isa Union{Symbol, Function} || throw(
            ArgumentError(
                "a reference must be a function of time or a diagnostic name",
            )
        )
        push!(pairs, Symbol(name) => reference)
    end
    return pairs
end

# Scalar diagnostics have one value column and no component names. A
# diagnostic without samples yet takes its column count from its metadata.
_columns(series::_DiagnosticSeries) = size(series.values, 2) > 0 ?
    size(series.values, 2) : max(length(series.info.components), 1)

_component_name(series, column) = column <= length(series.components) ?
    series.components[column] : column <= length(series.info.components) ?
    series.info.components[column] : string(column)

_series_label(series::_DiagnosticSeries, column) = _columns(series) > 1 ?
    "$(series.info.label) ($(_component_name(series, column)))" : series.info.label

function _transformed(values::AbstractVector, transform::Symbol)
    transform === :raw && return collect(Float64, values)
    index = findfirst(isfinite, values)
    isnothing(index) && return collect(Float64, values)
    initial = values[index]
    scale = iszero(initial) ? 1.0 : abs(initial)
    return [(value - initial) / scale for value in values]
end

_log_safe(values, scale::Symbol) =
    scale === :log ? [value > 0 ? value : NaN for value in values] : values

_last_time(::Nothing) = Inf
_last_time(until::Real) = Float64(until)

function _points(times, values, transform, scale)
    shown = _log_safe(_transformed(values, transform), scale)
    return Makie.Point2f[(time, value) for (time, value) in zip(times, shown)]
end

# Samples of one column up to the time `until`.
function _series_points(series, column, transform, scale, until)
    size(series.values, 2) >= column || return Makie.Point2f[]
    shown = searchsortedlast(series.times, _last_time(until))
    return _points(
        view(series.times, 1:shown), view(series.values, 1:shown, column), transform, scale,
    )
end

function _reference_points(reference::Function, series, column, transform, scale, until)
    isempty(series.times) && return Makie.Point2f[]
    final = min(last(series.times), _last_time(until))
    final >= first(series.times) || return Makie.Point2f[]
    times = collect(range(first(series.times), final; length = max(256, length(series.times))))
    values = map(times) do time
        value = reference(time)
        value isa Real && _columns(series) == 1 ? Float64(value) : Float64(value[column])
    end
    return _points(times, values, transform, scale)
end

# References are dashed: black for a scalar diagnostic, else in the colour of
# the matching component.
_reference_color(columns, column) = columns == 1 ? :black : Makie.Cycled(column)

# `data` holds the series by name and the last time to show.
function _plot_series!(axis, data, name, group_scale, transform, hidden)
    series = data[][1][name]
    for column in 1:_columns(series)
        (name, column) in hidden && continue
        points = Makie.lift(
            d -> _series_points(d[1][name], column, transform, group_scale, d[2]), data,
        )
        Makie.lines!(axis, points; label = _series_label(series, column))
    end
    return nothing
end

function _plot_reference!(axis, data, name, reference::Symbol, group_scale, transform, hidden)
    target = data[][1][reference]
    for column in 1:_columns(target)
        (name, column) in hidden && continue
        points = Makie.lift(
            d -> _series_points(d[1][reference], column, transform, group_scale, d[2]), data,
        )
        Makie.lines!(
            axis, points; linestyle = :dash,
            color = _reference_color(_columns(target), column),
            label = _series_label(target, column),
        )
    end
    return nothing
end

function _plot_reference!(axis, data, name, reference::Function, group_scale, transform, hidden)
    series = data[][1][name]
    columns = _columns(series)
    for column in 1:columns
        (name, column) in hidden && continue
        points = Makie.lift(
            d -> _reference_points(reference, d[1][name], column, transform, group_scale, d[2]),
            data,
        )
        label = columns > 1 ?
            "$(series.info.label) ($(_component_name(series, column)), reference)" :
            "$(series.info.label) (reference)"
        Makie.lines!(
            axis, points; linestyle = :dash, color = _reference_color(columns, column), label,
        )
    end
    return nothing
end

_cursor_values(cursor::Real) = [Float64(cursor)]
_cursor_values(cursor::AbstractVector{<:Real}) = Float64.(cursor)

function _plot_cursor!(axis, cursor)
    values = cursor isa Makie.Observable ? Makie.lift(_cursor_values, cursor) :
        _cursor_values(cursor)
    Makie.vlines!(axis, values; color = (:black, 0.55), linewidth = 1.5, inspectable = false)
    return nothing
end

function _diagnostic_axis_groups(selected, group::Symbol)
    infos = Pair{Symbol, DiagnosticInfo}[series.name => series.info for series in selected]
    group === :dimension && return diagnostic_groups(infos)
    group === :name && return [
        DiagnosticGroup(info.label, info.kind, info.dimension, info.scale, info.units, [name])
            for (name, info) in infos
    ]
    throw(ArgumentError("group must be :dimension or :name"))
end

const _DIMENSIONLESS = PhysicalDimension(0, 0, 0)

function _quantity_suffix(info::DiagnosticInfo)
    isempty(info.units) || return " [$(info.units)]"
    dimension = info.dimension
    return isnothing(dimension) || dimension == _DIMENSIONLESS ? "" : " ($(dimension))"
end

# An axis with one diagnostic is titled by its label and dimension (it has no
# legend); shared axes by the quantity, naming the counted items.
function _axis_title(group::DiagnosticGroup, lookup)
    infos = [lookup[name].info for name in group.names]
    length(infos) == 1 && return only(infos).label * _quantity_suffix(only(infos))
    group.kind === :count &&
        return "Count (" * join((lowercase(info.label) for info in infos), ", ") * ")"
    group.dimension == _DIMENSIONLESS && isempty(group.units) && return "Dimensionless"
    return group.title
end

_series_lookup(source) =
    Dict(series.name => series for series in _diagnostic_series(source))

# Growing copies of a store's mirrored diagnostics for live plots. A refresh
# holds the store lock only to copy the samples appended since the previous
# refresh, so a long history never makes the solver wait.
struct _DiagnosticCache
    store::SnapshotStore
    series::Dict{Symbol, Tuple{DiagnosticInfo, Vector{String}, Vector{Float64}, Vector{Vector{Float64}}}}
end

_DiagnosticCache(store::SnapshotStore) = _DiagnosticCache(store, Dict())

function _series_lookup(cache::_DiagnosticCache)
    store = cache.store
    Base.@lock store.lock begin
        for mirror in store.diagnostics
            entry = get(cache.series, mirror.name, nothing)
            # A replaced mirror (a new run on the same store) is copied anew.
            if isnothing(entry) || length(entry[3]) > length(mirror.times) ||
                    length(entry[4]) != length(mirror.columns)
                entry = (
                    mirror.info, copy(mirror.components), Float64[],
                    [Float64[] for _ in mirror.columns],
                )
                cache.series[mirror.name] = entry
            end
            copied = length(entry[3])
            append!(entry[3], view(mirror.times, (copied + 1):length(mirror.times)))
            for (column, values) in zip(entry[4], mirror.columns)
                append!(column, view(values, (copied + 1):length(mirror.times)))
            end
        end
    end
    return Dict(
        name => _DiagnosticSeries(
            name, info, times, components,
            isempty(columns) ? zeros(length(times), 0) : reduce(hcat, columns),
        ) for (name, (info, components, times, columns)) in cache.series
    )
end

_live_source(source::SnapshotStore) = _DiagnosticCache(source)
_live_source(source::Recorder) = _DiagnosticCache(snapshot_store(source))
_live_source(source) = source

function _diagnostic_selection(source, names, group, references, diagnostic_options; live)
    ordered = _diagnostic_series(source)
    lookup = Dict(series.name => series for series in ordered)
    reference_pairs = _reference_pairs(references)
    referenced = Set(reference for (_, reference) in reference_pairs if reference isa Symbol)
    for name in union(first.(reference_pairs), referenced)
        haskey(lookup, name) || throw(ArgumentError("unknown diagnostic :$name"))
    end
    requested = _name_list(names)
    for name in something(requested, Symbol[])
        haskey(lookup, name) || throw(ArgumentError("unknown diagnostic :$name"))
    end
    selected = isnothing(requested) ? ordered : [lookup[name] for name in requested]
    filter!(series -> !(series.name in referenced), selected)
    groups = _diagnostic_axis_groups(selected, group)
    counts = Dict(series.name => _columns(series) for series in selected)
    configs = TwoPhaseAMR._validated_diagnostic_options(diagnostic_options, groups, counts)
    # Preserve the historical default of skipping unsampled static series.
    # Explicit group options retain empty groups, like a growing live plot.
    if !live && (isnothing(configs) || isempty(configs))
        filter!(series -> size(series.values, 2) > 0, selected)
        groups = _diagnostic_axis_groups(selected, group)
    end
    isempty(selected) && throw(ArgumentError("no diagnostics to plot"))
    return (; lookup, reference_pairs, groups, configs)
end

function TwoPhaseAMR.diagnosticsgrid!(
        position::_GridPosition,
        source::Union{SnapshotStore, Recorder, FlowSimulation, DiagnosticCollection};
        names = :all, group::Symbol = :dimension, transform::Symbol = :raw,
        cursor = nothing, until = nothing, references = Dict{Symbol, Any}(),
        diagnostic_options = nothing, direction::Symbol = :column, legend = :rt, xlabel = "t",
    )
    transform in (:raw, :relative) || throw(ArgumentError("transform must be :raw or :relative"))
    direction in (:column, :row) || throw(ArgumentError("direction must be :column or :row"))
    live = until isa Makie.Observable
    (; lookup, reference_pairs, groups, configs) =
        _diagnostic_selection(source, names, group, references, diagnostic_options; live)
    # A live plot reads its source again whenever `until` changes; a store is
    # read incrementally.
    live_source = live ? _live_source(source) : nothing
    data = live ? Makie.lift(time -> (_series_lookup(live_source), time), until) :
        Makie.Observable((lookup, until))
    layout = Makie.GridLayout(position)
    plot_axes = Makie.Axis[]
    for (k, axis_group) in pairs(groups)
        option_index = isnothing(configs) ? nothing : findfirst(
                option -> length(option.names) == length(axis_group.names) &&
                all(name -> name in axis_group.names, option.names), configs,
            )
        option = isnothing(option_index) ? nothing : configs[option_index]
        scale = isnothing(option) ? axis_group.scale : option.scale
        hidden = isnothing(option) ? Set{Tuple{Symbol, Int}}() :
            Set((item.name, item.component) for item in option.hidden)
        last_axis = direction === :row || k == length(groups)
        axis = Makie.Axis(
            direction === :column ? layout[k, 1] : layout[1, k];
            title = _axis_title(axis_group, lookup), xlabel = last_axis ? xlabel : "",
            yscale = scale === :log ? log10 : identity,
            titlesize = 13, xlabelsize = 13, xticklabelsize = 11, yticklabelsize = 11,
        )
        for name in axis_group.names
            _plot_series!(axis, data, name, scale, transform, hidden)
            for (target, reference) in reference_pairs
                target === name || continue
                _plot_reference!(axis, data, name, reference, scale, transform, hidden)
            end
        end
        isnothing(cursor) || _plot_cursor!(axis, cursor)
        labelled = count(plot -> haskey(plot, :label) && !isnothing(plot.label[]), axis.scene.plots)
        !isnothing(legend) && labelled > 1 && Makie.axislegend(
            axis; position = legend, labelsize = 11, framevisible = false,
            backgroundcolor = (:white, 0.8), padding = (4, 4, 2, 2), rowgap = 0,
        )
        push!(plot_axes, axis)
    end
    if direction === :column && length(plot_axes) > 1
        Makie.linkxaxes!(plot_axes...)
        for axis in plot_axes[1:(end - 1)]
            Makie.hidexdecorations!(axis; ticks = false, grid = false)
        end
    end
    # Registered after the lines, so the limits follow the updated data.
    live && Makie.on(_ -> foreach(Makie.reset_limits!, plot_axes), data)
    return plot_axes
end

# ---------------------------------------------------------------------------
# Export

function _stored_frames(store::SnapshotStore, window, name::Symbol, selection::Symbol)
    selection in (:all, :keyframes) ||
        throw(ArgumentError("frames must be :all or :keyframes, got $selection"))
    infos = selection === :all ? TwoPhaseAMR.frames(store) : keyframes(store)
    selected = Frame[store[info] for info in infos]
    if !isnothing(window)
        first_time, last_time = Float64(window[1]), Float64(window[2])
        first_time <= last_time ||
            throw(ArgumentError("window must be (t0, t1) with t0 <= t1, got $window"))
        filter!(frame -> first_time <= frame.time <= last_time, selected)
    end
    isempty(selected) && throw(
        ArgumentError(
            isnothing(window) ?
                "the store holds no $(selection === :all ? "frames" : "keyframes")" :
                "the store holds no $(selection === :all ? "frames" : "keyframes") in the window $window",
        )
    )
    for frame in selected
        haskey(frame, name) || throw(ArgumentError("frame $(frame.id) has no field :$name"))
    end
    return selected
end

_title_text(title::Makie.Automatic, frame) =
    isfinite(frame.time) ? "t = $(round(frame.time; sigdigits = 4))" : ""
_title_text(title::Nothing, frame) = ""
_title_text(title::AbstractString, frame) = title
_title_text(title::Function, frame) = string(title(frame))

_valid_camera(camera) = camera === :fit ||
    camera isa Union{ViewCamera, NTuple{4, Real}, Frame, TopologySnapshot, BlockTreeMesh}

# Options shared by every exported panel, with their defaults.
function _panel_options(;
        diagnostics = nothing, diagnostic_options = nothing, size = _automatic, camera = :fit, colorrange = nothing,
        colormap = nothing, phase_rendering::Symbol = :cell, interface::Bool = true,
        geometry::Bool = true, blocks::Bool = false, decorations::Symbol = :minimal,
        title = _automatic, group::Symbol = :dimension, references = Dict{Symbol, Any}(),
        transform::Symbol = :raw, arrangement::Symbol = :auto,
        interfacecolor = :black, blockcolor = (:gray40, 0.8),
    )
    _valid_camera(camera) || throw(
        ArgumentError("camera must be :fit, a ViewCamera, a rectangle, or a domain"),
    )
    arrangement in (:auto, :right, :below) ||
        throw(ArgumentError("arrangement must be :auto, :right, or :below"))
    if isnothing(diagnostics) && !isnothing(diagnostic_options)
        TwoPhaseAMR._validated_diagnostic_options(diagnostic_options, DiagnosticGroup[], Dict())
    end
    return (;
        diagnostics, diagnostic_options, size, camera, colorrange, colormap, phase_rendering, interface,
        geometry, blocks, decorations, title, group, references, transform, arrangement, interfacecolor,
        blockcolor,
    )
end

# The colour range of frames known in advance: requested, registered, or
# their merged statistics.
function _frames_colorrange(frames, name, options)
    entry = TwoPhaseAMR._frame_entry(first(frames), name)
    colormap = _resolve_colormap(entry, options.colormap)
    stats = merge((frame.stats[name] for frame in frames)...)
    return _resolve_colorrange(entry, stats, options.colorrange, colormap), colormap
end

# Field panels share one colourbar. Diagnostics are on the right or below,
# automatically choosing below for multiple fields and right for one. A
# rectangular view normally sizes its columns to the view's aspect. A
# fixed-size output with diagnostics to the right instead shares the available
# width, keeping the diagnostics readable for very wide views. `colorrange` is
# fixed or an `Observable`.
function _export_scene(
        store, panels::Vector{<:Makie.Observable{Frame}}, name::Symbol, colorrange,
        colormap, options; cursor, until = nothing,
    )
    first_frame = panels[1][]
    entry = TwoPhaseAMR._frame_entry(first_frame, name)
    camera = options.camera
    # A fitted view is pinned to the domain, so limits never depend on data.
    view = camera === :fit ? _domain_rect(first_frame) :
        camera isa ViewCamera ? nothing : _camera_rect(camera)
    aspect = isnothing(view) ? 1.5 : (view[2] - view[1]) / (view[4] - view[3])
    npanels = length(panels)
    diagnostics = options.diagnostics
    below = !isnothing(diagnostics) &&
        (options.arrangement === :below || (options.arrangement === :auto && npanels > 1))
    automatic_size = options.size isa Makie.Automatic
    height = round(Int, min(npanels > 1 ? 280 : 360, 1000 / aspect))
    figure = Makie.Figure(; size = automatic_size ? (1000, 600) : options.size)
    layout = figure.layout
    plot = nothing
    field_axes = Makie.Axis[]
    for (k, frame) in pairs(panels)
        axis = TwoPhaseAMR.fieldaxis(
            figure[1, k]; camera = isnothing(view) ? camera : view,
            options.decorations, titlesize = 14, titlefont = :regular,
            title = Makie.lift(f -> _title_text(options.title, f), frame),
        )
        plot = amrfield!(
            axis, frame, name; colormap, colorrange, options.phase_rendering,
            options.interface, options.geometry, options.blocks, options.interfacecolor,
            options.blockcolor,
        )
        push!(field_axes, axis)
        Makie.colsize!(
            layout, k, !automatic_size && !isnothing(diagnostics) && !below ? Makie.Auto() :
                !isnothing(view) ? Makie.Aspect(1, aspect) :
                automatic_size ? Makie.Fixed(round(Int, aspect * height)) : Makie.Auto(),
        )
    end
    Makie.Colorbar(figure[1, npanels + 1], plot; label = _field_title(entry))
    diagnostic_axes = Makie.Axis[]
    if !isnothing(diagnostics)
        position = below ? figure[2, 1:(npanels + 1)] : figure[1, npanels + 2]
        append!(
            diagnostic_axes, TwoPhaseAMR.diagnosticsgrid!(
                position, store; names = diagnostics, options.group, options.references,
                options.transform, options.diagnostic_options, cursor, until,
                direction = below ? :row : :column,
            )
        )
        if automatic_size
            below ? Makie.rowsize!(layout, 2, Makie.Fixed(220)) :
                Makie.colsize!(layout, npanels + 2, Makie.Fixed(380))
        end
    end
    if automatic_size
        Makie.rowsize!(layout, 1, Makie.Fixed(height))
        Makie.resize_to_layout!(figure)
    end
    return (; figure, field_axes, plot, diagnostic_axes)
end

# Stored exports default to 20 frames per second and a play time that scales
# with the candidate count, between two and twenty seconds.
const _EXPORT_FPS = 20
const _EXPORT_DURATION_MIN = 2.0
const _EXPORT_DURATION_MAX = 20.0
_default_duration(count::Integer, fps::Real) =
    clamp(count / fps, _EXPORT_DURATION_MIN, _EXPORT_DURATION_MAX)

# Nearest-frame sampling over the stored frames of the window. The output
# times are uniform in simulation time, so stored frames repeat or are skipped.
function _video_schedule(times, window, fps, duration, nframes)
    isnothing(duration) || isnothing(nframes) ||
        throw(ArgumentError("give at most one of duration and nframes"))
    t_start, t_end = isnothing(window) ? (first(times), last(times)) :
        (Float64(window[1]), Float64(window[2]))
    isnothing(duration) && isnothing(nframes) &&
        (duration = _default_duration(length(times), fps))
    return isnothing(nframes) ?
        video_frame_schedule(times, t_start, t_end; fps, duration) :
        video_frame_schedule(times, t_start, t_end; fps, nframes)
end

# Animation of stored frames on a uniformly sampled simulation-time line; the
# figure is built once.
function _animation_scene(
        source, field; window = nothing, frames::Symbol = :keyframes,
        fps::Real = _EXPORT_FPS,
        duration::Union{Nothing, Real} = nothing,
        nframes::Union{Nothing, Integer} = nothing, kwargs...,
    )
    options = _panel_options(; kwargs...)
    store = _store(source)
    name = Symbol(field)
    selected = _stored_frames(store, window, name, frames)
    schedule = _video_schedule(
        [frame.time for frame in selected], window, fps, duration, nframes,
    )
    shown = selected[schedule.indices]
    colorrange, colormap = _frames_colorrange(selected, name, options)
    current = Makie.Observable(first(shown))
    scene = _export_scene(
        store, [current], name, colorrange, colormap, options;
        cursor = Makie.lift(frame -> frame.time, current),
    )
    return (; scene..., frame = current, frames = shown, schedule)
end

"""
Handle of a live recording started by `export_animation(path, recorder;
source = :live)`. `close` it after the run: it stops receiving frames, encodes
the remaining output frames, writes the file, and reports the counts.
`encoded` counts the frames written, `dropped` counts frames lost because the
encoder fell behind, and `ids` and `times` list the ids and simulation times
of the frames written, in output order.
"""
mutable struct LiveRecording
    const path::String
    const subscription::TwoPhaseAMR.FrameSubscription
    task::Union{Nothing, Task}
    const ids::Vector{Int}
    const times::Vector{Float64}
    @atomic encoded::Int
end

function Base.getproperty(recording::LiveRecording, name::Symbol)
    if name === :dropped
        subscription = getfield(recording, :subscription)
        return Base.@lock subscription.condition subscription.dropped
    end
    return getfield(recording, name)
end

Base.propertynames(::LiveRecording, private::Bool = false) =
    (:path, :encoded, :dropped, :ids, :times)

Base.isopen(recording::LiveRecording) = isopen(getfield(recording, :subscription))

function Base.show(io::IO, recording::LiveRecording)
    return print(
        io, "LiveRecording(", repr(recording.path), ", encoded=", recording.encoded,
        ", dropped=", recording.dropped, isopen(recording) ? ", open)" : ", closed)",
    )
end

function Base.close(recording::LiveRecording)
    close(getfield(recording, :subscription))
    wait(getfield(recording, :task)::Task)
    @info "Live recording wrote $(recording.encoded) frames to $(recording.path); " *
        "$(recording.dropped) dropped."
    return nothing
end

# A thread pool other than the caller's, so that encoding never shares the
# solver's thread; `nothing` if there is none.
function _encoder_threadpool()
    current = Threads.threadpool()
    current === :default && Threads.nthreads(:interactive) > 0 && return :interactive
    current === :interactive && Threads.nthreads(:default) > 0 && return :default
    Threads.nthreads(:default) > 1 && return :default
    return nothing
end

_solver_running(recorder) =
    TwoPhaseAMR.run_state(recorder) in (:running, :pause_requested, :paused, :stopping)

const _VIDEO_FORMATS = ("gif", "mp4", "mkv", "webm")

# Last-resort live default when the run and the store give no time span.
const _LIVE_SIM_PER_SECOND = 1.0

# Aim at ten seconds of video for the run's remaining simulation time, or for
# the span of the stored frames when the run's end is unknown.
function _live_default_rate(recorder, store)
    times = TwoPhaseAMR.step_log(store).time
    stored = TwoPhaseAMR.frames(store)
    t_start = isempty(times) ? (isempty(stored) ? nothing : newest_frame(store).time) : last(times)
    isnothing(t_start) && return _LIVE_SIM_PER_SECOND
    driver = Base.@lock recorder.lock recorder.driver
    t_final = driver isa TwoPhaseAMR.StudioDriver ? driver.final_time : nothing
    isnothing(t_final) && (t_final = isempty(stored) ? nothing : last(stored).time)
    isnothing(t_final) && return _LIVE_SIM_PER_SECOND
    span = t_final - t_start
    return span > 0 ? span / 10 : _LIVE_SIM_PER_SECOND
end

# Encode output frames on a uniform simulation-time grid. The arrival of a
# frame decides every output time up to its own time, using that frame and the
# previously arrived one; the nearest wins and ties keep the earlier frame.
# This repeats frames for sparse streams and skips them for dense streams, and
# it needs no buffer beyond the two frames involved.
function _encode_live!(
        recording, recorder, name, options, format, fps, px_per_unit,
        sim_per_second, deferred,
    )
    subscription = getfield(recording, :subscription)
    store = snapshot_store(recorder)
    stream = nothing
    current = colorrange = stats = entry = colormap = nothing
    growing = false
    recorder_frames = TwoPhaseAMR.frames(store)
    seed = isempty(recorder_frames) ? nothing : newest_frame(store)
    # Start at the simulation time at job start, not at the possibly lagging
    # newest stored frame. The seed only anchors the left neighbour.
    step_times = TwoPhaseAMR.step_log(store).time
    t_start = isempty(step_times) ? (isnothing(seed) ? nothing : seed.time) :
        last(step_times)
    sampler = isnothing(t_start) ? nothing :
        LiveVideoSampler{Frame}(t_start, sim_per_second / fps)
    seed_id = isnothing(seed) ? 0 : seed.id

    function show!(frame)
        if isnothing(stream)
            entry = TwoPhaseAMR._frame_entry(frame, name)
            colormap = _resolve_colormap(entry, options.colormap)
            stats = frame.stats[name]
            # The range of the frames recorded so far, which only grows.
            growing = !_requested(options.colorrange) && !(entry.colorrange isa Tuple)
            colorrange = Makie.Observable(
                _resolve_colorrange(entry, stats, options.colorrange, colormap),
            )
            current = Makie.Observable(frame)
            cursor = Makie.lift(f -> f.time, current)
            scene = _export_scene(
                store, [current], name, colorrange, colormap, options;
                cursor, until = cursor,
            )
            stream = Makie.VideoStream(
                scene.figure; format, framerate = fps, px_per_unit, visible = false,
            )
        else
            if growing
                stats = merge(stats, frame.stats[name])
                widened = _automatic_colorrange(stats, _symmetric(entry, colormap))
                widened == colorrange[] || (colorrange[] = widened)
            end
            current[] = frame
        end
        Makie.recordframe!(stream)
        push!(getfield(recording, :ids), frame.id)
        push!(getfield(recording, :times), Float64(frame.time))
        @atomic recording.encoded += 1
        return nothing
    end

    function decide!(frames)
        for frame in frames
            haskey(frame, name) ||
                throw(ArgumentError("frame $(frame.id) has no field :$name"))
            show!(frame)
        end
        return nothing
    end

    try
        isnothing(seed) || decide!(push_live_frame!(sampler, seed, seed.time))
        while (frame = take!(subscription)) !== nothing
            # Without a thread of its own, the encoder waits for the run to end.
            while deferred && _solver_running(recorder) && isopen(subscription)
                sleep(0.05)
            end
            frame.id <= seed_id && continue
            if isnothing(sampler)
                # The store held no frame when the recording started.
                sampler = LiveVideoSampler{Frame}(frame.time, sim_per_second / fps)
            end
            decide!(push_live_frame!(sampler, frame, frame.time))
        end
        if !isnothing(sampler)
            # Hold the last arrived frame through the simulation time at stop,
            # so sparse streams still cover the elapsed time.
            stop_times = TwoPhaseAMR.step_log(store).time
            t_end = isempty(stop_times) ? sampler.left_time : last(stop_times)
            decide!(flush_live_frames!(sampler, t_end))
        end
        isnothing(stream) &&
            throw(ArgumentError("the live recording produced no output frames"))
        Makie.save(getfield(recording, :path), stream)
    catch
        close(subscription)
        rethrow()
    end
    return nothing
end

function _record_live(
        path, recorder::Recorder, name, options;
        fps, sim_per_second, px_per_unit, capacity,
    )
    format = lowercase(lstrip(last(splitext(path)), '.'))
    format in _VIDEO_FORMATS ||
        throw(ArgumentError("a live recording writes one of $(_VIDEO_FORMATS), got $path"))
    subscription = TwoPhaseAMR.subscribe(snapshot_store(recorder); capacity)
    recording = LiveRecording(String(path), subscription, nothing, Int[], Float64[], 0)
    pool = _encoder_threadpool()
    deferred = isnothing(pool)
    deferred && @info "Live recording: no thread besides the solver's is available, so " *
        "frames are queued (capacity $capacity) and encoded after the run. Start Julia " *
        "with an interactive thread (the default since Julia 1.12 unless `-t 1`) to " *
        "encode while the solver runs."
    encode() = _encode_live!(
        recording, recorder, name, options, format, fps, px_per_unit,
        sim_per_second, deferred,
    )
    recording.task = pool === :interactive ? Threads.@spawn(:interactive, encode()) :
        Threads.@spawn(:default, encode())
    return recording
end

function _export_cancelled(cancel)
    isnothing(cancel) && return false
    result = cancel()
    result isa Bool || throw(ArgumentError("cancel must return a Bool"))
    return result
end

# Makie.record does not close its VideoStream when its callback throws. Own
# the stream for cooperative jobs so cancellation and callback failures close
# and reap FFmpeg, release the tick filter, and remove the temporary video.
function _record_stored(animation, path; fps, px_per_unit, progress, cancel)
    _export_cancelled(cancel) && return nothing
    format = lstrip(last(splitext(path)), '.')
    stream = Makie.VideoStream(animation.figure; format, framerate = fps, px_per_unit)
    try
        total = length(animation.frames)
        for (index, frame) in enumerate(animation.frames)
            _export_cancelled(cancel) && return nothing
            animation.frame[] = frame
            Makie.recordframe!(stream)
            isnothing(progress) || progress(index, total)
            yield()
        end
        _export_cancelled(cancel) && return nothing
        # Save to a private sibling first: cancellation or a failed save must
        # neither expose a partial file nor destroy an existing destination.
        return mktempdir(dirname(abspath(path))) do directory
            staged = joinpath(directory, basename(path))
            Makie.save(staged, stream)
            success(stream.process) || throw(ErrorException("FFmpeg failed while encoding the animation"))
            _export_cancelled(cancel) && return nothing
            mv(staged, path; force = true)
            return path
        end
    finally
        try
            try
                isopen(stream.io) && close(stream.process)
            finally
                wait(stream.process)
            end
        finally
            try
                Makie.stop!(stream.tick_controller)
            finally
                rm(dirname(stream.path); recursive = true, force = true)
            end
        end
    end
end

function TwoPhaseAMR.export_animation(
        path::AbstractString, recording::Union{SnapshotStore, Recorder};
        field, source::Symbol = :stored, window = nothing, frames::Symbol = :keyframes,
        fps::Real = _EXPORT_FPS,
        duration::Union{Nothing, Real} = nothing,
        nframes::Union{Nothing, Integer} = nothing, sim_per_second::Union{Nothing, Real} = nothing,
        px_per_unit::Real = 1, capacity = nothing,
        progress = nothing, cancel = nothing, kwargs...,
    )
    TwoPhaseAMR._positive_finite(fps, "fps")
    if source === :stored
        isnothing(capacity) ||
            throw(ArgumentError("`capacity` applies to source = :live"))
        isnothing(sim_per_second) ||
            throw(ArgumentError("`sim_per_second` applies to source = :live"))
        isnothing(progress) || applicable(progress, 1, 1) ||
            throw(ArgumentError("progress must be callable as progress(i, n)"))
        isnothing(cancel) || applicable(cancel) ||
            throw(ArgumentError("cancel must be a zero-argument predicate"))
        animation = _animation_scene(
            recording, field; window, frames, fps, duration, nframes, kwargs...,
        )
        return _record_stored(animation, path; fps, px_per_unit, progress, cancel)
    elseif source === :live
        recording isa Recorder ||
            throw(ArgumentError("source = :live records the frames of a running Recorder"))
        isnothing(window) || throw(ArgumentError("`window` applies to source = :stored"))
        frames === :keyframes || throw(ArgumentError("`frames` applies to source = :stored"))
        isnothing(duration) && isnothing(nframes) ||
            throw(ArgumentError("`duration` and `nframes` apply to source = :stored"))
        isnothing(progress) && isnothing(cancel) ||
            throw(ArgumentError("progress and cancel apply only to source = :stored"))
        rate = something(sim_per_second, _live_default_rate(recording, snapshot_store(recording)))
        TwoPhaseAMR._positive_finite(rate, "sim_per_second")
        options = _panel_options(; kwargs...)
        if !isnothing(options.diagnostics) && !isnothing(options.diagnostic_options)
            isnothing(recording.diagnostics) &&
                throw(ArgumentError("the recorder has no diagnostics"))
            store = snapshot_store(recording)
            diagnostic_source = isempty(diagnostic_names(store)) ? recording.diagnostics : store
            selection = _diagnostic_selection(
                diagnostic_source, options.diagnostics, options.group,
                options.references, options.diagnostic_options; live = true,
            )
            options = merge(options, (; diagnostic_options = selection.configs))
        end
        return _record_live(
            path, recording, Symbol(field), options; fps,
            sim_per_second = rate, px_per_unit, capacity = something(capacity, 32),
        )
    end
    throw(ArgumentError("source must be :stored or :live"))
end

_figure_frames(store::SnapshotStore, ::Nothing) = Frame[newest_frame(store)]
_figure_frames(store::SnapshotStore, at::Real) = Frame[frame_at(store, at)]
_figure_frames(store::SnapshotStore, at::AbstractVector{<:Real}) =
    Frame[frame_at(store, time) for time in at]

function TwoPhaseAMR.export_figure(
        path::AbstractString,
        source::Union{SnapshotStore, Recorder, Frame, VisualizationFieldRegistry};
        field, at = nothing, kwargs...,
    )
    options = _panel_options(; kwargs...)
    name = Symbol(field)
    store = source isa Union{SnapshotStore, Recorder} ? _store(source) : nothing
    shown = if isnothing(store)
        isnothing(at) || throw(ArgumentError("`at` requires a SnapshotStore or Recorder"))
        isnothing(options.diagnostics) ||
            throw(ArgumentError("diagnostics require a SnapshotStore or Recorder"))
        Frame[_frame(source)]
    else
        isempty(TwoPhaseAMR.frames(store)) && throw(ArgumentError("the store holds no frames"))
        _figure_frames(store, at)
    end
    for frame in shown
        haskey(frame, name) || throw(ArgumentError("the frame has no field :$name"))
    end
    colorrange, colormap = _frames_colorrange(shown, name, options)
    scene = _export_scene(
        store, [Makie.Observable(frame) for frame in shown], name, colorrange, colormap,
        options; cursor = [frame.time for frame in shown],
    )
    Makie.save(path, scene.figure)
    return path
end

end
