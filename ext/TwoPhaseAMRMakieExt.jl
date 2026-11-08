module TwoPhaseAMRMakieExt

using TwoPhaseAMR
import Makie

const SnapshotSource = Union{VisualizationFieldRegistry, Frame}

_snapshot_frame(frame::Frame) = frame
_snapshot_frame(registry::VisualizationFieldRegistry) =
    TwoPhaseAMR._registry_frame(registry)

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

# A requested range wins, then the registered one, then the field statistics.
function _colorrange(entry, stats, requested)
    isnothing(requested) || return Float32.(requested)
    entry.colorrange isa Tuple && return Float32.(entry.colorrange)
    return _automatic_colorrange(stats, entry.symmetric)
end

_domain_rect(topology) = (
    Float64(topology.origin[1]), Float64(topology.origin[1] + topology.extent[1]),
    Float64(topology.origin[2]), Float64(topology.origin[2] + topology.extent[2]),
)

function _raster_size(rect, resolution)
    width, height = rect[2] - rect[1], rect[4] - rect[3]
    scale = resolution / max(width, height)
    return (max(1, round(Int, width * scale)), max(1, round(Int, height * scale)))
end

function _block_segments(topology)
    nx, ny = topology.block_size
    points = Makie.Point2f[]
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

function _solid_mesh(topology)
    points = Makie.Point2f[]
    faces = Makie.GLTriangleFace[]
    for solid in topology.solids
        offset = length(points)
        append!(points, Makie.Point2f.(solid.polygons.points))
        append!(
            faces, Makie.GLTriangleFace[
                Makie.GLTriangleFace((offset .+ triangle)...)
                    for triangle in solid.polygons.triangles
            ]
        )
    end
    return Makie.GeometryBasics.Mesh(points, faces)
end

# Draw one field of an observable frame. The figure is built once; updating
# the observable re-rasterizes the field and its overlays.
function _snapshot_panel!(
        position, frame::Makie.Observable{Frame}; field, phase_rendering = :cell,
        blocks = true, interface = true, geometry = true, colorbar = true,
        colormap = :viridis, colorrange = nothing, resolution = 1024,
        title = nothing,
    )
    name = Symbol(field)
    current = frame[]
    entry = TwoPhaseAMR._frame_entry(current, name)
    rect = _domain_rect(current.topology)
    raster_size = _raster_size(rect, resolution)
    image = Makie.lift(frame) do snapshot
        rasterize(snapshot, name, rect, raster_size; phase_rendering)
    end
    axis = Makie.Axis(
        position[1, 1]; aspect = Makie.DataAspect(),
        title = isnothing(title) ? _field_title(entry) : title,
    )
    plot = Makie.image!(
        axis, (rect[1], rect[2]), (rect[3], rect[4]), image; colormap,
        colorrange = _colorrange(entry, current.stats[name], colorrange),
        nan_color = :transparent, interpolate = false,
    )
    if geometry && any(solid -> !isempty(solid.polygons.triangles), current.topology.solids)
        Makie.mesh!(
            axis, Makie.lift(snapshot -> _solid_mesh(snapshot.topology), frame);
            color = :black, shading = Makie.NoShading,
        )
    end
    if blocks
        Makie.linesegments!(
            axis, Makie.lift(snapshot -> _block_segments(snapshot.topology), frame);
            color = :gray, linewidth = 0.75,
        )
    end
    if interface && !isempty(current.phases)
        Makie.linesegments!(
            axis, Makie.lift(_interface_segments, frame); color = :black, linewidth = 2,
        )
    end
    Makie.limits!(axis, rect...)
    bar = colorbar ? Makie.Colorbar(position[1, 2], plot) : nothing
    return axis, bar
end

function TwoPhaseAMR.snapshot_figure(source::SnapshotSource; size = (600, 500), kwargs...)
    figure = Makie.Figure(; size)
    _snapshot_panel!(figure[1, 1], Makie.Observable(_snapshot_frame(source)); kwargs...)
    return figure
end

function TwoPhaseAMR.snapshot_figure!(position, source::SnapshotSource; kwargs...)
    axis, _ = _snapshot_panel!(
        position, Makie.Observable(_snapshot_frame(source)); kwargs...,
    )
    return axis
end

function TwoPhaseAMR.plot_diagnostics!(
        axis::Makie.Axis, flow::FlowSimulation, names;
        analytical = nothing, labels = names,
    )
    series = names isa Symbol ? (names,) : names
    series_labels = labels isa Union{Symbol, AbstractString} ? (labels,) : labels
    length(series) == length(series_labels) ||
        throw(ArgumentError("labels must match the number of diagnostic names"))
    for (name, label) in zip(series, series_labels)
        history = diagnostic_history(flow, Symbol(name))
        Makie.lines!(axis, history.times, history.values; label = string(label))
        if !isnothing(analytical)
            Makie.lines!(axis, history.times, analytical.(history.times);
                linestyle = :dash, label = "$(label) analytical")
        end
    end
    return axis
end

_record_store(recorder::Recorder) = snapshot_store(recorder)
_record_store(store::SnapshotStore) = store

function TwoPhaseAMR.record_checkpoints(
        path::AbstractString, source::Union{Recorder, SnapshotStore};
        field, framerate = 10, size = (600, 500), colorrange = nothing, kwargs...,
    )
    store = _record_store(source)
    name = Symbol(field)
    selected = Frame[store[info] for info in keyframes(store)]
    filter!(frame -> haskey(frame, name), selected)
    isempty(selected) && throw(ArgumentError("the store holds no keyframes of :$name"))
    range = isnothing(colorrange) ? _automatic_colorrange(
            merge((frame.stats[name] for frame in selected)...),
            TwoPhaseAMR._frame_entry(first(selected), name).symmetric,
        ) : colorrange
    entry = TwoPhaseAMR._frame_entry(first(selected), name)
    entry.colorrange isa Tuple && isnothing(colorrange) &&
        (range = Float32.(entry.colorrange))
    figure = Makie.Figure(; size)
    frame = Makie.Observable(first(selected))
    _snapshot_panel!(figure[1, 1], frame; field = name, colorrange = range, kwargs...)
    Makie.record(figure, path, selected; framerate) do snapshot
        frame[] = snapshot
    end
    return path
end

end
