const _HISTOGRAM_BINS = 64

"""
    FieldStats

Statistics of one field in one frame, or merged over frames: `minimum`,
`maximum`, `absmax`, `mean` of the finite values, their number `finite`, and a
64-bin `histogram` over `[minimum, maximum]`. Phase fields include both phase
values. Colour policies use these instead of the field data.
"""
struct FieldStats
    minimum::Float64
    maximum::Float64
    absmax::Float64
    mean::Float64
    finite::Int
    histogram::Vector{Int}
end

function _accumulate_extrema(values, state)
    lower, upper, total, count = state
    for value in values
        isfinite(value) || continue
        v = Float64(value)
        lower = min(lower, v)
        upper = max(upper, v)
        total += v
        count += 1
    end
    return lower, upper, total, count
end

function _accumulate_histogram!(histogram, values, lower, upper)
    width = upper - lower
    bins = length(histogram)
    for value in values
        isfinite(value) || continue
        bin = width > 0 ? floor(Int, (Float64(value) - lower) / width * bins) + 1 : 1
        @inbounds histogram[clamp(bin, 1, bins)] += 1
    end
    return histogram
end

_stat_arrays(values::AbstractVector) = (values,)
_stat_arrays(values::TwoPhaseVisualizationValues) = (values.liquid, values.gas)

function _field_stats(values)
    arrays = _stat_arrays(values)
    state = (Inf, -Inf, 0.0, 0)
    for array in arrays
        state = _accumulate_extrema(array, state)
    end
    lower, upper, total, count = state
    histogram = zeros(Int, _HISTOGRAM_BINS)
    count == 0 && return FieldStats(NaN, NaN, NaN, NaN, 0, histogram)
    for array in arrays
        _accumulate_histogram!(histogram, array, lower, upper)
    end
    return FieldStats(
        lower, upper, max(abs(lower), abs(upper)), total / count, count, histogram,
    )
end

"""
    merge(stats::FieldStats...)

Combine per-frame statistics. Extrema, mean, and count are exact; histogram
bins are redistributed by their centres onto the merged range.
"""
function Base.merge(first::FieldStats, rest::FieldStats...)
    all_stats = (first, rest...)
    finite = filter(stats -> stats.finite > 0, collect(all_stats))
    histogram = zeros(Int, _HISTOGRAM_BINS)
    isempty(finite) && return FieldStats(NaN, NaN, NaN, NaN, 0, histogram)
    lower = minimum(stats -> stats.minimum, finite)
    upper = maximum(stats -> stats.maximum, finite)
    count = sum(stats -> stats.finite, finite)
    mean = sum(stats -> stats.mean * stats.finite, finite) / count
    width = upper - lower
    for stats in finite
        bins = length(stats.histogram)
        step = (stats.maximum - stats.minimum) / bins
        for (bin, number) in pairs(stats.histogram)
            iszero(number) && continue
            centre = stats.minimum + (bin - 0.5) * step
            target = width > 0 ? floor(Int, (centre - lower) / width * _HISTOGRAM_BINS) + 1 : 1
            histogram[clamp(target, 1, _HISTOGRAM_BINS)] += number
        end
    end
    return FieldStats(lower, upper, max(abs(lower), abs(upper)), mean, count, histogram)
end

"""
    FrameInfo

Metadata of a stored frame: `id`, simulation `time`, `step`, `kind`
(`:initial`, `:keyframe`, `:preview`, `:manual`, `:pause`, or `:final`),
recorder-clock `wall_time` at capture, `topology` id, and the scheduled
keyframe times the frame satisfies (`requested_times`).
"""
struct FrameInfo
    id::Int
    time::Float64
    step::Int
    kind::Symbol
    wall_time::Float64
    topology::Int
    requested_times::Vector{Float64}
end

# Never-dropped kinds outrank previews; initial and final outrank the rest.
_kind_rank(kind::Symbol) = kind === :preview ? 0 : kind === :keyframe ? 1 :
    kind === :manual ? 2 : kind === :pause ? 3 : 4

"""Manifest entry of one captured field."""
struct FieldEntry
    name::Symbol
    label::String
    units::String
    colormap::Any
    colorrange::Any
    placement::AbstractPlacement
    # Index of the interface (phase snapshot) separating liquid and gas
    # values, or 0 for single-valued fields.
    interface::Int
    slot::Int
    symmetric::Bool
end

"""Names, placements, and display metadata of the fields captured in a frame."""
struct FieldManifest
    entries::Vector{FieldEntry}
    index::Dict{Symbol, Int}
end

field_names(manifest::FieldManifest) = [entry.name for entry in manifest.entries]

"""
Interface state of one frame: the majority phase of every cell (conditional
liquid fraction of at least 1/2), the PLIC planes `(n_x, n_y, d)` of mixed
cells in local cell coordinates (liquid where `n⋅ξ <= d`), the physical PLIC
segments, and the index of the solid it is conditional on (0 = none).
"""
struct PhaseSnapshot{F}
    liquid::BitVector
    mixed::Vector{Int32}
    planes::Vector{NTuple{3, F}}
    segments::Vector{NTuple{2, NTuple{2, F}}}
    solid::Int
end

"""
    Frame

Owned snapshot of the registered fields at one completed simulation state.
`frame[name]` returns the captured values: a vector in block-major cell order
or in per-axis face order (interior faces, then boundary faces), or a
`TwoPhaseVisualizationValues` with `liquid` and `gas` vectors for phase fields.
Arrays never alias solver storage and remain valid after mesh adaptation; do
not modify them. `frame.time`, `frame.step`, `frame.kind`, and `frame.id`
forward to the frame's [`FrameInfo`](@ref); `frame.topology` is its
[`TopologySnapshot`](@ref).
"""
struct Frame
    info::FrameInfo
    topology::TopologySnapshot
    manifest::FieldManifest
    values::Vector{Any}
    phases::Vector{PhaseSnapshot}
    stats::Dict{Symbol, FieldStats}
    bytes::Int
end

const _FRAME_INFO_PROPERTIES = (:id, :time, :step, :kind, :wall_time, :requested_times)

function Base.getproperty(frame::Frame, name::Symbol)
    name in _FRAME_INFO_PROPERTIES && return getfield(getfield(frame, :info), name)
    return getfield(frame, name)
end

Base.propertynames(frame::Frame, private::Bool = false) =
    (fieldnames(Frame)..., _FRAME_INFO_PROPERTIES...)

Base.haskey(frame::Frame, name::Symbol) = haskey(frame.manifest.index, name)

function _frame_entry(frame::Frame, name::Symbol)
    index = get(frame.manifest.index, name, 0)
    iszero(index) && throw(ArgumentError("the frame has no field :$name"))
    return frame.manifest.entries[index]
end

Base.getindex(frame::Frame, name::Symbol) = frame.values[_frame_entry(frame, name).slot]

field_names(frame::Frame) = field_names(frame.manifest)

locate(frame::Frame, point) = locate(frame.topology, point)
locate_face(frame::Frame, point, axis::Integer) = locate_face(frame.topology, point, axis)

function Base.show(io::IO, frame::Frame)
    return print(
        io, "Frame(id=", frame.id, ", kind=:", frame.kind, ", time=", frame.time,
        ", step=", frame.step, ", fields=", length(frame.manifest.entries), ")",
    )
end

_value_bytes(values::AbstractVector) = sizeof(values)
_value_bytes(values::TwoPhaseVisualizationValues) =
    sizeof(values.liquid) + sizeof(values.gas)

_phase_bytes(phase::PhaseSnapshot) = sizeof(phase.liquid.chunks) +
    sizeof(phase.mixed) + sizeof(phase.planes) + sizeof(phase.segments)
