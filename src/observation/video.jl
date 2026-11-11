"""
    VideoSchedule

Output frames of an exported video and the stored frame shown at each one.

`times[k]` is the simulation time at which output frame `k` is played, on a
uniform grid from the window start to its end. `indices[k]` is the 1-based
index into the frame-time vector passed to [`video_frame_schedule`](@ref) of
the stored frame nearest to `times[k]`; ties go to the earlier frame. Stored
frames may therefore be repeated (sparse recording) or skipped (dense
recording).
"""
struct VideoSchedule
    indices::Vector{Int}
    times::Vector{Float64}
end

Base.length(schedule::VideoSchedule) = length(schedule.indices)

function Base.show(io::IO, schedule::VideoSchedule)
    window = isempty(schedule.times) ? "empty" :
        "$(first(schedule.times)) .. $(last(schedule.times))"
    return print(
        io, "VideoSchedule(", length(schedule), " output frames, ",
        length(unique(schedule.indices)), " stored frames, sim time $window)",
    )
end

"""
    video_frame_count(fps, duration)

Number of output frames of a video that plays `duration` seconds at `fps`
frames per second, including both the first and the last frame. With
`n` output frames the spacing between displayed simulation times is
`(t_end - t_start) / (n - 1)`, so the play time between the first and last
frame is exactly `duration`.
"""
function video_frame_count(fps::Real, duration::Real)
    _positive_finite(fps, "fps")
    _positive_finite(duration, "duration")
    frames = Float64(fps) * Float64(duration)
    frames <= typemax(Int) - 1 || throw(
        ArgumentError("fps × duration is too large for a frame count, got $frames"),
    )
    return max(2, round(Int, frames) + 1)
end

function _positive_finite(value::Real, name::String)
    isfinite(value) && value > 0 ||
        throw(ArgumentError("$name must be finite and positive, got $value"))
    return Float64(value)
end

# Index of the entry nearest to `target`; ties go to the earlier entry.
function _nearest_index(times::AbstractVector{<:Real}, target::Float64)
    index = searchsortedfirst(times, target)
    index > length(times) && return length(times)
    index == 1 && return 1
    before, after = times[index - 1], times[index]
    return target - before <= after - target ? index - 1 : index
end

"""
    video_frame_schedule(times, t_start, t_end, count)
    video_frame_schedule(times, t_start, t_end; fps, duration=nothing, nframes=nothing)

Map a uniformly sampled video time line onto stored frames. Return a
[`VideoSchedule`](@ref) of `count` output frames whose target times are
uniformly spaced over `[t_start, t_end]` (both end points included), each
showing the frame in `times` nearest to its target time; ties go to the
earlier frame.

`times` are the increasing simulation times of the stored frames. It may
contain repeated times. The keyword form derives `count` from
`fps` and `duration` (see [`video_frame_count`](@ref)) or takes it directly
as `nframes`; give exactly one of the two. The mapping is independent of how
the stored frames are spaced: output times are linear in simulation time, and
stored frames are repeated when the recording is sparse or skipped when it
is dense.
"""
function video_frame_schedule(
        times::AbstractVector{<:Real}, t_start::Real, t_end::Real, count::Integer,
    )
    isempty(times) && throw(ArgumentError("the recording holds no frames"))
    isfinite(t_start) && isfinite(t_end) ||
        throw(ArgumentError("window bounds must be finite, got ($t_start, $t_end)"))
    t_start <= t_end ||
        throw(ArgumentError("window must be (t0, t1) with t0 <= t1, got ($t_start, $t_end)"))
    count >= 1 || throw(ArgumentError("the frame count must be positive, got $count"))
    for (index, time) in pairs(times)
        isfinite(time) || throw(ArgumentError("frame time $index is not finite"))
        index > 1 && time < times[index - 1] &&
            throw(ArgumentError("frame times must be nondecreasing"))
    end
    targets = count == 1 ? Float64[t_start] :
        collect(range(Float64(t_start), Float64(t_end); length = count))
    indices = Int[_nearest_index(times, target) for target in targets]
    return VideoSchedule(indices, targets)
end

function video_frame_schedule(
        times::AbstractVector{<:Real}, t_start::Real, t_end::Real;
        fps::Real, duration::Union{Nothing, Real} = nothing,
        nframes::Union{Nothing, Integer} = nothing,
    )
    isnothing(duration) == isnothing(nframes) && throw(
        ArgumentError("give exactly one of duration and nframes"),
    )
    count = isnothing(nframes) ? video_frame_count(fps, duration) : nframes
    return video_frame_schedule(times, t_start, t_end, count)
end

"""
    LiveVideoSampler{F}(t_start, step)

Incremental nearest-frame sampler for a live export. Output frame `k`
(1-based) targets simulation time `t_start + (k - 1) * step`; it shows the
arrived frame nearest to that time, and ties go to the earlier frame. This is
the same rule as [`video_frame_schedule`](@ref), so a live export and a stored
export over the same window pick the same frames when no frame is lost.

Frames arrive in nondecreasing simulation time. Feed each one to
[`push_live_frame!`](@ref), which returns the frames to show for every output time
the arrival decides, in order. Call [`flush_live_frames!`](@ref) after the last
arrival; it returns the last arrived frame for the remaining output times up
to that frame's time. A sparse stream therefore repeats frames and a dense
stream skips them. The sampler keeps only the latest arrived frame.
"""
mutable struct LiveVideoSampler{F}
    t_start::Float64
    step::Float64
    next_index::Int
    left::Union{Nothing, F}
    left_time::Float64

    function LiveVideoSampler{F}(t_start::Real, step::Real) where {F}
        _positive_finite(step, "step")
        isfinite(t_start) ||
            throw(ArgumentError("the live start time must be finite, got $t_start"))
        return new{F}(Float64(t_start), Float64(step), 1, nothing, -Inf)
    end
end

"""
    LiveVideoSampler(t_start, step)

Untyped [`LiveVideoSampler`](@ref); prefer `LiveVideoSampler{F}` when the
frame type is known.
"""
LiveVideoSampler(t_start::Real, step::Real) = LiveVideoSampler{Any}(t_start, step)

_output_time(sampler::LiveVideoSampler, index::Integer) =
    sampler.t_start + (index - 1) * sampler.step

"""
    push_live_frame!(sampler, frame, time)

Feed the frame arrived at simulation time `time` and return the frames to
show for the output times this arrival decides, in output order. Times must be
nondecreasing.
"""
function push_live_frame!(sampler::LiveVideoSampler{F}, frame::F, time::Real) where {F}
    isfinite(time) || throw(ArgumentError("live frame times must be finite"))
    time >= sampler.left_time ||
        throw(ArgumentError("live frame times must be nondecreasing"))
    decided = F[]
    while true
        target = _output_time(sampler, sampler.next_index)
        target <= time || break
        # `left` is the latest arrived frame before this one; equal distances
        # keep it, so ties go to the earlier frame.
        chosen = isnothing(sampler.left) ||
            time - target < target - sampler.left_time ? frame : sampler.left
        push!(decided, chosen)
        sampler.next_index += 1
    end
    sampler.left = frame
    sampler.left_time = Float64(time)
    return decided
end

"""
    flush_live_frames!(sampler)

Return the frames for the remaining output times up to the last arrived
frame's time, then empty the sampler's history.
"""
function flush_live_frames!(sampler::LiveVideoSampler{F}) where {F}
    isnothing(sampler.left) && return F[]
    decided = F[]
    while _output_time(sampler, sampler.next_index) <= sampler.left_time
        push!(decided, sampler.left)
        sampler.next_index += 1
    end
    return decided
end
