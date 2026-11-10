"""
    AtTimes(times)

Keyframe schedule at explicit absolute simulation times. Times are sorted and
deduplicated; each must lie in the simulated interval.
"""
struct AtTimes
    times::Vector{Float64}

    function AtTimes(times)
        values = Float64[Float64(time) for time in times]
        all(isfinite, values) ||
            throw(ArgumentError("keyframe times must be finite"))
        return new(unique!(sort!(values)))
    end
end

"""
    EquispacedTimes(count)

Keyframe schedule of `count` equispaced times that include the initial and
final time of each `simulate!` call. A count of one requests only the final
time.
"""
struct EquispacedTimes
    count::Int

    function EquispacedTimes(count)
        count isa Integer && !(count isa Bool) && count >= 1 || throw(
            ArgumentError("the keyframe count must be a positive integer"),
        )
        return new(Int(count))
    end
end

"""
    EveryWallInterval(seconds)

Preview schedule: at most one preview per `seconds` of wall-clock time.
"""
struct EveryWallInterval
    seconds::Float64

    function EveryWallInterval(seconds::Real)
        isfinite(seconds) && seconds > 0 || throw(
            ArgumentError("the preview wall-clock interval must be finite and positive"),
        )
        return new(Float64(seconds))
    end
end

const _KeyframeSchedule = Union{
    Nothing, AtTimes, EquispacedTimes, EveryTimeInterval, EveryNSteps, EveryStep,
}
const _PreviewSchedule = Union{
    Nothing, EveryWallInterval, EveryTimeInterval, EveryNSteps, EveryStep,
}

_check_keyframe_schedule(schedule::_KeyframeSchedule) = schedule
_check_keyframe_schedule(schedule) = throw(
    ArgumentError(
        "keyframes must be nothing, AtTimes, EquispacedTimes, " *
            "EveryTimeInterval, EveryNSteps, or EveryStep; got $(typeof(schedule))",
    ),
)
_check_preview_schedule(schedule::_PreviewSchedule) = schedule
_check_preview_schedule(schedule) = throw(
    ArgumentError(
        "preview must be nothing, EveryWallInterval, EveryTimeInterval, " *
            "EveryNSteps, or EveryStep; got $(typeof(schedule))",
    ),
)

_time_tolerance(epsilon, initial_time, final_time) =
    32 * epsilon * max(1.0, abs(initial_time), abs(final_time))

# Scheduled keyframe times of one `simulate!` call. Step schedules have none.
_keyframe_times(::Union{Nothing, EveryNSteps, EveryStep}, t0, tf, tolerance) =
    Float64[]

function _keyframe_times(schedule::EquispacedTimes, t0, tf, tolerance)
    schedule.count == 1 && return Float64[tf]
    return collect(range(t0, tf; length = schedule.count))
end

function _keyframe_times(schedule::EveryTimeInterval, t0, tf, tolerance)
    interval = Float64(schedule.interval)
    times = Float64[t0]
    count = 1
    while true
        time = t0 + count * interval
        time < tf - tolerance || break
        push!(times, time)
        count += 1
    end
    if tf - last(times) > tolerance
        push!(times, tf)
    else
        times[end] = tf
    end
    return times
end

function _keyframe_times(schedule::AtTimes, t0, tf, tolerance)
    for time in schedule.times
        t0 - tolerance <= time <= tf + tolerance || throw(
            ArgumentError("keyframe time $time lies outside [$t0, $tf]"),
        )
    end
    return unique!([clamp(time, t0, tf) for time in schedule.times])
end

_step_keyframe_due(::Union{Nothing, AtTimes, EquispacedTimes, EveryTimeInterval}, step) =
    false
_step_keyframe_due(::EveryStep, step) = true
_step_keyframe_due(schedule::EveryNSteps, step) = mod(step, schedule.n) == 0
