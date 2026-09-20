"""Supertype for diagnostic sampling schedules."""
abstract type AbstractDiagnosticSchedule end

"""Sample a diagnostic after every completed simulation step."""
struct EveryStep <: AbstractDiagnosticSchedule end

"""Sample every `n` completed simulation steps."""
struct EveryNSteps <: AbstractDiagnosticSchedule
    n::Int
    function EveryNSteps(n)
        n isa Integer && !(n isa Bool) && n > 0 || throw(
            ArgumentError("the diagnostic step interval must be a positive integer"),
        )
        return new(Int(n))
    end
end

"""Sample whenever simulation time crosses a multiple of `interval`."""
struct EveryTimeInterval{T <: Real} <: AbstractDiagnosticSchedule
    interval::T
    function EveryTimeInterval(interval::T) where {T <: Real}
        isfinite(interval) && interval > zero(T) || throw(
            ArgumentError("the diagnostic time interval must be finite and positive"),
        )
        return new{T}(interval)
    end
end
