"""Internal common interface for typed diagnostic time series."""
abstract type AbstractDiagnosticSeries{T} end

mutable struct DiagnosticSeries{
        T, V, D <: AbstractDiagnostic,
        S <: AbstractDiagnosticSchedule,
    } <: AbstractDiagnosticSeries{T}
    diagnostic::D
    schedule::S
    times::Vector{T}
    steps::Vector{Int}
    values::Vector{V}
    next_time::T
end

"""Collection of independently scheduled, typed diagnostic histories."""
mutable struct DiagnosticCollection{T}
    series::Vector{AbstractDiagnosticSeries{T}}
    indices::Dict{Symbol, Int}
end

DiagnosticCollection(::Type{T}) where {T} =
    DiagnosticCollection{T}(AbstractDiagnosticSeries{T}[], Dict{Symbol, Int}())

"""No-op diagnostic target used by the allocation-free low-level API."""
struct NoDiagnostics end

"""A detached, read-only snapshot returned by [`diagnostic_history`](@ref)."""
struct DiagnosticHistory{T, V}
    name::Symbol
    times::Vector{T}
    steps::Vector{Int}
    values::Vector{V}
end

Base.length(history::DiagnosticHistory) = length(history.values)
Base.isempty(history::DiagnosticHistory) = isempty(history.values)
Base.getindex(history::DiagnosticHistory, index) = history.values[index]

function _diagnostic_specs(specifications)
    isnothing(specifications) && return ()
    return specifications isa Tuple ? specifications : (specifications,)
end

function _split_diagnostic_spec(specification)
    if specification isa Pair
        diagnostic, schedule = specification.first, specification.second
    else
        diagnostic, schedule = specification, EveryStep()
    end
    diagnostic isa AbstractDiagnostic || throw(
        ArgumentError("diagnostics must contain AbstractDiagnostic values"),
    )
    schedule isa AbstractDiagnosticSchedule || throw(
        ArgumentError("a diagnostic schedule must subtype AbstractDiagnosticSchedule"),
    )
    return diagnostic, schedule
end

_first_next_time(::Union{EveryStep, EveryNSteps}, time) = time
_first_next_time(schedule::EveryTimeInterval, time) =
    time + typeof(time)(schedule.interval)

function _add_diagnostic!(
        collection::DiagnosticCollection{T}, diagnostic::AbstractDiagnostic,
        state::SimulationState; schedule = EveryStep(),
    ) where {T}
    schedule isa AbstractDiagnosticSchedule || throw(
        ArgumentError("schedule must subtype AbstractDiagnosticSchedule"),
    )
    name = diagnostic_name(diagnostic)
    haskey(collection.indices, name) && throw(
        ArgumentError("diagnostic :$name is already registered"),
    )
    value = diagnostic_value(diagnostic, state)
    series = DiagnosticSeries(
        diagnostic, schedule, T[state.time], Int[state.step],
        typeof(value)[value], _first_next_time(schedule, T(state.time)),
    )
    push!(collection.series, series)
    collection.indices[name] = length(collection.series)
    return collection
end

function _configure_diagnostics!(
        collection::DiagnosticCollection, specifications,
        state::SimulationState,
    )
    for specification in _diagnostic_specs(specifications)
        diagnostic, schedule = _split_diagnostic_spec(specification)
        _add_diagnostic!(collection, diagnostic, state; schedule)
    end
    return collection
end

function _diagnostic_due!(
        series::DiagnosticSeries{T}, state::SimulationState,
        event::Symbol,
    ) where {T}
    last_time = series.times[end]
    last_step = series.steps[end]
    state.time == last_time && state.step == last_step && return false
    event === :completed && return true
    schedule = series.schedule
    schedule isa EveryStep && return true
    schedule isa EveryNSteps && return mod(state.step, schedule.n) == 0
    tolerance = T(16) * eps(T) * max(one(T), abs(state.time))
    state.time + tolerance >= series.next_time || return false
    while series.next_time <= state.time + tolerance
        series.next_time += T(schedule.interval)
    end
    return true
end

function _record_series!(
        series::DiagnosticSeries{T, V}, state::SimulationState;
        event = :step, force = false,
    ) where {T, V}
    (force || _diagnostic_due!(series, state, event)) || return nothing
    if series.times[end] == state.time && series.steps[end] == state.step
        return series.values[end]
    end
    value = diagnostic_value(series.diagnostic, state)
    value isa V || throw(
        ArgumentError(
            "diagnostic :$(diagnostic_name(series.diagnostic)) changed value " *
                "type from $V to $(typeof(value))",
        ),
    )
    push!(series.times, T(state.time))
    push!(series.steps, state.step)
    push!(series.values, value)
    return value
end

_record_diagnostics!(::NoDiagnostics, state; event = :step) = nothing

function _record_diagnostics!(
        collection::DiagnosticCollection, state::SimulationState;
        event = :step,
    )
    for series in collection.series
        _record_series!(series, state; event)
    end
    return collection
end

function _diagnostic_series(collection::DiagnosticCollection, name::Symbol)
    index = get(collection.indices, name, 0)
    iszero(index) && throw(ArgumentError("unknown diagnostic :$name"))
    return collection.series[index]
end

function _diagnostic_value_at!(
        collection::DiagnosticCollection, name::Symbol,
        state::SimulationState,
    )
    series = _diagnostic_series(collection, name)
    value = _record_series!(series, state; force = true)
    return isnothing(value) ? series.values[end] : value
end

diagnostic_names(collection::DiagnosticCollection) =
    [diagnostic_name(series.diagnostic) for series in collection.series]

function diagnostic_history(
        collection::DiagnosticCollection{T}, name::Symbol,
    ) where {T}
    series = _diagnostic_series(collection, name)
    return DiagnosticHistory(
        name, copy(series.times), copy(series.steps), copy(series.values),
    )
end
