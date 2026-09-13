abstract type AbstractCheckpointSpecification end

struct CheckpointCount <: AbstractCheckpointSpecification
    count::Int

    function CheckpointCount(count::Integer)
        count >= 1 ||
            throw(ArgumentError("the checkpoint count must be positive"))
        new(Int(count))
    end
end

struct CheckpointInterval{T<:Real} <: AbstractCheckpointSpecification
    interval::T

    function CheckpointInterval(interval::T) where {T<:Real}
        isfinite(interval) && interval > zero(T) ||
            throw(ArgumentError(
                "the checkpoint time step must be finite and positive"))
        new{T}(interval)
    end
end

struct ExplicitCheckpointTimes{T<:Real} <: AbstractCheckpointSpecification
    times::Vector{T}
end

"""
    VisualizationFrame

Snapshot metadata and copied primitive field data for one completed solver
state. Primitive arrays never alias solver-owned storage. Registered scalar
views are accessed with `frame[name]`; derived views are recomputed on demand.
"""
struct VisualizationFrame{T}
    time::T
    step::Int
    registry::VisualizationFieldRegistry
    primitives::IdDict{Any,Any}
end

Base.haskey(frame::VisualizationFrame, name::Symbol) =
    haskey(frame.registry.indices, name)

function Base.getindex(frame::VisualizationFrame, name::Symbol)
    entry = _field_entry(frame.registry, name)
    return _frame_values(frame, entry.source)
end

_frame_values(frame::VisualizationFrame, source::CellVisualizationSource) =
    frame.primitives[source.field]

_frame_values(frame::VisualizationFrame, source::FaceVisualizationSource) =
    visualization_values(frame.primitives[source.field])

function _frame_values(frame::VisualizationFrame,
                       source::FaceComponentVisualizationSource{Axis}) where {Axis}
    velocity = frame.primitives[source.field]
    return visualization_values(velocity, Face{Axis}())
end

function _frame_values(frame::VisualizationFrame,
                       source::DerivedFaceVisualizationSource)
    field = source.field
    if field isa FaceVelocity
        return visualization_values(
            frame.primitives[field], source.derived)
    end
    fields = (frame.primitives[field[1]], frame.primitives[field[2]])
    return visualization_values(fields, source.derived)
end

"""A retained frame together with the time at which it was requested."""
struct VisualizationCheckpoint{T}
    requested_time::T
    frame::VisualizationFrame{T}
end

"""
    VisualizationSession(registry; viewer=NoVisualization(), checkpoints=nothing,
                         checkpoint_dt=nothing, checkpoint_times=nothing,
                         exact_checkpoints=false)

Own live copied frames and a fixed checkpoint schedule for one simulation.
Specify at most one of:

- `checkpoints=<integer>` for that many equispaced checkpoints;
- `checkpoint_dt=<real>` for a checkpoint time interval; or
- `checkpoint_times=<vector>` for explicit absolute simulation times.

Count and interval schedules include the initial and final times. A count of
one retains only the final state. Explicit times are sorted and deduplicated
and must fall within the simulated interval.

By default, the completed solution closest to every requested time is retained.
With `exact_checkpoints=true`, the session invokes the independent step
procedure supplied when the simulation initializes it. Each off-grid state is
advanced from copied solver state and retained without changing the simulation
path, whose step sequence and state are unchanged.
"""
mutable struct VisualizationSession{T,R,V,S} <: AbstractVisualization
    registry::R
    viewer::V
    specification::S
    exact_checkpoints::Bool
    checkpoint_times::Vector{T}
    stored_checkpoints::Vector{VisualizationCheckpoint{T}}
    latest_frame_data::Union{Nothing,VisualizationFrame{T}}
    previous_frame::Union{Nothing,VisualizationFrame{T}}
    next_checkpoint::Int
    initial_time::T
    final_time::T
    initialized::Bool
    state::Symbol
    mode::Symbol
    selected_checkpoint::Union{Nothing,Int}
    step_procedure::Any
    checkpoint_velocity::Any
    checkpoint_pressure::Any
    checkpoint_state_time::T
end

function _checkpoint_specification(::Type{T}; checkpoints=nothing,
                                   checkpoint_dt=nothing,
                                   checkpoint_times=nothing) where {T}
    supplied = count(!isnothing,
                     (checkpoints, checkpoint_dt, checkpoint_times))
    supplied <= 1 ||
        throw(ArgumentError(
            "specify only one of checkpoints, checkpoint_dt, and checkpoint_times"))

    if !isnothing(checkpoint_dt)
        return CheckpointInterval(T(checkpoint_dt))
    elseif !isnothing(checkpoint_times)
        times = T.(collect(checkpoint_times))
        all(isfinite, times) ||
            throw(ArgumentError("checkpoint times must be finite"))
        sort!(times)
        unique!(times)
        return ExplicitCheckpointTimes(times)
    elseif isnothing(checkpoints)
        return ExplicitCheckpointTimes(T[])
    elseif checkpoints isa Bool
        throw(ArgumentError(
            "checkpoints must not be a boolean"))
    elseif checkpoints isa Integer
        return CheckpointCount(checkpoints)
    elseif checkpoints isa Real
        return CheckpointInterval(T(checkpoints))
    elseif checkpoints isa AbstractVector || checkpoints isa Tuple
        times = T.(collect(checkpoints))
        all(isfinite, times) ||
            throw(ArgumentError("checkpoint times must be finite"))
        sort!(times)
        unique!(times)
        return ExplicitCheckpointTimes(times)
    end
    throw(ArgumentError(
        "checkpoints must be an integer count, real time step, or time vector"))
end

function VisualizationSession(registry::VisualizationFieldRegistry;
                              viewer::AbstractVisualization=NoVisualization(),
                              checkpoints=nothing, checkpoint_dt=nothing,
                              checkpoint_times=nothing,
                              exact_checkpoints=false)
    isempty(registry.entries) &&
        throw(ArgumentError("a visualization session needs a nonempty registry"))
    mesh = registry.mesh::BlockTreeMesh
    T = eltype(mesh.origin)
    specification = _checkpoint_specification(
        T; checkpoints, checkpoint_dt, checkpoint_times)
    return VisualizationSession{T,typeof(registry),typeof(viewer),
                                typeof(specification)}(
        registry, viewer, specification, Bool(exact_checkpoints), T[],
        VisualizationCheckpoint{T}[], nothing, nothing, 1,
        zero(T), zero(T), false, :created, :live, nothing,
        nothing, nothing, nothing, zero(T))
end

function _checkpoint_tolerance(initial_time::T, final_time::T) where {T}
    return T(32) * eps(T) *
           max(one(T), abs(initial_time), abs(final_time))
end

function _resolve_checkpoint_times(specification::CheckpointCount,
                                   initial_time::T, final_time::T) where {T}
    specification.count == 1 && return T[final_time]
    return collect(range(initial_time, final_time; length=specification.count))
end

function _resolve_checkpoint_times(specification::CheckpointInterval,
                                   initial_time::T, final_time::T) where {T}
    interval = T(specification.interval)
    tolerance = _checkpoint_tolerance(initial_time, final_time)
    times = T[initial_time]
    step = 1
    while true
        time = initial_time + step * interval
        time < final_time - tolerance || break
        push!(times, time)
        step += 1
    end
    if final_time - last(times) > tolerance
        push!(times, final_time)
    else
        times[end] = final_time
    end
    return times
end

function _resolve_checkpoint_times(specification::ExplicitCheckpointTimes,
                                   initial_time::T, final_time::T) where {T}
    tolerance = _checkpoint_tolerance(initial_time, final_time)
    times = T.(specification.times)
    for time in times
        initial_time - tolerance <= time <= final_time + tolerance ||
            throw(ArgumentError(
                "checkpoint time $time lies outside [$initial_time, $final_time]"))
    end
    resolved = [clamp(time, initial_time, final_time) for time in times]
    unique!(resolved)
    return resolved
end

function _initialize_session!(session::VisualizationSession{T},
                              initial_time, final_time) where {T}
    start = T(initial_time)
    stop = T(final_time)
    stop >= start ||
        throw(ArgumentError("final_time must not precede the initial time"))
    session.checkpoint_times = _resolve_checkpoint_times(
        session.specification, start, stop)
    empty!(session.stored_checkpoints)
    session.latest_frame_data = nothing
    session.previous_frame = nothing
    session.next_checkpoint = 1
    session.initial_time = start
    session.final_time = stop
    session.initialized = true
    session.state = :running
    session.mode = :live
    session.selected_checkpoint = nothing
    session.step_procedure = nothing
    session.checkpoint_velocity = nothing
    session.checkpoint_pressure = nothing
    session.checkpoint_state_time = start
    return session
end

function _capture_frame(registry::VisualizationFieldRegistry, time::T,
                        step::Integer;
                        replacements=IdDict{Any,Any}()) where {T}
    primitives = IdDict{Any,Any}()
    for entry in registry.entries
        source = entry.source
        if source isa CellVisualizationSource
            field = get(replacements, source.field, source.field)
            get!(primitives, source.field) do
                visualization_values(field)
            end
        elseif source isa FaceVisualizationSource
            field = get(replacements, source.field, source.field)
            get!(primitives, source.field) do
                copy(field)
            end
        elseif source isa FaceComponentVisualizationSource
            live_velocity = source.field
            velocity = get(replacements, live_velocity, live_velocity)
            get!(primitives, live_velocity) do
                copy(velocity)
            end
        else
            field = source.field
            if field isa FaceVelocity
                velocity = get(replacements, field, field)
                get!(primitives, field) do
                    copy(velocity)
                end
            else
                for component in field
                    captured = get(replacements, component, component)
                    get!(primitives, component) do
                        copy(captured)
                    end
                end
            end
        end
    end
    return VisualizationFrame(time, Int(step), registry, primitives)
end

function _store_crossed_checkpoints!(session::VisualizationSession{T},
                                     frame::VisualizationFrame{T}) where {T}
    tolerance = _checkpoint_tolerance(
        session.initial_time, session.final_time)
    while session.next_checkpoint <= length(session.checkpoint_times)
        requested_time = session.checkpoint_times[session.next_checkpoint]
        requested_time <= frame.time + tolerance || break
        previous = session.previous_frame
        selected = if isnothing(previous)
            frame
        elseif abs(previous.time - requested_time) <=
               abs(frame.time - requested_time)
            previous
        else
            frame
        end
        push!(session.stored_checkpoints,
              VisualizationCheckpoint(requested_time, selected))
        session.next_checkpoint += 1
    end
    session.previous_frame = frame
    session.latest_frame_data = frame
    return session
end

function initialize_visualization!(session::VisualizationSession;
                                   step_procedure=nothing, kwargs...)
    update_visualization!(session; event=:initial, kwargs...)
    session.step_procedure = step_procedure
    return session
end

function _cache_checkpoint_state!(session::VisualizationSession{T}, time,
                                  velocity, pressure) where {T}
    session.exact_checkpoints || return session
    isnothing(velocity) &&
        throw(ArgumentError("exact checkpoints require simulation velocity"))
    isnothing(pressure) &&
        throw(ArgumentError("exact checkpoints require simulation pressure"))
    session.checkpoint_velocity = copy(velocity)
    session.checkpoint_pressure = copy(pressure)
    session.checkpoint_state_time = T(time)
    return session
end

function _capture_exact_checkpoints!(session::VisualizationSession{T},
                                     time, step,
                                     live_velocity, live_pressure) where {T}
    session.exact_checkpoints || return session
    current_time = T(time)
    start_time = session.checkpoint_state_time
    current_time > start_time || return session
    isnothing(session.step_procedure) &&
        throw(ArgumentError(
            "exact checkpoints require an independent step procedure"))
    tolerance = _checkpoint_tolerance(
        session.initial_time, session.final_time)

    while session.next_checkpoint <= length(session.checkpoint_times)
        checkpoint_time = session.checkpoint_times[session.next_checkpoint]
        if checkpoint_time <= start_time + tolerance
            session.next_checkpoint += 1
            continue
        end
        checkpoint_time < current_time - tolerance || break
        checkpoint_velocity = copy(session.checkpoint_velocity)
        checkpoint_pressure = copy(session.checkpoint_pressure)
        session.step_procedure(
            checkpoint_velocity, checkpoint_pressure,
            checkpoint_time - start_time; time=start_time)
        _store_exact_checkpoint!(
            session, checkpoint_time, step,
            live_velocity, checkpoint_velocity,
            live_pressure, checkpoint_pressure)
    end
    return session
end

function update_visualization!(session::VisualizationSession{T};
                               event=:step, time, step=0,
                               final_time=nothing, velocity=nothing,
                               pressure=nothing, kwargs...) where {T}
    if event == :initial
        isnothing(final_time) &&
            throw(ArgumentError(
                "the initial visualization update requires final_time"))
        _initialize_session!(session, time, final_time)
    elseif !session.initialized
        throw(ArgumentError(
            "the visualization session has not received its initial update"))
    end

    if event == :failed
        session.state = :failed
        return session
    end

    frame_time = T(time)
    frame = session.latest_frame_data
    if isnothing(frame) || frame.time != frame_time || frame.step != step
        event == :initial || _capture_exact_checkpoints!(
            session, frame_time, step, velocity, pressure)
        frame = _capture_frame(session.registry, frame_time, step)
        _store_crossed_checkpoints!(session, frame)
        _cache_checkpoint_state!(session, frame_time, velocity, pressure)
    end

    if session.mode == :live
        update_visualization!(session.viewer, frame)
    end
    session.state = event == :completed ? :completed : :running
    return session
end

"""Return a shallow copy of the retained checkpoint records."""
visualization_checkpoints(session::VisualizationSession) =
    copy(session.stored_checkpoints)

"""Return the most recent complete copied frame, or `nothing`."""
latest_frame(session::VisualizationSession) = session.latest_frame_data

"""Display a retained checkpoint while continuing to ingest live frames."""
function select_checkpoint!(session::VisualizationSession, index::Integer)
    checkbounds(session.stored_checkpoints, index)
    session.mode = :checkpoint
    session.selected_checkpoint = Int(index)
    update_visualization!(
        session.viewer, session.stored_checkpoints[index].frame)
    return session
end

"""Return to the most recent complete live frame."""
function live_mode!(session::VisualizationSession)
    session.mode = :live
    session.selected_checkpoint = nothing
    isnothing(session.latest_frame_data) ||
        update_visualization!(session.viewer, session.latest_frame_data)
    return session
end

function _store_exact_checkpoint!(
        session::VisualizationSession{T}, requested_time, step,
        live_velocity, checkpoint_velocity,
        live_pressure, checkpoint_pressure) where {T}
    requested = T(requested_time)
    session.next_checkpoint <= length(session.checkpoint_times) ||
        throw(ArgumentError("no pending visualization checkpoint"))
    expected = session.checkpoint_times[session.next_checkpoint]
    tolerance = _checkpoint_tolerance(
        session.initial_time, session.final_time)
    abs(requested - expected) <= tolerance ||
        throw(ArgumentError(
            "exact checkpoint $requested does not match pending time $expected"))
    replacements = IdDict{Any,Any}(
        live_velocity => checkpoint_velocity,
        live_pressure => checkpoint_pressure)
    frame = _capture_frame(
        session.registry, requested, step; replacements)
    push!(session.stored_checkpoints,
          VisualizationCheckpoint(expected, frame))
    session.next_checkpoint += 1
    return session
end
