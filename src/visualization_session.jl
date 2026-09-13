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

mutable struct LatestFrameMailbox{F}
    condition::Base.GenericCondition{ReentrantLock}
    pending::Union{Nothing,F}
    closed::Bool
    dropped::Int
end

LatestFrameMailbox{F}() where {F} =
    LatestFrameMailbox{F}(Threads.Condition(), nothing, false, 0)

function _publish_latest!(mailbox::LatestFrameMailbox{F}, frame::F) where {F}
    lock(mailbox.condition)
    try
        mailbox.closed && return false
        isnothing(mailbox.pending) || (mailbox.dropped += 1)
        mailbox.pending = frame
        notify(mailbox.condition)
        return true
    finally
        unlock(mailbox.condition)
    end
end

function _take_latest!(mailbox::LatestFrameMailbox)
    lock(mailbox.condition)
    try
        while isnothing(mailbox.pending) && !mailbox.closed
            wait(mailbox.condition)
        end
        frame = mailbox.pending
        mailbox.pending = nothing
        return frame
    finally
        unlock(mailbox.condition)
    end
end

function _discard_pending!(mailbox::LatestFrameMailbox)
    lock(mailbox.condition)
    try
        mailbox.pending = nothing
    finally
        unlock(mailbox.condition)
    end
    return mailbox
end

function _close_mailbox!(mailbox::LatestFrameMailbox)
    lock(mailbox.condition)
    try
        mailbox.closed = true
        notify(mailbox.condition)
    finally
        unlock(mailbox.condition)
    end
    return mailbox
end

function _dropped_frames(mailbox::LatestFrameMailbox)
    lock(mailbox.condition)
    try
        return mailbox.dropped
    finally
        unlock(mailbox.condition)
    end
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
                         exact_checkpoints=false, live_every=nothing,
                         live_dt=nothing, asynchronous=false)

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

Live frames are sampled after every completed step by default. Set
`live_every=n` to sample every `n` steps or `live_dt=interval` to sample in
simulation time. Initial and final states are always sampled. With
`asynchronous=true`, sampled frames are sent to the viewer through a
capacity-one latest-wins mailbox; checkpoints remain lossless.
"""
mutable struct VisualizationSession{T,R,V,S} <: AbstractVisualization
    registry::R
    viewer::V
    specification::S
    exact_checkpoints::Bool
    live_every::Union{Nothing,Int}
    live_dt::Union{Nothing,T}
    asynchronous::Bool
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
    next_live_time::T
    last_update_time::T
    last_update_step::Int
    last_published_time::T
    last_published_step::Int
    mailbox::Any
    publisher_task::Any
    session_lock::ReentrantLock
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

function _live_sampling(::Type{T}; live_every=nothing,
                        live_dt=nothing) where {T}
    !isnothing(live_every) && !isnothing(live_dt) &&
        throw(ArgumentError("specify only one of live_every and live_dt"))
    if !isnothing(live_dt)
        interval = T(live_dt)
        isfinite(interval) && interval > zero(T) ||
            throw(ArgumentError(
                "the live-frame time interval must be finite and positive"))
        return nothing, interval
    end
    every = isnothing(live_every) ? 1 : live_every
    every isa Bool &&
        throw(ArgumentError("live_every must not be a boolean"))
    every isa Integer && every >= 1 ||
        throw(ArgumentError("live_every must be a positive integer"))
    return Int(every), nothing
end

function VisualizationSession(registry::VisualizationFieldRegistry;
                              viewer::AbstractVisualization=NoVisualization(),
                              checkpoints=nothing, checkpoint_dt=nothing,
                              checkpoint_times=nothing,
                              exact_checkpoints=false, live_every=nothing,
                              live_dt=nothing, asynchronous=false)
    isempty(registry.entries) &&
        throw(ArgumentError("a visualization session needs a nonempty registry"))
    mesh = registry.mesh::BlockTreeMesh
    T = eltype(mesh.origin)
    specification = _checkpoint_specification(
        T; checkpoints, checkpoint_dt, checkpoint_times)
    every, interval = _live_sampling(T; live_every, live_dt)
    session_lock = ReentrantLock()
    session = VisualizationSession{T,typeof(registry),typeof(viewer),
                                   typeof(specification)}(
        registry, viewer, specification, Bool(exact_checkpoints),
        every, interval, Bool(asynchronous), T[],
        VisualizationCheckpoint{T}[], nothing, nothing, 1,
        zero(T), zero(T), false, :created, :live, nothing,
        nothing, nothing, nothing, zero(T), zero(T), zero(T), -1,
        zero(T), -1, nothing, nothing, session_lock)
    _start_publisher!(session)
    return session
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

function _check_publisher!(session::VisualizationSession)
    task = session.publisher_task
    if !isnothing(task) && istaskfailed(task)
        fetch(task)
    end
    return session
end

function _stop_publisher!(session::VisualizationSession; wait_for_task=false)
    mailbox = session.mailbox
    isnothing(mailbox) || _close_mailbox!(mailbox)
    task = session.publisher_task
    if wait_for_task && !isnothing(task)
        fetch(task)
    end
    return session
end

function _start_publisher!(session::VisualizationSession{T}) where {T}
    session.asynchronous || return session
    mailbox = LatestFrameMailbox{VisualizationFrame{T}}()
    session.mailbox = mailbox
    session.publisher_task = @async begin
        try
            while true
                frame = _take_latest!(mailbox)
                isnothing(frame) && break
                lock(session.session_lock)
                show_live = try
                    session.mode == :live
                finally
                    unlock(session.session_lock)
                end
                if show_live
                    update_visualization!(session.viewer, frame)
                else
                    _update_visualization_state!(session.viewer, session)
                end
            end
        catch
            lock(session.session_lock)
            try
                session.state = :failed
            finally
                unlock(session.session_lock)
            end
            rethrow()
        finally
            _close_mailbox!(mailbox)
        end
    end
    return session
end

function _initialize_session!(session::VisualizationSession{T},
                              initial_time, final_time) where {T}
    if session.initialized
        _stop_publisher!(session)
        task = session.publisher_task
        if !isnothing(task) && !istaskdone(task)
            throw(ArgumentError(
                "wait for asynchronous visualization before reusing the session"))
        end
        isnothing(task) || fetch(task)
        session.mailbox = nothing
        session.publisher_task = nothing
        _start_publisher!(session)
    elseif session.asynchronous && isnothing(session.publisher_task)
        _start_publisher!(session)
    end
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
    session.next_live_time = isnothing(session.live_dt) ? start :
                             start + session.live_dt
    session.last_update_time = start
    session.last_update_step = -1
    session.last_published_time = start
    session.last_published_step = -1
    _bind_visualization_session!(session.viewer, session)
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

function _checkpoint_frame_required(session::VisualizationSession{T},
                                    time::T) where {T}
    session.next_checkpoint <= length(session.checkpoint_times) || return false
    session.exact_checkpoints || return true
    tolerance = _checkpoint_tolerance(
        session.initial_time, session.final_time)
    requested_time = session.checkpoint_times[session.next_checkpoint]
    return requested_time <= time + tolerance
end

function _live_frame_due!(session::VisualizationSession{T}, time::T,
                          step::Int, event::Symbol) where {T}
    if event == :initial
        return true
    elseif event == :completed
        return session.last_published_step != step ||
               session.last_published_time != time
    elseif !isnothing(session.live_every)
        return mod(step, session.live_every) == 0
    end

    tolerance = _checkpoint_tolerance(
        session.initial_time, session.final_time)
    time + tolerance >= session.next_live_time || return false
    while session.next_live_time <= time + tolerance
        session.next_live_time += session.live_dt
    end
    return true
end

function _publish_live_frame!(session::VisualizationSession,
                              frame::VisualizationFrame)
    if session.asynchronous
        _check_publisher!(session)
        _publish_latest!(session.mailbox, frame) ||
            throw(ErrorException("the live-frame publisher is closed"))
        session.last_published_time = frame.time
        session.last_published_step = frame.step
    elseif session.mode == :live
        update_visualization!(session.viewer, frame)
        session.last_published_time = frame.time
        session.last_published_step = frame.step
    else
        _update_visualization_state!(session.viewer, session)
    end
    return session
end

function _visualization_session_state(session::VisualizationSession)
    lock(session.session_lock)
    try
        return (
            state=session.state,
            mode=session.mode,
            selected_checkpoint=session.selected_checkpoint,
            checkpoint_count=length(session.stored_checkpoints),
            latest_frame=session.latest_frame_data,
        )
    finally
        unlock(session.session_lock)
    end
end

function initialize_visualization!(session::VisualizationSession;
                                   step_procedure=nothing, kwargs...)
    update_visualization!(session; event=:initial, kwargs...)
    lock(session.session_lock)
    try
        session.step_procedure = step_procedure
    finally
        unlock(session.session_lock)
    end
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

function _update_visualization!(session::VisualizationSession{T};
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
        _stop_publisher!(session)
        return session
    end

    frame_time = T(time)
    frame_step = Int(step)
    is_new_state = session.last_update_step != frame_step ||
                   session.last_update_time != frame_time
    if event != :initial
        _check_publisher!(session)
    end

    if is_new_state && event != :initial
        _capture_exact_checkpoints!(
            session, frame_time, step, velocity, pressure)
    end

    live_due = _live_frame_due!(session, frame_time, frame_step, event)
    checkpoint_due = is_new_state &&
                     _checkpoint_frame_required(session, frame_time)
    frame = session.latest_frame_data
    frame_is_current = !isnothing(frame) && frame.time == frame_time &&
                       frame.step == frame_step
    if (live_due || checkpoint_due) && !frame_is_current
        frame = _capture_frame(session.registry, frame_time, frame_step)
        _store_crossed_checkpoints!(session, frame)
    end

    if is_new_state
        _cache_checkpoint_state!(session, frame_time, velocity, pressure)
        session.last_update_time = frame_time
        session.last_update_step = frame_step
    end

    session.state = event == :completed ? :completed : :running
    if live_due
        isnothing(frame) &&
            throw(ErrorException("live-frame sampling did not capture a frame"))
        _publish_live_frame!(session, frame)
    end
    event == :completed && _stop_publisher!(session)
    return session
end

function update_visualization!(session::VisualizationSession; kwargs...)
    lock(session.session_lock)
    try
        return _update_visualization!(session; kwargs...)
    finally
        unlock(session.session_lock)
    end
end

"""Return a shallow copy of the retained checkpoint records."""
function visualization_checkpoints(session::VisualizationSession)
    lock(session.session_lock)
    try
        return copy(session.stored_checkpoints)
    finally
        unlock(session.session_lock)
    end
end

"""Return the most recent complete copied frame, or `nothing`."""
function latest_frame(session::VisualizationSession)
    lock(session.session_lock)
    try
        return session.latest_frame_data
    finally
        unlock(session.session_lock)
    end
end

"""Display a retained checkpoint while continuing to ingest live frames."""
function select_checkpoint!(session::VisualizationSession, index::Integer)
    lock(session.session_lock)
    try
        checkbounds(session.stored_checkpoints, index)
        _check_publisher!(session)
        session.asynchronous && _discard_pending!(session.mailbox)
        session.mode = :checkpoint
        session.selected_checkpoint = Int(index)
        update_visualization!(
            session.viewer, session.stored_checkpoints[index].frame)
    finally
        unlock(session.session_lock)
    end
    return session
end

"""Return to the most recent complete live frame."""
function live_mode!(session::VisualizationSession)
    lock(session.session_lock)
    try
        _check_publisher!(session)
        session.asynchronous && _discard_pending!(session.mailbox)
        session.mode = :live
        session.selected_checkpoint = nothing
        frame = session.latest_frame_data
        if !isnothing(frame)
            update_visualization!(session.viewer, frame)
            session.last_published_time = frame.time
            session.last_published_step = frame.step
        end
    finally
        unlock(session.session_lock)
    end
    return session
end

"""Wait for an asynchronous viewer to consume its final live frame."""
function wait_visualization!(session::VisualizationSession)
    lock(session.session_lock)
    task = try
        session.state in (:completed, :failed) ||
            throw(ArgumentError(
                "wait_visualization! requires a completed or failed session"))
        session.publisher_task
    finally
        unlock(session.session_lock)
    end
    isnothing(task) || fetch(task)
    return session
end

"""Return the number of transient live frames replaced before rendering."""
function dropped_visualization_frames(session::VisualizationSession)
    lock(session.session_lock)
    mailbox = try
        session.mailbox
    finally
        unlock(session.session_lock)
    end
    return isnothing(mailbox) ? 0 : _dropped_frames(mailbox)
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
