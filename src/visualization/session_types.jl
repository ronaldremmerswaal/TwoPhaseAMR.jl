"""
    VisualizationSession(registry; viewer=NoVisualization(), checkpoints=nothing,
                         checkpoint_dt=nothing, checkpoint_times=nothing,
                         exact_checkpoints=false, live_every=nothing,
                         live_dt=nothing, diagnostic_every=nothing,
                         diagnostic_dt=nothing, asynchronous=false)

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
capacity-one latest-wins mailbox; checkpoints remain lossless. `simulate!`
yields at visualization updates so the publisher and GUI can run on the
creating thread even with one Julia thread. Numerical steps run on the caller,
so a long step can delay GUI updates. Call
`request_pause!` to request a pause at the next completed-step boundary and
`resume_simulation!` to continue. The Makie viewer exposes the same control.
Registered diagnostics are sampled after every step by default. Set
`diagnostic_every` or `diagnostic_dt` independently of the live-frame policy.
"""
mutable struct VisualizationSession{T, R, V, S} <: AbstractVisualization
    registry::R
    viewer::V
    specification::S
    exact_checkpoints::Bool
    live_every::Union{Nothing, Int}
    live_dt::Union{Nothing, T}
    asynchronous::Bool
    diagnostic_history::VisualizationDiagnosticHistory{T}
    checkpoint_times::Vector{T}
    stored_checkpoints::Vector{VisualizationCheckpoint{T}}
    latest_frame_data::Union{Nothing, VisualizationFrame{T}}
    previous_frame::Union{Nothing, VisualizationFrame{T}}
    next_checkpoint::Int
    initial_time::T
    final_time::T
    initialized::Bool
    state::Symbol
    mode::Symbol
    selected_checkpoint::Union{Nothing, Int}
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
    topology_keys::Vector{BlockKey}
    topology::Any
    session_lock::ReentrantLock
    control_condition::Base.GenericCondition{ReentrantLock}
end

_supports_dynamic_topology(session::VisualizationSession) =
    !session.exact_checkpoints

function _checkpoint_specification(
        ::Type{T}; checkpoints = nothing,
        checkpoint_dt = nothing,
        checkpoint_times = nothing
    ) where {T}
    supplied = count(
        !isnothing,
        (checkpoints, checkpoint_dt, checkpoint_times)
    )
    supplied <= 1 ||
        throw(
        ArgumentError(
            "specify only one of checkpoints, checkpoint_dt, and checkpoint_times"
        )
    )

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
        throw(
            ArgumentError(
                "checkpoints must not be a boolean"
            )
        )
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
    throw(
        ArgumentError(
            "checkpoints must be an integer count, real time step, or time vector"
        )
    )
end

function _live_sampling(
        ::Type{T}; live_every = nothing,
        live_dt = nothing
    ) where {T}
    !isnothing(live_every) && !isnothing(live_dt) &&
        throw(ArgumentError("specify only one of live_every and live_dt"))
    if !isnothing(live_dt)
        interval = T(live_dt)
        isfinite(interval) && interval > zero(T) ||
            throw(
            ArgumentError(
                "the live-frame time interval must be finite and positive"
            )
        )
        return nothing, interval
    end
    every = isnothing(live_every) ? 1 : live_every
    every isa Bool &&
        throw(ArgumentError("live_every must not be a boolean"))
    every isa Integer && every >= 1 ||
        throw(ArgumentError("live_every must be a positive integer"))
    return Int(every), nothing
end

function _diagnostic_sampling(
        ::Type{T}; diagnostic_every = nothing,
        diagnostic_dt = nothing
    ) where {T}
    !isnothing(diagnostic_every) && !isnothing(diagnostic_dt) &&
        throw(
        ArgumentError(
            "specify only one of diagnostic_every and diagnostic_dt"
        )
    )
    if !isnothing(diagnostic_dt)
        interval = T(diagnostic_dt)
        isfinite(interval) && interval > zero(T) ||
            throw(
            ArgumentError(
                "the diagnostic time interval must be finite and positive"
            )
        )
        return nothing, interval
    end
    every = isnothing(diagnostic_every) ? 1 : diagnostic_every
    every isa Bool &&
        throw(ArgumentError("diagnostic_every must not be a boolean"))
    every isa Integer && every >= 1 ||
        throw(ArgumentError("diagnostic_every must be a positive integer"))
    return Int(every), nothing
end

function VisualizationSession(
        registry::VisualizationFieldRegistry;
        viewer::AbstractVisualization = NoVisualization(),
        checkpoints = nothing, checkpoint_dt = nothing,
        checkpoint_times = nothing,
        exact_checkpoints = false, live_every = nothing,
        live_dt = nothing, diagnostic_every = nothing,
        diagnostic_dt = nothing, asynchronous = false
    )
    isempty(registry.entries) &&
        throw(ArgumentError("a visualization session needs a nonempty registry"))
    mesh = registry.mesh::BlockTreeMesh
    T = eltype(mesh.origin)
    specification = _checkpoint_specification(
        T; checkpoints, checkpoint_dt, checkpoint_times
    )
    every, interval = _live_sampling(T; live_every, live_dt)
    diagnostic_every_value, diagnostic_interval = _diagnostic_sampling(
        T; diagnostic_every, diagnostic_dt
    )
    diagnostic_history = VisualizationDiagnosticHistory{T}(
        diagnostic_every_value, diagnostic_interval, zero(T), zero(T), -1,
        VisualizationDiagnosticSample{T}[]
    )
    session_lock = ReentrantLock()
    control_condition = Threads.Condition(session_lock)
    session = VisualizationSession{
        T, typeof(registry), typeof(viewer),
        typeof(specification),
    }(
        registry, viewer, specification, Bool(exact_checkpoints),
        every, interval, Bool(asynchronous), diagnostic_history, T[],
        VisualizationCheckpoint{T}[], nothing, nothing, 1,
        zero(T), zero(T), false, :created, :live, nothing,
        nothing, nothing, nothing, zero(T), zero(T), zero(T), -1,
        zero(T), -1, nothing, nothing, BlockKey[], nothing,
        session_lock, control_condition
    )
    # Bind controls on the task that constructs the viewer/session. In an
    # interactive application that is the GUI task; doing this lazily from
    # `simulate!` would mutate GL resources from its worker thread.
    _bind_visualization_session!(session.viewer, session)
    _start_publisher!(session)
    return session
end

function _checkpoint_tolerance(initial_time::T, final_time::T) where {T}
    return T(32) * eps(T) *
        max(one(T), abs(initial_time), abs(final_time))
end

function _resolve_checkpoint_times(
        specification::CheckpointCount,
        initial_time::T, final_time::T
    ) where {T}
    specification.count == 1 && return T[final_time]
    return collect(range(initial_time, final_time; length = specification.count))
end

function _resolve_checkpoint_times(
        specification::CheckpointInterval,
        initial_time::T, final_time::T
    ) where {T}
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

function _resolve_checkpoint_times(
        specification::ExplicitCheckpointTimes,
        initial_time::T, final_time::T
    ) where {T}
    tolerance = _checkpoint_tolerance(initial_time, final_time)
    times = T.(specification.times)
    for time in times
        initial_time - tolerance <= time <= final_time + tolerance ||
            throw(
            ArgumentError(
                "checkpoint time $time lies outside [$initial_time, $final_time]"
            )
        )
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

function _yield_visualization!(session::VisualizationSession)
    if session.asynchronous
        yield()
        _check_publisher!(session)
    end
    return nothing
end

function _stop_publisher!(session::VisualizationSession; wait_for_task = false)
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
                frame, state_pending, diagnostics_pending =
                    _take_latest!(mailbox)
                isnothing(frame) && !state_pending &&
                    !diagnostics_pending && break
                if !isnothing(frame)
                    show_live = Base.@lock session.session_lock begin
                        session.mode == :live
                    end
                    if show_live
                        update_visualization!(session.viewer, frame)
                    else
                        _update_visualization_state!(session.viewer, session)
                    end
                end
                if state_pending
                    _update_visualization_state!(session.viewer, session)
                end
                if diagnostics_pending
                    _update_visualization_diagnostics!(session.viewer, session)
                end
            end
        catch
            Base.@lock session.session_lock begin
                session.state = :failed
                notify(session.control_condition; all = true)
            end
            rethrow()
        finally
            _close_mailbox!(mailbox)
        end
    end
    return session
end

function _notify_visualization_state!(session::VisualizationSession)
    if session.asynchronous
        mailbox = session.mailbox
        isnothing(mailbox) || _publish_state!(mailbox)
    else
        _update_visualization_state!(session.viewer, session)
    end
    return session
end

function _notify_visualization_diagnostics!(session::VisualizationSession)
    if session.asynchronous
        mailbox = session.mailbox
        isnothing(mailbox) || _publish_diagnostics!(mailbox)
    else
        _update_visualization_diagnostics!(session.viewer, session)
    end
    return session
end

function _pause_at_step_boundary!(session::VisualizationSession)
    session.state == :pause_requested || return session
    session.state = :paused
    notify(session.control_condition; all = true)
    _notify_visualization_state!(session)
    while session.state == :paused
        wait(session.control_condition)
    end
    session.state == :failed && _check_publisher!(session)
    return session
end

function _initialize_session!(
        session::VisualizationSession{T},
        initial_time, final_time
    ) where {T}
    pause_on_start = session.state == :pause_requested
    if session.initialized
        _stop_publisher!(session)
        task = session.publisher_task
        if !isnothing(task) && !istaskdone(task)
            throw(
                ArgumentError(
                    "wait for asynchronous visualization before reusing the session"
                )
            )
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
        session.specification, start, stop
    )
    empty!(session.stored_checkpoints)
    history = session.diagnostic_history
    empty!(history.samples)
    history.next_time = isnothing(history.interval) ? start :
        start + history.interval
    history.last_time = start
    history.last_step = -1
    session.latest_frame_data = nothing
    session.previous_frame = nothing
    session.next_checkpoint = 1
    session.initial_time = start
    session.final_time = stop
    session.initialized = true
    session.state = pause_on_start ? :pause_requested : :running
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
    empty!(session.topology_keys)
    session.topology = nothing
    return session
end
