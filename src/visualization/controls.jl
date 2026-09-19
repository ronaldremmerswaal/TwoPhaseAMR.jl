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

"""Return a shallow copy of the retained scalar-diagnostic history."""
function visualization_diagnostics(session::VisualizationSession)
    lock(session.session_lock)
    try
        return copy(session.diagnostic_history.samples)
    finally
        unlock(session.session_lock)
    end
end

"""Return the session lifecycle state without exposing its synchronization lock."""
function simulation_state(session::VisualizationSession)
    lock(session.session_lock)
    try
        return session.state
    finally
        unlock(session.session_lock)
    end
end

"""
    request_pause!(session)

Request a cooperative simulation pause. The state changes immediately from
`:created` or `:running` to `:pause_requested`; the simulation changes it to
`:paused` and blocks when its next completed-step visualization update is
reached. Requesting before initialization pauses on the initial state.
"""
function request_pause!(session::VisualizationSession)
    changed = false
    lock(session.session_lock)
    try
        if session.state in (:created, :running)
            session.state = :pause_requested
            changed = true
        elseif session.state in (:pause_requested, :paused)
            return session
        else
            throw(
                ArgumentError(
                    "cannot pause a visualization session in state :$(session.state)"
                )
            )
        end
    finally
        unlock(session.session_lock)
    end
    changed && _notify_visualization_state!(session)
    return session
end

"""
    resume_simulation!(session)

Resume a paused simulation, or cancel a pause request that has not yet reached
a completed-step boundary.
"""
function resume_simulation!(session::VisualizationSession)
    changed = false
    lock(session.session_lock)
    try
        if session.state in (:pause_requested, :paused)
            session.state = session.initialized ? :running : :created
            notify(session.control_condition; all = true)
            changed = true
        elseif session.state == :running
            return session
        else
            throw(
                ArgumentError(
                    "cannot resume a visualization session in state :$(session.state)"
                )
            )
        end
    finally
        unlock(session.session_lock)
    end
    changed && _notify_visualization_state!(session)
    return session
end

"""
    wait_until_paused(session)

Wait until a pending pause request reaches a completed-step boundary. An error
is raised if the session completes or fails before becoming paused.
"""
function wait_until_paused(session::VisualizationSession)
    lock(session.session_lock)
    try
        session.state in (:pause_requested, :paused) ||
            throw(
            ArgumentError(
                "wait_until_paused requires a pending or reached pause"
            )
        )
        while session.state == :pause_requested
            wait(session.control_condition)
        end
        session.state == :paused || throw(
            ErrorException(
                "simulation entered state :$(session.state) before pausing"
            )
        )
    finally
        unlock(session.session_lock)
    end
    return session
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
            session.viewer, session.stored_checkpoints[index].frame
        )
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
            throw(
            ArgumentError(
                "wait_visualization! requires a completed or failed session"
            )
        )
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
        live_pressure, checkpoint_pressure;
        additional_replacements = IdDict{Any, Any}()
    ) where {T}
    requested = T(requested_time)
    session.next_checkpoint <= length(session.checkpoint_times) ||
        throw(ArgumentError("no pending visualization checkpoint"))
    expected = session.checkpoint_times[session.next_checkpoint]
    tolerance = _checkpoint_tolerance(
        session.initial_time, session.final_time
    )
    abs(requested - expected) <= tolerance ||
        throw(
        ArgumentError(
            "exact checkpoint $requested does not match pending time $expected"
        )
    )
    replacements = IdDict{Any, Any}(
        live_velocity => checkpoint_velocity,
        live_pressure => checkpoint_pressure
    )
    if live_velocity isa TwoPhaseVelocity &&
            checkpoint_velocity isa TwoPhaseVelocity
        for phase in 1:2
            replacements[live_velocity[phase]] = checkpoint_velocity[phase]
        end
    end
    for (field, replacement) in additional_replacements
        replacements[field] = replacement
    end
    frame = _capture_frame(
        session.registry, requested, step; replacements,
        topology = _session_topology!(session)
    )
    push!(
        session.stored_checkpoints,
        VisualizationCheckpoint(expected, frame)
    )
    session.next_checkpoint += 1
    return session
end
