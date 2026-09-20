_independent_field_replacements(step_procedure) = IdDict{Any, Any}()

_copy_visualization_velocity(velocity) = copy(velocity)
_copy_visualization_velocity(velocity::TwoPhaseVelocity) = map(copy, velocity)

function _update_visualization!(
        session::VisualizationSession{T};
        event = :step, time, step = 0,
        final_time = nothing, velocity = nothing,
        pressure = nothing, solver = nothing,
        kwargs...
    ) where {T}
    if event == :initial
        isnothing(final_time) &&
            throw(
            ArgumentError(
                "the initial visualization update requires final_time"
            )
        )
        _initialize_session!(session, time, final_time)
    elseif !session.initialized
        throw(
            ArgumentError(
                "the visualization session has not received its initial update"
            )
        )
    end

    if event == :failed
        session.state = :failed
        notify(session.control_condition; all = true)
        _notify_visualization_state!(session)
        _stop_publisher!(session)
        return session
    end

    checkpoint_count_before = length(session.stored_checkpoints)
    frame_time = T(time)
    frame_step = Int(step)
    is_new_state = session.last_update_step != frame_step ||
        session.last_update_time != frame_time
    if event != :initial
        _check_publisher!(session)
    end

    if is_new_state && event != :initial
        _capture_exact_checkpoints!(
            session, frame_time, step, velocity, pressure
        )
    end

    if _diagnostic_due!(session, frame_time, frame_step, event)
        _capture_diagnostic_sample!(
            session, frame_time, frame_step, velocity, pressure, solver
        )
    end

    live_due = _live_frame_due!(session, frame_time, frame_step, event)
    checkpoint_due = is_new_state &&
        _checkpoint_frame_required(session, frame_time)
    frame = session.latest_frame_data
    frame_is_current = !isnothing(frame) && frame.time == frame_time &&
        frame.step == frame_step
    if (live_due || checkpoint_due) && !frame_is_current
        frame = _capture_frame(
            session.registry, frame_time, frame_step;
            topology = _session_topology!(session)
        )
        _store_crossed_checkpoints!(session, frame)
    end

    if is_new_state
        _cache_checkpoint_state!(session, frame_time, velocity, pressure)
        session.last_update_time = frame_time
        session.last_update_step = frame_step
    end

    if event == :completed
        session.state = :completed
    elseif session.state != :pause_requested
        session.state = :running
    end
    if live_due
        isnothing(frame) &&
            throw(ErrorException("live-frame sampling did not capture a frame"))
        _publish_live_frame!(session, frame)
    end
    if length(session.stored_checkpoints) != checkpoint_count_before &&
            !live_due
        _notify_visualization_state!(session)
    end
    if event == :completed
        notify(session.control_condition; all = true)
        _notify_visualization_state!(session)
        _stop_publisher!(session)
    else
        _pause_at_step_boundary!(session)
    end
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
