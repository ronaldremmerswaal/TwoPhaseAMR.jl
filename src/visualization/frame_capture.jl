function _visualization_topology(mesh::BlockTreeMesh{T}) where {T}
    return VisualizationTopology(
        visualization_geometry(mesh),
        (
            visualization_geometry(mesh, Face{1}()),
            visualization_geometry(mesh, Face{2}()),
        ),
        copy(mesh.faces), copy(mesh.boundary_faces)
    )
end

function _session_topology!(session::VisualizationSession)
    mesh = session.registry.mesh::BlockTreeMesh
    unchanged = length(session.topology_keys) == nblocks(mesh)
    if unchanged
        for (index, block) in pairs(mesh.blocks)
            if session.topology_keys[index] != block.key
                unchanged = false
                break
            end
        end
    end
    if !unchanged || isnothing(session.topology)
        empty!(session.topology_keys)
        append!(session.topology_keys, (block.key for block in mesh.blocks))
        session.topology = _visualization_topology(mesh)
    end
    return session.topology
end

function _capture_frame(
        registry::VisualizationFieldRegistry, time::T,
        step::Integer;
        replacements = IdDict{Any, Any}(),
        topology = _visualization_topology(
            registry.mesh::BlockTreeMesh
        )
    ) where {T}
    primitives = IdDict{Any, Any}()
    phase_geometries = IdDict{Any, Dict{DataType, Any}}()
    for interface in registry.interfaces
        primitives[interface] = plic_segments(interface)
    end
    for geometry in registry.geometries
        primitives[geometry] = plic_solid_geometry(geometry)
    end
    for entry in registry.entries
        source = entry.source
        if _is_discontinuous_source(source)
            values = source isa PhaseVisualizationSource ?
                _phase_visualization_values(source, replacements) :
                _phase_velocity_visualization_values(source, replacements)
            interface = _source_interface(source)
            geometries = get!(phase_geometries, interface) do
                Dict{DataType, Any}()
            end
            placement_type = typeof(_source_placement(source))
            geometry = get!(geometries, placement_type) do
                _phase_visualization_geometry(
                    source, topology, replacements,
                )
            end
            primitives[source] = PLICVisualizationSnapshot(values, geometry)
        elseif source isa CellVisualizationSource
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
    frame = VisualizationFrame(
        time, Int(step), registry, primitives, Dict{Symbol, Any}(), topology
    )
    # Cache every registered scalar while the captured arrays and current mesh
    # schedules still describe the same topology. This keeps old checkpoints
    # valid after the live mesh is adapted, including derived face scalars.
    for entry in registry.entries
        frame.values[entry.name] = copy(_frame_values(frame, entry.source))
    end
    return frame
end

function _store_crossed_checkpoints!(
        session::VisualizationSession{T},
        frame::VisualizationFrame{T}
    ) where {T}
    tolerance = _checkpoint_tolerance(
        session.initial_time, session.final_time
    )
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
        push!(
            session.stored_checkpoints,
            VisualizationCheckpoint(requested_time, selected)
        )
        session.next_checkpoint += 1
    end
    session.previous_frame = frame
    session.latest_frame_data = frame
    return session
end

function _checkpoint_frame_required(
        session::VisualizationSession{T},
        time::T
    ) where {T}
    session.next_checkpoint <= length(session.checkpoint_times) || return false
    session.exact_checkpoints || return true
    tolerance = _checkpoint_tolerance(
        session.initial_time, session.final_time
    )
    requested_time = session.checkpoint_times[session.next_checkpoint]
    return requested_time <= time + tolerance
end

function _live_frame_due!(
        session::VisualizationSession{T}, time::T,
        step::Int, event::Symbol
    ) where {T}
    if event == :initial
        return true
    elseif event == :completed
        return session.last_published_step != step ||
            session.last_published_time != time
    elseif !isnothing(session.live_every)
        return mod(step, session.live_every) == 0
    end

    tolerance = _checkpoint_tolerance(
        session.initial_time, session.final_time
    )
    time + tolerance >= session.next_live_time || return false
    while session.next_live_time <= time + tolerance
        session.next_live_time += session.live_dt
    end
    return true
end

function _diagnostic_due!(
        session::VisualizationSession{T}, time::T,
        step::Int, event::Symbol
    ) where {T}
    isempty(session.registry.diagnostics) && return false
    history = session.diagnostic_history
    if event == :initial
        return true
    elseif event == :completed
        return history.last_step != step || history.last_time != time
    elseif !isnothing(history.every)
        return mod(step, history.every) == 0
    end

    tolerance = _checkpoint_tolerance(
        session.initial_time, session.final_time
    )
    time + tolerance >= history.next_time || return false
    while history.next_time <= time + tolerance
        history.next_time += history.interval
    end
    return true
end

function _capture_diagnostic_sample!(
        session::VisualizationSession{T},
        time::T, step::Int,
        velocity, pressure, solver
    ) where {T}
    values = Dict{Symbol, T}()
    for entry in session.registry.diagnostics
        value = entry.reduction(velocity, pressure, time, step, solver)
        value isa Real || throw(
            ArgumentError(
                "visualization diagnostic :$(entry.name) returned " *
                    "$(typeof(value)); expected a real scalar"
            )
        )
        values[entry.name] = T(value)
    end
    sample = VisualizationDiagnosticSample(time, step, values)
    history = session.diagnostic_history
    push!(history.samples, sample)
    history.last_time = time
    history.last_step = step
    _notify_visualization_diagnostics!(session)
    return sample
end

function _publish_live_frame!(
        session::VisualizationSession,
        frame::VisualizationFrame
    )
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
            state = session.state,
            mode = session.mode,
            selected_checkpoint = session.selected_checkpoint,
            checkpoint_count = length(session.stored_checkpoints),
            latest_frame = session.latest_frame_data,
        )
    finally
        unlock(session.session_lock)
    end
end

function initialize_visualization!(
        session::VisualizationSession;
        step_procedure = nothing, kwargs...
    )
    update_visualization!(session; event = :initial, kwargs...)
    lock(session.session_lock)
    try
        session.step_procedure = step_procedure
    finally
        unlock(session.session_lock)
    end
    return session
end

function _cache_checkpoint_state!(
        session::VisualizationSession{T}, time,
        velocity, pressure
    ) where {T}
    session.exact_checkpoints || return session
    isnothing(velocity) &&
        throw(ArgumentError("exact checkpoints require simulation velocity"))
    isnothing(pressure) &&
        throw(ArgumentError("exact checkpoints require simulation pressure"))
    session.checkpoint_velocity = _copy_visualization_velocity(velocity)
    session.checkpoint_pressure = copy(pressure)
    session.checkpoint_state_time = T(time)
    return session
end

function _capture_exact_checkpoints!(
        session::VisualizationSession{T},
        time, step,
        live_velocity, live_pressure
    ) where {T}
    session.exact_checkpoints || return session
    current_time = T(time)
    start_time = session.checkpoint_state_time
    current_time > start_time || return session
    isnothing(session.step_procedure) &&
        throw(
        ArgumentError(
            "exact checkpoints require an independent step procedure"
        )
    )
    tolerance = _checkpoint_tolerance(
        session.initial_time, session.final_time
    )

    while session.next_checkpoint <= length(session.checkpoint_times)
        checkpoint_time = session.checkpoint_times[session.next_checkpoint]
        if checkpoint_time <= start_time + tolerance
            session.next_checkpoint += 1
            continue
        end
        checkpoint_time < current_time - tolerance || break
        checkpoint_velocity = _copy_visualization_velocity(
            session.checkpoint_velocity,
        )
        checkpoint_pressure = copy(session.checkpoint_pressure)
        session.step_procedure(
            checkpoint_velocity, checkpoint_pressure,
            checkpoint_time - start_time; time = start_time
        )
        replacements = _independent_field_replacements(
            session.step_procedure
        )
        _store_exact_checkpoint!(
            session, checkpoint_time, step,
            live_velocity, checkpoint_velocity,
            live_pressure, checkpoint_pressure;
            additional_replacements = replacements
        )
    end
    return session
end
