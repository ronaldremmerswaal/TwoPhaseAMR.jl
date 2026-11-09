# A job is owned by the server, never by a browser connection. All its fields
# except the phase-2 handle's own atomic counters are protected by app.lock.
mutable struct RecordingJob
    active::Bool
    source::Union{Nothing, Symbol}
    field::Union{Nothing, Symbol}
    path::Union{Nothing, String}
    diagnostics::Vector{Symbol}
    status::Symbol
    error::Union{Nothing, String}
    encoded::Int
    dropped::Int
    stop_requested::Bool
    selection::Union{Nothing, Vector{TwoPhaseAMR.FrameInfo}}
    diagnostic_counts::Union{Nothing, Vector{Int}}
    handle::Any
    task::Union{Nothing, Task}
end

RecordingJob() = RecordingJob(
    false, nothing, nothing, nothing, Symbol[], :idle,
    nothing, 0, 0, false, nothing, nothing, nothing, nothing
)

# The phase-2 handle's dropped count is a virtual property. Read the
# subscription directly so accounting is safe across extension world ages.
function _live_counts(handle)
    subscription = getfield(handle, :subscription)
    encoded = getfield(handle, :encoded, :sequentially_consistent)
    dropped = Base.@lock subscription.condition subscription.dropped
    return encoded, dropped
end

function _recording_state(app)
    state = Base.@lock app.lock begin
        job = app.recording
        (;
            job.active, source = isnothing(job.source) ? nothing : string(job.source),
            field = isnothing(job.field) ? nothing : string(job.field), job.path,
            job.encoded, job.dropped, status = string(job.status), job.error, handle = job.handle,
        )
    end
    encoded, dropped = state.encoded, state.dropped
    if !isnothing(state.handle)
        encoded, dropped = _live_counts(state.handle)
    end
    return (;
        state.active, state.source, state.field, state.path, encoded, dropped,
        state.status, state.error,
    )
end

function _registered_name(text, names, description)
    text isa AbstractString || throw(ArgumentError("$description must be a string"))
    index = findfirst(name -> string(name) == text, names)
    isnothing(index) && throw(ArgumentError("unknown $description"))
    return names[index]
end

function _validate_recording(app, source, path, field, diagnostics)
    source isa AbstractString && source in ("stored", "live") ||
        throw(ArgumentError("recording source must be stored or live"))
    path isa AbstractString && !isempty(path) && !occursin('\0', path) ||
        throw(ArgumentError("recording path must be a nonempty filename"))
    lowercase(last(splitext(path))) in (".gif", ".mp4", ".mkv", ".webm") ||
        throw(ArgumentError("recording path must end in .gif, .mp4, .mkv, or .webm"))
    names = TwoPhaseAMR.field_names(app.recorder.registry)
    name = _registered_name(field, names, "field")
    diagnostics isa AbstractVector || throw(ArgumentError("diagnostics must be an array of names"))
    available = isnothing(app.recorder.diagnostics) ? Symbol[] :
        TwoPhaseAMR.diagnostic_names(app.recorder.diagnostics)
    selected = Symbol[_registered_name(item, available, "diagnostic") for item in diagnostics]
    length(unique(selected)) == length(selected) || throw(ArgumentError("duplicate diagnostics"))
    if source == "live" && TwoPhaseAMR.run_state(app.recorder) in (:completed, :failed)
        throw(ArgumentError("a live recording needs a new or running recorder"))
    end
    return Symbol(source), String(path), name, selected
end

# Freeze a stored session's selected keyframes and detached diagnostics. The
# resulting ordinary SnapshotStore is consumed by the phase-2 implementation.
function _stored_snapshot(store, diagnostics, index, diagnostic_counts)
    snapshot = TwoPhaseAMR.SnapshotStore(store.retention)
    for info in index
        frame = store[info.id] # keyframes cannot be thinned
        # Promotion can replace its metadata after stop; keep the frozen index.
        frame = frame.info === info ? frame : TwoPhaseAMR._with_info(frame, info)
        push!(snapshot.records, info)
        snapshot.frames[frame.id] = frame
    end
    for (name, count) in zip(diagnostics, diagnostic_counts)
        history = TwoPhaseAMR.diagnostic_history(store, name)
        count = min(count, length(history.times))
        resize!(history.times, count)
        resize!(history.steps, count)
        mirror = TwoPhaseAMR.MirroredDiagnostic(
            name, history.info, nothing,
            count, history.components, history.times, history.steps,
            [history.values[1:count, j] for j in axes(history.values, 2)]
        )
        push!(snapshot.diagnostics, mirror)
        snapshot.diagnostic_index[name] = length(snapshot.diagnostics)
    end
    return snapshot
end

function _job_set!(
        app, job; active = nothing, status = nothing,
        error = nothing, encoded = nothing, dropped = nothing
    )
    Base.@lock app.lock begin
        isnothing(active) || (job.active = active)
        isnothing(status) || (job.status = status)
        isnothing(error) || (job.error = error)
        isnothing(encoded) || (job.encoded = encoded)
        isnothing(dropped) || (job.dropped = dropped)
    end
    return nothing
end

function _detach_subscription(store, subscription)
    Base.@lock store.lock filter!(candidate -> candidate !== subscription, store.subscriptions)
    return nothing
end

function _unsubscribe(store, subscription)
    close(subscription)
    _detach_subscription(store, subscription)
    return nothing
end

# The phase-2 single-thread encoder waits for the solver to finish only while
# its subscription is open. Detach it at stop, but keep that bounded queue
# open until the run finishes; closing it early would start encoding on the
# solver's sole thread. The app reports :stopping until finalization finishes.
function _stop_live_subscription!(app, handle)
    subscription = getfield(handle, :subscription)
    single = Threads.nthreads(:interactive) == 0 && Threads.nthreads(:default) == 1
    if single && TwoPhaseAMR.run_state(app.recorder) in (:running, :pause_requested, :paused, :stopping)
        _detach_subscription(app.store, subscription)
        Threads.@spawn :default begin
            while isopen(subscription) &&
                    TwoPhaseAMR.run_state(app.recorder) in (:running, :pause_requested, :paused, :stopping)
                sleep(0.05)
            end
            close(subscription)
        end
    else
        _unsubscribe(app.store, subscription)
    end
    return nothing
end

function _record_job!(app, job)
    handle = nothing
    try
        if job.source === :stored
            # A stored session selects retained keyframes up to the stop
            # command; it does not depend on any browser's playhead.
            while !(Base.@lock app.lock job.stop_requested)
                sleep(0.05)
            end
            # Without an independent thread defer rendering until the solver
            # is idle. The selected frame index is already frozen at stop.
            if Threads.nthreads(:interactive) == 0 && Threads.nthreads(:default) == 1
                while TwoPhaseAMR.run_state(app.recorder) in (:running, :pause_requested, :paused, :stopping)
                    sleep(0.05)
                end
            end
        elseif Base.@lock app.lock job.stop_requested
            _job_set!(
                app, job; active = false, status = :error,
                error = "live recording stopped before its subscription was ready"
            )
            return nothing
        end
        diagnostics = isempty(job.diagnostics) ? nothing : job.diagnostics
        if job.source === :stored
            selection, counts = Base.@lock app.lock (job.selection, job.diagnostic_counts)
            snapshot = _stored_snapshot(app.store, job.diagnostics, selection, counts)
            count = length(snapshot.records)
            Base.invokelatest(
                TwoPhaseAMR.export_animation, job.path, snapshot;
                field = job.field, source = :stored, diagnostics
            )
            _job_set!(app, job; active = false, status = :completed, encoded = count)
        else
            # Phase 2 chooses the *opposite* pool to its caller. Start the
            # live handle on :default so it uses :interactive when available.
            starter = Threads.@spawn :default Base.invokelatest(
                TwoPhaseAMR.export_animation, job.path, app.recorder;
                field = job.field, source = :live, diagnostics,
            )
            handle = fetch(starter)
            stopped = Base.@lock app.lock begin
                job.handle = handle
                job.status = job.stop_requested ? :stopping : :recording
                job.stop_requested
            end
            stopped && _stop_live_subscription!(app, handle)
            wait(getfield(handle, :task)::Task)
            # Flush/finalize the phase-2 stream and preserve its accounting.
            Base.invokelatest(close, handle)
            _unsubscribe(app.store, getfield(handle, :subscription))
            encoded, dropped = _live_counts(handle)
            encoded > 0 || throw(ArgumentError("live recording produced no frames"))
            _job_set!(app, job; active = false, status = :completed, encoded, dropped)
        end
    catch err
        if !isnothing(handle)
            _unsubscribe(app.store, getfield(handle, :subscription))
        end
        message = sprint(showerror, err)
        encoded, dropped = isnothing(handle) ? (0, 0) : _live_counts(handle)
        _job_set!(app, job; active = false, status = :error, error = message, encoded, dropped)
    end
    return nothing
end

function _start_recording!(app, source, path, field, diagnostics)
    source, path, field, diagnostics = _validate_recording(app, source, path, field, diagnostics)
    job = RecordingJob(
        true, source, field, path, diagnostics,
        source === :stored ? :recording : :starting, nothing, 0, 0,
        false, nothing, nothing, nothing, nothing
    )
    Base.@lock app.lock begin
        app.status === :started || throw(ArgumentError("Studio is closed"))
        app.recording.active && throw(ArgumentError("a recording is already active"))
        app.recording = job
        # Run stored rendering away from the HTTP handler. On a one-default-
        # thread configuration prefer the independent pool, if available.
        if Threads.nthreads(:interactive) > 0 &&
                (source === :live || Threads.nthreads(:default) == 1)
            job.task = Threads.@spawn :interactive _record_job!(app, job)
        else
            job.task = Threads.@spawn :default _record_job!(app, job)
        end
    end
    return nothing
end

function _stop_recording!(app; closing = false)
    job = Base.@lock app.lock begin
        job = app.recording
        if !job.active || job.stop_requested
            closing && return nothing
            throw(ArgumentError("no recording is accepting a stop command"))
        end
        job
    end
    # Capture the immutable index reference and diagnostic lengths together;
    # filter outside the store lock. Later solver samples cannot leak into the
    # diagnostic plots of this stored export.
    selection, counts = if job.source === :stored
        records, lengths = Base.@lock app.store.lock begin
            store = app.store
            counts = map(job.diagnostics) do name
                index = get(store.diagnostic_index, name, 0)
                iszero(index) ? 0 : length(store.diagnostics[index].times)
            end
            (store.records, counts)
        end
        (filter(info -> info.kind !== :preview, records), lengths)
    else
        (nothing, nothing)
    end
    handle = Base.@lock app.lock begin
        if app.recording !== job || !job.active || job.stop_requested
            closing && return nothing
            throw(ArgumentError("recording already stopped"))
        end
        job.selection = selection
        job.diagnostic_counts = counts
        job.stop_requested = true
        job.status = :stopping
        job.handle
    end
    isnothing(handle) || _stop_live_subscription!(app, handle)
    return nothing
end
