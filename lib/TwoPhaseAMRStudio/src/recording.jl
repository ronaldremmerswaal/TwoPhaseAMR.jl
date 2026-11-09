# A job is owned by the server, never by a browser connection. All its fields
# except the phase-2 handle's own atomic counters are protected by app.lock.
mutable struct RecordingJob
    active::Bool
    id::Union{Nothing, String}
    source::Union{Nothing, Symbol}
    field::Union{Nothing, Symbol}
    path::Union{Nothing, String}
    diagnostics::Vector{Symbol}
    options::NamedTuple
    status::Symbol
    error::Union{Nothing, String}
    encoded::Int
    dropped::Int
    skipped::Int
    total::Union{Nothing, Int}
    stop_requested::Bool
    cancel_requested::Bool
    selection::Union{Nothing, Vector{TwoPhaseAMR.Frame}}
    diagnostic_counts::Union{Nothing, Vector{Int}}
    handle::Any
    task::Union{Nothing, Task}
end

RecordingJob() = RecordingJob(
    false, nothing, nothing, nothing, nothing, Symbol[], (;), :idle,
    nothing, 0, 0, 0, nothing, false, false, nothing, nothing, nothing, nothing
)

# The phase-2 handle's dropped count is a virtual property. Read the
# subscription directly so accounting is safe across extension world ages.
function _live_counts(handle)
    subscription = getfield(handle, :subscription)
    encoded = getfield(handle, :encoded, :sequentially_consistent)
    skipped = getfield(handle, :skipped, :sequentially_consistent)
    dropped = Base.@lock subscription.condition subscription.dropped
    return encoded, dropped, skipped
end

function _recording_state(app)
    state = Base.@lock app.lock begin
        job = app.recording
        (;
            job.active, job.id, source = isnothing(job.source) ? nothing : string(job.source),
            field = isnothing(job.field) ? nothing : string(job.field), job.path,
            job.encoded, job.dropped, job.skipped, job.total,
            status = string(job.status), job.error, handle = job.handle,
            url = job.status === :completed ? "/download/$(job.id)?token=$(app.token)" : nothing,
        )
    end
    encoded, dropped, skipped = state.encoded, state.dropped, state.skipped
    if !isnothing(state.handle)
        encoded, dropped, skipped = _live_counts(state.handle)
    end
    # This is frame-submission progress, not file-finalization progress.
    progress = state.status == "completed" ? 1.0 :
        state.source == "stored" && !isnothing(state.total) && state.total > 0 ?
        encoded / state.total : nothing
    return (;
        state.active, state.id, state.source, state.field, state.path, encoded, dropped, skipped, state.total,
        state.status, state.error, progress, state.url,
    )
end

function _registered_name(text, names, description)
    text isa AbstractString || throw(ArgumentError("$description must be a string"))
    index = findfirst(name -> string(name) == text, names)
    isnothing(index) && throw(ArgumentError("unknown $description"))
    return names[index]
end

function _number(value, name)
    value isa Real && !(value isa Bool) && isfinite(value) ||
        throw(ArgumentError("$name must be a finite number"))
    result = Float64(value)
    isfinite(result) || throw(ArgumentError("$name must be finite"))
    return result
end

function _numeric_tuple(value, count, name)
    value isa AbstractVector && length(value) == count ||
        throw(ArgumentError("$name must be an array of $count numbers"))
    return Tuple(_number(item, name) for item in value)
end

function _positive_integer(value, name)
    value isa Integer && !(value isa Bool) && 1 <= value <= typemax(Int) ||
        throw(ArgumentError("$name must be a positive integer"))
    return Int(value)
end

function _validate_recording(app, message, one_shot)
    allowed = (
        "type", "protocol", "request", "action", "source", "path", "field", "diagnostics",
        "frames", "window", "every", "capacity", "camera", "colorrange", "colormap",
        "phase_rendering", "size", "framerate", "theme", "arrangement",
        "interface", "geometry", "blocks",
    )
    for key in keys(message)
        key in allowed || throw(ArgumentError("unsupported recording option: $key"))
    end
    source = get(message, "source", one_shot ? "stored" : nothing)
    source isa AbstractString && source in ("stored", "live") ||
        throw(ArgumentError("recording source must be stored or live"))
    one_shot && source != "stored" &&
        throw(ArgumentError("one-shot export requires source stored"))
    path = get(message, "path", nothing)
    path isa AbstractString && !isempty(path) && !occursin('\0', path) ||
        throw(ArgumentError("recording path must be a nonempty filename"))
    lowercase(last(splitext(path))) in (".gif", ".mp4", ".mkv", ".webm") ||
        throw(ArgumentError("recording path must end in .gif, .mp4, .mkv, or .webm"))
    names = TwoPhaseAMR.field_names(app.recorder.registry)
    name = _registered_name(get(message, "field", nothing), names, "field")
    diagnostics = get(message, "diagnostics", Any[])
    diagnostics isa AbstractVector || throw(ArgumentError("diagnostics must be an array of names"))
    available = isnothing(app.recorder.diagnostics) ? Symbol[] :
        TwoPhaseAMR.diagnostic_names(app.recorder.diagnostics)
    selected = Symbol[_registered_name(item, available, "diagnostic") for item in diagnostics]
    length(unique(selected)) == length(selected) || throw(ArgumentError("duplicate diagnostics"))
    if source == "live" && TwoPhaseAMR.run_state(app.recorder) in (:completed, :failed)
        throw(ArgumentError("a live recording needs a new or running recorder"))
    end

    frames = get(message, "frames", "keyframes")
    frames in ("keyframes", "all") || throw(ArgumentError("frames must be keyframes or all"))
    source == "live" && haskey(message, "frames") &&
        throw(ArgumentError("frames selection applies only to stored recordings"))
    window = get(message, "window", nothing)
    if !isnothing(window)
        source == "stored" || throw(ArgumentError("window applies only to stored recordings"))
        window = _numeric_tuple(window, 2, "window")
        window[1] <= window[2] || throw(ArgumentError("window must have increasing bounds"))
    end
    every = _positive_integer(get(message, "every", 1), "every")
    capacity = _positive_integer(get(message, "capacity", 32), "capacity")
    source == "stored" && haskey(message, "capacity") &&
        throw(ArgumentError("capacity applies only to live recordings"))
    camera = get(message, "camera", "fit")
    if camera == "fit"
        camera = :fit
    else
        camera = _numeric_tuple(camera, 4, "camera")
        camera[1] < camera[2] && camera[3] < camera[4] ||
            throw(ArgumentError("camera limits must have positive extent"))
    end
    colorrange = get(message, "colorrange", nothing)
    if isnothing(colorrange) || colorrange == "auto"
        colorrange = nothing
    elseif colorrange == "global"
        source == "stored" || throw(ArgumentError("global colorrange requires stored frames"))
        colorrange = :global
    else
        colorrange = _numeric_tuple(colorrange, 2, "colorrange")
        colorrange[1] < colorrange[2] || throw(ArgumentError("colorrange must increase"))
    end
    colormap = get(message, "colormap", nothing)
    if !isnothing(colormap)
        colormap isa AbstractString && !isempty(colormap) ||
            throw(ArgumentError("colormap must be a nonempty name"))
        colormap = Symbol(colormap)
    end
    phase = get(message, "phase_rendering", "cell")
    phase in ("cell", "plic") || throw(ArgumentError("phase_rendering must be cell or plic"))
    size = get(message, "size", nothing)
    if !isnothing(size)
        size isa AbstractVector && length(size) == 2 ||
            throw(ArgumentError("size must be an array of two positive integers"))
        size = (_positive_integer(size[1], "width"), _positive_integer(size[2], "height"))
        all(iseven, size) ||
            throw(ArgumentError("size width and height must be even to avoid encoder padding"))
    end
    # Makie rounds noninteger frame rates; do not silently change a client's FPS.
    framerate = _positive_integer(get(message, "framerate", 12), "framerate")
    theme = get(message, "theme", nothing)
    if !isnothing(theme)
        theme in ("light", "dark") || throw(ArgumentError("theme must be light or dark"))
        source == "stored" || throw(ArgumentError("theme is not supported for live recording"))
        theme = Symbol(theme)
    end
    arrangement = get(message, "arrangement", "right")
    arrangement in ("right", "below") ||
        throw(ArgumentError("arrangement must be right or below"))
    interface = get(message, "interface", true)
    geometry = get(message, "geometry", true)
    blocks = get(message, "blocks", false)
    all(value -> value isa Bool, (interface, geometry, blocks)) ||
        throw(ArgumentError("interface, geometry, and blocks must be booleans"))
    options = (;
        frames = Symbol(frames), window, every, capacity, camera, colorrange, colormap,
        phase_rendering = Symbol(phase), size, framerate, theme, arrangement = Symbol(arrangement),
        interface, geometry, blocks,
    )
    return Symbol(source), abspath(path), name, selected, options
end

# Pin selected immutable frame objects at the command boundary. In particular,
# retaining just preview ids would race store thinning before the encoder runs.
function _freeze_selection(store, diagnostics, options)
    selected, counts = Base.@lock store.lock begin
        selected = TwoPhaseAMR.Frame[]
        for info in store.records
            options.frames === :keyframes && info.kind === :preview && continue
            if !isnothing(options.window) && !(options.window[1] <= info.time <= options.window[2])
                continue
            end
            push!(selected, store.frames[info.id])
        end
        counts = map(diagnostics) do name
            index = get(store.diagnostic_index, name, 0)
            iszero(index) ? 0 : length(store.diagnostics[index].times)
        end
        (selected, counts)
    end
    return selected[1:options.every:end], counts
end

# Phase 2 selects only keyframes. For an explicit "all" export, promote
# previews in this private store only; the original store is never modified.
function _stored_snapshot(store, diagnostics, selection, diagnostic_counts)
    snapshot = TwoPhaseAMR.SnapshotStore(store.retention)
    for frame in selection
        info = frame.info
        if info.kind === :preview
            info = TwoPhaseAMR.FrameInfo(
                info.id, info.time, info.step, :keyframe, info.wall_time,
                info.topology, info.requested_times,
            )
            frame = TwoPhaseAMR._with_info(frame, info)
        end
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

# Explicit global colour policy overrides a registered fixed range, using the
# same symmetry, diverging-map and constant-field rules as the phase-2 plot.
function _export_options(job)
    options = job.options
    colorrange = options.colorrange
    if colorrange === :global
        selection = job.selection
        isempty(selection) && throw(ArgumentError("no stored frames match the selection"))
        extension = Base.get_extension(TwoPhaseAMR, :TwoPhaseAMRMakieExt)
        isnothing(extension) && throw(ArgumentError("the Makie export extension is not loaded"))
        first_frame = first(selection)
        entry = TwoPhaseAMR._frame_entry(first_frame, job.field)
        stats = mapreduce(frame -> frame.stats[job.field], merge, selection)
        colormap = Base.invokelatest(extension._resolve_colormap, entry, options.colormap)
        symmetric = Base.invokelatest(extension._symmetric, entry, colormap)
        colorrange = Base.invokelatest(extension._automatic_colorrange, stats, symmetric)
    end
    kwargs = (;
        diagnostics = isempty(job.diagnostics) ? nothing : job.diagnostics,
        options.camera, colorrange, options.colormap, options.phase_rendering,
        options.framerate, options.arrangement, options.interface, options.geometry, options.blocks,
    )
    isnothing(options.size) || (kwargs = merge(kwargs, (; options.size)))
    if options.theme === :dark
        kwargs = merge(kwargs, (; interfacecolor = :white, blockcolor = (:white, 0.2)))
    end
    if job.source === :live
        kwargs = merge(kwargs, (; options.every, options.capacity))
    end
    return kwargs
end

function _complete_job!(app, job, encoded, dropped, skipped)
    isfile(job.path) && filesize(job.path) > 0 ||
        throw(ArgumentError("the encoder did not write an output file"))
    Base.@lock app.lock begin
        app.downloads[job.id] = job.path
        job.encoded = encoded
        job.dropped = dropped
        job.skipped = skipped
        # Release pinned previews as soon as encoding finishes.
        job.selection = nothing
        job.diagnostic_counts = nothing
        job.handle = nothing
        job.task = nothing
        job.active = false
        job.status = :completed
    end
    return nothing
end

function _job_set!(
        app, job; active = nothing, status = nothing,
        error = nothing, encoded = nothing, dropped = nothing, skipped = nothing
    )
    Base.@lock app.lock begin
        isnothing(active) || (job.active = active)
        isnothing(status) || (job.status = status)
        isnothing(error) || (job.error = error)
        isnothing(encoded) || (job.encoded = encoded)
        isnothing(dropped) || (job.dropped = dropped)
        isnothing(skipped) || (job.skipped = skipped)
        if active === false
            job.selection = nothing
            job.diagnostic_counts = nothing
            job.handle = nothing
            job.task = nothing
        end
    end
    return nothing
end

_cancel_requested(app, job) = Base.@lock app.lock job.cancel_requested

function _finish_cancelled!(app, job)
    _job_set!(app, job; active = false, status = :cancelled)
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
            # A stored session freezes its selection at stop; a one-shot job
            # froze it at submission. Neither consults a browser playhead.
            while !(Base.@lock app.lock job.stop_requested)
                sleep(0.05)
            end
            if _cancel_requested(app, job)
                return _finish_cancelled!(app, job)
            end
            # Without an independent thread defer rendering until the solver
            # is idle. The selected frame index is already frozen at stop.
            if Threads.nthreads(:interactive) == 0 && Threads.nthreads(:default) == 1
                while TwoPhaseAMR.run_state(app.recorder) in (:running, :pause_requested, :paused, :stopping)
                    _cancel_requested(app, job) && return _finish_cancelled!(app, job)
                    sleep(0.05)
                end
            end
            _cancel_requested(app, job) && return _finish_cancelled!(app, job)
        elseif Base.@lock app.lock job.stop_requested
            _job_set!(
                app, job; active = false, status = :error,
                error = "live recording stopped before its subscription was ready"
            )
            return nothing
        end
        kwargs = _export_options(job)
        if job.source === :stored
            selection, counts = Base.@lock app.lock (job.selection, job.diagnostic_counts)
            isempty(selection) && throw(ArgumentError("no stored frames match the selection"))
            snapshot = _stored_snapshot(app.store, job.diagnostics, selection, counts)
            count = length(snapshot.records)
            ready = Base.@lock app.lock begin
                if job.cancel_requested
                    false
                else
                    job.status = :exporting
                    true
                end
            end
            ready || return _finish_cancelled!(app, job)
            progress(index, total) = Base.@lock app.lock begin
                job.encoded = index
                job.total = total
            end
            cancel() = _cancel_requested(app, job)
            render() = Base.invokelatest(
                TwoPhaseAMR.export_animation, job.path, snapshot;
                field = job.field, source = :stored, progress, cancel, kwargs...,
            )
            result = if isnothing(job.options.theme)
                render()
            elseif job.options.theme === :dark
                Base.invokelatest(CairoMakie.with_theme, render, CairoMakie.theme_dark(); textcolor = :gray85)
            else
                Base.invokelatest(CairoMakie.with_theme, render, CairoMakie.Theme())
            end
            isnothing(result) ? _finish_cancelled!(app, job) : _complete_job!(app, job, count, 0, 0)
        else
            # Phase 2 chooses the *opposite* pool to its caller. Start the
            # live handle on :default so it uses :interactive when available.
            starter = Threads.@spawn :default Base.invokelatest(
                TwoPhaseAMR.export_animation, job.path, app.recorder;
                field = job.field, source = :live, kwargs...,
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
            encoded, dropped, skipped = _live_counts(handle)
            encoded > 0 || throw(ArgumentError("live recording produced no frames"))
            _complete_job!(app, job, encoded, dropped, skipped)
        end
    catch err
        if !isnothing(handle)
            _unsubscribe(app.store, getfield(handle, :subscription))
        end
        message = sprint(showerror, err)
        if isnothing(handle)
            _job_set!(app, job; active = false, status = :error, error = message)
        else
            encoded, dropped, skipped = _live_counts(handle)
            _job_set!(app, job; active = false, status = :error, error = message, encoded, dropped, skipped)
        end
    end
    return nothing
end

function _start_recording!(app, message; one_shot = false)
    source, path, field, diagnostics, options = _validate_recording(app, message, one_shot)
    # Reject a busy server before pinning a potentially large stored selection.
    # Recheck under the lock when installing the job, since another client can
    # start one while the selection is being captured.
    Base.@lock app.lock begin
        app.status === :started || throw(ArgumentError("Studio is closed"))
        app.recording.active && throw(ArgumentError("a recording is already active"))
    end
    selection, counts = if one_shot
        selection, counts = _freeze_selection(app.store, diagnostics, options)
        isempty(selection) && throw(ArgumentError("no stored frames match the selection"))
        (selection, counts)
    else
        (nothing, nothing)
    end
    id = bytes2hex(Random.rand(Random.RandomDevice(), UInt8, 16))
    job = RecordingJob(
        true, id, source, field, path, diagnostics, options,
        one_shot ? :starting : source === :stored ? :recording : :starting,
        nothing, 0, 0, 0, isnothing(selection) ? nothing : length(selection),
        one_shot, false, selection, counts, nothing, nothing,
    )
    Base.@lock app.lock begin
        app.status === :started || throw(ArgumentError("Studio is closed"))
        app.recording.active && throw(ArgumentError("a recording is already active"))
        # A new write can replace the same path; never serve an older job id
        # against a partially overwritten or subsequently changed result.
        filter!(pair -> last(pair) != path, app.downloads)
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
    # Pin frame references and diagnostic lengths together at stop. Later
    # thinning, promotions and samples cannot change the exported selection.
    selection, counts = job.source === :stored ?
        _freeze_selection(app.store, job.diagnostics, job.options) : (nothing, nothing)
    handle = Base.@lock app.lock begin
        if app.recording !== job || !job.active || job.stop_requested
            closing && return nothing
            throw(ArgumentError("recording already stopped"))
        end
        job.selection = selection
        job.diagnostic_counts = counts
        job.total = isnothing(selection) ? nothing : length(selection)
        job.stop_requested = true
        job.status = :stopping
        job.handle
    end
    isnothing(handle) || _stop_live_subscription!(app, handle)
    return nothing
end

function _cancel_recording!(app)
    Base.@lock app.lock begin
        job = app.recording
        job.active || throw(ArgumentError("no recording is active"))
        job.source === :stored || throw(ArgumentError("live recordings must be stopped, not cancelled"))
        job.cancel_requested && throw(ArgumentError("cancellation already requested"))
        job.cancel_requested = true
        job.stop_requested = true
        job.status = :cancelling
    end
    return nothing
end
