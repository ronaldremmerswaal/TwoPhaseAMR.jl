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
    total::Union{Nothing, Int}
    stop_requested::Bool
    cancel_requested::Bool
    selection::Union{Nothing, Vector{TwoPhaseAMR.Frame}}
    diagnostic_counts::Union{Nothing, Vector{Int}}
    handle::Any
    times::Union{Nothing, Vector{Float64}}
    task::Union{Nothing, Task}
end

RecordingJob() = RecordingJob(
    false, nothing, nothing, nothing, nothing, Symbol[], (;), :idle,
    nothing, 0, 0, nothing, false, false, nothing, nothing, nothing, nothing, nothing
)

# CairoMakie previews must not overlap Studio encoders (including jobs owned
# by another Studio instance), or one another. A reservation covers startup
# through task completion, without delaying a live frame subscription.
const _RENDER_GATE_LOCK = ReentrantLock()
const _RENDER_RECORDINGS = Ref(0)
const _RENDER_PREVIEW = Ref(false)

function _reserve_recording_render!()
    Base.@lock _RENDER_GATE_LOCK begin
        _RENDER_PREVIEW[] && throw(ArgumentError("cannot start a recording while a preview is rendering"))
        _RENDER_RECORDINGS[] += 1
    end
    return nothing
end

function _release_recording_render!()
    Base.@lock _RENDER_GATE_LOCK _RENDER_RECORDINGS[] -= 1
    return nothing
end

function _check_preview_render_available()
    _RENDER_RECORDINGS[] > 0 &&
        throw(ArgumentError("preview unavailable while a recording or encoding is active"))
    _RENDER_PREVIEW[] && throw(ArgumentError("another preview is already rendering"))
    return nothing
end

function _reserve_preview_render!()
    Base.@lock _RENDER_GATE_LOCK begin
        _check_preview_render_available()
        _RENDER_PREVIEW[] = true
    end
    return nothing
end

function _release_preview_render!()
    Base.@lock _RENDER_GATE_LOCK _RENDER_PREVIEW[] = false
    return nothing
end

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
            job.active, job.id, source = isnothing(job.source) ? nothing : string(job.source),
            field = isnothing(job.field) ? nothing : string(job.field), job.path,
            job.encoded, job.dropped, job.total, job.times,
            status = string(job.status), job.error, handle = job.handle,
            url = job.status === :completed ? "/download/$(job.id)?token=$(app.token)" : nothing,
        )
    end
    encoded, dropped = state.encoded, state.dropped
    if !isnothing(state.handle)
        encoded, dropped = _live_counts(state.handle)
    end
    # This is frame-submission progress, not file-finalization progress.
    progress = state.status == "completed" ? 1.0 :
        state.source == "stored" && !isnothing(state.total) && state.total > 0 ?
        encoded / state.total : nothing
    return (;
        state.active, state.id, state.source, state.field, state.path, encoded, dropped, state.total,
        state.status, state.error, progress, state.url, times = state.times,
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

function _diagnostic_component_counts(app, selected, infos)
    collection = app.recorder.diagnostics
    return Dict(
        name => begin
            count = Base.@lock app.store.lock begin
                index = get(app.store.diagnostic_index, name, 0)
                iszero(index) ? 0 : length(app.store.diagnostics[index].columns)
            end
            if iszero(count)
                series = collection.series[collection.indices[name]]
                sample = findfirst(value -> TwoPhaseAMR._value_components(value) > 0, series.values)
                count = isnothing(sample) ? max(length(infos[name].components), 1) :
                    TwoPhaseAMR._value_components(series.values[sample])
            end
            count
        end for name in selected
    )
end

function _diagnostic_options(app, value, selected)
    value isa AbstractVector || throw(ArgumentError("diagnostic_options must be an array"))
    configs = NamedTuple{
        (:names, :scale, :hidden), Tuple{
            Vector{Symbol}, Symbol, Vector{NamedTuple{(:name, :component), Tuple{Symbol, Int}}},
        },
    }[]
    for item in value
        item isa AbstractDict && Set(keys(item)) == Set(("names", "scale", "hidden")) ||
            throw(ArgumentError("each diagnostic option needs names, scale, and hidden"))
        names = item["names"]
        names isa AbstractVector && !isempty(names) ||
            throw(ArgumentError("diagnostic option names must be a nonempty array"))
        names = Symbol[_registered_name(name, selected, "diagnostic") for name in names]
        scale = item["scale"]
        scale isa AbstractString && scale in ("linear", "log") ||
            throw(ArgumentError("diagnostic option scale must be linear or log"))
        entries = item["hidden"]
        entries isa AbstractVector || throw(ArgumentError("diagnostic option hidden must be an array"))
        hidden = NamedTuple{(:name, :component), Tuple{Symbol, Int}}[]
        for entry in entries
            entry isa AbstractDict && Set(keys(entry)) == Set(("name", "component")) ||
                throw(ArgumentError("each hidden component needs name and component"))
            name = _registered_name(entry["name"], names, "hidden diagnostic")
            component = _positive_integer(entry["component"], "component")
            push!(hidden, (; name, component))
        end
        push!(configs, (; names, scale = Symbol(scale), hidden))
    end
    isempty(configs) && return configs
    infos = Dict(TwoPhaseAMR.diagnostic_info(app.recorder.diagnostics))
    groups = TwoPhaseAMR.diagnostic_groups(
        Pair{Symbol, TwoPhaseAMR.DiagnosticInfo}[name => infos[name] for name in selected],
    )
    counts = _diagnostic_component_counts(app, selected, infos)
    return TwoPhaseAMR._validated_diagnostic_options(configs, groups, counts)
end

function _validate_recording(app, message, one_shot)
    allowed = (
        "type", "protocol", "request", "action", "source", "path", "field", "diagnostics", "diagnostic_options",
        "frames", "window", "duration", "fps", "sim_per_second", "capacity", "camera", "colorrange", "colormap",
        "phase_rendering", "size", "theme", "arrangement",
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
    one_shot || source == "live" ||
        throw(ArgumentError("stored frames export in one shot; send action export"))
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
    diagnostic_options = haskey(message, "diagnostic_options") ?
        _diagnostic_options(app, message["diagnostic_options"], selected) : nothing
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
    duration = get(message, "duration", nothing)
    if !isnothing(duration)
        source == "stored" || throw(ArgumentError("duration applies only to stored recordings"))
        duration = _number(duration, "duration")
        duration > 0 || throw(ArgumentError("duration must be positive"))
    end
    source == "stored" && isnothing(duration) && (duration = 2.0)
    # Makie rounds noninteger frame rates; do not silently change a client's FPS.
    fps = _positive_integer(get(message, "fps", 20), "fps")
    sim_per_second = get(message, "sim_per_second", nothing)
    if !isnothing(sim_per_second)
        source == "live" || throw(ArgumentError("sim_per_second applies only to live recordings"))
        sim_per_second = _number(sim_per_second, "sim_per_second")
        sim_per_second > 0 || throw(ArgumentError("sim_per_second must be positive"))
    end
    source == "live" && isnothing(sim_per_second) && (sim_per_second = 1.0)
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
    theme = get(message, "theme", nothing)
    if !isnothing(theme)
        theme in ("light", "dark") || throw(ArgumentError("theme must be light or dark"))
        # The live figure is created lazily on the encoder task, outside the
        # caller's with_theme block, so only stored exports can apply one.
        source == "stored" || throw(
            ArgumentError("theme applies only to stored exports; the live figure is built while encoding"),
        )
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
        frames = Symbol(frames), window, duration, fps, sim_per_second, capacity, camera, colorrange,
        colormap, phase_rendering = Symbol(phase), size, theme, arrangement = Symbol(arrangement),
        interface, geometry, blocks, diagnostic_options,
    )
    return Symbol(source), abspath(path), name, selected, options
end

# Pin selected immutable frame objects at the command boundary. In particular,
# retaining just preview ids would race store thinning before the encoder runs.
# `frames = :keyframes` excludes previews; the time-linear mapping only uses
# these candidates.
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
    return selected, counts
end

# The export reads a private snapshot of the frozen frames, so later thinning,
# promotion and diagnostic samples cannot change it.
function _stored_snapshot(store, diagnostics, selection, diagnostic_counts)
    snapshot = TwoPhaseAMR.SnapshotStore(store.retention)
    for frame in selection
        push!(snapshot.records, frame.info)
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
        diagnostics = isempty(job.diagnostics) ? nothing : job.diagnostics, options.diagnostic_options,
        options.camera, colorrange, options.colormap, options.phase_rendering,
        options.fps, options.arrangement, options.interface, options.geometry, options.blocks,
    )
    isnothing(options.size) || (kwargs = merge(kwargs, (; options.size)))
    if options.theme === :dark
        kwargs = merge(kwargs, (; interfacecolor = :white, blockcolor = (:white, 0.2)))
    end
    if job.source === :live
        kwargs = merge(
            kwargs, (; sim_per_second = options.sim_per_second, capacity = options.capacity),
        )
    else
        kwargs = merge(
            kwargs, (; window = options.window, frames = options.frames, duration = options.duration),
        )
    end
    return kwargs
end

# Render through the animation scene itself, so range selection, layout,
# cursors, and diagnostic options are identical to the stored export.
function _preview_image(snapshot, frame, job, directory)
    extension = Base.get_extension(TwoPhaseAMR, :TwoPhaseAMRMakieExt)
    isnothing(extension) && throw(ArgumentError("the Makie export extension is not loaded"))
    options = _export_options(job)
    panels = (;
        (key => value for (key, value) in pairs(options)
         if !(key in (:fps, :duration, :sim_per_second, :capacity)))...
    )
    render() = begin
        animation = Base.invokelatest(extension._animation_scene, snapshot, job.field; panels...)
        Base.invokelatest(setindex!, animation.frame, frame)
        path = joinpath(directory, "preview.png")
        Base.invokelatest(CairoMakie.save, path, animation.figure; px_per_unit = 1)
        return read(path)
    end
    if isnothing(job.options.theme)
        return render()
    elseif job.options.theme === :dark
        return Base.invokelatest(CairoMakie.with_theme, render, CairoMakie.theme_dark(); textcolor = :gray85)
    end
    return Base.invokelatest(CairoMakie.with_theme, render, CairoMakie.Theme())
end

function _cleanup_preview_directory!(app, directory)
    Base.@lock app.lock begin
        if directory in app.preview_dirs
            rm(directory; recursive = true, force = true)
            delete!(app.preview_dirs, directory)
        end
    end
    return nothing
end

"""
Compute the stored export's output-frame plan without starting a job: the
output frame count, the number of distinct candidate frames it shows, and the
uniform simulation-time spacing. The dialog uses it to show the resulting
video summary and warn about a sparse recording before exporting.
"""
function _recording_schedule(app, socket, message)
    request = _optional_request(message)
    isnothing(request) && throw(ArgumentError("schedule requires a request id"))
    command = Dict{String, Any}(key => value for (key, value) in pairs(message))
    delete!(command, "action")
    get!(command, "path", "schedule.gif") # Validated but never written.
    _, _, _, _, options = _validate_recording(app, command, true)
    ids = Int[]
    times = Float64[]
    Base.@lock app.store.lock begin
        for info in app.store.records
            options.frames === :keyframes && info.kind === :preview && continue
            if !isnothing(options.window) && !(options.window[1] <= info.time <= options.window[2])
                continue
            end
            push!(ids, info.id)
            push!(times, info.time)
        end
    end
    isempty(times) && throw(ArgumentError("no stored frames match the selection"))
    t_start, t_end = isnothing(options.window) ? (first(times), last(times)) :
        (options.window[1], options.window[2])
    schedule = TwoPhaseAMR.video_frame_schedule(
        times, t_start, t_end; fps = options.fps, duration = options.duration,
    )
    count = length(schedule)
    distinct = length(unique(schedule.indices))
    spacing = count > 1 ? (t_end - t_start) / (count - 1) : 0.0
    return _send(socket, (;
        type = "recording_schedule", request, frames = count, distinct,
        repeats = count - distinct, sparse = 2 * distinct < count,
        window = [t_start, t_end], fps = options.fps, duration = options.duration,
        spacing, sim_per_second = spacing * options.fps,
        first_frame = ids[first(schedule.indices)],
    ))
end

function _start_preview!(app, socket, message)
    request = _optional_request(message)
    isnothing(request) && throw(ArgumentError("preview requires a request id"))
    Base.@lock _RENDER_GATE_LOCK _check_preview_render_available()
    id = _integer_selector(message, "frame")
    frame = app.store[id]
    command = Dict{String, Any}(key => value for (key, value) in pairs(message))
    delete!(command, "frame")
    get!(command, "path", "preview.gif") # Validated but never written.
    source, _, field, diagnostics, options = _validate_recording(app, command, true)
    haskey(frame, field) || throw(ArgumentError("the preview frame has no selected field"))
    selection, counts = _freeze_selection(app.store, diagnostics, options)
    isempty(selection) && throw(ArgumentError("no stored frames match the selection"))
    snapshot = _stored_snapshot(app.store, diagnostics, selection, counts)
    job = RecordingJob()
    job.source = source
    job.field = field
    job.diagnostics = diagnostics
    job.options = options
    job.selection = selection
    generation, directory = Base.@lock app.lock begin
        app.status === :started || throw(ArgumentError("Studio is closed"))
        _reserve_preview_render!()
        directory = nothing
        try
            directory = mktempdir()
            push!(app.preview_dirs, directory)
            app.preview_generation += 1
            (app.preview_generation, directory)
        catch
            try
                if !isnothing(directory)
                    delete!(app.preview_dirs, directory)
                    rm(directory; recursive = true, force = true)
                end
            finally
                _release_preview_render!()
            end
            rethrow()
        end
    end
    try
        Threads.@spawn :default begin
            response = try
                image = _preview_image(snapshot, frame, job, directory)
                preview_id = bytes2hex(Random.rand(Random.RandomDevice(), UInt8, 16))
                published = Base.@lock app.lock begin
                    if app.status === :started && app.preview_generation == generation
                        app.preview = (preview_id, image)
                        true
                    else
                        false
                    end
                end
                published || throw(ArgumentError("preview superseded"))
                (; type = "recording_preview", request, url = "/preview/$preview_id?token=$(app.token)")
            catch err
                text = err isa ArgumentError || err isa KeyError ? sprint(showerror, err) : "preview rendering failed"
                (; type = "error", request, message = text)
            finally
                try
                    _cleanup_preview_directory!(app, directory)
                finally
                    _release_preview_render!()
                end
            end
            try
                _send(socket, response)
            catch
                # The client may have disconnected while rendering.
            end
        end
    catch
        try
            _cleanup_preview_directory!(app, directory)
        finally
            _release_preview_render!()
        end
        rethrow()
    end
    return nothing
end

function _complete_job!(app, job, encoded, dropped)
    isfile(job.path) && filesize(job.path) > 0 ||
        throw(ArgumentError("the encoder did not write an output file"))
    Base.@lock app.lock begin
        app.downloads[job.id] = job.path
        job.encoded = encoded
        job.dropped = dropped
        # Release pinned previews/keys as soon as encoding finishes.
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
        error = nothing, encoded = nothing, dropped = nothing
    )
    Base.@lock app.lock begin
        isnothing(active) || (job.active = active)
        isnothing(status) || (job.status = status)
        isnothing(error) || (job.error = error)
        isnothing(encoded) || (job.encoded = encoded)
        isnothing(dropped) || (job.dropped = dropped)
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
            _cancel_requested(app, job) && return _finish_cancelled!(app, job)
            # Without an independent thread defer rendering until the solver
            # is idle. The selected candidate frames are already frozen.
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
            count = TwoPhaseAMR.video_frame_count(job.options.fps, job.options.duration)
            ready = Base.@lock app.lock begin
                if job.cancel_requested
                    false
                else
                    job.status = :exporting
                    job.total = count
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
            isnothing(result) ? _finish_cancelled!(app, job) : _complete_job!(app, job, count, 0)
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
            encoded, dropped = _live_counts(handle)
            encoded > 0 || throw(ArgumentError("live recording produced no frames"))
            shown = copy(getfield(handle, :times))
            Base.@lock(app.lock, job.times = shown)
            _complete_job!(app, job, encoded, dropped)
        end
    catch err
        if !isnothing(handle)
            _unsubscribe(app.store, getfield(handle, :subscription))
        end
        message = sprint(showerror, err)
        if isnothing(handle)
            _job_set!(app, job; active = false, status = :error, error = message)
        else
            encoded, dropped = _live_counts(handle)
            _job_set!(app, job; active = false, status = :error, error = message, encoded, dropped)
        end
    finally
        _release_recording_render!()
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
    total = one_shot ? TwoPhaseAMR.video_frame_count(options.fps, options.duration) : nothing
    job = RecordingJob(
        true, id, source, field, path, diagnostics, options, :starting,
        nothing, 0, 0, total, false, false, selection, counts, nothing, nothing, nothing,
    )
    Base.@lock app.lock begin
        app.status === :started || throw(ArgumentError("Studio is closed"))
        app.recording.active && throw(ArgumentError("a recording is already active"))
        _reserve_recording_render!()
        previous = app.recording
        try
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
        catch
            app.recording = previous
            _release_recording_render!()
            rethrow()
        end
    end
    return nothing
end

function _stop_recording!(app; closing = false)
    job = Base.@lock app.lock app.recording
    if job.active && job.source === :stored
        closing || throw(ArgumentError("stored exports are cancelled, not stopped"))
        Base.@lock app.lock job.cancel_requested = true
        return nothing
    end
    job = Base.@lock app.lock begin
        job = app.recording
        if !job.active || job.stop_requested
            closing && return nothing
            throw(ArgumentError("no recording is accepting a stop command"))
        end
        job
    end
    handle = Base.@lock app.lock begin
        if app.recording !== job || !job.active || job.stop_requested
            closing && return nothing
            throw(ArgumentError("recording already stopped"))
        end
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
