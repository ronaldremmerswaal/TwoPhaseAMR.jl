const _ASSETS = joinpath(@__DIR__, "..", "assets")
const _ASSET_ROUTES = Dict(
    "/" => ("index.html", "text/html; charset=utf-8"),
    "/index.html" => ("index.html", "text/html; charset=utf-8"),
    "/studio.js" => ("studio.js", "text/javascript; charset=utf-8"),
    "/studio.css" => ("studio.css", "text/css; charset=utf-8"),
)

mutable struct Studio
    recorder::TwoPhaseAMR.Recorder
    store::TwoPhaseAMR.SnapshotStore
    name::String
    token::String
    host::String
    port::Int
    layout::Symbol
    theme::Symbol
    lock::ReentrantLock
    status::Symbol
    server::Any
    clients::IdDict{Any, Nothing}
    launchdir::Union{Nothing, String}
    recording::RecordingJob
    downloads::Dict{String, String}
    preview::Union{Nothing, Tuple{String, Vector{UInt8}}}
    preview_generation::Int
    preview_dirs::Set{String}
end

mutable struct StudioClient
    socket::Any
    subscription::TwoPhaseAMR.FrameSubscription
    active::Threads.Atomic{Bool}
end

# Avoid accidentally printing the credential in REPL displays or error logs.
function Base.show(io::IO, app::Studio)
    status = Base.@lock app.lock app.status
    return print(io, "Studio(", repr(app.host), ":", app.port, ", status=:", status, ")")
end

function _url_host(host)
    name = host == "0.0.0.0" ? "127.0.0.1" : host == "::" ? "::1" : host
    return occursin(':', name) ? "[$name]" : name
end

"""
    url(app) -> String

Return the Studio URL, including its secret token. Treat it as a credential:
anyone possessing it can read the run and issue solver and recording commands.
"""
url(app::Studio) = "http://$(_url_host(app.host)):$(app.port)/?token=$(app.token)"

function _launch_browser(app, address, mode)
    mode === :none && return nothing
    executable = if mode === :app
        found = nothing
        for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable")
            found = Sys.which(name)
            isnothing(found) || break
        end
        isnothing(found) && throw(ArgumentError("open=:app requires a Chromium-family browser"))
        found
    else
        found = Sys.isapple() ? Sys.which("open") : Sys.iswindows() ? nothing : Sys.which("xdg-open")
        isnothing(found) && throw(ArgumentError("no browser opener found; use open=:none and open the printed URL"))
        found
    end
    # Passing a tokenized URL in argv exposes it to other local users through
    # process listings. Open a private local redirect document instead.
    directory = mktempdir()
    Base.@lock app.lock app.launchdir = directory
    path = joinpath(directory, "studio.html")
    Base.open(path, "w") do io
        write(
            io, "<!doctype html><meta name=\"referrer\" content=\"no-referrer\">" *
                "<meta http-equiv=\"refresh\" content=\"0;url=$address\">"
        )
    end
    chmod(path, 0o600)
    if mode === :app
        fileurl = "file://" * HTTP.escapepath(path)
        run(pipeline(`$executable --app=$fileurl`, stdout = devnull, stderr = devnull); wait = false)
    else
        run(pipeline(`$executable $path`, stdout = devnull, stderr = devnull); wait = false)
    end
    return nothing
end

"""
    studio(recorder; name="Run 1", layout=:auto, host="127.0.0.1", port=0,
           open=:browser, theme=:system) -> Studio

Start a Studio server for `recorder`. The default binds only IPv4 loopback;
`port=0` asks the OS for an available port. The printed URL contains a fresh
128-bit cryptographically random token required on every HTTP and WebSocket
request. `name` labels the run. Set `open=:none` to avoid launching a browser,
or `open=:app` for a Chromium app window. `theme` is `:system`, `:light`, or
`:dark`.

For a resumable run, first construct `driver = StudioDriver(flow, recorder;
simulation_options...)`, start `studio(recorder)`, then run
`simulate!(driver, final_time)`. The driver retains the flow, recorder, and
fixed simulation options. Its initial call and all continuations are mutually
exclusive. A recorder without a driver remains supported but cannot continue.
After completion or a requested stop, `control` action `continue` accepts a
finite `final_time` strictly later than the actual completed time (and, after
ordinary completion, the previous target) and continues in the same store.
`state` reports `resumable`, `t_final`, and a
finite `suggested_final_time` or null.

`close(app)` closes the server and stops/finalizes its app-owned recording; it
does not stop the solver. `wait(app)` waits for the server to stop. With no
independent Julia thread, network I/O runs cooperatively at solver yields and
stored rendering is deferred until the solver is idle.

A `recording` WebSocket command with action `start` records a live stream:
source `live`, `path`, `field`, and a `diagnostics` name array, until `stop`
closes its bounded subscription and finalizes the file. Action `export` starts
a one-shot stored export immediately; `cancel` cooperatively cancels an active
stored job. Stored jobs accept `frames` (`keyframes` or `all`), an inclusive
`window` `[t0,t1]`, and `duration` in seconds; `fps` sets the frame rate.
Both sources accept `camera` (`fit` or `[xmin,xmax,ymin,ymax]`), `colorrange`
(`auto` or `[min,max]`; stored also accepts `global`), `colormap`,
`phase_rendering` (`cell` or `plic`), `size` `[width,height]` with positive even
integer dimensions, positive integer `fps`, and boolean `interface`,
`geometry`, and `blocks`. The diagnostics `arrangement` is `right` (default) or
`below`. Optional `diagnostic_options` is an array of `{names, scale, hidden}`
objects, at most one per selected diagnostic group. `names` is the nonempty
array of exact selected diagnostic names in that group, `scale` is `linear` or
`log`, and `hidden` is an array of `{name, component}` objects with one-based
component indices. Unlisted groups keep their metadata scale; hidden components
and their reference lines do not enter the legend or automatic limits.
Nonpositive values are omitted on log axes. Duplicate groups, names,
components, or JSON object keys, unknown keys, and invalid component indices
are rejected.
Both sources accept `theme` (`light` or `dark`). Live also accepts queue
`capacity` and `sim_per_second`, the simulation time covered by one second of
video; its output times are `t_start + k · sim_per_second / fps`, so video time
advances linearly in simulation time and frames repeat or are skipped. Stored
exports use the same uniform grid over the window (default: the whole
recording) and repeat or skip candidate frames. Unrecognized or unsupported
options are rejected.

Action `schedule` validates a stored command without starting a job and replies
`recording_schedule` with the output `frames`, `distinct` candidates,
`repeats`, `sparse` flag, `window`, `fps`, `duration`, uniform `spacing`,
`sim_per_second`, and the `first_frame` id the time line selects first.

For a single-frame preview, send `action: "preview"`, a required request id
and stored `frame` id, and the same stored export settings (the optional `path`
is validated but never written). The reply is `recording_preview` with the
request id and an authenticated relative PNG `url`. The scene uses the same
selected-frame colour range and rendering options as the animation. A newer
preview replaces the previous image; previewing never changes the recording
job or its download. To avoid concurrent CairoMakie rendering, a preview is
rejected while any Studio recording or encoding is active, and new recordings
or previews are rejected while a preview is rendering.

All clients receive the same run-level `recording` state, independent of their
timelines. It contains a job `id`, `source`, `status`, actual `encoded` and
`dropped` counts, `total`, and an `error` if it failed. Stored `progress` is the
fraction of output frames submitted to the encoder; it is null until the total
is known and may be 1 while finalization is still in progress. Live progress is
null until completion. Cancellation goes through `cancelling` to `cancelled`, but a
job already committing its result may complete first. Once complete, `url` is
an authenticated relative download URL; it is null before completion, on
cancellation and on failure. Starting another job at the same path invalidates
the older download URL.
"""
function studio(
        recorder::TwoPhaseAMR.Recorder; name::AbstractString = "Run 1",
        layout = :auto, host = "127.0.0.1", port = 0, open = :browser, theme = :system
    )
    layout === :auto || throw(ArgumentError("layout must be :auto"))
    theme in (:system, :light, :dark) || throw(ArgumentError("theme must be :system, :light, or :dark"))
    open in (:none, :browser, :app) || throw(ArgumentError("open must be :none, :browser, or :app"))
    host isa AbstractString && occursin(r"^[A-Za-z0-9._:-]+$", host) ||
        throw(ArgumentError("host must be a hostname or IP address"))
    port isa Integer && !(port isa Bool) && 0 <= port <= 65535 ||
        throw(ArgumentError("port must be between 0 and 65535"))
    token = bytes2hex(Random.rand(Random.RandomDevice(), UInt8, 16))
    app = Studio(
        recorder, TwoPhaseAMR.snapshot_store(recorder), String(name), token, String(host), 0,
        layout, theme, ReentrantLock(), :starting, nothing, IdDict{Any, Nothing}(), nothing,
        RecordingJob(), Dict{String, String}(), nothing, 0, Set{String}()
    )
    server = HTTP.listen!(
        String(host), Int(port); verbose = false,
        allow_h2c = false, max_header_bytes = 16384, read_header_timeout = 15
    ) do stream
        _serve(app, stream)
    end
    Base.@lock app.lock begin
        app.server = server
        app.port = HTTP.port(server)
        app.status = :started
    end
    try
        println("TwoPhaseAMR Studio: ", url(app))
        println("SSH tunnel: ssh -N -L ", app.port, ":127.0.0.1:", app.port, " <remote-host>")
        _launch_browser(app, url(app), open)
    catch
        close(app)
        rethrow()
    end
    return app
end

Base.isopen(app::Studio) = (Base.@lock app.lock app.status === :started) && isopen(app.server)

"""Wait until the Studio server stops accepting connections."""
Base.wait(app::Studio) = wait(app.server)

"""
    close(app::Studio)

Close all Studio connections and stop/finalize any app-owned recording without
stopping or resuming the simulation. An in-progress encoder may finish in the
background; its final status remains available on `app`.
"""
function Base.close(app::Studio)
    clients, launchdir, preview_dirs = Base.@lock app.lock begin
        app.status === :closed && return nothing
        app.status = :closed
        app.preview = nothing
        app.preview_generation += 1
        directories = collect(app.preview_dirs)
        empty!(app.preview_dirs)
        (collect(keys(app.clients)), app.launchdir, directories)
    end
    for directory in preview_dirs
        rm(directory; recursive = true, force = true)
    end
    _stop_recording!(app; closing = true)
    for client in clients
        client.active[] = false
        _unsubscribe(app.store, client.subscription)
    end
    try
        isnothing(app.server) || HTTP.forceclose(app.server)
    finally
        isnothing(launchdir) || rm(launchdir; recursive = true, force = true)
    end
    return nothing
end

# Fixed routes plus exact raw path matching avoid decoding or joining any
# attacker-controlled pathname. Reject duplicate token parameters.
function _authorized(target, token)
    parts = split(target, '?'; limit = 2)
    length(parts) == 2 || return false
    found = nothing
    for part in split(parts[2], '&')
        pair = split(part, '='; limit = 2)
        if length(pair) == 2 && pair[1] == "token"
            isnothing(found) || return false
            found = pair[2]
        end
    end
    isnothing(found) && return false
    ncodeunits(found) == ncodeunits(token) || return false
    difference = UInt8(0)
    for (a, b) in zip(codeunits(found), codeunits(token))
        difference |= a ⊻ b
    end
    return iszero(difference)
end

function _authority(req)
    host = HTTP.header(req.headers, "Host", nothing)
    return isnothing(host) ? req.host : host
end

function _allowed_host(app, authority)
    authority isa AbstractString || return false
    actual = lowercase(String(authority))
    configured = lowercase(app.host)
    if configured in ("127.0.0.1", "localhost")
        # A tunnel may expose a different local port. Permit only loopback
        # hostnames, and check the Origin against the full authority separately.
        match = Base.match(r"^(127\.0\.0\.1|localhost)(?::([0-9]{1,5}))?$", actual)
        isnothing(match) && return false
        port = match.captures[2]
        return isnothing(port) || 1 <= parse(Int, port) <= 65535
    end
    suffix = ":$(app.port)"
    # Port 80 can be elided from Host and Origin by browsers.
    hostpart = if endswith(actual, suffix)
        actual[1:(end - length(suffix))]
    elseif app.port == 80
        actual
    else
        return false
    end
    if configured == "0.0.0.0"
        return occursin(r"^[0-9]+(?:\.[0-9]+){3}$", hostpart)
    elseif configured == "::"
        return occursin(r"^\[[0-9a-f:.]+\]$", hostpart) ||
            occursin(r"^[0-9]+(?:\.[0-9]+){3}$", hostpart)
    end
    return hostpart == lowercase(_url_host(configured))
end

function _same_origin(req, authority)
    origin = HTTP.header(req.headers, "Origin", nothing)
    isnothing(origin) && return true # command-line and other non-browser clients
    authority isa AbstractString || return false
    actual = lowercase(origin)
    expected = "http://" * lowercase(authority)
    actual == expected && return true
    # Browsers omit the default port even when Host explicitly includes it.
    return (endswith(expected, ":80") && actual == expected[1:(end - 3)]) ||
        (endswith(actual, ":80") && expected == actual[1:(end - 3)])
end

function _respond(stream, status, body, content_type = "text/plain; charset=utf-8")
    HTTP.setstatus(stream, status)
    HTTP.setheader(stream, "Content-Type", content_type)
    HTTP.setheader(stream, "Cache-Control", "no-store")
    HTTP.setheader(stream, "Referrer-Policy", "no-referrer")
    HTTP.setheader(stream, "X-Content-Type-Options", "nosniff")
    HTTP.setheader(
        stream, "Content-Security-Policy",
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; " *
            "connect-src 'self' ws:; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    )
    write(stream, body)
    return nothing
end

function _download(app, stream, id)
    filename = Base.@lock app.lock get(app.downloads, id, nothing)
    (isnothing(filename) || !isfile(filename)) && return _respond(stream, 404, "Not found")
    extension = lowercase(last(splitext(filename)))
    content_type = get(
        Dict(
            ".gif" => "image/gif", ".mp4" => "video/mp4",
            ".mkv" => "video/x-matroska", ".webm" => "video/webm",
        ), extension, "application/octet-stream"
    )
    open(filename, "r") do io
        HTTP.setstatus(stream, 200)
        HTTP.setheader(stream, "Content-Type", content_type)
        HTTP.setheader(stream, "Content-Disposition", "attachment; filename=\"studio-export$extension\"")
        HTTP.setheader(stream, "Cache-Control", "no-store")
        HTTP.setheader(stream, "Referrer-Policy", "no-referrer")
        HTTP.setheader(stream, "X-Content-Type-Options", "nosniff")
        HTTP.setheader(stream, "Content-Length", string(filesize(filename)))
        HTTP.startwrite(stream)
        buffer = Vector{UInt8}(undef, 64 * 1024)
        while !eof(io)
            count = readbytes!(io, buffer)
            write(stream, view(buffer, 1:count))
        end
    end
    return nothing
end

function _serve(app, stream)
    # Catch transport and handler errors here: HTTP.jl must never log a request
    # target (which contains the token), and no consumer exception reaches the solver.
    try
        req = stream.message
        _authorized(req.target, app.token) || return _respond(stream, 403, "Forbidden")
        authority = _authority(req)
        _allowed_host(app, authority) || return _respond(stream, 403, "Forbidden")
        req.method == "GET" || return _respond(stream, 405, "Method not allowed")
        path = first(split(req.target, '?'; limit = 2))
        if path == "/ws"
            _same_origin(req, authority) || return _respond(stream, 403, "Forbidden")
            HTTP.WebSockets.isupgrade(req) || return _respond(stream, 400, "WebSocket upgrade required")
            HTTP.WebSockets.upgrade(stream; maxframesize = 65536) do socket
                _session(app, socket)
            end
            return nothing
        end
        download = match(r"^/download/([0-9a-f]{32})$", path)
        isnothing(download) || return _download(app, stream, String(download.captures[1]))
        preview = match(r"^/preview/([0-9a-f]{32})$", path)
        if !isnothing(preview)
            image = Base.@lock app.lock begin
                value = app.preview
                isnothing(value) || value[1] != preview.captures[1] ? nothing : value[2]
            end
            return isnothing(image) ? _respond(stream, 404, "Not found") :
                _respond(stream, 200, image, "image/png")
        end
        asset = get(_ASSET_ROUTES, path, nothing)
        isnothing(asset) && return _respond(stream, 404, "Not found")
        file, content_type = asset
        filename = joinpath(_ASSETS, file)
        isfile(filename) || return _respond(stream, 404, "Not found")
        return _respond(stream, 200, read(filename), content_type)
    catch
        # The response may already be committed (or the peer may have gone
        # away); in that case there is nothing left to write.
        try
            _respond(stream, 500, "Internal server error")
        catch
        end
        return nothing
    end
end

_send(socket, object) = HTTP.WebSockets.send(socket, JSON.json(object))

function _state(app)
    driver = Base.@lock app.recorder.lock app.recorder.driver
    isnothing(driver) && return _state_with_driver(app, driver)
    # A completed state must not combine the old store with a newly reserved
    # target (or precede FlowSimulation's update of its time and step count).
    return Base.@lock driver.lock _state_with_driver(app, driver)
end

function _state_with_driver(app, driver)
    time, step, dt, bytes, budget, count, rate, event = Base.@lock app.store.lock begin
        store = app.store
        log = store.steps
        time, step, dt = if !isempty(log.step)
            (last(log.time), last(log.step), last(log.dt))
        elseif !isempty(store.records)
            info = last(store.records)
            (info.time, info.step, 0.0)
        else
            (nothing, nothing, nothing)
        end
        elapsed = 0.0
        samples = 0
        for i in max(1, length(log.wall_time) - 19):length(log.wall_time)
            wall = log.wall_time[i]
            if isfinite(wall) && wall > 0
                elapsed += wall
                samples += 1
            end
        end
        event = isempty(store.events) ? nothing : last(store.events)
        (
            time, step, dt, store.frame_bytes + store.topology_bytes,
            store.retention.memory, length(store.records),
            iszero(samples) ? nothing : samples / elapsed, event,
        )
    end
    run, previous, state = Base.@lock app.recorder.lock begin
        recorder = app.recorder
        (recorder.run, recorder.previous_run, recorder.state)
    end
    recorded = isnothing(run) ? previous : run
    if isnothing(time) && !isnothing(recorded)
        time, step, dt = recorded.completed_time, recorded.completed_step, 0.0
    end
    snapshot = isnothing(driver) ? nothing : TwoPhaseAMR._driver_snapshot(driver)
    final_time = isnothing(snapshot) ?
        (isnothing(run) ? (isnothing(previous) ? nothing : _finite(previous.final_time)) : _finite(run.final_time)) :
        snapshot.final_time
    active = !isnothing(snapshot) && snapshot.active
    state = !isnothing(snapshot) && !snapshot.resumable ? :failed :
        active && state === :completed ? :running : state
    if state === :completed && !active && !isnothing(recorded)
        time, step = recorded.completed_time, recorded.completed_step
    end
    resumable = !isnothing(snapshot) && snapshot.resumable && state !== :failed
    suggested = resumable ? snapshot.suggested : nothing
    stopped = !isnothing(event) && event.kind === :stop && event.step == step &&
        event.time == time && state === :completed && !active
    status = Base.@lock app.lock app.status
    return (;
        type = "state", run_state = string(state),
        time = isnothing(time) ? nothing : _finite(time), t_final = final_time, step,
        resumable, suggested_final_time = suggested,
        dt = isnothing(dt) ? nothing : _finite(dt), store_bytes = bytes,
        store_budget = budget, frame_count = count,
        steps_per_second = isnothing(rate) ? nothing : _finite(rate), stopped,
        recording = _recording_state(app), server_status = string(status),
    )
end

function _hello(app, index)
    return (;
        type = "hello", protocol = Int(PROTOCOL_VERSION),
        runs = [_run_metadata(app.recorder, app.store, app.name)],
        frames = _frame_index(app.store, index), recording = _recording_state(app),
        layout = string(app.layout), theme = string(app.theme),
    )
end

function _frame_loop(client)
    try
        while client.active[]
            frame = take!(client.subscription)
            (isnothing(frame) || !client.active[]) && break
            _send(client.socket, _frame_message(frame))
        end
    catch
        # Network failure is local to this client, never an observer failure.
    finally
        client.active[] = false
        close(client.subscription)
    end
    return nothing
end

function _state_loop(app, client, previous_index, previous_state)
    try
        while client.active[]
            sleep(0.2)
            client.active[] || break
            index = TwoPhaseAMR.frames(app.store)
            if index !== previous_index
                _send(client.socket, (; type = "frames", frames = _frame_index(app.store, index)))
                previous_index = index
            end
            state = JSON.json(_state(app))
            if state != previous_state
                HTTP.WebSockets.send(client.socket, state)
                previous_state = state
            end
        end
    catch
        client.active[] = false
        close(client.subscription)
    end
    return nothing
end

function _optional_request(message)
    return haskey(message, "request") ? _request_id(message["request"]) : nothing
end

function _integer_selector(message, name)
    value = get(message, name, nothing)
    value isa Integer && !(value isa Bool) && 0 <= value <= typemax(Int) ||
        throw(ArgumentError("$name must be a nonnegative integer"))
    return Int(value)
end

function _get_message(app, socket, message)
    what = get(message, "what", nothing)
    request = _optional_request(message)
    if what == "frames"
        return _send(
            socket, (;
                type = "frames", request,
                frames = _frame_index(app.store, TwoPhaseAMR.frames(app.store)),
            )
        )
    elseif what == "field"
        isnothing(request) && throw(ArgumentError("field requests require a request id"))
        id = _integer_selector(message, "frame")
        frame = app.store[id]
        name = _registered_name(
            get(message, "field", nothing),
            TwoPhaseAMR.field_names(frame), "field"
        )
        encoding = get(message, "encoding", nothing)
        encoding in ("f32", "u16") || throw(ArgumentError("encoding must be f32 or u16"))
        mode = get(message, "mode", "cell")
        mode in ("cell", "plic") || throw(ArgumentError("mode must be cell or plic"))
        values = frame[name]
        phase = get(message, "phase", nothing)
        if values isa TwoPhaseAMR.TwoPhaseVisualizationValues
            phase in ("liquid", "gas") || throw(ArgumentError("phase must be liquid or gas for a phase field"))
            values = phase == "liquid" ? values.liquid : values.gas
        elseif !isnothing(phase)
            throw(ArgumentError("phase is only valid for a two-phase field"))
        end
        packet = encode_field(values, request; encoding = encoding == "f32" ? :f32 : :u16)
        return HTTP.WebSockets.send(socket, packet)
    elseif what == "topology"
        key = haskey(message, "topology") ? "topology" : "id"
        id = _integer_selector(message, key)
        return _send(socket, _topology_message(TwoPhaseAMR.topology(app.store, id), request))
    elseif what == "planes"
        id = _integer_selector(message, "frame")
        return _send(socket, _planes_message(app.store[id], request))
    elseif what == "diagnostics"
        available = TwoPhaseAMR.diagnostic_names(app.store)
        selection = get(message, "names", nothing)
        names = if isnothing(selection)
            available
        else
            selection isa AbstractVector || throw(ArgumentError("names must be an array"))
            Symbol[_registered_name(name, available, "diagnostic") for name in selection]
        end
        return _send(socket, _diagnostics_message(app.store, names, request))
    elseif what == "steplog"
        return _send(socket, _steplog_message(app.store, request))
    end
    throw(ArgumentError("unsupported get request"))
end

function _command(app, socket, message)
    message isa AbstractDict || throw(ArgumentError("message must be a JSON object"))
    if haskey(message, "protocol")
        version = message["protocol"]
        version isa Integer && !(version isa Bool) && version == PROTOCOL_VERSION ||
            throw(ArgumentError("unsupported protocol version"))
    end
    kind = get(message, "type", nothing)
    if kind == "get"
        return _get_message(app, socket, message)
    elseif kind == "control"
        action = get(message, "action", nothing)
        if action == "pause"
            TwoPhaseAMR.pause!(app.recorder)
        elseif action == "resume"
            TwoPhaseAMR.resume!(app.recorder)
        elseif action == "stop"
            TwoPhaseAMR.request_stop!(app.recorder)
        elseif action == "keyframe"
            TwoPhaseAMR.request_keyframe!(app.recorder)
        elseif action == "continue"
            all(key -> key in ("type", "protocol", "request", "action", "final_time"), keys(message)) ||
                throw(ArgumentError("continue accepts only final_time"))
            driver = Base.@lock app.recorder.lock app.recorder.driver
            isnothing(driver) && throw(ArgumentError("this run is not resumable"))
            target = TwoPhaseAMR._reserve_driver!(
                driver, get(message, "final_time", nothing); continuation = true,
            )
            Threads.@spawn :default begin
                try
                    TwoPhaseAMR._run_driver_reserved!(driver, target)
                catch exception
                    @error "Studio continuation failed" exception = (exception, catch_backtrace())
                end
            end
        else
            throw(ArgumentError("unknown control action"))
        end
        return _send(socket, _state(app))
    elseif kind == "recording"
        action = get(message, "action", nothing)
        if action == "preview"
            return _start_preview!(app, socket, message)
        elseif action == "schedule"
            return _recording_schedule(app, socket, message)
        elseif action == "start" || action == "export"
            _start_recording!(app, message; one_shot = action == "export")
        elseif action == "stop" || action == "cancel"
            all(key -> key in ("type", "protocol", "request", "action"), keys(message)) ||
                throw(ArgumentError("$action accepts no recording options"))
            action == "stop" ? _stop_recording!(app) : _cancel_recording!(app)
        else
            throw(ArgumentError("unknown recording action"))
        end
        return _send(socket, _state(app))
    end
    throw(ArgumentError("unsupported message type"))
end

# Do not let duplicate object keys silently replace validated command options.
struct _CommandObject <: AbstractDict{String, Any}
    data::Dict{String, Any}
end
_CommandObject() = _CommandObject(Dict{String, Any}())
Base.length(object::_CommandObject) = length(object.data)
Base.iterate(object::_CommandObject, state...) = iterate(object.data, state...)
Base.haskey(object::_CommandObject, key) = haskey(object.data, key)
Base.get(object::_CommandObject, key, default) = get(object.data, key, default)
Base.getindex(object::_CommandObject, key) = object.data[key]
function Base.setindex!(object::_CommandObject, value, key)
    haskey(object.data, key) && throw(ArgumentError("duplicate JSON object key"))
    object.data[key] = value
    return value
end

function _session(app, socket)
    subscription = TwoPhaseAMR.subscribe(app.store; capacity = 1)
    client = StudioClient(socket, subscription, Threads.Atomic{Bool}(true))
    accepted = Base.@lock app.lock begin
        if app.status !== :started
            false
        else
            app.clients[client] = nothing
            true
        end
    end
    if !accepted
        _unsubscribe(app.store, subscription)
        return nothing
    end
    try
        index = TwoPhaseAMR.frames(app.store)
        _send(socket, _hello(app, index))
        initial_state = JSON.json(_state(app))
        HTTP.WebSockets.send(socket, initial_state)
        @async _frame_loop(client)
        @async _state_loop(app, client, index, initial_state)
        for raw in socket
            request = nothing
            try
                raw isa AbstractString || throw(ArgumentError("commands must be JSON text"))
                message = JSON.parse(raw; dicttype = _CommandObject)
                if message isa AbstractDict && haskey(message, "request")
                    request = _optional_request(message)
                end
                _command(app, socket, message)
            catch err
                # Error strings are generated by our validators or core API;
                # never echo the input JSON or the URL credential.
                text = err isa ArgumentError || err isa KeyError ? sprint(showerror, err) : "invalid or failed request"
                _send(socket, (; type = "error", message = text, request))
            end
        end
    catch
        # Closing a socket only affects that client. Recording remains app-owned.
    finally
        client.active[] = false
        _unsubscribe(app.store, subscription)
        Base.@lock app.lock delete!(app.clients, client)
    end
    return nothing
end
