using Test
using TwoPhaseAMR
using TwoPhaseAMRStudio
using HTTP
using JSON

const _STUDIO_TIMEOUT = 90.0
const _RECORDING_TIMEOUT = 180.0

function _await(predicate, description; timeout = _STUDIO_TIMEOUT)
    timedwait(predicate, timeout; pollint = 0.005) === :ok ||
        error("Timed out waiting for $description")
    return nothing
end

function _take_studio(channel, description; timeout = _STUDIO_TIMEOUT)
    _await(() -> isready(channel), description; timeout)
    return take!(channel)
end

function _message(ws, predicate; timeout = _STUDIO_TIMEOUT)
    deadline = time() + timeout
    while time() < deadline
        receive_task = @async HTTP.WebSockets.receive(ws)
        _await(
            () -> istaskdone(receive_task), "WebSocket message";
            timeout = max(0.01, deadline - time())
        )
        raw = fetch(receive_task)
        message = raw isa AbstractString ? JSON.parse(raw) : raw
        predicate(message) && return message
    end
    error("Timed out waiting for matching WebSocket message")
end

_istype(message, type) = message isa AbstractDict && get(message, "type", "") == type
_send(ws, object) = HTTP.WebSockets.send(ws, JSON.json(object))

function _frames_reply(ws)
    _send(ws, Dict("type" => "get", "what" => "frames"))
    return _message(ws, message -> _istype(message, "frames"))["frames"]
end

function _field_reply(ws, frame, field, encoding, request)
    _send(
        ws, Dict(
            "type" => "get", "what" => "field", "frame" => frame,
            "field" => field, "encoding" => encoding, "request" => request,
        )
    )
    return _message(ws, message -> message isa AbstractVector{UInt8})
end

function _gif_envelope(path)
    isfile(path) && filesize(path) > 6 || return false
    return open(path, "r") do io
        signature = String(read(io, 6))
        seekend(io)
        seek(io, position(io) - 1)
        signature in ("GIF87a", "GIF89a") && read(io, UInt8) == 0x3b
    end
end

function _run_fixture()
    mesh = BlockTreeMesh(block_size = (4, 4))
    flow = FlowSimulation(
        mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.05);
        boundary = FreeSlip(), dt = 0.01, diagnostics = (KineticEnergy(),),
        velocity = (x, y, t, axis) -> axis == 1 ? sinpi(2y) : 0.0,
    )
    fields = VisualizationFieldRegistry()
    register_field!(fields, :pressure, flow.pressure)
    register_field!(fields, :velocity_x, flow.velocity, Face{1}())
    # A genuine phase field on the simulation mesh exercises intrinsic liquid
    # and gas arrays, including absent-phase NaNs, without mocking a Frame.
    interface = InterfaceField(mesh, (x, y) -> x - 0.625; lipschitz = 1)
    scalar = PhaseScalar(interface; liquid = 7, gas = -11)
    for ref in TwoPhaseAMR._cell_references(mesh)
        TwoPhaseAMR._phase_cell_is_active(interface, ref, :liquid) ||
            TwoPhaseAMR._set!(scalar.liquid, NaN, ref)
        TwoPhaseAMR._phase_cell_is_active(interface, ref, :gas) ||
            TwoPhaseAMR._set!(scalar.gas, NaN, ref)
    end
    register_field!(fields, :phase_scalar, scalar)
    recorder = Recorder(
        fields; diagnostics = flow, keyframes = EveryNSteps(3), preview = EveryStep(),
        retention = Retention(preview = :all), max_capture_overhead = 1.0,
        clock = () -> 0.0,
    )
    return flow, recorder
end

function _address(app)
    address = string(TwoPhaseAMRStudio.url(app))
    matched = match(r"^(https?://[^/]+)(?:/[^?]*)?\?(.*)$", address)
    isnothing(matched) && error("Studio URL has no authority/query")
    token = match(r"(?:^|&)token=([^&]+)", matched.captures[2])
    isnothing(token) && error("Studio URL has no token")
    return (; base = matched.captures[1], token = token.captures[1], address)
end

function _ws_address(address; token = address.token)
    scheme = replace(address.base, r"^http" => "ws")
    return scheme * "/ws?token=" * token
end

@testset "Ephemeral server and authentication" begin
    _, recorder = _run_fixture()
    app = studio(recorder; port = 0, open = :none)
    second = nothing
    try
        @test isopen(app)
        address = _address(app)
        @test occursin(r"^http://127\.0\.0\.1:[1-9][0-9]*$", address.base)
        @test length(address.token) >= 22 # at least 128 bits in URL-safe base64
        second = studio(recorder; port = 0, open = :none)
        other = _address(second)
        @test other.base != address.base
        @test other.token != address.token

        root = address.base * "/"
        for target in (root, root * "?token=wrong")
            response = HTTP.get(target; status_exception = false, request_timeout = 10)
            @test response.status in (401, 403)
        end
        response = HTTP.get(address.address; status_exception = false, request_timeout = 10)
        @test response.status == 200
        html = String(response.body)
        @test occursin(r"<html"i, html)
        # Use an asset actually advertised by the authenticated shell, rather
        # than checking only a made-up path that might simply return 404.
        assets = [
            m.captures[1] for m in eachmatch(
                    r"(?:src|href)=[\"']([^\"']+\.(?:js|css)(?:\?[^\"']*)?)[\"']"i, html,
                )
        ]
        @test !isempty(assets)
        for asset in assets
            path = first(split(asset, '?'; limit = 2))
            startswith(path, "/") || (path = "/" * path)
            for suffix in ("", "?token=wrong")
                denied = HTTP.get(address.base * path * suffix; status_exception = false, request_timeout = 10)
                @test denied.status in (401, 403)
            end
            allowed = HTTP.get(
                address.base * path * "?token=" * address.token;
                status_exception = false, request_timeout = 10
            )
            @test allowed.status == 200
            @test !isempty(allowed.body)
        end
        # A valid token must not grant access outside the asset directory.
        # Encoded separators/dot segments keep the request from being
        # normalized by the client before it reaches the server.
        for path in (
                "/%2e%2e/Project.toml", "/%2e%2e%2fProject.toml",
                "/..%2f..%2fProject.toml", "/assets/%2e%2e/Project.toml",
                "/assets/%2e%2e%2fProject.toml", "/assets/..%2f..%2fProject.toml",
            )
            response = HTTP.get(
                address.base * path * "?token=" * address.token;
                status_exception = false, request_timeout = 10
            )
            @test response.status in (400, 403, 404)
        end

        upgrade_headers = [
            "Connection" => "Upgrade", "Upgrade" => "websocket",
            "Sec-WebSocket-Version" => "13",
            "Sec-WebSocket-Key" => "dGhlIHNhbXBsZSBub25jZQ==",
        ]
        for target in (address.base * "/ws", address.base * "/ws?token=wrong")
            denied = HTTP.get(target, upgrade_headers; status_exception = false, request_timeout = 10)
            @test denied.status in (401, 403)
        end
        cross_origin = HTTP.get(
            address.base * "/ws?token=" * address.token,
            [upgrade_headers; "Origin" => "https://attacker.invalid"];
            status_exception = false, request_timeout = 10
        )
        @test cross_origin.status in (401, 403)

        # Perform genuine RFC 6455 handshakes, not ordinary GETs to /ws.
        for wsurl in (
                replace(address.base, r"^http" => "ws") * "/ws",
                _ws_address(address; token = "wrong"),
            )
            @test_throws Exception HTTP.WebSockets.open(wsurl; response_header_timeout = 10) do ws
                HTTP.WebSockets.receive(ws)
            end
        end
        @test_throws Exception HTTP.WebSockets.open(
            _ws_address(address);
            headers = ["Origin" => "https://attacker.invalid"], response_header_timeout = 10
        ) do ws
            HTTP.WebSockets.receive(ws)
        end
        HTTP.WebSockets.open(
            _ws_address(address);
            headers = ["Origin" => address.base], response_header_timeout = 10
        ) do ws
            hello = _message(ws, message -> _istype(message, "hello"))
            state = _message(ws, message -> _istype(message, "state"))
            @test hello["protocol"] == 1
            @test haskey(hello, "runs") && length(hello["runs"]) == 1
            @test haskey(hello, "frames") && haskey(hello, "recording")
            @test haskey(state, "run_state") && haskey(state, "recording")
        end

        # An SSH local forward connects to the actual listener but presents
        # its own local authority. Auth and same-origin checks must use that
        # presented authority, not require the server's listening port.
        port = parse(Int, last(split(address.base, ':')))
        forwarded_port = port == 43210 ? 43211 : 43210
        authority = "127.0.0.1:$forwarded_port"
        forwarded = ["Host" => authority]
        forwarded_html = HTTP.get(
            address.address, forwarded;
            status_exception = false, request_timeout = 10
        )
        @test forwarded_html.status == 200
        for suffix in ("", "?token=wrong")
            denied_html = HTTP.get(
                address.base * "/" * suffix, forwarded;
                status_exception = false, request_timeout = 10
            )
            @test denied_html.status in (401, 403)
        end
        if !isempty(assets)
            path = first(split(first(assets), '?'; limit = 2))
            startswith(path, "/") || (path = "/" * path)
            forwarded_asset = HTTP.get(
                address.base * path * "?token=" * address.token,
                forwarded; status_exception = false, request_timeout = 10
            )
            @test forwarded_asset.status == 200
        end
        denied_upgrade = HTTP.get(
            address.base * "/ws?token=" * address.token,
            [upgrade_headers; "Host" => authority; "Origin" => address.base];
            status_exception = false, request_timeout = 10
        )
        @test denied_upgrade.status in (401, 403)
        HTTP.WebSockets.open(
            _ws_address(address);
            headers = ["Host" => authority, "Origin" => "http://" * authority],
            response_header_timeout = 10
        ) do ws
            hello = _message(ws, message -> _istype(message, "hello"))
            state = _message(ws, message -> _istype(message, "state"))
            @test hello["protocol"] == 1
            @test haskey(state, "run_state")
            @test isempty(_frames_reply(ws))
        end
    finally
        isnothing(second) || close(second)
        close(app)
    end
    @test !isopen(app)
    @test !isopen(second)
end

@testset "Real simulation, WebSocket controls, history and recording" begin
    flow, recorder = _run_fixture()
    store = snapshot_store(recorder)
    entered, proceed = Channel{Float64}(1), Channel{Nothing}(1)
    last_midpoint = Ref(-Inf)
    function gated_force(x, y, midpoint, axis)
        if midpoint > last_midpoint[]
            last_midpoint[] = midpoint
            put!(entered, midpoint)
            _take_studio(proceed, "test release of simulation step")
        end
        return 0.0
    end
    # Start at a known boundary; all later advancement is a genuine step!.
    pause!(recorder)
    simulation = Threads.@spawn simulate!(flow, 100.0; observer = recorder, force = gated_force)
    app = nothing
    try
        _await(() -> run_state(recorder) === :paused || istaskdone(simulation), "initial pause")
        istaskdone(simulation) && fetch(simulation)
        @test run_state(recorder) === :paused
        app = studio(recorder; port = 0, open = :none)
        address = _address(app)
        mktempdir() do directory
            live_path = joinpath(directory, "live.gif")
            stored_path = joinpath(directory, "stored.gif")
            # Start a second, independently controlled WebSocket client.
            HTTP.WebSockets.open(_ws_address(address)) do slow
                _await(
                    () -> (Base.@lock app.lock length(app.clients)) == 1,
                    "slow client subscription"
                )
                slow_client = Base.@lock app.lock only(keys(app.clients))
                HTTP.WebSockets.open(_ws_address(address)) do ws
                    hello = _message(ws, message -> _istype(message, "hello"))
                    initial_state = _message(ws, message -> _istype(message, "state"))
                    @test hello["protocol"] == 1
                    @test length(hello["runs"]) == 1
                    run = only(hello["runs"])
                    @test all(
                        haskey(run, key) for key in
                            ("id", "name", "domain", "block_size", "t0", "t_final", "fields", "diagnostics")
                    )
                    @test run["block_size"] == [4, 4]
                    @test run["t0"] == 0.0 && run["t_final"] == 100.0
                    @test Set(field["name"] for field in run["fields"]) ==
                        Set(["pressure", "velocity_x", "phase_scalar"])
                    @test any(diagnostic -> diagnostic["name"] == "kinetic_energy", run["diagnostics"])
                    @test initial_state["run_state"] == "paused"
                    @test all(
                        haskey(initial_state, key) for key in
                            ("time", "step", "dt", "store_bytes", "recording")
                    )
                    @test initial_state["time"] == 0.0 && initial_state["step"] == 0
                    @test initial_state["store_bytes"] > 0
                    initial_id = first(frames(store)).id
                    @test any(frame -> frame["id"] == initial_id, hello["frames"])
                    @test !hello["recording"]["active"]

                    _send(
                        ws, Dict(
                            "type" => "recording", "action" => "start", "source" => "live",
                            "path" => live_path, "field" => "velocity_x", "diagnostics" => String[]
                        )
                    )
                    started = _message(
                        ws, message -> _istype(message, "state") &&
                            get(message["recording"], "status", "") in ("recording", "error");
                        timeout = _RECORDING_TIMEOUT
                    )["recording"]
                    @test started["status"] != "error"
                    @test started["active"] && started["source"] == "live"
                    @test started["field"] == "velocity_x" && started["path"] == live_path
                    started["status"] == "recording" ||
                        error("Live recording failed: $(get(started, "error", nothing))")

                    # Both encodings carry the real stored frame, not synthetic
                    # fixture data. The nonconstant face values catch order and
                    # size mistakes as well as a constant-only encoder.
                    expected = Float32.(store[initial_id][:velocity_x])
                    @test length(unique(expected)) > 1
                    packet = _field_reply(ws, initial_id, "velocity_x", "f32", UInt32(17))
                    decoded = decode_field(packet)
                    @test decoded.request == 17 && decoded.encoding === :f32
                    @test isequal(decoded.values, expected)
                    packet = _field_reply(ws, initial_id, "velocity_x", "u16", UInt32(18))
                    decoded = decode_field(packet)
                    @test decoded.request == 18 && decoded.encoding === :u16
                    # Half a quantization bin plus Float32 scale/offset rounding.
                    bound = Float64(decoded.scale) / 2 + 8eps(Float32) * maximum(abs, expected)
                    @test all(abs.(Float64.(decoded.values) .- expected) .<= bound)

                    frame = store[initial_id]
                    for (number, (phase, mode, encoding)) in enumerate(
                            Iterators.product(("liquid", "gas"), ("cell", "plic"), ("f32", "u16"))
                        )
                        request = 30 + number
                        _send(
                            ws, Dict(
                                "type" => "get", "what" => "field", "frame" => initial_id,
                                "field" => "phase_scalar", "phase" => phase, "mode" => mode,
                                "encoding" => encoding, "request" => request
                            )
                        )
                        packet = _message(ws, message -> message isa AbstractVector{UInt8})
                        decoded = decode_field(packet)
                        values = Float32.(getproperty(frame[:phase_scalar], Symbol(phase)))
                        @test any(isnan, values)
                        @test any(isfinite, values)
                        @test decoded.request == request && decoded.encoding === Symbol(encoding)
                        @test isequal(decoded.values, values)
                    end
                    for (request, options) in (
                            (50, Dict("mode" => "cell")), # phase is required
                            (51, Dict("phase" => "solid", "mode" => "cell")),
                            (52, Dict("phase" => "liquid", "mode" => "invalid")),
                        )
                        command = Dict{String, Any}(
                            "type" => "get", "what" => "field",
                            "frame" => initial_id, "field" => "phase_scalar",
                            "encoding" => "f32", "request" => request
                        )
                        merge!(command, options)
                        _send(ws, command)
                        reply = _message(
                            ws, message -> _istype(message, "error") &&
                                get(message, "request", nothing) == request
                        )
                        @test !isempty(reply["message"])
                    end
                    _send(
                        ws, Dict(
                            "type" => "get", "what" => "topology",
                            "topology" => frame.info.topology, "request" => 60
                        )
                    )
                    topology = _message(
                        ws, message -> _istype(message, "topology") &&
                            get(message, "request", nothing) == 60
                    )
                    @test topology["id"] == frame.info.topology
                    @test Tuple(topology["block_size"]) == frame.topology.block_size
                    @test length(topology["blocks"]) == length(frame.topology.keys)
                    _send(
                        ws, Dict(
                            "type" => "get", "what" => "planes",
                            "frame" => initial_id, "request" => 61
                        )
                    )
                    planes = _message(
                        ws, message -> _istype(message, "planes") &&
                            get(message, "request", nothing) == 61
                    )
                    @test planes["frame"] == initial_id
                    actual_phase, captured_phase = only(planes["phases"]), only(frame.phases)
                    @test actual_phase["liquid"] == collect(captured_phase.liquid)
                    @test actual_phase["mixed"] == captured_phase.mixed
                    @test !isempty(captured_phase.mixed)
                    @test [Float32.(plane) for plane in actual_phase["planes"]] ==
                        [Float32[plane...] for plane in captured_phase.planes]
                    for (id, name, encoding, request) in (
                            (initial_id, "missing", "f32", 21),
                            (typemax(Int32), "velocity_x", "f32", 22),
                            (initial_id, "velocity_x", "invalid", 23),
                        )
                        _send(
                            ws, Dict(
                                "type" => "get", "what" => "field", "frame" => id,
                                "field" => name, "encoding" => encoding, "request" => request
                            )
                        )
                        error_reply = _message(
                            ws, message -> _istype(message, "error") &&
                                get(message, "request", nothing) == request
                        )
                        @test !isempty(error_reply["message"])
                    end
                    _send(
                        ws, Dict(
                            "type" => "get", "what" => "frames",
                            "protocol" => 2, "request" => 70
                        )
                    )
                    invalid_version = _message(
                        ws, message -> _istype(message, "error") &&
                            get(message, "request", nothing) == 70
                    )
                    @test !isempty(invalid_version["message"])
                    HTTP.WebSockets.send(ws, "{")
                    invalid_json = _message(ws, message -> _istype(message, "error"))
                    @test !isempty(invalid_json["message"])
                    @test any(frame -> frame["id"] == initial_id, _frames_reply(ws))
                    @test run_state(recorder) === :paused


                    _send(ws, Dict("type" => "control", "action" => "resume"))
                    _take_studio(entered, "first physical step")
                    @test run_state(recorder) === :running
                    _send(ws, Dict("type" => "control", "action" => "keyframe"))
                    _frames_reply(ws) # A reply is a barrier after the control.
                    put!(proceed, nothing)
                    manual = _message(
                        ws, message -> _istype(message, "frame") &&
                            get(message, "kind", "") == "manual"
                    )
                    @test manual["step"] == 1 && manual["run"] == run["id"]
                    @test all(haskey(manual, key) for key in ("id", "time", "topology", "stats"))
                    @test haskey(store, manual["id"])
                    stats = manual["stats"]["velocity_x"]
                    captured_stats = store[manual["id"]].stats[:velocity_x]
                    @test stats["finite"] == captured_stats.finite
                    @test stats["minimum"] == captured_stats.minimum
                    @test stats["maximum"] == captured_stats.maximum

                    _take_studio(entered, "second physical step")
                    _send(ws, Dict("type" => "control", "action" => "pause"))
                    _await(() -> run_state(recorder) === :pause_requested, "pause request")
                    put!(proceed, nothing)
                    _await(() -> run_state(recorder) === :paused, "pause at step boundary")
                    @test last(frames(store)).kind === :pause
                    @test last(frames(store)).step == 2
                    @test any(frame -> frame["id"] == initial_id, _frames_reply(ws))
                    @test decode_field(_field_reply(ws, initial_id, "velocity_x", "f32", 24)).values == expected
                    @test run_state(recorder) === :paused

                    # Recording is run-level state, shared with a newly opened
                    # client even when the first client asks for old frames.
                    HTTP.WebSockets.open(_ws_address(address)) do observer
                        other = _message(observer, message -> _istype(message, "hello"))
                        rec = other["recording"]
                        @test rec["active"] && rec["source"] == "live"
                        @test rec["field"] == "velocity_x" && rec["path"] == live_path
                    end
                    # HTTP.jl 2.x eagerly drains client sockets to an unbounded
                    # channel, even if receive() is never called. Hold this
                    # connection's real send lock to model a blocked transport
                    # deterministically, then inspect its latest-wins mailbox.
                    subscription = slow_client.subscription
                    lock(slow_client.socket.sendlock)
                    try
                        _send(ws, Dict("type" => "control", "action" => "resume"))
                        for step in 3:10
                            _take_studio(entered, "physical step $step")
                            @test run_state(recorder) === :running
                            @test any(frame -> frame["id"] == initial_id, _frames_reply(ws))
                            @test decode_field(_field_reply(ws, initial_id, "velocity_x", "f32", 100 + step)).values == expected
                            put!(proceed, nothing)
                            _await(
                                () -> !isempty(frames(store)) && last(frames(store)).step >= step,
                                "completed step $step"
                            )
                        end
                        # Entering step 11 guarantees step 10 was committed,
                        # published to every subscription, and logged.
                        _take_studio(entered, "step after stalled delivery")
                        dropped, queued = Base.@lock subscription.condition (
                            subscription.dropped, [frame.id for frame in subscription.queue],
                        )
                        @test dropped > 0
                        @test queued == [last(frames(store)).id]
                    finally
                        unlock(slow_client.socket.sendlock)
                    end
                    latest_slow = _message(
                        slow, message -> _istype(message, "frame") &&
                            message["step"] == 10
                    )
                    @test latest_slow["id"] == last(frames(store)).id
                    @test length(step_log(store)) >= 10
                    @test diagnostic_history(store, :kinetic_energy).steps ==
                        diagnostic_history(flow, :kinetic_energy).steps

                    # Tear down the real client transport without a WebSocket
                    # close handshake, then advance the solver after the fault.
                    slow.close_transport!()
                    put!(proceed, nothing)
                    _await(() -> last(frames(store)).step >= 11, "progress after disconnect")
                    _take_studio(entered, "step at stop request")
                    _send(ws, Dict("type" => "control", "action" => "stop"))
                    _await(() -> run_state(recorder) === :stopping, "stop request")
                    put!(proceed, nothing)
                    _await(() -> istaskdone(simulation), "stopped simulation")
                    result = fetch(simulation)
                    @test result.stopped && result.steps >= 12
                    @test run_state(recorder) === :completed
                    @test all(event -> event.kind !== :failure, run_events(store))

                    _send(ws, Dict("type" => "recording", "action" => "stop"))
                    stopped = _message(
                        ws, message -> _istype(message, "state") &&
                            get(message["recording"], "status", "") in ("completed", "error");
                        timeout = _RECORDING_TIMEOUT
                    )["recording"]
                    @test stopped["status"] == "completed"
                    @test !stopped["active"] && stopped["source"] == "live"
                    @test stopped["encoded"] > 0 && stopped["dropped"] >= 0
                    @test _gif_envelope(live_path)

                    _send(
                        ws, Dict(
                            "type" => "recording", "action" => "start", "source" => "stored",
                            "path" => stored_path, "field" => "velocity_x", "diagnostics" => String[]
                        )
                    )
                    stored_started = _message(
                        ws, message -> _istype(message, "state") &&
                            get(message["recording"], "source", "") == "stored" &&
                            get(message["recording"], "status", "") in ("recording", "error");
                        timeout = _RECORDING_TIMEOUT
                    )["recording"]
                    @test stored_started["active"] && stored_started["status"] == "recording"
                    @test stored_started["field"] == "velocity_x" && stored_started["path"] == stored_path
                    stored_started["active"] || error("Stored recording failed: $(get(stored_started, "error", nothing))")
                    @test any(frame -> frame["id"] == initial_id, _frames_reply(ws))
                    @test isequal(decode_field(_field_reply(ws, initial_id, "velocity_x", "f32", 200)).values, expected)
                    HTTP.WebSockets.open(_ws_address(address)) do observer
                        other = _message(observer, message -> _istype(message, "hello"))
                        @test other["recording"]["active"]
                        @test other["recording"]["source"] == "stored"
                        @test other["recording"]["path"] == stored_path
                    end
                    expected_count = length(keyframes(store))
                    _send(ws, Dict("type" => "recording", "action" => "stop"))
                    stored = _message(
                        ws, message -> _istype(message, "state") &&
                            get(message["recording"], "source", "") == "stored" &&
                            get(message["recording"], "status", "") in ("completed", "error");
                        timeout = _RECORDING_TIMEOUT
                    )["recording"]
                    @test stored["status"] == "completed"
                    @test !stored["active"] && stored["path"] == stored_path
                    @test stored["encoded"] == expected_count && stored["dropped"] == 0
                    @test _gif_envelope(stored_path)

                    # An actual export failure must reach clients as recording
                    # state rather than kill the server or claim a completed file.
                    not_directory = joinpath(directory, "regular-file")
                    open(not_directory, "w") do io
                        write(io, "not a directory")
                    end
                    invalid_path = joinpath(not_directory, "invalid.gif")
                    _send(
                        ws, Dict(
                            "type" => "recording", "action" => "start", "source" => "stored",
                            "path" => invalid_path, "field" => "velocity_x", "diagnostics" => String[]
                        )
                    )
                    failed_start = _message(
                        ws, message -> _istype(message, "state") &&
                            get(message["recording"], "path", nothing) == invalid_path &&
                            get(message["recording"], "active", false)
                    )["recording"]
                    @test failed_start["status"] == "recording"
                    _send(ws, Dict("type" => "recording", "action" => "stop"))
                    failed = _message(
                        ws, message -> _istype(message, "state") &&
                            get(message["recording"], "path", nothing) == invalid_path &&
                            get(message["recording"], "status", "") in ("completed", "error");
                        timeout = _RECORDING_TIMEOUT
                    )["recording"]
                    @test failed["status"] == "error" && !failed["active"]
                    @test get(failed, "error", nothing) isa AbstractString && !isempty(failed["error"])
                    @test !isfile(invalid_path)
                    @test any(frame -> frame["id"] == initial_id, _frames_reply(ws))
                end
            end
        end
    finally
        try
            if !istaskdone(simulation)
                run_state(recorder) in (:created, :running, :pause_requested, :paused, :stopping) &&
                    request_stop!(recorder)
                isready(proceed) || put!(proceed, nothing)
                _await(() -> istaskdone(simulation), "fixture teardown")
            end
        finally
            isnothing(app) || close(app)
        end
    end
end
