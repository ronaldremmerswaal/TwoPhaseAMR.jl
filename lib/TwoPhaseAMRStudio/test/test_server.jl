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
    add_diagnostic!(
        flow, :signed_pair, state -> (state.time - 0.03, 1000.0 + state.time);
        label = "Signed pair", components = ["signal", "large"],
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
    app = studio(recorder; name = "Adaptive run", port = 0, open = :none)
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
            @test only(hello["runs"])["name"] == "Adaptive run"
            @test haskey(hello, "frames") && haskey(hello, "recording")
            @test haskey(state, "run_state") && haskey(state, "recording")
            # A one-shot export with no captured frames fails at submission,
            # rather than producing an empty or fabricated job result.
            _send(
                ws, Dict(
                    "type" => "recording", "action" => "export", "source" => "stored",
                    "path" => joinpath(tempdir(), "empty-studio.gif"), "field" => "velocity_x",
                    "request" => 901,
                )
            )
            empty_error = _message(
                ws, message -> _istype(message, "error") &&
                    get(message, "request", nothing) == 901
            )
            @test occursin("no stored frames", empty_error["message"])
            _send(ws, Dict("type" => "recording", "action" => "cancel", "request" => 902))
            inactive = _message(
                ws, message -> _istype(message, "error") &&
                    get(message, "request", nothing) == 902
            )
            @test occursin("no recording is active", inactive["message"])
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

@testset "Natural completion is not a requested stop" begin
    flow, recorder = _run_fixture()
    result = simulate!(flow, 0.02; observer = recorder)
    @test !result.stopped
    app = studio(recorder; port = 0, open = :none)
    try
        HTTP.WebSockets.open(_ws_address(_address(app))) do ws
            _message(ws, message -> _istype(message, "hello"))
            state = _message(ws, message -> _istype(message, "state"))
            @test state["run_state"] == "completed"
            @test !state["stopped"]
            @test state["steps_per_second"] > 0
        end
    finally
        close(app)
    end
end

@testset "Recording diagnostic option protocol" begin
    flow, recorder = _run_fixture()
    # This vector has no component metadata and has not yet been mirrored.
    add_diagnostic!(flow, :unnamed_pair, state -> (1.0, 2.0))
    app = studio(recorder; port = 0, open = :none)
    try
        base = Dict{String, Any}(
            "type" => "recording", "action" => "export", "source" => "stored",
            "path" => "diagnostic.gif", "field" => "velocity_x",
            "diagnostics" => ["signed_pair", "unnamed_pair"],
        )
        pair = Dict(
            "names" => ["signed_pair"], "scale" => "log",
            "hidden" => [Dict("name" => "signed_pair", "component" => 2)],
        )
        unnamed = Dict("names" => ["unnamed_pair"], "scale" => "linear", "hidden" => [])
        command = merge(base, Dict("diagnostic_options" => [pair, unnamed]))
        expected = [
            (; names = [:signed_pair], scale = :log, hidden = [(; name = :signed_pair, component = 2)]),
            (; names = [:unnamed_pair], scale = :linear, hidden = NamedTuple[]),
        ]
        for (source, one_shot) in (("stored", true), ("live", false))
            request = merge(command, Dict("source" => source, "action" => one_shot ? "export" : "start"))
            _, _, _, names, options = TwoPhaseAMRStudio._validate_recording(app, request, one_shot)
            @test names == [:signed_pair, :unnamed_pair]
            @test options.diagnostic_options == expected
            job = TwoPhaseAMRStudio.RecordingJob()
            job.source = Symbol(source)
            job.diagnostics = names
            job.options = options
            @test TwoPhaseAMRStudio._export_options(job).diagnostic_options == expected
        end
        _, _, _, _, defaults = TwoPhaseAMRStudio._validate_recording(app, base, true)
        @test defaults.diagnostic_options === nothing
        @test defaults.duration === nothing # resolved from the frozen candidates
        _, _, _, _, empty_options = TwoPhaseAMRStudio._validate_recording(
            app, merge(base, Dict("diagnostic_options" => [])), true,
        )
        @test isempty(empty_options.diagnostic_options)

        bad_options = (
            nothing, "log", [1], [Dict("names" => ["signed_pair"], "scale" => "log")],
            [merge(pair, Dict("extra" => 1))], [merge(pair, Dict("names" => []))],
            [merge(pair, Dict("names" => ["missing"]))],
            [merge(pair, Dict("names" => ["signed_pair", "signed_pair"]))],
            [merge(pair, Dict("names" => ["signed_pair", "unnamed_pair"]))],
            [merge(pair, Dict("scale" => "symlog"))], [merge(pair, Dict("hidden" => nothing))],
            [merge(pair, Dict("hidden" => [Dict("name" => "signed_pair")]))],
            [merge(pair, Dict("hidden" => [Dict("name" => "signed_pair", "component" => 1, "extra" => 1)]))],
            [merge(pair, Dict("hidden" => [Dict("name" => "unnamed_pair", "component" => 1)]))],
            [merge(pair, Dict("hidden" => [Dict("name" => "signed_pair", "component" => 0)]))],
            [merge(pair, Dict("hidden" => [Dict("name" => "signed_pair", "component" => 3)]))],
            [merge(pair, Dict("hidden" => [Dict("name" => "signed_pair", "component" => true)]))],
            [merge(pair, Dict("hidden" => [Dict("name" => "signed_pair", "component" => 1.0)]))],
            [merge(pair, Dict("hidden" => [pair["hidden"][1], pair["hidden"][1]]))],
            [pair, pair],
        )
        for bad in bad_options, source in ("stored", "live")
            request = merge(
                base, Dict(
                    "source" => source, "action" => source == "stored" ? "export" : "start",
                    "diagnostic_options" => bad,
                ),
            )
            @test_throws ArgumentError TwoPhaseAMRStudio._validate_recording(app, request, source == "stored")
        end
        parsed = JSON.parse(JSON.json(command); dicttype = TwoPhaseAMRStudio._CommandObject)
        @test get(parsed, "type", nothing) == "recording"
        @test get(parsed, "absent", :fallback) === :fallback
        @test get(parsed["diagnostic_options"][1], "scale", nothing) == "log"
        _, _, _, _, parsed_options = TwoPhaseAMRStudio._validate_recording(app, parsed, true)
        @test parsed_options.diagnostic_options == expected
        @test_throws ArgumentError JSON.parse(
            """{"diagnostic_options":[{"names":["signed_pair"],"scale":"log","scale":"linear","hidden":[]}]}""";
            dicttype = TwoPhaseAMRStudio._CommandObject,
        )

        # Reservations are process-wide: a second Studio cannot start a
        # recording or preview while the first preview owns CairoMakie.
        other = studio(recorder; port = 0, open = :none)
        try
            TwoPhaseAMRStudio._reserve_preview_render!()
            try
                @test_throws ArgumentError TwoPhaseAMRStudio._reserve_preview_render!()
                @test_throws ArgumentError TwoPhaseAMRStudio._start_recording!(
                    other, merge(base, Dict("action" => "start", "source" => "live")); one_shot = false,
                )
                @test !other.recording.active
            finally
                TwoPhaseAMRStudio._release_preview_render!()
            end
            TwoPhaseAMRStudio._reserve_recording_render!()
            try
                @test_throws ArgumentError TwoPhaseAMRStudio._reserve_preview_render!()
            finally
                TwoPhaseAMRStudio._release_recording_render!()
            end
        finally
            close(other)
        end
    finally
        close(app)
    end
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
                            (
                                "time", "t_final", "step", "dt", "store_bytes", "store_budget",
                                "frame_count", "steps_per_second", "stopped", "recording",
                            )
                    )
                    @test initial_state["time"] == 0.0 && initial_state["step"] == 0
                    @test initial_state["store_bytes"] > 0
                    @test initial_state["store_budget"] == store.retention.memory
                    @test initial_state["frame_count"] == length(frames(store))
                    @test initial_state["steps_per_second"] === nothing
                    @test initial_state["t_final"] == 100.0
                    @test !initial_state["stopped"]
                    initial_id = first(frames(store)).id
                    @test any(frame -> frame["id"] == initial_id, hello["frames"])
                    initial_summary = only(filter(frame -> frame["id"] == initial_id, hello["frames"]))
                    initial_stats = store[initial_id].stats[:velocity_x]
                    @test initial_summary["stats"]["velocity_x"]["minimum"] == initial_stats.minimum
                    @test initial_summary["stats"]["velocity_x"]["maximum"] == initial_stats.maximum
                    @test initial_summary["stats"]["velocity_x"]["histogram"] == initial_stats.histogram
                    listed = only(filter(frame -> frame["id"] == initial_id, _frames_reply(ws)))
                    @test listed["stats"] == initial_summary["stats"]
                    @test !hello["recording"]["active"]
                    # Exercise a valid parsed WebSocket command, not just
                    # duplicate-key rejection or direct dictionary calls.
                    HTTP.WebSockets.send(ws, """{"type":"get","what":"frames","request":1098}""")
                    parsed_reply = _message(
                        ws, message -> get(message, "request", nothing) == 1098 &&
                            (_istype(message, "frames") || _istype(message, "error")),
                    )
                    @test parsed_reply["type"] == "frames"

                    # Reject invalid or unsupported input instead of silently
                    # dropping options, before creating any recording job.
                    valid_export = Dict{String, Any}(
                        "type" => "recording", "action" => "export", "source" => "stored",
                        "path" => stored_path, "field" => "velocity_x",
                    )
                    invalid_options = (
                        Dict("field" => "missing"),
                        Dict("frames" => "unknown"),
                        Dict("source" => "live"),
                        Dict("diagnostics" => ["kinetic_energy", "kinetic_energy"]),
                        Dict("diagnostic_options" => nothing),
                        Dict(
                            "diagnostics" => ["signed_pair"],
                            "diagnostic_options" => [
                                Dict(
                                    "names" => ["signed_pair"], "scale" => "log",
                                    "hidden" => [Dict("name" => "signed_pair", "component" => 3)]
                                ),
                            ],
                        ),
                        Dict("every" => 0),
                        Dict("duration" => 0),
                        Dict("duration" => -1),
                        Dict("fps" => 0),
                        Dict("fps" => 8.5),
                        Dict("action" => "start", "source" => "live", "sim_per_second" => 0),
                        Dict("action" => "start", "source" => "live", "capacity" => 0),
                        Dict("capacity" => 4),
                        Dict("window" => [2.0, 1.0]),
                        Dict("window" => [1.0e6, 2.0e6]),
                        Dict("camera" => [0, 0, 0, 1]),
                        Dict("colorrange" => [1, 1]),
                        Dict("size" => [0, 100]),
                        Dict("size" => [301, 200]),
                        Dict("size" => [300, 201]),
                        Dict("action" => "start", "source" => "live", "size" => [301, 200]),
                        Dict("arrangement" => "diagonal"),
                        Dict("theme" => "system"),
                        Dict("unrecognized" => true),
                        Dict("action" => "start", "source" => "live", "frames" => "all"),
                        Dict("action" => "start", "source" => "live", "colorrange" => "global"),
                        Dict("action" => "start", "source" => "live", "theme" => "dark"),
                    )
                    for (i, options) in enumerate(invalid_options)
                        request = 1000 + i
                        command = merge(valid_export, options, Dict("request" => request))
                        _send(ws, command)
                        reply = _message(
                            ws, message -> _istype(message, "error") &&
                                get(message, "request", nothing) == request
                        )
                        @test !isempty(reply["message"])
                        if haskey(options, "size") && all(>(0), options["size"])
                            @test occursin("even", reply["message"])
                        end
                    end
                    missing_path = copy(valid_export)
                    delete!(missing_path, "path")
                    missing_path["request"] = 1100
                    _send(ws, missing_path)
                    @test !isempty(
                        _message(
                            ws, message -> _istype(message, "error") &&
                                get(message, "request", nothing) == 1100
                        )["message"]
                    )
                    missing_field = copy(valid_export)
                    delete!(missing_field, "field")
                    missing_field["request"] = 1101
                    _send(ws, missing_field)
                    @test !isempty(
                        _message(
                            ws, message -> _istype(message, "error") &&
                                get(message, "request", nothing) == 1101
                        )["message"]
                    )
                    HTTP.WebSockets.send(
                        ws,
                        """{"type":"recording","action":"preview","frame":1,"scale":"log","scale":"linear"}""",
                    )
                    duplicate_key = _message(
                        ws, message -> _istype(message, "error") &&
                            occursin("duplicate JSON", message["message"]),
                    )
                    @test duplicate_key["request"] === nothing

                    _send(
                        ws, Dict(
                            "type" => "recording", "action" => "start", "source" => "live",
                            "path" => live_path, "field" => "velocity_x", "diagnostics" => ["signed_pair"],
                            "diagnostic_options" => [
                                Dict(
                                    "names" => ["signed_pair"], "scale" => "log",
                                    "hidden" => [Dict("name" => "signed_pair", "component" => 2)]
                                ),
                            ],
                            "camera" => [0, 1, 0, 1], "colorrange" => [-1, 1],
                            "colormap" => "viridis", "phase_rendering" => "cell",
                            "size" => [400, 300], "fps" => 8, "sim_per_second" => 0.08,
                            "capacity" => 32, "arrangement" => "below", "theme" => "dark",
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
                    @test started["progress"] === nothing && started["url"] === nothing
                    _send(
                        ws, Dict(
                            "type" => "recording", "action" => "preview", "request" => 1099,
                            "frame" => initial_id, "field" => "velocity_x",
                        ),
                    )
                    preview_busy = _message(
                        ws, message -> _istype(message, "error") && get(message, "request", nothing) == 1099,
                    )
                    @test occursin("recording or encoding is active", preview_busy["message"])
                    @test TwoPhaseAMRStudio._recording_state(app).id == started["id"]
                    @test TwoPhaseAMRStudio._recording_state(app).active
                    _send(ws, Dict("type" => "recording", "action" => "cancel", "request" => 1102))
                    live_cancel = _message(
                        ws, message -> _istype(message, "error") &&
                            get(message, "request", nothing) == 1102
                    )
                    @test occursin("live recordings must be stopped", live_cancel["message"])

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
                    measured = _message(
                        ws, message -> _istype(message, "state") &&
                            get(message, "step", 0) >= 2 &&
                            get(message, "steps_per_second", nothing) !== nothing &&
                            get(message, "frame_count", -1) == length(frames(store))
                    )
                    @test measured["steps_per_second"] > 0
                    durations = filter(x -> isfinite(x) && x > 0, step_log(store).wall_time)
                    recent = last(durations, min(20, length(durations)))
                    @test measured["steps_per_second"] ≈ length(recent) / sum(recent)
                    @test measured["store_budget"] == store.retention.memory
                    @test measured["frame_count"] == length(frames(store))
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
                    stopped_state = _message(
                        ws, message -> _istype(message, "state") &&
                            get(message, "run_state", "") == "completed" &&
                            get(message, "stopped", false)
                    )
                    @test stopped_state["stopped"]
                    _send(ws, Dict("type" => "get", "what" => "frames", "request" => 1200))
                    summaries = _message(
                        ws, message -> _istype(message, "frames") &&
                            get(message, "request", nothing) == 1200
                    )["frames"]
                    @test [entry["id"] for entry in summaries] == [info.id for info in frames(store)]
                    for entry in summaries
                        stats = store[entry["id"]].stats[:velocity_x]
                        @test entry["stats"]["velocity_x"]["minimum"] == stats.minimum
                        @test entry["stats"]["velocity_x"]["maximum"] == stats.maximum
                        @test entry["stats"]["velocity_x"]["histogram"] == stats.histogram
                    end

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
                    @test stopped["progress"] == 1.0 && stopped["error"] === nothing
                    @test occursin(r"^/download/[0-9a-f]{32}\?token=", stopped["url"])
                    live_download = HTTP.get(address.base * stopped["url"]; status_exception = false)
                    @test live_download.status == 200 && live_download.body == read(live_path)

                    # Gate theme activation so the real one-shot job is
                    # cancellable before its encoder begins, without racing
                    # a fast machine or relying on a synthetic job.
                    cancel_path = joinpath(directory, "cancelled.gif")
                    theme_lock = TwoPhaseAMRStudio.CairoMakie.Makie.THEME_LOCK
                    lock(theme_lock)
                    try
                        _send(
                            ws, Dict(
                                "type" => "recording", "action" => "export", "source" => "stored",
                                "path" => cancel_path, "field" => "velocity_x", "frames" => "all",
                                "size" => [320, 240], "theme" => "dark",
                            )
                        )
                        pending = _message(
                            ws, message -> _istype(message, "state") &&
                                get(message["recording"], "path", nothing) == cancel_path &&
                                get(message["recording"], "status", "") in ("exporting", "error");
                            timeout = _RECORDING_TIMEOUT,
                        )["recording"]
                        @test pending["status"] == "exporting"
                        @test pending["total"] == TwoPhaseAMR.video_frame_count(
                            20, clamp(length(keyframes(store)) / 20, 2.0, 20.0),
                        )
                        @test pending["progress"] == 0.0 && pending["url"] === nothing
                        _send(ws, Dict("type" => "recording", "action" => "cancel"))
                        cancelling = _message(
                            ws, message -> _istype(message, "state") &&
                                get(message["recording"], "path", nothing) == cancel_path &&
                                get(message["recording"], "status", "") == "cancelling",
                        )["recording"]
                        @test cancelling["active"] && cancelling["url"] === nothing
                    finally
                        unlock(theme_lock)
                    end
                    cancelled = _message(
                        ws, message -> _istype(message, "state") &&
                            get(message["recording"], "path", nothing) == cancel_path &&
                            get(message["recording"], "status", "") in ("cancelled", "error");
                        timeout = _RECORDING_TIMEOUT,
                    )["recording"]
                    @test cancelled["status"] == "cancelled" && !cancelled["active"]
                    @test cancelled["encoded"] == 0 && cancelled["progress"] == 0.0
                    @test cancelled["error"] === nothing && cancelled["url"] === nothing
                    @test !isfile(cancel_path)
                    @test HTTP.get(address.base * stopped["url"]; status_exception = false).body == read(live_path)

                    # A stored export is one shot: it freezes its candidates at
                    # submission and encodes the default 20 fps over 2 s.
                    _send(
                        ws, Dict(
                            "type" => "recording", "action" => "export", "source" => "stored",
                            "path" => stored_path, "field" => "velocity_x", "request" => 1202,
                        )
                    )
                    stored = _message(
                        ws, message -> _istype(message, "state") &&
                            get(message["recording"], "source", "") == "stored" &&
                            get(message["recording"], "status", "") in ("completed", "error");
                        timeout = _RECORDING_TIMEOUT
                    )["recording"]
                    @test stored["status"] == "completed"
                    stored["status"] == "completed" || error("Stored export failed: $(stored["error"])")
                    @test !stored["active"] && stored["path"] == stored_path
                    @test stored["encoded"] == TwoPhaseAMR.video_frame_count(
                        20, clamp(length(keyframes(store)) / 20, 2.0, 20.0),
                    )
                    @test stored["dropped"] == 0
                    @test _gif_envelope(stored_path)

                    # The schedule preflight reports the same time-linear plan
                    # the export uses, without starting a job.
                    _send(
                        ws, Dict(
                            "type" => "recording", "action" => "schedule", "request" => 1210,
                            "source" => "stored", "field" => "velocity_x",
                            "frames" => "keyframes", "fps" => 20, "duration" => 2.0,
                        )
                    )
                    plan = _message(
                        ws, message -> _istype(message, "recording_schedule") &&
                            get(message, "request", nothing) == 1210,
                    )
                    keyed = keyframes(store)
                    expected_duration = clamp(length(keyed) / 20, 2.0, 20.0)
                    expected_frames = TwoPhaseAMR.video_frame_count(20, expected_duration)
                    @test plan["frames"] == expected_frames
                    @test plan["distinct"] == length(keyed)
                    @test plan["repeats"] == expected_frames - length(keyed)
                    @test plan["sparse"] == (2 * length(keyed) < expected_frames)
                    @test plan["fps"] == 20 && plan["duration"] ≈ expected_duration
                    @test plan["window"] == [first(keyed).time, last(keyed).time]
                    @test plan["spacing"] ≈ (last(keyed).time - first(keyed).time) / (expected_frames - 1)
                    @test plan["sim_per_second"] ≈ plan["spacing"] * 20
                    @test plan["first_frame"] == first(keyed).id
                    @test !Base.@lock(app.lock, app.recording.active)

                    # Freeze an all-frame, windowed, subsampled selection.
                    # Preview promotion is confined to the export snapshot.
                    index = frames(store)
                    @test any(info -> info.kind === :preview, index)
                    window = [index[2].time, index[end - 1].time]
                    command = Dict{String, Any}(
                        "type" => "recording", "action" => "export", "source" => "stored",
                        "path" => stored_path, "field" => "velocity_x",
                        "diagnostics" => ["kinetic_energy", "signed_pair"], "frames" => "all",
                        "diagnostic_options" => [
                            Dict(
                                "names" => ["signed_pair"], "scale" => "log",
                                "hidden" => [Dict("name" => "signed_pair", "component" => 2)]
                            ),
                        ],
                        "window" => window, "camera" => [0.1, 0.9, 0.2, 0.7],
                        "colorrange" => "global", "colormap" => "viridis",
                        "phase_rendering" => "cell", "theme" => "dark", "arrangement" => "below",
                        "size" => [400, 300], "fps" => 6, "duration" => 1.0,
                    )
                    _, _, _, diagnostics, options = TwoPhaseAMRStudio._validate_recording(app, command, true)
                    frozen, counts = TwoPhaseAMRStudio._freeze_selection(store, diagnostics, options)
                    expected_all = filter(info -> window[1] <= info.time <= window[2], index)
                    expected_ids = [info.id for info in expected_all]
                    @test [frame.id for frame in frozen] == expected_ids
                    @test counts == [
                        length(diagnostic_history(store, :kinetic_energy).times),
                        length(diagnostic_history(store, :signed_pair).times),
                    ]
                    all_frozen, _ = TwoPhaseAMRStudio._freeze_selection(store, Symbol[], options)
                    @test any(frame -> frame.kind === :preview, all_frozen)
                    private_store = TwoPhaseAMRStudio._stored_snapshot(store, Symbol[], all_frozen, Int[])
                    @test [info.id for info in frames(private_store)] == [frame.id for frame in all_frozen]
                    @test any(frame -> frame.kind === :preview, all_frozen)
                    stale = frames(private_store)
                    removed = first(stale).id
                    delete!(private_store.frames, removed)
                    @test all(summary -> summary.id != removed, TwoPhaseAMRStudio._frame_index(private_store, stale))
                    @test first(all_frozen).id == removed # pinned even after removal
                    original_kinds = [info.kind for info in index]
                    options_job = TwoPhaseAMRStudio.RecordingJob()
                    options_job.source = :stored
                    options_job.field = :velocity_x
                    options_job.diagnostics = diagnostics
                    options_job.options = options
                    options_job.selection = frozen
                    exported_options = TwoPhaseAMRStudio._export_options(options_job)
                    selected_stats = mapreduce(frame -> frame.stats[:velocity_x], merge, frozen)
                    @test exported_options.colorrange ==
                        (Float32(selected_stats.minimum), Float32(selected_stats.maximum))
                    @test exported_options.camera == (0.1, 0.9, 0.2, 0.7)
                    @test exported_options.diagnostics == [:kinetic_energy, :signed_pair]
                    @test exported_options.diagnostic_options == [
                        (; names = [:signed_pair], scale = :log, hidden = [(; name = :signed_pair, component = 2)]),
                    ]
                    @test exported_options.colormap == :viridis
                    @test exported_options.phase_rendering == :cell
                    @test exported_options.arrangement == :below
                    @test exported_options.size == (400, 300) && exported_options.fps == 6
                    @test exported_options.duration == 1.0 && exported_options.frames == :all

                    _send(ws, command)
                    exported = _message(
                        ws, message -> _istype(message, "state") &&
                            get(message["recording"], "path", nothing) == stored_path &&
                            get(message["recording"], "id", nothing) != stored["id"] &&
                            get(message["recording"], "status", "") in ("completed", "error");
                        timeout = _RECORDING_TIMEOUT,
                    )["recording"]
                    @test exported["status"] == "completed"
                    exported["status"] == "completed" || error("One-shot export failed: $(exported["error"])")
                    @test exported["encoded"] == TwoPhaseAMR.video_frame_count(6, 1.0)
                    @test exported["dropped"] == 0
                    @test exported["progress"] == 1.0 && exported["error"] === nothing
                    job = Base.@lock app.lock app.recording
                    @test job.selection === nothing && job.diagnostic_counts === nothing
                    @test job.options.theme == :dark
                    @test _gif_envelope(stored_path)
                    gif = read(stored_path)
                    @test Int(gif[7]) + 256 * Int(gif[8]) == 400
                    @test Int(gif[9]) + 256 * Int(gif[10]) == 300
                    @test [info.kind for info in frames(store)] == original_kinds
                    target = exported["url"]
                    @test HTTP.get(address.base * target; status_exception = false).body == gif
                    route = first(split(target, '?'; limit = 2))
                    for suffix in ("", "?token=wrong", "?token=$(address.token)&token=$(address.token)")
                        @test HTTP.get(address.base * route * suffix; status_exception = false).status == 403
                    end
                    @test HTTP.get(address.base * stored["url"]; status_exception = false).status == 404
                    @test HTTP.get(address.base * stopped["url"]; status_exception = false).body == read(live_path)
                    unknown = address.base * "/download/" * repeat("0", 32) * "?token=" * address.token
                    @test HTTP.get(unknown; status_exception = false).status == 404

                    # Preview is a real PNG made with the same export scene,
                    # but does not alter the finished job or its download.
                    preview_command = merge(
                        command, Dict("action" => "preview", "request" => 1300, "frame" => index[2].id),
                    )
                    delete!(preview_command, "path")
                    _send(ws, preview_command)
                    preview = _message(
                        ws, message -> get(message, "request", nothing) == 1300 &&
                            (_istype(message, "recording_preview") || _istype(message, "error"));
                        timeout = _RECORDING_TIMEOUT,
                    )
                    @test preview["type"] == "recording_preview"
                    preview["type"] == "recording_preview" || error("Preview failed: $(preview["message"])")
                    first_preview = preview["url"]
                    png_response = HTTP.get(address.base * first_preview; status_exception = false)
                    @test png_response.status == 200
                    png = png_response.body
                    @test png[1:8] == UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
                    @test Int(ntoh(reinterpret(UInt32, png[17:20])[1])) == 400
                    @test Int(ntoh(reinterpret(UInt32, png[21:24])[1])) == 300
                    preview_route = first(split(first_preview, '?'; limit = 2))
                    @test HTTP.get(address.base * preview_route; status_exception = false).status == 403
                    @test HTTP.get(address.base * preview_route * "?token=wrong"; status_exception = false).status == 403

                    unused_preview_path = joinpath(directory, "never-written.gif")
                    alternate = merge(
                        preview_command, Dict(
                            "request" => 1301, "path" => unused_preview_path,
                            "diagnostic_options" => [
                                Dict("names" => ["signed_pair"], "scale" => "linear", "hidden" => []),
                            ],
                        ),
                    )
                    _send(ws, alternate)
                    second_preview = _message(
                        ws, message -> get(message, "request", nothing) == 1301 &&
                            (_istype(message, "recording_preview") || _istype(message, "error"));
                        timeout = _RECORDING_TIMEOUT,
                    )
                    @test second_preview["type"] == "recording_preview"
                    second_preview["type"] == "recording_preview" ||
                        error("Second preview failed: $(second_preview["message"])")
                    second_png = HTTP.get(address.base * second_preview["url"]; status_exception = false).body
                    @test second_png != png
                    @test !isfile(unused_preview_path)
                    @test HTTP.get(address.base * first_preview; status_exception = false).status == 404
                    @test HTTP.get(address.base * target; status_exception = false).body == gif
                    @test TwoPhaseAMRStudio._recording_state(app).id == exported["id"]
                    @test TwoPhaseAMRStudio._recording_state(app).status == "completed"

                    other_settings = merge(
                        preview_command, Dict(
                            "request" => 1303, "field" => "pressure", "camera" => "fit",
                            "colorrange" => [-1, 1], "theme" => "light", "arrangement" => "right",
                        ),
                    )
                    _send(ws, other_settings)
                    third_preview = _message(
                        ws, message -> get(message, "request", nothing) == 1303 &&
                            (_istype(message, "recording_preview") || _istype(message, "error"));
                        timeout = _RECORDING_TIMEOUT,
                    )
                    @test third_preview["type"] == "recording_preview"
                    third_preview["type"] == "recording_preview" ||
                        error("Third preview failed: $(third_preview["message"])")
                    @test HTTP.get(address.base * third_preview["url"]; status_exception = false).body != second_png
                    @test HTTP.get(address.base * second_preview["url"]; status_exception = false).status == 404

                    _send(ws, merge(preview_command, Dict("request" => 1302, "frame" => typemax(Int))))
                    @test _istype(
                        _message(ws, message -> _istype(message, "error") && get(message, "request", nothing) == 1302),
                        "error",
                    )
                    @test HTTP.get(address.base * third_preview["url"]; status_exception = false).status == 200

                    # An actual export failure must reach clients as recording
                    # state rather than kill the server or claim a completed file.
                    not_directory = joinpath(directory, "regular-file")
                    open(not_directory, "w") do io
                        write(io, "not a directory")
                    end
                    invalid_path = joinpath(not_directory, "invalid.gif")
                    _send(
                        ws, Dict(
                            "type" => "recording", "action" => "export", "source" => "stored",
                            "path" => invalid_path, "field" => "velocity_x", "diagnostics" => String[]
                        )
                    )
                    failed = _message(
                        ws, message -> _istype(message, "state") &&
                            get(message["recording"], "path", nothing) == invalid_path &&
                            get(message["recording"], "status", "") in ("completed", "error");
                        timeout = _RECORDING_TIMEOUT
                    )["recording"]
                    @test failed["status"] == "error" && !failed["active"]
                    @test get(failed, "error", nothing) isa AbstractString && !isempty(failed["error"])
                    @test failed["url"] === nothing
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
            if !isnothing(app)
                close(app)
                @test app.preview === nothing
                @test isempty(app.preview_dirs)
            end
        end
    end
end

@testset "Resumable driver controls and broadcasts" begin
    flow, recorder = _run_fixture()
    store = snapshot_store(recorder)
    entered, release = Channel{Nothing}(1), Channel{Nothing}(1)
    armed = Threads.Atomic{Bool}(true)
    function force(x, y, time, axis)
        if Threads.atomic_cas!(armed, true, false)
            put!(entered, nothing)
            take!(release)
        end
        return 0.0
    end
    driver = StudioDriver(flow, recorder; dt = 0.01, force)
    app = studio(recorder; port = 0, open = :none)
    initial = Threads.@spawn simulate!(driver, 0.02)
    try
        _take_studio(entered, "initial driver step")
        HTTP.WebSockets.open(_ws_address(_address(app))) do first_client
            _message(first_client, message -> _istype(message, "hello"))
            initial_state = _message(first_client, message -> _istype(message, "state"))
            @test initial_state["resumable"]
            @test initial_state["run_state"] == "running"
            @test initial_state["t_final"] == 0.02
            @test initial_state["suggested_final_time"] > 0.02
            HTTP.WebSockets.open(_ws_address(_address(app))) do second_client
                _message(second_client, message -> _istype(message, "hello"))
                _message(second_client, message -> _istype(message, "state"))
                _send(
                    first_client, Dict(
                        "type" => "control", "action" => "continue",
                        "final_time" => 0.03, "request" => 2001,
                    )
                )
                @test _istype(
                    _message(
                        first_client, message ->
                        _istype(message, "error") && get(message, "request", nothing) == 2001
                    ), "error"
                )
                put!(release, nothing)
                _await(() -> istaskdone(initial), "initial driver completion")
                first_result = fetch(initial)
                @test first_result.total_steps == 2 && flow.time == 0.02
                for client in (first_client, second_client)
                    completed = _message(
                        client, message -> _istype(message, "state") &&
                            get(message, "run_state", "") == "completed" &&
                            get(message, "t_final", nothing) == 0.02
                    )
                    @test completed["time"] == 0.02 && !completed["stopped"]
                end
                original_ids = getproperty.(frames(store), :id)
                for (request, value) in zip(2010:2014, (nothing, true, "0.03", 0.02, 0.01))
                    _send(
                        first_client, Dict(
                            "type" => "control", "action" => "continue",
                            "final_time" => value, "request" => request,
                        )
                    )
                    @test _istype(
                        _message(
                            first_client, message ->
                            _istype(message, "error") && get(message, "request", nothing) == request
                        ), "error"
                    )
                end
                _send(
                    first_client, Dict(
                        "type" => "control", "action" => "continue", "request" => 2015,
                    )
                )
                @test _istype(
                    _message(
                        first_client, message ->
                        _istype(message, "error") && get(message, "request", nothing) == 2015
                    ), "error"
                )
                HTTP.WebSockets.send(
                    first_client,
                    """{"type":"control","action":"continue","final_time":1e9999,"request":2016}"""
                )
                @test _istype(_message(first_client, message -> _istype(message, "error")), "error")
                for value in (NaN, Inf, -Inf)
                    @test_throws ArgumentError TwoPhaseAMR._reserve_driver!(driver, value; continuation = true)
                end
                @test flow.time == 0.02

                armed[] = true
                _send(
                    first_client, Dict(
                        "type" => "control", "action" => "continue", "final_time" => 0.04,
                    )
                )
                _take_studio(entered, "continued driver step")
                for client in (first_client, second_client)
                    running = _message(
                        client, message -> _istype(message, "state") &&
                            get(message, "t_final", nothing) == 0.04 &&
                            get(message, "run_state", "") == "running"
                    )
                    @test running["resumable"] && !running["stopped"]
                end
                for (client, request, target) in (
                        (first_client, 2020, 0.04), (second_client, 2021, 0.06),
                    )
                    _send(
                        client, Dict(
                            "type" => "control", "action" => "continue",
                            "final_time" => target, "request" => request,
                        )
                    )
                    @test _istype(
                        _message(
                            client, message ->
                            _istype(message, "error") && get(message, "request", nothing) == request
                        ), "error"
                    )
                end
                put!(release, nothing)
                _await(
                    () -> !TwoPhaseAMR._driver_snapshot(driver).active && flow.time == 0.04,
                    "continued driver completion",
                )
                for client in (first_client, second_client)
                    completed = _message(
                        client, message -> _istype(message, "state") &&
                            get(message, "run_state", "") == "completed" &&
                            get(message, "t_final", nothing) == 0.04
                    )
                    @test completed["time"] == 0.04 && completed["step"] == 4
                    @test completed["suggested_final_time"] ≈ 0.06
                end
                @test all(id -> id in getproperty.(frames(store), :id), original_ids)
                @test count(info -> info.time == 0.02, frames(store)) == 1
                @test step_log(store).step == collect(1:4)
                @test diagnostic_history(store, :kinetic_energy).steps == collect(0:4)

                _send(
                    second_client, Dict(
                        "type" => "control", "action" => "continue", "final_time" => 0.06,
                    )
                )
                _await(
                    () -> !TwoPhaseAMR._driver_snapshot(driver).active && flow.time == 0.06,
                    "second continuation",
                )
                for client in (first_client, second_client)
                    completed = _message(
                        client, message -> _istype(message, "state") &&
                            get(message, "run_state", "") == "completed" &&
                            get(message, "t_final", nothing) == 0.06
                    )
                    @test completed["time"] == 0.06 && completed["step"] == 6
                    @test completed["suggested_final_time"] ≈ 0.08
                end
                @test step_log(store).step == collect(1:6)
                @test diagnostic_history(store, :kinetic_energy).steps == collect(0:6)
            end
        end
    finally
        isready(release) || put!(release, nothing)
        _await(() -> istaskdone(initial), "initial driver teardown")
        _await(() -> !TwoPhaseAMR._driver_snapshot(driver).active, "driver teardown")
        close(app)
    end
end

@testset "Stopped and unowned Studio runs" begin
    flow, recorder = _run_fixture()
    armed = Threads.Atomic{Bool}(true)
    function stop_once(x, y, time, axis)
        Threads.atomic_cas!(armed, true, false) && request_stop!(recorder)
        return 0.0
    end
    driver = StudioDriver(flow, recorder; force = stop_once)
    stopped = simulate!(driver, 0.05)
    @test stopped.stopped && stopped.time == 0.01
    app = studio(recorder; port = 0, open = :none)
    try
        HTTP.WebSockets.open(_ws_address(_address(app))) do ws
            _message(ws, message -> _istype(message, "hello"))
            state = _message(ws, message -> _istype(message, "state"))
            @test state["resumable"] && state["stopped"]
            @test state["time"] == 0.01 && state["t_final"] == 0.05
            @test state["suggested_final_time"] ≈ 0.06
            _send(
                ws, Dict(
                    "type" => "control", "action" => "continue", "final_time" => 0.02,
                )
            )
            completed = _message(
                ws, message -> _istype(message, "state") &&
                    get(message, "run_state", "") == "completed" &&
                    get(message, "t_final", nothing) == 0.02
            )
            @test completed["time"] == 0.02 && !completed["stopped"]
            @test completed["suggested_final_time"] ≈ 0.07
            @test flow.steps == 2
            @test count(event -> event.kind === :stop, run_events(snapshot_store(recorder))) == 1
        end
    finally
        close(app)
    end

    flow, recorder = _run_fixture()
    # A bare closure may run a simulation, but is not an owned run driver.
    run = () -> simulate!(flow, 0.02; observer = recorder)
    run()
    app = studio(recorder; port = 0, open = :none)
    try
        HTTP.WebSockets.open(_ws_address(_address(app))) do ws
            hello = _message(ws, message -> _istype(message, "hello"))
            state = _message(ws, message -> _istype(message, "state"))
            @test !state["resumable"] && isnothing(state["suggested_final_time"])
            @test state["t_final"] == 0.02
            @test only(hello["runs"])["t_final"] == 0.02
            _send(
                ws, Dict(
                    "type" => "control", "action" => "continue",
                    "final_time" => 0.04, "request" => 2030,
                )
            )
            @test _istype(
                _message(
                    ws, message ->
                    _istype(message, "error") && get(message, "request", nothing) == 2030
                ), "error"
            )
            @test flow.time == 0.02 && flow.steps == 2
        end
    finally
        close(app)
    end
end

@testset "Stopped state without captured fields" begin
    flow, _ = _run_fixture()
    recorder = Recorder(VisualizationFieldRegistry(); diagnostics = flow, keyframes = nothing, preview = nothing)
    driver = StudioDriver(flow, recorder)
    request_stop!(recorder)
    @test simulate!(driver, 0.05).stopped
    app = studio(recorder; port = 0, open = :none)
    try
        HTTP.WebSockets.open(_ws_address(_address(app))) do ws
            _message(ws, message -> _istype(message, "hello"))
            state = _message(ws, message -> _istype(message, "state"))
            @test state["time"] == 0 && state["step"] == 0
            @test state["t_final"] == 0.05 && state["stopped"] && state["resumable"]
            _send(ws, Dict("type" => "control", "action" => "continue", "final_time" => 0.02))
            completed = _message(
                ws, message -> _istype(message, "state") &&
                    get(message, "run_state", "") == "completed" &&
                    get(message, "t_final", nothing) == 0.02
            )
            @test completed["time"] == 0.02 && completed["step"] == 2
            @test !completed["stopped"]
        end
    finally
        close(app)
    end
end

@testset "State reports a later target without another step" begin
    flow, recorder = _run_fixture()
    driver = StudioDriver(flow, recorder)
    simulate!(driver, 0.02)
    store = snapshot_store(recorder)
    old = last(frames(store))
    app = studio(recorder; port = 0, open = :none)
    try
        HTTP.WebSockets.open(_ws_address(_address(app))) do ws
            _message(ws, message -> _istype(message, "hello"))
            _message(ws, message -> _istype(message, "state"))
            target = nextfloat(0.02)
            _send(ws, Dict("type" => "control", "action" => "continue", "final_time" => target))
            completed = _message(
                ws, message -> _istype(message, "state") &&
                    get(message, "run_state", "") == "completed" &&
                    get(message, "t_final", nothing) == target
            )
            @test completed["time"] == target && completed["step"] == 2
            @test !completed["stopped"]
            @test store[old.id].time == 0.02
            @test last(frames(store)).time == target
        end
    finally
        close(app)
    end
end
