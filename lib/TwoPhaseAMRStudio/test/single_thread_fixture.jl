using Test
using TwoPhaseAMR
using TwoPhaseAMRStudio
using HTTP
using JSON

function await_condition(predicate, description; timeout = 90.0)
    timedwait(predicate, timeout; pollint = 0.005) === :ok ||
        error("Timed out waiting for $description")
    return nothing
end

function await_message(ws, predicate; timeout = 90.0)
    deadline = time() + timeout
    while time() < deadline
        task = @async HTTP.WebSockets.receive(ws)
        await_condition(
            () -> istaskdone(task), "WebSocket message";
            timeout = max(0.01, deadline - time())
        )
        raw = fetch(task)
        message = raw isa AbstractString ? JSON.parse(raw) : raw
        predicate(message) && return message
    end
    error("Timed out waiting for matching message")
end

kind(message, type) = message isa AbstractDict && get(message, "type", "") == type
send_message(ws, value) = HTTP.WebSockets.send(ws, JSON.json(value))

@testset "Single-thread live recording detaches without blocking the solver" begin
    @test Threads.nthreads(:default) == 1
    @test Threads.nthreads(:interactive) == 0
    mesh = BlockTreeMesh(block_size = (4, 4))
    flow = FlowSimulation(
        mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.05);
        boundary = FreeSlip(), dt = 0.01,
        velocity = (x, y, t, axis) -> axis == 1 ? sinpi(2y) : 0.0,
    )
    fields = VisualizationFieldRegistry()
    register_field!(fields, :velocity_x, flow.velocity, Face{1}())
    recorder = Recorder(
        fields; keyframes = nothing, preview = EveryStep(),
        retention = Retention(preview = :all), max_capture_overhead = 1.0, clock = () -> 0.0
    )
    store = snapshot_store(recorder)
    entered, proceed = Channel{Nothing}(1), Channel{Nothing}(1)
    last_midpoint = Ref(-Inf)
    function force(x, y, midpoint, axis)
        if midpoint > last_midpoint[]
            last_midpoint[] = midpoint
            put!(entered, nothing)
            await_condition(() -> isready(proceed), "simulation release")
            take!(proceed)
        end
        return 0.0
    end
    pause!(recorder)
    simulation = Threads.@spawn simulate!(flow, 1.0; observer = recorder, force)
    app = nothing
    try
        await_condition(() -> run_state(recorder) === :paused || istaskdone(simulation), "initial pause")
        istaskdone(simulation) && fetch(simulation)
        @test run_state(recorder) === :paused
        app = studio(recorder; port = 0, open = :none)
        wsurl = replace(TwoPhaseAMRStudio.url(app), r"^http" => "ws")
        wsurl = replace(wsurl, "/?token=" => "/ws?token=")
        mktempdir() do dir
            path = joinpath(dir, "deferred.gif")
            HTTP.WebSockets.open(wsurl) do ws
                hello = await_message(ws, message -> kind(message, "hello"))
                await_message(ws, message -> kind(message, "state"))
                initial = first(hello["frames"])["id"]
                send_message(
                    ws, Dict(
                        "type" => "recording", "action" => "start", "source" => "live",
                        "path" => path, "field" => "velocity_x", "diagnostics" => String[]
                    )
                )
                started = await_message(
                    ws, message -> kind(message, "state") &&
                        get(message["recording"], "status", "") in ("recording", "error")
                )["recording"]
                @test started["active"] && started["status"] == "recording"
                started["active"] || error("Could not start live recording")
                send_message(ws, Dict("type" => "control", "action" => "resume"))
                await_condition(() -> isready(entered), "first step")
                take!(entered)
                put!(proceed, nothing)
                await_condition(() -> isready(entered), "second step")
                take!(entered)
                send_message(ws, Dict("type" => "recording", "action" => "stop"))
                stopping = await_message(
                    ws, message -> kind(message, "state") &&
                        get(message["recording"], "status", "") in ("stopping", "completed", "error")
                )["recording"]
                @test stopping["status"] == "stopping"
                @test !isfile(path)
                @test run_state(recorder) === :running
                send_message(
                    ws, Dict(
                        "type" => "get", "what" => "field", "frame" => initial,
                        "field" => "velocity_x", "encoding" => "f32", "request" => 80
                    )
                )
                payload = await_message(ws, message -> message isa AbstractVector{UInt8})
                @test decode_field(payload).request == 80
                @test run_state(recorder) === :running
                put!(proceed, nothing)
                for _ in 3:4
                    await_condition(() -> isready(entered), "further step after recording stop")
                    take!(entered)
                    put!(proceed, nothing)
                end
                await_condition(() -> isready(entered), "step at solver stop")
                take!(entered)
                @test length(step_log(store)) >= 4
                @test !istaskdone(simulation) && !isfile(path)
                send_message(ws, Dict("type" => "control", "action" => "stop"))
                await_condition(() -> run_state(recorder) === :stopping, "solver stop request")
                put!(proceed, nothing)
                await_condition(() -> istaskdone(simulation), "solver completion")
                @test fetch(simulation).stopped
                finished = await_message(
                    ws, message -> kind(message, "state") &&
                        get(message["recording"], "status", "") in ("completed", "error");
                    timeout = 240.0
                )["recording"]
                @test finished["status"] == "completed"
                @test !finished["active"] && finished["source"] == "live"
                # Only step 1 arrived before recording stop; later steps must
                # not be mixed into the detached live stream.
                @test finished["encoded"] == 1 && finished["dropped"] == 0
                @test isfile(path) && filesize(path) > 0
            end
        end
    finally
        try
            if !istaskdone(simulation)
                run_state(recorder) in (:created, :running, :pause_requested, :paused, :stopping) &&
                    request_stop!(recorder)
                isready(proceed) || put!(proceed, nothing)
                await_condition(() -> istaskdone(simulation), "simulation teardown")
            end
        finally
            isnothing(app) || close(app)
        end
    end
end
