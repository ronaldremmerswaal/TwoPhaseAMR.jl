using Test, TwoPhaseAMR

const _RECORDING_TIMEOUT = 60.0

function _fetch_with_timeout(task; timeout = _RECORDING_TIMEOUT)
    status = timedwait(() -> istaskdone(task), timeout; pollint = 0.001)
    status === :ok || error("recording test task timed out")
    return fetch(task)
end

function _take_with_timeout(channel; timeout = _RECORDING_TIMEOUT)
    status = timedwait(() -> isready(channel), timeout; pollint = 0.001)
    status === :ok || error("recording test channel timed out")
    return take!(channel)
end

_wait_paused_with_timeout(recorder) =
    _fetch_with_timeout(@async wait_paused(recorder))

# Drive a recorder by hand through a cell field whose value is the time.
mutable struct ManualRun
    recorder::Recorder
    field::CellField
    final_time::Float64
    time::Float64
end

function _start!(recorder, field; time = 0.0, final_time = 1.0)
    initialize!(field, (x, y) -> time)
    observe!(
        recorder, :initial, SimulationState(nothing, field, nothing, time, 0),
        StepInfo(; dt = 0.0, final_time),
    )
    return ManualRun(recorder, field, final_time, time)
end

function _step!(run::ManualRun, time, step; event = :step)
    initialize!(run.field, (x, y) -> time)
    dt = time - run.time
    run.time = time
    observe!(
        run.recorder, event, SimulationState(nothing, run.field, nothing, time, step),
        StepInfo(; dt, final_time = run.final_time),
    )
    return run
end

_scheduled(store) = [
    (requested, info.time) for info in frames(store)
        for requested in info.requested_times
]

function _state_registry(mesh = BlockTreeMesh(block_size = (4, 4)))
    field = CellField(mesh)
    registry = VisualizationFieldRegistry()
    register_field!(registry, :state, field)
    return registry, field
end

function _decaying_flow(; dt = 0.01, diagnostics = (KineticEnergy(),))
    mesh = BlockTreeMesh(block_size = (4, 4))
    return FlowSimulation(
        mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.05);
        boundary = FreeSlip(), dt, diagnostics,
        velocity = (x, y, time, axis) -> axis == 1 ? sinpi(2y) : 0.0,
    )
end

mutable struct DispatchObserver <: AbstractObserver
    flow::FlowSimulation
    events::Vector{NamedTuple}
    stop_at::Int
    fail_event::Symbol
    fail_poll::Int
    polls::Int
end

DispatchObserver(flow; stop_at = typemax(Int), fail_event = :none, fail_poll = 0) =
    DispatchObserver(flow, NamedTuple[], stop_at, fail_event, fail_poll, 0)

function TwoPhaseAMR.observe!(observer::DispatchObserver, event::Symbol, state, info::StepInfo)
    history = diagnostic_history(observer.flow, :step_marker)
    push!(
        observer.events, (;
            event, time = state.time, step = state.step, dt = info.dt,
            final_time = info.final_time, topology_changed = info.topology_changed,
            converged = info.projection.converged,
            independent_step = !isnothing(info.step_procedure),
            exception = info.exception, wall_time = info.wall_time,
            diagnostic_step = last(history.steps),
            same_fields = state.velocity === observer.flow.velocity &&
                state.pressure === observer.flow.pressure &&
                state.solver === observer.flow.solver &&
                state.workspace === observer.flow.workspace,
        )
    )
    event === observer.fail_event && throw(ErrorException("observer callback failed"))
    return nothing
end

function TwoPhaseAMR.stop_requested(observer::DispatchObserver)
    observer.polls += 1
    observer.polls == observer.fail_poll && throw(ErrorException("observer poll failed"))
    return !isempty(observer.events) && last(observer.events).step >= observer.stop_at
end

mutable struct InterruptObserver <: AbstractObserver
    fail_event::Symbol
    fail_poll::Int
    polls::Int
    events::Vector{Symbol}
end

function TwoPhaseAMR.observe!(observer::InterruptObserver, event::Symbol, state, info::StepInfo)
    push!(observer.events, event)
    event === observer.fail_event && throw(InterruptException())
    return nothing
end

function TwoPhaseAMR.stop_requested(observer::InterruptObserver)
    observer.polls += 1
    observer.polls == observer.fail_poll && throw(InterruptException())
    return false
end

function _observer_flow()
    flow = _decaying_flow()
    add_diagnostic!(flow, :step_marker, state -> state.step)
    return flow
end

@testset "Custom observers receive completed solver states" begin
    flow = _observer_flow()
    observer = DispatchObserver(flow)
    result = simulate!(flow, 0.025; observer)
    events = observer.events
    @test !result.stopped && result.steps == 3 && flow.steps == 3
    @test getproperty.(events, :event) == [:initial, :step, :step, :step, :completed]
    @test getproperty.(events, :step) == [0, 1, 2, 3, 3]
    # Three additions/subtractions of a Float64 step incur only roundoff.
    @test getproperty.(events, :time) ≈ [0.0, 0.01, 0.02, 0.025, 0.025] atol = 16eps(Float64) rtol = 0
    @test getproperty.(events, :dt) ≈ [0.0, 0.01, 0.01, 0.005, 0.0] atol = 16eps(Float64) rtol = 0
    @test all(event -> event.final_time == 0.025, events)
    @test all(event -> event.diagnostic_step == event.step, events)
    @test all(event -> event.same_fields && !event.topology_changed, events)
    @test all(event -> isnothing(event.exception) && event.wall_time >= 0, events)
    @test all(event -> event.converged, events[2:4])
    @test getproperty.(events, :independent_step) == [true, false, false, false, false]
    @test observer.polls == 4

    # A resumed call starts at the accumulated step and diagnostic history.
    resumed = DispatchObserver(flow)
    result = simulate!(flow, 0.035; observer = resumed)
    @test !result.stopped && result.steps == 1 && result.total_steps == 4
    @test getproperty.(resumed.events, :event) == [:initial, :step, :completed]
    @test getproperty.(resumed.events, :step) == [3, 4, 4]
    @test all(event -> event.diagnostic_step == event.step, resumed.events)

    stopped_flow = _observer_flow()
    stopped = DispatchObserver(stopped_flow; stop_at = 2)
    result = simulate!(stopped_flow, 0.05; observer = stopped)
    @test result.stopped && result.steps == 2 && result.time ≈ 0.02
    @test getproperty.(stopped.events, :event) == [:initial, :step, :step, :completed]
    @test getproperty.(stopped.events, :step) == [0, 1, 2, 2]
    @test all(event -> event.diagnostic_step == event.step, stopped.events)
end

@testset "Observer errors are isolated after initialization" begin
    baseline = _observer_flow()
    simulate!(baseline, 0.025)

    for (fail_event, fail_poll) in ((:step, 0), (:completed, 0), (:none, 1), (:none, 2))
        flow = _observer_flow()
        observer = DispatchObserver(flow; fail_event, fail_poll)
        result = simulate!(flow, 0.025; observer)
        @test !result.stopped && result.steps == 3 && result.time == 0.025
        @test flow.steps == 3 && flow.time == 0.025
        @test fail_event === :none || any(event -> event.event === fail_event, observer.events)
        @test iszero(fail_poll) || observer.polls >= fail_poll
        # Identical 4×4 solves should differ by at most floating-point roundoff.
        @test flow.velocity.interior ≈ baseline.velocity.interior atol = 1.0e-12 rtol = 1.0e-12
        @test visualization_values(flow.pressure) ≈ visualization_values(baseline.pressure) atol = 1.0e-12 rtol = 1.0e-12
    end

    initial_flow = _observer_flow()
    initial = DispatchObserver(initial_flow; fail_event = :initial)
    @test_throws ErrorException simulate!(initial_flow, 0.025; observer = initial)
    @test initial_flow.time == 0.0 && initial_flow.steps == 0
    @test getproperty.(initial.events, :event) == [:initial]

    # A failing error callback must not replace the actual solver exception.
    failing_flow = _observer_flow()
    failing = DispatchObserver(failing_flow; fail_event = :failed)
    solver_error = ErrorException("force evaluation failed")
    force = (x, y, time, axis) -> throw(solver_error)
    caught = try
        simulate!(failing_flow, 0.025; observer = failing, force)
        nothing
    catch exception
        exception
    end
    @test caught === solver_error
    @test getproperty.(failing.events, :event) == [:initial, :failed]
    @test last(failing.events).exception === solver_error
    @test last(failing.events).step == 0
end

@testset "Observer interrupts escape simulation hooks" begin
    for event in (:initial, :step, :completed)
        observer = InterruptObserver(event, 0, 0, Symbol[])
        @test_throws InterruptException simulate!(_decaying_flow(), 0.02; observer)
        @test event in observer.events
    end
    for poll in (1, 2)
        observer = InterruptObserver(:none, poll, 0, Symbol[])
        @test_throws InterruptException simulate!(_decaying_flow(), 0.02; observer)
        @test observer.polls == poll
    end
end

@testset "Keyframe schedules select the nearest completed state" begin
    registry, field = _state_registry()
    recorder = Recorder(registry; keyframes = EquispacedTimes(5), preview = nothing)
    @test isnothing(newest_frame(snapshot_store(recorder)))
    run = _start!(recorder, field)
    for (step, time) in enumerate((0.18, 0.44, 0.68))
        _step!(run, time, step)
    end
    _step!(run, 1.0, 4)
    _step!(run, 1.0, 4; event = :completed)
    store = snapshot_store(recorder)
    @test _scheduled(store) == [
        (0.0, 0.0), (0.25, 0.18), (0.5, 0.44), (0.75, 0.68), (1.0, 1.0),
    ]
    @test [info.kind for info in keyframes(store)] ==
        [:initial, :keyframe, :keyframe, :keyframe, :final]
    @test all(all(==(info.time), store[info][:state]) for info in frames(store))
    # Frames own their values: later solver updates do not change them.
    fill!(interior(field, 1), 99.0)
    @test all(iszero, store[first(frames(store))][:state])
    @test run_state(recorder) === :completed

    interval = Recorder(registry; keyframes = EveryTimeInterval(0.3), preview = nothing)
    interval_run = _start!(interval, field)
    for (step, time) in enumerate(0.1:0.1:1.0)
        _step!(interval_run, time, step)
    end
    _step!(interval_run, 1.0, 10; event = :completed)
    @test first.(_scheduled(snapshot_store(interval))) ≈ [0.0, 0.3, 0.6, 0.9, 1.0]
    @test last.(_scheduled(snapshot_store(interval))) ≈ [0.0, 0.3, 0.6, 0.9, 1.0]

    explicit = Recorder(
        registry; keyframes = AtTimes([0.7, 0.2, 0.7]), preview = nothing,
    )
    explicit_run = _start!(explicit, field)
    for (step, time) in enumerate((0.3, 0.6, 0.9, 1.0))
        _step!(explicit_run, time, step)
    end
    _step!(explicit_run, 1.0, 4; event = :completed)
    explicit_store = snapshot_store(explicit)
    @test _scheduled(explicit_store) == [(0.2, 0.3), (0.7, 0.6)]
    # Initial and final states are always stored.
    @test [info.kind for info in frames(explicit_store)] ==
        [:initial, :keyframe, :keyframe, :final]
    @test newest_frame(explicit_store).time == 1.0
    @test frame_at(explicit_store, 0.62).time == 0.6

    single = Recorder(registry; keyframes = EquispacedTimes(1), preview = nothing)
    single_run = _start!(single, field; final_time = 2.0)
    _step!(single_run, 2.0, 1)
    _step!(single_run, 2.0, 1; event = :completed)
    @test _scheduled(snapshot_store(single)) == [(2.0, 2.0)]

    stepped = Recorder(registry; keyframes = EveryNSteps(2), preview = nothing)
    stepped_run = _start!(stepped, field)
    for step in 1:5
        _step!(stepped_run, step / 10, step)
    end
    _step!(stepped_run, 0.5, 5; event = :completed)
    @test [info.step for info in keyframes(snapshot_store(stepped))] == [0, 2, 4, 5]

    outside = Recorder(registry; keyframes = AtTimes([-0.1, 0.5]))
    @test_throws ArgumentError _start!(outside, field)
    @test run_state(outside) === :failed

    @test_throws ArgumentError EquispacedTimes(0)
    @test_throws ArgumentError EquispacedTimes(true)
    @test_throws ArgumentError AtTimes([NaN])
    @test_throws ArgumentError EveryWallInterval(0)
    @test_throws ArgumentError Recorder(registry; keyframes = EveryWallInterval(1))
    @test_throws ArgumentError Recorder(registry; preview = AtTimes([0.5]))
    @test_throws ArgumentError Recorder(registry; max_capture_overhead = 0)
    @test_throws ArgumentError Recorder(registry; preview_precision = Int)
    @test_throws ArgumentError Recorder(registry; diagnostics = 1)
end

@testset "Preview promotion preserves native precision" begin
    value = 0.1
    rounded_time = nextfloat(value)
    @test Float64(Float32(value)) != value

    registry, field = _state_registry()
    recorder = Recorder(
        registry; keyframes = nothing, preview = EveryStep(),
        preview_precision = Float32, retention = Retention(preview = :all),
        max_capture_overhead = 1.0, clock = () -> 0.0,
    )
    store = snapshot_store(recorder)
    subscription = subscribe(store; capacity = 4)
    run = _start!(recorder, field; final_time = rounded_time)
    _step!(run, value, 1)
    preview = newest_frame(store)
    @test preview.kind === :preview && preview.step == 1
    @test eltype(preview[:state]) === Float32
    @test all(==(Float32(value)), preview[:state])
    before = store_usage(store)
    @test take!(subscription).kind === :initial
    @test take!(subscription) === preview

    # The solver can relabel the final time without changing the fields.
    observe!(
        recorder, :completed, SimulationState(nothing, field, nothing, rounded_time, 1),
        StepInfo(; dt = 0.0, final_time = rounded_time),
    )
    final = newest_frame(store)
    @test length(frames(store)) == 2
    @test final.id == preview.id && final.step == 1
    @test final.kind === :final && final.time == rounded_time
    @test eltype(final[:state]) === Float64
    @test all(==(value), final[:state])
    @test final.stats[:state].minimum == value
    @test preview.kind === :preview && eltype(preview[:state]) === Float32
    @test all(==(Float32(value)), preview[:state])
    @test all(info -> store[info].info === info, frames(store))
    @test frame_at(store, value) === final
    @test store_usage(store).bytes - before.bytes ==
        length(final[:state]) * (sizeof(Float64) - sizeof(Float32))
    close(subscription)

    registry, field = _state_registry()
    paused_recorder = Recorder(
        registry; keyframes = nothing, preview = EveryStep(),
        preview_precision = Float32, retention = Retention(preview = :all),
        max_capture_overhead = 1.0, clock = () -> 0.0,
    )
    paused_store = snapshot_store(paused_recorder)
    subscription = subscribe(paused_store; capacity = 4)
    run = _start!(paused_recorder, field)
    pause!(paused_recorder)
    task = @async _step!(run, value, 1)
    try
        _wait_paused_with_timeout(paused_recorder)
        paused = newest_frame(paused_store)
        @test length(frames(paused_store)) == 2
        @test paused.kind === :pause && paused.time == value && paused.step == 1
        @test eltype(paused[:state]) === Float64
        @test all(==(value), paused[:state])
        @test paused.stats[:state].minimum == value
        @test all(info -> paused_store[info].info === info, frames(paused_store))
        if length(frames(paused_store)) == 2
            @test take!(subscription).kind === :initial
            preview = take!(subscription)
            @test preview.kind === :preview && preview.id == paused.id
            @test eltype(preview[:state]) === Float32
            @test all(==(Float32(value)), preview[:state])
        end
    finally
        if !istaskdone(task)
            state = run_state(paused_recorder)
            state in (:completed, :failed) || request_stop!(paused_recorder)
            _fetch_with_timeout(task)
        end
        close(subscription)
    end
    _fetch_with_timeout(task)
    _step!(run, value, 1; event = :completed)
    final = newest_frame(paused_store)
    @test final.kind === :final && final.time == value && final.step == 1
    @test eltype(final[:state]) === Float64
    @test all(==(value), final[:state])
    @test length(frames(paused_store)) == 2
    @test all(info -> paused_store[info].info === info, frames(paused_store))

    # Final-time relabeling must retain an already committed schedule entry.
    registry, field = _state_registry()
    scheduled = Recorder(
        registry; keyframes = AtTimes([value]), preview = EveryStep(),
        preview_precision = Float32, max_capture_overhead = 1.0, clock = () -> 0.0,
    )
    run = _start!(scheduled, field; final_time = rounded_time)
    _step!(run, value, 1)
    store = snapshot_store(scheduled)
    keyframe = newest_frame(store)
    observe!(
        scheduled, :completed, SimulationState(nothing, field, nothing, rounded_time, 1),
        StepInfo(; dt = 0.0, final_time = rounded_time),
    )
    final = newest_frame(store)
    @test final.id == keyframe.id && final.time == rounded_time && final.step == 1
    @test final.kind === :final && final.requested_times == [value]
    @test eltype(final[:state]) === Float64 && all(==(value), final[:state])
    @test length(frames(store)) == 2
    @test all(info -> store[info].info === info, frames(store))

    # An uncommitted native nearest-state candidate can become the final state.
    registry, field = _state_registry()
    pending = Recorder(registry; keyframes = AtTimes([0.15]), preview = nothing)
    run = _start!(pending, field)
    _step!(run, value, 1)
    store = snapshot_store(pending)
    @test length(frames(store)) == 1
    _step!(run, value, 1; event = :completed)
    final = newest_frame(store)
    @test final.kind === :final && final.step == 1 && final.time == value
    @test eltype(final[:state]) === Float64 && all(==(value), final[:state])
    @test length(frames(store)) == 2
    @test all(info -> store[info].info === info, frames(store))

    # A larger next step can select a seemingly distant preview as nearest.
    registry, field = _state_registry()
    nearest = Recorder(
        registry; keyframes = AtTimes([0.35]), preview = EveryStep(),
        preview_precision = Float32, retention = Retention(preview = :all),
        max_capture_overhead = 1.0, clock = () -> 0.0,
    )
    run = _start!(nearest, field)
    _step!(run, value, 1)
    store = snapshot_store(nearest)
    preview = newest_frame(store)
    @test preview.kind === :preview && eltype(preview[:state]) === Float32
    @test all(==(Float32(value)), preview[:state])
    _step!(run, 0.8, 2)
    selected = only(filter(info -> info.requested_times == [0.35], keyframes(store)))
    @test selected.id == preview.id && selected.step == 1 && selected.time == value
    @test eltype(store[selected][:state]) === Float64
    @test all(==(value), store[selected][:state])
end

@testset "Frames share captured sources and own derived values" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    velocity = FaceVelocity(mesh, (x, y, time, axis) -> axis == 1 ? 3.0 : 4.0)
    registry = VisualizationFieldRegistry()
    register_field!(registry, :velocity_x, velocity, Face{1}())
    register_field!(registry, :again_x, velocity, Face{1}())
    register_field!(registry, :velocity_magnitude, velocity, CentredMagnitude())
    register_field!(registry, :vorticity, velocity, Vorticity())
    xfield, yfield = FaceField{1}(velocity), FaceField{2}(velocity)
    register_field!(registry, :x, xfield)
    register_field!(registry, :magnitude, (xfield, yfield), CentredMagnitude())
    frame = TwoPhaseAMR._registry_frame(registry)
    @test frame[:again_x] === frame[:velocity_x]
    @test frame[:velocity_x] == visualization_values(velocity, Face{1}())
    @test frame[:x] == visualization_values(xfield)
    @test all(==(5.0), frame[:velocity_magnitude])
    @test all(==(5.0), frame[:magnitude])
    @test all(iszero, frame[:vorticity])
    @test field_names(frame) == field_names(registry)
    @test frame.stats[:velocity_magnitude].minimum == 5.0
    @test frame.stats[:velocity_magnitude].finite == ncells(mesh)
    # A frame lists only the fields captured with it.
    register_field!(registry, :late, CellField(mesh))
    @test !haskey(frame, :late)
    @test_throws ArgumentError frame[:late]
end

@testset "Preview throttle on wall-clock time and capture overhead" begin
    registry, field = _state_registry()
    now = Ref(0.0)
    recorder = Recorder(
        registry; keyframes = nothing, preview = EveryWallInterval(1.0),
        clock = () -> now[], retention = Retention(preview = :all),
    )
    run = _start!(recorder, field)
    for step in 1:10
        now[] = 0.4step
        _step!(run, step / 10, step)
    end
    previews = filter(info -> info.kind === :preview, frames(snapshot_store(recorder)))
    @test [info.step for info in previews] == [3, 6, 9]
    @test [info.wall_time for info in previews] ≈ [1.2, 2.4, 3.6]

    # The clock advances by one unit per reading. Steps that are long compared
    # with a capture keep every preview; short steps are throttled so previews
    # cost at most max_capture_overhead of the elapsed time.
    for (advance, throttled) in ((1000.0, false), (5.0, true))
        ticks = Ref(0.0)
        ticking = Recorder(
            registry; keyframes = nothing, preview = EveryStep(),
            clock = () -> (ticks[] += 1.0), max_capture_overhead = 0.05,
            retention = Retention(preview = :all),
        )
        ticking_run = _start!(ticking, field)
        for step in 1:40
            ticks[] += advance
            _step!(ticking_run, step / 40, step)
        end
        usage = store_usage(snapshot_store(ticking))
        @test (usage.skipped_previews > 0) == throttled
        @test usage.previews + usage.skipped_previews == 40
    end
end

@testset "Retention keeps keyframes, a dense recent window, and the budget" begin
    registry, field = _state_registry(BlockTreeMesh(block_size = (8, 8)))
    function preview_run(retention; steps = 300, keyframes = EveryNSteps(100))
        recorder = Recorder(
            registry; keyframes, preview = EveryStep(), retention,
            max_capture_overhead = 1.0, clock = () -> 0.0,
        )
        run = _start!(recorder, field; final_time = steps / 100)
        for step in 1:steps
            _step!(run, step / 100, step)
        end
        _step!(run, steps / 100, steps; event = :completed)
        return snapshot_store(recorder)
    end

    store = preview_run(Retention(preview = :logarithmic, recent = 8))
    infos = frames(store)
    previews = filter(info -> info.kind === :preview, infos)
    @test [info.step for info in keyframes(store)] == [0, 100, 200, 300]
    @test issorted([info.time for info in infos])
    # The newest previews are dense; older ones thin out logarithmically.
    recent_steps = [info.step for info in previews[(end - 7):end]]
    @test recent_steps == collect((recent_steps[1]):(recent_steps[1] + 7))
    @test length(previews) <= 8 * (1 + log2(297 / 8)) + 1
    @test store_usage(store).thinned_previews == 297 - length(previews)
    @test all(info -> haskey(store, info.id), infos)

    @test store_usage(preview_run(Retention(preview = :none))).previews == 1
    @test store_usage(preview_run(Retention(preview = :all))).previews == 297

    frame_bytes = TwoPhaseAMR._registry_frame(registry).bytes
    budget = 40 * frame_bytes
    bounded = preview_run(Retention(memory = budget, preview = :uniform))
    usage = store_usage(bounded)
    @test usage.bytes <= budget
    @test usage.keyframes == 4
    @test usage.previews > 4
    @test issorted([info.time for info in frames(bounded)])

    # Keyframes are never dropped, even when they alone exceed the budget.
    over = @test_logs (:warn,) match_mode = :any preview_run(
        Retention(memory = frame_bytes, preview = :uniform); steps = 20,
        keyframes = EveryStep(),
    )
    @test store_usage(over).keyframes == 21
    @test_throws ArgumentError Retention(memory = 0)
    @test_throws ArgumentError Retention(preview = :fifo)
    @test_throws ArgumentError Retention(recent = 0)
end

@testset "Topology snapshots are shared until the leaf set changes" begin
    mesh = BlockTreeMesh(block_size = (4, 4), max_level = 1)
    field = CellField(mesh)
    initialize!(field, (x, y) -> x + y)
    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y),
    )
    registry = VisualizationFieldRegistry()
    register_field!(registry, :state, field)
    register_field!(registry, :vorticity, velocity, Vorticity())
    recorder = Recorder(registry; keyframes = EveryStep(), preview = nothing)
    state(time, step) = SimulationState(velocity, field, nothing, time, step)
    observe!(recorder, :initial, state(0.0, 0), StepInfo(; dt = 0.0, final_time = 1.0))
    amr = AMRWorkspace(mesh, field, velocity)
    generation = mesh.topology_cache.generation
    @test !adapt!(amr, fill(:keep, nblocks(mesh)); step = 0).changed
    @test mesh.topology_cache.generation == generation
    observe!(recorder, :step, state(0.5, 1), StepInfo(; dt = 0.5, final_time = 1.0))
    @test adapt!(amr, [:refine]; step = 1).changed
    @test mesh.topology_cache.generation != generation
    initialize!(field, (x, y) -> 2x - y)
    observe!(recorder, :step, state(1.0, 2), StepInfo(; dt = 0.5, final_time = 1.0))
    observe!(recorder, :completed, state(1.0, 2), StepInfo(; dt = 0.0, final_time = 1.0))

    store = snapshot_store(recorder)
    coarse, unchanged, fine = (store[info] for info in frames(store))
    @test coarse.topology === unchanged.topology
    @test fine.topology !== coarse.topology
    @test store_usage(store).topologies == 2
    @test length(coarse[:state]) == length(coarse[:vorticity]) == 16
    @test length(fine[:state]) == length(fine[:vorticity]) == 64
    # Old frames stay valid after adaptation of the live mesh.
    for (frame, sample) in ((coarse, (x, y) -> x + y), (fine, (x, y) -> 2x - y))
        cell = locate(frame.topology, (0.3, 0.6))
        index = TwoPhaseAMR._cell_index(cell, 4, 4)
        center = (
            frame.topology.origins[cell.block] .+
                (cell.i - 0.5, cell.j - 0.5) .* frame.topology.spacings[cell.block]
        )
        @test frame[:state][index] ≈ sample(center...)
    end
end

@testset "Exact keyframes use independent side steps" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    force = (x, y, time, axis) -> axis == 1 ? 0.1sinpi(2y) : 0.0
    function solve(observer)
        velocity = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
        pressure = CellField(mesh)
        solver = IncompressibleNavierStokesSolver(
            mesh; density = 1.0, dynamic_viscosity = 0.01,
            boundary = FreeSlip(), coarse_size = 8,
        )
        workspace = NavierStokesWorkspace(solver)
        isnothing(observer) || begin
            register_field!(observer.registry, :pressure, pressure)
            register_field!(observer.registry, :velocity_x, velocity, Face{1}())
        end
        result = simulate!(
            velocity, pressure, solver, 1.0, workspace;
            dt = 0.3, force, tolerance = 1.0e-10, max_cycles = 20,
            observer = isnothing(observer) ? NoObserver() : observer,
        )
        return result, velocity, pressure
    end
    baseline, baseline_velocity, baseline_pressure = solve(nothing)
    recorder = Recorder(
        VisualizationFieldRegistry(); keyframes = AtTimes([0.2, 0.65]),
        exact_keyframes = true, preview = nothing,
    )
    result, velocity, pressure = solve(recorder)
    @test result.steps == baseline.steps == 4
    @test result.time == 1.0
    @test velocity.interior == baseline_velocity.interior
    @test velocity.boundary == baseline_velocity.boundary
    @test visualization_values(pressure) == visualization_values(baseline_pressure)
    store = snapshot_store(recorder)
    @test _scheduled(store) == [(0.2, 0.2), (0.65, 0.65)]
    exact = store[only(filter(info -> info.time == 0.2, frames(store)))]
    @test exact.kind === :keyframe
    @test any(!iszero, exact[:velocity_x])
    @test exact[:velocity_x] != store[first(frames(store))][:velocity_x]

    adaptive = Recorder(
        VisualizationFieldRegistry(); keyframes = AtTimes([0.5]),
        exact_keyframes = true,
    )
    @test !TwoPhaseAMR._supports_dynamic_topology(adaptive)
end

@testset "Recording during simulate! mirrors diagnostics and logs steps" begin
    flow = _decaying_flow(; diagnostics = (KineticEnergy(), LinearMomentum(), CellCount()))
    fields = VisualizationFieldRegistry()
    register_default_fields!(fields, flow)
    recorder = Recorder(fields; diagnostics = flow, keyframes = EquispacedTimes(3), preview = nothing)
    result = simulate!(flow, 0.1; observer = recorder)
    @test !result.stopped
    @test result.steps == 10
    store = snapshot_store(recorder)
    @test [info.time for info in keyframes(store)] ≈ [0.0, 0.05, 0.1]
    @test diagnostic_names(store) == [:kinetic_energy, :linear_momentum, :cells]
    for name in diagnostic_names(flow)
        mirrored = diagnostic_history(store, name)
        original = diagnostic_history(flow, name)
        @test mirrored.times == original.times
        @test mirrored.steps == original.steps
        @test mirrored.values ==
            reduce(vcat, (permutedims(Float64[value...]) for value in original.values))
    end
    @test diagnostic_history(store, :linear_momentum).components == ["x", "y"]
    @test diagnostic_history(store, :kinetic_energy).info == diagnostic_info(KineticEnergy())
    log = step_log(store)
    @test length(log) == 10
    @test log.step == 1:10
    @test log.time ≈ 0.01:0.01:0.1
    @test all(==(0.01), @view log.dt[1:(end - 1)])
    # The last step reaches the target exactly; accumulated time can differ
    # from the ideal 0.09 by a few ulps.
    @test last(log.dt) == 0.1 - log.time[end - 1]
    @test isapprox(last(log.dt), 0.01; atol = 16eps(0.1), rtol = 0)
    @test all(log.converged)
    @test !any(log.topology_changed)
    @test all(>=(0), log.wall_time)
    @test all(>=(0), log.capture_time)

    # A second call continues the same recording.
    simulate!(flow, 0.15; observer = recorder)
    @test length(step_log(store)) == 15
    @test diagnostic_history(store, :kinetic_energy).times ==
        diagnostic_history(flow, :kinetic_energy).times
    @test last(frames(store)).time ≈ 0.15

    # Configuration errors are reported before the first step.
    other = Recorder(_state_registry()[1])
    @test_throws ArgumentError simulate!(flow, 0.2; observer = other)
    @test flow.time ≈ 0.15
    @test run_state(other) === :failed
end

@testset "A slow consumer never blocks the solver" begin
    registry, field = _state_registry()
    gate = Channel{Nothing}(1)
    recorder = Recorder(
        registry; keyframes = nothing, preview = EveryStep(),
        max_capture_overhead = 1.0, clock = () -> 0.0,
    )
    subscription = subscribe(snapshot_store(recorder))
    received = Frame[]
    consumer = @async begin
        while true
            frame = take!(subscription)
            isnothing(frame) && break
            push!(received, frame)
            # Deliberately stall on the first frame until the run has ended.
            length(received) == 1 && _take_with_timeout(gate)
        end
    end
    run = _start!(recorder, field)
    yield()
    for step in 1:50
        _step!(run, step / 50, step)
    end
    _step!(run, 1.0, 50; event = :completed)
    # The run finished while the consumer was stalled.
    @test length(received) == 1
    @test !istaskdone(consumer)
    put!(gate, nothing)
    timedwait(() -> length(received) >= 2, _RECORDING_TIMEOUT; pollint = 0.001)
    close(subscription)
    _fetch_with_timeout(consumer)
    # Only the newest frame was delivered after the stall.
    @test last(received).id == last(frames(snapshot_store(recorder))).id
    @test subscription.dropped == length(frames(snapshot_store(recorder))) - 2

    flow = _decaying_flow()
    timed = Recorder(VisualizationFieldRegistry(); diagnostics = flow, keyframes = nothing)
    elapsed_free = @elapsed simulate!(flow, 0.3; observer = timed)
    fields = VisualizationFieldRegistry()
    register_field!(fields, :pressure, flow.pressure)
    stalled = Recorder(fields; keyframes = nothing, preview = EveryStep())
    blocked = subscribe(snapshot_store(stalled))
    stuck = Channel{Nothing}(1)
    stalled_consumer = @async (take!(blocked); _take_with_timeout(stuck))
    elapsed_stalled = @elapsed simulate!(flow, 0.6; observer = stalled)
    @test elapsed_stalled < 4elapsed_free + 2.0
    put!(stuck, nothing)
    _fetch_with_timeout(stalled_consumer)
    @test blocked.dropped > 0
end

@testset "Subscriptions keep the frames that retention thins" begin
    registry, field = _state_registry()
    recorder = Recorder(
        registry; keyframes = nothing, preview = EveryStep(),
        retention = Retention(preview = :none), max_capture_overhead = 1.0,
        clock = () -> 0.0,
    )
    store = snapshot_store(recorder)
    queued = subscribe(store; capacity = 64)
    latest = subscribe(store; capacity = 1)
    run = _start!(recorder, field)
    for step in 1:20
        _step!(run, step / 20, step)
    end
    _step!(run, 1.0, 20; event = :completed)
    close(queued)
    received = Frame[]
    while (frame = take!(queued)) !== nothing
        push!(received, frame)
    end
    usage = store_usage(store)
    # Every committed frame arrived in order, including thinned previews,
    # whose values stay readable.
    @test length(received) == usage.frames + usage.thinned_previews
    @test issorted([frame.id for frame in received]) && queued.dropped == 0
    thinned = filter(frame -> !haskey(store, frame.id), received)
    @test length(thinned) == usage.thinned_previews > 0
    @test all(frame -> frame[:state] isa AbstractVector, thinned)
    # A full queue drops its oldest frame.
    @test take!(latest).id == last(received).id
    @test latest.dropped == length(received) - 1
    @test_throws ArgumentError subscribe(store; capacity = 0)
end

@testset "Pause, resume, stop, and keyframe requests" begin
    flow = _decaying_flow()
    fields = VisualizationFieldRegistry()
    register_field!(fields, :pressure, flow.pressure)
    recorder = Recorder(fields; diagnostics = flow, keyframes = nothing, preview = nothing)
    @test run_state(recorder) === :created
    pause!(recorder)
    @test run_state(recorder) === :pause_requested
    resume!(recorder)
    @test run_state(recorder) === :created

    # A pause requested before the run holds the initial state.
    pause!(recorder)
    task = Threads.@spawn simulate!(flow, 0.05; observer = recorder)
    _wait_paused_with_timeout(recorder)
    @test run_state(recorder) === :paused
    @test [info.kind for info in frames(snapshot_store(recorder))] == [:initial]
    resume!(recorder)
    result = _fetch_with_timeout(task)
    @test result.steps == 5
    @test run_state(recorder) === :completed
    @test_throws ArgumentError pause!(recorder)
    @test_throws ArgumentError wait_paused(recorder)
    @test_throws ArgumentError request_stop!(recorder)

    # Pause at a step boundary, request a keyframe, then stop while paused.
    entered = Channel{Nothing}(1)
    proceed = Channel{Nothing}(1)
    last_midpoint = Ref(-Inf)
    force_steps = Ref(0)
    force = function (x, y, time, axis)
        if time > last_midpoint[]
            last_midpoint[] = time
            force_steps[] += 1
            if force_steps[] in (1, 3)
                put!(entered, nothing)
                _take_with_timeout(proceed)
            end
        end
        return 0.0
    end
    task = Threads.@spawn simulate!(flow, 1.0; observer = recorder, force)
    _take_with_timeout(entered)
    pause!(recorder)
    put!(proceed, nothing)
    _wait_paused_with_timeout(recorder)
    store = snapshot_store(recorder)
    paused = last(frames(store))
    @test paused.kind === :pause
    @test paused.time ≈ 0.06
    request_keyframe!(recorder)
    resume!(recorder)
    # Hold the next physical step so the run cannot finish between the
    # manual-keyframe observation and the second pause request.
    _take_with_timeout(entered)
    @test any(info -> info.kind === :manual && info.time ≈ 0.07, frames(store))
    pause!(recorder)
    put!(proceed, nothing)
    _wait_paused_with_timeout(recorder)
    stop_time = last(frames(store)).time
    request_stop!(recorder)
    result = _fetch_with_timeout(task)
    @test result.stopped
    @test result.time ≈ stop_time
    @test flow.time ≈ stop_time
    @test last(frames(store)).kind in (:pause, :final)
    @test run_state(recorder) === :completed
    @test [event.kind for event in run_events(store)][(end - 1):end] == [:resume, :stop]

    # A stop requested before the run ends it at the initial state.
    early = Recorder(fields; keyframes = nothing, preview = nothing)
    request_stop!(early)
    @test stop_requested(early)
    start = flow.time
    result = simulate!(flow, 1.0; observer = early)
    @test result.stopped
    @test result.steps == 0
    @test flow.time == start
    @test [info.kind for info in frames(snapshot_store(early))] == [:initial]
    @test !stop_requested(early)

    # A stop supersedes a pause requested before the run starts.
    early_paused = Recorder(fields; keyframes = nothing, preview = nothing)
    pause!(early_paused)
    request_stop!(early_paused)
    @test stop_requested(early_paused)
    start = flow.time
    task = @async simulate!(flow, 1.0; observer = early_paused)
    result = try
        _fetch_with_timeout(task)
    finally
        if !istaskdone(task)
            state = run_state(early_paused)
            state in (:completed, :failed) || request_stop!(early_paused)
            _fetch_with_timeout(task)
        end
    end
    @test result.stopped && result.steps == 0 && result.time == start
    @test flow.time == start
    @test run_state(early_paused) === :completed
    @test !stop_requested(early_paused)
    @test [info.kind for info in frames(snapshot_store(early_paused))] == [:initial]
    @test !any(event -> event.kind === :pause, run_events(snapshot_store(early_paused)))
end

@testset "Breakpoints pause where a diagnostic condition starts to hold" begin
    flow = _decaying_flow()
    fields = VisualizationFieldRegistry()
    register_field!(fields, :pressure, flow.pressure)
    recorder = Recorder(fields; diagnostics = flow, keyframes = nothing, preview = nothing)
    initial_energy = diagnostic_history(flow, :kinetic_energy).values[1]
    threshold = 0.9initial_energy
    pause_when!(recorder, :kinetic_energy, energy -> energy < threshold)
    task = Threads.@spawn simulate!(flow, 1.0; observer = recorder)
    _wait_paused_with_timeout(recorder)
    history = diagnostic_history(flow, :kinetic_energy)
    store = snapshot_store(recorder)
    paused = last(frames(store))
    @test paused.kind === :pause
    @test last(history.values) < threshold
    @test history.values[end - 1] >= threshold
    @test paused.time == last(history.times)
    breakpoint = only(filter(event -> event.kind === :breakpoint, run_events(store)))
    @test breakpoint.time == paused.time
    @test breakpoint.step == paused.step
    resume!(recorder)
    # The condition keeps holding, so the run is not paused again.
    result = _fetch_with_timeout(task)
    @test result.time == 1.0
    @test count(event -> event.kind === :pause, run_events(store)) == 1
end

@testset "Recorder failures never abort the simulation" begin
    flow = _decaying_flow()
    fields = VisualizationFieldRegistry()
    register_field!(fields, :pressure, flow.pressure)
    recorder = Recorder(fields; diagnostics = flow, keyframes = EveryNSteps(5), preview = nothing)
    calls = Ref(0)
    pause_when!(recorder, :kinetic_energy, energy -> (calls[] += 1) > 3 ? error("broken breakpoint") : false)
    result = @test_logs (:error,) match_mode = :any simulate!(
        flow, 0.1; observer = recorder,
    )
    @test result.steps == 10
    @test !result.stopped
    store = snapshot_store(recorder)
    failures = filter(event -> event.kind === :failure, run_events(store))
    @test length(failures) == calls[] - 3 >= 7
    @test issorted([event.step for event in failures])
    # Event times are accumulated from at most ten Float64 steps.
    @test all(event -> isapprox(event.time, 0.01 * event.step; atol = 16eps(Float64), rtol = 0), failures)
    @test [info.step for info in keyframes(store)] == [0, 5, 10]
    @test run_state(recorder) === :completed

    # A failing simulation is reported, and the error reaches the caller.
    failing = Recorder(fields; diagnostics = flow, keyframes = nothing, preview = nothing)
    add_diagnostic!(flow, :fragile, state -> state.time > 0.12 ? error("diagnostic failure") : 0.0)
    @test_throws ErrorException simulate!(flow, 0.2; observer = failing)
    @test run_state(failing) === :failed
    @test last(run_events(snapshot_store(failing))).kind === :failure
end

@testset "Recorder interrupts propagate without leaving a pause blocked" begin
    # Breakpoint hooks run both at completed steps and at the final event.
    for schedule in (EveryStep(), EveryNSteps(10))
        flow = _decaying_flow(; diagnostics = (KineticEnergy() => schedule,))
        recorder = Recorder(; diagnostics = flow, keyframes = nothing, preview = nothing)
        calls = Ref(0)
        pause_when!(
            recorder, :kinetic_energy, value -> begin
                calls[] += 1
                calls[] == 2 && throw(InterruptException())
                false
            end
        )
        @test_throws InterruptException simulate!(flow, 0.02; observer = recorder)
        @test calls[] == 2
        @test run_state(recorder) === :failed
    end

    # Interrupt each capture handler without depending on solver timing.
    for stage in (:step, :completed, :pause)
        registry, field = _state_registry()
        armed = Ref(false)
        clock = function ()
            if armed[]
                armed[] = false
                throw(InterruptException())
            end
            return 0.0
        end
        recorder = Recorder(
            registry; keyframes = nothing, preview = stage === :step ? EveryStep() : nothing,
            clock,
        )
        run = _start!(recorder, field)
        stage === :completed && _step!(run, 0.1, 1)
        stage === :pause && pause!(recorder)
        armed[] = true
        event = stage === :completed ? :completed : :step
        @test_throws InterruptException _step!(run, 0.1, 1; event)
    end

    # Wake the actual condition wait with an interrupt, at the initial and a
    # completed-step boundary. Cleanup also runs if an assertion fails.
    for initial_pause in (true, false)
        flow = _decaying_flow()
        fields = VisualizationFieldRegistry()
        register_field!(fields, :pressure, flow.pressure)
        recorder = Recorder(
            fields; diagnostics = flow, keyframes = nothing, preview = nothing,
            clock = () -> 0.0,
        )
        if initial_pause
            pause!(recorder)
        else
            calls = Ref(0)
            pause_when!(
                recorder, :kinetic_energy, value -> begin
                    calls[] += 1
                    calls[] == 2
                end
            )
        end
        task = @async try
            simulate!(flow, 0.02; observer = recorder)
        catch exception
            exception
        end
        try
            _wait_paused_with_timeout(recorder)
            @test run_state(recorder) === :paused
            @test newest_frame(snapshot_store(recorder)).step == (initial_pause ? 0 : 1)
            waiting = timedwait(_RECORDING_TIMEOUT; pollint = 0.001) do
                Base.@lock recorder.lock !isempty(recorder.condition)
            end
            @test waiting === :ok
            if waiting === :ok
                Base.@lock recorder.lock notify(
                    recorder.condition, InterruptException(); all = true, error = true,
                )
                @test _fetch_with_timeout(task) isa InterruptException
                @test istaskdone(task)
                @test run_state(recorder) === :failed
                @test isnothing(recorder.run)
                @test !stop_requested(recorder)
            end
        finally
            if !istaskdone(task)
                state = run_state(recorder)
                state in (:completed, :failed) || request_stop!(recorder)
                _fetch_with_timeout(task)
            end
        end
    end
end
