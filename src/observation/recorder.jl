struct Breakpoint
    name::Symbol
    predicate::Any
end

# State of one `simulate!` call, owned by the solver task.
mutable struct RecorderRun
    plan::Union{Nothing, CapturePlan}
    topology::Union{Nothing, CaptureTopology}
    keyframe_times::Vector{Float64}
    next_keyframe::Int
    tolerance::Float64
    initial_time::Float64
    final_time::Float64
    completed_time::Float64
    completed_step::Int
    # Nearest-state keyframes: a captured state that a later step may select.
    candidate::Union{Nothing, Frame}
    candidate_committed::Bool
    candidate_native::Bool
    # Exact keyframes: copied state at the previous step boundary.
    step_procedure::Any
    exact_velocity::Any
    exact_pressure::Any
    exact_time::Float64
    exact_step::Int
    # Identifier of the stored frame holding the current state, if any.
    current_frame::Union{Nothing, Frame}
    current_committed::Bool
    current_native::Bool
    previous_breakpoint::Dict{Symbol, Bool}
    pause_now::Bool
    pause_message::String
    preview_wall::Float64
    next_preview_time::Float64
    # Smoothed cost of a preview capture and the number of measured previews.
    preview_cost::Float64
    previews_measured::Int
end

"""
    Recorder(fields; diagnostics=nothing, keyframes=EquispacedTimes(21),
             exact_keyframes=false, preview=EveryWallInterval(0.2),
             max_capture_overhead=0.05, retention=Retention(),
             preview_precision=Float32, clock=time)

Observer for [`simulate!`](@ref) that copies owned frames of the fields
registered in `fields` (a [`VisualizationFieldRegistry`](@ref)) into a bounded
[`SnapshotStore`](@ref) (see [`snapshot_store`](@ref)). The solver never waits
for a consumer of the store.

- `keyframes` are scheduled, lossless frames in native precision: an
  [`EquispacedTimes`](@ref) count or [`EveryTimeInterval`](@ref) (both include
  the initial and final time), explicit [`AtTimes`](@ref), a step schedule
  ([`EveryNSteps`](@ref), [`EveryStep`](@ref)), or `nothing`. By default the
  completed state nearest to each requested time is kept. With
  `exact_keyframes=true`, an independent copy of the state is advanced to each
  requested time without changing the simulation path (fixed mesh only).
- `preview` frames show recent progress in `preview_precision`; they are
  throttled with [`EveryWallInterval`](@ref), [`EveryNSteps`](@ref),
  [`EveryStep`](@ref), or [`EveryTimeInterval`](@ref), and further so that
  capture time stays below `max_capture_overhead` of the wall time. Use
  `nothing` for headless scripts. Older previews are thinned by `retention`.
- The initial and final states are always stored. Initial, final, and pause
  frames use native precision, including when a preview is promoted.
- `diagnostics` (a `FlowSimulation` or `DiagnosticCollection`) are mirrored
  at every step boundary into the store with their [`DiagnosticInfo`](@ref).
- `clock()` returns wall-clock seconds; it stamps frames and drives the
  preview throttle.

Controls are thread-safe: [`run_state`](@ref), [`pause!`](@ref),
[`wait_paused`](@ref), [`resume!`](@ref), [`request_stop!`](@ref),
[`request_keyframe!`](@ref), and [`pause_when!`](@ref). Errors after the
initial state are isolated: they are logged and stored as `:failure` events,
and the simulation continues. `InterruptException` propagates to the caller.
Configuration errors (for example a registry on another mesh) throw at the
initial event, before the first step.
"""
mutable struct Recorder <: AbstractObserver
    registry::VisualizationFieldRegistry
    diagnostics::Union{Nothing, DiagnosticCollection}
    keyframes::_KeyframeSchedule
    exact_keyframes::Bool
    preview::_PreviewSchedule
    max_capture_overhead::Float64
    preview_precision::Union{Nothing, DataType}
    clock::Any
    store::SnapshotStore
    lock::ReentrantLock
    condition::Threads.Condition
    state::Symbol
    stop::Bool
    keyframe_requested::Bool
    breakpoints::Vector{Breakpoint}
    run::Union{Nothing, RecorderRun}
    previous_run::Union{Nothing, RecorderRun}
    driver::Any
    estimate_logged::Bool
    failures_logged::Set{Symbol}
end

_diagnostic_collection(::Nothing) = nothing
_diagnostic_collection(collection::DiagnosticCollection) = collection
_diagnostic_collection(source) = hasproperty(source, :diagnostics) &&
    getproperty(source, :diagnostics) isa DiagnosticCollection ?
    getproperty(source, :diagnostics) : throw(
        ArgumentError(
            "diagnostics must be nothing, a DiagnosticCollection, or a " *
            "FlowSimulation; got $(typeof(source))",
        ),
    )

function Recorder(
        fields::VisualizationFieldRegistry = VisualizationFieldRegistry();
        diagnostics = nothing, keyframes = EquispacedTimes(21),
        exact_keyframes::Bool = false, preview = EveryWallInterval(0.2),
        max_capture_overhead::Real = 0.05, retention::Retention = Retention(),
        preview_precision = Float32, clock = time,
    )
    0 < max_capture_overhead <= 1 || throw(
        ArgumentError("max_capture_overhead must lie in (0, 1]"),
    )
    isnothing(preview_precision) || preview_precision in (Float32, Float64) ||
        throw(ArgumentError("preview_precision must be Float32, Float64, or nothing"))
    lock = ReentrantLock()
    return Recorder(
        fields, _diagnostic_collection(diagnostics),
        _check_keyframe_schedule(keyframes), exact_keyframes,
        _check_preview_schedule(preview), Float64(max_capture_overhead),
        preview_precision, clock, SnapshotStore(retention), lock,
        Threads.Condition(lock), :created, false, false, Breakpoint[], nothing,
        nothing, nothing, false, Set{Symbol}(),
    )
end

_supports_dynamic_topology(recorder::Recorder) = !recorder.exact_keyframes

"""Return the [`SnapshotStore`](@ref) written by a [`Recorder`](@ref)."""
snapshot_store(recorder::Recorder) = recorder.store

function Base.show(io::IO, recorder::Recorder)
    return print(
        io, "Recorder(state=:", run_state(recorder), ", ", recorder.store, ")",
    )
end

"""
    run_state(recorder)

Lifecycle state: `:created`, `:running`, `:pause_requested`, `:paused`,
`:stopping`, `:completed`, or `:failed`.
"""
run_state(recorder::Recorder) = Base.@lock recorder.lock recorder.state

"""
    pause!(recorder)

Request a pause. The solver blocks at its next completed-step boundary (at the
initial state if the run has not started), stores the paused state as a
`:pause` frame, and waits for [`resume!`](@ref) or [`request_stop!`](@ref).
"""
function pause!(recorder::Recorder)
    Base.@lock recorder.lock begin
        state = recorder.state
        if state in (:created, :running)
            recorder.state = :pause_requested
        elseif !(state in (:pause_requested, :paused))
            throw(ArgumentError("cannot pause a recorder in state :$state"))
        end
    end
    return recorder
end

"""
    resume!(recorder)

Resume a paused run, or cancel a pause that has not been reached yet.
"""
function resume!(recorder::Recorder)
    Base.@lock recorder.lock begin
        state = recorder.state
        if state === :pause_requested
            recorder.state = isnothing(recorder.run) ? :created : :running
        elseif state === :paused
            recorder.state = :running
            notify(recorder.condition; all = true)
        elseif state !== :running
            throw(ArgumentError("cannot resume a recorder in state :$state"))
        end
    end
    return recorder
end

"""
    wait_paused(recorder)

Wait until the run is paused, by [`pause!`](@ref) or a breakpoint (see
[`pause_when!`](@ref)). Throws if the run completes, fails, or stops first.
"""
function wait_paused(recorder::Recorder)
    Base.@lock recorder.lock begin
        recorder.state in (:completed, :failed, :stopping) && throw(
            ArgumentError("cannot wait for a pause in state :$(recorder.state)"),
        )
        while recorder.state in (:created, :running, :pause_requested)
            wait(recorder.condition)
        end
        recorder.state === :paused || throw(
            ErrorException("the run entered state :$(recorder.state) before pausing"),
        )
    end
    return recorder
end

"""
    request_stop!(recorder)

Make `simulate!` return at the next completed-step boundary with
`stopped = true`; a paused run resumes and stops immediately. A stop also
supersedes a pause requested before the run starts.
"""
function request_stop!(recorder::Recorder)
    Base.@lock recorder.lock begin
        recorder.state in (:completed, :failed) && throw(
            ArgumentError("cannot stop a recorder in state :$(recorder.state)"),
        )
        recorder.stop = true
        if recorder.state in (:running, :paused) ||
                (recorder.state === :pause_requested && !isnothing(recorder.run))
            recorder.state = :stopping
        end
        notify(recorder.condition; all = true)
    end
    return recorder
end

stop_requested(recorder::Recorder) = Base.@lock recorder.lock recorder.stop

"""
    request_keyframe!(recorder)

Store the state at the next step boundary as a `:manual` keyframe.
"""
function request_keyframe!(recorder::Recorder)
    Base.@lock recorder.lock recorder.keyframe_requested = true
    return recorder
end

"""
    pause_when!(recorder, name, predicate)

Breakpoint: pause the run at the step boundary where `predicate(value)` first
holds for a new sample of the simulation diagnostic `name` (for example
`pause_when!(recorder, :divergence_norm, >(1e-8))`). The breakpoint fires
again only after the predicate has been false.
"""
function pause_when!(recorder::Recorder, name::Symbol, predicate)
    Base.@lock recorder.lock push!(recorder.breakpoints, Breakpoint(name, predicate))
    return recorder
end

function _record_failure!(recorder::Recorder, stage::Symbol, exception, backtrace, time, step)
    message = sprint(showerror, exception)
    _record_event!(
        recorder.store,
        RunEvent(:failure, Float64(time), step, recorder.clock(), "$stage: $message"),
    )
    if !(stage in recorder.failures_logged)
        push!(recorder.failures_logged, stage)
        @error "Recorder $stage failed at t = $time; the simulation continues" exception = (
            exception, backtrace,
        )
    end
    return recorder
end

_log_iterations(stats) = hasproperty(stats, :iterations) ? Int(stats.iterations) :
    hasproperty(stats, :cycles) ? Int(stats.cycles) : -1
_log_initial_residual(stats) = hasproperty(stats, :initial_residual) ?
    Float64(stats.initial_residual) : hasproperty(stats, :initial_linf) ?
    Float64(stats.initial_linf) : NaN
_log_final_residual(stats) = hasproperty(stats, :final_residual) ?
    Float64(stats.final_residual) : hasproperty(stats, :final_linf) ?
    Float64(stats.final_linf) : NaN
_log_converged(stats) = hasproperty(stats, :converged) ? Bool(stats.converged) : true

function _current_topology!(recorder::Recorder, run::RecorderRun)
    mesh = recorder.registry.mesh::BlockTreeMesh
    geometries = run.plan.geometries
    topology = run.topology
    if isnothing(topology) || !_topology_current(topology.snapshot, mesh, geometries)
        id = _next_topology_id!(recorder.store)
        topology = _capture_topology(id, mesh, geometries)
        _add_topology!(recorder.store, topology.snapshot)
        run.topology = topology
    end
    return topology
end

function _new_frame(
        recorder::Recorder, run::RecorderRun, time, step, kind;
        native::Bool, requested_times = Float64[], replacements = IdDict{Any, Any}(),
    )
    topology = _current_topology!(recorder, run)
    info = FrameInfo(
        _next_frame_id!(recorder.store), Float64(time), Int(step), kind,
        recorder.clock(), topology.snapshot.id, Float64.(requested_times),
    )
    precision = native ? nothing : recorder.preview_precision
    return _capture_frame(
        run.plan, recorder.registry, topology, info; precision, replacements,
    )
end

_has_fields(run::RecorderRun) = !isnothing(run.plan) && !isempty(run.plan.sources)

function _native_frame(recorder::Recorder, run::RecorderRun, frame::Frame)
    topology = _current_topology!(recorder, run)
    return _capture_frame(run.plan, recorder.registry, topology, frame.info)
end

# Reuse a native capture of this state, or recapture a lossy preview before it
# becomes a keyframe. A committed preview keeps its id and metadata until stored.
function _current_frame!(recorder::Recorder, run::RecorderRun, state, kind; native = true)
    frame = run.current_frame
    if isnothing(frame)
        frame = _new_frame(recorder, run, state.time, state.step, kind; native)
        run.current_frame = frame
        run.current_committed = false
        run.current_native = native || isnothing(recorder.preview_precision)
    elseif native && !run.current_native
        candidate = run.candidate
        frame = !isnothing(candidate) && candidate.id == frame.id && run.candidate_native ?
            candidate : _native_frame(recorder, run, frame)
        run.current_frame = frame
        run.current_native = true
    end
    return frame
end

function _store_frame!(store, frame, committed, kind, requested_times; time = nothing)
    if committed
        frame = Base.@lock store.lock begin
            stored = store.frames[frame.id]
            replaced = frame.values !== stored.values || frame.phases !== stored.phases
            if replaced
                store.frames[frame.id] = _with_info(frame, stored.info)
                store.frame_bytes += frame.bytes - stored.bytes
            end
            promoted = _promote_frame!(store, frame.id, kind, requested_times; time)
            replaced && _apply_retention!(store)
            promoted
        end
    else
        rank_kind = _kind_rank(kind) >= _kind_rank(frame.kind) ? kind : frame.kind
        info = FrameInfo(
            frame.id, isnothing(time) ? frame.info.time : Float64(time),
            frame.info.step, rank_kind, frame.info.wall_time, frame.info.topology,
            Float64.(requested_times),
        )
        frame = _commit_frame!(store, _with_info(frame, info))
    end
    return frame
end

function _store_current!(
        recorder::Recorder, run::RecorderRun, kind, requested_times = Float64[];
        time = nothing,
    )
    frame = _store_frame!(
        recorder.store, run.current_frame, run.current_committed, kind, requested_times; time,
    )
    run.current_committed = true
    run.current_frame = frame
    if !isnothing(run.candidate) && run.candidate.id == frame.id
        run.candidate = frame
        run.candidate_committed = true
        run.candidate_native = run.current_native
    end
    return recorder
end

function _evaluate_breakpoints!(recorder::Recorder, run::RecorderRun, samples)
    breakpoints = Base.@lock recorder.lock copy(recorder.breakpoints)
    isempty(breakpoints) && return nothing
    for (name, value, time, step) in samples
        for breakpoint in breakpoints
            breakpoint.name === name || continue
            holds = breakpoint.predicate(value) === true
            key = Symbol(name, :_, objectid(breakpoint))
            previous = get(run.previous_breakpoint, key, false)
            run.previous_breakpoint[key] = holds
            if holds && !previous
                run.pause_now = true
                run.pause_message = "breakpoint :$name at value $value"
                _record_event!(
                    recorder.store,
                    RunEvent(:breakpoint, time, step, recorder.clock(), run.pause_message),
                )
            end
        end
    end
    return nothing
end

function _cache_exact_state!(run::RecorderRun, state)
    run.exact_velocity = _copy_observed_velocity(state.velocity)
    run.exact_pressure = copy(state.pressure)
    run.exact_time = Float64(state.time)
    run.exact_step = state.step
    return run
end

_copy_observed_velocity(velocity) = copy(velocity)
_copy_observed_velocity(velocity::TwoPhaseVelocity) = map(copy, velocity)

# Interface and phase-scalar copies advanced by an independent step procedure
# (specialized for the Navier--Stokes procedure).
_independent_field_replacements(procedure) = IdDict{Any, Any}()

function _exact_replacements(
        procedure, live_velocity, velocity, live_pressure, pressure,
    )
    replacements = IdDict{Any, Any}(live_velocity => velocity, live_pressure => pressure)
    if live_velocity isa TwoPhaseVelocity
        for phase in 1:2
            replacements[live_velocity[phase]] = velocity[phase]
        end
    end
    for (field, replacement) in _independent_field_replacements(procedure)
        replacements[field] = replacement
    end
    return replacements
end

# Advance copies of the previous step-boundary state to every requested time
# strictly inside the completed step, and store each as a keyframe.
function _exact_keyframes!(recorder::Recorder, run::RecorderRun, state)
    current = Float64(state.time)
    while run.next_keyframe <= length(run.keyframe_times)
        target = run.keyframe_times[run.next_keyframe]
        if target <= run.exact_time + run.tolerance
            run.next_keyframe += 1
            continue
        end
        target < current - run.tolerance || break
        velocity = _copy_observed_velocity(run.exact_velocity)
        pressure = copy(run.exact_pressure)
        dt = typeof(state.time)(target - run.exact_time)
        run.step_procedure(
            velocity, pressure, dt; time = typeof(state.time)(run.exact_time),
        )
        if _has_fields(run)
            replacements = _exact_replacements(
                run.step_procedure, state.velocity, velocity, state.pressure, pressure,
            )
            frame = _new_frame(
                recorder, run, target, run.exact_step, :keyframe;
                native = true, requested_times = [target], replacements,
            )
            _commit_frame!(recorder.store, frame)
        end
        run.next_keyframe += 1
    end
    return nothing
end

# Select the nearer of the candidate (previous captured state) and the
# current state for every requested time reached by this step.
function _nearest_keyframes!(recorder::Recorder, run::RecorderRun, state)
    current = Float64(state.time)
    candidate_times = Float64[]
    current_times = Float64[]
    while run.next_keyframe <= length(run.keyframe_times)
        target = run.keyframe_times[run.next_keyframe]
        target <= current + run.tolerance || break
        candidate = run.candidate
        if !isnothing(candidate) &&
                abs(candidate.time - target) <= abs(current - target)
            push!(candidate_times, target)
        else
            push!(current_times, target)
        end
        run.next_keyframe += 1
    end
    if !isempty(candidate_times) && _has_fields(run)
        candidate = _store_frame!(
            recorder.store, run.candidate::Frame, run.candidate_committed,
            :keyframe, candidate_times,
        )
        run.candidate = candidate
        run.candidate_committed = true
    end
    return current_times
end

function _preview_due!(recorder::Recorder, run::RecorderRun, state)
    schedule = recorder.preview
    isnothing(schedule) && return false
    due = if schedule isa EveryWallInterval
        recorder.clock() - run.preview_wall >= schedule.seconds
    elseif schedule isa EveryStep
        true
    elseif schedule isa EveryNSteps
        mod(state.step, schedule.n) == 0
    else
        time = Float64(state.time)
        reached = time + run.tolerance >= run.next_preview_time
        while run.next_preview_time <= time + run.tolerance
            run.next_preview_time += Float64(schedule.interval)
        end
        reached
    end
    due || return false
    # Overhead budget: a preview may cost at most `max_capture_overhead` of
    # the wall time since the previous one. The first preview's cost includes
    # compilation and is not used.
    if run.previews_measured >= 2 && recorder.clock() - run.preview_wall <
            run.preview_cost / recorder.max_capture_overhead
        Base.@lock recorder.store.lock recorder.store.skipped_previews += 1
        return false
    end
    return true
end

function _log_memory_estimate(recorder::Recorder, frame::Frame)
    recorder.estimate_logged && return nothing
    recorder.estimate_logged = true
    budget = recorder.store.retention.memory
    bytes = max(frame.bytes + _topology_bytes(frame.topology), 1)
    @info "Recorder: a native-precision frame needs $(Base.format_bytes(bytes)); " *
        "the memory budget of $(Base.format_bytes(budget)) holds about " *
        "$(budget ÷ bytes) frames."
    return nothing
end

# Mesh of an observed state; custom time loops may omit the solver.
function _state_mesh(state)
    hasproperty(state.solver, :mesh) && return state.solver.mesh
    for field in (state.velocity, state.pressure)
        field isa TwoPhaseVelocity && return field[1].mesh
        hasproperty(field, :mesh) && return field.mesh
    end
    return nothing
end

function _validate_recording_state(recorder::Recorder, state, info::StepInfo)
    registry = recorder.registry
    mesh = _state_mesh(state)
    if !isnothing(registry.mesh) && !isnothing(mesh)
        registry.mesh === mesh || throw(
            ArgumentError(
                "the recorder's field registry belongs to a different mesh than the simulation",
            ),
        )
    end
    recorder.exact_keyframes && isnothing(info.step_procedure) && throw(
        ArgumentError("exact keyframes require the independent step procedure of simulate!"),
    )
    return nothing
end

function _initialize_run!(recorder::Recorder, state, info::StepInfo)
    _validate_recording_state(recorder, state, info)
    registry = recorder.registry
    t0 = Float64(state.time)
    tf = Float64(info.final_time)
    tolerance = _time_tolerance(Float64(eps(typeof(state.time))), t0, tf)
    plan = isnothing(registry.mesh) ? nothing : _capture_plan(registry)
    return RecorderRun(
        plan, nothing, _keyframe_times(recorder.keyframes, t0, tf, tolerance), 1,
        tolerance, t0, tf, t0, state.step, nothing, false, false, info.step_procedure, nothing, nothing,
        t0, state.step, nothing, false, false, Dict{Symbol, Bool}(), false, "",
        recorder.clock(), recorder.preview isa EveryTimeInterval ?
            t0 + Float64(recorder.preview.interval) : t0,
        0.0, 0,
    )
end

# Preserve interval phase and unserved absolute targets across a stop.
# EquispacedTimes is instead defined over each individual simulate! call.
function _resume_run!(recorder::Recorder, run::RecorderRun, state, info::StepInfo)
    _validate_recording_state(recorder, state, info)
    t0 = Float64(state.time)
    tf = Float64(info.final_time)
    tolerance = _time_tolerance(Float64(eps(typeof(state.time))), run.initial_time, tf)
    t0 == run.completed_time && state.step == run.completed_step ||
        throw(ArgumentError("a recorder can only continue from its last completed state"))
    tf >= t0 || throw(ArgumentError("final_time must not precede the completed state"))

    schedule = recorder.keyframes
    pending = schedule isa EquispacedTimes ? Float64[] :
        run.keyframe_times[run.next_keyframe:end]
    additional = if schedule isa AtTimes
        schedule.times
    elseif schedule isa EveryTimeInterval
        _keyframe_times(schedule, run.initial_time, tf, tolerance)
    elseif schedule isa EquispacedTimes
        _keyframe_times(schedule, t0, tf, tolerance)
    else
        Float64[]
    end
    # The start of an equispaced segment is itself a requested keyframe.
    # Attach it to the existing final frame rather than capturing it again.
    boundary = run.current_frame
    if schedule isa EquispacedTimes && schedule.count > 1 &&
            !isnothing(boundary) && run.current_committed &&
            !any(time -> abs(time - t0) <= tolerance, boundary.info.requested_times)
        _store_current!(recorder, run, :keyframe, [t0])
    end
    times = sort!(
        [
            time for time in Iterators.flatten((pending, additional))
                if t0 + tolerance < time
        ]
    )
    unique_times = Float64[]
    for time in times
        value = time
        if isempty(unique_times) || value - last(unique_times) > tolerance
            push!(unique_times, value)
        end
    end
    run.keyframe_times = unique_times
    run.next_keyframe = 1
    run.tolerance = tolerance
    run.final_time = tf
    run.step_procedure = info.step_procedure
    run.pause_now = false
    run.pause_message = ""
    return run
end

function _observe_initial!(recorder::Recorder, state, info::StepInfo)
    pause_requested, previous = Base.@lock recorder.lock begin
        recorder.state in (:running, :paused, :stopping) && throw(
            ArgumentError("the recorder is already observing a run"),
        )
        requested = recorder.state === :pause_requested && !recorder.stop
        previous = recorder.previous_run
        recorder.state = requested ? :pause_requested : :running
        (requested, previous)
    end
    run = try
        isnothing(previous) ? _initialize_run!(recorder, state, info) :
            _resume_run!(recorder, previous, state, info)
    catch
        Base.@lock recorder.lock begin
            recorder.state = :failed
            recorder.stop = false
            recorder.run = nothing
            notify(recorder.condition; all = true)
        end
        rethrow()
    end
    Base.@lock recorder.lock recorder.run = run
    try
        samples = _mirror_diagnostics!(recorder.store, recorder.diagnostics)
        _evaluate_breakpoints!(recorder, run, samples)
        if _has_fields(run) && isnothing(previous)
            requested_times = Float64[]
            while run.next_keyframe <= length(run.keyframe_times) &&
                    run.keyframe_times[run.next_keyframe] <= run.initial_time + run.tolerance
                push!(requested_times, run.keyframe_times[run.next_keyframe])
                run.next_keyframe += 1
            end
            _current_frame!(recorder, run, state, :initial)
            _store_current!(recorder, run, :initial, requested_times)
            run.candidate = run.current_frame
            run.candidate_committed = true
            run.candidate_native = run.current_native
            _log_memory_estimate(recorder, run.current_frame)
        end
        if !isnothing(previous) && _has_fields(run)
            # The preceding final frame is also this segment's initial state.
            # Reuse it for a pause or nearest-state capture, never publish a
            # second frame or initial event at this boundary.
            run.candidate = run.current_frame
            run.candidate_committed = run.current_committed
            run.candidate_native = run.current_native
        end
        recorder.exact_keyframes && _cache_exact_state!(run, state)
        run.pause_now |= pause_requested
        _pause_gate!(recorder, run, state)
        run.current_frame = nothing
    catch
        Base.@lock recorder.lock begin
            recorder.state = :failed
            recorder.stop = false
            recorder.run = nothing
            notify(recorder.condition; all = true)
        end
        rethrow()
    end
    return recorder
end

function _observe_step!(recorder::Recorder, state, info::StepInfo)
    run = recorder.run::RecorderRun
    started = time_ns()
    try
        samples = _mirror_diagnostics!(recorder.store, recorder.diagnostics)
        _evaluate_breakpoints!(recorder, run, samples)
    catch exception
        exception isa InterruptException && rethrow()
        _record_failure!(recorder, :diagnostics, exception, catch_backtrace(), state.time, state.step)
    end
    if _has_fields(run)
        try
            _capture_step!(recorder, run, state, info)
        catch exception
            exception isa InterruptException && rethrow()
            _record_failure!(recorder, :capture, exception, catch_backtrace(), state.time, state.step)
            run.current_frame = nothing
            run.candidate = nothing
        end
    end
    capture_time = (time_ns() - started) / 1.0e9
    stats = info.projection
    _append_step_log!(
        recorder.store, (;
            time = Float64(state.time), step = state.step, dt = Float64(info.dt),
            wall_time = info.wall_time, iterations = _log_iterations(stats),
            initial_residual = _log_initial_residual(stats),
            final_residual = _log_final_residual(stats),
            converged = _log_converged(stats),
            topology_changed = info.topology_changed, capture_time,
        )
    )
    Base.@lock recorder.lock begin
        recorder.state === :pause_requested && (run.pause_now = true)
    end
    _pause_gate!(recorder, run, state)
    run.current_frame = nothing
    yield()
    return recorder
end

function _capture_step!(recorder::Recorder, run::RecorderRun, state, info::StepInfo)
    current_times = if recorder.exact_keyframes
        _exact_keyframes!(recorder, run, state)
        times = Float64[]
        while run.next_keyframe <= length(run.keyframe_times) &&
                run.keyframe_times[run.next_keyframe] <= Float64(state.time) + run.tolerance
            push!(times, run.keyframe_times[run.next_keyframe])
            run.next_keyframe += 1
        end
        times
    else
        _nearest_keyframes!(recorder, run, state)
    end
    manual = Base.@lock recorder.lock begin
        requested = recorder.keyframe_requested
        recorder.keyframe_requested = false
        requested
    end
    kind = if manual
        :manual
    elseif !isempty(current_times) || _step_keyframe_due(recorder.keyframes, state.step)
        :keyframe
    else
        nothing
    end
    # A state the next step may select as the nearest keyframe.
    needs_candidate = !recorder.exact_keyframes &&
        run.next_keyframe <= length(run.keyframe_times) &&
        run.keyframe_times[run.next_keyframe] - Float64(state.time) <=
        Float64(info.dt) + run.tolerance
    preview = isnothing(kind) && _preview_due!(recorder, run, state)
    native_candidate = nothing
    if !isnothing(kind)
        _current_frame!(recorder, run, state, kind)
        _store_current!(recorder, run, kind, current_times)
    elseif preview
        started = recorder.clock()
        _current_frame!(recorder, run, state, :preview; native = needs_candidate)
        _store_current!(recorder, run, :preview)
        if !run.current_native && !recorder.exact_keyframes &&
                run.next_keyframe <= length(run.keyframe_times)
            # A growing step can make even a distant preview the nearest
            # keyframe. Retain its native state without changing the preview.
            native_candidate = _native_frame(recorder, run, run.current_frame)
        end
        finished = recorder.clock()
        run.previews_measured += 1
        cost = finished - started
        run.preview_cost = run.previews_measured <= 2 ? cost :
            0.8 * run.preview_cost + 0.2 * cost
        run.preview_wall = finished
    elseif needs_candidate
        _current_frame!(recorder, run, state, :keyframe)
    end
    if needs_candidate || !isnothing(kind) || preview
        run.candidate = isnothing(native_candidate) ? run.current_frame : native_candidate
        run.candidate_committed = run.current_committed
        run.candidate_native = run.current_native || !isnothing(native_candidate)
    else
        run.candidate = nothing
    end
    recorder.exact_keyframes && _cache_exact_state!(run, state)
    return nothing
end

function _observe_completed!(recorder::Recorder, state, info::StepInfo)
    run = recorder.run::RecorderRun
    try
        samples = _mirror_diagnostics!(recorder.store, recorder.diagnostics)
        _evaluate_breakpoints!(recorder, run, samples)
    catch exception
        exception isa InterruptException && rethrow()
        _record_failure!(recorder, :diagnostics, exception, catch_backtrace(), state.time, state.step)
    end
    if _has_fields(run)
        try
            current_times = if recorder.exact_keyframes
                _exact_keyframes!(recorder, run, state)
                times = Float64[]
                while run.next_keyframe <= length(run.keyframe_times) &&
                        run.keyframe_times[run.next_keyframe] <= Float64(state.time) + run.tolerance
                    push!(times, run.keyframe_times[run.next_keyframe])
                    run.next_keyframe += 1
                end
                times
            else
                _nearest_keyframes!(recorder, run, state)
            end
            # Reuse the last step at this boundary, including its rounded
            # final time. A later target reached without a physical step is
            # a new time boundary and must not relabel the previous frame.
            frame = run.candidate
            same_step_new_time = state.step == run.completed_step &&
                Float64(state.time) != run.completed_time
            if same_step_new_time
                run.current_frame = nothing
            elseif !isnothing(frame) && frame.step == state.step
                run.current_frame = frame
                run.current_committed = run.candidate_committed
                run.current_native = run.candidate_native
            end
            Base.@lock recorder.lock recorder.keyframe_requested = false
            _current_frame!(recorder, run, state, :final)
            _store_current!(recorder, run, :final, current_times; time = state.time)
        catch exception
            exception isa InterruptException && rethrow()
            _record_failure!(recorder, :capture, exception, catch_backtrace(), state.time, state.step)
        end
    end
    run.completed_time = Float64(state.time)
    run.completed_step = state.step
    if _has_fields(run) && !isnothing(run.current_frame)
        run.candidate = run.current_frame
        run.candidate_committed = run.current_committed
        run.candidate_native = run.current_native
    end
    # The next call supplies a fresh independent-step procedure and caches its
    # own starting state; do not retain these potentially large copies idle.
    run.exact_velocity = nothing
    run.exact_pressure = nothing
    run.step_procedure = nothing
    stopped = Base.@lock recorder.lock begin
        stopped = recorder.stop
        recorder.stop = false
        recorder.previous_run = run
        recorder.state = :completed
        notify(recorder.condition; all = true)
        stopped
    end
    stopped && _record_event!(
        recorder.store,
        RunEvent(:stop, Float64(state.time), state.step, recorder.clock(), "stop requested"),
    )
    recorder.run = nothing
    return recorder
end

function _observe_failed!(recorder::Recorder, state, info::StepInfo)
    _record_event!(
        recorder.store, RunEvent(
            :failure, Float64(state.time), state.step, recorder.clock(),
            "simulation: " * sprint(showerror, info.exception),
        ),
    )
    Base.@lock recorder.lock begin
        recorder.stop = false
        recorder.state = :failed
        notify(recorder.condition; all = true)
    end
    recorder.run = nothing
    return recorder
end

# Block at a step boundary while paused. The paused state is stored first.
function _pause_gate!(recorder::Recorder, run::RecorderRun, state)
    run.pause_now || return nothing
    run.pause_now = false
    if _has_fields(run)
        try
            _current_frame!(recorder, run, state, :pause)
            _store_current!(recorder, run, :pause)
        catch exception
            exception isa InterruptException && rethrow()
            _record_failure!(recorder, :capture, exception, catch_backtrace(), state.time, state.step)
        end
    end
    message = isempty(run.pause_message) ? "pause requested" : run.pause_message
    run.pause_message = ""
    paused = Base.@lock recorder.lock begin
        (recorder.stop || recorder.state in (:stopping, :completed, :failed)) && return nothing
        recorder.state = :paused
        notify(recorder.condition; all = true)
        true
    end
    _record_event!(
        recorder.store,
        RunEvent(:pause, Float64(state.time), state.step, recorder.clock(), message),
    )
    try
        Base.@lock recorder.lock begin
            while recorder.state === :paused
                wait(recorder.condition)
            end
        end
    catch
        Base.@lock recorder.lock begin
            recorder.state = :failed
            notify(recorder.condition; all = true)
        end
        rethrow()
    end
    _record_event!(
        recorder.store,
        RunEvent(:resume, Float64(state.time), state.step, recorder.clock(), ""),
    )
    return paused
end

function observe!(recorder::Recorder, event::Symbol, state, info::StepInfo)
    if event === :initial
        _observe_initial!(recorder, state, info)
    elseif isnothing(recorder.run)
        # A failure before the initial event leaves nothing to record.
        event === :failed && Base.@lock recorder.lock begin
            recorder.state = :failed
            notify(recorder.condition; all = true)
        end
    elseif event === :step
        _observe_step!(recorder, state, info)
    elseif event === :completed
        _observe_completed!(recorder, state, info)
    elseif event === :failed
        _observe_failed!(recorder, state, info)
    else
        throw(ArgumentError("unknown observer event :$event"))
    end
    return recorder
end

"""
    StudioDriver(flow, recorder; simulation_options...)
    StudioDriver(advance!, recorder; time=0)

Bind one recorder to a resumable simulation. For a `FlowSimulation`, the driver
owns the flow and the fixed keyword arguments supplied to `simulate!`; the
recorder is always its observer. Call `simulate!(driver, final_time)` for the
initial run and later absolute targets. A continuation target must be finite
and later than the actual time; after an ordinary completion it must also
be later than the previous requested target. Only one call can run at a time.

Custom stateful loops may instead provide `advance!(final_time, recorder)`,
which must advance their own persistent state, send the recorder's observer
events, and return a result with its actual `time` and boolean `stopped`.
This explicit driver is distinct from an unowned, non-resumable closure run.
"""
mutable struct StudioDriver
    simulation::Any
    advance::Any
    recorder::Recorder
    options::NamedTuple
    lock::ReentrantLock
    time::Float64
    final_time::Union{Nothing, Float64}
    initial_interval::Float64
    active::Bool
    stopped::Bool
    failed::Bool
end

function _bind_studio_driver!(driver::StudioDriver)
    recorder = driver.recorder
    Base.@lock recorder.lock begin
        isnothing(recorder.driver) ||
            throw(ArgumentError("the recorder already has a Studio driver"))
        recorder.state in (:created, :pause_requested) ||
            throw(ArgumentError("a Studio driver must be attached before the run starts"))
        recorder.driver = driver
    end
    return driver
end

function StudioDriver(flow, recorder::Recorder; options...)
    flow isa FlowSimulation || throw(ArgumentError("a flow driver requires a FlowSimulation"))
    for key in (:observer, :time, :diagnostics, :step_offset)
        haskey(options, key) && throw(ArgumentError("the driver owns $key"))
    end
    time = Float64(flow.time)
    return _bind_studio_driver!(
        StudioDriver(
            flow, nothing, recorder, (; options...), ReentrantLock(),
            time, nothing, 0.0, false, false, false
        ),
    )
end

function StudioDriver(advance!::Function, recorder::Recorder; time::Real = 0)
    isfinite(time) && isfinite(Float64(time)) ||
        throw(ArgumentError("the initial time must be finite"))
    initial_time = Float64(time)
    return _bind_studio_driver!(
        StudioDriver(
            nothing, advance!, recorder, (;), ReentrantLock(),
            initial_time, nothing, 0.0, false, false, false
        ),
    )
end

function _reserve_driver!(driver::StudioDriver, final_time; continuation::Bool = false)
    final_time isa Real && !(final_time isa Bool) && isfinite(final_time) ||
        throw(ArgumentError("final_time must be a finite number"))
    target = Float64(final_time)
    isfinite(target) || throw(ArgumentError("final_time must be a finite number"))
    if !isnothing(driver.simulation)
        target = Float64(typeof(driver.simulation.time)(target))
        isfinite(target) || throw(ArgumentError("final_time must be representable by the simulation"))
    end
    return Base.@lock driver.lock begin
        driver.active && throw(ArgumentError("the simulation is already running"))
        driver.failed && throw(ArgumentError("a failed simulation cannot be continued"))
        state = run_state(driver.recorder)
        if isnothing(driver.final_time)
            continuation && throw(ArgumentError("the simulation has not completed"))
            state in (:created, :pause_requested) ||
                throw(ArgumentError("the simulation has already started"))
            target >= driver.time || throw(ArgumentError("final_time must not precede the initial time"))
            driver.initial_interval = target - driver.time
        else
            state === :completed || throw(ArgumentError("the simulation has not completed"))
            target > driver.time && (driver.stopped || target > driver.final_time) ||
                throw(ArgumentError("final_time must be later than the completed time"))
        end
        driver.final_time = target
        driver.active = true
        target
    end
end

function _run_driver_reserved!(driver::StudioDriver, target)
    try
        result = if isnothing(driver.simulation)
            driver.advance(target, driver.recorder)
        else
            simulate!(driver.simulation, target; observer = driver.recorder, driver.options...)
        end
        actual = Float64(result.time)
        isfinite(actual) || throw(ArgumentError("the simulation returned a non-finite time"))
        stopped = result.stopped
        stopped isa Bool || throw(ArgumentError("the simulation must report whether it stopped"))
        Base.@lock driver.lock begin
            driver.time = actual
            driver.stopped = stopped
        end
        return result
    catch
        Base.@lock driver.lock driver.failed = true
        rethrow()
    finally
        Base.@lock driver.lock driver.active = false
    end
end

function simulate!(driver::StudioDriver, final_time)
    target = _reserve_driver!(driver, final_time)
    return _run_driver_reserved!(driver, target)
end

function _driver_snapshot(driver::StudioDriver)
    return Base.@lock driver.lock begin
        target = driver.final_time
        suggested = if isnothing(target)
            nothing
        else
            span = driver.initial_interval
            base = driver.stopped && !driver.active ? driver.time : target
            candidate = base + span
            isfinite(candidate) && candidate > base ? candidate : nothing
        end
        (; final_time = target, driver.active, resumable = !driver.failed, suggested)
    end
end
