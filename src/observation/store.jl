const _RETENTION_POLICIES = (:logarithmic, :uniform, :none, :all)

"""
    Retention(; memory=:auto, preview=:logarithmic, recent=64)

Retention policy of a [`SnapshotStore`](@ref). Keyframes (initial,
scheduled, manual, pause, and final frames) are never dropped. Previews follow
`preview`:

- `:logarithmic`: the newest `recent` previews at full density, older ones
  thinned so that density halves each time their age doubles;
- `:uniform`: previews are thinned evenly in time, only when over budget;
- `:none`: only the newest preview is kept;
- `:all`: every preview is kept while the budget allows.

`memory` is the budget in bytes; `:auto` is min(2 GiB, 20 % of physical
memory). Over budget, previews are thinned evenly until the store fits. If
keyframes alone exceed the budget they stay in memory and one warning is
logged; recording never aborts the simulation.
"""
struct Retention
    memory::Int
    preview::Symbol
    recent::Int

    function Retention(; memory = :auto, preview = :logarithmic, recent = 64)
        bytes = if memory === :auto
            min(2^31, floor(Int, 0.2 * Sys.total_memory()))
        elseif memory isa Integer && !(memory isa Bool) && memory > 0
            Int(memory)
        else
            throw(ArgumentError("memory must be :auto or a positive number of bytes"))
        end
        preview in _RETENTION_POLICIES || throw(
            ArgumentError("preview retention must be one of $(_RETENTION_POLICIES)"),
        )
        recent isa Integer && !(recent isa Bool) && recent >= 1 ||
            throw(ArgumentError("recent must be a positive integer"))
        return new(bytes, preview, Int(recent))
    end
end

"""Mirrored samples of one simulation diagnostic, split into components."""
mutable struct MirroredDiagnostic
    name::Symbol
    info::DiagnosticInfo
    series::Any
    consumed::Int
    components::Vector{String}
    times::Vector{Float64}
    steps::Vector{Int}
    columns::Vector{Vector{Float64}}
end

"""
    DiagnosticColumns

Detached copy of a mirrored diagnostic, returned by
`diagnostic_history(store, name)`: `name`, [`DiagnosticInfo`](@ref) `info`,
sample `times` and `steps`, component names, and a `values` matrix with one
column per component.
"""
struct DiagnosticColumns
    name::Symbol
    info::DiagnosticInfo
    times::Vector{Float64}
    steps::Vector{Int}
    components::Vector{String}
    values::Matrix{Float64}
end

Base.length(history::DiagnosticColumns) = length(history.times)

"""One step-log column set; see [`step_log`](@ref)."""
struct StepLog
    time::Vector{Float64}
    step::Vector{Int}
    dt::Vector{Float64}
    wall_time::Vector{Float64}
    iterations::Vector{Int}
    initial_residual::Vector{Float64}
    final_residual::Vector{Float64}
    converged::Vector{Bool}
    topology_changed::Vector{Bool}
    capture_time::Vector{Float64}
end

StepLog() = StepLog(
    Float64[], Int[], Float64[], Float64[], Int[], Float64[], Float64[], Bool[],
    Bool[], Float64[],
)

Base.copy(log::StepLog) = StepLog(
    (copy(getfield(log, name)) for name in fieldnames(StepLog))...,
)
Base.length(log::StepLog) = length(log.time)

"""
    RunEvent

A recorded event: `kind` (`:pause`, `:resume`, `:breakpoint`, `:stop`, or
`:failure`), simulation `time` and `step`, recorder-clock `wall_time`, and a
`message`.
"""
struct RunEvent
    kind::Symbol
    time::Float64
    step::Int
    wall_time::Float64
    message::String
end

"""
    FrameSubscription

Latest-wins notification of newly stored frames, created by
[`subscribe`](@ref). `take!(subscription)` waits for and returns the newest
[`FrameInfo`](@ref) not yet taken; frames committed while the consumer was busy
are skipped and counted in `subscription.dropped`. Returns `nothing` once the
subscription is closed and empty. Notifications never wait for the consumer.
"""
mutable struct FrameSubscription
    condition::Threads.Condition
    latest::Union{Nothing, FrameInfo}
    closed::Bool
    dropped::Int
end

FrameSubscription() = FrameSubscription(Threads.Condition(), nothing, false, 0)

"""
    SnapshotStore

Bounded in-memory store of the frames, topology snapshots, mirrored
diagnostics, step log, and events of one recorded run. The frame index is
published copy-on-write and frames are immutable, so readers never block the
solver. Obtain it with [`snapshot_store`](@ref).
"""
mutable struct SnapshotStore
    lock::ReentrantLock
    records::Vector{FrameInfo}
    frames::Dict{Int, Frame}
    topologies::Dict{Int, TopologySnapshot}
    retention::Retention
    frame_bytes::Int
    topology_bytes::Int
    next_frame::Int
    next_topology::Int
    preview_sequence::Dict{Int, Int}
    next_preview::Int
    thinned_previews::Int
    skipped_previews::Int
    budget_warned::Bool
    diagnostics::Vector{MirroredDiagnostic}
    diagnostic_index::Dict{Symbol, Int}
    steps::StepLog
    events::Vector{RunEvent}
    subscriptions::Vector{FrameSubscription}
end

SnapshotStore(retention::Retention) = SnapshotStore(
    ReentrantLock(), FrameInfo[], Dict{Int, Frame}(),
    Dict{Int, TopologySnapshot}(), retention, 0, 0, 1, 1, Dict{Int, Int}(), 0,
    0, 0, false, MirroredDiagnostic[], Dict{Symbol, Int}(), StepLog(),
    RunEvent[], FrameSubscription[],
)

function Base.show(io::IO, store::SnapshotStore)
    usage = store_usage(store)
    return print(
        io, "SnapshotStore(", usage.frames, " frames, ", usage.keyframes,
        " keyframes, ", round(usage.bytes / 2^20; digits = 1), " MiB of ",
        round(usage.budget / 2^20; digits = 1), " MiB)",
    )
end

"""Return the [`FrameInfo`](@ref) of every stored frame in commit order."""
frames(store::SnapshotStore) = Base.@lock store.lock store.records

"""Return the stored frames that are never dropped (all except previews)."""
keyframes(store::SnapshotStore) =
    filter(info -> info.kind !== :preview, frames(store))

function Base.getindex(store::SnapshotStore, id::Integer)
    Base.@lock store.lock begin
        frame = get(store.frames, Int(id), nothing)
        isnothing(frame) && throw(KeyError(id))
        return frame
    end
end

Base.getindex(store::SnapshotStore, info::FrameInfo) = store[info.id]
Base.haskey(store::SnapshotStore, id::Integer) =
    Base.@lock store.lock haskey(store.frames, Int(id))
Base.length(store::SnapshotStore) = length(frames(store))

"""Return the most recently stored frame, or `nothing`."""
function latest_frame(store::SnapshotStore)
    Base.@lock store.lock begin
        isempty(store.records) && return nothing
        return store.frames[last(store.records).id]
    end
end

"""
    frame_at(store, time)

Return the stored frame nearest to `time` (the earlier one on ties).
"""
function frame_at(store::SnapshotStore, time::Real)
    Base.@lock store.lock begin
        isempty(store.records) && throw(ArgumentError("the store holds no frames"))
        best = first(store.records)
        for info in store.records
            if abs(info.time - time) < abs(best.time - time)
                best = info
            end
        end
        return store.frames[best.id]
    end
end

"""Return the topology snapshot with the given id."""
topology(store::SnapshotStore, id::Integer) =
    Base.@lock store.lock store.topologies[Int(id)]

"""
    frame_stats(store, name)

Merge the [`FieldStats`](@ref) of field `name` over all stored frames.
"""
function frame_stats(store::SnapshotStore, name::Symbol)
    collected = Base.@lock store.lock FieldStats[
        frame.stats[name] for frame in values(store.frames) if haskey(frame.stats, name)
    ]
    isempty(collected) && throw(ArgumentError("no stored frame has field :$name"))
    return merge(collected...)
end

"""
    store_usage(store)

Memory and retention summary: number of `frames`, `keyframes`, `previews`, and
`topologies`; `bytes` used (frames plus topology snapshots) and the `budget`;
`thinned_previews` dropped by retention and `skipped_previews` not captured
because of the capture-overhead budget.
"""
function store_usage(store::SnapshotStore)
    Base.@lock store.lock begin
        previews = count(info -> info.kind === :preview, store.records)
        return (;
            frames = length(store.records),
            keyframes = length(store.records) - previews, previews,
            topologies = length(store.topologies),
            bytes = store.frame_bytes + store.topology_bytes,
            budget = store.retention.memory,
            thinned_previews = store.thinned_previews,
            skipped_previews = store.skipped_previews,
        )
    end
end

"""Return a copy of the [`StepLog`](@ref) columns: one entry per completed step."""
step_log(store::SnapshotStore) = Base.@lock store.lock copy(store.steps)

"""Return a copy of the recorded [`RunEvent`](@ref)s in order."""
run_events(store::SnapshotStore) = Base.@lock store.lock copy(store.events)

"""
    diagnostic_names(source)

Names of the diagnostics of a `DiagnosticCollection`, `FlowSimulation`, or
[`SnapshotStore`](@ref) (its mirrored diagnostics), in registration order.
"""
diagnostic_names(store::SnapshotStore) =
    Base.@lock store.lock Symbol[mirror.name for mirror in store.diagnostics]

"""
    diagnostic_history(store, name)

Return the mirrored samples of a simulation diagnostic as
[`DiagnosticColumns`](@ref). Vector values are split into components.
"""
function diagnostic_history(store::SnapshotStore, name::Symbol)
    Base.@lock store.lock begin
        index = get(store.diagnostic_index, name, 0)
        iszero(index) && throw(ArgumentError("unknown diagnostic :$name"))
        mirror = store.diagnostics[index]
        values = Matrix{Float64}(undef, length(mirror.times), length(mirror.columns))
        for (component, column) in pairs(mirror.columns)
            values[:, component] = column
        end
        return DiagnosticColumns(
            mirror.name, mirror.info, copy(mirror.times), copy(mirror.steps),
            copy(mirror.components), values,
        )
    end
end

diagnostic_info(store::SnapshotStore) = Base.@lock store.lock Pair{Symbol, DiagnosticInfo}[
    mirror.name => mirror.info for mirror in store.diagnostics
]
diagnostic_groups(store::SnapshotStore) = diagnostic_groups(diagnostic_info(store))

"""
    subscribe(store)::FrameSubscription

Receive latest-wins notifications of frames stored from now on.
"""
function subscribe(store::SnapshotStore)
    subscription = FrameSubscription()
    Base.@lock store.lock push!(store.subscriptions, subscription)
    return subscription
end

function Base.take!(subscription::FrameSubscription)
    Base.@lock subscription.condition begin
        while isnothing(subscription.latest) && !subscription.closed
            wait(subscription.condition)
        end
        latest = subscription.latest
        subscription.latest = nothing
        return latest
    end
end

function Base.close(subscription::FrameSubscription)
    Base.@lock subscription.condition begin
        subscription.closed = true
        notify(subscription.condition; all = true)
    end
    return subscription
end

Base.isopen(subscription::FrameSubscription) =
    Base.@lock subscription.condition !subscription.closed

function _notify_subscriptions!(store::SnapshotStore, info::FrameInfo)
    subscriptions = Base.@lock store.lock begin
        filter!(isopen, store.subscriptions)
        copy(store.subscriptions)
    end
    for subscription in subscriptions
        Base.@lock subscription.condition begin
            subscription.closed && continue
            isnothing(subscription.latest) || (subscription.dropped += 1)
            subscription.latest = info
            notify(subscription.condition)
        end
    end
    return store
end

function _add_topology!(store::SnapshotStore, topology::TopologySnapshot)
    Base.@lock store.lock begin
        store.topologies[topology.id] = topology
        store.topology_bytes += _topology_bytes(topology)
    end
    return topology
end

function _next_topology_id!(store::SnapshotStore)
    Base.@lock store.lock begin
        id = store.next_topology
        store.next_topology += 1
        return id
    end
end

function _next_frame_id!(store::SnapshotStore)
    Base.@lock store.lock begin
        id = store.next_frame
        store.next_frame += 1
        return id
    end
end

_with_info(frame::Frame, info::FrameInfo) = Frame(
    info, frame.topology, frame.manifest, frame.values, frame.phases,
    frame.stats, frame.bytes,
)

function _commit_frame!(store::SnapshotStore, frame::Frame)
    Base.@lock store.lock begin
        store.frames[frame.id] = frame
        store.records = push!(copy(store.records), frame.info)
        store.frame_bytes += frame.bytes
        if frame.kind === :preview
            store.preview_sequence[frame.id] = store.next_preview
            store.next_preview += 1
        end
        _apply_retention!(store)
    end
    _notify_subscriptions!(store, frame.info)
    return frame
end

# Turn a stored frame into a keyframe (it was committed as a preview, or the
# same state now also satisfies scheduled keyframe times). `time` relabels the
# final state, whose time `simulate!` rounds to the requested final time.
function _promote_frame!(
        store::SnapshotStore, id::Int, kind::Symbol, requested_times;
        time = nothing,
    )
    Base.@lock store.lock begin
        frame = store.frames[id]
        new_kind = _kind_rank(kind) > _kind_rank(frame.kind) ? kind : frame.kind
        info = FrameInfo(
            id, isnothing(time) ? frame.info.time : Float64(time), frame.info.step,
            new_kind, frame.info.wall_time, frame.info.topology,
            sort!(unique!(vcat(frame.info.requested_times, requested_times))),
        )
        store.frames[id] = _with_info(frame, info)
        store.records = [record.id == id ? info : record for record in store.records]
        new_kind === :preview || delete!(store.preview_sequence, id)
        return store.frames[id]
    end
end

function _drop_frames!(store::SnapshotStore, ids)
    isempty(ids) && return store
    dropped = Set(ids)
    for id in dropped
        frame = pop!(store.frames, id)
        store.frame_bytes -= frame.bytes
        delete!(store.preview_sequence, id)
    end
    store.records = filter(info -> !(info.id in dropped), store.records)
    store.thinned_previews += length(dropped)
    referenced = Set(info.topology for info in store.records)
    for id in collect(keys(store.topologies))
        id in referenced && continue
        store.topology_bytes -= _topology_bytes(pop!(store.topologies, id))
    end
    return store
end

# Previews eligible for thinning, oldest first. The newest preview is the
# live head and is always kept.
function _thinnable_previews(store::SnapshotStore)
    previews = [info.id for info in store.records if info.kind === :preview]
    isempty(previews) || pop!(previews)
    return previews
end

function _keep_logarithmic(sequence, newest, recent)
    age = newest - sequence
    age < recent && return true
    level = floor(Int, log2(age / recent)) + 1
    return iszero(mod(sequence, 1 << level))
end

function _apply_retention!(store::SnapshotStore)
    retention = store.retention
    candidates = _thinnable_previews(store)
    if retention.preview === :none
        _drop_frames!(store, candidates)
    elseif retention.preview === :logarithmic && !isempty(candidates)
        newest = store.next_preview - 1
        _drop_frames!(
            store, [
                id for id in candidates if
                    !_keep_logarithmic(store.preview_sequence[id], newest, retention.recent)
            ]
        )
    end
    while store.frame_bytes + store.topology_bytes > retention.memory
        candidates = _thinnable_previews(store)
        isempty(candidates) && break
        # Drop every other preview: an even thinning in time.
        _drop_frames!(store, candidates[1:2:end])
    end
    if store.frame_bytes + store.topology_bytes > retention.memory && !store.budget_warned
        store.budget_warned = true
        @warn "Recorded keyframes exceed the memory budget of " *
            "$(round(retention.memory / 2^20; digits = 1)) MiB; they are kept in memory."
    end
    return store
end

function _record_event!(store::SnapshotStore, event::RunEvent)
    Base.@lock store.lock push!(store.events, event)
    return event
end

function _append_step_log!(store::SnapshotStore, values)
    Base.@lock store.lock begin
        log = store.steps
        push!(log.time, values.time)
        push!(log.step, values.step)
        push!(log.dt, values.dt)
        push!(log.wall_time, values.wall_time)
        push!(log.iterations, values.iterations)
        push!(log.initial_residual, values.initial_residual)
        push!(log.final_residual, values.final_residual)
        push!(log.converged, values.converged)
        push!(log.topology_changed, values.topology_changed)
        push!(log.capture_time, values.capture_time)
    end
    return store
end

_value_components(value::Real) = 1
_value_components(value::Union{AbstractVector{<:Real}, Tuple{Vararg{Real}}}) = length(value)
_value_components(value) = 0

_component_value(value::Real, component) = Float64(value)
_component_value(value, component) = Float64(value[component])

_component_names(info::DiagnosticInfo, count) =
    length(info.components) == count ? copy(info.components) :
    count == 1 ? String[] : [string(component) for component in 1:count]

# Copy diagnostic samples appended since the previous call. Returns the new
# raw samples as `(name, value)` pairs for breakpoint evaluation.
function _mirror_diagnostics!(store::SnapshotStore, collection)
    fresh = Tuple{Symbol, Any, Float64, Int}[]
    isnothing(collection) && return fresh
    Base.@lock store.lock begin
        for series in collection.series
            name = diagnostic_name(series.diagnostic)
            index = get(store.diagnostic_index, name, 0)
            if iszero(index) || store.diagnostics[index].series !== series
                mirror = MirroredDiagnostic(
                    name, diagnostic_info(series.diagnostic), series, 0,
                    String[], Float64[], Int[], Vector{Float64}[],
                )
                if iszero(index)
                    push!(store.diagnostics, mirror)
                    store.diagnostic_index[name] = length(store.diagnostics)
                else
                    store.diagnostics[index] = mirror
                end
            end
            mirror = store.diagnostics[store.diagnostic_index[name]]
            for sample in (mirror.consumed + 1):length(series.values)
                value = series.values[sample]
                mirror.consumed = sample
                count = _value_components(value)
                count == 0 && continue
                if isempty(mirror.columns)
                    mirror.components = _component_names(mirror.info, count)
                    mirror.columns = [Float64[] for _ in 1:count]
                end
                count == length(mirror.columns) || continue
                time = Float64(series.times[sample])
                push!(mirror.times, time)
                push!(mirror.steps, series.steps[sample])
                for component in 1:count
                    push!(mirror.columns[component], _component_value(value, component))
                end
                push!(fresh, (name, value, time, series.steps[sample]))
            end
        end
    end
    return fresh
end
