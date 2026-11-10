# Store indices are copy-on-write; frames and their topology are owned snapshots.
_finite(x::Real) = isfinite(x) ? x : nothing
_placement(::TwoPhaseAMR.Cell) = "cell"
_placement(::TwoPhaseAMR.Face{1}) = "face_x"
_placement(::TwoPhaseAMR.Face{2}) = "face_y"

function _field_metadata(entry)
    colormap = entry.colormap isa Union{Symbol, AbstractString} ? string(entry.colormap) : nothing
    colorrange = entry.colorrange isa Tuple && length(entry.colorrange) == 2 &&
        all(x -> x isa Real && isfinite(x), entry.colorrange) ? collect(entry.colorrange) : nothing
    return (;
        name = string(entry.name), entry.label, entry.units, colormap, colorrange,
        placement = _placement(entry.placement), entry.interface, entry.symmetric,
    )
end

function _diagnostic_metadata(pair)
    name, info = pair
    dimension = isnothing(info.dimension) ? nothing : (;
            mass = Int(info.dimension.mass), length = Int(info.dimension.length),
            time = Int(info.dimension.time),
        )
    return (;
        name = string(name), info.label, info.units, dimension,
        components = copy(info.components), kind = string(info.kind), scale = string(info.scale),
    )
end

function _field_manifest(recorder, store)
    frame = TwoPhaseAMR.newest_frame(store)
    # Before the first frame the same capture plan supplies the eventual manifest.
    manifest = isnothing(frame) ? TwoPhaseAMR._capture_plan(recorder.registry).manifest : frame.manifest
    return [_field_metadata(entry) for entry in manifest.entries]
end

function _diagnostic_manifest(recorder, store)
    infos = TwoPhaseAMR.diagnostic_info(store)
    if isempty(infos) && !isnothing(recorder.diagnostics)
        infos = TwoPhaseAMR.diagnostic_info(recorder.diagnostics)
    end
    return [_diagnostic_metadata(pair) for pair in infos]
end

_frame_info(info) = (;
    info.id, time = _finite(info.time), info.step,
    kind = string(info.kind), topology = info.topology,
)

function _frame_stats(frame)
    return Dict(
        string(name) => (;
            minimum = _finite(stats.minimum), maximum = _finite(stats.maximum),
            absmax = _finite(stats.absmax), mean = _finite(stats.mean), stats.finite,
            histogram = stats.histogram,
        ) for (name, stats) in frame.stats
    )
end

# The index is copy-on-write, but previews in an older index may already have
# been thinned. Pin the surviving immutable frames briefly under the store
# lock, then serialize their statistics without holding the solver's lock.
function _frame_index(store, index)
    entries = Base.@lock store.lock [
        (info, frame) for info in index
            for frame in (get(store.frames, info.id, nothing),) if !isnothing(frame)
    ]
    return [(; _frame_info(info)..., stats = _frame_stats(frame)) for (info, frame) in entries]
end

_frame_message(frame) = (;
    type = "frame", run = "run-1", id = frame.id,
    time = _finite(frame.time), step = frame.step, kind = string(frame.kind),
    topology = frame.info.topology, stats = _frame_stats(frame),
)

function _run_metadata(recorder, store, name)
    frame = TwoPhaseAMR.newest_frame(store)
    mesh = isnothing(frame) ? recorder.registry.mesh : frame.topology
    domain = isnothing(mesh) ? nothing : (; origin = mesh.origin, extent = mesh.extent)
    block_size = isnothing(mesh) ? nothing : mesh.block_size
    index = TwoPhaseAMR.frames(store)
    run = Base.@lock recorder.lock recorder.run
    t0 = !isnothing(run) ? _finite(run.initial_time) : isempty(index) ? nothing : _finite(first(index).time)
    tf = !isnothing(run) ? _finite(run.final_time) :
        TwoPhaseAMR.run_state(recorder) === :completed && !isempty(index) ? _finite(last(index).time) : nothing
    return (;
        id = "run-1", name, domain, block_size, t0, t_final = tf,
        fields = _field_manifest(recorder, store), diagnostics = _diagnostic_manifest(recorder, store),
    )
end

function _topology_message(topology, request)
    blocks = [
        (;
            level = key.level, i = key.i, j = key.j,
            origin = topology.origins[k], spacing = topology.spacings[k],
        )
            for (k, key) in pairs(topology.keys)
    ]
    solids = [
        (;
            full = collect(solid.full), cut = solid.cut, planes = solid.planes,
            revision = solid.revision, polygons = (;
                points = solid.polygons.points,
                triangles = solid.polygons.triangles,
            ),
        ) for solid in topology.solids
    ]
    # Block and cell/face indices are Julia's one-based indices; zero in a
    # face tile means no face. Block-key i,j remain zero-based mesh coordinates.
    return (;
        type = "topology", request, id = topology.id, generation = topology.generation,
        blocks, block_size = topology.block_size, base_mesh = topology.base_mesh,
        domain = (; origin = topology.origin, extent = topology.extent),
        periodic = topology.periodic, face_counts = topology.face_counts,
        face_tiles = topology.face_tiles, solids, max_level = topology.max_level,
    )
end

function _planes_message(frame, request)
    phases = [
        (;
            liquid = collect(phase.liquid), mixed = phase.mixed,
            planes = phase.planes, segments = phase.segments, solid = phase.solid,
        )
            for phase in frame.phases
    ]
    return (; type = "planes", request, frame = frame.id, phases)
end

function _diagnostics_message(store, names, request)
    series = map(names) do name
        history = TwoPhaseAMR.diagnostic_history(store, name)
        return (;
            name = string(name), info = _diagnostic_metadata(name => history.info),
            times = _finite.(history.times), steps = history.steps,
            components = history.components,
            values = [
                [_finite(history.values[i, j]) for j in axes(history.values, 2)]
                    for i in axes(history.values, 1)
            ],
        )
    end
    return (; type = "diagnostics", request, series)
end

function _steplog_message(store, request)
    log = TwoPhaseAMR.step_log(store)
    return (;
        type = "steplog", request, time = _finite.(log.time), step = log.step,
        dt = _finite.(log.dt), wall_time = _finite.(log.wall_time), iterations = log.iterations,
        initial_residual = _finite.(log.initial_residual), final_residual = _finite.(log.final_residual),
        converged = log.converged, topology_changed = log.topology_changed,
        capture_time = _finite.(log.capture_time),
    )
end
