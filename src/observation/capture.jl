"""
Capture plan of one registry: deduplicated value slots, their sources, and the
interfaces and solids drawn with them. Built once per `simulate!` call.
"""
struct CapturePlan
    manifest::FieldManifest
    sources::Vector{AbstractVisualizationFieldSource}
    interfaces::Vector{Any}
    geometries::Vector{Any}
    interface_solids::Vector{Int}
end

_slot_key(source::CellVisualizationSource) = (:cell, source.field)
_slot_key(source::FaceVisualizationSource) = (:face, source.field)
_slot_key(source::FaceComponentVisualizationSource{Axis}) where {Axis} =
    (:component, source.field, Axis)
_slot_key(source::DerivedFaceVisualizationSource) =
    (:derived, typeof(source.derived), source.field)
_slot_key(source::PhaseVisualizationSource) = (:phase, source.quantity)
_slot_key(source::PhaseVelocityComponentVisualizationSource{Axis}) where {Axis} =
    (:phase_component, source.velocity, source.interface, Axis)
_slot_key(source::PhaseVelocityDerivedVisualizationSource) =
    (:phase_derived, typeof(source.derived), source.velocity, source.interface)

_symmetric_source(source) = false
_symmetric_source(source::DerivedFaceVisualizationSource) =
    source.derived isa Vorticity
_symmetric_source(source::PhaseVelocityDerivedVisualizationSource) =
    source.derived isa Vorticity

function _identity_index(items, item)
    index = findfirst(candidate -> candidate === item, items)
    return isnothing(index) ? 0 : index
end

function _capture_plan(registry::VisualizationFieldRegistry)
    interfaces = copy(registry.interfaces)
    geometries = copy(registry.geometries)
    sources = AbstractVisualizationFieldSource[]
    slots = IdDict{Any, Int}()
    entries = FieldEntry[]
    index = Dict{Symbol, Int}()
    for entry in registry.entries
        source = entry.source
        slot = get!(slots, _slot_key(source)) do
            push!(sources, source)
            length(sources)
        end
        interface = _is_discontinuous_source(source) ?
            _identity_index(interfaces, _source_interface(source)) : 0
        push!(
            entries, FieldEntry(
                entry.name, entry.label, entry.units, entry.colormap,
                entry.colorrange, _source_placement(source), interface, slot,
                _symmetric_source(source),
            )
        )
        index[entry.name] = length(entries)
    end
    interface_solids = Int[
        isnothing(interface.geometry) ? 0 :
            _identity_index(geometries, interface.geometry)
            for interface in interfaces
    ]
    return CapturePlan(
        FieldManifest(entries, index), sources, interfaces, geometries,
        interface_solids,
    )
end

"""Snapshot and value gather schedules of the current mesh topology."""
struct CaptureTopology{T}
    snapshot::TopologySnapshot{T}
    # Interior and boundary schedule indices of the faces normal to each axis,
    # in the order of the per-axis staggered values.
    interior_faces::NTuple{2, Vector{Int32}}
    boundary_faces::NTuple{2, Vector{Int32}}
end

function _capture_topology(id::Int, mesh::BlockTreeMesh, geometries)
    interior = ntuple(axis -> Int32[i for (i, face) in pairs(mesh.faces) if face.axis == axis], 2)
    boundary = ntuple(
        axis -> Int32[i for (i, face) in pairs(mesh.boundary_faces) if face.axis == axis], 2,
    )
    snapshot = _topology_snapshot(id, mesh, geometries, interior, boundary)
    return CaptureTopology(snapshot, interior, boundary)
end

_target_type(precision, ::Type{T}) where {T} = isnothing(precision) ? T : precision

function _copy_cell_values(precision, field::CellField{T}) where {T}
    E = _target_type(precision, T)
    nx, ny = field.mesh.block_size
    values = Vector{E}(undef, ncells(field.mesh))
    _parallel_block_columns(field.mesh) do block, first_j, last_j
        owned = interior(field, block)
        offset = (block - 1) * nx * ny
        for j in first_j:last_j, i in 1:nx
            @inbounds values[offset + (j - 1) * nx + i] = owned[i, j]
        end
    end
    return values
end

function _copy_face_values(precision, field::FaceField{Axis, T}) where {Axis, T}
    E = _target_type(precision, T)
    count = length(field.interior)
    values = Vector{E}(undef, count + length(field.boundary))
    copyto!(values, 1, field.interior, 1, count)
    copyto!(values, count + 1, field.boundary, 1, length(field.boundary))
    return values
end

function _gather_face_component(
        precision, velocity::FaceVelocity{T}, axis::Int,
        topology::CaptureTopology,
    ) where {T}
    E = _target_type(precision, T)
    interior_faces = topology.interior_faces[axis]
    boundary_faces = topology.boundary_faces[axis]
    count = length(interior_faces)
    values = Vector{E}(undef, count + length(boundary_faces))
    source_interior, source_boundary = velocity.interior, velocity.boundary
    _parallel_for(count) do k
        @inbounds values[k] = source_interior[interior_faces[k]]
    end
    for k in eachindex(boundary_faces)
        @inbounds values[count + k] = source_boundary[boundary_faces[k]]
    end
    return values
end

_convert_values(precision, values::AbstractVector{T}) where {T} =
    _target_type(precision, T) === T ? values :
    convert(Vector{_target_type(precision, T)}, values)

_replacement(replacements, field) = get(replacements, field, field)

_capture_slot(source::CellVisualizationSource, context) =
    _copy_cell_values(context.precision, _replacement(context.replacements, source.field))

_capture_slot(source::FaceVisualizationSource, context) =
    _copy_face_values(context.precision, _replacement(context.replacements, source.field))

_capture_slot(source::FaceComponentVisualizationSource{Axis}, context) where {Axis} =
    _gather_face_component(
    context.precision, _replacement(context.replacements, source.field), Axis,
    context.topology,
)

function _capture_slot(source::DerivedFaceVisualizationSource, context)
    field = source.field
    values = if field isa FaceVelocity
        visualization_values(_replacement(context.replacements, field), source.derived)
    else
        components = (
            _replacement(context.replacements, field[1]),
            _replacement(context.replacements, field[2]),
        )
        visualization_values(components, source.derived)
    end
    return _convert_values(context.precision, values)
end

function _capture_slot(source::PhaseVisualizationSource{<:PhaseScalar}, context)
    scalar = source.quantity
    return TwoPhaseVisualizationValues(
        _copy_cell_values(context.precision, _replacement(context.replacements, scalar.liquid)),
        _copy_cell_values(context.precision, _replacement(context.replacements, scalar.gas)),
    )
end

function _capture_slot(source::PhaseVisualizationSource{<:PhaseFaceScalar}, context)
    scalar = source.quantity
    return TwoPhaseVisualizationValues(
        _copy_face_values(context.precision, _replacement(context.replacements, scalar.liquid)),
        _copy_face_values(context.precision, _replacement(context.replacements, scalar.gas)),
    )
end

function _capture_slot(
        source::PhaseVelocityComponentVisualizationSource{Axis}, context,
    ) where {Axis}
    return TwoPhaseVisualizationValues(
        _gather_face_component(
            context.precision, _replacement(context.replacements, source.velocity[1]),
            Axis, context.topology,
        ),
        _gather_face_component(
            context.precision, _replacement(context.replacements, source.velocity[2]),
            Axis, context.topology,
        ),
    )
end

function _capture_slot(source::PhaseVelocityDerivedVisualizationSource, context)
    values = _cell_phase_values!(
        context.registry, source, context.replacements, context.cache,
    )
    return TwoPhaseVisualizationValues(
        _convert_values(context.precision, values.liquid),
        _convert_values(context.precision, values.gas),
    )
end

function _capture_phase(precision, interface, solid::Int)
    T = eltype(interface)
    F = _target_type(precision, T)
    mesh = interface.mesh
    nx, ny = mesh.block_size
    liquid = falses(ncells(mesh))
    mixed = Int32[]
    planes = NTuple{3, F}[]
    floor = interface.fraction_floor
    fraction = interface.conditional_fraction
    for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        ref = CellRef(block, i, j)
        index = _cell_index(ref, nx, ny)
        value = _get(fraction, ref)
        liquid[index] = value >= T(0.5)
        if floor < value < one(T) - floor
            push!(mixed, Int32(index))
            push!(
                planes, (
                    F(_get(interface.normal_x, ref)), F(_get(interface.normal_y, ref)),
                    F(_get(interface.intercept, ref)),
                )
            )
        end
    end
    segments = NTuple{2, NTuple{2, F}}[
        (F.(segment[1]), F.(segment[2])) for segment in plic_segments(interface)
    ]
    return PhaseSnapshot{F}(liquid, mixed, planes, segments, solid)
end

"""
Capture the registered fields of the current (or, with `replacements`, an
independently advanced) state into a new frame.
"""
function _capture_frame(
        plan::CapturePlan, registry::VisualizationFieldRegistry,
        topology::CaptureTopology, info::FrameInfo;
        precision = nothing, replacements = IdDict{Any, Any}(),
    )
    context = (;
        precision, replacements, topology, registry,
        cache = Dict{Tuple{Any, Any, Symbol}, Any}(),
    )
    values = Any[_capture_slot(source, context) for source in plan.sources]
    phases = PhaseSnapshot[
        _capture_phase(
            precision, _visualization_sampling_interface(interface, replacements),
            plan.interface_solids[index],
        ) for (index, interface) in pairs(plan.interfaces)
    ]
    slot_stats = [_field_stats(slot_values) for slot_values in values]
    stats = Dict{Symbol, FieldStats}(
        entry.name => slot_stats[entry.slot] for entry in plan.manifest.entries
    )
    bytes = sum(_value_bytes, values; init = 0) + sum(_phase_bytes, phases; init = 0)
    return Frame(info, topology.snapshot, plan.manifest, values, phases, stats, bytes)
end

# Frame of the current state of a registry outside any recorded run (static
# figures of live fields); `replacements` substitute independent copies.
function _registry_frame(
        registry::VisualizationFieldRegistry; time = NaN, step = 0,
        replacements = IdDict{Any, Any}(),
    )
    mesh = registry.mesh
    isnothing(mesh) && throw(ArgumentError("the field registry is empty"))
    plan = _capture_plan(registry)
    topology = _capture_topology(0, mesh, plan.geometries)
    info = FrameInfo(0, Float64(time), Int(step), :manual, Base.time(), 0, Float64[])
    return _capture_frame(plan, registry, topology, info; replacements)
end
