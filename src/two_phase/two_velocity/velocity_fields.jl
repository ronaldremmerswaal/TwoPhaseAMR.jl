"""Zero phase weight excludes a value; in particular, never compute 0*NaN."""
@inline _phase_product(weight, value) = iszero(weight) ? zero(weight) : weight * value

@inline function _phase_cell_is_active(interface, cell, phase)
    return _phase_cell_is_active(interface, interface.geometry, cell, phase)
end

@inline function _phase_cell_is_active(interface, geometry, cell, phase)
    !isnothing(geometry) && iszero(_get(geometry.open_fraction, cell)) && return false
    fraction = _conditional_fraction(interface, cell)
    return phase == :liquid ? fraction > 0 : fraction < 1
end

@inline function _phase_face_is_active(interface, index, face, phase)
    return _phase_face_is_active(interface, interface.geometry, index, face, phase)
end

@inline function _phase_face_is_active(interface, geometry, index, face, phase)
    !isnothing(geometry) && iszero(geometry.face_dual_volumes[index]) && return false
    fraction = _staggered_liquid_fraction(interface, geometry, face)
    return phase == :liquid ? fraction > 0 : fraction < 1
end

"""Mark absent intrinsic velocities, using the open staggered phase volume.

Absence means exactly zero phase volume, not a density-dependent tolerance.
Values on arbitrarily small but positive phase volumes remain physical. A
missing value on such an active face is an error, not a request to use zero.
The interior faces are marked in fixed chunks, which record their first
offending face in `failures` (caller-owned scratch, one entry per chunk; the
per-step callers pass their workspace's, the default allocates); the error
names the first such face, as a serial pass would, after every interior face
has been marked. Without a solid the chunks run threaded over the phase face
arrays and the conditional-fraction block vectors, which are not plain data and
are therefore boxed per launch; the cut-cell fraction reads the solid
reconstruction and runs serially.
"""
function _mark_inactive_phase_velocities!(
        velocity::TwoPhaseVelocity, interface::InterfaceField{T},
        failures::Vector{Int} = Int[],
    ) where {T}
    geometry = interface.geometry
    chunks = _reduction_chunks(length(interface.mesh.faces))
    _reduction_partials!(failures, length(interface.mesh.faces))
    _mark_inactive_interior!(failures, velocity[1].interior, velocity[2].interior, interface, geometry)
    for chunk in 1:chunks
        failure = @inbounds failures[chunk]
        iszero(failure) && continue
        phase = isodd(failure) ? 1 : 2
        throw(
            ArgumentError(
                "non-finite active phase-$phase velocity on face $(cld(failure, 2))"
            )
        )
    end
    for (index, face) in pairs(interface.mesh.boundary_faces)
        fraction = _conditional_fraction(interface, face.cell)
        open = isnothing(geometry) || _get(geometry.open_fraction, face.cell) > zero(T)
        for phase in 1:2
            active = open && (phase == 1 ? fraction > zero(T) : fraction < one(T))
            if active
                isfinite(velocity[phase].boundary[index]) || throw(
                    ArgumentError(
                        "non-finite active phase-$phase boundary velocity on face $index"
                    )
                )
            else
                velocity[phase].boundary[index] = T(NaN)
            end
        end
    end
    return velocity
end

# Full Cartesian faces: a threaded launch over the phase face arrays and the
# conditional-fraction block vectors (boxed per launch, see `_parallel_for`).
function _mark_inactive_interior!(failures, liquid, gas, interface, ::Nothing)
    T = eltype(liquid)
    faces = interface.mesh.faces
    leaf_blocks = interface.mesh.blocks
    fraction_blocks = interface.conditional_fraction.blocks
    halo = interface.mesh.halo
    number_of_faces = length(faces)
    @_parallel work = _REDUCTION_CHUNK for chunk in 1:_reduction_chunks(number_of_faces)
        failure = 0
        for index in _reduction_range(chunk, number_of_faces)
            fraction = _staggered_liquid_fraction(
                leaf_blocks, fraction_blocks, halo, @inbounds(faces[index]), T,
            )
            failure = _mark_inactive_face!(liquid, gas, index, fraction, true, failure)
        end
        @inbounds failures[chunk] = failure
    end
    return failures
end

# Cut cells: the open-volume weighted fraction reads the solid's
# reconstruction through its fields; a threaded launch would box them on every
# call, so these chunks run serially.
function _mark_inactive_interior!(failures, liquid, gas, interface, geometry::GeometryField)
    faces = interface.mesh.faces
    number_of_faces = length(faces)
    for chunk in 1:_reduction_chunks(number_of_faces)
        failure = 0
        for index in _reduction_range(chunk, number_of_faces)
            fraction = _staggered_liquid_fraction(interface, geometry, faces[index])
            face_open = geometry.face_dual_volumes[index] > zero(eltype(liquid))
            failure = _mark_inactive_face!(liquid, gas, index, fraction, face_open, failure)
        end
        failures[chunk] = failure
    end
    return failures
end

"""Mark the absent phases of interior face `index` with `NaN`; return
`failure`, or `2index - 1` (liquid) / `2index` (gas) for the first active
non-finite velocity when `failure` is still zero."""
@inline function _mark_inactive_face!(liquid, gas, index, fraction, face_open, failure)
    T = eltype(liquid)
    if face_open && fraction > zero(T)
        iszero(failure) && !isfinite(@inbounds liquid[index]) && (failure = 2index - 1)
    else
        @inbounds liquid[index] = T(NaN)
    end
    if face_open && fraction < one(T)
        iszero(failure) && !isfinite(@inbounds gas[index]) && (failure = 2index)
    else
        @inbounds gas[index] = T(NaN)
    end
    return failure
end

"""Two-layer, same-component extension in *working storage* only.

The owned phase velocity remains NaN where absent. This finite extension
supplies missing interpolation/transport stencil entries, exclusively from
valid neighbours. Beyond the stencil band zero is an algebraic placeholder,
never an intrinsic velocity or a donor on a zero-volume phase.
"""
function _extend_phase_face_values!(face_values, next, work; layers = 2, zero_missing = true)
    any(isnan, face_values) || return face_values
    T = eltype(face_values)
    offsets = work.dual_offsets
    dual_faces = work.dual_faces
    schedule_faces = work.schedule.faces
    for _ in 1:layers
        # Each output averages its finite neighbours of the previous layer in
        # incidence order: owned outputs, the same sums at every thread count.
        @_parallel for output in 1:length(face_values)
            value = @inbounds face_values[output]
            if isfinite(value)
                @inbounds next[output] = value
                continue
            end
            total = zero(T)
            neighbours = 0
            for slot in @inbounds(offsets[output]):(@inbounds(offsets[output + 1]) - 1)
                face = @inbounds schedule_faces[dual_faces[slot]]
                neighbour = face.minus == output ? face.plus : face.minus
                neighbour_value = @inbounds face_values[neighbour]
                isfinite(neighbour_value) || continue
                total += neighbour_value
                neighbours += 1
            end
            @inbounds next[output] = iszero(neighbours) ? T(NaN) : total / neighbours
        end
        copyto!(face_values, next)
    end
    if zero_missing
        @_parallel for index in 1:length(face_values)
            isnan(@inbounds face_values[index]) && (@inbounds face_values[index] = zero(T))
        end
    end
    return face_values
end

function _extend_phase_velocity_for_transfer!(
        velocity::FaceVelocity{T}, interface,
        scratch::PhaseTransferScratch{T} = PhaseTransferScratch(T)
    ) where {T}
    # AMR calls this on its finite source buffer, not on owned phase storage.
    # Connectivity still belongs to the old face schedule even if the tree
    # has already been edited. Both phases share it; donor masks do not.
    mesh = velocity.mesh
    _extend_phase_transfer_component!(
        velocity,
        _phase_transport_connectivity(mesh, Val(1), T), scratch
    )
    _extend_phase_transfer_component!(
        velocity,
        _phase_transport_connectivity(mesh, Val(2), T), scratch
    )
    return velocity
end

function _extend_phase_transfer_component!(
        velocity,
        graph::PhaseTransportConnectivity, scratch
    )
    values = scratch.values
    interior_count = length(graph.interior_indices)
    resize!(values, interior_count + length(graph.boundary_indices))
    for (local_index, index) in pairs(graph.interior_indices)
        values[local_index] = velocity.interior[index]
    end
    for (local_index, index) in pairs(graph.boundary_indices)
        values[interior_count + local_index] = velocity.boundary[index]
    end
    _extend_phase_transfer_values!(values, graph, velocity.mesh, graph.centers, scratch)
    for (local_index, index) in pairs(graph.interior_indices)
        velocity.interior[index] = values[local_index]
    end
    for (local_index, index) in pairs(graph.boundary_indices)
        velocity.boundary[index] = values[interior_count + local_index]
    end
    return velocity
end

function _extend_phase_transfer_values!(values, work, mesh, axis)
    T = eltype(values)
    centers = Tuple{T, T}[face.center for face in mesh.faces if face.axis == axis]
    append!(centers, (face.center for face in mesh.boundary_faces if face.axis == axis))
    return _extend_phase_transfer_values!(values, work, mesh, centers, PhaseTransferScratch(T))
end

function _extend_phase_transfer_values!(
        values, work, mesh, centers,
        scratch::PhaseTransferScratch
    )
    any(isnan, values) || return values
    T = eltype(values)
    original, known, next = scratch.original, scratch.known, scratch.next
    resize!(original, length(values))
    resize!(known, length(values))
    resize!(next, length(values))
    copyto!(original, values)
    for index in eachindex(values)
        known[index] = isfinite(original[index])
    end
    # The Cartesian interpolation stencil also reaches closed solid faces.
    # Extend into that scratch band from valid phase data, not from dormant
    # solid/other-phase velocities. No extended value becomes a fit donor.
    _extend_phase_face_values!(values, next, work; zero_missing = false)
    candidates = scratch.candidates
    for target in eachindex(values)
        known[target] && continue
        if isnan(values[target])
            values[target] = zero(T)  # Outside the transfer stencil band.
            continue
        end
        empty!(candidates)
        push!(candidates, target)
        first = last = 1
        for _ in 1:3
            for current in first:last
                index = candidates[current]
                for pointer in work.dual_offsets[index]:(work.dual_offsets[index + 1] - 1)
                    face = work.schedule.faces[work.dual_faces[pointer]]
                    neighbour = face.minus == index ? face.plus : face.minus
                    neighbour in candidates || push!(candidates, neighbour)
                end
            end
            first, last = last + 1, length(candidates)
        end
        sw = sx = sy = sxx = sxy = syy = su = sxu = syu = zero(T)
        for donor in candidates
            known[donor] || continue
            dx = T(
                _wrapped_delta(
                    centers[donor][1] - centers[target][1],
                    mesh.extent[1], mesh.periodic[1]
                )
            )
            dy = T(
                _wrapped_delta(
                    centers[donor][2] - centers[target][2],
                    mesh.extent[2], mesh.periodic[2]
                )
            )
            distance_squared = dx^2 + dy^2
            distance_squared > zero(T) || continue
            w = inv(distance_squared)
            value = original[donor]
            sw += w; sx += w * dx; sy += w * dy
            sxx += w * dx^2; sxy += w * dx * dy; syy += w * dy^2
            su += w * value; sxu += w * dx * value; syu += w * dy * value
        end
        iszero(sw) && continue
        mx, my, value = sx / sw, sy / sw, su / sw
        a, b, c = sxx - sx * mx, sxy - sx * my, syy - sy * my
        determinant = a * c - b^2
        # An affine extension retains the linear-exact AMR interpolation.
        # Rank-deficient donor sets keep the constant working extension.
        if determinant > 128eps(T) * max(abs(a * c), b^2)
            bx, by = sxu - sx * value, syu - sy * value
            values[target] = value -
                ((c * bx - b * by) * mx + (a * by - b * bx) * my) / determinant
        end
    end
    return values
end

"""Validate the storage contract shared by all two-velocity entry points."""

function _validate_two_phase_velocity(
        velocity::TwoPhaseVelocity, mesh::BlockTreeMesh, ::Type{T},
    ) where {T}
    for (phase, component) in zip((:liquid, :gas), velocity)
        component.mesh === mesh || throw(
            ArgumentError(
                "$phase velocity and the two-phase field must share a mesh",
            )
        )
        eltype(component) === T || throw(
            ArgumentError(
                "$phase velocity has element type $(eltype(component)); expected $T",
            )
        )
        length(component.interior) == length(mesh.faces) || throw(
            DimensionMismatch("$phase velocity has the wrong interior-face layout"),
        )
        length(component.boundary) == length(mesh.boundary_faces) || throw(
            DimensionMismatch("$phase velocity has the wrong boundary-face layout"),
        )
    end
    return velocity
end

function register_adaptive_field!(
        workspace::AMRWorkspace,
        velocity::TwoPhaseVelocity, interface::InterfaceField
    )
    interface.mesh === workspace.mesh || throw(ArgumentError("AMR interface mesh differs"))
    register_adaptive_field!(workspace, velocity)
    register_adaptive_field!(workspace, interface)
    any(
        pair -> pair[1] === velocity[1] && pair[2] === velocity[2],
        workspace.phase_velocities
    ) || push!(
        workspace.phase_velocities,
        (velocity[1], velocity[2], interface)
    )
    return workspace
end

function _associate_adaptive_phase_velocities!(workspace::AMRWorkspace, fields)
    interfaces = filter(field -> field isa InterfaceField, fields)
    velocities = filter(field -> field isa TwoPhaseVelocity, fields)
    isempty(velocities) && return workspace
    isempty(interfaces) && return workspace
    length(interfaces) == 1 || throw(
        ArgumentError(
            "register each two-phase velocity with its interface explicitly when AMR uses multiple interfaces"
        )
    )
    for velocity in velocities
        register_adaptive_field!(workspace, velocity, only(interfaces))
    end
    return workspace
end

_adaptive_refinement_labels!(workspace, indicator, time) =
    _refinement_labels_at_time!(workspace.indicator_labels, indicator, time)

function _adaptive_refinement_labels!(
        workspace,
        indicator::VorticityRefinementIndicator, time
    )
    if isnothing(indicator.interface)
        for (liquid, gas, interface) in workspace.phase_velocities
            phase = indicator.field === liquid ? :liquid :
                indicator.field === gas ? :gas : nothing
            isnothing(phase) && continue
            return refinement_labels!(
                workspace.indicator_labels, indicator;
                time, interface, phase, _vorticity_workspace = workspace.vorticity
            )
        end
    end
    return refinement_labels!(
        workspace.indicator_labels, indicator; time,
        _vorticity_workspace = workspace.vorticity
    )
end

function _prepare_adaptive_phase_velocities!(workspace::AMRWorkspace)
    for (liquid, gas, interface) in workspace.phase_velocities
        _prepare_adaptive_phase_pair!(workspace, liquid, gas, interface)
    end
    return
end

function _prepare_adaptive_phase_pair!(workspace, liquid, gas, interface)
    _mark_inactive_phase_velocities!((liquid, gas), interface)
    for velocity in (liquid, gas)
        buffer = workspace.field_buffers[_find_field_buffer(workspace, velocity)]
        _stage_phase_velocity!(buffer, velocity, interface, workspace.phase_transfer)
    end
    return
end

function _stage_phase_velocity!(buffer::FaceAMRFieldBuffer, velocity, interface, scratch)
    resize!(buffer.old_interior, length(velocity.interior))
    resize!(buffer.old_boundary, length(velocity.boundary))
    copyto!(buffer.old_interior, velocity.interior)
    copyto!(buffer.old_boundary, velocity.boundary)
    working = FaceVelocity(velocity.mesh, buffer.old_interior, buffer.old_boundary)
    return _extend_phase_velocity_for_transfer!(working, interface, scratch)
end

function _finish_adaptive_phase_velocities!(workspace::AMRWorkspace)
    for (liquid, gas, interface) in workspace.phase_velocities
        _mark_inactive_phase_velocities!((liquid, gas), interface)
    end
    return
end

"""Set both phase velocities from `(liquid_function, gas_function)`."""
function set_velocities!(
        velocity::TwoPhaseVelocity, functions::Tuple; time = 0,
    )
    length(functions) == 2 || throw(
        ArgumentError(
            "two phase-velocity initializers are required",
        )
    )
    liquid, gas = velocity
    _validate_two_phase_velocity(velocity, liquid.mesh, eltype(liquid))
    set_velocities!(liquid, functions[1]; time)
    set_velocities!(gas, functions[2]; time)
    return velocity
end

"""Apply the impermeable-domain condition to both phase velocities."""
function enforce_no_permeation!(velocity::TwoPhaseVelocity)
    for component in velocity, index in eachindex(component.boundary)
        isnan(component.boundary[index]) && continue
        component.boundary[index] = zero(eltype(component))
    end
    return velocity
end

"""Apply the fixed-solid no-slip mask to both phase velocities."""
function enforce_geometry_no_slip!(
        velocity::TwoPhaseVelocity, geometry::GeometryField,
    )
    _validate_two_phase_velocity(
        velocity, geometry.mesh, eltype(geometry),
    )
    foreach(
        component -> enforce_geometry_no_slip!(
            component, geometry;
            preserve_absent = true
        ), velocity
    )
    return velocity
end

"""
Register both intrinsic velocity fields for AMR transfer.

The existing face interpolation and divergence-preserving refinement are used
for each phase.  The global face-integral correction is disabled: these fields
store velocity, not phase momentum.
"""
function register_adaptive_field!(
        workspace::AMRWorkspace{T}, velocity::TwoPhaseVelocity,
    ) where {T}
    liquid, gas = velocity
    _validate_two_phase_velocity(
        velocity, workspace.mesh, T,
    )
    for component in (liquid, gas)
        register_adaptive_field!(
            workspace, component;
            piecewise_constant = false, conserve_integral = false,
        )
    end
    return workspace
end
