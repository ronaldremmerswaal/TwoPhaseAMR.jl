"""Volume-weighted interpolation from centred cells to staggered dual volumes."""
function _phase_face_average!(destination::FaceField{Axis}, source::CellField) where {Axis}
    destination.mesh === source.mesh || throw(ArgumentError("fields must share a mesh"))
    mesh = source.mesh
    index = 0
    for face in mesh.faces
        face.axis == Axis || continue
        index += 1
        hm = mesh.blocks[face.minus.block].spacing[Axis]
        hp = mesh.blocks[face.plus.block].spacing[Axis]
        destination.interior[index] = (hm * _get(source, face.minus) + hp * _get(source, face.plus)) / (hm + hp)
    end
    index = 0
    for face in mesh.boundary_faces
        face.axis == Axis || continue
        index += 1
        destination.boundary[index] = _get(source, face.cell)
    end
    return destination
end

"""
    PhaseFaceScalar{Axis}(interface; liquid=1, gas=1)

Intrinsic scalar values on staggered dual volumes, one value per phase. Pass
through the `staggered_scalars` keyword of geometric `advect!`. Conservative
phase-weighted products are formed only inside the update. Dual phase fluxes
are the algebraic interpolant of the centred WY volume fluxes.
"""
mutable struct PhaseFaceScalar{Axis, T, M <: BlockTreeMesh}
    interface::InterfaceField{T, M}
    liquid::FaceField{Axis, T, M}
    gas::FaceField{Axis, T, M}
    fraction::FaceField{Axis, T, M}
    workspace::StaggeredAdvectionWorkspace{Axis, T, M}
end

"""Scratch for compatible advection of one stored velocity field.

All arrays contain velocity, phase fraction, or a velocity increment. Momentum
is formed only in local expressions and is never a persistent field.
"""
struct MomentumComponentWorkspace{Axis, T, M <: BlockTreeMesh, W}
    value::FaceField{Axis, T, M}
    source::FaceField{Axis, T, M}
    fraction::FaceField{Axis, T, M}
    predicted_fraction::FaceField{Axis, T, M}
    increment::FaceField{Axis, T, M}
    advection::W
    old_fraction::Vector{T}
    new_fraction::Vector{T}
    values::Vector{T}
    sources::Vector{T}
    delta::Vector{T}
    rates::Vector{T}
end

@inline function _pack_phase_faces!(destination, source::FaceField)
    interior_count = length(source.interior)
    copyto!(destination, 1, source.interior, 1, interior_count)
    copyto!(
        destination, interior_count + 1,
        source.boundary, 1, length(source.boundary)
    )
    return destination
end

@inline function _unpack_phase_faces!(destination::FaceField, source)
    interior_count = length(destination.interior)
    copyto!(
        destination.interior, 1, source, 1, interior_count
    )
    copyto!(
        destination.boundary, 1, source, interior_count + 1,
        length(destination.boundary)
    )
    return destination
end

struct MomentumAdvectionWorkspace{T, M <: BlockTreeMesh, WX, WY}
    interface::InterfaceField{T, M}
    x::MomentumComponentWorkspace{1, T, M, WX}
    y::MomentumComponentWorkspace{2, T, M, WY}
end

function _momentum_component_workspace(
        interface::InterfaceField{T, M},
        ::Val{Axis}
    ) where {T, M, Axis}
    mesh = interface.mesh
    value = FaceField{Axis}(mesh, T)
    advection = StaggeredAdvectionWorkspace(value)
    if !all(mesh.periodic)
        advection = StaggeredAdvectionWorkspace(
            advection.tendency, advection.rates,
            _bounded_phase_schedule(mesh, Val(Axis), T), Ref(true)
        )
    end
    face_count = length(value.interior) + length(value.boundary)
    return MomentumComponentWorkspace(
        value, FaceField{Axis}(mesh, T),
        FaceField{Axis}(mesh, T), FaceField{Axis}(mesh, T), FaceField{Axis}(mesh, T),
        advection, ntuple(_ -> zeros(T, face_count), 6)...
    )
end

function MomentumAdvectionWorkspace(interface::InterfaceField{T, M}) where {T, M}
    x = _momentum_component_workspace(interface, Val(1))
    y = _momentum_component_workspace(interface, Val(2))
    return MomentumAdvectionWorkspace{T, M, typeof(x.advection), typeof(y.advection)}(
        interface, x, y
    )
end

function _resize_zero_face_field!(field::FaceField{Axis, T}) where {Axis, T}
    resize!(field.interior, _nfaces(field.mesh, Axis))
    resize!(field.boundary, _nboundary_faces(field.mesh, Axis))
    fill!(field.interior, zero(T))
    fill!(field.boundary, zero(T))
    return field
end

function _reconfigure_momentum_component!(
        component::MomentumComponentWorkspace{Axis, T},
        interface::InterfaceField{T},
    ) where {Axis, T}
    mesh = interface.mesh
    for field in (
            component.value, component.source, component.fraction,
            component.predicted_fraction, component.increment,
            component.advection.tendency, component.advection.rates,
        )
        _resize_zero_face_field!(field)
    end
    schedule = all(mesh.periodic) ?
        _dual_advection_schedule(mesh, Val(Axis), T, true) :
        _bounded_phase_schedule(mesh, Val(Axis), T)
    advection = StaggeredAdvectionWorkspace(
        component.advection.tendency, component.advection.rates,
        schedule, component.advection.sweep_forward,
    )
    face_count = length(component.value.interior) +
        length(component.value.boundary)
    for values in (
            component.old_fraction, component.new_fraction,
            component.values, component.sources, component.delta,
            component.rates,
        )
        resize!(values, face_count)
        fill!(values, zero(T))
    end
    return MomentumComponentWorkspace(
        component.value, component.source, component.fraction,
        component.predicted_fraction, component.increment, advection,
        component.old_fraction, component.new_fraction,
        component.values, component.sources, component.delta, component.rates,
    )
end

"""Rebuild only momentum connectivity and retain all field/vector capacity."""
function _reconfigure_momentum_workspace!(
        workspace::MomentumAdvectionWorkspace{T},
        interface::InterfaceField{T, M},
    ) where {T, M}
    workspace.interface === interface || return MomentumAdvectionWorkspace(interface)
    x = _reconfigure_momentum_component!(workspace.x, interface)
    y = _reconfigure_momentum_component!(workspace.y, interface)
    return MomentumAdvectionWorkspace{T, M, typeof(x.advection), typeof(y.advection)}(
        interface, x, y,
    )
end

function _prepare_momentum_advection!(
        work::MomentumAdvectionWorkspace,
        velocity::FaceVelocity
    )
    copyto!(work.x.value, velocity);copyto!(work.y.value, velocity)
    copyto!(work.x.source, velocity);copyto!(work.y.source, velocity)
    return work
end

function _advect_momentum_component!(
        component::MomentumComponentWorkspace,
        work::WYAdvectionWorkspace, field::InterfaceField,
        axis, density, ctu_threshold, scheme,
        amr_interface_scheme
    )
    q = component.value
    mesh = q.mesh
    _phase_face_average!(component.fraction, field.fraction)
    @_parallel for cell in 1:length(work.cells)
        ref = work.cells[cell]
        _set!(
            work.incoming, clamp(
                _get(field.fraction, ref) +
                    _get(work.increment, ref), zero(eltype(q)), one(eltype(q))
            ), ref
        )
    end
    _phase_face_average!(component.predicted_fraction, work.incoming)
    oldc = _pack_phase_faces!(component.old_fraction, component.fraction)
    newc = _pack_phase_faces!(
        component.new_fraction, component.predicted_fraction
    )
    values = _pack_phase_faces!(component.values, q)
    sources = _pack_phase_faces!(component.sources, component.source)
    delta = component.delta
    schedule = component.advection.schedule
    @_parallel for output in 1:length(values)
        value = zero(eltype(delta))
        for (rho, flux, liquid) in (
                (density[1], work.liquid_flux, true),
                (density[2], work.gas_flux, false),
            )
            for pointer in component.advection.dual_offsets[output]:(
                    component.advection.dual_offsets[output + 1] - 1
                )
                face_index = @inbounds component.advection.dual_faces[pointer]
                face = schedule.faces[face_index]
                face.axis == axis || continue
                m, p = face.minus, face.plus
                cm, cp = liquid ? (oldc[m], oldc[p]) :
                    (1 - oldc[m], 1 - oldc[p])
                nm, np = liquid ? (newc[m], newc[p]) :
                    (1 - newc[m], 1 - newc[p])
                _, minus_difference, plus_difference = _phase_transport(
                    face, schedule, flux, mesh, values,
                    cm, cp, nm, np, ctu_threshold,
                    scheme, amr_interface_scheme,
                )
                sign = @inbounds component.advection.dual_signs[pointer]
                value += rho * sign * (
                    sign > 0 ? minus_difference : plus_difference
                ) / _phase_dual_volume(schedule, output)
            end
        end
        @inbounds delta[output] = value
    end
    # WY's directional dilation term, with the phase density applied before
    # the centred-to-staggered interpolation as required by compatibility.
    @_parallel for cell in 1:length(work.cells)
        ref = work.cells[cell]
        chi = _get(work.indicator, ref)
        rhochi = density[1] * chi + density[2] * (1 - chi)
        _set!(work.incoming, rhochi * _get(work.outgoing, ref), ref)
    end
    _phase_face_average!(component.advection.rates, work.incoming)
    rates = _pack_phase_faces!(component.rates, component.advection.rates)
    @_parallel for i in 1:length(values)
        rhonew = density[1] * newc[i] + density[2] * (1 - newc[i])
        values[i] += (delta[i] + rates[i] * (sources[i] - values[i])) / rhonew
    end
    _unpack_phase_faces!(q, values)
    return component
end

function PhaseFaceScalar{Axis}(interface::InterfaceField{T, M}; liquid = 1, gas = 1) where {Axis, T, M}
    mesh = interface.mesh
    c = _phase_face_average!(FaceField{Axis}(mesh, T), interface.fraction)
    l, g = FaceField{Axis}(mesh, T), FaceField{Axis}(mesh, T)
    index = 0
    for face in mesh.faces
        face.axis == Axis || continue
        index += 1
        x, y = face.center
        lv = liquid isa Number ? liquid : liquid(x, y)
        gv = gas isa Number ? gas : gas(x, y)
        isfinite(lv) && isfinite(gv) || throw(ArgumentError("phase scalar values must be finite"))
        l.interior[index] = lv
        g.interior[index] = gv
    end
    index = 0
    for face in mesh.boundary_faces
        face.axis == Axis || continue
        index += 1
        x, y = face.center
        lv = liquid isa Number ? liquid : liquid(x, y)
        gv = gas isa Number ? gas : gas(x, y)
        l.boundary[index] = lv
        g.boundary[index] = gv
    end
    workspace = StaggeredAdvectionWorkspace(l)
    if !all(mesh.periodic)
        schedule = _bounded_phase_schedule(mesh, Val(Axis), T)
        workspace = StaggeredAdvectionWorkspace(workspace.tendency, workspace.rates, schedule, Ref(true))
    end
    return PhaseFaceScalar{Axis, T, M}(interface, l, g, c, workspace)
end

@inline _phase_dual_flux(face, schedule::UniformDualAdvectionSchedule, flux, mesh) =
    (
    flux[face.flux_a] / mesh.faces[face.flux_a].area +
        flux[face.flux_b] / mesh.faces[face.flux_b].area
) * face.area / 2
@inline function _phase_dual_flux(face, schedule::AMRDualAdvectionSchedule, flux, mesh)
    result = zero(eltype(flux))
    for k in face.first_weight:(face.first_weight + face.weight_count - 1)
        result += schedule.flux_weights[k] * flux[schedule.flux_indices[k]]
    end
    return result
end
@inline _phase_dual_volume(schedule::UniformDualAdvectionSchedule, i) = schedule.volume
@inline _phase_dual_volume(schedule::AMRDualAdvectionSchedule, i) = schedule.volumes[i]

@inline function _ctu_phase_transport(
        face, schedule::UniformDualAdvectionSchedule,
        flux, mesh, values
    )
    a = face.area * flux[face.flux_a] / (2mesh.faces[face.flux_a].area)
    b = face.area * flux[face.flux_b] / (2mesh.faces[face.flux_b].area)
    m, p = face.minus, face.plus
    qa = a >= 0 ? values[m] : values[p]
    qb = b >= 0 ? values[m] : values[p]
    return a + b,
        a * (values[m] - qa) + b * (values[m] - qb),
        a * (values[p] - qa) + b * (values[p] - qb)
end
@inline function _ctu_phase_transport(
        face, schedule::AMRDualAdvectionSchedule,
        flux, mesh, values
    )
    volume = minus_difference = plus_difference = zero(eltype(flux))
    m, p = face.minus, face.plus
    for k in face.first_weight:(face.first_weight + face.weight_count - 1)
        partial = schedule.flux_weights[k] * flux[schedule.flux_indices[k]]
        volume += partial
        upwind = partial >= 0 ? values[m] : values[p]
        minus_difference += partial * (values[m] - upwind)
        plus_difference += partial * (values[p] - upwind)
    end
    return volume, minus_difference, plus_difference
end

@inline _phase_connection_scheme(face::DualFace, scheme, amr_interface_scheme) =
    scheme
@inline _phase_connection_scheme(
    face::AMRDualFace, scheme,
    amr_interface_scheme
) =
    face.refinement_interface ? amr_interface_scheme : scheme

@inline function _phase_bulk_state(face, schedule, values, volume, vm, vp, ::Upwind)
    return volume >= 0 ? values[face.minus] : values[face.plus]
end
@inline function _phase_bulk_state(face, schedule, values, volume, vm, vp, ::Centred)
    return (values[face.minus] + values[face.plus]) / 2
end
@inline function _phase_bulk_state(
        face, schedule, values, volume, vm, vp,
        ::LaxWendroff
    )
    upwind, downwind = volume >= 0 ?
        (values[face.minus], values[face.plus]) :
        (values[face.plus], values[face.minus])
    courant = abs(volume) / min(vm, vp)
    return upwind + (1 - courant) * (downwind - upwind) / 2
end
@inline function _phase_bulk_state(
        face, schedule, values, volume, vm, vp,
        scheme::AbstractThreePointFluxScheme
    )
    positive = volume >= 0
    upwind, downwind = positive ?
        (values[face.minus], values[face.plus]) :
        (values[face.plus], values[face.minus])
    stencil = positive ? face.minus_stencil : face.plus_stencil
    iszero(stencil.upstream) && return upwind
    high_order = _phase_bulk_state(
        face, schedule, values, volume, vm, vp,
        LaxWendroff()
    )
    upstream = @inbounds values[stencil.upstream]
    return _three_point_state(
        upwind, downwind, upstream, stencil.ratio_scale,
        high_order, scheme
    )
end

@inline function _phase_transport(
        face, schedule, flux, mesh, values,
        cm, cp, nm, np, ctu_threshold, scheme,
        amr_interface_scheme
    )
    if min(nm, np) < ctu_threshold
        # For a split method, term-wise upwinding of the partial interpolated
        # volume fluxes is the CTU flux of thesis §pt1:momentum:flux_interp_ctu.
        # Keeping the partial terms separate cancels fluid in transit.
        return _ctu_phase_transport(face, schedule, flux, mesh, values)
    end
    volume = _phase_dual_flux(face, schedule, flux, mesh)
    vm = _phase_dual_volume(schedule, face.minus)
    vp = _phase_dual_volume(schedule, face.plus)
    selected = _phase_connection_scheme(face, scheme, amr_interface_scheme)
    state = _phase_bulk_state(face, schedule, values, volume, vm, vp, selected)
    return volume, volume * (values[face.minus] - state),
        volume * (values[face.plus] - state)
end

function _phase_face_sources(q)
    _phase_face_average!(q.fraction, q.interface.fraction)
    return (
        vcat(q.liquid.interior, q.liquid.boundary),
        vcat(q.gas.interior, q.gas.boundary),
    )
end
function _advect_phase_faces!(
        q, work, axis, sources, fraction_floor, ctu_threshold,
        scheme, amr_interface_scheme
    )
    mesh = work.mesh
    _phase_face_average!(q.fraction, q.interface.fraction)
    schedule = q.workspace.schedule
    fractions = vcat(q.fraction.interior, q.fraction.boundary)
    @_parallel for cell in 1:length(work.cells)
        ref = work.cells[cell]
        _set!(work.incoming, clamp(_get(q.interface.fraction, ref) + _get(work.increment, ref), 0, 1), ref)
    end
    _phase_face_average!(q.workspace.rates, work.incoming)
    predicted = vcat(q.workspace.rates.interior, q.workspace.rates.boundary)
    for (field, flux, source, liquid) in (
            (q.liquid, work.liquid_flux, sources[1], true),
            (q.gas, work.gas_flux, sources[2], false),
        )
        values = vcat(field.interior, field.boundary)
        inc = zeros(eltype(values), length(values))
        @_parallel for output in 1:length(values)
            value = zero(eltype(values))
            for pointer in q.workspace.dual_offsets[output]:(
                    q.workspace.dual_offsets[output + 1] - 1
                )
                face_index = @inbounds q.workspace.dual_faces[pointer]
                face = schedule.faces[face_index]
                face.axis == axis || continue
                m, p = face.minus, face.plus
                cm, cp = liquid ? (fractions[m], fractions[p]) :
                    (1 - fractions[m], 1 - fractions[p])
                nm, np = liquid ? (predicted[m], predicted[p]) :
                    (1 - predicted[m], 1 - predicted[p])
                _, minus_difference, plus_difference = _phase_transport(
                    face, schedule, flux, mesh, values,
                    cm, cp, nm, np, ctu_threshold,
                    scheme, amr_interface_scheme,
                )
                sign = @inbounds q.workspace.dual_signs[pointer]
                value += sign * (
                    sign > 0 ? minus_difference : plus_difference
                ) / _phase_dual_volume(schedule, output)
            end
            @inbounds inc[output] = value
        end
        # Interpolate the product chi*div_k(u), not its two factors separately.
        @_parallel for cell in 1:length(work.cells)
            ref = work.cells[cell]
            chi = _get(work.indicator, ref)
            _set!(work.incoming, (liquid ? chi : 1 - chi) * _get(work.outgoing, ref), ref)
        end
        _phase_face_average!(q.workspace.rates, work.incoming)
        rates = vcat(q.workspace.rates.interior, q.workspace.rates.boundary)
        @_parallel for i in 1:length(inc)
            intrinsic = values[i]
            target = liquid ? predicted[i] : 1 - predicted[i]
            values[i] = intrinsic +
                (inc[i] + (source[i] - intrinsic) * rates[i]) /
                max(target, fraction_floor)
        end
        n = length(field.interior)
        field.interior .= @view(values[1:n])
        field.boundary .= @view(values[(n + 1):end])
    end
    return q
end

# Complete the dual transport partition with physical boundary half-volumes.
# Primary boundary flux is zero, but an interior primary flux also crosses the
# boundary half-volume's *interior* side. Omitting those connections breaks the
# phase one-function identity near a wall.
function _bounded_phase_schedule(mesh, ::Val{Axis}, ::Type{T}) where {Axis, T}
    incidences, global_to_local, volumes = _component_incidences(mesh, Val(Axis), T)
    nx, ny = mesh.block_size
    local_index = length(volumes)
    for (boundary_index, face) in pairs(mesh.boundary_faces)
        face.axis == Axis || continue
        local_index += 1
        block = mesh.blocks[face.cell.block]
        fraction = face.area / block.spacing[3 - Axis]
        push!(volumes, T(face.area * block.spacing[Axis] / 2))
        push!(
            incidences[_cell_index(face.cell, nx, ny)], ComponentFaceIncidence(
                local_index, length(mesh.faces) + boundary_index, face.side, T(fraction)
            )
        )
    end
    coefficients = Dict{NTuple{3, Int}, T}()
    for (index, face) in pairs(mesh.faces)
        if face.axis == Axis
            centre = global_to_local[index]
            for (cell, sign) in ((face.minus, one(T)), (face.plus, -one(T)))
                for other in incidences[_cell_index(cell, nx, ny)]
                    other.global_index == index && continue
                    _add_amr_flux_weight!(coefficients, other.local_index, centre, index, sign * other.fraction / 2)
                end
            end
        else
            minus = incidences[_cell_index(face.minus, nx, ny)]
            plus = incidences[_cell_index(face.plus, nx, ny)]
            for side in (-1, 1),a in minus,b in plus
                a.side == side == b.side || continue
                _add_amr_flux_weight!(coefficients, a.local_index, b.local_index, index, a.fraction * b.fraction / 2)
            end
        end
    end
    keys_order = sort!(collect(keys(coefficients)); by = k -> (k[1], k[2], mesh.faces[k[3]].axis, k[3]))
    faces = AMRDualFace{T}[];indices = Int[];weights = T[]
    cursor = 1
    while cursor <= length(keys_order)
        first, second, flux = keys_order[cursor]
        axis = mesh.faces[flux].axis;start = length(indices) + 1
        while cursor <= length(keys_order)
            key = keys_order[cursor]
            key[1] == first && key[2] == second && mesh.faces[key[3]].axis == axis || break
            push!(indices, key[3]);push!(weights, coefficients[key]);cursor += 1
        end
        stencil = ThreePointStencil(0, zero(T))
        push!(faces, AMRDualFace(first, second, start, length(indices) - start + 1, axis, true, stencil, stencil))
    end
    return AMRDualAdvectionSchedule(faces, indices, weights, volumes)
end

"""Register intrinsic staggered phase scalars for accurate face transfer."""
function register_adaptive_field!(workspace::AMRWorkspace, q::PhaseFaceScalar)
    register_adaptive_field!(workspace, q.interface)
    index = _find_field_buffer(workspace, q.interface.fraction)
    strategy = workspace.field_buffers[index].prolongation
    any(s -> s === q, strategy.staggered) && return workspace
    _phase_face_average!(q.fraction, q.interface.fraction)
    push!(strategy.staggered, q)
    # Intrinsic values use limited linear face interpolation. A global face
    # integral correction is inappropriate because this is not a stored mass.
    for field in (q.liquid, q.gas)
        register_adaptive_field!(
            workspace, field;
            piecewise_constant = false, conserve_integral = false
        )
    end
    return workspace
end
function _finish_phase_face_transfer!(q::PhaseFaceScalar{Axis, T}) where {Axis, T}
    mesh = q.interface.mesh
    _resize_face_field!(q.fraction, mesh)
    current = _phase_face_average!(FaceField{Axis}(mesh, T), q.interface.fraction)
    copyto!(q.fraction.interior, current.interior)
    copyto!(q.fraction.boundary, current.boundary)
    work = StaggeredAdvectionWorkspace(q.liquid)
    if !all(mesh.periodic)
        work = StaggeredAdvectionWorkspace(
            work.tendency, work.rates,
            _bounded_phase_schedule(mesh, Val(Axis), T), Ref(true)
        )
    end
    q.workspace = work
    return q
end
