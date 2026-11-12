"""
    _overdrain_suspects(workspace)

Unknowns whose phase update can drain more phase than their staggered content
holds. A level transition weighs a partial flux by the strip of the receiving
unknown while the cut-donor bound limits it by the strip of the donating cell,
so only those unknowns need the outflow limiter of `_phase_drain_limit!`; on a
conforming patch the cell-wise outflow bound keeps the own-value coefficient
non-negative.
"""
function _overdrain_suspects(
        workspace::StaggeredAdvectionWorkspace{Axis, T, M, S},
    ) where {Axis, T, M, S}
    return falses(length(workspace.dual_offsets) - 1)
end
function _overdrain_suspects(
        workspace::StaggeredAdvectionWorkspace{Axis, T, M, S},
    ) where {Axis, T, M, S <: AMRDualAdvectionSchedule}
    schedule = workspace.schedule
    suspect = falses(length(workspace.dual_offsets) - 1)
    for output in eachindex(suspect)
        for pointer in workspace.dual_offsets[output]:(
                workspace.dual_offsets[output + 1] - 1
            )
            if schedule.faces[workspace.dual_faces[pointer]].refinement_interface
                suspect[output] = true
                break
            end
        end
    end
    return suspect
end

function _phase_advection_workspace(
        tendency::FaceField{Axis, T}, rates::FaceField{Axis, T}, geometry::G,
        forward; reuse_flux = nothing,
    ) where {Axis, T, G}
    if isnothing(geometry)
        # The cache lookup/geometry dispatch happens once per construction,
        # never through Any in the numerical transport loop.
        graph = _phase_transport_connectivity(tendency.mesh, Val(Axis), T)
        return StaggeredAdvectionWorkspace(tendency, rates, graph, forward; reuse_flux)
    end
    # Open-volume connectivity depends on the geometry, not only the mesh.
    # Rebuild this part; never alias it to the uncut mesh-only cache.
    return StaggeredAdvectionWorkspace(
        tendency, rates,
        _open_strip_phase_schedule(
            _geometry_phase_schedule(geometry, Val(Axis)), geometry, Val(Axis),
        ),
        forward
    )
end

# The open staggered volume interpolates each cell with half of the open strip
# behind the face (eq:momentum:volumes, eq:momentum:interpolation). At a split
# coarse side that strip is generally not the same fraction of the open cell as
# the geometric fraction of the side, so the phase-flux factorization must use
# the open-volume fractions for its divergence to commute with the same
# interpolation that forms the staggered fraction (eq:momentum:mass-identity,
# condition (b)). Every phase schedule -- one-velocity momentum, two-velocity
# momentum and the staggered phase scalars -- shares this factorization.
function _open_strip_phase_schedule(
        schedule, geometry::GeometryField{T}, ::Val{Axis},
    ) where {T, Axis}
    mesh = geometry.mesh
    # Uniform cut cells and uncut hanging cells already use the correct
    # geometric weights. Avoid rebuilding incidences for that common case.
    needs_correction = false
    for face in mesh.faces
        face.axis == Axis || continue
        for ref in (face.minus, face.plus)
            side_fraction = face.area / mesh.blocks[ref.block].spacing[3 - Axis]
            open = _get(geometry.open_fraction, ref)
            if side_fraction < one(T) && zero(T) < open < one(T)
                needs_correction = true
                break
            end
        end
        needs_correction && break
    end
    needs_correction || return schedule

    incidences, global_to_local, _ = _component_incidences(mesh, Val(Axis), T)
    nx, ny = mesh.block_size
    local_index = length(geometry.dual_volumes[Axis])
    for (boundary_index, face) in pairs(mesh.boundary_faces)
        face.axis == Axis || continue
        local_index += 1
        fraction = T(face.area / mesh.blocks[face.cell.block].spacing[3 - Axis])
        push!(
            incidences[_cell_index(face.cell, nx, ny)],
            ComponentFaceIncidence(
                local_index, length(mesh.faces) + boundary_index, face.side, fraction,
            ),
        )
    end

    changed = false
    for (cell_index, cell_faces) in pairs(incidences)
        block = (cell_index - 1) ÷ (nx * ny) + 1
        local_cell = (cell_index - 1) % (nx * ny)
        ref = CellRef(block, local_cell % nx + 1, local_cell ÷ nx + 1)
        open = _get(geometry.open_fraction, ref)
        zero(T) < open < one(T) || continue
        for index in eachindex(cell_faces)
            incidence = cell_faces[index]
            incidence.fraction < one(T) || continue
            face = incidence.global_index <= length(mesh.faces) ?
                mesh.faces[incidence.global_index] :
                mesh.boundary_faces[incidence.global_index - length(mesh.faces)]
            lo, hi = _face_local_tangent_interval(mesh, ref, face)
            bounds = Axis == 1 ? (-0.5, 0.5, lo, hi) : (lo, hi, -0.5, 0.5)
            fraction = T(_open_subcell_area(geometry.solid, ref, bounds...) / open)
            changed |= fraction != incidence.fraction
            cell_faces[index] = ComponentFaceIncidence(
                incidence.local_index, incidence.global_index, incidence.side, fraction,
            )
        end
    end

    changed || return schedule
    # Match `_bounded_phase_schedule`'s normal stars and transverse product
    # plan, changing only the cell-to-face partition fractions.
    weights = _amr_flux_weights(
        mesh, Val(Axis), T, incidences, global_to_local, local_index,
    )
    for face in schedule.faces
        for pointer in face.first_weight:(face.first_weight + face.weight_count - 1)
            schedule.flux_weights[pointer] = _amr_flux_weight(
                weights, face.minus, face.plus, schedule.flux_indices[pointer],
            )
        end
    end
    return schedule
end

function _phase_advection_workspace(field::FaceField{Axis, T}, geometry) where {Axis, T}
    return _phase_advection_workspace(
        FaceField{Axis}(field.mesh, T),
        FaceField{Axis}(field.mesh, T), geometry, Ref(true)
    )
end

function PhaseFaceScalar{Axis}(interface::InterfaceField{T, M}; liquid = 1, gas = 1) where {Axis, T, M}
    mesh = interface.mesh
    c = _phase_face_average!(
        FaceField{Axis}(mesh, T), interface.conditional_fraction,
        interface.geometry,
    )
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
    workspace = _phase_advection_workspace(l, interface.geometry)
    count = length(l.interior) + length(l.boundary)
    scratch = PhaseFaceAdvectionScratch(T, count, workspace)
    return PhaseFaceScalar{
        Axis, T, M, typeof(scratch), typeof(workspace), typeof(interface),
    }(
        interface, l, g, c, workspace, scratch,
    )
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

"""
    integrate(quantity::PhaseFaceScalar, phase)

Return the phase-volume-weighted integral of an intrinsic staggered scalar.
`phase` is `:liquid` or `:gas`. The phase amount is formed only for this
reduction; `PhaseFaceScalar` continues to store intrinsic values.
"""
function integrate(quantity::PhaseFaceScalar, phase::Symbol)
    liquid = if phase === :liquid
        true
    elseif phase === :gas
        false
    else
        throw(ArgumentError("phase must be :liquid or :gas"))
    end
    _phase_face_average!(
        quantity.fraction, quantity.interface.conditional_fraction,
        quantity.interface.geometry,
    )
    field = liquid ? quantity.liquid : quantity.gas
    values = Iterators.flatten((field.interior, field.boundary))
    fractions = Iterators.flatten(
        (
            quantity.fraction.interior, quantity.fraction.boundary,
        )
    )
    result = zero(eltype(field))
    for (index, (value, fraction)) in enumerate(zip(values, fractions))
        weight = liquid ? fraction : one(fraction) - fraction
        result += _phase_dual_volume(quantity.workspace.schedule, index) *
            weight * value
    end
    return result
end

# The fluctuation form of an unknown is a convex combination of its old value
# and the values that enter it exactly when the phase volume leaving it,
# O = sum of its positive partial volumes, is at most the phase content
# V_f * alpha_f it holds at the start of the sweep: the own-value coefficient
# is (V alpha - O)/(V alpha_new) (eq:momentum:own-coefficient). On a conforming
# patch the cell-wise outflow bound gives O <= V alpha. At a hanging segment the
# factorization weighs a partial flux by the strip of the receiving unknown and
# bounds it by the strip of the donating cell, so a draining control volume can
# lose more phase than its interpolated content holds, for every state
# (including the net-volume upwind one): a phase scalar 0 among neighbours 1
# became 1.98 there (issue #70).
#
# The excess O - V alpha of such an unknown leaves at the mean of the values
# that enter it instead of at its own value. Its update stays the conservative
# one -- every partial volume is valued once, for its sender and its receiver
# alike -- and becomes a convex combination, because the coefficients of the
# old value, (V alpha - O(1 - excess)), and of the entering values, I - O
# excess, are then both non-negative and sum to V alpha_new.
@inline function _phase_fraction(values, index, liquid::Bool)
    return liquid ? @inbounds(values[index]) :
        one(eltype(values)) - @inbounds(values[index])
end

@inline function _phase_limited_value(values, excess, flush, index)
    value = @inbounds values[index]
    weight = @inbounds excess[index]
    return weight > zero(weight) ?
        value + weight * (@inbounds(flush[index]) - value) : value
end

# Excess outflow of every unknown of one phase, and the mean value of what
# enters it, for the values of this sweep. Convergent patches are skipped: the
# cell-wise outflow bound keeps their own-value coefficient non-negative.
function _phase_drain_limit!(
        excess, flush, suspect, workspace, schedule, flux, values,
        old_fraction, liquid::Bool, axis,
    )
    T = eltype(values)
    fill!(excess, zero(T))
    @_parallel for output in 1:length(values)
        suspect[output] || continue
        outflow = inflow = entering = zero(T)
        for pointer in workspace.dual_offsets[output]:(
                workspace.dual_offsets[output + 1] - 1
            )
            face_index = @inbounds workspace.dual_faces[pointer]
            face = schedule.faces[face_index]
            face.axis == axis || continue
            minus_side = @inbounds workspace.dual_signs[pointer] > 0
            for k in face.first_weight:(face.first_weight + face.weight_count - 1)
                partial = @inbounds schedule.flux_weights[k] *
                    flux[schedule.flux_indices[k]]
                leaving = minus_side ? max(partial, zero(T)) : max(-partial, zero(T))
                inflow_part = minus_side ? max(-partial, zero(T)) : max(partial, zero(T))
                outflow += leaving
                inflow += inflow_part
                entering += inflow_part *
                    (
                    minus_side ? @inbounds(values[face.plus]) :
                        @inbounds(values[face.minus])
                )
            end
        end
        content = _phase_dual_volume(schedule, output) *
            _phase_fraction(old_fraction, output, liquid)
        # A relative tolerance: a vanishing content with a round-off outflow
        # is not a drained unknown.
        drained = outflow - content
        drained > T(128) * eps(T) * outflow || continue
        # Without an inflow there is nothing to blend towards: keep the own
        # value, never a value left over from an earlier sweep or phase.
        @inbounds flush[output] = values[output]
        inflow > zero(T) && (@inbounds flush[output] = entering / inflow)
        @inbounds excess[output] = drained / outflow
    end
    return excess
end

@inline function _ctu_phase_transport(
        face, schedule::UniformDualAdvectionSchedule,
        flux, mesh, values, excess, flush,
    )
    a = face.area * flux[face.flux_a] / (2mesh.faces[face.flux_a].area)
    b = face.area * flux[face.flux_b] / (2mesh.faces[face.flux_b].area)
    m, p = face.minus, face.plus
    qm = _phase_limited_value(values, excess, flush, m)
    qp = _phase_limited_value(values, excess, flush, p)
    qa = a >= 0 ? qm : qp
    qb = b >= 0 ? qm : qp
    return a + b,
        a * (values[m] - qa) + b * (values[m] - qb),
        a * (values[p] - qa) + b * (values[p] - qb)
end
@inline function _ctu_phase_transport(
        face, schedule::AMRDualAdvectionSchedule,
        flux, mesh, values, excess, flush,
    )
    volume = minus_difference = plus_difference = zero(eltype(flux))
    m, p = face.minus, face.plus
    qm = _phase_limited_value(values, excess, flush, m)
    qp = _phase_limited_value(values, excess, flush, p)
    for k in face.first_weight:(face.first_weight + face.weight_count - 1)
        partial = schedule.flux_weights[k] * flux[schedule.flux_indices[k]]
        volume += partial
        upwind = partial >= 0 ? qm : qp
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
        amr_interface_scheme, excess, flush,
    )
    excess_minus = @inbounds excess[face.minus]
    excess_plus = @inbounds excess[face.plus]
    if min(nm, np) < ctu_threshold ||
            excess_minus > zero(excess_minus) || excess_plus > zero(excess_plus)
        # For a split method, term-wise upwinding of the partial interpolated
        # volume fluxes is the corner-transport-upwind flux of
        # eq:momentum:ctu-exchange: upwinding the net volume plus a
        # counter-flow exchange between the two unknowns of the dual face.
        # A limited unknown takes this flux in the bulk branch as well: the
        # bulk state moves the net volume of the face at one value, while
        # `_phase_drain_limit!` counts the partial volumes by sign, so the two
        # only agree on the outflow of a limited unknown when every partial is
        # upwinded.
        return _ctu_phase_transport(
            face, schedule, flux, mesh, values, excess, flush,
        )
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
    _phase_face_average!(
        q.fraction, q.interface.conditional_fraction, q.interface.geometry,
    )
    _pack_phase_faces!(q.scratch.source_liquid, q.liquid)
    _pack_phase_faces!(q.scratch.source_gas, q.gas)
    return (q.scratch.source_liquid, q.scratch.source_gas)
end
function _advect_phase_faces!(
        q, work, axis, sources, ctu_threshold,
        scheme, amr_interface_scheme,
    )
    # Resolve the two-way schedule union once per sweep. The numerical loops
    # below then specialize on the actual concrete workspace and face type.
    return _advect_phase_faces_workspace!(
        q.workspace, q, work, axis, sources, ctu_threshold,
        scheme, amr_interface_scheme,
    )
end

function _advect_phase_faces_workspace!(
        workspace::W, q, work, axis, sources, ctu_threshold,
        scheme, amr_interface_scheme,
    ) where {W <: StaggeredAdvectionWorkspace}
    mesh = work.mesh
    _phase_face_average!(
        q.fraction, q.interface.conditional_fraction, q.interface.geometry,
    )
    schedule = workspace.schedule
    liquid_fraction = _pack_phase_faces!(
        q.scratch.liquid_fraction, q.fraction,
    )
    nx, _ = mesh.block_size
    halo = mesh.halo
    T = eltype(q.liquid)

    # The predicted liquid carrier is the same one used by the cell update.
    _parallel_block_columns(mesh) do b, first_j, last_j
        fraction = @inbounds q.interface.conditional_fraction.blocks[b]
        increment = @inbounds work.increment.blocks[b]
        incoming = @inbounds work.incoming.blocks[b]
        @inbounds for j in first_j:last_j, i in 1:nx
            ii, jj = i + halo, j + halo
            incoming[ii, jj] = clamp(
                fraction[ii, jj] + increment[ii, jj], zero(T), one(T),
            )
        end
    end
    _phase_face_average!(workspace.rates, work.incoming, q.interface.geometry)
    predicted_liquid = _pack_phase_faces!(
        q.scratch.predicted_liquid, workspace.rates,
    )

    for (field, flux, source, liquid) in (
            (q.liquid, work.liquid_flux, sources[1], true),
            (q.gas, work.gas_flux, sources[2], false),
        )
        values = _pack_phase_faces!(q.scratch.values, field)
        inc = q.scratch.increment
        fill!(inc, zero(T))
        limiter = q.scratch.limiter
        excess, flush = limiter.excess, limiter.flush
        _phase_drain_limit!(
            excess, flush, limiter.suspect, workspace, schedule, flux,
            values, liquid_fraction, liquid, axis,
        )
        @_parallel for output in 1:length(values)
            value = zero(eltype(values))
            for pointer in workspace.dual_offsets[output]:(
                    workspace.dual_offsets[output + 1] - 1
                )
                face_index = @inbounds workspace.dual_faces[pointer]
                face = schedule.faces[face_index]
                face.axis == axis || continue
                m, p = face.minus, face.plus
                cm = liquid ? liquid_fraction[m] :
                    one(T) - liquid_fraction[m]
                cp = liquid ? liquid_fraction[p] :
                    one(T) - liquid_fraction[p]
                nm = liquid ? predicted_liquid[m] :
                    one(T) - predicted_liquid[m]
                np = liquid ? predicted_liquid[p] :
                    one(T) - predicted_liquid[p]
                _, minus_difference, plus_difference = _phase_transport(
                    face, schedule, flux, mesh, values,
                    cm, cp, nm, np, ctu_threshold,
                    scheme, amr_interface_scheme, excess, flush,
                )
                sign = @inbounds workspace.dual_signs[pointer]
                value += sign * (
                    sign > 0 ? minus_difference : plus_difference
                ) / _phase_dual_volume(schedule, output)
            end
            @inbounds inc[output] = value
        end
        # Interpolate the product chi*div_k(u), not its two factors separately.
        _parallel_block_columns(mesh) do b, first_j, last_j
            indicator = @inbounds work.indicator.blocks[b]
            outgoing = @inbounds work.outgoing.blocks[b]
            incoming = @inbounds work.incoming.blocks[b]
            @inbounds for j in first_j:last_j, i in 1:nx
                ii, jj = i + halo, j + halo
                chi = indicator[ii, jj]
                incoming[ii, jj] = liquid ?
                    chi * outgoing[ii, jj] :
                    (one(T) - chi) * outgoing[ii, jj]
            end
        end
        _phase_face_average!(workspace.rates, work.incoming, q.interface.geometry)
        rates = _pack_phase_faces!(q.scratch.rates, workspace.rates)
        @_parallel for i in 1:length(inc)
            intrinsic = values[i]
            target = liquid ? predicted_liquid[i] :
                one(T) - predicted_liquid[i]
            if target > q.interface.fraction_floor
                values[i] = intrinsic +
                    (inc[i] + (source[i] - intrinsic) * rates[i]) /
                    target
            end
        end
        _unpack_phase_faces!(field, values)
    end
    return q
end

# Complete the dual transport partition with physical boundary half-volumes.
# Primary boundary flux is zero, but an interior primary flux also crosses the
# boundary half-volume's *interior* side. Omitting those connections breaks the
# phase one-function identity near a wall.
function _bounded_phase_schedule(mesh, ::Val{Axis}, ::Type{T}) where {Axis, T}
    incidences, global_to_local, volumes = _component_incidences(mesh, Val(Axis), T)
    interior_count = length(volumes)
    nx, ny = mesh.block_size
    local_index = length(volumes)
    for (boundary_index, face) in pairs(mesh.boundary_faces)
        face.axis == Axis || continue
        local_index += 1
        block = mesh.blocks[face.cell.block]
        push!(volumes, T(face.area * block.spacing[Axis] / 2))
        push!(
            incidences[_cell_index(face.cell, nx, ny)], ComponentFaceIncidence(
                local_index, length(mesh.faces) + boundary_index, face.side, one(T)
            )
        )
    end
    weights = _amr_flux_weights(
        mesh, Val(Axis), T, incidences, global_to_local, length(volumes),
    )
    return _assemble_amr_dual_schedule(
        mesh, Val(Axis), T, weights, volumes; interior_count,
    )
end
