"""Integral of an intrinsic phase scalar over the part occupied by `phase`."""

function integrate(q::PhaseScalar, phase::Symbol)
    phase in (:liquid, :gas) || throw(ArgumentError("phase must be :liquid or :gas"))
    total = zero(eltype(q.liquid))
    for (b, block) in pairs(q.interface.mesh.blocks)
        liquid_fraction = interior(q.interface.fraction, b)
        values = interior(phase === :liquid ? q.liquid : q.gas, b)
        open = isnothing(q.interface.geometry) ? one.(liquid_fraction) :
            interior(q.interface.geometry.open_fraction, b)
        total += block.volume * sum(
            (phase === :liquid ? liquid_fraction : open .- liquid_fraction) .*
                values,
        )
    end
    return total
end

# Use a higher-order state only in the bulk. Every face incident to an old
# or newly mixed control volume uses phase upwinding, so a vanishing carrier
# never receives a high-order extrapolated amount. Recovery uses the
# interface's finite fraction floor, which defaults to `sqrt(eps(T))`.
@inline function _phase_flux_state(
        qm, qp, cm, cp, newcm, newcp, volume, vm, vp,
        scheme::AbstractStaggeredAdvectionScheme, upstream = qm,
        ratio_scale = one(volume)
    )
    upwind = volume >= 0 ? qm : qp
    iszero(volume) && return zero(upwind)
    tol = 64eps(typeof(cm))
    min(cm, cp, newcm, newcp) >= 1 - tol || return upwind
    downwind = volume >= 0 ? qp : qm
    scheme isa Upwind && return upwind
    scheme isa Centred && return (qm + qp) / 2
    courant = abs(volume) / min(vm, vp)
    high_order = upwind + (1 - courant) * (downwind - upwind) / 2
    scheme isa LaxWendroff && return high_order
    return _three_point_state(
        upwind, downwind, upstream, ratio_scale, high_order, scheme
    )
end

@inline _phase_upstream_value(
    values::CellField, ref::CellRef, axis, direction
) = @inbounds values.blocks[ref.block][
    ref.i + values.mesh.halo + (axis == 1 ? direction : 0),
    ref.j + values.mesh.halo + (axis == 2 ? direction : 0),
]

function _swept_fraction(
        field, face, donor, u, dt, normalized_open_volume = nothing,
    )
    geometry = field.geometry
    c = _conditional_fraction(field, donor)
    (c == 0 || c == 1) && return c
    mesh = field.mesh
    block = mesh.blocks[donor.block]
    axis = face.axis
    tlow, thigh = _face_local_tangent_interval(mesh, donor, face)
    distance = abs(u) * dt / block.spacing[axis]
    low, high = u >= 0 ?
        (0.5 - distance, 0.5) : (-0.5, -0.5 + distance)
    xlo, xhi, ylo, yhi = axis == 1 ? (low, high, tlow, thigh) : (tlow, thigh, low, high)
    isnothing(geometry) && return _plane_fraction(
        _get(field.normal_x, donor), _get(field.normal_y, donor),
        _get(field.intercept, donor), xlo, xhi, ylo, yhi
    )
    T = eltype(field)
    solid_fraction = _get(geometry.solid.fraction, donor)
    solid_fraction < one(T) || return zero(T)
    target = isnothing(normalized_open_volume) ?
        T(abs(u) * dt * face.area / block.volume) :
        T(normalized_open_volume)
    target > zero(T) || return zero(T)
    function swept_bounds(trial_distance)
        trial_low, trial_high = u >= 0 ?
            (T(0.5) - trial_distance, T(0.5)) :
            (-T(0.5), -T(0.5) + trial_distance)
        return axis == 1 ?
            (trial_low, trial_high, T(tlow), T(thigh)) :
            (T(tlow), T(thigh), trial_low, trial_high)
    end
    if solid_fraction <= zero(T)
        transverse_width = T(thigh - tlow)
        trial_distance = clamp(
            target / transverse_width, zero(T), one(T),
        )
        bounds = swept_bounds(trial_distance)
        return _plane_fraction(
            _get(field.normal_x, donor), _get(field.normal_y, donor),
            _get(field.intercept, donor), bounds...,
        )
    end
    # The flux is prescribed on the open face.  For a sloped cut cell, the
    # Cartesian distance |u|dt does not generally enclose that same open
    # volume.  Invert the clipped PLIC area so the swept region represents the
    # conservative volume flux exactly.  This is also what makes its liquid
    # and gas parts a bounded partition of the donor volume.
    entire_bounds = swept_bounds(one(T))
    maximum_area = _open_subcell_area(
        geometry.solid, donor, entire_bounds...,
    )
    tolerance = T(256) * eps(T) * max(one(T), maximum_area)
    target <= maximum_area + tolerance || error(
        "cut-cell face flux exceeds the donor's open subcell volume",
    )
    lo, hi = zero(T), one(T)
    for _ in 1:54
        middle = (lo + hi) / 2
        bounds = swept_bounds(middle)
        candidate_area = _open_subcell_area(
            geometry.solid, donor, bounds...,
        )
        if candidate_area < target
            lo = middle
        else
            hi = middle
        end
    end
    xlo, xhi, ylo, yhi = swept_bounds((lo + hi) / 2)
    rectangle = NTuple{2, T}[
        (xlo, ylo), (xhi, ylo), (xhi, yhi), (xlo, yhi),
    ]
    open_polygon = solid_fraction <= zero(T) ? rectangle :
        _clip_polygon(
            rectangle,
            _get(geometry.solid.normal_x, donor),
            _get(geometry.solid.normal_y, donor),
            _get(geometry.solid.intercept, donor); keep_less = false,
        )
    liquid_polygon = _clip_polygon(
        open_polygon,
        _get(field.normal_x, donor), _get(field.normal_y, donor),
        _get(field.intercept, donor); keep_less = true,
    )
    return clamp(_polygon_area(liquid_polygon) / target, zero(T), one(T))
end

"""Populate the carrier and phase swept volumes for one one-velocity sweep."""
function _compute_directional_wy_fluxes!(
        field::InterfaceField{T}, velocity::FaceVelocity, dt,
        work::WYAdvectionWorkspace{T}, axis,
    ) where {T}
    geometry = field.geometry
    mesh = field.mesh
    @_parallel for i in 1:length(mesh.faces)
        face = mesh.faces[i]
        face.axis == axis || continue
        u = velocity.interior[i]
        if iszero(u)
            @inbounds begin
                work.open_flux[i] = zero(T)
                work.liquid_flux[i] = zero(T)
                work.gas_flux[i] = zero(T)
            end
            continue
        end
        donor = u >= 0 ? face.minus : face.plus
        aperture = isnothing(geometry) ? one(T) : geometry.aperture[i]
        volume = dt * u * face.area * aperture
        normalized_open_volume = abs(volume) /
            mesh.blocks[donor.block].volume
        liquid = volume * _swept_fraction(
            field, face, donor, u, dt, normalized_open_volume,
        )
        gas = volume - liquid
        @inbounds begin
            work.open_flux[i] = volume
            work.liquid_flux[i] = liquid
            work.gas_flux[i] = gas
        end
    end
    return work
end

"""
    advect!(interface, velocity, dt, work::WYAdvectionWorkspace;
            scalars=(), sweep_order=(1,2))

Directionally split geometric Weymouth--Yue transport with a frozen `c > 1/2`
indicator and reconstruction between sweeps. Phase scalars use these exact
volume fluxes and upwind phase concentrations. Their dilation concentrations
are frozen for the whole step, so dilation cancels for solenoidal velocity.
Fractions within `interface.fraction_floor` of either pure phase are snapped to
that phase; larger bound violations raise errors.
"""
function _validate_phase_momentum!(args...)
    throw(
        ArgumentError(
            "momentum destination, source, and workspace must select the same " *
                "one- or two-velocity model",
        )
    )
end
@inline function _wy_fraction_in_bounds(c::T, open::Real) where {T}
    tolerance = T(128) * eps(T)
    if iszero(open) || open == one(T)
        return -tolerance <= c <= one(T) + tolerance
    end
    # PLIC clipping and the conservative swept-volume sum are formed in
    # units of the full cell. Test their round-off in those same units rather
    # than amplifying it by dividing by a possibly tiny open fraction.
    liquid = open * c
    return -tolerance <= liquid <= open + tolerance
end

function _advect_wy!(
        field::InterfaceField{T}, velocity, dt,
        work::WYAdvectionWorkspace{T}; scalars = (), staggered_scalars = (),
        momentum = nothing, density = nothing,
        momentum_workspace = nothing, momentum_source = velocity,
        sweep_order = (1, 2),
        scheme = LaxWendroff(),
        reset_diagnostics = true,
        check_timestep = true,
        initial_geometry_prepared = false,
        amr_interface_scheme =
            default_amr_interface_scheme(scheme)
    ) where {T}
    field === work.interface || throw(
        ArgumentError(
            "interface and workspace must be the same object",
        )
    )
    sweep_order in ((1, 2), (2, 1)) || throw(ArgumentError("sweep_order must be (1,2) or (2,1)"))
    isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be finite and positive"))
    if check_timestep
        limit = stable_timestep(velocity, work)
        dt <= limit * (1 + 1.0e-6 + 32eps(T)) || throw(
            ArgumentError(
                "time step exceeds the WY CFL limit",
            )
        )
    end
    all(q -> q.interface === field, scalars) || throw(ArgumentError("scalars must belong to this interface"))
    all(q -> q.interface === field, staggered_scalars) || throw(ArgumentError("staggered scalars must belong to this interface"))
    scheme = _canonical_staggered_scheme(scheme)
    amr_interface_scheme = _canonical_staggered_scheme(amr_interface_scheme)
    _validate_amr_interface_scheme(scheme, amr_interface_scheme)
    for k in eachindex(scalars), j in firstindex(scalars):(k - 1)
        scalars[j].liquid === scalars[k].liquid &&
            throw(ArgumentError("each scalar may only be advanced once"))
    end
    if !isnothing(momentum)
        density isa NTuple{2} && all(r -> isfinite(r)&&r > 0, density) ||
            throw(ArgumentError("momentum advection requires two positive phase densities"))
        _validate_phase_momentum!(
            momentum, momentum_source, momentum_workspace, field, T,
        )
    elseif !isnothing(density)||!isnothing(momentum_workspace)
        throw(ArgumentError("density and momentum_workspace require a momentum destination"))
    end
    # A frozen-interface caller may have reconstructed this same fraction
    # state while preparing the time-step-start apertures/CFL bound. The
    # geometry is then already the PLIC needed by the first directional sweep.
    initial_geometry_prepared || reconstruct!(field)
    geometry = field.geometry
    face_sources = map(_phase_face_sources, staggered_scalars)
    momentum_sources = isnothing(momentum) ? nothing :
        _phase_momentum_sources(momentum_workspace)
    mesh = field.mesh
    nx, ny = mesh.block_size
    halo = mesh.halo
    cells_per_block = nx * ny
    _parallel_block_columns(mesh) do b, first_j, last_j
        fraction = @inbounds field.conditional_fraction.blocks[b]
        indicator = @inbounds work.indicator.blocks[b]
        gas_carrier = @inbounds work.gas_carrier.blocks[b]
        @inbounds for j in first_j:last_j, i in 1:nx
            ii, jj = i + halo, j + halo
            indicator[ii, jj] = fraction[ii, jj] > 0.5
            gas_carrier[ii, jj] = one(T) - fraction[ii, jj]
        end
    end
    # Freeze dilation concentrations before the directional sweeps.
    sources = work.scalar_sources
    increments = work.scalar_increments
    while length(sources) < length(scalars)
        push!(sources, (CellField(mesh, T), CellField(mesh, T)))
        push!(increments, (CellField(mesh, T), CellField(mesh, T)))
    end
    for k in eachindex(scalars)
        q = scalars[k]
        l, g = sources[k]
        _parallel_block_columns(mesh) do b, first_j, last_j
            indicator = @inbounds work.indicator.blocks[b]
            liquid = @inbounds q.liquid.blocks[b]
            gas = @inbounds q.gas.blocks[b]
            liquid_source = @inbounds l.blocks[b]
            gas_source = @inbounds g.blocks[b]
            @inbounds for j in first_j:last_j, i in 1:nx
                ii, jj = i + halo, j + halo
                chi = indicator[ii, jj]
                liquid_source[ii, jj] = chi * liquid[ii, jj]
                gas_source[ii, jj] = (1 - chi) * gas[ii, jj]
            end
        end
    end
    reset_diagnostics && _reset!(work.diagnostics)
    fill!(work.open_flux, zero(T))
    fill!(work.liquid_flux, zero(T)); fill!(work.gas_flux, zero(T))
    for axis in sweep_order
        for q in scalars
            fill_halo!(q.liquid)
            fill_halo!(q.gas)
        end
        _compute_directional_wy_fluxes!(field, velocity, dt, work, axis)
        _parallel_block_columns(mesh) do b, first_j, last_j
            indicator = @inbounds work.indicator.blocks[b]
            outgoing_values = @inbounds work.outgoing.blocks[b]
            liquid_dilation = @inbounds work.liquid_dilation.blocks[b]
            gas_dilation = @inbounds work.gas_dilation.blocks[b]
            gas_increment_values = @inbounds work.gas_increment.blocks[b]
            increment_values = @inbounds work.increment.blocks[b]
            cellvolume = mesh.blocks[b].volume
            offset = (b - 1) * cells_per_block
            @inbounds for j in first_j:last_j, i in 1:nx
                cell = offset + (j - 1) * nx + i
                outgoing = zero(T)
                fraction_increment = zero(T)
                gas_fraction_increment = zero(T)
                for pointer in work.cell_face_offsets[cell]:(
                        work.cell_face_offsets[cell + 1] - 1
                    )
                    face_index = work.cell_faces[pointer]
                    face = mesh.faces[face_index]
                    face.axis == axis || continue
                    orientation = work.cell_face_orientations[pointer]
                    volume = work.open_flux[face_index]
                    liquid_flux = work.liquid_flux[face_index]
                    gas_flux = work.gas_flux[face_index]
                    outgoing += orientation * volume
                    fraction_increment += orientation * (
                        indicator[i + halo, j + halo] * volume - liquid_flux
                    )
                    gas_fraction_increment += orientation * (
                        (1 - indicator[i + halo, j + halo]) * volume - gas_flux
                    )
                end
                open = isnothing(geometry) ? one(T) :
                    geometry.open_fraction.blocks[b][i + halo, j + halo]
                effective_volume = cellvolume * open
                divergence = effective_volume > zero(T) ?
                    outgoing / effective_volume : zero(T)
                outgoing_values[i + halo, j + halo] = divergence
                liquid_dilation[i + halo, j + halo] =
                    indicator[i + halo, j + halo] * divergence
                gas_dilation[i + halo, j + halo] =
                    (1 - indicator[i + halo, j + halo]) * divergence
                increment_values[i + halo, j + halo] =
                    effective_volume > zero(T) ?
                    fraction_increment / effective_volume : zero(T)
                gas_increment_values[i + halo, j + halo] =
                    effective_volume > zero(T) ?
                    gas_fraction_increment / effective_volume : zero(T)
            end
        end
        _parallel_block_columns(mesh) do b, first_j, last_j
            indicator = @inbounds work.indicator.blocks[b]
            cellvolume = mesh.blocks[b].volume
            offset = (b - 1) * cells_per_block
            @inbounds for j in first_j:last_j, i in 1:nx
                cell = offset + (j - 1) * nx + i
                ii, jj = i + halo, j + halo
                for k in eachindex(scalars), phase in 1:2
                    liquid_phase = phase == 1
                    values = liquid_phase ? scalars[k].liquid : scalars[k].gas
                    values_block = values.blocks[b]
                    source_block = sources[k][phase].blocks[b]
                    increment = zero(T)
                    for pointer in work.cell_face_offsets[cell]:(
                            work.cell_face_offsets[cell + 1] - 1
                        )
                        face_index = work.cell_faces[pointer]
                        face = mesh.faces[face_index]
                        face.axis == axis || continue
                        orientation = work.cell_face_orientations[pointer]
                        liquid_cm = _conditional_fraction(field, face.minus)
                        liquid_cp = _conditional_fraction(field, face.plus)
                        cm = liquid_phase ? liquid_cm :
                            _get(work.gas_carrier, face.minus)
                        cp = liquid_phase ? liquid_cp :
                            _get(work.gas_carrier, face.plus)
                        nm = cm + (
                            liquid_phase ?
                                _get(work.increment, face.minus) :
                                _get(work.gas_increment, face.minus)
                        )
                        np = cp + (
                            liquid_phase ?
                                _get(work.increment, face.plus) :
                                _get(work.gas_increment, face.plus)
                        )
                        vm = mesh.blocks[face.minus.block].volume *
                            (
                            isnothing(geometry) ? one(T) :
                                _get(geometry.open_fraction, face.minus)
                        )
                        vp = mesh.blocks[face.plus.block].volume *
                            (
                            isnothing(geometry) ? one(T) :
                                _get(geometry.open_fraction, face.plus)
                        )
                        volume = work.open_flux[face_index]
                        flux = liquid_phase ?
                            work.liquid_flux[face_index] :
                            work.gas_flux[face_index]
                        state = _phase_flux_state(
                            _get(values, face.minus), _get(values, face.plus),
                            cm, cp, nm, np,
                            flux, vm, vp,
                            mesh.blocks[face.minus.block].key.level ==
                                mesh.blocks[face.plus.block].key.level ? scheme :
                                amr_interface_scheme,
                            flux >= 0 ?
                                _phase_upstream_value(
                                    values, face.minus, axis, -1
                                ) :
                                _phase_upstream_value(
                                    values, face.plus, axis, 1
                                ),
                            (
                                mesh.blocks[face.minus.block].spacing[axis] +
                                    mesh.blocks[face.plus.block].spacing[axis]
                            ) / 2 / (
                                flux >= 0 ?
                                    mesh.blocks[face.minus.block].spacing[axis] :
                                    mesh.blocks[face.plus.block].spacing[axis]
                            ),
                        )
                        intrinsic = values_block[ii, jj]
                        chi = liquid_phase ? indicator[ii, jj] :
                            1 - indicator[ii, jj]
                        open = isnothing(geometry) ? one(T) :
                            geometry.open_fraction.blocks[b][ii, jj]
                        effective_volume = cellvolume * open
                        effective_volume > zero(T) || continue
                        increment += orientation / effective_volume * (
                            (source_block[ii, jj] - chi * intrinsic) * volume +
                                flux * (intrinsic - state)
                        )
                    end
                    increments[k][phase].blocks[b][ii, jj] = increment
                end
            end
        end
        for k in eachindex(staggered_scalars)
            _advect_phase_faces!(
                staggered_scalars[k], work, axis, face_sources[k],
                work.ctu_threshold, scheme,
                amr_interface_scheme,
            )
        end
        isnothing(momentum) || _advect_phase_momentum!(
            momentum_workspace, work, field, axis, momentum_sources,
            density, scheme, amr_interface_scheme,
        )
        _parallel_block_columns(mesh) do b, first_j, last_j
            fraction = @inbounds field.fraction.blocks[b]
            conditional = @inbounds field.conditional_fraction.blocks[b]
            indicator_values = @inbounds work.indicator.blocks[b]
            fraction_increment = @inbounds work.increment.blocks[b]
            gas_fraction_increment = @inbounds work.gas_increment.blocks[b]
            gas_carrier_values = @inbounds work.gas_carrier.blocks[b]
            open_values = isnothing(geometry) ? nothing :
                @inbounds geometry.open_fraction.blocks[b]
            offset = (b - 1) * cells_per_block
            @inbounds for j in first_j:last_j, i in 1:nx
                ii, jj = i + halo, j + halo
                c = conditional[ii, jj] + fraction_increment[ii, jj]
                open = isnothing(open_values) ? one(T) : open_values[ii, jj]
                if !_wy_fraction_in_bounds(c, open)
                    cell = offset + (j - 1) * nx + i
                    flux_details = map(
                        work.cell_face_offsets[cell]:
                            (work.cell_face_offsets[cell + 1] - 1),
                    ) do pointer
                        face_index = work.cell_faces[pointer]
                        face = mesh.faces[face_index]
                        (
                            face = face_index, face_axis = face.axis,
                            orientation = work.cell_face_orientations[pointer],
                            open_flux = work.open_flux[face_index],
                            liquid_flux = work.liquid_flux[face_index],
                        )
                    end
                    error(
                        "WY conditional fraction out of bounds: $c " *
                            "in cell $(CellRef(b, i, j)); axis=$axis, " *
                            "old=$(conditional[ii, jj]), " *
                            "increment=$(fraction_increment[ii, jj]), " *
                            "open=$open, " *
                            "indicator=$(indicator_values[ii, jj]), faces=$flux_details",
                    )
                end
                c = clamp(c, zero(T), one(T))
                snap = field.fraction_floor
                c = c <= snap ? zero(T) : c >= 1 - snap ? one(T) : c
                fraction[ii, jj] = open * c
                conditional[ii, jj] = c
                gas_carrier_values[ii, jj] = one(T) - c
                # Algebraically conservative fluctuation form: outgoing
                # upwind differences are exactly zero, including a completely
                # drained cell.
                for k in eachindex(scalars)
                    q = scalars[k]
                    c > field.fraction_floor &&
                        (
                        q.liquid.blocks[b][ii, jj] +=
                            increments[k][1].blocks[b][ii, jj] / c
                    )
                    gas_denominator = one(T) - c
                    gas_denominator > field.fraction_floor &&
                        (
                        q.gas.blocks[b][ii, jj] +=
                            increments[k][2].blocks[b][ii, jj] /
                            gas_denominator
                    )
                end
            end
        end
        reconstruct!(field)
    end
    for q in staggered_scalars
        _phase_face_average!(
            q.fraction, field.conditional_fraction, field.geometry,
        )
    end
    isnothing(momentum) ||
        _finish_phase_momentum!(momentum, momentum_workspace)
    _update_phase_flux_diagnostics!(work)
    return field
end

function advect!(
        field::InterfaceField{T}, velocity::FaceVelocity, dt,
        work::WYAdvectionWorkspace{T}; kwargs...,
    ) where {T}
    return _advect_wy!(field, velocity, dt, work; kwargs...)
end
