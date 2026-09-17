"""
    PhaseScalar(interface; liquid=1, gas=1)

Intrinsic phase values `liquid = s_l` and `gas = s_g`. The conservative
products `f_l*s_l` and `(f_open-f_l)*s_g` exist only as temporary terms during
an update; they are never stored in the field. Without embedded geometry,
`f_open=1`. Initialization arguments are
constants or `(x,y)` functions. Advect together with their interface via
`advect!(interface, velocity, dt, work; scalars=(q,))`.
"""
struct PhaseScalar{T, M <: BlockTreeMesh}
    interface::InterfaceField{T, M}
    liquid::CellField{T, M}
    gas::CellField{T, M}
end
function PhaseScalar(interface::InterfaceField{T}; liquid = 1, gas = 1) where {T}
    mesh = interface.mesh
    q = PhaseScalar(interface, CellField(mesh, T), CellField(mesh, T))
    nx, _ = mesh.block_size
    h = mesh.halo
    _parallel_block_columns(mesh) do b, first_j, last_j
        liquid_values = @inbounds q.liquid.blocks[b]
        gas_values = @inbounds q.gas.blocks[b]
        block = @inbounds mesh.blocks[b]
        dx, dy = block.spacing
        @inbounds for j in first_j:last_j, i in 1:nx
            x = block.origin[1] + (i - 1 / 2) * dx
            y = block.origin[2] + (j - 1 / 2) * dy
            l = liquid isa Number ? liquid : liquid(x, y)
            g = gas isa Number ? gas : gas(x, y)
            isfinite(l) && isfinite(g) ||
                throw(ArgumentError("phase scalar values must be finite"))
            liquid_values[i + h, j + h] = l
            gas_values[i + h, j + h] = g
        end
    end
    return q
end

"""
    WYAdvectionWorkspace(interface;
        fraction_floor=sqrt(eps(eltype(interface))), ctu_threshold=0.5)

Reusable WY sweep arrays; reconstruct after regridding by creating a new workspace.
`liquid_flux` and `gas_flux` contain signed, time-integrated volumes on owned
faces from the most recent complete step, in the same order as `mesh.faces`.
Intrinsic phase values are recovered with `max(fraction, fraction_floor)`.
The modified staggered flux interpolant switches to CTU when either adjacent
predicted phase fraction is below `ctu_threshold`.
"""
struct WYAdvectionWorkspace{T, M <: BlockTreeMesh}
    mesh::M
    indicator::CellField{T, M}
    increment::CellField{T, M}
    incoming::CellField{T, M}
    outgoing::CellField{T, M}
    liquid_flux::Vector{T}
    gas_flux::Vector{T}
    cell_face_offsets::Vector{Int}
    cell_faces::Vector{Int}
    cell_face_orientations::Vector{Int8}
    geometry::Any
    topology::Vector{BlockKey}
    fraction_floor::T
    ctu_threshold::T
end
function WYAdvectionWorkspace(
        field::InterfaceField{T, M};
        fraction_floor = sqrt(eps(T)),
        ctu_threshold = 0.5
    ) where {T, M}
    mesh = field.mesh
    floor = T(fraction_floor)
    isfinite(floor) && zero(T) < floor < T(0.5) || throw(
        ArgumentError(
            "fraction_floor must be finite and in (0, 0.5)"
        )
    )
    threshold = T(ctu_threshold)
    isfinite(threshold) && zero(T) < threshold <= one(T) || throw(
        ArgumentError(
            "ctu_threshold must be finite and in (0, 1]"
        )
    )
    offsets, faces, signs = _cell_face_incidence(mesh)
    return WYAdvectionWorkspace(
        mesh, CellField(mesh, T), CellField(mesh, T),
        CellField(mesh, T), CellField(mesh, T),
        zeros(T, length(mesh.faces)), zeros(T, length(mesh.faces)),
        offsets, faces, Int8.(-signs),
        field.geometry, [block.key for block in mesh.blocks], floor, threshold
    )
end

function _resize_zero_cell_field!(field::CellField{T}) where {T}
    mesh = field.mesh
    nx, ny = mesh.block_size
    halo = mesh.halo
    old_length = length(field.blocks)
    resize!(field.blocks, nblocks(mesh))
    for block in eachindex(field.blocks)
        if block > old_length
            field.blocks[block] = zeros(T, nx + 2halo, ny + 2halo)
        else
            fill!(field.blocks[block], zero(T))
        end
    end
    return field
end

"""Retarget WY scratch after AMR while retaining all growable storage."""
function _reconfigure_wy_workspace!(
        workspace::WYAdvectionWorkspace{T}, field::InterfaceField{T},
    ) where {T}
    workspace.mesh === field.mesh || return WYAdvectionWorkspace(
        field; fraction_floor = workspace.fraction_floor,
        ctu_threshold = workspace.ctu_threshold,
    )
    workspace.geometry === field.geometry || return WYAdvectionWorkspace(
        field; fraction_floor = workspace.fraction_floor,
        ctu_threshold = workspace.ctu_threshold,
    )
    for cell_field in (
            workspace.indicator, workspace.increment,
            workspace.incoming, workspace.outgoing,
        )
        _resize_zero_cell_field!(cell_field)
    end
    for flux in (workspace.liquid_flux, workspace.gas_flux)
        resize!(flux, length(field.mesh.faces))
        fill!(flux, zero(T))
    end
    offsets, faces, signs = _cell_face_incidence(field.mesh)
    empty!(workspace.cell_face_offsets)
    append!(workspace.cell_face_offsets, offsets)
    empty!(workspace.cell_faces)
    append!(workspace.cell_faces, faces)
    empty!(workspace.cell_face_orientations)
    append!(workspace.cell_face_orientations, Int8.(-signs))
    empty!(workspace.topology)
    append!(workspace.topology, (block.key for block in field.mesh.blocks))
    return workspace
end
function _check_wy_workspace(velocity, work)
    velocity.mesh === work.mesh || throw(ArgumentError("velocity and workspace must share a mesh"))
    length(work.topology) == nblocks(work.mesh) &&
        all(work.topology[i] == work.mesh.blocks[i].key for i in eachindex(work.topology)) ||
        throw(ArgumentError("rebuild WYAdvectionWorkspace after mesh adaptation"))
    all(isfinite, velocity.interior) && all(isfinite, velocity.boundary) ||
        throw(ArgumentError("velocity must be finite"))
    return all(iszero, velocity.boundary) ||
        throw(ArgumentError("WY currently requires impermeable or periodic boundaries"))
end

"""
    stable_timestep(velocity, work::WYAdvectionWorkspace; cfl=0.5)

WY incoming-volume CFL: `dt*sum(max(-outward_volume_rate,0))/cell_volume`.
Also constrain outgoing volume and each face extrusion to half the donor cell
width; the latter protects subfaces on mixed-level meshes. `0 < cfl <= 0.5`.
"""
function stable_timestep(velocity::FaceVelocity, work::WYAdvectionWorkspace{T}; cfl = 0.5) where {T}
    _check_wy_workspace(velocity, work)
    isfinite(cfl) && 0 < cfl <= 0.5 || throw(ArgumentError("WY requires 0 < cfl <= 0.5"))
    mesh = work.mesh
    geometry = work.geometry
    nx, ny = mesh.block_size
    halo = mesh.halo
    cells_per_block = nx * ny
    _parallel_block_columns(mesh) do b, first_j, last_j
        incoming_values = @inbounds work.incoming.blocks[b]
        outgoing_values = @inbounds work.outgoing.blocks[b]
        volume = mesh.blocks[b].volume
        offset = (b - 1) * cells_per_block
        @inbounds for j in first_j:last_j, i in 1:nx
            cell = offset + (j - 1) * nx + i
            incoming = zero(T)
            outgoing = zero(T)
            for pointer in work.cell_face_offsets[cell]:(
                    work.cell_face_offsets[cell + 1] - 1
                )
                face_index = work.cell_faces[pointer]
                orientation = work.cell_face_orientations[pointer]
                face = mesh.faces[face_index]
                aperture = isnothing(geometry) ? one(T) :
                    geometry.aperture[face_index]
                outward_rate = orientation * velocity.interior[face_index] *
                    face.area * aperture
                outgoing += max(outward_rate, zero(T))
                incoming += max(-outward_rate, zero(T))
            end
            open = isnothing(geometry) ? one(T) :
                geometry.open_fraction.blocks[b][i + halo, j + halo]
            effective_volume = volume * open
            incoming_values[i + halo, j + halo] =
                effective_volume > zero(T) ? incoming / effective_volume : zero(T)
            outgoing_values[i + halo, j + halo] =
                effective_volume > zero(T) ? outgoing / effective_volume : zero(T)
        end
    end
    maxrate = zero(T)
    for (i, face) in pairs(mesh.faces)
        u = velocity.interior[i]
        donor = u >= 0 ? face.minus : face.plus
        maxrate = max(
            maxrate,
            abs(u) / mesh.blocks[donor.block].spacing[face.axis],
        )
    end
    for b in eachindex(mesh.blocks)
        incoming_values = @inbounds work.incoming.blocks[b]
        outgoing_values = @inbounds work.outgoing.blocks[b]
        @inbounds for j in 1:ny, i in 1:nx
            maxrate = max(
                maxrate, incoming_values[i + halo, j + halo],
                outgoing_values[i + halo, j + halo],
            )
        end
    end
    return maxrate == 0 ? T(Inf) : T(cfl) / maxrate
end

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
# never receives a high-order extrapolated amount. Recovery uses the workspace's
# finite fraction floor, which defaults to `sqrt(eps(T))`.
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
    c = _conditional_fraction(field, donor)
    (c == 0 || c == 1) && return c
    mesh = field.mesh
    block = mesh.blocks[donor.block]
    axis, tangent = face.axis, 3 - face.axis
    center = cellcenter(mesh, donor.block, donor.i, donor.j)
    transverse = _wrapped_delta(
        face.center[tangent] - center[tangent],
        mesh.extent[tangent], mesh.periodic[tangent]
    ) / block.spacing[tangent]
    halfwidth = face.area / (2block.spacing[tangent])
    tlow, thigh = transverse - halfwidth, transverse + halfwidth
    distance = abs(u) * dt / block.spacing[axis]
    low, high = u >= 0 ?
        (0.5 - distance, 0.5) : (-0.5, -0.5 + distance)
    xlo, xhi, ylo, yhi = axis == 1 ? (low, high, tlow, thigh) : (tlow, thigh, low, high)
    isnothing(field.geometry) && return _plane_fraction(
        _get(field.normal_x, donor), _get(field.normal_y, donor),
        _get(field.intercept, donor), xlo, xhi, ylo, yhi
    )
    T = eltype(field)
    geometry = field.geometry
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

"""
    advect!(interface, velocity, dt, work::WYAdvectionWorkspace;
            scalars=(), sweep_order=(1,2))

Directionally split geometric Weymouth--Yue transport with a frozen `c > 1/2`
indicator and reconstruction between sweeps. Phase scalars use these exact
volume fluxes and upwind phase concentrations. Their dilation concentrations
are frozen for the whole step, so dilation cancels for solenoidal velocity.
Fractions are only snapped for roundoff; larger bound violations raise errors.
"""
function advect!(
        field::InterfaceField{T}, velocity::FaceVelocity, dt,
        work::WYAdvectionWorkspace{T}; scalars = (), staggered_scalars = (),
        momentum = nothing, density = nothing,
        momentum_workspace = nothing, sweep_order = (1, 2),
        scheme = LW(),
        amr_interface_scheme =
            default_amr_interface_scheme(scheme)
    ) where {T}
    field.mesh === work.mesh || throw(ArgumentError("interface and workspace must share a mesh"))
    sweep_order in ((1, 2), (2, 1)) || throw(ArgumentError("sweep_order must be (1,2) or (2,1)"))
    isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be finite and positive"))
    limit = stable_timestep(velocity, work)
    dt <= limit * (1 + 32eps(T)) || throw(ArgumentError("time step exceeds the WY CFL limit"))
    all(q -> q.interface === field, scalars) || throw(ArgumentError("scalars must belong to this interface"))
    all(q -> q.interface === field, staggered_scalars) || throw(ArgumentError("staggered scalars must belong to this interface"))
    scheme = _canonical_staggered_scheme(scheme)
    amr_interface_scheme = _canonical_staggered_scheme(amr_interface_scheme)
    _validate_amr_interface_scheme(scheme, amr_interface_scheme)
    length(unique(objectid(q.liquid) for q in scalars)) == length(scalars) ||
        throw(ArgumentError("each scalar may only be advanced once"))
    if !isnothing(momentum)
        momentum isa FaceVelocity && momentum.mesh === field.mesh ||
            throw(ArgumentError("momentum destination must be a FaceVelocity on the interface mesh"))
        density isa NTuple{2} && all(r -> isfinite(r)&&r > 0, density) ||
            throw(ArgumentError("momentum advection requires two positive phase densities"))
        momentum_workspace isa MomentumAdvectionWorkspace ||
            throw(ArgumentError("momentum advection requires MomentumAdvectionWorkspace"))
        momentum_workspace.interface === field ||
            throw(ArgumentError("momentum workspace belongs to a different interface"))
        _prepare_momentum_advection!(momentum_workspace, velocity)
    elseif !isnothing(density)||!isnothing(momentum_workspace)
        throw(ArgumentError("density and momentum_workspace require a momentum destination"))
    end
    reconstruct!(field)
    geometry = field.geometry
    face_sources = map(_phase_face_sources, staggered_scalars)
    mesh = field.mesh
    nx, ny = mesh.block_size
    halo = mesh.halo
    cells_per_block = nx * ny
    _parallel_block_columns(mesh) do b, first_j, last_j
        fraction = @inbounds field.conditional_fraction.blocks[b]
        indicator = @inbounds work.indicator.blocks[b]
        @inbounds for j in first_j:last_j, i in 1:nx
            indicator[i + halo, j + halo] = fraction[i + halo, j + halo] > 0.5
        end
    end
    # Freeze dilation concentrations before the directional sweeps.
    sources = map(scalars) do q
        l, g = copy(q.liquid), copy(q.gas)
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
        (l, g)
    end
    increments = map(q -> (CellField(mesh, T), CellField(mesh, T)), scalars)
    fill!(work.liquid_flux, zero(T)); fill!(work.gas_flux, zero(T))
    for axis in sweep_order
        for q in scalars
            fill_halo!(q.liquid)
            fill_halo!(q.gas)
        end
        @_parallel for i in 1:length(mesh.faces)
            face = mesh.faces[i]
            face.axis == axis || continue
            u = velocity.interior[i]
            if iszero(u)
                @inbounds begin
                    work.liquid_flux[i] = zero(T)
                    work.gas_flux[i] = zero(T)
                end
                continue
            end
            donor = u >= 0 ? face.minus : face.plus
            aperture = isnothing(geometry) ? one(T) :
                geometry.aperture[i]
            volume = dt * u * face.area * aperture
            normalized_open_volume = abs(volume) /
                mesh.blocks[donor.block].volume
            liquid = volume * _swept_fraction(
                field, face, donor, u, dt, normalized_open_volume,
            )
            gas = volume - liquid
            @inbounds begin
                work.liquid_flux[i] = liquid
                work.gas_flux[i] = gas
            end
        end
        _parallel_block_columns(mesh) do b, first_j, last_j
            indicator = @inbounds work.indicator.blocks[b]
            outgoing_values = @inbounds work.outgoing.blocks[b]
            increment_values = @inbounds work.increment.blocks[b]
            cellvolume = mesh.blocks[b].volume
            offset = (b - 1) * cells_per_block
            @inbounds for j in first_j:last_j, i in 1:nx
                cell = offset + (j - 1) * nx + i
                outgoing = zero(T)
                fraction_increment = zero(T)
                for pointer in work.cell_face_offsets[cell]:(
                        work.cell_face_offsets[cell + 1] - 1
                    )
                    face_index = work.cell_faces[pointer]
                    face = mesh.faces[face_index]
                    face.axis == axis || continue
                    orientation = work.cell_face_orientations[pointer]
                    aperture = isnothing(geometry) ? one(T) :
                        geometry.aperture[face_index]
                    volume = dt * velocity.interior[face_index] * face.area *
                        aperture
                    liquid_flux = work.liquid_flux[face_index]
                    outgoing += orientation * volume
                    fraction_increment += orientation * (
                        indicator[i + halo, j + halo] * volume - liquid_flux
                    )
                end
                open = isnothing(geometry) ? one(T) :
                    geometry.open_fraction.blocks[b][i + halo, j + halo]
                effective_volume = cellvolume * open
                outgoing_values[i + halo, j + halo] = effective_volume > zero(T) ?
                    outgoing / effective_volume : zero(T)
                increment_values[i + halo, j + halo] =
                    effective_volume > zero(T) ?
                    fraction_increment / effective_volume : zero(T)
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
                        cm = _conditional_fraction(field, face.minus)
                        cp = _conditional_fraction(field, face.plus)
                        nm = cm + _get(work.increment, face.minus)
                        np = cp + _get(work.increment, face.plus)
                        vm = mesh.blocks[face.minus.block].volume *
                            (isnothing(geometry) ? one(T) :
                                _get(geometry.open_fraction, face.minus))
                        vp = mesh.blocks[face.plus.block].volume *
                            (isnothing(geometry) ? one(T) :
                                _get(geometry.open_fraction, face.plus))
                        aperture = isnothing(geometry) ? one(T) :
                            geometry.aperture[face_index]
                        volume = dt * velocity.interior[face_index] *
                            face.area * aperture
                        flux = liquid_phase ?
                            work.liquid_flux[face_index] :
                            work.gas_flux[face_index]
                        state = _phase_flux_state(
                            _get(values, face.minus), _get(values, face.plus),
                            liquid_phase ? cm : 1 - cm,
                            liquid_phase ? cp : 1 - cp,
                            liquid_phase ? nm : 1 - nm,
                            liquid_phase ? np : 1 - np,
                            flux, vm, vp,
                            mesh.blocks[face.minus.block].key.level ==
                                mesh.blocks[face.plus.block].key.level ? scheme :
                                amr_interface_scheme,
                            volume >= 0 ?
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
                                volume >= 0 ?
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
                work.fraction_floor, work.ctu_threshold, scheme,
                amr_interface_scheme
            )
        end
        if !isnothing(momentum)
            _advect_momentum_component!(
                momentum_workspace.x, work, field, axis,
                density, work.ctu_threshold, scheme, amr_interface_scheme
            )
            _advect_momentum_component!(
                momentum_workspace.y, work, field, axis,
                density, work.ctu_threshold, scheme, amr_interface_scheme
            )
        end
        _parallel_block_columns(mesh) do b, first_j, last_j
            fraction = @inbounds field.fraction.blocks[b]
            conditional = @inbounds field.conditional_fraction.blocks[b]
            indicator_values = @inbounds work.indicator.blocks[b]
            fraction_increment = @inbounds work.increment.blocks[b]
            open_values = isnothing(geometry) ? nothing :
                @inbounds geometry.open_fraction.blocks[b]
            offset = (b - 1) * cells_per_block
            @inbounds for j in first_j:last_j, i in 1:nx
                ii, jj = i + halo, j + halo
                c = conditional[ii, jj] + fraction_increment[ii, jj]
                if !(-128eps(T) <= c <= 1 + 128eps(T))
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
                            velocity = velocity.interior[face_index],
                            aperture = isnothing(geometry) ? one(T) :
                                geometry.aperture[face_index],
                            liquid_flux = work.liquid_flux[face_index],
                        )
                    end
                    error(
                        "WY conditional fraction out of bounds: $c " *
                        "in cell $(CellRef(b, i, j)); axis=$axis, " *
                        "old=$(conditional[ii, jj]), " *
                        "increment=$(fraction_increment[ii, jj]), " *
                        "open=$(isnothing(open_values) ? one(T) : open_values[ii, jj]), " *
                        "indicator=$(indicator_values[ii, jj]), faces=$flux_details",
                    )
                end
                c = clamp(c, zero(T), one(T))
                snap = 128eps(T)
                c = c <= snap ? zero(T) : c >= 1 - snap ? one(T) : c
                open = isnothing(open_values) ? one(T) : open_values[ii, jj]
                fraction[ii, jj] = open * c
                conditional[ii, jj] = c
                # Algebraically conservative fluctuation form: outgoing
                # upwind differences are exactly zero, including a completely
                # drained cell.
                for k in eachindex(scalars)
                    q = scalars[k]
                    q.liquid.blocks[b][ii, jj] +=
                        increments[k][1].blocks[b][ii, jj] /
                        max(c, work.fraction_floor)
                    q.gas.blocks[b][ii, jj] +=
                        increments[k][2].blocks[b][ii, jj] /
                        max(1 - c, work.fraction_floor)
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
    if !isnothing(momentum)
        copyto!(momentum, momentum_workspace.x.value)
        copyto!(momentum, momentum_workspace.y.value)
    end
    return field
end

# Intrinsic values use limited linear prolongation. A local correction makes
# the phase-volume-weighted sum of the four fine values equal the coarse phase
# amount. Coarsening uses the corresponding phase-volume-weighted average.
struct PhaseProlongation{B, F, T}
    fraction_buffer::B
    interface::F
    liquid::Bool
    old_fraction::Dict{BlockKey, Matrix{T}}
    old_open::Dict{BlockKey, Matrix{T}}
    new_indices::Dict{BlockKey, Int}
end
function _prepare_prolongation!(p::PhaseProlongation, field, workspace)
    empty!(p.old_fraction)
    empty!(p.old_open)
    empty!(p.new_indices)
    buffer = p.fraction_buffer
    source = isempty(buffer.staging) ? buffer.field.blocks : buffer.staging
    for (b, block) in pairs(field.mesh.blocks)
        p.old_fraction[block.key] = copy(source[b])
        if !isnothing(p.interface.geometry)
            p.old_open[block.key] = copy(
                p.interface.geometry.open_fraction.blocks[b],
            )
        end
    end
    return
end
function _prepare_parallel_prolongation!(p::PhaseProlongation, workspace)
    empty!(p.new_indices)
    for (index, block) in pairs(workspace.mesh.blocks)
        p.new_indices[block.key] = index
    end
    return nothing
end
function _new_phase_fraction(p::PhaseProlongation, parent_key, fi, fj, nx, ny, halo)
    cx, ii = _fine_location(fi, nx)
    cy, jj = _fine_location(fj, ny)
    child_key = _child_keys(parent_key)[cx + 2(cy - 1)]
    block = p.new_indices[child_key]
    c = p.fraction_buffer.field.blocks[block][ii + halo, jj + halo]
    p.liquid && return c
    isnothing(p.interface.geometry) && return one(c) - c
    strategy = p.fraction_buffer.prolongation
    solid_x, solid_y, solid_intercept = strategy.geometry_planes[parent_key]
    pi, pj = (fi + 1) ÷ 2 + halo, (fj + 1) ÷ 2 + halo
    xlo = isodd(fi) ? -0.5 : 0.0
    ylo = isodd(fj) ? -0.5 : 0.0
    T = typeof(c)
    rectangle = NTuple{2, T}[
        (xlo, ylo), (xlo + 0.5, ylo),
        (xlo + 0.5, ylo + 0.5), (xlo, ylo + 0.5),
    ]
    open_polygon = _clip_polygon(
        rectangle, solid_x[pi, pj], solid_y[pi, pj],
        solid_intercept[pi, pj]; keep_less = false,
    )
    return clamp(4 * _polygon_area(open_polygon) - c, zero(T), one(T))
end
function (p::PhaseProlongation)(child, parent, halo, key)
    parent_key = _parent_key(key)
    fractions = p.old_fraction[parent_key]
    nx, ny = size(parent) .- 2halo
    for j in 1:ny,i in 1:nx
        fi = mod(key.i, 2) * nx + i
        fj = mod(key.j, 2) * ny + j
        pi = (fi + 1) ÷ 2 + halo
        pj = (fj + 1) ÷ 2 + halo
        value = parent[pi, pj]
        slope_x = _mc_slope(value - parent[pi - 1, pj], parent[pi + 1, pj] - value)
        slope_y = _mc_slope(value - parent[pi, pj - 1], parent[pi, pj + 1] - value)
        if iszero(slope_x) && iszero(slope_y)
            child[i + halo, j + halo] = value
            continue
        end
        quarter = one(value) / 4
        candidate(ii, jj) = value + (isodd(ii) ? -quarter : quarter) * slope_x +
            (isodd(jj) ? -quarter : quarter) * slope_y
        weighted = weight = zero(value)
        for fine_j in (2(pj - halo) - 1):2(pj - halo),
                fine_i in (2(pi - halo) - 1):2(pi - halo)
            w = _new_phase_fraction(p, parent_key, fine_i, fine_j, nx, ny, halo)
            weighted += w * candidate(fine_i, fine_j)
            weight += w
        end
        parent_c = fractions[pi, pj]
        parent_open = isnothing(p.interface.geometry) ? one(parent_c) :
            p.old_open[parent_key][pi, pj]
        parent_weight = p.liquid ? parent_c : parent_open - parent_c
        correction = weight > sqrt(eps(eltype(parent))) ?
            (4parent_weight * value - weighted) / weight : zero(value)
        child[i + halo, j + halo] = candidate(fi, fj) + correction
    end
    return child
end
function _restrict_parent!(p::PhaseProlongation, parent, children, halo, key)
    fraction_children = ntuple(i -> p.old_fraction[_child_keys(key)[i]], 4)
    nx, ny = size(parent) .- 2halo
    for j in 1:ny,i in 1:nx
        weighted = weight = plain = zero(eltype(parent))
        first_value = nothing
        constant = true
        for fj in (2j - 1):2j,fi in (2i - 1):2i
            cx, ii = _fine_location(fi, nx);cy, jj = _fine_location(fj, ny)
            child_index = cx + 2(cy - 1)
            value = children[child_index][ii + halo, jj + halo]
            if isnothing(first_value)
                first_value = value
            else
                constant &= value == first_value
            end
            c = fraction_children[child_index][ii + halo, jj + halo]
            child_key = _child_keys(key)[child_index]
            open = isnothing(p.interface.geometry) ? one(c) :
                p.old_open[child_key][ii + halo, jj + halo]
            w = p.liquid ? c : open - c
            weighted += w * value;weight += w;plain += value
        end
        parent[i + halo, j + halo] = constant ? first_value :
            (weight > 0 ? weighted / weight : plain / 4)
    end
    return parent
end
function register_adaptive_field!(workspace::AMRWorkspace, q::PhaseScalar{T}) where {T}
    register_adaptive_field!(workspace, q.interface)
    index = _find_field_buffer(workspace, q.interface.fraction)
    buffer = workspace.field_buffers[index]
    for (field, liquid) in ((q.liquid, true), (q.gas, false))
        strategy = PhaseProlongation(
            buffer, q.interface, liquid,
            Dict{BlockKey, Matrix{T}}(), Dict{BlockKey, Matrix{T}}(),
            Dict{BlockKey, Int}()
        )
        register_adaptive_field!(workspace, field; prolongation = strategy)
    end
    return workspace
end
