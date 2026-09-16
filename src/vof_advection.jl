"""
    PhaseScalar(interface; liquid=1, gas=1)

Intrinsic phase values `liquid = s_l` and `gas = s_g`. The conservative
products `c*s_l` and `(1-c)*s_g` exist only as temporary terms during an
update; they are never stored in the field. Initialization arguments are
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
    for ref in _cell_references(mesh)
        x, y = cellcenter(mesh, ref.block, ref.i, ref.j)
        c = _get(interface.fraction, ref)
        l = liquid isa Number ? liquid : liquid(x, y)
        g = gas isa Number ? gas : gas(x, y)
        isfinite(l) && isfinite(g) || throw(ArgumentError("phase scalar values must be finite"))
        _set!(q.liquid, l, ref)
        _set!(q.gas, g, ref)
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
    cells::Vector{CellRef}
    indicator::CellField{T, M}
    increment::CellField{T, M}
    incoming::CellField{T, M}
    outgoing::CellField{T, M}
    liquid_flux::Vector{T}
    gas_flux::Vector{T}
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
    return WYAdvectionWorkspace(
        mesh, _cell_references(mesh), CellField(mesh, T),
        CellField(mesh, T), CellField(mesh, T), CellField(mesh, T),
        zeros(T, length(mesh.faces)), zeros(T, length(mesh.faces)),
        [block.key for block in mesh.blocks], floor, threshold
    )
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
    _zero_interior!(work.incoming)
    _zero_interior!(work.outgoing)
    mesh = work.mesh
    maxrate = zero(T)
    for (i, face) in pairs(mesh.faces)
        u = velocity.interior[i]
        donor, receiver = u >= 0 ? (face.minus, face.plus) : (face.plus, face.minus)
        rate = abs(u) * face.area
        _add!(work.outgoing, rate / mesh.blocks[donor.block].volume, donor)
        _add!(work.incoming, rate / mesh.blocks[receiver.block].volume, receiver)
        maxrate = max(maxrate, abs(u) / mesh.blocks[donor.block].spacing[face.axis])
    end
    for ref in work.cells
        maxrate = max(maxrate, _get(work.incoming, ref), _get(work.outgoing, ref))
    end
    return maxrate == 0 ? T(Inf) : T(cfl) / maxrate
end

"""Integral of an intrinsic phase scalar over the part occupied by `phase`."""
function integrate(q::PhaseScalar, phase::Symbol)
    phase in (:liquid, :gas) || throw(ArgumentError("phase must be :liquid or :gas"))
    total = zero(eltype(q.liquid))
    for (b, block) in pairs(q.interface.mesh.blocks)
        c = interior(q.interface, b)
        values = interior(phase === :liquid ? q.liquid : q.gas, b)
        total += block.volume * sum((phase === :liquid ? c : 1 .- c) .* values)
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

function _swept_fraction(field, face, donor, u, dt)
    c = _get(field.fraction, donor)
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
    distance = abs(u) * dt / block.spacing[axis]
    low, high = u >= 0 ? (0.5 - distance, 0.5) : (-0.5, -0.5 + distance)
    tlow, thigh = transverse - halfwidth, transverse + halfwidth
    xlo, xhi, ylo, yhi = axis == 1 ? (low, high, tlow, thigh) : (tlow, thigh, low, high)
    return _plane_fraction(
        _get(field.normal_x, donor), _get(field.normal_y, donor),
        _get(field.intercept, donor), xlo, xhi, ylo, yhi
    )
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
    face_sources = map(_phase_face_sources, staggered_scalars)
    mesh = field.mesh
    for ref in work.cells
        _set!(work.indicator, _get(field.fraction, ref) > 0.5, ref)
    end
    # Freeze dilation concentrations before the directional sweeps.
    sources = map(scalars) do q
        l, g = copy(q.liquid), copy(q.gas)
        for ref in work.cells
            c = _get(field.fraction, ref)
            chi = _get(work.indicator, ref)
            _set!(l, chi * _get(q.liquid, ref), ref)
            _set!(g, (1 - chi) * _get(q.gas, ref), ref)
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
        _zero_interior!(work.increment)
        _zero_interior!(work.outgoing)
        for pair in increments, increment in pair
            _zero_interior!(increment)
        end
        for (i, face) in pairs(mesh.faces)
            face.axis == axis || continue
            u = velocity.interior[i]
            iszero(u) && continue
            donor = u >= 0 ? face.minus : face.plus
            volume = dt * u * face.area
            liquid = volume * _swept_fraction(field, face, donor, u, dt)
            gas = volume - liquid
            work.liquid_flux[i], work.gas_flux[i] = liquid, gas
            cdonor = _get(field.fraction, donor)
            for (ref, sign) in ((face.minus, 1), (face.plus, -1))
                invvolume = sign / mesh.blocks[ref.block].volume
                _add!(work.outgoing, invvolume * volume, ref)
                _add!(work.increment, invvolume * (_get(work.indicator, ref) * volume - liquid), ref)
            end
        end
        for (i, face) in pairs(mesh.faces)
            face.axis == axis || continue
            cm, cp = _get(field.fraction, face.minus), _get(field.fraction, face.plus)
            nm = cm + _get(work.increment, face.minus)
            np = cp + _get(work.increment, face.plus)
            vm, vp = mesh.blocks[face.minus.block].volume, mesh.blocks[face.plus.block].volume
            volume = dt * velocity.interior[i] * face.area
            for k in eachindex(scalars),phase in 1:2
                liquid = phase == 1
                values = liquid ? scalars[k].liquid : scalars[k].gas
                flux = liquid ? work.liquid_flux[i] : work.gas_flux[i]
                state = _phase_flux_state(
                    _get(values, face.minus), _get(values, face.plus),
                    liquid ? cm : 1 - cm, liquid ? cp : 1 - cp,
                    liquid ? nm : 1 - nm, liquid ? np : 1 - np,
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
                        (
                            mesh.blocks[face.minus.block].spacing[axis] +
                                mesh.blocks[face.plus.block].spacing[axis]
                        ) / 2
                    ) /
                        (
                        volume >= 0 ?
                            mesh.blocks[face.minus.block].spacing[axis] :
                            mesh.blocks[face.plus.block].spacing[axis]
                    )
                )
                for (ref, sign, cellvolume) in ((face.minus, 1, vm), (face.plus, -1, vp))
                    fraction = liquid ? _get(field.fraction, ref) : 1 - _get(field.fraction, ref)
                    intrinsic = _get(values, ref)
                    chi = liquid ? _get(work.indicator, ref) : 1 - _get(work.indicator, ref)
                    _add!(
                        increments[k][phase], sign / cellvolume * (
                            (_get(sources[k][phase], ref) - chi * intrinsic) * volume +
                                flux * (intrinsic - state)
                        ), ref
                    )
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
        for ref in work.cells
            oldc = _get(field.fraction, ref)
            c = oldc + _get(work.increment, ref)
            -128eps(T) <= c <= 1 + 128eps(T) || error("WY fraction out of bounds: $c")
            c = clamp(c, zero(T), one(T))
            snap = 128eps(T)
            c = c <= snap ? zero(T) : c >= 1 - snap ? one(T) : c
            _set!(field.fraction, c, ref)
            # Algebraically conservative fluctuation form: outgoing upwind
            # differences are exactly zero, including a completely drained cell.
            for k in eachindex(scalars)
                q = scalars[k]
                _add!(
                    q.liquid, _get(increments[k][1], ref) /
                        max(c, work.fraction_floor), ref
                )
                _add!(
                    q.gas, _get(increments[k][2], ref) /
                        max(1 - c, work.fraction_floor), ref
                )
            end
        end
        reconstruct!(field)
    end
    for q in staggered_scalars
        _phase_face_average!(q.fraction, field.fraction)
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
struct PhaseProlongation{B, T}
    fraction_buffer::B
    liquid::Bool
    old_fraction::Dict{BlockKey, Matrix{T}}
    new_indices::Dict{BlockKey, Int}
end
function _prepare_prolongation!(p::PhaseProlongation, field)
    empty!(p.old_fraction)
    empty!(p.new_indices)
    buffer = p.fraction_buffer
    source = isempty(buffer.staging) ? buffer.field.blocks : buffer.staging
    for (b, block) in pairs(field.mesh.blocks)
        p.old_fraction[block.key] = copy(source[b])
    end
    return
end
function _new_phase_fraction(p::PhaseProlongation, parent_key, fi, fj, nx, ny, halo)
    if isempty(p.new_indices)
        for (index, block) in pairs(p.fraction_buffer.field.mesh.blocks)
            p.new_indices[block.key] = index
        end
    end
    cx, ii = _fine_location(fi, nx)
    cy, jj = _fine_location(fj, ny)
    child_key = _child_keys(parent_key)[cx + 2(cy - 1)]
    block = p.new_indices[child_key]
    c = p.fraction_buffer.field.blocks[block][ii + halo, jj + halo]
    return p.liquid ? c : 1 - c
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
        parent_weight = p.liquid ? parent_c : 1 - parent_c
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
            w = p.liquid ? c : 1 - c
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
            buffer, liquid,
            Dict{BlockKey, Matrix{T}}(), Dict{BlockKey, Int}()
        )
        register_adaptive_field!(workspace, field; prolongation = strategy)
    end
    return workspace
end
