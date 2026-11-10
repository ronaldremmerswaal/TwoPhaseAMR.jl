"""
    WYAdvectionWorkspace(interface, velocity::TwoPhaseVelocity; kwargs...)

Construct the existing geometric-advection workspace after validating the
liquid/gas velocity tuple.  Phase flux storage continues to live in the WY
workspace rather than in the velocity fields.
"""
function WYAdvectionWorkspace(
        field::InterfaceField{T}, velocity::TwoPhaseVelocity;
        interface_transport = nothing, kwargs...,
    ) where {T}
    _validate_two_phase_velocity(velocity, field.mesh, T)
    transport = _select_interface_transport(interface_transport, FrozenInterfaceFluxes())
    transport isa _TwoVelocityInterfaceAdvection || throw(
        ArgumentError(
            "two-velocity interface_transport must be " *
                "ProjectedLiquidVelocity() or FrozenInterfaceFluxes()",
        ),
    )
    return WYAdvectionWorkspace(field; interface_transport = transport, kwargs...)
end

"""
Conservative pre-transport CFL bound for a two-velocity tuple.

Frozen-interface transport uses the time-step-start phase apertures. With a
solid, it also freezes subface ordering and bounds the open donor strip of
each carrier to keep opposing swept open regions disjoint. Projected transport
bounds both the projected liquid velocity and extended gas velocity. For
another workspace policy, the bound is the minimum of the two individual
velocity bounds.
"""
function stable_timestep(
        velocity::TwoPhaseVelocity, work::WYAdvectionWorkspace{T}; cfl = 0.5,
    ) where {T}
    _validate_two_phase_velocity(velocity, work.mesh, T)
    return _stable_timestep(work.interface_transport, velocity, work; cfl)
end

function _stable_timestep(
        ::FrozenInterfaceFluxes, velocity::TwoPhaseVelocity,
        work::WYAdvectionWorkspace; cfl,
    )
    _freeze_phase_apertures!(work.interface, work)
    return _frozen_interface_stable_timestep(velocity, work; cfl)
end

function _stable_timestep(
        ::ProjectedLiquidVelocity, velocity::TwoPhaseVelocity,
        work::WYAdvectionWorkspace; cfl,
    )
    projected = _project_liquid_velocity!(work, velocity)
    gas = _extend_projected_gas_velocity!(work, velocity)
    return min(
        stable_timestep(projected, work; cfl),
        stable_timestep(gas, work; cfl),
    )
end

function _stable_timestep(
        ::Any, velocity::TwoPhaseVelocity, work::WYAdvectionWorkspace; cfl,
    )
    return min(
        stable_timestep(velocity[1], work; cfl),
        stable_timestep(velocity[2], work; cfl),
    )
end

"""
    advect!(interface, velocity::TwoPhaseVelocity, dt, workspace; kwargs...)

Common tuple entry point. Every velocity tuple uses the explicitly selected
projected-liquid or frozen-interface policy, including tuples whose phase
velocities happen to have equal values. Both policies advance centred phase
scalars with the phase fluxes used by the interface update.
"""
function advect!(
        field::InterfaceField{T}, velocity::TwoPhaseVelocity, dt,
        work::WYAdvectionWorkspace{T}; check_timestep = true, kwargs...,
    ) where {T}
    _validate_two_phase_velocity(velocity, field.mesh, T)
    momentum = get(kwargs, :momentum, nothing)
    return _advect_two_velocity!(
        work.interface_transport, field, velocity, dt, work, momentum;
        check_timestep, kwargs...,
    )
end

@inline function _validate_two_velocity_momentum(momentum)
    isnothing(momentum) || momentum isa TwoPhaseVelocity || throw(
        ArgumentError(
            "two-phase velocities require a liquid/gas momentum " *
                "destination tuple",
        ),
    )
    return nothing
end

function _advect_two_velocity!(
        ::FrozenInterfaceFluxes, field::InterfaceField{T},
        velocity::TwoPhaseVelocity, dt, work::WYAdvectionWorkspace{T},
        momentum; check_timestep, kwargs...,
    ) where {T}
    _validate_two_velocity_momentum(momentum)
    # Public advection still prepares its own time-step-start geometry and
    # validates an independently chosen step. The CFL query already
    # reconstructs/finalizes the phase apertures, so do not redo that work
    # at the beginning of the directional WY sweeps.
    if check_timestep
        limit = stable_timestep(velocity, work)
        dt <= limit * (1 + 1.0e-6 + 32eps(T)) || throw(
            ArgumentError("time step exceeds the WY CFL limit"),
        )
    else
        _freeze_phase_apertures!(field, work)
    end
    return _advect_wy!(
        field, velocity, dt, work;
        check_timestep = false, initial_geometry_prepared = true,
        kwargs...,
    )
end

function _advect_two_velocity!(
        ::ProjectedLiquidVelocity, field::InterfaceField{T},
        velocity::TwoPhaseVelocity, dt, work::WYAdvectionWorkspace{T},
        momentum; check_timestep, kwargs...,
    ) where {T}
    _validate_two_velocity_momentum(momentum)
    _reset!(work.diagnostics)
    projected = _project_liquid_velocity!(work, velocity)
    gas = _extend_projected_gas_velocity!(work, velocity)
    limit = min(
        stable_timestep(projected, work),
        stable_timestep(gas, work),
    )
    dt <= limit * (1 + 1.0e-6 + 32eps(T)) || throw(
        ArgumentError(
            "time step exceeds the projected two-velocity WY CFL limit",
        )
    )
    return _advect_wy!(
        field, (projected, gas), dt, work;
        momentum_source = velocity, reset_diagnostics = false,
        check_timestep = false, kwargs...,
    )
end

function _advect_two_velocity!(
        ::Any, field, velocity, dt, work, momentum;
        check_timestep, kwargs...,
    )
    error("unsupported two-velocity interface-advection policy")
end

"""Advance after the flow-step guard prepared/validated frozen geometry.

Only the integrated step path calls this; public `advect!` always prepares
geometry itself. The aperture vectors must remain those of the time-step-start
interface throughout both directional sweeps.
"""
function _advect_transport_substep!(
        ::FrozenInterfaceFluxes, field::InterfaceField,
        velocity::TwoPhaseVelocity, dt,
        work::WYAdvectionWorkspace{T, M, <:FrozenInterfaceFluxes}; kwargs...,
    ) where {T, M}
    return _advect_wy!(
        field, velocity, dt, work;
        check_timestep = false, initial_geometry_prepared = true,
        kwargs...,
    )
end

function _advect_transport_substep!(
        ::Any, field, velocity, dt, work; kwargs...,
    )
    return advect!(field, velocity, dt, work; kwargs...)
end

function _uses_auxiliary_gas_carrier(
        ::TwoPhaseVelocity, ::WYAdvectionWorkspace{T, M, P},
    ) where {T, M, P <: ProjectedLiquidVelocity}
    return true
end

"""
Populate the projected-liquid physical flux and the auxiliary gas-carrier
flux for one split sweep. The liquid flux is cut from the donating region of
the projected liquid velocity. The gas flux is cut independently from the
donating region of the gas velocity; it therefore need not complement the
liquid flux. `open_flux` is the projected carrier flux used by the physical
liquid WY update.
"""
function _compute_directional_wy_fluxes!(
        field::InterfaceField{T}, velocity::TwoPhaseVelocity, dt,
        work::WYAdvectionWorkspace{T, M, <:ProjectedLiquidVelocity}, axis,
    ) where {T, M}
    mesh = field.mesh
    geometry = field.geometry
    liquid_velocity, gas_velocity = velocity
    @_parallel for face_index in eachindex(mesh.faces)
        face = mesh.faces[face_index]
        face.axis == axis || continue
        aperture = isnothing(geometry) ? one(T) :
            @inbounds(geometry.aperture[face_index])

        liquid_u = @inbounds liquid_velocity.interior[face_index]
        liquid_volume = T(dt) * liquid_u * face.area * aperture
        liquid_flux = zero(T)
        if !iszero(liquid_volume)
            donor = liquid_u >= 0 ? face.minus : face.plus
            liquid_fraction = _swept_fraction(
                field, face, donor, liquid_u, dt,
                abs(liquid_volume) / mesh.blocks[donor.block].volume,
            )
            liquid_flux = liquid_volume * liquid_fraction
        end

        gas_u = @inbounds gas_velocity.interior[face_index]
        gas_volume = T(dt) * gas_u * face.area * aperture
        gas_flux = zero(T)
        if !iszero(gas_volume)
            donor = gas_u >= 0 ? face.minus : face.plus
            liquid_fraction = _swept_fraction(
                field, face, donor, gas_u, dt,
                abs(gas_volume) / mesh.blocks[donor.block].volume,
            )
            gas_flux = gas_volume * (one(T) - liquid_fraction)
        end

        @inbounds begin
            work.open_flux[face_index] = liquid_volume
            work.liquid_flux[face_index] = liquid_flux
            work.gas_flux[face_index] = gas_flux
        end
    end
    return work
end

@inline function _averaged_tangent_normal(field, face, tangent)
    return (
        (
            tangent == 1 ? _get(field.normal_x, face.minus) :
                _get(field.normal_y, face.minus)
        ) +
            (
            tangent == 1 ? _get(field.normal_x, face.plus) :
                _get(field.normal_y, face.plus)
        )
    ) / 2
end

"""Freeze phase apertures and the cut-cell subface ordering.

The PLIC is reconstructed between directional sweeps. With a solid, freezing
the ordering with the apertures keeps the swept intervals equal to those used
by the pre-transport geometric timestep bound.
"""
function _freeze_phase_apertures!(
        field::InterfaceField{T, M}, work::WYAdvectionWorkspace{T, M},
    ) where {T, M}
    field.mesh === work.mesh || throw(
        ArgumentError(
            "interface and WY workspace must share a mesh",
        )
    )
    liquid, gas = _current_phase_face_apertures!(field)
    length(work.liquid_aperture) == length(liquid) &&
        length(work.gas_aperture) == length(gas) || throw(
        DimensionMismatch("WY aperture workspace does not match the mesh"),
    )
    copyto!(work.liquid_aperture, liquid)
    copyto!(work.gas_aperture, gas)
    if work.interface_transport isa FrozenInterfaceFluxes && !isnothing(field.geometry)
        length(work.liquid_subface_lower) == length(liquid) || throw(
            DimensionMismatch("WY subface workspace does not match the mesh"),
        )
        geometry = field.geometry
        for (index, face) in pairs(field.mesh.faces)
            open = @inbounds geometry.aperture[index]
            liquid_partial = zero(T) < liquid[index] < open
            gas_partial = zero(T) < gas[index] < open
            # Full-width and empty subfaces do not use their ordering.
            @inbounds work.liquid_subface_lower[index] =
                (liquid_partial || gas_partial) &&
                _averaged_tangent_normal(field, face, 3 - face.axis) >= zero(T)
        end
    end
    return work
end

"""Frozen open/phase subface interval in coordinates of the donor cell."""
function _frozen_subface_interval(
        field::InterfaceField{T}, work::WYAdvectionWorkspace{T},
        face, face_index, donor, liquid_carrier::Bool, geometry,
    ) where {T}
    full_lo, full_hi = _face_local_tangent_interval(
        field.mesh, donor, face,
    )
    full_width = full_hi - full_lo
    open_aperture = isnothing(geometry) ? one(T) :
        @inbounds(geometry.aperture[face_index])
    open_width = clamp(open_aperture, zero(T), one(T)) * full_width
    open_lo, open_hi = full_lo, full_hi
    if open_width < full_width
        tangent = 3 - face.axis
        solid_tangent = _averaged_tangent_normal(
            geometry.solid, face, tangent,
        )
        if solid_tangent > zero(T)
            open_lo = full_hi - open_width
        else
            open_hi = full_lo + open_width
        end
    end
    phase_aperture = liquid_carrier ?
        @inbounds(work.liquid_aperture[face_index]) :
        @inbounds(work.gas_aperture[face_index])
    phase_width = clamp(phase_aperture, zero(T), open_aperture) *
        full_width
    phase_width > zero(T) || return open_lo, open_lo
    phase_width >= open_width && return open_lo, open_hi

    # Without a solid the velocity-based extrusion bound keeps the donating
    # regions disjoint regardless of ordering. Preserve the current PLIC
    # ordering there, and freeze it only where the cut bound depends on it.
    liquid_is_lower = isnothing(geometry) ?
        _averaged_tangent_normal(field, face, 3 - face.axis) >= zero(T) :
        @inbounds(work.liquid_subface_lower[face_index])
    phase_is_lower = liquid_carrier == liquid_is_lower
    return phase_is_lower ?
        (open_lo, open_lo + phase_width) :
        (open_hi - phase_width, open_hi)
end

"""Liquid fraction of one frozen phase-subface donating region."""
function _frozen_subface_swept_fraction(
        field::InterfaceField{T}, face, donor, u, dt,
        tangent_lo, tangent_hi, target, geometry,
    ) where {T}
    target > zero(T) || return zero(T)
    transverse_width = T(tangent_hi - tangent_lo)
    transverse_width > zero(T) || return zero(T)
    mesh = field.mesh
    axis = face.axis
    function swept_bounds(distance)
        normal_lo, normal_hi = u >= 0 ?
            (T(0.5) - distance, T(0.5)) :
            (-T(0.5), -T(0.5) + distance)
        return axis == 1 ?
            (normal_lo, normal_hi, T(tangent_lo), T(tangent_hi)) :
            (T(tangent_lo), T(tangent_hi), normal_lo, normal_hi)
    end

    distance = clamp(target / transverse_width, zero(T), one(T))
    if isnothing(geometry)
        # For Stokes3 and other problems without an embedded solid, the swept
        # donor region is a rectangle clipped only by the liquid PLIC. The
        # analytic rectangle/plane fraction is the same geometric integral as
        # constructing and clipping two heap-allocated polygons on every
        # active face. Retain the represented/target factor for the rare case
        # in which the requested swept volume exceeds the clamped rectangle.
        xlo, xhi, ylo, yhi = swept_bounds(distance)
        liquid = _plane_fraction(
            _get(field.normal_x, donor), _get(field.normal_y, donor),
            _get(field.intercept, donor), xlo, xhi, ylo, yhi,
        )
        represented = (xhi - xlo) * (yhi - ylo)
        return clamp(liquid * represented / target, zero(T), one(T))
    end
    if !isnothing(geometry)
        maximum_area = _open_subcell_area(
            geometry.solid, donor, swept_bounds(one(T))...,
        )
        tolerance = T(256) * eps(T) * max(one(T), maximum_area)
        target <= maximum_area + tolerance || error(
            "frozen phase-subface flux exceeds the donor open volume",
        )
        lo, hi = zero(T), one(T)
        for _ in 1:54
            middle = (lo + hi) / 2
            candidate = _open_subcell_area(
                geometry.solid, donor, swept_bounds(middle)...,
            )
            if candidate < target
                lo = middle
            else
                hi = middle
            end
        end
        distance = (lo + hi) / 2
    end
    bounds = swept_bounds(distance)
    xlo, xhi, ylo, yhi = bounds
    rectangle = NTuple{2, T}[
        (xlo, ylo), (xhi, ylo), (xhi, yhi), (xlo, yhi),
    ]
    open_polygon = isnothing(geometry) ? rectangle : _clip_polygon(
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

function _compute_directional_wy_fluxes!(
        field::InterfaceField{T}, velocity::TwoPhaseVelocity, dt,
        work::WYAdvectionWorkspace{T, M, <:FrozenInterfaceFluxes}, axis,
    ) where {T, M}
    geometry = field.geometry
    mesh = field.mesh
    liquid_velocity, gas_velocity = velocity
    @_parallel for face_index in eachindex(mesh.faces)
        face = mesh.faces[face_index]
        face.axis == axis || continue
        open_flux = liquid_flux = zero(T)
        for liquid_carrier in (true, false)
            aperture = liquid_carrier ?
                work.liquid_aperture[face_index] :
                work.gas_aperture[face_index]
            iszero(aperture) && continue
            u = liquid_carrier ?
                liquid_velocity.interior[face_index] :
                gas_velocity.interior[face_index]
            iszero(u) && continue
            donor = u >= 0 ? face.minus : face.plus
            volume = T(dt) * u * face.area * aperture
            tangent_lo, tangent_hi = _frozen_subface_interval(
                field, work, face, face_index, donor, liquid_carrier,
                geometry,
            )
            fraction = _frozen_subface_swept_fraction(
                field, face, donor, u, dt, tangent_lo, tangent_hi,
                abs(volume) / mesh.blocks[donor.block].volume,
                geometry,
            )
            open_flux += volume
            liquid_flux += volume * fraction
        end
        @inbounds begin
            work.open_flux[face_index] = open_flux
            work.liquid_flux[face_index] = liquid_flux
            work.gas_flux[face_index] = open_flux - liquid_flux
        end
    end
    return work
end
