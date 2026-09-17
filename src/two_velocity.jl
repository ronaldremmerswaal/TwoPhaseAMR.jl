"""
    TwoPhaseVelocity

A liquid/gas pair of existing [`FaceVelocity`](@ref) fields, stored as the
ordinary tuple `(liquid_velocity, gas_velocity)`.  The alias adds dispatch and
documentation without introducing another field container or duplicating
velocity storage.
"""
const TwoPhaseVelocity = Tuple{<:FaceVelocity, <:FaceVelocity}

"""Validate the storage contract shared by all two-velocity entry points."""
function _validate_two_phase_velocity(
        velocity::TwoPhaseVelocity, mesh::BlockTreeMesh, ::Type{T},
    ) where {T}
    for (phase, component) in zip((:liquid, :gas), velocity)
        component.mesh === mesh || throw(ArgumentError(
            "$phase velocity and the two-phase field must share a mesh",
        ))
        eltype(component) === T || throw(ArgumentError(
            "$phase velocity has element type $(eltype(component)); expected $T",
        ))
        length(component.interior) == length(mesh.faces) || throw(
            DimensionMismatch("$phase velocity has the wrong interior-face layout"),
        )
        length(component.boundary) == length(mesh.boundary_faces) || throw(
            DimensionMismatch("$phase velocity has the wrong boundary-face layout"),
        )
    end
    return velocity
end

"""Set both phase velocities from `(liquid_function, gas_function)`."""
function set_velocities!(
        velocity::TwoPhaseVelocity, functions::Tuple; time = 0,
    )
    length(functions) == 2 || throw(ArgumentError(
        "two phase-velocity initializers are required",
    ))
    liquid, gas = velocity
    _validate_two_phase_velocity(velocity, liquid.mesh, eltype(liquid))
    set_velocities!(liquid, functions[1]; time)
    set_velocities!(gas, functions[2]; time)
    return velocity
end

"""Apply the impermeable-domain condition to both phase velocities."""
function enforce_no_permeation!(velocity::TwoPhaseVelocity)
    foreach(enforce_no_permeation!, velocity)
    return velocity
end

"""Apply the fixed-solid no-slip mask to both phase velocities."""
function enforce_geometry_no_slip!(
        velocity::TwoPhaseVelocity, geometry::GeometryField,
    )
    _validate_two_phase_velocity(
        velocity, geometry.mesh, eltype(geometry),
    )
    foreach(component -> enforce_geometry_no_slip!(component, geometry), velocity)
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

"""
    WYAdvectionWorkspace(interface, velocity::TwoPhaseVelocity; kwargs...)

Construct the existing geometric-advection workspace after validating the
liquid/gas velocity tuple.  Phase flux storage continues to live in the WY
workspace rather than in the velocity fields.
"""
function WYAdvectionWorkspace(
        field::InterfaceField{T}, velocity::TwoPhaseVelocity; kwargs...,
    ) where {T}
    _validate_two_phase_velocity(velocity, field.mesh, T)
    return WYAdvectionWorkspace(field; kwargs...)
end

"""
Conservative pre-transport CFL bound for a two-velocity tuple.

Until a method-specific interface flux is selected, both phase fields must
individually satisfy the existing multidimensional WY incoming/outgoing-volume
and face-extrusion restrictions.  The minimum of those bounds is therefore a
safe common step.
"""
function stable_timestep(
        velocity::TwoPhaseVelocity, work::WYAdvectionWorkspace{T}; cfl = 0.5,
    ) where {T}
    _validate_two_phase_velocity(velocity, work.mesh, T)
    return min(
        stable_timestep(velocity[1], work; cfl),
        stable_timestep(velocity[2], work; cfl),
    )
end

@inline function _equal_phase_velocities(velocity::TwoPhaseVelocity)
    liquid, gas = velocity
    return liquid === gas || (
        liquid.interior == gas.interior && liquid.boundary == gas.boundary
    )
end

"""
    advect!(interface, velocity::TwoPhaseVelocity, dt, workspace; kwargs...)

Common tuple entry point.  This first implementation supports the exact
one-velocity reduction.  Construction of unequal liquid/gas interface fluxes
is intentionally deferred to the projected-extension and frozen-interface
policies; silently choosing either velocity here would violate compatibility.
"""
function advect!(
        field::InterfaceField{T}, velocity::TwoPhaseVelocity, dt,
        work::WYAdvectionWorkspace{T}; kwargs...,
    ) where {T}
    _validate_two_phase_velocity(velocity, field.mesh, T)
    _equal_phase_velocities(velocity) || throw(ArgumentError(
        "unequal liquid and gas velocities require an explicit two-velocity " *
            "interface-flux method",
    ))
    return advect!(field, velocity[1], dt, work; kwargs...)
end

"""Physical liquid and gas fractions of a full Cartesian cell."""
@inline function _phase_cell_fractions(
        field::InterfaceField{T}, ::Nothing, ref::CellRef,
    ) where {T}
    liquid = _get(field.fraction, ref)
    return liquid, one(T) - liquid
end

@inline function _phase_cell_fractions(
        field::InterfaceField{T}, geometry::GeometryField{T}, ref::CellRef,
    ) where {T}
    liquid = _get(field.fraction, ref)
    open = _get(geometry.open_fraction, ref)
    return liquid, open - liquid
end

@inline _phase_cell_fractions(field::InterfaceField, ref::CellRef) =
    _phase_cell_fractions(field, field.geometry, ref)

@inline _select_phase(values, ::Val{:liquid}) = values[1]
@inline _select_phase(values, ::Val{:gas}) = values[2]

"""Whether a phase occupies a positive physical volume in `ref`."""
@inline function _phase_cell_active(
        field::InterfaceField, geometry, ref::CellRef,
        phase::Union{Val{:liquid}, Val{:gas}},
    )
    return _select_phase(
        _phase_cell_fractions(field, geometry, ref), phase,
    ) > 0
end

@inline function _phase_cell_active(
        field::InterfaceField, ref::CellRef, phase::Union{Val{:liquid}, Val{:gas}},
    )
    return _phase_cell_active(field, field.geometry, ref, phase)
end

@inline function _staggered_open_volume(
        field::InterfaceField, ::Nothing, fraction::FaceField{Axis}, index;
        boundary = false,
    ) where {Axis}
    return one(eltype(field))
end

@inline function _staggered_open_volume(
        field::InterfaceField, geometry::GeometryField,
        fraction::FaceField{Axis}, index; boundary = false,
    ) where {Axis}
    volumes = boundary ? geometry.boundary_dual_volumes[Axis] :
        geometry.dual_volumes[Axis]
    return @inbounds volumes[index]
end

"""
Liquid and gas fractions relative to the open staggered control volume.

`fraction` is the existing geometry-weighted staggered interpolation of the
conditional liquid fraction.  A fully solid dual volume has `(0, 0)` so that
neither velocity is active there.
"""
@inline function _phase_staggered_fractions(
        field::InterfaceField{T}, geometry, fraction::FaceField, index;
        boundary = false,
    ) where {T}
    open_volume = _staggered_open_volume(
        field, geometry, fraction, index; boundary,
    )
    open_volume > zero(T) || return zero(T), zero(T)
    liquid = boundary ? @inbounds(fraction.boundary[index]) :
        @inbounds(fraction.interior[index])
    liquid = clamp(liquid, zero(T), one(T))
    return liquid, one(T) - liquid
end

@inline function _phase_staggered_fractions(
        field::InterfaceField, fraction::FaceField, index;
        boundary = false,
    )
    return _phase_staggered_fractions(
        field, field.geometry, fraction, index; boundary,
    )
end

"""Whether a phase velocity is active on a staggered control volume."""
@inline function _phase_staggered_active(
        field::InterfaceField, geometry, fraction::FaceField, index,
        phase::Union{Val{:liquid}, Val{:gas}}; boundary = false,
    )
    values = _phase_staggered_fractions(
        field, geometry, fraction, index; boundary,
    )
    return _select_phase(values, phase) > 0
end

@inline function _phase_staggered_active(
        field::InterfaceField, fraction::FaceField, index,
        phase::Union{Val{:liquid}, Val{:gas}}; boundary = false,
    )
    return _phase_staggered_active(
        field, field.geometry, fraction, index, phase; boundary,
    )
end

"""Physical phase apertures on one owned interior face."""
@inline function _phase_face_apertures(
        field::InterfaceField{T}, geometry, index,
    ) where {T}
    face = @inbounds field.mesh.faces[index]
    open = isnothing(geometry) ? one(T) :
        @inbounds(geometry.aperture[index])
    liquid = (
        _face_aperture(field, geometry, face, face.minus) +
            _face_aperture(field, geometry, face, face.plus)
    ) / 2
    liquid = clamp(liquid, zero(T), open)
    return liquid, open - liquid
end

@inline _phase_face_apertures(field::InterfaceField, index) =
    _phase_face_apertures(field, field.geometry, index)

"""Physical phase apertures on one domain-boundary face."""
@inline function _phase_boundary_face_apertures(
        field::InterfaceField{T}, geometry, index,
    ) where {T}
    face = @inbounds field.mesh.boundary_faces[index]
    open = isnothing(geometry) ? one(T) :
        @inbounds(geometry.boundary_aperture[index])
    liquid = clamp(
        _face_aperture(field, geometry, face, face.cell), zero(T), open,
    )
    return liquid, open - liquid
end

@inline _phase_boundary_face_apertures(field::InterfaceField, index) =
    _phase_boundary_face_apertures(field, field.geometry, index)
