function _frozen_interface_stable_timestep(
        velocity::TwoPhaseVelocity, work::WYAdvectionWorkspace{T};
        cfl = 0.5,
    ) where {T}
    isfinite(cfl) && 0 < cfl <= 0.5 || throw(
        ArgumentError("WY requires 0 < cfl <= 0.5"),
    )
    foreach(component -> _check_wy_workspace(component, work), velocity)
    mesh = work.mesh
    nx, ny = mesh.block_size
    halo = mesh.halo
    cells_per_block = nx * ny
    _parallel_block_columns(mesh) do block, first_j, last_j
        incoming_values = @inbounds work.incoming.blocks[block]
        outgoing_values = @inbounds work.outgoing.blocks[block]
        offset = (block - 1) * cells_per_block
        cell_volume = mesh.blocks[block].volume
        @inbounds for j in first_j:last_j, i in 1:nx
            cell = offset + (j - 1) * nx + i
            incoming = outgoing = zero(T)
            for pointer in work.cell_face_offsets[cell]:(
                    work.cell_face_offsets[cell + 1] - 1
                )
                face_index = work.cell_faces[pointer]
                face = mesh.faces[face_index]
                orientation = work.cell_face_orientations[pointer]
                for phase in 1:2
                    rate = velocity[phase].interior[face_index] * face.area *
                        (
                        phase == 1 ? work.liquid_aperture[face_index] :
                            work.gas_aperture[face_index]
                    )
                    outward_rate = orientation * rate
                    outgoing += max(outward_rate, zero(T))
                    incoming += max(-outward_rate, zero(T))
                end
            end
            open = isnothing(work.geometry) ? one(T) :
                work.geometry.open_fraction.blocks[block][i + halo, j + halo]
            effective_volume = cell_volume * open
            incoming_values[i + halo, j + halo] =
                effective_volume > zero(T) ? incoming / effective_volume : zero(T)
            outgoing_values[i + halo, j + halo] =
                effective_volume > zero(T) ? outgoing / effective_volume : zero(T)
        end
    end
    maxrate = zero(T)
    for (face_index, face) in pairs(mesh.faces), phase in 1:2
        aperture = phase == 1 ? work.liquid_aperture[face_index] :
            work.gas_aperture[face_index]
        aperture > zero(T) || continue
        u = velocity[phase].interior[face_index]
        donor = u >= 0 ? face.minus : face.plus
        maxrate = max(
            maxrate,
            abs(u) / mesh.blocks[donor.block].spacing[face.axis],
        )
    end
    for block in eachindex(mesh.blocks)
        incoming = @inbounds work.incoming.blocks[block]
        outgoing = @inbounds work.outgoing.blocks[block]
        @inbounds for j in 1:ny, i in 1:nx
            maxrate = max(
                maxrate, incoming[i + halo, j + halo],
                outgoing[i + halo, j + halo],
            )
        end
    end
    return iszero(maxrate) ? T(Inf) : T(cfl) / maxrate
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
