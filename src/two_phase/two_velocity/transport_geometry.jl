function _frozen_interface_stable_timestep(
        velocity::TwoPhaseVelocity, work::WYAdvectionWorkspace{T, M};
        cfl = 0.5,
    ) where {T, M}
    isfinite(cfl) && 0 < cfl <= 0.5 || throw(
        ArgumentError("WY requires 0 < cfl <= 0.5"),
    )
    _check_wy_topology(velocity[1], work)
    velocity[2].mesh === work.mesh || throw(ArgumentError("phase velocities must share a mesh"))
    geometry = work.interface.geometry
    mesh = work.mesh
    # Validate intrinsic values only in physical phase volumes.
    _check_phase_wy_velocity(velocity[1], work, geometry, :liquid)
    _check_phase_wy_velocity(velocity[2], work, geometry, :gas)
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
                    aperture = phase == 1 ? work.liquid_aperture[face_index] :
                        work.gas_aperture[face_index]
                    iszero(aperture) && continue
                    rate = velocity[phase].interior[face_index] * face.area * aperture
                    outward_rate = orientation * rate
                    outgoing += max(outward_rate, zero(T))
                    incoming += max(-outward_rate, zero(T))
                end
            end
            open = isnothing(geometry) ? one(T) :
                geometry.open_fraction.blocks[block][i + halo, j + halo]
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
        iszero(u) && continue
        donor = u >= 0 ? face.minus : face.plus
        block = mesh.blocks[donor.block]
        maxrate = max(maxrate, abs(u) / block.spacing[face.axis])

        # This is the same donor and frozen phase-subface used by the sweep.
        # A cut donor may have little open volume behind just this subface,
        # even when its whole-cell incoming and outgoing bounds are satisfied.
        lo, hi = _frozen_subface_interval(
            work.interface, work, face, face_index, donor, phase == 1,
            geometry,
        )
        rate = abs(u) * face.area * aperture
        if isnothing(geometry)
            # The subface spans its aperture, so its width is exactly the
            # aperture times the full interval; differencing the clipped
            # interval endpoints would shrink it by round-off.
            width = _frozen_subface_width(
                work.interface, work, face, face_index, donor, phase == 1,
                geometry,
            )
            area = T(0.5) * width
            maxrate = max(maxrate, _wy_strip_rate(rate, area, block.volume))
        else
            solid = geometry.solid
            area = _wy_open_strip_area(
                solid, donor, face.axis, lo, hi, -T(0.5), T(0.5),
            )
            maxrate = max(
                maxrate,
                _wy_cut_strip_rate(
                    rate, area, aperture, block.volume, face_index, face, donor,
                    phase,
                ),
            )

            # Opposing cut-cell subfaces can overlap even when each swept
            # volume fits its own strip. Split their common open support and
            # constrain each volume to its own side. Visit each pair once,
            # from the positive face, including both phase carriers.
            if u > zero(T) && _get(solid.fraction, donor) > zero(T)
                cell = (donor.block - 1) * cells_per_block +
                    (donor.j - 1) * nx + donor.i
                for pointer in work.cell_face_offsets[cell]:(
                        work.cell_face_offsets[cell + 1] - 1
                    )
                    work.cell_face_orientations[pointer] == -1 || continue
                    other_index = work.cell_faces[pointer]
                    other_face = mesh.faces[other_index]
                    other_face.axis == face.axis || continue
                    for other_phase in 1:2
                        other_aperture = other_phase == 1 ?
                            work.liquid_aperture[other_index] :
                            work.gas_aperture[other_index]
                        other_aperture > zero(T) || continue
                        other_u = velocity[other_phase].interior[other_index]
                        other_u < zero(T) || continue
                        other_lo, other_hi = _frozen_subface_interval(
                            work.interface, work, other_face, other_index,
                            donor, other_phase == 1, geometry,
                        )
                        areas = _wy_opposing_strip_areas(
                            solid, donor, face.axis, other_lo, other_hi, lo, hi,
                        )
                        isnothing(areas) && continue
                        left_area, right_area = areas
                        other_rate = abs(other_u) * other_face.area * other_aperture
                        maxrate = max(
                            maxrate,
                            _wy_cut_strip_rate(
                                other_rate, left_area, other_aperture,
                                block.volume, other_index, other_face, donor,
                                other_phase,
                            ),
                            _wy_cut_strip_rate(
                                rate, right_area, aperture, block.volume,
                                face_index, face, donor, phase,
                            ),
                        )
                    end
                end
            end
        end
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

"""
Phase aperture below which a face is treated as single-phase.

The two adjacent face traces are clipped independently, so their average can
miss the shared open aperture by a few ulp and leave a residue on a phase
that does not reach the face. Such a residue is not a physical opening: its
subface has essentially zero width and no open strip, which would make the
cut-donor bound infinite.
"""
@inline _phase_aperture_residue(::Type{T}, open) where {T} =
    T(64) * eps(T) * open

"""
    _snap_phase_apertures(liquid, open)

Split `open` into liquid and gas apertures, collapsing a round-off residue on
either phase to exactly zero.
"""
@inline function _snap_phase_apertures(liquid::T, open::T) where {T}
    liquid = clamp(liquid, zero(T), open)
    gas = open - liquid
    residue = _phase_aperture_residue(T, open)
    liquid <= residue && return zero(T), open
    gas <= residue && return open, zero(T)
    return liquid, gas
end

"""Physical phase apertures on one owned interior face."""
@inline function _phase_face_apertures(
        field::InterfaceField{T}, geometry, index,
    ) where {T}
    face = @inbounds field.mesh.faces[index]
    open = isnothing(geometry) ? one(T) :
        @inbounds(geometry.aperture[index])
    # In a pure phase the complementary aperture is exactly zero. Clipping
    # two independently reconstructed polygons can otherwise leave a 1e-17
    # opening on an absent phase, which must never access its NaN velocity.
    liquid_possible = gas_possible = false
    for cell in (face.minus, face.plus)
        !isnothing(geometry) && iszero(_get(geometry.open_fraction, cell)) && continue
        fraction = _conditional_fraction(field, cell)
        liquid_possible |= fraction > zero(T)
        gas_possible |= fraction < one(T)
    end
    liquid_possible || return zero(T), open
    gas_possible || return open, zero(T)
    liquid = (
        _face_aperture(field, geometry, face, face.minus) +
            _face_aperture(field, geometry, face, face.plus)
    ) / 2
    return _snap_phase_apertures(T(liquid), open)
end

@inline _phase_face_apertures(field::InterfaceField, index) =
    _phase_face_apertures(field, field.geometry, index)

"""Return the physical phase apertures for the current reconstructed state.

The two vectors are interface-owned so pressure assembly and the next CFL
query see exactly the same geometric face openings. A WY sweep copies them to
its separate frozen arrays: those must remain at the *old* interface state
while the sweep changes fractions and invalidates this cache. Direct writes
to the public fraction arrays are detected by `reconstruct!` before reuse.
"""
function _current_phase_face_apertures!(field::InterfaceField{T, M}) where {T, M}
    reconstruct!(field)
    number_of_faces = length(field.mesh.faces)
    if field.phase_apertures_valid[] &&
            length(field.phase_liquid_aperture) == number_of_faces &&
            length(field.phase_gas_aperture) == number_of_faces
        return field.phase_liquid_aperture, field.phase_gas_aperture
    end
    field.phase_apertures_valid[] = false
    resize!(field.phase_liquid_aperture, number_of_faces)
    resize!(field.phase_gas_aperture, number_of_faces)
    geometry = field.geometry
    liquid_values = field.phase_liquid_aperture
    gas_values = field.phase_gas_aperture
    @_parallel for index in eachindex(field.mesh.faces)
        liquid, gas = _phase_face_apertures(field, geometry, index)
        @inbounds begin
            liquid_values[index] = liquid
            gas_values[index] = gas
        end
    end
    field.phase_aperture_updates[] += UInt(1)
    field.phase_apertures_valid[] = true
    return field.phase_liquid_aperture, field.phase_gas_aperture
end


"""Physical phase apertures on one domain-boundary face."""
@inline function _phase_boundary_face_apertures(
        field::InterfaceField{T}, geometry, index,
    ) where {T}
    face = @inbounds field.mesh.boundary_faces[index]
    open = isnothing(geometry) ? one(T) :
        @inbounds(geometry.boundary_aperture[index])
    fraction = _conditional_fraction(field, face.cell)
    iszero(fraction) && return zero(T), open
    fraction == one(T) && return open, zero(T)
    liquid = clamp(
        _face_aperture(field, geometry, face, face.cell), zero(T), open,
    )
    return liquid, open - liquid
end

@inline _phase_boundary_face_apertures(field::InterfaceField, index) =
    _phase_boundary_face_apertures(field, field.geometry, index)
