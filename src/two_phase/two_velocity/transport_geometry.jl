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
    _check_phase_wy_velocities(velocity, work, geometry)
    _, ny = mesh.block_size
    # Each block column records its largest incoming/outgoing rate; with the
    # fixed-chunk face maxima below the bound is an exact maximum, so it does
    # not depend on the thread count.
    column_rates = work.column_rates
    length(column_rates) == nblocks(mesh) * ny || resize!(column_rates, nblocks(mesh) * ny)
    _frozen_column_rates!(column_rates, velocity, work, geometry)
    maxrate = _frozen_face_maximum(velocity, work, geometry)
    for index in eachindex(column_rates)
        maxrate = max(maxrate, @inbounds column_rates[index])
    end
    return iszero(maxrate) ? T(Inf) : T(cfl) / maxrate
end

"""Set `column_rates[(block - 1) * ny + j]` to the largest incoming or
outgoing rate of the cells in column `j` of a block, writing the per-cell
rates into `work.incoming` and `work.outgoing`. Every block column owns its
entries; the launch captures array and scalar locals, including the block
vectors, rather than the workspace, velocity or geometry structs. Without a
solid the open fractions are `nothing`."""
function _frozen_column_rates!(
        column_rates, velocity::TwoPhaseVelocity, work::WYAdvectionWorkspace{T, M},
        geometry::G,
    ) where {T, M, G}
    mesh = work.mesh
    nx, ny = mesh.block_size
    halo = mesh.halo
    cells_per_block = nx * ny
    faces = mesh.faces
    leaf_blocks = mesh.blocks
    incoming_blocks = work.incoming.blocks
    outgoing_blocks = work.outgoing.blocks
    cell_face_offsets = work.cell_face_offsets
    cell_faces = work.cell_faces
    cell_face_orientations = work.cell_face_orientations
    liquid_aperture = work.liquid_aperture
    gas_aperture = work.gas_aperture
    liquid = velocity[1].interior
    gas = velocity[2].interior
    open_blocks = isnothing(geometry) ? nothing : geometry.open_fraction.blocks
    _parallel_block_columns(mesh) do block, first_j, last_j
        incoming_values = @inbounds incoming_blocks[block]
        outgoing_values = @inbounds outgoing_blocks[block]
        offset = (block - 1) * cells_per_block
        cell_volume = @inbounds leaf_blocks[block].volume
        @inbounds for j in first_j:last_j
            column_rate = zero(T)
            for i in 1:nx
                cell = offset + (j - 1) * nx + i
                incoming = outgoing = zero(T)
                for slot in cell_face_offsets[cell]:(cell_face_offsets[cell + 1] - 1)
                    face_index = cell_faces[slot]
                    face = faces[face_index]
                    orientation = cell_face_orientations[slot]
                    for phase in 1:2
                        aperture = phase == 1 ? liquid_aperture[face_index] :
                            gas_aperture[face_index]
                        iszero(aperture) && continue
                        u = phase == 1 ? liquid[face_index] : gas[face_index]
                        rate = u * face.area * aperture
                        outward_rate = orientation * rate
                        outgoing += max(outward_rate, zero(T))
                        incoming += max(-outward_rate, zero(T))
                    end
                end
                cell_open = open_blocks === nothing ? one(T) :
                    open_blocks[block][i + halo, j + halo]
                effective_volume = cell_volume * cell_open
                incoming_rate = effective_volume > zero(T) ? incoming / effective_volume : zero(T)
                outgoing_rate = effective_volume > zero(T) ? outgoing / effective_volume : zero(T)
                incoming_values[i + halo, j + halo] = incoming_rate
                outgoing_values[i + halo, j + halo] = outgoing_rate
                column_rate = max(
                    column_rate, _canonical_nan(incoming_rate),
                    _canonical_nan(outgoing_rate),
                )
            end
            column_rates[(block - 1) * ny + j] = column_rate
        end
    end
    return column_rates
end

"""Largest frozen-subface rate over all faces and phase carriers, without a
solid: the extrusion rate `|u|/h` and the donor strip rate of each subface.
Threaded fixed chunks over plain arrays; the chunk maxima are folded in chunk
order (`max` is exact, `NaN` terms canonical), so the bound does not depend on
the thread count."""
function _frozen_face_maximum(velocity, work::WYAdvectionWorkspace{T}, ::Nothing) where {T}
    mesh = work.mesh
    faces = mesh.faces
    leaf_blocks = mesh.blocks
    extent = mesh.extent
    periodic = mesh.periodic
    liquid_aperture = work.liquid_aperture
    gas_aperture = work.gas_aperture
    liquid = velocity[1].interior
    gas = velocity[2].interior
    number_of_faces = length(faces)
    chunks = _reduction_chunks(number_of_faces)
    partials = _reduction_partials!(work.rate_partials, number_of_faces)
    @_parallel work = _REDUCTION_CHUNK for chunk in 1:chunks
        chunk_rate = zero(T)
        for face_index in _reduction_range(chunk, number_of_faces), phase in 1:2
            aperture = phase == 1 ? @inbounds(liquid_aperture[face_index]) :
                @inbounds(gas_aperture[face_index])
            u = phase == 1 ? @inbounds(liquid[face_index]) : @inbounds(gas[face_index])
            rate = _frozen_full_face_rate(
                leaf_blocks, extent, periodic, @inbounds(faces[face_index]), aperture, u,
            )
            chunk_rate = max(chunk_rate, _canonical_nan(rate))
        end
        @inbounds partials[chunk] = chunk_rate
    end
    return _fold_chunk_maxima(zero(T), partials, chunks)
end

# Cut cells: the strip rates read the solid reconstruction and raise the
# empty-strip error in face order, so this pass runs serially.
function _frozen_face_maximum(velocity, work::WYAdvectionWorkspace{T}, geometry::GeometryField) where {T}
    maxrate = zero(T)
    for face_index in eachindex(work.mesh.faces), phase in 1:2
        maxrate = max(
            maxrate,
            _canonical_nan(_frozen_cut_face_rate(velocity, work, geometry, face_index, phase)),
        )
    end
    return maxrate
end

"""Rate bound of one frozen phase subface of a face without a solid: the
subface spans its aperture, so its width is the aperture times the full
interval (differencing the clipped endpoints would shrink it by round-off)."""
@inline function _frozen_full_face_rate(leaf_blocks, extent, periodic, face, aperture, u)
    T = typeof(aperture)
    # A subface at round-off width carries no flux the sweep can resolve.
    aperture > _wy_strip_area_tolerance(T) || return zero(T)
    iszero(u) && return zero(T)
    donor = u >= 0 ? face.minus : face.plus
    block = @inbounds leaf_blocks[donor.block]
    full_lo, full_hi = _face_local_tangent_interval(leaf_blocks, extent, periodic, donor, face)
    width = _frozen_subface_width(T(full_hi - full_lo), aperture, one(T))
    rate = abs(u) * face.area * aperture
    return max(
        abs(u) / block.spacing[face.axis],
        _wy_strip_rate(rate, T(0.5) * width, block.volume),
    )
end

"""
Largest WY rate bound of the frozen `phase` subface of one face behind a
possibly cut donor: the extrusion rate, the donor strip rate and the rates of
opposing strip pairs. A positive flux through a cut donor strip without
resolvable open area throws the strip error.
"""
@inline function _frozen_cut_face_rate(
        velocity, work::WYAdvectionWorkspace{T}, geometry::GeometryField, face_index, phase,
    ) where {T}
    mesh = work.mesh
    face = @inbounds mesh.faces[face_index]
    aperture = phase == 1 ? @inbounds(work.liquid_aperture[face_index]) :
        @inbounds(work.gas_aperture[face_index])
    # A subface at round-off width carries no flux the sweep can resolve.
    aperture > _wy_strip_area_tolerance(T) || return zero(T)
    u = @inbounds velocity[phase].interior[face_index]
    iszero(u) && return zero(T)
    donor = u >= 0 ? face.minus : face.plus
    block = @inbounds mesh.blocks[donor.block]
    face_rate = abs(u) / block.spacing[face.axis]

    # This is the same donor and frozen phase-subface used by the sweep.
    # A cut donor may have little open volume behind just this subface,
    # even when its whole-cell incoming and outgoing bounds are satisfied.
    rate = abs(u) * face.area * aperture
    lo, hi = _frozen_subface_interval(
        work.interface, work, face, face_index, donor, phase == 1, geometry,
    )
    solid = geometry.solid
    area = _wy_open_strip_area(solid, donor, face.axis, lo, hi, -T(0.5), T(0.5))
    face_rate = max(
        face_rate,
        _wy_cut_strip_rate(
            rate, area, aperture, block.volume, face_index, face, donor, phase,
        ),
    )

    # Opposing cut-cell subfaces can overlap even when each swept
    # volume fits its own strip. Split their common open support and
    # constrain each volume to its own side. Visit each pair once,
    # from the positive face, including both phase carriers.
    if u > zero(T) && _get(solid.fraction, donor) > zero(T)
        nx, ny = mesh.block_size
        cell = (donor.block - 1) * nx * ny + (donor.j - 1) * nx + donor.i
        for slot in work.cell_face_offsets[cell]:(work.cell_face_offsets[cell + 1] - 1)
            work.cell_face_orientations[slot] == -1 || continue
            other_index = work.cell_faces[slot]
            other_face = mesh.faces[other_index]
            other_face.axis == face.axis || continue
            for other_phase in 1:2
                other_aperture = other_phase == 1 ?
                    work.liquid_aperture[other_index] :
                    work.gas_aperture[other_index]
                other_aperture > _wy_strip_area_tolerance(T) || continue
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
                face_rate = max(
                    face_rate,
                    _wy_side_strip_rate(
                        other_rate, left_area, other_aperture, block.volume,
                    ),
                    _wy_side_strip_rate(rate, right_area, aperture, block.volume),
                )
            end
        end
    end
    return face_rate
end

"""
Validate the intrinsic velocities of both phases in their physical volumes:
finite on active interior faces, zero on active boundary faces. The interior
faces are checked in fixed chunks that record the failing phases (bit 1
liquid, bit 2 gas) in the workspace's owned flags; the errors are raised
afterwards in the serial order (liquid interior, liquid boundary, gas
interior, gas boundary).
"""
function _check_phase_wy_velocities(velocity, work::WYAdvectionWorkspace, geometry)
    interface = work.interface
    number_of_faces = length(work.mesh.faces)
    chunks = _reduction_chunks(number_of_faces)
    failures = _reduction_partials!(work.velocity_failure, number_of_faces)
    _check_phase_wy_interior!(
        failures, velocity[1].interior, velocity[2].interior, interface, geometry,
    )
    for (bit, phase, phase_velocity) in ((1, :liquid, velocity[1]), (2, :gas, velocity[2]))
        for chunk in 1:chunks
            iszero(@inbounds(failures[chunk]) & bit) ||
                throw(ArgumentError("active $phase velocity must be finite"))
        end
        for (index, face) in pairs(work.mesh.boundary_faces)
            _phase_cell_is_active(interface, geometry, face.cell, phase) || continue
            isfinite(phase_velocity.boundary[index]) && iszero(phase_velocity.boundary[index]) ||
                throw(ArgumentError("WY requires zero active phase boundary velocity"))
        end
    end
    return nothing
end

# Full Cartesian faces: a threaded launch over the phase face arrays and the
# conditional-fraction block vectors (boxed per launch, see `_parallel_for`).
function _check_phase_wy_interior!(failures, liquid, gas, interface, ::Nothing)
    T = eltype(liquid)
    faces = interface.mesh.faces
    leaf_blocks = interface.mesh.blocks
    fraction_blocks = interface.conditional_fraction.blocks
    halo = interface.mesh.halo
    number_of_faces = length(faces)
    @_parallel work = _REDUCTION_CHUNK for chunk in 1:_reduction_chunks(number_of_faces)
        failed = 0
        for index in _reduction_range(chunk, number_of_faces)
            fraction = _staggered_liquid_fraction(
                leaf_blocks, fraction_blocks, halo, @inbounds(faces[index]), T,
            )
            failed |= _active_nonfinite_bits(liquid, gas, index, fraction)
        end
        @inbounds failures[chunk] = failed
    end
    return failures
end

# Cut cells read the solid reconstruction: serial chunks (see
# `_mark_inactive_interior!`).
function _check_phase_wy_interior!(failures, liquid, gas, interface, geometry::GeometryField)
    faces = interface.mesh.faces
    number_of_faces = length(faces)
    for chunk in 1:_reduction_chunks(number_of_faces)
        failed = 0
        for index in _reduction_range(chunk, number_of_faces)
            iszero(geometry.face_dual_volumes[index]) && continue
            fraction = _staggered_liquid_fraction(interface, geometry, faces[index])
            failed |= _active_nonfinite_bits(liquid, gas, index, fraction)
        end
        failures[chunk] = failed
    end
    return failures
end

"""Bit 1 (liquid) and bit 2 (gas) when that phase is active on the face
(`fraction > 0`, `fraction < 1`) and its velocity is not finite."""
@inline function _active_nonfinite_bits(liquid, gas, index, fraction)
    failed = 0
    fraction > 0 && !isfinite(@inbounds liquid[index]) && (failed |= 1)
    fraction < 1 && !isfinite(@inbounds gas[index]) && (failed |= 2)
    return failed
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
cut-donor bound infinite. Both traces are fractions of the *whole* face, so
their rounding error is absolute rather than proportional to the aperture.
"""
@inline _phase_aperture_residue(::Type{T}) where {T} = T(64) * eps(T)

"""
    _snap_phase_apertures(liquid, open)

Split `open` into liquid and gas apertures, collapsing a round-off residue on
either phase to exactly zero. The pair always sums to `open`, and a face whose
own open extent is at round-off level keeps both apertures: the cut-donor
bounds treat such a face as carrying no resolvable flux.
"""
@inline function _snap_phase_apertures(liquid::T, open::T) where {T}
    liquid = clamp(liquid, zero(T), open)
    gas = open - liquid
    residue = _phase_aperture_residue(T)
    open > residue || return liquid, gas
    liquid <= residue && return zero(T), open
    gas <= residue && return open, zero(T)
    return liquid, gas
end

"""Physical phase apertures on one owned interior face.

The liquid aperture is the mean of the two adjacent PLIC traces. With the
interface's `:pure_neighbour` rule (`MimeticProjection`), a face between a
pure cell and a cut cell whose PLIC line crosses it strictly inside takes the
pure cell's value instead: such a crossing is a corner clip of a line that
leaves the cell nearby, and halving it gives the cut cell a sliver polygon."""
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
    minus = _face_aperture(field, geometry, face, face.minus)
    plus = _face_aperture(field, geometry, face, face.plus)
    if field.phase_aperture_rule[] === :pure_neighbour
        pure = _pure_neighbour_aperture(field, face, minus, plus)
        isnothing(pure) || return _snap_phase_apertures(T(pure), open)
    end
    liquid = (minus + plus) / 2
    return _snap_phase_apertures(T(liquid), open)
end

"""The pure neighbour's liquid aperture (`0` or `1`) if exactly one cell of
`face` is pure and the other cell's PLIC trace on the face is strictly
partial, else `nothing`."""
@inline function _pure_neighbour_aperture(field, face, minus, plus)
    minus_fraction = _conditional_fraction(field, face.minus)
    plus_fraction = _conditional_fraction(field, face.plus)
    minus_pure = iszero(minus_fraction) || isone(minus_fraction)
    plus_pure = iszero(plus_fraction) || isone(plus_fraction)
    minus_pure == plus_pure && return nothing
    pure, trace = minus_pure ? (minus_fraction, plus) : (plus_fraction, minus)
    return zero(trace) < trace < one(trace) ? pure : nothing
end

"""Select how face apertures combine the two adjacent traces (`:average` or
`:pure_neighbour`); a change invalidates the cached apertures."""
function _set_phase_aperture_rule!(field::InterfaceField, rule::Symbol)
    rule in (:average, :pure_neighbour) || throw(
        ArgumentError("phase aperture rule must be :average or :pure_neighbour"),
    )
    field.phase_aperture_rule[] === rule && return field
    field.phase_aperture_rule[] = rule
    field.phase_apertures_valid[] = false
    return field
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
    return _snap_phase_apertures(T(liquid), open)
end

@inline _phase_boundary_face_apertures(field::InterfaceField, index) =
    _phase_boundary_face_apertures(field, field.geometry, index)
