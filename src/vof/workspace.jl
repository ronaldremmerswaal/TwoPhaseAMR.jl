"""
    WYAdvectionWorkspace(interface; ctu_threshold=0.5)

Reusable WY sweep arrays; reconstruct after regridding by creating a new workspace.
`liquid_flux` and `gas_flux` contain signed, time-integrated volumes on owned
faces from the most recent complete step, in the same order as `mesh.faces`.
Intrinsic phase values are recovered with the interface's
`fraction_floor`; the same threshold removes small VOF fragments.
The modified staggered flux interpolant switches to CTU when either adjacent
predicted phase fraction is below `ctu_threshold`. The workspace uses its
interface's fixed geometry and, for a geometry field, freezes the liquid/gas
subface ordering there.
Cell-scalar source and increment buffers grow on demand and are reused across
steps and resized when the workspace is retargeted after mesh adaptation.
"""
mutable struct WYAdvectionWorkspace{T, M <: BlockTreeMesh, F}
    # Reference identity avoids boxing this large, concretely typed workspace
    # when an AMR-retargetable parent stores it behind a partial type.
    const mesh::M
    const interface::F
    const indicator::CellField{T, M}
    const increment::CellField{T, M}
    const incoming::CellField{T, M}
    const outgoing::CellField{T, M}
    const liquid_dilation::CellField{T, M}
    const gas_dilation::CellField{T, M}
    const gas_carrier::CellField{T, M}
    const gas_increment::CellField{T, M}
    const scalar_sources::Vector{NTuple{2, CellField{T, M}}}
    const scalar_increments::Vector{NTuple{2, CellField{T, M}}}
    const open_flux::Vector{T}
    const liquid_flux::Vector{T}
    const gas_flux::Vector{T}
    const liquid_aperture::Vector{T}
    const gas_aperture::Vector{T}
    const liquid_subface_lower::Vector{Bool}
    const cell_face_offsets::Vector{Int}
    const cell_faces::Vector{Int}
    const cell_face_orientations::Vector{Int8}
    const topology::Vector{BlockKey}
    # Owned WY bound flags, one zero entry per block column: the value of an
    # entry is the column index of a cell whose conditional fraction left
    # `[0, open]` in the sweep being validated.
    const conditional_failure::Vector{Int32}
    const ctu_threshold::T
    const diagnostics::PhaseFluxDiagnostics{T}
end
function WYAdvectionWorkspace(
        field::InterfaceField{T, M};
        ctu_threshold = 0.5,
    ) where {T, M}
    mesh = field.mesh
    threshold = T(ctu_threshold)
    isfinite(threshold) && zero(T) < threshold <= one(T) || throw(
        ArgumentError(
            "ctu_threshold must be finite and in (0, 1]"
        )
    )
    offsets, faces, signs = _cell_face_incidence(mesh)
    number_of_faces = length(mesh.faces)
    return WYAdvectionWorkspace(
        mesh, field, CellField(mesh, T), CellField(mesh, T),
        CellField(mesh, T), CellField(mesh, T),
        CellField(mesh, T), CellField(mesh, T), CellField(mesh, T),
        CellField(mesh, T),
        NTuple{2, CellField{T, typeof(mesh)}}[],
        NTuple{2, CellField{T, typeof(mesh)}}[],
        zeros(T, number_of_faces),
        zeros(T, number_of_faces), zeros(T, number_of_faces),
        zeros(T, number_of_faces), zeros(T, number_of_faces),
        fill(false, isnothing(field.geometry) ? 0 : number_of_faces),
        offsets, faces, Int8.(-signs),
        [block.key for block in mesh.blocks],
        zeros(Int32, nblocks(mesh) * mesh.block_size[2]), threshold,
        PhaseFluxDiagnostics(T),
    )
end

"""Clear and return the owned WY bound flags of one column-scheduled sweep."""
function _wy_failure_flags!(workspace::WYAdvectionWorkspace)
    mesh = workspace.mesh
    flags = workspace.conditional_failure
    length(flags) == nblocks(mesh) * mesh.block_size[2] ||
        resize!(flags, nblocks(mesh) * mesh.block_size[2])
    fill!(flags, zero(eltype(flags)))
    return flags
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
    workspace.interface === field || return WYAdvectionWorkspace(
        field; ctu_threshold = workspace.ctu_threshold,
    )
    workspace.mesh === field.mesh || return WYAdvectionWorkspace(
        field; ctu_threshold = workspace.ctu_threshold,
    )
    for cell_field in (
            workspace.indicator, workspace.increment,
            workspace.incoming, workspace.outgoing,
            workspace.liquid_dilation, workspace.gas_dilation,
            workspace.gas_carrier, workspace.gas_increment,
        )
        _resize_zero_cell_field!(cell_field)
    end
    for buffers in (workspace.scalar_sources, workspace.scalar_increments)
        for pair in buffers, cell_field in pair
            _resize_zero_cell_field!(cell_field)
        end
    end
    for flux in (
            workspace.open_flux, workspace.liquid_flux, workspace.gas_flux,
            workspace.liquid_aperture, workspace.gas_aperture,
        )
        resize!(flux, length(field.mesh.faces))
        fill!(flux, zero(T))
    end
    if !isnothing(field.geometry)
        resize!(workspace.liquid_subface_lower, length(field.mesh.faces))
        fill!(workspace.liquid_subface_lower, false)
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
    resize!(
        workspace.conditional_failure,
        nblocks(field.mesh) * field.mesh.block_size[2],
    )
    fill!(workspace.conditional_failure, zero(Int32))
    _reset!(workspace.diagnostics)
    return workspace
end

function _update_phase_flux_diagnostics!(workspace::WYAdvectionWorkspace{T}) where {T}
    maximum_defect = zero(T)
    absolute_defect = zero(T)
    signed_defect = zero(T)
    @inbounds for face in eachindex(workspace.open_flux)
        defect = workspace.liquid_flux[face] + workspace.gas_flux[face] -
            workspace.open_flux[face]
        maximum_defect = max(maximum_defect, abs(defect))
        absolute_defect += abs(defect)
        signed_defect += defect
    end
    diagnostics = workspace.diagnostics
    diagnostics.maximum_partition_defect = maximum_defect
    diagnostics.absolute_partition_defect = absolute_defect
    diagnostics.signed_partition_defect = signed_defect
    return diagnostics
end

function _check_wy_topology(velocity, work)
    velocity.mesh === work.mesh || throw(ArgumentError("velocity and workspace must share a mesh"))
    length(work.topology) == nblocks(work.mesh) &&
        all(work.topology[i] == work.mesh.blocks[i].key for i in eachindex(work.topology)) ||
        throw(ArgumentError("rebuild WYAdvectionWorkspace after mesh adaptation"))
    return nothing
end

function _check_wy_workspace(velocity, work)
    _check_wy_topology(velocity, work)
    all(isfinite, velocity.interior) && all(isfinite, velocity.boundary) ||
        throw(ArgumentError("velocity must be finite"))
    all(iszero, velocity.boundary) ||
        throw(ArgumentError("WY currently requires impermeable or periodic boundaries"))
    return nothing
end

function _check_phase_wy_velocity(velocity, work, geometry::G, phase) where {G}
    for (index, face) in pairs(work.mesh.faces)
        _phase_face_is_active(work.interface, geometry, index, face, phase) || continue
        isfinite(velocity.interior[index]) ||
            throw(ArgumentError("active $phase velocity must be finite"))
    end
    for (index, face) in pairs(work.mesh.boundary_faces)
        _phase_cell_is_active(work.interface, geometry, face.cell, phase) || continue
        isfinite(velocity.boundary[index]) && iszero(velocity.boundary[index]) ||
            throw(ArgumentError("WY requires zero active phase boundary velocity"))
    end
    return nothing
end

"""Open area of a donor rectangle in normalized cell coordinates."""
@inline function _wy_open_strip_area(
        solid::InterfaceField{T}, donor, axis, tangent_lo, tangent_hi,
        normal_lo, normal_hi,
    ) where {T}
    lo, hi = T(tangent_lo), T(tangent_hi)
    nlo, nhi = T(normal_lo), T(normal_hi)
    bounds = axis == 1 ? (nlo, nhi, lo, hi) : (lo, hi, nlo, nhi)
    return _open_subcell_area(solid, donor, bounds...)
end

"""
Areas available to opposing outgoing strips, separated inside their common
open support. Return `nothing` when the strips have no open overlap.
"""
@inline function _wy_opposing_strip_areas(
        solid::InterfaceField{T}, donor, axis,
        left_lo, left_hi, right_lo, right_hi,
    ) where {T}
    lo, hi = max(T(left_lo), T(right_lo)), min(T(left_hi), T(right_hi))
    lo < hi || return nothing
    half = T(0.5)
    overlap = _wy_open_strip_area(solid, donor, axis, lo, hi, -half, half)
    # A common open support at round-off level leaves nothing to separate.
    overlap > _wy_strip_area_tolerance(T) || return nothing

    # The open solid PLIC is a half-plane. Find its normal support in the
    # intersection of the two tangential intervals, then split that support
    # in its interior. A fixed cell midpoint can have zero open area on one
    # side even though both outgoing strips can be swept disjointly.
    normal_lo, normal_hi = -half, half
    fraction = _get(solid.fraction, donor)
    if zero(T) < fraction < one(T)
        nx = _get(solid.normal_x, donor)
        ny = _get(solid.normal_y, donor)
        normal, tangent = axis == 1 ? (nx, ny) : (ny, nx)
        intercept = _get(solid.intercept, donor)
        tangent_max = max(tangent * lo, tangent * hi)
        if normal > zero(T)
            normal_lo = clamp((intercept - tangent_max) / normal, -half, half)
        elseif normal < zero(T)
            normal_hi = clamp((intercept - tangent_max) / normal, -half, half)
        end
    end
    split = (normal_lo + normal_hi) / 2
    left_area = _wy_open_strip_area(
        solid, donor, axis, left_lo, left_hi, -half, split,
    )
    right_area = _wy_open_strip_area(
        solid, donor, axis, right_lo, right_hi, split, half,
    )
    return left_area, right_area
end

@inline function _wy_strip_rate(rate::T, area, volume) where {T}
    return iszero(area) ? T(Inf) : rate / (2area * volume)
end

"""
Relative slack between a bound and the step accepted by the checks.

`advect!` and `step!` accept a step up to `1 + _wy_step_slack(T)` times the
returned bound. The frozen subface sweep widens its area tolerance by twice
that slack, so a step the checks accept is not rejected there.
"""
@inline _wy_step_slack(::Type{T}) where {T} = T(1.0e-6) + T(32) * eps(T)

"""
Absolute strip-area floor of a cut-donor bound, as a cell fraction.

A face (or phase subface) aperture below it carries no flux the sweep can
resolve, so the geometric strip constraints skip it.
"""
@inline _wy_strip_area_tolerance(::Type{T}) where {T} = T(256) * eps(T)

"""Floor of a strip's open area behind a face with aperture `aperture`."""
@inline _wy_strip_floor(::Type{T}, aperture) where {T} =
    T(256) * eps(T) * aperture

"""
Error for an outgoing face whose open donor strip cannot hold its flux.

A positive flux through a strip with no open area, or with an open area below
the sweep's round-off area tolerance, cannot be represented by any donating
region: no usable time step exists. A strip that is merely small, but
resolvable, slows the step proportionally and is bounded normally.
"""
@inline function _wy_strip_error(face_index, face, donor, phase::Int, area, floor)
    carrier = phase == 1 ? ", liquid subface" : phase == 2 ? ", gas subface" : ""
    return ArgumentError(
        "cut-donor strip on face $face_index (axis $(face.axis)$carrier) is empty: " *
            "the open area of the strip behind this face in cell " *
            "(block $(donor.block), i $(donor.i), j $(donor.j)) is $area, at most " *
            "the sweep's resolvable $floor, so no positive WY time step can " *
            "represent its flux",
    )
end

"""
Cut-donor strip rate for one outgoing face, validated against the sweep.

The strip's open area must exceed the sweep's area tolerance, scaled by the
face aperture; otherwise the prescribed flux has no donating region.
"""
@inline function _wy_cut_strip_rate(
        rate::T, area::T, aperture, volume, face_index, face, donor, phase::Int,
    ) where {T}
    floor = _wy_strip_floor(T, aperture)
    area > floor || throw(_wy_strip_error(face_index, face, donor, phase, area, floor))
    return _wy_strip_rate(rate, area, volume)
end

"""Strip rate of one side of an opposing pair, if that side is resolvable."""
@inline function _wy_side_strip_rate(rate::T, area::T, aperture, volume) where {T}
    area > _wy_strip_floor(T, aperture) || return zero(T)
    return _wy_strip_rate(rate, area, volume)
end

"""
    stable_timestep(velocity, work::WYAdvectionWorkspace; cfl=0.5)

WY incoming-volume CFL: `dt*sum(max(-outward_volume_rate,0))/cell_volume`.
Also constrain outgoing volume and each face extrusion. With an embedded
solid, limit each outgoing flux by its donor's open strip and constrain pairs
of opposing strips to disjoint open regions, including hanging faces; a
positive flux whose strip has no resolvable open area raises an
`ArgumentError` naming the face and donor cell. `0 < cfl <= 0.5`.
"""
function stable_timestep(
        velocity::FaceVelocity, work::WYAdvectionWorkspace{T, M};
        cfl = 0.5,
    ) where {T, M}
    _check_wy_workspace(velocity, work)
    isfinite(cfl) && 0 < cfl <= 0.5 || throw(ArgumentError("WY requires 0 < cfl <= 0.5"))
    geometry = work.interface.geometry
    mesh = work.mesh
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
        if !isnothing(geometry) && !iszero(u)
            aperture = geometry.aperture[i]
            solid = geometry.solid
            # A face at round-off open extent carries no resolvable flux.
            if aperture > _wy_strip_area_tolerance(T) &&
                    _get(solid.fraction, donor) > zero(T)
                lo, hi = _face_local_tangent_interval(mesh, donor, face)
                volume = mesh.blocks[donor.block].volume
                rate = abs(u) * face.area * aperture
                area = _wy_open_strip_area(solid, donor, face.axis, lo, hi, -0.5, 0.5)
                maxrate = max(
                    maxrate,
                    _wy_cut_strip_rate(rate, area, aperture, volume, i, face, donor, 0),
                )

                # Visit each pair once, from the positive (right/top) face.
                # Both rectangles use the same open solid half-plane, so
                # separating their open parts suffices for boundedness.
                if u > zero(T)
                    cell = (donor.block - 1) * cells_per_block + (donor.j - 1) * nx + donor.i
                    for pointer in work.cell_face_offsets[cell]:(work.cell_face_offsets[cell + 1] - 1)
                        other_index = work.cell_faces[pointer]
                        other = mesh.faces[other_index]
                        other.axis == face.axis && other.plus == donor || continue
                        other_u = velocity.interior[other_index]
                        other_u < zero(T) || continue
                        other_aperture = geometry.aperture[other_index]
                        other_aperture > _wy_strip_area_tolerance(T) || continue
                        other_lo, other_hi = _face_local_tangent_interval(mesh, donor, other)
                        areas = _wy_opposing_strip_areas(
                            solid, donor, face.axis, other_lo, other_hi, lo, hi,
                        )
                        isnothing(areas) && continue
                        left_area, right_area = areas
                        other_rate = abs(other_u) * other.area * other_aperture
                        maxrate = max(
                            maxrate,
                            _wy_side_strip_rate(
                                other_rate, left_area, other_aperture, volume,
                            ),
                            _wy_side_strip_rate(
                                rate, right_area, aperture, volume,
                            ),
                        )
                    end
                end
            end
        end
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
