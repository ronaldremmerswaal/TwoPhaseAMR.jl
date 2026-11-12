"""Volume-weighted interpolation from centred cells to staggered dual volumes.

The interior faces of `Axis` are a contiguous range of the axis-major
`mesh.faces` (`_axis_face_offset`); each face owns its output, so the threaded
loop is bitwise serial. The few boundary faces stay serial."""
function _phase_face_average!(destination::FaceField{Axis}, source::CellField) where {Axis}
    destination.mesh === source.mesh || throw(ArgumentError("fields must share a mesh"))
    mesh = source.mesh
    axis = Axis
    interior_values = destination.interior
    faces = mesh.faces
    leaf_blocks = mesh.blocks
    source_blocks = source.blocks
    halo = mesh.halo
    offset = _axis_face_offset(faces, axis, length(interior_values))
    @_parallel for local_index in 1:length(interior_values)
        face = @inbounds faces[offset + local_index]
        minus, plus = face.minus, face.plus
        hm = @inbounds leaf_blocks[minus.block].spacing[axis]
        hp = @inbounds leaf_blocks[plus.block].spacing[axis]
        @inbounds interior_values[local_index] = (
            hm * source_blocks[minus.block][minus.i + halo, minus.j + halo] +
                hp * source_blocks[plus.block][plus.i + halo, plus.j + halo]
        ) / (hm + hp)
    end
    _phase_face_boundary_average!(destination, source)
    return destination
end

function _phase_face_boundary_average!(destination::FaceField{Axis}, source::CellField) where {Axis}
    boundary_faces = source.mesh.boundary_faces
    offset = _axis_face_offset(boundary_faces, Axis, length(destination.boundary))
    for local_index in eachindex(destination.boundary)
        @inbounds destination.boundary[local_index] =
            _get(source, boundary_faces[offset + local_index].cell)
    end
    return destination
end

_phase_face_average!(destination, source, ::Nothing) =
    _phase_face_average!(destination, source)

function _phase_face_average!(
        destination::FaceField{Axis, T}, source::CellField,
        geometry::GeometryField,
    ) where {Axis, T}
    destination.mesh === source.mesh === geometry.mesh || throw(
        ArgumentError("fields and geometry must share a mesh"),
    )
    mesh = source.mesh
    interior_values = destination.interior
    faces = mesh.faces
    leaf_blocks = mesh.blocks
    extent, periodic = mesh.extent, mesh.periodic
    source_blocks = source.blocks
    halo = mesh.halo
    solid = geometry.solid
    offset = _axis_face_offset(faces, Axis, length(interior_values))
    # Captures the solid interface (one small box per threaded launch).
    @_parallel for local_index in 1:length(interior_values)
        face = @inbounds faces[offset + local_index]
        weighted = volume = zero(T)
        for ref in (face.minus, face.plus)
            lo, hi = _face_local_tangent_interval(leaf_blocks, extent, periodic, ref, face)
            bounds = Axis == 1 ? (-0.5, 0.5, lo, hi) :
                (lo, hi, -0.5, 0.5)
            contribution = T(0.5) * _open_subcell_area(solid, ref, bounds...) *
                @inbounds(leaf_blocks[ref.block].volume)
            weighted += contribution *
                @inbounds(source_blocks[ref.block][ref.i + halo, ref.j + halo])
            volume += contribution
        end
        @inbounds interior_values[local_index] = volume > zero(T) ?
            weighted / volume : zero(T)
    end
    _phase_face_boundary_average!(destination, source)
    return destination
end

"""
    PhaseFaceScalar{Axis}(interface; liquid=1, gas=1)

Intrinsic scalar values on staggered dual volumes, one value per phase. Pass
through the `staggered_scalars` keyword of geometric `advect!`. Conservative
phase-weighted products are formed only inside the update. Dual phase fluxes
are the algebraic interpolant of the centred WY volume fluxes.
"""
struct PhaseFaceAdvectionScratch{T}
    source_liquid::Vector{T}
    source_gas::Vector{T}
    liquid_fraction::Vector{T}
    predicted_liquid::Vector{T}
    values::Vector{T}
    increment::Vector{T}
    rates::Vector{T}
end

function PhaseFaceAdvectionScratch(::Type{T}, count) where {T}
    return PhaseFaceAdvectionScratch(ntuple(_ -> zeros(T, count), 7)...)
end

"""
    PhaseFaceScalar

Liquid/gas intrinsic scalar values on one face orientation, together with the
phase fraction and reusable staggered-advection scratch needed for compatible
two-velocity transport.
"""
mutable struct PhaseFaceScalar{
        Axis, T, M <: BlockTreeMesh, S,
        # W records the constructor's workspace type for compatible type
        # parameters; adaptation can change the current schedule in place.
        W <: StaggeredAdvectionWorkspace{Axis, T, M},
        F <: InterfaceField{T, M},
    }
    interface::F
    liquid::FaceField{Axis, T, M}
    gas::FaceField{Axis, T, M}
    fraction::FaceField{Axis, T, M}
    # Adaptation may switch between the uniform and AMR dual schedules while
    # the registered PhaseFaceScalar retains its identity. Keep the field a
    # bounded union of the two concrete workspace types so Julia can split
    # transport specializations without an abstract hot-loop field.
    workspace::Union{
        StaggeredAdvectionWorkspace{Axis, T, M, UniformDualAdvectionSchedule{T}},
        StaggeredAdvectionWorkspace{Axis, T, M, AMRDualAdvectionSchedule{T}},
    }
    scratch::S
end

"""Scratch for compatible advection of one stored velocity field.

All arrays contain velocity, phase fraction, or a velocity increment. Momentum
is formed only in local expressions and is never a persistent field.
"""
struct MomentumComponentWorkspace{Axis, T, M <: BlockTreeMesh, W}
    value::FaceField{Axis, T, M}
    source::FaceField{Axis, T, M}
    fraction::FaceField{Axis, T, M}
    predicted_fraction::FaceField{Axis, T, M}
    increment::FaceField{Axis, T, M}
    advection::W
    old_fraction::Vector{T}
    new_fraction::Vector{T}
    values::Vector{T}
    sources::Vector{T}
    delta::Vector{T}
    rates::Vector{T}
end

@inline function _pack_phase_faces!(destination, source::FaceField)
    interior_count = length(source.interior)
    copyto!(destination, 1, source.interior, 1, interior_count)
    copyto!(
        destination, interior_count + 1,
        source.boundary, 1, length(source.boundary)
    )
    return destination
end

@inline function _unpack_phase_faces!(destination::FaceField, source)
    interior_count = length(destination.interior)
    copyto!(
        destination.interior, 1, source, 1, interior_count
    )
    copyto!(
        destination.boundary, 1, source, interior_count + 1,
        length(destination.boundary)
    )
    return destination
end
