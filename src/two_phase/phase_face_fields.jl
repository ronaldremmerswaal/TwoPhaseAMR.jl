"""Volume-weighted interpolation from centred cells to staggered dual volumes."""
function _phase_face_average!(destination::FaceField{Axis}, source::CellField) where {Axis}
    destination.mesh === source.mesh || throw(ArgumentError("fields must share a mesh"))
    mesh = source.mesh
    index = 0
    for face in mesh.faces
        face.axis == Axis || continue
        index += 1
        hm = mesh.blocks[face.minus.block].spacing[Axis]
        hp = mesh.blocks[face.plus.block].spacing[Axis]
        destination.interior[index] = (hm * _get(source, face.minus) + hp * _get(source, face.plus)) / (hm + hp)
    end
    index = 0
    for face in mesh.boundary_faces
        face.axis == Axis || continue
        index += 1
        destination.boundary[index] = _get(source, face.cell)
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
    local_index = 0
    for face in mesh.faces
        face.axis == Axis || continue
        local_index += 1
        weighted = volume = zero(T)
        for ref in (face.minus, face.plus)
            tangent = 3 - Axis
            center = cellcenter(mesh, ref.block, ref.i, ref.j)
            spacing = mesh.blocks[ref.block].spacing[tangent]
            lo = _wrapped_delta(
                face.center[tangent] - face.area / 2 - center[tangent],
                mesh.extent[tangent], mesh.periodic[tangent],
            ) / spacing
            hi = lo + face.area / spacing
            bounds = Axis == 1 ? (-0.5, 0.5, lo, hi) :
                (lo, hi, -0.5, 0.5)
            contribution = T(0.5) * _open_subcell_area(
                geometry.solid, ref, bounds...,
            ) * mesh.blocks[ref.block].volume
            weighted += contribution * _get(source, ref)
            volume += contribution
        end
        @inbounds destination.interior[local_index] = volume > zero(T) ?
            weighted / volume : zero(T)
    end
    local_index = 0
    for face in mesh.boundary_faces
        face.axis == Axis || continue
        local_index += 1
        @inbounds destination.boundary[local_index] = _get(source, face.cell)
    end
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
    gas_fraction::Vector{T}
    predicted_gas::Vector{T}
    values::Vector{T}
    increment::Vector{T}
    rates::Vector{T}
end

function PhaseFaceAdvectionScratch(::Type{T}, count) where {T}
    return PhaseFaceAdvectionScratch(ntuple(_ -> zeros(T, count), 9)...)
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
    }
    interface::InterfaceField{T, M}
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
