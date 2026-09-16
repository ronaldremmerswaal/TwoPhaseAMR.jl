"""
    CellField(mesh, [T=eltype(mesh.origin)]; init=zero(T))
    CellField(mesh, f; T=eltype(mesh.origin))

A cell-centred scalar field stored as one dense haloed matrix per active
block. The function constructor samples `f(x, y)` in owned cells; halo values
are populated separately by `fill_halo!`.
"""
struct CellField{T, M <: BlockTreeMesh} <: AbstractMeshField{T}
    mesh::M
    blocks::Vector{Matrix{T}}
end

"""
    FaceField{Axis}(mesh, [T=eltype(mesh.origin)]; init=zero(T))
    FaceField(mesh, Face{Axis}(), [T=eltype(mesh.origin)]; init=zero(T))

A scalar field on the owned faces normal to `Axis`. Values are stored compactly
in the order of the corresponding entries in the mesh's immutable face
schedules. `interior` contains faces between two cells (including periodic
seams), while `boundary` contains physical-boundary faces.
"""
struct FaceField{Axis, T, M <: BlockTreeMesh} <: AbstractMeshField{T}
    mesh::M
    interior::Vector{T}
    boundary::Vector{T}

    function FaceField{Axis, T, M}(
            mesh::M, interior::Vector{T},
            boundary::Vector{T}
        ) where {Axis, T, M <: BlockTreeMesh}
        Face{Axis}() # Validate the directional placement.
        return new{Axis, T, M}(mesh, interior, boundary)
    end
end

"""Return the discrete placement associated with a mesh field."""
placement(::FaceField{Axis}) where {Axis} = Face{Axis}()
Base.eltype(::Type{FaceField{Axis, T, M}}) where {Axis, T, M} = T
Base.eltype(::FaceField{Axis, T}) where {Axis, T} = T

@inline _nfaces(mesh::BlockTreeMesh, axis::Integer) =
    count(face -> face.axis == axis, mesh.faces)
@inline _nboundary_faces(mesh::BlockTreeMesh, axis::Integer) =
    count(face -> face.axis == axis, mesh.boundary_faces)

function FaceField{Axis}(
        mesh::M, ::Type{T} = eltype(mesh.origin);
        init = zero(T)
    ) where {Axis, M <: BlockTreeMesh, T}
    Face{Axis}()
    interior = fill(convert(T, init), _nfaces(mesh, Axis))
    boundary = fill(convert(T, init), _nboundary_faces(mesh, Axis))
    return FaceField{Axis, T, M}(mesh, interior, boundary)
end

FaceField(mesh::BlockTreeMesh, ::Face{Axis}, args...; kwargs...) where {Axis} =
    FaceField{Axis}(mesh, args...; kwargs...)

function FaceField{Axis}(mesh::M, f; T = eltype(mesh.origin)) where {Axis, M <: BlockTreeMesh}
    field = FaceField{Axis}(mesh, T)
    return initialize!(field, f)
end

FaceField(mesh::BlockTreeMesh, ::Face{Axis}, f; kwargs...) where {Axis} =
    FaceField{Axis}(mesh, f; kwargs...)

placement(::CellField) = Cell()
Base.eltype(::Type{CellField{T, M}}) where {T, M} = T
Base.eltype(::CellField{T}) where {T} = T

function CellField(mesh::M, ::Type{T} = eltype(mesh.origin); init = zero(T)) where {M <: BlockTreeMesh, T}
    nx, ny = mesh.block_size
    h = mesh.halo
    blocks = [fill(convert(T, init), nx + 2h, ny + 2h) for _ in 1:nblocks(mesh)]
    return CellField{T, M}(mesh, blocks)
end

function CellField(mesh::M, f; T = eltype(mesh.origin)) where {M <: BlockTreeMesh}
    field = CellField(mesh, T)
    return initialize!(field, f)
end

"""Return the haloed dense array owned by one active block."""
@inline blockdata(field::CellField, block::Integer) = field.blocks[block]

"""Return a non-copying view of a block interior."""
@inline function interior(field::CellField, block::Integer)
    nx, ny = field.mesh.block_size
    h = field.mesh.halo
    return @view field.blocks[block][(h + 1):(h + nx), (h + 1):(h + ny)]
end

@inline function _get(field::CellField, ref::CellRef)
    h = field.mesh.halo
    return @inbounds field.blocks[ref.block][ref.i + h, ref.j + h]
end

@inline function _set!(field::CellField, value, ref::CellRef)
    h = field.mesh.halo
    @inbounds field.blocks[ref.block][ref.i + h, ref.j + h] = value
    return value
end

@inline function _add!(field::CellField, value, ref::CellRef)
    h = field.mesh.halo
    @inbounds field.blocks[ref.block][ref.i + h, ref.j + h] += value
    return value
end

function Base.copy(field::CellField{T}) where {T}
    result = CellField(field.mesh, T)
    for b in eachindex(field.blocks)
        copyto!(result.blocks[b], field.blocks[b])
    end
    return result
end

function Base.copy(field::FaceField{Axis, T}) where {Axis, T}
    return FaceField{Axis, T, typeof(field.mesh)}(
        field.mesh, copy(field.interior), copy(field.boundary)
    )
end

"""
    initialize!(field, f)

Set every interior cell to `f(x, y)`. Halos are untouched.
"""
function initialize!(field::CellField, f)
    nx, ny = field.mesh.block_size
    h = field.mesh.halo
    _parallel_block_columns(field.mesh) do b, first_j, last_j
        values = @inbounds field.blocks[b]
        block = @inbounds field.mesh.blocks[b]
        dx, dy = block.spacing
        @inbounds for j in first_j:last_j, i in 1:nx
            x = block.origin[1] + (i - 1 / 2) * dx
            y = block.origin[2] + (j - 1 / 2) * dy
            values[i + h, j + h] = f(x, y)
        end
    end
    return field
end

"""Set an owned directional face field to `f(x, y)`."""
function initialize!(field::FaceField{Axis}, f) where {Axis}
    local_index = 0
    for face in field.mesh.faces
        face.axis == Axis || continue
        local_index += 1
        @inbounds field.interior[local_index] = f(face.center[1], face.center[2])
    end
    local_index = 0
    for face in field.mesh.boundary_faces
        face.axis == Axis || continue
        local_index += 1
        @inbounds field.boundary[local_index] = f(face.center[1], face.center[2])
    end
    return field
end

"""Volume integral of a cell-average field over active leaves."""
function integrate(field::CellField)
    nx, ny = field.mesh.block_size
    h = field.mesh.halo
    result = zero(eltype(field))
    for (b, block) in pairs(field.mesh.blocks)
        values = @inbounds field.blocks[b]
        @inbounds for j in 1:ny, i in 1:nx
            result += values[i + h, j + h] * block.volume
        end
    end
    return result
end

"""Interface for physical boundary values used by cell-field halos."""
abstract type AbstractBoundaryCondition end

"""Copy the closest interior value into a physical-boundary halo."""
struct ZeroGradient <: AbstractBoundaryCondition end

"""A physical-boundary value supplied as `value(x, y, time)`."""
struct Dirichlet{F} <: AbstractBoundaryCondition
    value::F
end

@inline _boundary_value(::ZeroGradient, field, entry, time) =
    _get(field, field.mesh.halo_donors[entry.first_donor])
@inline _boundary_value(bc::Dirichlet, field, entry, time) =
    bc.value(entry.center[1], entry.center[2], time)

"""
    fill_halo!(field, boundary=ZeroGradient(); time=0)

Fill same-level, coarse--fine, periodic, and physical halos using the schedule
built with the mesh. Fine-to-coarse values are conservative area averages;
coarse-to-fine values use piecewise-constant prolongation.
"""
function fill_halo!(field::CellField, boundary::AbstractBoundaryCondition = ZeroGradient(); time = 0)
    mesh = field.mesh
    @_parallel for entry_index in 1:length(mesh.halo_entries)
        entry = mesh.halo_entries[entry_index]
        if entry.physical_boundary
            _set!(field, _boundary_value(boundary, field, entry, time), entry.destination)
        else
            value = zero(eltype(field))
            stop = entry.first_donor + entry.donor_count - 1
            for index in entry.first_donor:stop
                @inbounds value += mesh.halo_weights[index] * _get(field, mesh.halo_donors[index])
            end
            _set!(field, value, entry.destination)
        end
    end
    return field
end
