struct CellField{T,M<:BlockTreeMesh}
    mesh::M
    blocks::Vector{Matrix{T}}
end

placement(::CellField) = Cell()
Base.eltype(::Type{CellField{T,M}}) where {T,M} = T
Base.eltype(::CellField{T}) where {T} = T

"""Allocate a haloed, dense array for every leaf block."""
function CellField(mesh::M, ::Type{T}=Float64; init=zero(T)) where {M<:BlockTreeMesh,T}
    nx, ny = mesh.block_size
    h = mesh.halo
    blocks = [fill(convert(T, init), nx + 2h, ny + 2h) for _ in 1:nblocks(mesh)]
    return CellField{T,M}(mesh, blocks)
end

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

"""
    initialize!(field, f)

Set every interior cell to `f(x, y)`. Halos are untouched.
"""
function initialize!(field::CellField, f)
    nx, ny = field.mesh.block_size
    h = field.mesh.halo
    for b in eachindex(field.blocks), j in 1:ny, i in 1:nx
        x, y = cellcenter(field.mesh, b, i, j)
        @inbounds field.blocks[b][i + h, j + h] = f(x, y)
    end
    return field
end

"""Volume integral of a cell-average field over active leaves."""
function integrate(field::CellField)
    nx, ny = field.mesh.block_size
    h = field.mesh.halo
    result = zero(eltype(field))
    for (b, block) in pairs(field.mesh.blocks), j in 1:ny, i in 1:nx
        @inbounds result += field.blocks[b][i + h, j + h] * block.volume
    end
    return result
end

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
function fill_halo!(field::CellField, boundary::AbstractBoundaryCondition=ZeroGradient(); time=0)
    mesh = field.mesh
    for entry in mesh.halo_entries
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
