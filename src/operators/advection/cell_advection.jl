"""A two-component velocity stored on owned coordinate-normal mesh faces."""
struct FaceVelocity{T, M <: BlockTreeMesh} <: AbstractMeshField{T}
    mesh::M
    interior::Vector{T}
    boundary::Vector{T}
end

Base.eltype(::Type{FaceVelocity{T, M}}) where {T, M} = T
Base.eltype(::FaceVelocity{T}) where {T} = T

"""
    FaceVelocity(mesh, [T=eltype(mesh.origin)]; init=zero(T))
    FaceVelocity(mesh, f; time=0, T=eltype(mesh.origin))

Create an owned coordinate-normal face velocity. The first form fills every
component with `init`; the second samples `f(x, y, time, axis)` on the immutable
face schedule. Coarse--fine faces have a single owner and a single value,
which is the conservation invariant used by advection.
"""
function FaceVelocity(
        mesh::M, ::Type{T} = eltype(mesh.origin);
        init = zero(T)
    ) where {M <: BlockTreeMesh, T}
    value = convert(T, init)
    return FaceVelocity{T, M}(
        mesh, fill(value, length(mesh.faces)),
        fill(value, length(mesh.boundary_faces))
    )
end

function FaceVelocity(mesh::M, f; time = 0, T = eltype(mesh.origin)) where {M <: BlockTreeMesh}
    interior = Vector{T}(undef, length(mesh.faces))
    boundary = Vector{T}(undef, length(mesh.boundary_faces))
    velocity = FaceVelocity{T, M}(mesh, interior, boundary)
    return set_velocities!(velocity, f; time)
end

function Base.copy(velocity::FaceVelocity{T, M}) where {T, M}
    return FaceVelocity{T, M}(velocity.mesh, copy(velocity.interior), copy(velocity.boundary))
end

"""Copy one directional component out of a `FaceVelocity`."""
function Base.copyto!(
        destination::FaceField{Axis},
        source::FaceVelocity
    ) where {Axis}
    destination.mesh === source.mesh ||
        throw(ArgumentError("destination and source must share a mesh"))
    local_index = 0
    for (global_index, face) in pairs(source.mesh.faces)
        face.axis == Axis || continue
        local_index += 1
        @inbounds destination.interior[local_index] = source.interior[global_index]
    end
    local_index = 0
    for (global_index, face) in pairs(source.mesh.boundary_faces)
        face.axis == Axis || continue
        local_index += 1
        @inbounds destination.boundary[local_index] = source.boundary[global_index]
    end
    return destination
end

"""Copy a directional `FaceField` into the corresponding velocity component."""
function Base.copyto!(
        destination::FaceVelocity,
        source::FaceField{Axis}
    ) where {Axis}
    destination.mesh === source.mesh ||
        throw(ArgumentError("destination and source must share a mesh"))
    local_index = 0
    for (global_index, face) in pairs(destination.mesh.faces)
        face.axis == Axis || continue
        local_index += 1
        @inbounds destination.interior[global_index] = source.interior[local_index]
    end
    local_index = 0
    for (global_index, face) in pairs(destination.mesh.boundary_faces)
        face.axis == Axis || continue
        local_index += 1
        @inbounds destination.boundary[global_index] = source.boundary[local_index]
    end
    return destination
end

function FaceField{Axis}(velocity::FaceVelocity{T}) where {Axis, T}
    field = FaceField{Axis}(velocity.mesh, T)
    return copyto!(field, velocity)
end

"""Assemble an owned staggered velocity from its two directional components."""
function FaceVelocity(x::FaceField{1, TX, M}, y::FaceField{2, TY, M}) where {TX, TY, M}
    x.mesh === y.mesh || throw(ArgumentError("face components must share a mesh"))
    T = promote_type(TX, TY)
    velocity = FaceVelocity{T, M}(
        x.mesh, zeros(T, length(x.mesh.faces)), zeros(T, length(x.mesh.boundary_faces))
    )
    copyto!(velocity, x)
    copyto!(velocity, y)
    return velocity
end

"""Resample `f(x, y, time, axis)` into an existing `FaceVelocity`."""
function set_velocities!(velocity::FaceVelocity, f; time = 0)
    @_parallel for index in 1:length(velocity.mesh.faces)
        face = velocity.mesh.faces[index]
        @inbounds velocity.interior[index] = f(face.center[1], face.center[2], time, face.axis)
    end
    @_parallel for index in 1:length(velocity.mesh.boundary_faces)
        face = velocity.mesh.boundary_faces[index]
        @inbounds velocity.boundary[index] = f(face.center[1], face.center[2], time, face.axis)
    end
    return velocity
end

"""
    enforce_no_permeation!(velocity)

Set the coordinate-normal velocity on every physical domain face to zero.
Periodic directions have no physical boundary entries, so a mesh with any
non-periodic direction automatically receives the impermeable-wall condition
without affecting periodic seams.
"""
function enforce_no_permeation!(velocity::FaceVelocity{T}) where {T}
    fill!(velocity.boundary, zero(T))
    return velocity
end

"""Reusable flux, incidence, increment, and rate storage for cell advection."""
struct AdvectionWorkspace{T, M <: BlockTreeMesh}
    increment::CellField{T, M}
    rates::CellField{T, M}
    interior_flux::Vector{T}
    boundary_flux::Vector{T}
    cell_offsets::Vector{Int}
    face_indices::Vector{Int}
    face_signs::Vector{Int8}
    boundary_entries::BitVector
end

function AdvectionWorkspace(field::CellField{T, M}) where {T, M}
    mesh = field.mesh
    targets = Int[]
    face_indices = Int[]
    face_signs = Int8[]
    boundary_entries = BitVector()
    nx, ny = mesh.block_size
    for (index, face) in pairs(mesh.faces)
        push!(targets, _cell_index(face.minus, nx, ny))
        push!(face_indices, index)
        push!(face_signs, -1)
        push!(boundary_entries, false)
        push!(targets, _cell_index(face.plus, nx, ny))
        push!(face_indices, index)
        push!(face_signs, 1)
        push!(boundary_entries, false)
    end
    for (index, face) in pairs(mesh.boundary_faces)
        push!(targets, _cell_index(face.cell, nx, ny))
        push!(face_indices, index)
        push!(face_signs, -1)
        push!(boundary_entries, true)
    end
    offsets, permutation = _grouped_output_schedule(targets, ncells(mesh))
    return AdvectionWorkspace(
        CellField(mesh, T), CellField(mesh, T),
        zeros(T, length(mesh.faces)), zeros(T, length(mesh.boundary_faces)),
        offsets, face_indices[permutation], face_signs[permutation],
        boundary_entries[permutation],
    )
end

function _zero_interior!(field::CellField)
    nx, ny = field.mesh.block_size
    h = field.mesh.halo
    _parallel_block_columns(field.mesh) do b, first_j, last_j
        values = @inbounds field.blocks[b]
        @inbounds for j in first_j:last_j, i in 1:nx
            values[i + h, j + h] = zero(eltype(field))
        end
    end
    return field
end

"""
    divergence!(destination, velocity)

Compute the finite-volume divergence of an owned face-normal velocity field at
cell centres. The same face schedule and metric weights are used by scalar and
staggered transport.
"""
function divergence!(destination::CellField{T}, velocity::FaceVelocity) where {T}
    destination.mesh === velocity.mesh ||
        throw(ArgumentError("destination and velocity must share a mesh"))
    _zero_interior!(destination)
    mesh = destination.mesh
    for (index, face) in pairs(mesh.faces)
        @inbounds flux = velocity.interior[index] * face.area
        _add!(destination, flux / mesh.blocks[face.minus.block].volume, face.minus)
        _add!(destination, -flux / mesh.blocks[face.plus.block].volume, face.plus)
    end
    for (index, face) in pairs(mesh.boundary_faces)
        @inbounds flux = face.side * velocity.boundary[index] * face.area
        _add!(destination, flux / mesh.blocks[face.cell.block].volume, face.cell)
    end
    return destination
end

@inline function _inflow_value(::ZeroGradient, scalar, face, time)
    return _get(scalar, face.cell)
end

@inline function _inflow_value(bc::Dirichlet, scalar, face, time)
    return bc.value(face.center[1], face.center[2], time)
end

"""
    advect!(scalar, velocity, dt, workspace; boundary=ZeroGradient(), time=0)

Advance cell averages with forward Euler and a first-order upwind finite-volume
flux. Every interior face, including every coarse--fine face segment, is owned
exactly once and applies equal-and-opposite integrated fluxes to its two cells.
After workspace construction the update is allocation-free.
"""
function advect!(
        scalar::CellField{T}, velocity::FaceVelocity, dt,
        workspace::AdvectionWorkspace{T};
        boundary::AbstractBoundaryCondition = ZeroGradient(), time = 0
    ) where {T}
    scalar.mesh === velocity.mesh === workspace.increment.mesh ||
        throw(ArgumentError("scalar, velocity, and workspace must share a mesh"))
    increment = workspace.increment
    mesh = scalar.mesh
    @_parallel for index in 1:length(mesh.faces)
        face = mesh.faces[index]
        @inbounds speed = velocity.interior[index]
        state = speed >= 0 ? _get(scalar, face.minus) : _get(scalar, face.plus)
        @inbounds workspace.interior_flux[index] =
            dt * speed * state * face.area
    end
    @_parallel for index in 1:length(mesh.boundary_faces)
        face = mesh.boundary_faces[index]
        @inbounds coordinate_speed = velocity.boundary[index]
        outward_speed = face.side * coordinate_speed
        state = outward_speed >= 0 ? _get(scalar, face.cell) :
            _inflow_value(boundary, scalar, face, time)
        @inbounds workspace.boundary_flux[index] =
            dt * outward_speed * state * face.area
    end
    nx, ny = mesh.block_size
    h = mesh.halo
    cells_per_block = nx * ny
    _parallel_block_columns(mesh) do b, first_j, last_j
        increment_block = @inbounds increment.blocks[b]
        scalar_block = @inbounds scalar.blocks[b]
        volume = mesh.blocks[b].volume
        cell_offset = (b - 1) * cells_per_block
        @inbounds for j in first_j:last_j, i in 1:nx
            cell = cell_offset + (j - 1) * nx + i
            integrated = zero(T)
            for pointer in workspace.cell_offsets[cell]:(
                    workspace.cell_offsets[cell + 1] - 1
                )
                face_index = workspace.face_indices[pointer]
                flux = workspace.boundary_entries[pointer] ?
                    workspace.boundary_flux[face_index] :
                    workspace.interior_flux[face_index]
                integrated += workspace.face_signs[pointer] * flux
            end
            change = integrated / volume
            increment_block[i + h, j + h] = change
            scalar_block[i + h, j + h] += change
        end
    end
    return scalar
end

"""
    stable_timestep(velocity, workspace; cfl=0.9)

Return a conservative multidimensional upwind CFL bound. Absolute face rates
are accumulated, making the result safe for arbitrary prescribed velocities.
"""
function stable_timestep(velocity::FaceVelocity{TV}, workspace::AdvectionWorkspace{TW}; cfl = 0.9) where {TV, TW}
    velocity.mesh === workspace.rates.mesh ||
        throw(ArgumentError("velocity and workspace must share a mesh"))
    T = promote_type(TV, TW, typeof(cfl))
    zero(T) < cfl <= one(T) || throw(ArgumentError("cfl must lie in (0, 1]"))
    rates = workspace.rates
    mesh = velocity.mesh
    @_parallel for index in 1:length(mesh.faces)
        face = mesh.faces[index]
        @inbounds workspace.interior_flux[index] =
            abs(velocity.interior[index]) * face.area
    end
    @_parallel for index in 1:length(mesh.boundary_faces)
        face = mesh.boundary_faces[index]
        @inbounds workspace.boundary_flux[index] =
            abs(velocity.boundary[index]) * face.area
    end
    nx, ny = mesh.block_size
    h = mesh.halo
    cells_per_block = nx * ny
    _parallel_block_columns(mesh) do b, first_j, last_j
        rates_block = @inbounds rates.blocks[b]
        volume = mesh.blocks[b].volume
        cell_offset = (b - 1) * cells_per_block
        @inbounds for j in first_j:last_j, i in 1:nx
            cell = cell_offset + (j - 1) * nx + i
            integrated = zero(T)
            for pointer in workspace.cell_offsets[cell]:(
                    workspace.cell_offsets[cell + 1] - 1
                )
                face_index = workspace.face_indices[pointer]
                integrated += workspace.boundary_entries[pointer] ?
                    workspace.boundary_flux[face_index] :
                    workspace.interior_flux[face_index]
            end
            rates_block[i + h, j + h] = integrated / volume
        end
    end
    maxrate = zero(T)
    for b in eachindex(mesh.blocks)
        rates_block = @inbounds rates.blocks[b]
        @inbounds for j in 1:ny, i in 1:nx
            maxrate = max(maxrate, rates_block[i + h, j + h])
        end
    end
    return iszero(maxrate) ? T(Inf) : T(cfl) / maxrate
end
