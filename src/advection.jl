struct FaceVelocity{T,M<:BlockTreeMesh}
    mesh::M
    interior::Vector{T}
    boundary::Vector{T}
end

Base.eltype(::Type{FaceVelocity{T,M}}) where {T,M} = T
Base.eltype(::FaceVelocity{T}) where {T} = T

"""
    FaceVelocity(mesh, f; time=0, T=eltype(mesh.origin))

Sample a prescribed coordinate-normal face velocity `f(x, y, time, axis)` on
the immutable face schedule. Coarse--fine faces have a single owner and a
single sampled velocity, which is the conservation invariant used by advection.
"""
function FaceVelocity(mesh::M, f; time=0, T=eltype(mesh.origin)) where {M<:BlockTreeMesh}
    interior = Vector{T}(undef, length(mesh.faces))
    boundary = Vector{T}(undef, length(mesh.boundary_faces))
    velocity = FaceVelocity{T,M}(mesh, interior, boundary)
    return set_velocities!(velocity, f; time)
end

function Base.copy(velocity::FaceVelocity{T,M}) where {T,M}
    FaceVelocity{T,M}(velocity.mesh, copy(velocity.interior), copy(velocity.boundary))
end

"""Copy one directional component out of a `FaceVelocity`."""
function Base.copyto!(destination::FaceField{Axis},
                      source::FaceVelocity) where {Axis}
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
function Base.copyto!(destination::FaceVelocity,
                      source::FaceField{Axis}) where {Axis}
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

function FaceField{Axis}(velocity::FaceVelocity{T}) where {Axis,T}
    field = FaceField{Axis}(velocity.mesh, T)
    return copyto!(field, velocity)
end

"""Assemble an owned staggered velocity from its two directional components."""
function FaceVelocity(x::FaceField{1,TX,M}, y::FaceField{2,TY,M}) where {TX,TY,M}
    x.mesh === y.mesh || throw(ArgumentError("face components must share a mesh"))
    T = promote_type(TX, TY)
    velocity = FaceVelocity{T,M}(
        x.mesh, zeros(T, length(x.mesh.faces)), zeros(T, length(x.mesh.boundary_faces)))
    copyto!(velocity, x)
    copyto!(velocity, y)
    return velocity
end

function set_velocities!(velocity::FaceVelocity, f; time=0)
    for (index, face) in pairs(velocity.mesh.faces)
        @inbounds velocity.interior[index] = f(face.center[1], face.center[2], time, face.axis)
    end
    for (index, face) in pairs(velocity.mesh.boundary_faces)
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

struct AdvectionWorkspace{T,M<:BlockTreeMesh}
    increment::CellField{T,M}
    rates::CellField{T,M}
end

function AdvectionWorkspace(field::CellField{T,M}) where {T,M}
    AdvectionWorkspace(CellField(field.mesh, T), CellField(field.mesh, T))
end

function _zero_interior!(field::CellField)
    nx, ny = field.mesh.block_size
    h = field.mesh.halo
    for block in field.blocks, j in 1:ny, i in 1:nx
        @inbounds block[i + h, j + h] = zero(eltype(field))
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
    _get(scalar, face.cell)
end

@inline function _inflow_value(bc::Dirichlet, scalar, face, time)
    bc.value(face.center[1], face.center[2], time)
end

"""
    advect!(scalar, velocity, dt, workspace; boundary=ZeroGradient(), time=0)

Advance cell averages with forward Euler and a first-order upwind finite-volume
flux. Every interior face, including every coarse--fine face segment, is owned
exactly once and applies equal-and-opposite integrated fluxes to its two cells.
After workspace construction the update is allocation-free.
"""
function advect!(scalar::CellField{T}, velocity::FaceVelocity, dt,
                 workspace::AdvectionWorkspace{T};
                 boundary::AbstractBoundaryCondition=ZeroGradient(), time=0) where {T}
    scalar.mesh === velocity.mesh === workspace.increment.mesh ||
        throw(ArgumentError("scalar, velocity, and workspace must share a mesh"))
    increment = workspace.increment
    _zero_interior!(increment)
    mesh = scalar.mesh
    for (index, face) in pairs(mesh.faces)
        @inbounds speed = velocity.interior[index]
        state = speed >= 0 ? _get(scalar, face.minus) : _get(scalar, face.plus)
        transported = dt * speed * state * face.area
        minus_volume = mesh.blocks[face.minus.block].volume
        plus_volume = mesh.blocks[face.plus.block].volume
        _add!(increment, -transported / minus_volume, face.minus)
        _add!(increment, transported / plus_volume, face.plus)
    end
    for (index, face) in pairs(mesh.boundary_faces)
        @inbounds coordinate_speed = velocity.boundary[index]
        outward_speed = face.side * coordinate_speed
        state = outward_speed >= 0 ? _get(scalar, face.cell) :
                _inflow_value(boundary, scalar, face, time)
        transported = dt * outward_speed * state * face.area
        volume = mesh.blocks[face.cell.block].volume
        _add!(increment, -transported / volume, face.cell)
    end
    nx, ny = mesh.block_size
    h = mesh.halo
    for b in eachindex(scalar.blocks), j in 1:ny, i in 1:nx
        @inbounds scalar.blocks[b][i + h, j + h] += increment.blocks[b][i + h, j + h]
    end
    return scalar
end

"""
    stable_timestep(velocity, workspace; cfl=0.9)

Return a conservative multidimensional upwind CFL bound. Absolute face rates
are accumulated, making the result safe for arbitrary prescribed velocities.
"""
function stable_timestep(velocity::FaceVelocity{TV}, workspace::AdvectionWorkspace{TW}; cfl=0.9) where {TV,TW}
    velocity.mesh === workspace.rates.mesh ||
        throw(ArgumentError("velocity and workspace must share a mesh"))
    T = promote_type(TV, TW, typeof(cfl))
    zero(T) < cfl <= one(T) || throw(ArgumentError("cfl must lie in (0, 1]"))
    rates = workspace.rates
    _zero_interior!(rates)
    mesh = velocity.mesh
    for (index, face) in pairs(mesh.faces)
        contribution = abs(@inbounds velocity.interior[index]) * face.area
        _add!(rates, contribution / mesh.blocks[face.minus.block].volume, face.minus)
        _add!(rates, contribution / mesh.blocks[face.plus.block].volume, face.plus)
    end
    for (index, face) in pairs(mesh.boundary_faces)
        contribution = abs(@inbounds velocity.boundary[index]) * face.area
        _add!(rates, contribution / mesh.blocks[face.cell.block].volume, face.cell)
    end
    maxrate = zero(T)
    nx, ny = mesh.block_size
    h = mesh.halo
    for block in rates.blocks, j in 1:ny, i in 1:nx
        @inbounds maxrate = max(maxrate, block[i + h, j + h])
    end
    return iszero(maxrate) ? T(Inf) : T(cfl) / maxrate
end
