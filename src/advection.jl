struct FaceVelocity{T,M<:BlockTreeMesh}
    mesh::M
    interior::Vector{T}
    boundary::Vector{T}
end

"""
    FaceVelocity(mesh, f; time=0, T=Float64)

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

function set_velocities!(velocity::FaceVelocity, f; time=0)
    for (index, face) in pairs(velocity.mesh.faces)
        @inbounds velocity.interior[index] = f(face.center[1], face.center[2], time, face.axis)
    end
    for (index, face) in pairs(velocity.mesh.boundary_faces)
        @inbounds velocity.boundary[index] = f(face.center[1], face.center[2], time, face.axis)
    end
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
