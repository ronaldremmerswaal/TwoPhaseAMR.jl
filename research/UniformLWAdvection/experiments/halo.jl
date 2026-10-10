module UniformLWHaloExperiment

using UniformLWAdvection
using Base.Threads

export PaddedState, HaloWorkspace, padded_step!, padded_simulate!, padded_mass,
       padded_memory_bytes, halo_step!, halo_simulate!, halo_values,
       halo_memory_bytes

const Tile = NTuple{4,Int}

"""Constant-carrier state with one periodic halo around every physical array."""
mutable struct PaddedState
    q::Matrix{Float64}
    u::Matrix{Float64}
    v::Matrix{Float64}
    nx::Int
    ny::Int
    u_speed::Float64
    v_speed::Float64
    t::Float64
end

function refresh_halo!(a::Matrix{Float64},nx,ny)
    @inbounds for j in 2:ny+1
        a[1,j] = a[nx+1,j]
        a[nx+2,j] = a[2,j]
    end
    @inbounds for i in 1:nx+2
        a[i,1] = a[i,ny+1]
        a[i,ny+2] = a[i,2]
    end
    nothing
end

"""Copy a validated core state once, outside a timed step."""
function PaddedState(s::UniformLWAdvection.State)
    s.constant_carrier || throw(ArgumentError("halo prototype requires a constant carrier"))
    nx,ny = size(s.q)
    size(s.u) == size(s.v) == (nx,ny) ||
        throw(DimensionMismatch("state arrays must have identical shapes"))
    q,u,v = (Matrix{Float64}(undef,nx+2,ny+2) for _ in 1:3)
    @inbounds for j in 1:ny, i in 1:nx
        q[i+1,j+1] = s.q[i,j]
        u[i+1,j+1] = s.u[i,j]
        v[i+1,j+1] = s.v[i,j]
    end
    refresh_halo!(u,nx,ny)
    refresh_halo!(v,nx,ny)
    PaddedState(q,u,v,nx,ny,s.u_speed,s.v_speed,s.t)
end

struct HaloWorkspace{B<:UniformLWAdvection.Backend}
    state::PaddedState
    first::Matrix{Float64}
    second::Matrix{Float64}
    backend::B
    tiles::Vector{Tile}
    chunks::Vector{UnitRange{Int}}
    threshold::Int
end

function HaloWorkspace(s::PaddedState;backend::Symbol=:serial,
                       tile::Tuple{Integer,Integer}=(128,32),
                       batches_per_worker::Integer=4,threshold::Integer=16384)
    backend in (:serial,:threads,:polyester) ||
        throw(ArgumentError("unknown backend $backend"))
    all(>(0),tile) || throw(ArgumentError("tile dimensions must be positive"))
    batches_per_worker > 0 || throw(ArgumentError("batch count must be positive"))
    threshold >= 0 || throw(ArgumentError("threshold must be nonnegative"))
    b = backend === :serial ? UniformLWAdvection.SerialBackend() :
        backend === :threads ? UniformLWAdvection.ThreadsBackend() :
        UniformLWAdvection.PolyesterBackend()
    nx,ny = s.nx,s.ny
    tx,ty = Int.(tile)
    tiles = Tile[]
    for j in 1:ty:ny, i in 1:tx:nx
        push!(tiles,(i,min(nx,i+tx-1),j,min(ny,j+ty-1)))
    end
    nt = length(tiles)
    nc = min(nt,Int(batches_per_worker)*Threads.nthreads(:default))
    chunks = [(div((c-1)*nt,nc)+1):div(c*nt,nc) for c in 1:nc]
    HaloWorkspace(s,similar(s.q),similar(s.q),b,tiles,chunks,Int(threshold))
end

"Numerical storage in five padded grids, excluding scheduler metadata."
padded_memory_bytes(s::PaddedState,w::HaloWorkspace) =
    sizeof(s.q)+sizeof(s.u)+sizeof(s.v)+sizeof(w.first)+sizeof(w.second)

"View of physical cell averages, suitable for resetting outside timed steps."
halo_values(w::HaloWorkspace) =
    @view w.state.q[2:w.state.nx+1,2:w.state.ny+1]

halo_memory_bytes(w::HaloWorkspace) = padded_memory_bytes(w.state,w)

padded_mass(s::PaddedState) =
    sum(@view s.q[2:s.nx+1,2:s.ny+1])/(s.nx*s.ny)

@inline function sweep_tile!(out,q,velocity,ratio,::Val{1},tile::Tile)
    ilo,ihi,jlo,jhi = tile
    for jj in jlo:jhi
        j = jj+1
        @inbounds @simd for ii in ilo:ihi
            i = ii+1
            qi = q[i,j]
            fr = UniformLWAdvection.lwflux(qi,q[i+1,j],velocity[i,j],ratio)
            fl = UniformLWAdvection.lwflux(q[i-1,j],qi,velocity[i-1,j],ratio)
            out[i,j] = qi-ratio*(fr-fl)
        end
    end
    nothing
end

@inline function sweep_tile!(out,q,velocity,ratio,::Val{2},tile::Tile)
    ilo,ihi,jlo,jhi = tile
    for jj in jlo:jhi
        j = jj+1
        @inbounds @simd for ii in ilo:ihi
            i = ii+1
            qi = q[i,j]
            ft = UniformLWAdvection.lwflux(qi,q[i,j+1],velocity[i,j],ratio)
            fb = UniformLWAdvection.lwflux(q[i,j-1],qi,velocity[i,j-1],ratio)
            out[i,j] = qi-ratio*(ft-fb)
        end
    end
    nothing
end

@inline function padded_sweep!(out,q,velocity,ratio,direction,s,w)
    refresh_halo!(q,s.nx,s.ny)
    serial = w.backend isa UniformLWAdvection.SerialBackend ||
             Threads.nthreads(:default) == 1 || s.nx*s.ny < w.threshold
    if serial
        sweep_tile!(out,q,velocity,ratio,direction,(1,s.nx,1,s.ny))
    else
        UniformLWAdvection.foreach_tile!(
            t -> sweep_tile!(out,q,velocity,ratio,direction,@inbounds(w.tiles[t])),
            w.backend,w.tiles,w.chunks,
        )
    end
    nothing
end

"Three synchronized Strang sweeps; each source halo costs O(nx+ny) to refresh."
function padded_step!(s::PaddedState,w::HaloWorkspace,dt::Real)
    delta = Float64(dt)
    s === w.state || throw(ArgumentError("workspace belongs to another state"))
    isfinite(delta) && delta >= 0 ||
        throw(ArgumentError("timestep must be finite and nonnegative"))
    dims = (s.nx+2,s.ny+2)
    size(s.q) == size(s.u) == size(s.v) == size(w.first) == size(w.second) == dims ||
        throw(DimensionMismatch("padded arrays must have identical shapes"))
    s.q === s.u || s.q === s.v || s.u === s.v ||
        s.q === w.first || s.q === w.second || w.first === w.second ||
        any(a -> a === s.u || a === s.v,(w.first,w.second)) ?
        throw(ArgumentError("state, carrier, and output arrays must be distinct")) : nothing
    abs(s.u_speed)*delta*s.nx*0.5 <= 1 && abs(s.v_speed)*delta*s.ny <= 1 ||
        throw(ArgumentError("a directional Courant number exceeds one"))
    rx,ry = delta*s.nx*0.5,delta*s.ny
    padded_sweep!(w.first,s.q,s.u,rx,Val(1),s,w)
    padded_sweep!(w.second,w.first,s.v,ry,Val(2),s,w)
    padded_sweep!(s.q,w.second,s.u,rx,Val(1),s,w)
    s.t += delta
    s
end

function padded_simulate!(s::PaddedState,w::HaloWorkspace,dt::Real,nsteps::Integer)
    nsteps >= 0 || throw(ArgumentError("step count must be nonnegative"))
    for _ in 1:nsteps
        padded_step!(s,w,dt)
    end
    s
end

halo_step!(w::HaloWorkspace,dt::Real) = padded_step!(w.state,w,dt)
halo_simulate!(w::HaloWorkspace,dt::Real,nsteps::Integer) =
    padded_simulate!(w.state,w,dt,nsteps)

end
