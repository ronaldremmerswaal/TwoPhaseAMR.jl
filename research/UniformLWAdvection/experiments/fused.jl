module UniformLWFusedExperiment

using UniformLWAdvection
using Base.Threads

export FusedWorkspace, fused_step!, fused_simulate!, fused_memory_bytes

const Tile = NTuple{4,Int}

# Scratch belongs to a stable chunk index, never to a Julia thread id: tasks
# may migrate. Every chunk processes its own tile range sequentially.
struct ChunkScratch
    xhalf::Matrix{Float64}
    yfull::Matrix{Float64}
    xwrap::Vector{Int}
    ywrap::Vector{Int}
end

function ChunkScratch(tx::Int, ty::Int)
    ChunkScratch(Matrix{Float64}(undef,tx+2,ty+2),
                 Matrix{Float64}(undef,tx+2,ty),
                 Vector{Int}(undef,tx+4),Vector{Int}(undef,ty+2))
end

"""
    FusedWorkspace(state; backend=:threads, tile=(128,32),
                   batches_per_worker=4, threshold=16384)

Experimental constant-carrier Strang fusion. One `next` full-grid array holds
the output; each chunk owns bounded scratch for x-half and y-full values in
its output tile's dependency halo. A second synchronized chunk pass copies
`next` into the original `state.q`, preserving that array's identity. The
workspace is exclusive to one in-flight step, but tasks may migrate freely.
"""
struct FusedWorkspace{B<:UniformLWAdvection.Backend}
    state::UniformLWAdvection.State
    next::Matrix{Float64}
    scratch::Vector{ChunkScratch}
    backend::B
    tiles::Vector{Tile}
    chunks::Vector{UnitRange{Int}}
    task_ranges::Vector{UnitRange{Int}}
    threshold::Int
end

function FusedWorkspace(s::UniformLWAdvection.State;
                        backend::Symbol=:threads,
                        tile::Tuple{Integer,Integer}=(128,32),
                        batches_per_worker::Integer=4,
                        threshold::Integer=16384)
    backend in (:serial,:threads,:spawn,:polyester) ||
        throw(ArgumentError("unknown backend $backend"))
    all(>(0),tile) || throw(ArgumentError("tile dimensions must be positive"))
    batches_per_worker > 0 || throw(ArgumentError("batch count must be positive"))
    threshold >= 0 || throw(ArgumentError("threshold must be nonnegative"))
    b = backend === :serial ? UniformLWAdvection.SerialBackend() :
        backend === :threads ? UniformLWAdvection.ThreadsBackend() :
        backend === :spawn ? UniformLWAdvection.SpawnBackend() :
        UniformLWAdvection.PolyesterBackend()
    nx, ny = size(s.q)
    tx, ty = Int.(tile)
    tiles = Tile[]
    for j in 1:ty:ny, i in 1:tx:nx
        push!(tiles,(i,min(nx,i+tx-1),j,min(ny,j+ty-1)))
    end
    nt = length(tiles)
    nc = min(nt,Int(batches_per_worker)*Threads.nthreads(:default))
    chunks = [(div((c-1)*nt,nc)+1):div(c*nt,nc) for c in 1:nc]
    scratch = [ChunkScratch(min(tx,nx),min(ty,ny)) for _ in 1:nc]
    task_ranges = [c:c for c in 1:nc]
    FusedWorkspace(s,similar(s.q),scratch,b,tiles,chunks,task_ranges,
                   Int(threshold))
end

"Numerical-array and chunk-index bytes, excluding Julia object headers."
function fused_memory_bytes(s::UniformLWAdvection.State,w::FusedWorkspace)
    total = sizeof(s.q)+sizeof(s.u)+sizeof(s.v)+sizeof(w.next)
    if s.velocity_scratch !== nothing
        total += sizeof(s.velocity_scratch[1])+sizeof(s.velocity_scratch[2])
    end
    for scratch in w.scratch
        total += sizeof(scratch.xhalf)+sizeof(scratch.yfull)+
                 sizeof(scratch.xwrap)+sizeof(scratch.ywrap)
    end
    total
end

@inline function foreach_chunk!(f,w::FusedWorkspace)
    nx, ny = size(w.state.q)
    serial = w.backend isa UniformLWAdvection.SerialBackend ||
             Threads.nthreads(:default) == 1 || nx*ny < w.threshold
    backend = serial ? UniformLWAdvection.SerialBackend() : w.backend
    # `foreach_tile!` iterates chunk IDs here. For :spawn, each one-element
    # task range owns exactly one ChunkScratch, irrespective of task migration.
    UniformLWAdvection.foreach_tile!(f,backend,w.chunks,w.task_ranges)
    nothing
end

@inline function periodic_indices!(scratch::ChunkScratch,tile::Tile,nx,ny)
    ilo, ihi, jlo, jhi = tile
    width, height = ihi-ilo+1, jhi-jlo+1
    @inbounds for k in 1:width+4
        scratch.xwrap[k] = mod1(ilo+k-3,nx)
    end
    @inbounds for k in 1:height+2
        scratch.ywrap[k] = mod1(jlo+k-2,ny)
    end
    nothing
end

@inline flux(left,right,speed,ratio) =
    UniformLWAdvection.lwflux(left,right,speed,ratio)

function fused_tile!(s::UniformLWAdvection.State,w::FusedWorkspace,
                     scratch::ChunkScratch,tile::Tile,rx,ry)
    nx, ny = size(s.q)
    ilo, ihi, jlo, jhi = tile
    width, height = ihi-ilo+1, jhi-jlo+1
    periodic_indices!(scratch,tile,nx,ny)
    xx, yy = scratch.xwrap,scratch.ywrap
    q0, u, v = s.q,s.u,s.v
    q1, q2, out = scratch.xhalf,scratch.yfull,w.next

    # q1 covers the output tile plus one cell in x and y. Its reads reach
    # radius two in x and one in y in the original global q0.
    for lj in 1:height+2
        j = @inbounds yy[lj]
        @inbounds for li in 1:width+2
            im, i, ip = xx[li],xx[li+1],xx[li+2]
            qi = q0[i,j]
            fr = flux(qi,q0[ip,j],u[i,j],rx)
            fl = flux(q0[im,j],qi,u[im,j],rx)
            q1[li,lj] = qi-rx*(fr-fl)
        end
    end

    # q2 needs x-halo values for the final x-half sweep, but only the
    # output tile's y extent. All required q1 y-halo values are local.
    for lj in 1:height
        jm, j, jp = @inbounds yy[lj],yy[lj+1],yy[lj+2]
        @inbounds for li in 1:width+2
            i = xx[li+1]
            qi = q1[li,lj+1]
            ft = flux(qi,q1[li,lj+2],v[i,j],ry)
            fb = flux(q1[li,lj],qi,v[i,jm],ry)
            q2[li,lj] = qi-ry*(ft-fb)
        end
    end

    # The final output is uniquely owned by this tile. Recomputed halo
    # faces use the same positive orientation and operation order as the
    # unfused package kernel, so neighboring tiles obtain identical fluxes.
    for lj in 1:height
        j = jlo+lj-1
        @inbounds for li in 1:width
            im, i = xx[li+1],xx[li+2]
            qi = q2[li+1,lj]
            fr = flux(qi,q2[li+2,lj],u[i,j],rx)
            fl = flux(q2[li,lj],qi,u[im,j],rx)
            out[ilo+li-1,j] = qi-rx*(fr-fl)
        end
    end
    nothing
end

@inline function copy_tile!(destination,source,tile::Tile)
    ilo, ihi, jlo, jhi = tile
    @inbounds for j in jlo:jhi, i in ilo:ihi
        destination[i,j] = source[i,j]
    end
    nothing
end

function validate_fused_step(s::UniformLWAdvection.State,w::FusedWorkspace,dt)
    s === w.state || throw(ArgumentError("workspace belongs to another state"))
    s.constant_carrier || throw(ArgumentError("fused prototype requires a constant carrier"))
    isfinite(dt) && dt >= 0 ||
        throw(ArgumentError("timestep must be finite and nonnegative"))
    size(s.q) == size(s.u) == size(s.v) == size(w.next) ||
        throw(DimensionMismatch("state and workspace shapes differ"))
    w.next === s.q || s.q === s.u || s.q === s.v ||
        w.next === s.u || w.next === s.v ?
        throw(ArgumentError("state, carrier, and output arrays must be distinct")) : nothing
    nx, ny = size(s.q)
    abs(s.u_speed)*dt*nx*0.5 <= 1 && abs(s.v_speed)*dt*ny <= 1 ||
        throw(ArgumentError("a directional Courant number exceeds one"))
    nothing
end

"One fused, conservative Strang step with an explicit final copy-back barrier."
function fused_step!(s::UniformLWAdvection.State,w::FusedWorkspace,dt::Real)
    delta = Float64(dt)
    validate_fused_step(s,w,delta)
    nx, ny = size(s.q)
    rx, ry = delta*nx*0.5,delta*ny
    foreach_chunk!(w) do c
        scratch = @inbounds w.scratch[c]
        for t in w.chunks[c]
            fused_tile!(s,w,scratch,w.tiles[t],rx,ry)
        end
    end
    # Completion of all tile tasks above is the barrier before copy-back.
    foreach_chunk!(w) do c
        for t in w.chunks[c]
            copy_tile!(s.q,w.next,w.tiles[t])
        end
    end
    s.t += delta
    s
end

function fused_simulate!(s::UniformLWAdvection.State,w::FusedWorkspace,
                         dt::Real,nsteps::Integer)
    nsteps >= 0 || throw(ArgumentError("step count must be nonnegative"))
    for _ in 1:nsteps
        fused_step!(s,w,dt)
    end
    s
end

end
