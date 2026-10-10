module CachePipeline

using Base.Threads
using Polyester
import UniformLWAdvection as Toy

export PipelineWorkspace, step!, simulate!, track_workers!, active_workers, memory_bytes

const Tile = NTuple{4,Int}

mutable struct TileScratch
    first::Matrix{Float64}
    second::Matrix{Float64}
    yindex::Vector{Int}
end

"""
    PipelineWorkspace(state; tile=(128,32), batches_per_worker=4, threaded=true)

One complete constant-carrier `x(dt/2), y(dt), x(dt/2)` step per output tile.
The input `state.q` remains frozen until all tiles have written the independent
`next` array, then a parallel copy publishes the complete step to the same
public `state.q` object. Each logical chunk owns its own halo scratch, so task
migration and worker scheduling cannot share scratch accidentally.
"""
mutable struct PipelineWorkspace
    next::Matrix{Float64}
    tiles::Vector{Tile}
    chunks::Vector{UnitRange{Int}}
    scratch::Vector{TileScratch}
    threaded::Bool
    record_workers::Bool
    worker_ids::Matrix{Int}
end

function PipelineWorkspace(s::Toy.State; tile::Tuple{Integer,Integer}=(128,32),
                           batches_per_worker::Integer=4, threaded::Bool=true)
    all(>(0), tile) || throw(ArgumentError("tile dimensions must be positive"))
    batches_per_worker > 0 || throw(ArgumentError("batch count must be positive"))
    nx, ny = size(s.q)
    tiles = Tile[]
    for j in 1:tile[2]:ny, i in 1:tile[1]:nx
        push!(tiles, (i,min(nx,i+tile[1]-1),j,min(ny,j+tile[2]-1)))
    end
    nchunks = min(length(tiles), batches_per_worker*Threads.nthreads(:default))
    chunks = [((div((k-1)*length(tiles),nchunks)+1):
               div(k*length(tiles),nchunks)) for k in 1:nchunks]
    scratch = [TileScratch(Matrix{Float64}(undef,tile[1]+2,tile[2]+2),
                           Matrix{Float64}(undef,tile[1]+2,tile[2]),
                           Vector{Int}(undef,tile[2]+2)) for _ in chunks]
    PipelineWorkspace(similar(s.q),tiles,chunks,scratch,threaded,false,
                      zeros(Int,2,nchunks))
end

function validate_step(s::Toy.State, w::PipelineWorkspace, dt::Float64)
    isfinite(dt) && dt >= 0 || throw(ArgumentError("timestep must be finite and nonnegative"))
    s.constant_carrier || throw(ArgumentError("pipeline prototype requires a constant carrier"))
    dims = size(s.q)
    dims == size(w.next) == size(s.u) == size(s.v) ||
        throw(DimensionMismatch("state and workspace shapes must agree"))
    arrays = (s.q,s.u,s.v,w.next)
    for i in 1:length(arrays), j in i+1:length(arrays)
        arrays[i] === arrays[j] && throw(ArgumentError("state and output arrays must be distinct"))
    end
    nx, ny = dims
    abs(s.u_speed)*dt*nx*0.5 <= 1 && abs(s.v_speed)*dt*ny <= 1 ||
        throw(ArgumentError("a directional Courant number exceeds one"))
    nothing
end

@inline function process_tile!(s::Toy.State, next::Matrix{Float64},
                               scratch::TileScratch, tile::Tile,
                               rx::Float64, ry::Float64)
    ilo, ihi, jlo, jhi = tile
    nx, ny = size(s.q)
    width, height = ihi-ilo+1, jhi-jlo+1
    first, second = scratch.first, scratch.second
    yj = scratch.yindex
    @inbounds for l in 1:height+2
        yj[l] = mod1(jlo+l-2,ny)
    end

    # Only the few k values whose *input* neighborhood crosses the x seam
    # need wrapped scalar work. The middle of every tile stays unit-stride.
    first_lo = max(1,4-ilo)
    first_hi = min(width+2,nx+1-ilo)
    first_right = max(first_lo,first_hi+1)
    y_lo = max(1,3-ilo)
    y_hi = min(width+2,nx+2-ilo)
    y_right = max(y_lo,y_hi+1)

    # First x intermediate: input footprint is output x halo 2, y halo 1.
    @inbounds for l in 1:height+2
        gj = yj[l]
        for k in 1:first_lo-1
            gi = mod1(ilo+k-2,nx)
            gim = gi == 1 ? nx : gi-1
            gip = gi == nx ? 1 : gi+1
            first[k,l] = Toy.xcell(s.q,s.u,gi,gj,gim,gip,rx)
        end
        @simd for k in first_lo:first_hi
            gi = ilo+k-2
            first[k,l] = Toy.xcell(s.q,s.u,gi,gj,gi-1,gi+1,rx)
        end
        for k in first_right:width+2
            gi = mod1(ilo+k-2,nx)
            gim = gi == 1 ? nx : gi-1
            gip = gi == nx ? 1 : gi+1
            first[k,l] = Toy.xcell(s.q,s.u,gi,gj,gim,gip,rx)
        end
    end

    # Y intermediate: output x halo 1, no y halo. The same physical face
    # orientation and Float64 operation order as the frozen toy are used.
    @inbounds for l in 1:height
        gj = yj[l+1]
        gjm = yj[l]
        for k in 1:y_lo-1
            gi = mod1(ilo+k-2,nx)
            qi = first[k,l+1]
            ft = Toy.lwflux(qi,first[k,l+2],s.v[gi,gj],ry)
            fb = Toy.lwflux(first[k,l],qi,s.v[gi,gjm],ry)
            second[k,l] = qi - ry*(ft-fb)
        end
        @simd for k in y_lo:y_hi
            gi = ilo+k-2
            qi = first[k,l+1]
            ft = Toy.lwflux(qi,first[k,l+2],s.v[gi,gj],ry)
            fb = Toy.lwflux(first[k,l],qi,s.v[gi,gjm],ry)
            second[k,l] = qi - ry*(ft-fb)
        end
        for k in y_right:width+2
            gi = mod1(ilo+k-2,nx)
            qi = first[k,l+1]
            ft = Toy.lwflux(qi,first[k,l+2],s.v[gi,gj],ry)
            fb = Toy.lwflux(first[k,l],qi,s.v[gi,gjm],ry)
            second[k,l] = qi - ry*(ft-fb)
        end
    end

    # Final x update owns only physical output cells; neighboring tiles write
    # no overlapping output and do not read one another's partial results.
    @inbounds for l in 1:height
        gj = yj[l+1]
        if ilo == 1
            qi = second[2,l]
            fr = Toy.lwflux(qi,second[3,l],s.u[1,gj],rx)
            fl = Toy.lwflux(second[1,l],qi,s.u[nx,gj],rx)
            next[1,gj] = qi - rx*(fr-fl)
        end
        @simd for k in (ilo == 1 ? 2 : 1):width
            gi = ilo+k-1
            qi = second[k+1,l]
            fr = Toy.lwflux(qi,second[k+2,l],s.u[gi,gj],rx)
            fl = Toy.lwflux(second[k,l],qi,s.u[gi-1,gj],rx)
            next[gi,gj] = qi - rx*(fr-fl)
        end
    end
    nothing
end

@inline function process_chunk!(s,w,c,rx,ry)
    scratch = @inbounds w.scratch[c]
    for t in @inbounds w.chunks[c]
        process_tile!(s,w.next,scratch,@inbounds(w.tiles[t]),rx,ry)
    end
    if w.record_workers
        @inbounds w.worker_ids[1,c] = Threads.threadid()
    end
    nothing
end

@inline function copy_chunk!(s,w,c)
    @inbounds for t in w.chunks[c]
        ilo,ihi,jlo,jhi = w.tiles[t]
        for j in jlo:jhi
            @simd for i in ilo:ihi
                s.q[i,j] = w.next[i,j]
            end
        end
    end
    if w.record_workers
        @inbounds w.worker_ids[2,c] = Threads.threadid()
    end
    nothing
end

function step!(s::Toy.State, w::PipelineWorkspace, dt::Real)
    delta = Float64(dt)
    validate_step(s,w,delta)
    nx, ny = size(s.q)
    rx, ry = delta*nx*0.5, delta*ny
    if w.threaded && Threads.nthreads(:default) > 1
        Polyester.@batch per=thread minbatch=1 for c in eachindex(w.chunks)
            process_chunk!(s,w,c,rx,ry)
        end
        Polyester.@batch per=thread minbatch=1 for c in eachindex(w.chunks)
            copy_chunk!(s,w,c)
        end
    else
        for c in eachindex(w.chunks)
            process_chunk!(s,w,c,rx,ry)
        end
        for c in eachindex(w.chunks)
            copy_chunk!(s,w,c)
        end
    end
    s.t += delta
    s
end

function simulate!(s::Toy.State,w::PipelineWorkspace,dt::Real,nsteps::Integer)
    nsteps >= 0 || throw(ArgumentError("step count must be nonnegative"))
    for _ in 1:nsteps
        step!(s,w,dt)
    end
    s
end

function track_workers!(w::PipelineWorkspace, enabled::Bool=true)
    fill!(w.worker_ids,0)
    w.record_workers = enabled
    w
end
function active_workers(w::PipelineWorkspace, phase::Symbol=:compute)
    row = phase === :compute ? 1 : phase === :copyback ? 2 :
        throw(ArgumentError("unknown phase $phase"))
    sort!(unique(filter(!iszero,@view(w.worker_ids[row,:]))))
end
memory_bytes(s::Toy.State,w::PipelineWorkspace) =
    sizeof(s.q)+sizeof(s.u)+sizeof(s.v)+sizeof(w.next)+
    sum(x -> sizeof(x.first)+sizeof(x.second)+sizeof(x.yindex),w.scratch)

end
