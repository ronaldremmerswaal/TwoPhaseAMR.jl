module UniformLWFusedAffineExperiment

using UniformLWAdvection
using Base.Threads
import UniformLWAdvection: active_workers

# Keep the frozen prototype's scheduling, validation, scratch ownership, and
# memory layout. Only the numerical tile kernel differs in this experiment.
include(joinpath(@__DIR__,"fused.jl"))
const BaseFused = UniformLWFusedExperiment
const FusedWorkspace = BaseFused.FusedWorkspace
const fused_memory_bytes = BaseFused.fused_memory_bytes

export FusedWorkspace, fused_step!, fused_simulate!,
       fused_simulate_amortized!, fused_memory_bytes, WorkerProbe,
       active_workers, active_copyback_workers

"Caller-owned worker IDs: one slot per chunk and compute pass."
struct WorkerProbe
    compute::Matrix{Int}
    copyback::Vector{Int}
end

function WorkerProbe(w::FusedWorkspace; nsteps::Integer=1)
    nsteps >= 0 || throw(ArgumentError("step count must be nonnegative"))
    nc = length(w.chunks)
    WorkerProbe(zeros(Int,nc,nsteps),zeros(Int,nc))
end

active_workers(probe::WorkerProbe,pass::Integer) =
    sort!(unique(filter(!iszero,@view probe.compute[:,pass])))
active_copyback_workers(probe::WorkerProbe) =
    sort!(unique(filter(!iszero,probe.copyback)))

function validate_probe(w::FusedWorkspace,probe::WorkerProbe,nsteps::Integer)
    size(probe.compute) == (length(w.chunks),nsteps) &&
        length(probe.copyback) == length(w.chunks) ||
        throw(DimensionMismatch("worker probe must match chunks and step count"))
    nothing
end

@inline flux(left,right,speed,ratio) =
    UniformLWAdvection.lwflux(left,right,speed,ratio)

@inline function xhalf_seam!(q1,q0,u,li,lj,ilo,j,nx,rx)
    i = mod1(ilo+li-2,nx)
    im,ip = i == 1 ? nx : i-1, i == nx ? 1 : i+1
    @inbounds begin
        qi = q0[i,j]
        fr = flux(qi,q0[ip,j],u[i,j],rx)
        fl = flux(q0[im,j],qi,u[im,j],rx)
        q1[li,lj] = qi-rx*(fr-fl)
    end
    nothing
end

@inline function yfull_seam!(q2,q1,v,li,lj,ilo,j,jm,nx,ry)
    i = mod1(ilo+li-2,nx)
    @inbounds begin
        qi = q1[li,lj+1]
        ft = flux(qi,q1[li,lj+2],v[i,j],ry)
        fb = flux(q1[li,lj],qi,v[i,jm],ry)
        q2[li,lj] = qi-ry*(ft-fb)
    end
    nothing
end

function affine_tile!(s::UniformLWAdvection.State,w::FusedWorkspace,
                      scratch::BaseFused.ChunkScratch,tile::BaseFused.Tile,
                      q0,out,rx,ry)
    nx,ny = size(s.q)
    ilo,ihi,jlo,jhi = tile
    width,height = ihi-ilo+1,jhi-jlo+1
    u,v = s.u,s.v
    q1,q2 = scratch.xhalf,scratch.yfull

    # q1 covers one x/y halo around the owned tile. The middle span has
    # affine q0/u reads, leaving only physical periodic seams to scalar code.
    xlo = clamp(4-ilo,1,width+3)
    xhi = clamp(nx+1-ilo,0,width+2)
    for lj in 1:height+2
        j = mod1(jlo+lj-2,ny)
        for li in 1:xlo-1
            xhalf_seam!(q1,q0,u,li,lj,ilo,j,nx,rx)
        end
        @inbounds @simd for li in xlo:xhi
            i = ilo+li-2
            qi = q0[i,j]
            fr = flux(qi,q0[i+1,j],u[i,j],rx)
            fl = flux(q0[i-1,j],qi,u[i-1,j],rx)
            q1[li,lj] = qi-rx*(fr-fl)
        end
        for li in max(xlo,xhi+1):width+2
            xhalf_seam!(q1,q0,u,li,lj,ilo,j,nx,rx)
        end
    end

    # q2 uses local q1 storage with affine x indices. Only the x carrier
    # lookup wraps at the physical seam; y wrapping is once per output row.
    ylo = clamp(3-ilo,1,width+3)
    yhi = clamp(nx+2-ilo,0,width+2)
    for lj in 1:height
        j = mod1(jlo+lj-1,ny)
        jm = j == 1 ? ny : j-1
        for li in 1:ylo-1
            yfull_seam!(q2,q1,v,li,lj,ilo,j,jm,nx,ry)
        end
        @inbounds @simd for li in ylo:yhi
            i = ilo+li-2
            qi = q1[li,lj+1]
            ft = flux(qi,q1[li,lj+2],v[i,j],ry)
            fb = flux(q1[li,lj],qi,v[i,jm],ry)
            q2[li,lj] = qi-ry*(ft-fb)
        end
        for li in max(ylo,yhi+1):width+2
            yfull_seam!(q2,q1,v,li,lj,ilo,j,jm,nx,ry)
        end
    end

    # q3 owns contiguous output cells. Only cell i=1 needs a wrapped left
    # face carrier; q2's local halo already provides both neighboring values.
    for lj in 1:height
        j = jlo+lj-1
        if ilo == 1
            @inbounds begin
                qi = q2[2,lj]
                fr = flux(qi,q2[3,lj],u[1,j],rx)
                fl = flux(q2[1,lj],qi,u[nx,j],rx)
                out[1,j] = qi-rx*(fr-fl)
            end
        end
        @inbounds @simd for li in (ilo == 1 ? 2 : 1):width
            i = ilo+li-1
            qi = q2[li+1,lj]
            fr = flux(qi,q2[li+2,lj],u[i,j],rx)
            fl = flux(q2[li,lj],qi,u[i-1,j],rx)
            out[i,j] = qi-rx*(fr-fl)
        end
    end
    nothing
end

# Force specialization on the closure; the frozen prototype's generic
# `foreach_chunk!` passes its function argument straight to a scheduler.
@inline function foreach_chunk!(f::F,w::FusedWorkspace) where F
    nx,ny = size(w.state.q)
    serial = w.backend isa UniformLWAdvection.SerialBackend ||
             Threads.nthreads(:default) == 1 || nx*ny < w.threshold
    backend = serial ? UniformLWAdvection.SerialBackend() : w.backend
    UniformLWAdvection.foreach_tile!(f,backend,w.chunks,w.task_ranges)
    nothing
end

@inline function compute_tiles!(s,w,q0,out,rx,ry)
    foreach_chunk!(w) do c
        scratch = @inbounds w.scratch[c]
        for t in w.chunks[c]
            affine_tile!(s,w,scratch,w.tiles[t],q0,out,rx,ry)
        end
    end
    nothing
end

@inline function compute_tiles!(s,w,q0,out,rx,ry,probe::WorkerProbe,pass::Int)
    foreach_chunk!(w) do c
        @inbounds probe.compute[c,pass] = Threads.threadid()
        scratch = @inbounds w.scratch[c]
        for t in w.chunks[c]
            affine_tile!(s,w,scratch,w.tiles[t],q0,out,rx,ry)
        end
    end
    nothing
end

@inline function copyback!(s,w,source)
    foreach_chunk!(w) do c
        for t in w.chunks[c]
            BaseFused.copy_tile!(s.q,source,w.tiles[t])
        end
    end
    nothing
end

@inline function copyback!(s,w,source,probe::WorkerProbe)
    foreach_chunk!(w) do c
        @inbounds probe.copyback[c] = Threads.threadid()
        for t in w.chunks[c]
            BaseFused.copy_tile!(s.q,source,w.tiles[t])
        end
    end
    nothing
end

# Polyester wraps arrays captured directly by @batch as pointer views. Doing
# the chunk loop inside the batch closure avoids boxing the eight tile-kernel
# arguments on every tile through a nested callback. Keep this optional so
# the standalone experiment loads without Polyester in its environment.
@static if Base.find_package("Polyester") !== nothing
    import Polyester
    include(joinpath(@__DIR__,"fused_affine_polyester.jl"))
end

function fused_step!(s::UniformLWAdvection.State,w::FusedWorkspace,dt::Real)
    delta = Float64(dt)
    BaseFused.validate_fused_step(s,w,delta)
    nx,ny = size(s.q)
    rx,ry = delta*nx*0.5,delta*ny
    compute_tiles!(s,w,s.q,w.next,rx,ry)
    copyback!(s,w,w.next)
    s.t += delta
    s
end

"Untimed observable step that records actual compute and copy-back workers."
function fused_step!(s::UniformLWAdvection.State,w::FusedWorkspace,
                     dt::Real,probe::WorkerProbe)
    delta = Float64(dt)
    BaseFused.validate_fused_step(s,w,delta)
    validate_probe(w,probe,1)
    fill!(probe.compute,0)
    fill!(probe.copyback,0)
    nx,ny = size(s.q)
    rx,ry = delta*nx*0.5,delta*ny
    compute_tiles!(s,w,s.q,w.next,rx,ry,probe,1)
    copyback!(s,w,w.next,probe)
    s.t += delta
    s
end

"""Multi-step trajectory with one tile barrier per step and one final copy-back.

The local source/destination matrices alternate; `State.q` is never rebound
and has its original identity on return. Intermediate steps are not exposed.
Use `fused_step!` when state observation is required after each step.
"""
function fused_simulate_amortized!(s::UniformLWAdvection.State,w::FusedWorkspace,
                                   dt::Real,nsteps::Integer)
    nsteps >= 0 || throw(ArgumentError("step count must be nonnegative"))
    nsteps == 0 && return s
    delta = Float64(dt)
    BaseFused.validate_fused_step(s,w,delta)
    nx,ny = size(s.q)
    rx,ry = delta*nx*0.5,delta*ny
    source,destination = s.q,w.next
    time = s.t
    for _ in 1:nsteps
        compute_tiles!(s,w,source,destination,rx,ry)
        source,destination = destination,source
        time += delta
    end
    source === s.q || copyback!(s,w,source)
    s.t = time
    s
end

"Untimed amortized trajectory with per-step compute and final copy-back IDs."
function fused_simulate_amortized!(s::UniformLWAdvection.State,w::FusedWorkspace,
                                   dt::Real,nsteps::Integer,probe::WorkerProbe)
    nsteps >= 0 || throw(ArgumentError("step count must be nonnegative"))
    validate_probe(w,probe,nsteps)
    nsteps == 0 && return s
    delta = Float64(dt)
    BaseFused.validate_fused_step(s,w,delta)
    fill!(probe.compute,0)
    fill!(probe.copyback,0)
    nx,ny = size(s.q)
    rx,ry = delta*nx*0.5,delta*ny
    source,destination = s.q,w.next
    time = s.t
    for pass in 1:nsteps
        compute_tiles!(s,w,source,destination,rx,ry,probe,pass)
        source,destination = destination,source
        time += delta
    end
    source === s.q || copyback!(s,w,source,probe)
    s.t = time
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
