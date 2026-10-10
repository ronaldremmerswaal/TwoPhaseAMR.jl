# Standalone numerical diagnostic for the frozen affine experiment. By
# default this checks a 1031×257 odd, multi-x-tile partition with poisoned
# scratch. Use nx=4096 ny=4096 steps=29 reps=5 to stress the reported p24
# intermittent case. It never edits the benchmarked kernel.
using LinearAlgebra
BLAS.set_num_threads(1)
push!(LOAD_PATH,normpath(joinpath(@__DIR__,"..")))
using UniformLWAdvection, Polyester
include(joinpath(@__DIR__,"fused_affine.jl"))
const Affine = UniformLWFusedAffineExperiment

const OPT = Dict(split(arg,'=';limit=2)[1] => split(arg,'=';limit=2)[2]
                 for arg in ARGS)
opt(k,default) = get(OPT,k,default)
const NX = parse(Int,opt("nx","1031"))
const NY = parse(Int,opt("ny","257"))
const NSTEPS = parse(Int,opt("steps","29"))
const REPS = parse(Int,opt("reps","20"))
min(NX,NY,NSTEPS,REPS)>0 || error("dimensions, steps, and repeats must be positive")
const TILE = (512,8)
const CPU_OPTION = opt("cpus","")
const CPUS = isempty(CPU_OPTION) ? Int[] : parse.(Int,split(CPU_OPTION,','))
isempty(CPUS) || length(CPUS)==Threads.nthreads(:default) ||
    error("CPU/thread mismatch")

function pin_workers!()
    Threads.@threads :static for slot in eachindex(CPUS)
        Threads.threadid()==slot || error("worker slot mismatch")
        mask=zeros(UInt8,128)
        cpu=CPUS[slot]
        mask[cpu÷8+1]=UInt8(1)<<(cpu%8)
        ccall(:sched_setaffinity,Cint,(Cint,Csize_t,Ptr{UInt8}),
              0,length(mask),mask)==0 || error("pin failed")
        actual=zeros(UInt8,128)
        ccall(:sched_getaffinity,Cint,(Cint,Csize_t,Ptr{UInt8}),
              0,length(actual),actual)==0 || error("affinity query failed")
        actual==mask || error("pin mismatch")
    end
end

@inline function compare(a,b)
    best=0.0
    best_index=0
    @inbounds for k in eachindex(a,b)
        delta=abs(a[k]-b[k])
        if delta>best
            best=delta
            best_index=k
        end
    end
    return best,best_index
end

function describe_error(candidate,reference,index,work)
    index==0 && return
    i,j=Tuple(CartesianIndices(candidate)[index])
    x_tile=cld(i,TILE[1]); y_tile=cld(j,TILE[2])
    tiles_per_row=cld(NX,TILE[1])
    tile_index=(y_tile-1)*tiles_per_row+x_tile
    chunk=findfirst(range -> tile_index in range,work.chunks)
    println("WORST i=",i," j=",j," tile=",tile_index,
            " chunk=",chunk," chunk_range=",work.chunks[chunk],
            " candidate=",candidate[index]," reference=",reference[index])
end

isempty(CPUS) || pin_workers!()
reference=State(NX,NY); initialize!(reference)
initial=copy(reference.q)
serial=Workspace(reference;backend=:serial,flux=:recompute)
dt=0.35/(0.7NX+0.3NY)
simulate!(reference,serial,dt,NSTEPS)
state=State(NX,NY); initialize!(state)
work=Affine.FusedWorkspace(state;backend=:polyester,tile=TILE,
                           threshold=16384)
original_q=state.q
ntiles=length(work.tiles)
coverage=zeros(Int,ntiles)
for range in work.chunks, tile in range
    coverage[tile]+=1
end
all(==(1),coverage) || error("chunk tile ownership is incomplete or repeated")
sum(prod((t[2]-t[1]+1,t[4]-t[3]+1)) for t in work.tiles)==NX*NY ||
    error("tile area does not cover the grid")
expected_tiles=[(i,min(NX,i+TILE[1]-1),j,min(NY,j+TILE[2]-1))
                for j in 1:TILE[2]:NY for i in 1:TILE[1]:NX]
work.tiles==expected_tiles || error("tile geometry has gaps or overlaps")
println("CONFIG nx=",NX," ny=",NY," threads=",Threads.nthreads(:default),
        " steps=",NSTEPS," reps=",REPS," tiles=",ntiles,
        " chunks=",length(work.chunks)," dt=",dt,
        " pinned=",!isempty(CPUS))

# Mirror screen_layouts' warm/compile/reset sequence before each trajectory.
tol=512eps(Float64)*max(1.0,maximum(abs,reference.q))
for trial in 1:REPS
    copyto!(state.q,initial);state.t=0.0
    Affine.fused_simulate_amortized!(state,work,dt,2)
    copyto!(state.q,initial);state.t=0.0
    Affine.fused_simulate_amortized!(state,work,dt,2)
    copyto!(state.q,initial);state.t=0.0
    Affine.fused_simulate_amortized!(state,work,dt,NSTEPS)
    copyto!(state.q,initial);state.t=0.0
    if opt("poison","1")=="1"
        fill!(work.next,NaN)
        for scratch in work.scratch
            fill!(scratch.xhalf,NaN)
            fill!(scratch.yfull,NaN)
        end
    end
    Affine.fused_simulate_amortized!(state,work,dt,NSTEPS)
    delta,index=compare(state.q,reference.q)
    println("TRIAL ",trial," delta=",delta," time=",state.t,
            " identity=",state.q===original_q)
    flush(stdout)
    if !(delta<=tol && state.q===original_q && state.t==reference.t)
        describe_error(state.q,reference.q,index,work)
        error("affine trajectory failed numerical/identity/time check; " *
              "delta=$delta tolerance=$tol")
    end
end

if opt("trace","1")=="1"
    let
        copyto!(state.q,initial);state.t=0.0
        source,destination=state.q,work.next
        traced_reference=State(NX,NY); initialize!(traced_reference)
        traced_serial=Workspace(traced_reference;backend=:serial,flux=:recompute)
        rx,ry=dt*NX*0.5,dt*NY
        for pass in 1:NSTEPS
            step!(traced_reference,traced_serial,dt)
            Affine.compute_tiles!(state,work,source,destination,rx,ry)
            delta,index=compare(destination,traced_reference.q)
            println("PASS ",pass," delta=",delta)
            flush(stdout)
            if delta>tol
                describe_error(destination,traced_reference.q,index,work)
                error("first divergent compute pass: $pass delta=$delta")
            end
            source,destination=destination,source
        end
    end
end

if opt("track","1")=="1"
    copyto!(state.q,initial);state.t=0.0
    probe=Affine.WorkerProbe(work;nsteps=NSTEPS)
    Affine.fused_simulate_amortized!(state,work,dt,NSTEPS,probe)
    counts=[length(Affine.active_workers(probe,pass)) for pass in 1:NSTEPS]
    delta,index=compare(state.q,reference.q)
    println("TRACK counts=",counts," copyback=",
            length(Affine.active_copyback_workers(probe))," delta=",delta)
    copyback_ok=isodd(NSTEPS) ? all(!iszero,probe.copyback) :
        all(iszero,probe.copyback)
    if !(delta<=tol && state.q===original_q && state.t==reference.t &&
         all(!iszero,probe.compute) && copyback_ok)
        describe_error(state.q,reference.q,index,work)
        error("tracked trajectory failed numerical/worker/identity/time check")
    end
end
