# Fresh-process, pinned full-step comparison of the frozen toy and the
# cache-preserving tile prototype. All diagnostics are outside the timer.
using LinearAlgebra, TOML, SHA
BLAS.set_num_threads(1)
push!(LOAD_PATH, normpath(joinpath(@__DIR__,"..","..")))
using UniformLWAdvection, Polyester
include(joinpath(@__DIR__,"CachePipeline.jl"))
const Toy = UniformLWAdvection
const CP = CachePipeline

function options(args)
    o = Dict{String,String}()
    for arg in args
        pair = split(arg,'=';limit=2)
        length(pair) == 2 || error("expected key=value: $arg")
        o[pair[1]] = pair[2]
    end
    o
end

function pin_workers(cpus)
    length(cpus) == Threads.nthreads(:default) || error("one CPU per worker required")
    length(unique(cpus)) == length(cpus) || error("CPU IDs must be distinct")
    Sys.islinux() || error("worker pinning requires Linux")
    observed = Vector{Dict{String,Int}}(undef,length(cpus))
    Threads.@threads :static for slot in eachindex(cpus)
        Threads.threadid() == slot || error("unexpected default-pool mapping")
        cpu = cpus[slot]
        0 <= cpu < 1024 || error("CPU ID outside cpu_set_t")
        mask = zeros(UInt8,128)
        mask[cpu ÷ 8 + 1] = UInt8(1) << (cpu % 8)
        ccall(:sched_setaffinity,Cint,(Cint,Csize_t,Ptr{UInt8}),
              0,length(mask),mask) == 0 || error("sched_setaffinity failed")
        actual = zeros(UInt8,128)
        ccall(:sched_getaffinity,Cint,(Cint,Csize_t,Ptr{UInt8}),
              0,length(actual),actual) == 0 || error("sched_getaffinity failed")
        actual == mask || error("worker affinity mismatch")
        running = ccall(:sched_getcpu,Cint,())
        running == cpu || error("worker not running on its pinned CPU")
        observed[slot] = Dict("worker"=>slot,"cpu"=>cpu,
                              "observed_cpu"=>running,
                              "linux_tid"=>ccall(:gettid,Cint,()))
    end
    observed
end

function reset!(s,initial)
    copyto!(s.q,initial)
    s.t = 0.0
    nothing
end

@inline advance!(::Val{:toy},s,w,dt) = Toy.step!(s,w,dt)
@inline advance!(::Val{:pipeline},s,w,dt) = CP.step!(s,w,dt)
@inline advance!(::Val{:oracle},s,w,dt) = Toy.step!(s,w,dt)
@inline trajectory!(::Val{:toy},s,w,dt,n) = Toy.simulate!(s,w,dt,n)
@inline trajectory!(::Val{:pipeline},s,w,dt,n) = CP.simulate!(s,w,dt,n)
@inline trajectory!(::Val{:oracle},s,w,dt,n) = Toy.simulate!(s,w,dt,n)

function probe!(mode::Val{:toy},s,w,initial,dt)
    reset!(s,initial)
    advance!(mode,s,w,dt)
    expected = copy(s.q)
    reset!(s,initial)
    Toy.track_workers!(w,true)
    ids = try
        advance!(mode,s,w,dt)
        nt = length(w.tiles)
        [sort!(unique(@view(w.worker_ids[(p-1)*nt+1:p*nt]))) for p in 1:3]
    finally
        Toy.track_workers!(w,false)
    end
    s.q == expected || error("toy tracking changed the one-step field")
    all(v -> all(!iszero,v),ids) || error("toy worker probe incomplete")
    reset!(s,initial)
    ids
end

function probe!(mode::Val{:pipeline},s,w,initial,dt)
    original_q = s.q
    reset!(s,initial)
    advance!(mode,s,w,dt)
    expected = copy(s.q)
    reset!(s,initial)
    CP.track_workers!(w,true)
    ids = try
        advance!(mode,s,w,dt)
        [CP.active_workers(w,:compute),CP.active_workers(w,:copyback)]
    finally
        CP.track_workers!(w,false)
    end
    s.q == expected || error("pipeline tracking changed the one-step field")
    s.q === original_q || error("public q object changed during the probe")
    reset!(s,initial)
    ids
end

function compare_oracle(q,path)
    expected = similar(q)
    open(path,"r") do io
        read!(io,expected)
        eof(io) || error("oracle field has trailing bytes")
    end
    mismatches = 0
    max_abs = 0.0
    @inbounds for k in eachindex(q)
        delta = abs(q[k]-expected[k])
        mismatches += q[k] != expected[k]
        max_abs = max(max_abs,delta)
    end
    mismatches == 0 || error("full field differs from serial toy oracle: $mismatches cells, maximum $max_abs")
    Dict("field_mismatches"=>mismatches,"field_max_abs_error"=>max_abs,
         "oracle_sha256"=>bytes2hex(sha256(read(path))))
end

function run(o)
    mode = Symbol(get(o,"mode","pipeline"))
    mode in (:init,:oracle,:toy,:pipeline) || error("unknown mode $mode")
    nx, ny = parse(Int,o["nx"]), parse(Int,o["ny"])
    nx > 0 && ny > 0 || error("dimensions must be positive")
    s = Toy.State(nx,ny;u=0.7,v=-0.3)
    if mode === :init
        Toy.initialize!(s)
        open(o["output"],"w") do io
            write(io,s.q)
        end
        return
    end
    steps = parse(Int,o["steps"])
    steps > 0 || error("steps must be positive")
    open(o["input"],"r") do io
        read!(io,s.q)
        eof(io) || error("input field has trailing bytes")
    end
    initial = copy(s.q)
    original_q = s.q
    tx = parse(Int,get(o,"tile-x",mode === :pipeline ? "128" : "512"))
    ty = parse(Int,get(o,"tile-y",mode === :pipeline ? "32" : "16"))
    batches = parse(Int,get(o,"batches-per-worker","4"))
    dt = 0.35/(0.7nx+0.3ny)
    backend = mode === :oracle ? :serial : :polyester
    w = mode === :pipeline ? CP.PipelineWorkspace(s;tile=(tx,ty),
                                                 batches_per_worker=batches) :
        Toy.Workspace(s;backend,tile=(tx,ty),flux=:recompute,
                      carrier=:constant,batches_per_worker=batches)
    cpus = parse.(Int,split(o["cpus"],','))
    affinity = pin_workers(cpus)
    method = Val(mode)
    trajectory!(method,s,w,dt,2) # compile and fault in arrays
    s.q === original_q || error("public q object changed during warmup")
    reset!(s,initial)
    alloc_bytes = @allocated advance!(method,s,w,dt)
    reset!(s,initial)
    pass_ids = mode === :oracle ? Any[] : probe!(method,s,w,initial,dt)
    if mode !== :oracle
        all(ids -> length(ids) == length(cpus),pass_ids) ||
            error("not all workers participated in every untimed phase")
    end
    reset!(s,initial)
    GC.gc()
    started = time_ns()
    trajectory!(method,s,w,dt,steps)
    elapsed = (time_ns()-started)/1e9
    s.q === original_q || error("public q object changed during timed trajectory")
    comparison = haskey(o,"oracle") ? compare_oracle(s.q,o["oracle"]) :
                 Dict{String,Any}()
    open(o["output"],"w") do io
        write(io,s.q)
    end
    source = Dict(
        "toy"=>bytes2hex(sha256(read(joinpath(@__DIR__,"..","..","src","UniformLWAdvection.jl")))),
        "pipeline"=>bytes2hex(sha256(read(joinpath(@__DIR__,"CachePipeline.jl")))),
        "driver"=>bytes2hex(sha256(read(@__FILE__))))
    record = Dict{String,Any}(
        "mode"=>String(mode),"nx"=>nx,"ny"=>ny,"steps"=>steps,
        "workers"=>Threads.nthreads(:default),"tile_x"=>tx,"tile_y"=>ty,
        "batches_per_worker"=>batches,"dt"=>dt,"final_time"=>s.t,
        "time_seconds"=>elapsed,
        "ns_per_unknown_step"=>1e9*elapsed/(nx*ny*steps),
        "allocation_bytes_per_warm_step"=>alloc_bytes,
        "numerical_array_bytes"=>mode === :pipeline ? CP.memory_bytes(s,w) : Toy.memory_bytes(s,w),
        "q_alias_preserved"=>s.q === original_q,
        "mass_initial"=>sum(initial)/length(initial),
        "mass_final"=>sum(s.q)/length(s.q),
        "field_sha256"=>bytes2hex(sha256(reinterpret(UInt8,vec(s.q)))),
        "active_worker_ids_per_phase"=>pass_ids,
        "active_worker_counts"=>length.(pass_ids),
        "affinity"=>affinity,"source_sha256"=>source,
        "julia_version"=>string(VERSION),"blas_threads"=>BLAS.get_num_threads())
    merge!(record,comparison)
    open(o["report"],"w") do io
        TOML.print(io,record)
    end
end

run(options(ARGS))
