"""
Diagnostic affine-fusion profiler. Run only after the frozen scaling matrix.

Example (launch Julia 1.10 with --threads=4 and the resolved benchmark project):

    julia --project=/tmp/twophase-compat110 --threads=4 \
      research/UniformLWAdvection/experiments/profile_affine.jl \
      --nx=4096 --ny=4096 --steps=16 --tile-x=512 --tile-y=8 \
      --backend=polyester --cpus=0,8,1,9 \
      --output-dir=/tmp/twophase-affine-profile-p4

The two complete trajectories use the original observable-step and amortized
APIs. Separate split trials call those implementations' existing compute and
copy-back functions; they do not copy the numerical tile kernel. Per-pass
clocks, allocation macros, and Profile sampling perturb scheduling and cache
state. Their fractions are diagnostic, never Stage 2 acceptance timings.
Profile output can include compilation and runtime frames and is not itself a
hot-loop time fraction. Smoke both odd and even step counts before large runs.
The amortized API exposes state only after its full trajectory; an even step
count does not require final copy-back. No carrier refresh is timed because
the frozen constant face arrays are the Stage 2 numerical contract.
"""

using LinearAlgebra: BLAS
using Profile
using SHA
using Statistics: median
using TOML

BLAS.set_num_threads(1)
push!(LOAD_PATH, normpath(joinpath(@__DIR__, "..")))
using UniformLWAdvection
using Polyester
include(joinpath(@__DIR__, "fused_affine.jl"))
const Toy = UniformLWAdvection
const Affine = UniformLWFusedAffineExperiment
const BaseFused = Affine.BaseFused

const ALLOWED = Set((
    "nx", "ny", "steps", "tile-x", "tile-y", "backend", "threshold",
    "batches-per-worker", "cpus", "output-dir", "repeats", "warmups",
    "profile-repeats",
))

function parse_options(args)
    options = Dict{String,String}()
    for argument in args
        startswith(argument, "--") && occursin('=', argument) ||
            error("expected --name=value: $argument")
        name, value = split(argument[3:end], '='; limit=2)
        name in ALLOWED || error("unknown option --$name")
        haskey(options, name) && error("duplicate option --$name")
        options[name] = value
    end
    options
end

option(options, name, fallback) = get(options, name, string(fallback))
function positive(options, name, fallback)
    value = parse(Int, option(options, name, fallback))
    value > 0 || error("--$name must be positive")
    value
end

sha_file(path) = isfile(path) ? bytes2hex(sha256(read(path))) : "absent"

function source_paths()
    names = (
        "profile_affine.jl", "fused_affine.jl", "fused_affine_polyester.jl",
        "fused.jl",
    )
    paths = Dict(name => joinpath(@__DIR__, name) for name in names)
    toy = normpath(joinpath(@__DIR__, ".."))
    paths["toy_src"] = joinpath(toy, "src", "UniformLWAdvection.jl")
    paths["toy_polyester_extension"] = joinpath(
        toy, "ext", "UniformLWAdvectionPolyesterExt.jl")
    paths["toy_project"] = joinpath(toy, "Project.toml")
    project = Base.active_project()
    if project !== nothing
        paths["active_project"] = project
        paths["active_manifest"] = joinpath(dirname(project), "Manifest.toml")
    end
    paths
end

source_hashes(paths) = Dict(name => sha_file(path) for (name,path) in paths)

function archive_sources!(paths,output_dir)
    destination = joinpath(output_dir,"source-snapshot")
    mkpath(destination)
    copied = Dict{String,String}()
    for (name,path) in paths
        isfile(path) || continue
        target = joinpath(destination,name*splitext(path)[2])
        cp(path,target;force=false)
        copied[name] = relpath(target,output_dir)
    end
    copied
end

function pin_workers!(cpus)
    Sys.islinux() || error("per-worker affinity probe requires Linux")
    length(cpus) == Threads.nthreads(:default) ||
        error("--cpus must specify one CPU per default-pool worker")
    length(unique(cpus)) == length(cpus) || error("CPU IDs must be distinct")
    all(cpu -> 0 <= cpu < 1024, cpus) || error("CPU outside cpu_set_t range")
    records = Vector{Dict{String,Any}}(undef,length(cpus))
    Threads.@threads :static for slot in eachindex(cpus)
        Threads.threadid() == slot || error("unexpected default-pool mapping")
        mask = zeros(UInt8,128)
        cpu = cpus[slot]
        mask[cpu ÷ 8 + 1] = UInt8(1) << (cpu % 8)
        ccall(:sched_setaffinity,Cint,(Cint,Csize_t,Ptr{UInt8}),
              0,length(mask),mask) == 0 || error("worker pinning failed")
        actual = zeros(UInt8,128)
        ccall(:sched_getaffinity,Cint,(Cint,Csize_t,Ptr{UInt8}),
              0,length(actual),actual) == 0 || error("affinity query failed")
        actual == mask || error("worker affinity differs from requested CPU")
        records[slot] = Dict("worker_id"=>slot,"cpu"=>cpu,
            "observed_cpu"=>ccall(:sched_getcpu,Cint,()),
            "linux_tid"=>ccall(:gettid,Cint,()))
    end
    records
end

function reset!(state, initial)
    copyto!(state.q,initial)
    state.t = 0.0
    nothing
end

function trajectory!(mode::Symbol,state,work,dt,steps)
    if mode === :observable
        Affine.fused_simulate!(state,work,dt,steps)
    elseif mode === :amortized
        Affine.fused_simulate_amortized!(state,work,dt,steps)
    else
        error("unknown affine profile mode")
    end
    nothing
end

function verify(state,reference,initial_mass,dt,steps,original_q)
    state.q === original_q || error("state.q identity changed")
    expected_time = steps*dt
    abs(state.t-expected_time) <=
        128eps(Float64)*steps*max(1.0,abs(expected_time)) ||
        error("affine profile ended at wrong time")
    all(isfinite,state.q) || error("nonfinite profile state")
    delta = maximum(abs,state.q .-reference)
    delta <= 512eps(Float64)*max(1.0,maximum(abs,reference)) ||
        error("affine profile differs from serial reference: $delta")
    mass_error = abs(Toy.mass(state)-initial_mass)
    mass_error <= 512eps(Float64) || error("affine profile changed mass")
    Dict{String,Any}(
        "max_state_delta"=>delta,"mass_error"=>mass_error,
        "translation_l2"=>Toy.translation_error(state),
        "checksum"=>sum(state.q),"observed_final_time"=>state.t,
    )
end

# Only orchestration is repeated here. The numerical tile, scheduler, and
# copy-back implementations are called directly from the frozen experiment.
function split_trajectory!(mode::Symbol,state,work,dt,steps,trial::Symbol)
    trial in (:clock,:allocation) || error("invalid split trial")
    compute_pass_seconds = zeros(Float64,trial === :clock ? steps : 0)
    copyback_pass_seconds = zeros(Float64,
        trial === :clock ? (mode === :observable ? steps : isodd(steps) ? 1 : 0) : 0)
    started = time_ns()
    compute_ns = copyback_ns = validation_ns = 0
    compute_bytes = copyback_bytes = validation_bytes = 0
    nx,ny = size(state.q)
    rx,ry = dt*nx*0.5,dt*ny
    source,destination = state.q,work.next
    time = state.t
    for pass in 1:steps
        if mode === :observable || pass == 1
            if trial === :clock
                t = time_ns()
                BaseFused.validate_fused_step(state,work,dt)
                validation_ns += time_ns()-t
            else
                validation_bytes += @allocated BaseFused.validate_fused_step(
                    state,work,dt)
            end
        end
        if trial === :clock
            t = time_ns()
            Affine.compute_tiles!(state,work,source,destination,rx,ry)
            duration = time_ns()-t
            compute_ns += duration
            compute_pass_seconds[pass] = duration*1.0e-9
        else
            compute_bytes += @allocated Affine.compute_tiles!(
                state,work,source,destination,rx,ry)
        end
        if mode === :observable
            if trial === :clock
                t = time_ns()
                Affine.copyback!(state,work,destination)
                duration = time_ns()-t
                copyback_ns += duration
                copyback_pass_seconds[pass] = duration*1.0e-9
            else
                copyback_bytes += @allocated Affine.copyback!(
                    state,work,destination)
            end
            state.t += dt
        elseif mode === :amortized
            source,destination = destination,source
            time += dt
        else
            error("invalid split mode")
        end
    end
    if mode === :amortized && source !== state.q
        if trial === :clock
            t = time_ns()
            Affine.copyback!(state,work,source)
            duration = time_ns()-t
            copyback_ns += duration
            copyback_pass_seconds[1] = duration*1.0e-9
        else
            copyback_bytes += @allocated Affine.copyback!(state,work,source)
        end
    end
    mode === :amortized && (state.t = time)
    Dict{String,Any}(
        "elapsed_seconds"=>(time_ns()-started)*1.0e-9,
        "compute_seconds"=>compute_ns*1.0e-9,
        "copyback_seconds"=>copyback_ns*1.0e-9,
        "compute_pass_seconds"=>compute_pass_seconds,
        "copyback_pass_seconds"=>copyback_pass_seconds,
        "validation_seconds"=>validation_ns*1.0e-9,
        "compute_allocated_bytes"=>compute_bytes,
        "copyback_allocated_bytes"=>copyback_bytes,
        "validation_allocated_bytes"=>validation_bytes,
    )
end

function worker_probe!(mode,state,work,dt,steps,initial)
    probe_steps = mode === :observable ? 1 : steps
    reset!(state,initial)
    trajectory!(mode,state,work,dt,probe_steps)
    expected = copy(state.q)
    expected_time = state.t
    reset!(state,initial)
    probe = Affine.WorkerProbe(work;nsteps=probe_steps)
    if mode === :observable
        Affine.fused_step!(state,work,dt,probe)
    else
        Affine.fused_simulate_amortized!(state,work,dt,probe_steps,probe)
    end
    state.q == expected && state.t == expected_time ||
        error("tracked affine path changed state or clock")
    all(!iszero,probe.compute) || error("compute probe missed a chunk")
    copy_expected = mode === :observable || isodd(probe_steps)
    if copy_expected
        all(!iszero,probe.copyback) || error("copy-back probe missed a chunk")
    else
        all(iszero,probe.copyback) ||
            error("even amortized trajectory unexpectedly copied back")
    end
    pass_ids = [Affine.active_workers(probe,pass) for pass in 1:probe_steps]
    copy_ids = Affine.active_copyback_workers(probe)
    Dict{String,Any}(
        "probe_steps"=>probe_steps,
        "compute_pass_worker_ids"=>pass_ids,
        "copyback_worker_ids"=>copy_ids,
        "copyback_executed"=>copy_expected,
        "active_worker_ids"=>sort!(unique(vcat(copy_ids,
            reduce(vcat,pass_ids;init=Int[])))),
    )
end

function profile_mode!(mode,nx,ny,initial,reference,initial_mass,dt,steps,
                       tile,backend,threshold,batches,warmups,repeats,
                       profile_repeats,output_dir)
    state = Toy.State(nx,ny)
    Toy.initialize!(state)
    original_q = state.q
    work = Affine.FusedWorkspace(state;backend,tile,threshold,
                                 batches_per_worker=batches)
    scratch_bytes = sum(sizeof(chunk.xhalf)+sizeof(chunk.yfull)+
        sizeof(chunk.xwrap)+sizeof(chunk.ywrap) for chunk in work.scratch)
    for _ in 1:warmups
        reset!(state,initial)
        trajectory!(mode,state,work,dt,steps)
        reset!(state,initial)
        split_trajectory!(mode,state,work,dt,steps,:clock)
    end
    # Compile the allocation-mode specialization before reporting bytes.
    reset!(state,initial)
    split_trajectory!(mode,state,work,dt,steps,:allocation)
    reset!(state,initial)
    @allocated trajectory!(mode,state,work,dt,steps)

    full_times = Float64[]
    full_allocations = Int[]
    full_gc = Float64[]
    split_samples = Dict{String,Any}[]
    for _ in 1:repeats
        reset!(state,initial)
        GC.gc()
        sample = @timed trajectory!(mode,state,work,dt,steps)
        push!(full_times,sample.time)
        push!(full_allocations,sample.bytes)
        push!(full_gc,sample.gctime)
        verify(state,reference,initial_mass,dt,steps,original_q)
        reset!(state,initial)
        push!(split_samples,split_trajectory!(
            mode,state,work,dt,steps,:clock))
        verify(state,reference,initial_mass,dt,steps,original_q)
    end
    reset!(state,initial)
    full_bytes = @allocated trajectory!(mode,state,work,dt,steps)
    complete_checks = verify(
        state,reference,initial_mass,dt,steps,original_q)
    reset!(state,initial)
    split_bytes = split_trajectory!(mode,state,work,dt,steps,:allocation)
    split_checks = verify(state,reference,initial_mass,dt,steps,original_q)
    probe = worker_probe!(mode,state,work,dt,steps,initial)

    Profile.clear()
    for _ in 1:profile_repeats
        reset!(state,initial)
        Profile.@profile trajectory!(mode,state,work,dt,steps)
    end
    profile_path = joinpath(output_dir,string(mode)*"_cpu_profile.txt")
    open(profile_path,"w") do io
        println(io,"Diagnostic Julia Profile samples for ",mode,
            "; sampling perturbs runtime and does not prove hardware traffic.")
        Profile.print(io)
    end
    profile_words = length(Profile.fetch())
    split_median = Dict(key=>median(Float64[sample[key]
        for sample in split_samples])
        for key in ("elapsed_seconds","compute_seconds",
                    "copyback_seconds","validation_seconds"))
    whole_split = split_median["elapsed_seconds"]
    whole_split > 0 || error("nonpositive split diagnostic time")
    directional_updates = 0
    for tile_bounds in work.tiles
        ilo,ihi,jlo,jhi = tile_bounds
        width,height = ihi-ilo+1,jhi-jlo+1
        directional_updates +=
            (width+2)*(height+2)+(width+2)*height+width*height
    end
    unfused_updates = 3nx*ny
    Dict{String,Any}(
        "mode"=>string(mode),
        "complete_seconds"=>full_times,
        "complete_median_seconds"=>median(full_times),
        "complete_allocated_bytes"=>full_allocations,
        "complete_gc_seconds"=>full_gc,
        "complete_separate_allocated_bytes"=>full_bytes,
        "split_clock_samples"=>split_samples,
        "split_clock_median"=>split_median,
        "diagnostic_compute_fraction"=>median(Float64[
            sample["compute_seconds"]/sample["elapsed_seconds"]
            for sample in split_samples]),
        "diagnostic_copyback_fraction"=>median(Float64[
            sample["copyback_seconds"]/sample["elapsed_seconds"]
            for sample in split_samples]),
        "diagnostic_validation_fraction"=>median(Float64[
            sample["validation_seconds"]/sample["elapsed_seconds"]
            for sample in split_samples]),
        "split_allocation_trial"=>split_bytes,
        "complete_checks"=>complete_checks,
        "split_checks"=>split_checks,
        "worker_probe"=>probe,
        "numeric_array_bytes"=>BaseFused.fused_memory_bytes(state,work),
        "state_carrier_bytes"=>sizeof(state.q)+sizeof(state.u)+sizeof(state.v),
        "next_bytes"=>sizeof(work.next),
        "chunk_scratch_bytes"=>scratch_bytes,
        "analytic_directional_updates_per_step"=>directional_updates,
        "unfused_directional_updates_per_step"=>unfused_updates,
        "analytic_halo_duplicate_update_fraction"=>
            (directional_updates-unfused_updates)/unfused_updates,
        "analytic_halo_count_scope"=>
            "directional cell updates, not measured memory traffic or timing",
        "tile_count"=>length(work.tiles),
        "chunk_count"=>length(work.chunks),
        "cpu_profile_file"=>basename(profile_path),
        "cpu_profile_words"=>profile_words,
        "peak_rss_bytes"=>Sys.maxrss(),
    )
end

function main()
    options = parse_options(ARGS)
    nx = positive(options,"nx",4096)
    ny = positive(options,"ny",nx)
    steps = positive(options,"steps",16)
    tile = (positive(options,"tile-x",512),positive(options,"tile-y",8))
    backend = Symbol(option(options,"backend","polyester"))
    backend in (:serial,:threads,:spawn,:polyester) || error("invalid backend")
    threshold = parse(Int,option(options,"threshold",16384))
    threshold >= 0 || error("negative threshold")
    batches = positive(options,"batches-per-worker",4)
    repeats = positive(options,"repeats",3)
    warmups = positive(options,"warmups",2)
    profile_repeats = positive(options,"profile-repeats",2)
    cpus = parse.(Int,split(option(options,"cpus","0"),','))
    output_dir = abspath(option(options,"output-dir","/tmp/twophase-affine-profile"))
    ispath(output_dir) && error("output directory already exists: $output_dir")
    affinity = pin_workers!(cpus)
    mkpath(output_dir)
    paths = source_paths()
    hashes = source_hashes(paths)
    snapshots = archive_sources!(paths,output_dir)
    dt = 0.35/(0.7nx+0.3ny)

    reference_state = Toy.State(nx,ny)
    Toy.initialize!(reference_state)
    initial = copy(reference_state.q)
    initial_mass = Toy.mass(reference_state)
    reference_work = Toy.Workspace(reference_state;backend=:serial,
                                    flux=:recompute)
    Toy.simulate!(reference_state,reference_work,dt,steps)
    reference = copy(reference_state.q)
    reference_state = reference_work = nothing
    GC.gc()

    modes = Dict{String,Any}()
    for mode in (:observable,:amortized)
        modes[string(mode)] = profile_mode!(
            mode,nx,ny,initial,reference,initial_mass,dt,steps,
            tile,backend,threshold,batches,warmups,repeats,
            profile_repeats,output_dir)
        GC.gc()
    end
    source_hashes(paths) == hashes ||
        error("source changed during diagnostic affine profiling")
    result = Dict{String,Any}(
        "kind"=>"diagnostic_affine_pass_profile",
        "caveat"=>"instrumented split and CPU profiles perturb runtime; not acceptance timing",
        "julia_version"=>string(VERSION),
        "default_workers"=>Threads.nthreads(:default),
        "backend"=>string(backend),"tile"=>collect(tile),
        "threshold"=>threshold,"batches_per_worker"=>batches,
        "nx"=>nx,"ny"=>ny,"steps"=>steps,"dt"=>dt,
        "final_time"=>steps*dt,
        "repeats"=>repeats,"warmups"=>warmups,
        "profile_repeats"=>profile_repeats,
        "affinity"=>affinity,
        "blas_threads"=>BLAS.get_num_threads(),
        "gc_threads_requested"=>get(ENV,"JULIA_NUM_GC_THREADS","unreported"),
        "active_project"=>string(Base.active_project()),
        "julia_executable"=>joinpath(Sys.BINDIR,"julia"),
        "cpu_name"=>Sys.CPU_NAME,
        "source_sha256"=>hashes,
        "source_snapshot"=>snapshots,
        "modes"=>modes,
    )
    open(joinpath(output_dir,"profile.toml"),"w") do io
        TOML.print(io,result)
    end
    println("Wrote diagnostic affine profile to ",output_dir)
end

main()
