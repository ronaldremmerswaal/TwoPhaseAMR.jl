#=
One fresh-process exploratory sample of standard, halo, or fused LW storage.
`fused_affine` times individually observable steps; `fused_amortized` times
the specialized complete trajectory that exposes state only on return.

Arguments use name=value, e.g. variant=halo nx=1024 steps=256 backend=polyester
tile-x=1024 tile-y=32 cpus=0,8,1,9 output=/tmp/halo.toml. Run Julia with the
matching thread count and resolved benchmark project. All variants retain
prescribed velocity arrays, identical physical work, and their complete-step
storage maintenance. Analytic error and actual worker probes run after the
timed trajectory; probe parity is checked against the same untracked path.
This is a bounded layout screen, not a scaling gate.
=#
using LinearAlgebra, TOML, SHA
BLAS.set_num_threads(1)
push!(LOAD_PATH, normpath(joinpath(@__DIR__, "..")))
using UniformLWAdvection
using Polyester
const Toy = UniformLWAdvection
const OPTIONS = Dict(split(arg, '='; limit=2)[1] => split(arg, '='; limit=2)[2] for arg in ARGS)
option(name, default) = get(OPTIONS, name, default)

const VARIANT = option("variant", "standard")
if VARIANT == "halo"
    include("halo.jl")
    using .UniformLWHaloExperiment
elseif VARIANT == "fused"
    include("fused.jl")
    using .UniformLWFusedExperiment
elseif VARIANT in ("fused_affine", "fused_amortized")
    include("fused_affine.jl")
    using .UniformLWFusedAffineExperiment
elseif VARIANT != "standard"
    error("unknown layout variant")
end

function pin_workers!()
    cpus = parse.(Int, split(option("cpus", "0"), ','))
    length(cpus) == Threads.nthreads(:default) || error("one CPU per worker required")
    length(unique(cpus)) == length(cpus) || error("CPUs must be distinct")
    Sys.islinux() || error("this pinning experiment requires Linux")
    records = Vector{Dict{String,Any}}(undef, length(cpus))
    Threads.@threads :static for slot in eachindex(cpus)
        Threads.threadid() == slot || error("unexpected default-pool mapping")
        cpu = cpus[slot]
        0 <= cpu < 1024 || error("CPU outside cpu_set_t range")
        mask = zeros(UInt8, 128)
        mask[cpu ÷ 8 + 1] = UInt8(1) << (cpu % 8)
        ccall(:sched_setaffinity, Cint, (Cint,Csize_t,Ptr{UInt8}),
              0, length(mask), mask) == 0 || error("pinning failed")
        actual = zeros(UInt8, 128)
        ccall(:sched_getaffinity, Cint, (Cint,Csize_t,Ptr{UInt8}),
              0, length(actual), actual) == 0 || error("affinity query failed")
        actual == mask || error("worker affinity differs")
        records[slot] = Dict("worker"=>slot, "cpu"=>cpu,
            "observed_cpu"=>ccall(:sched_getcpu,Cint,()),
            "linux_tid"=>ccall(:gettid,Cint,()))
    end
    records
end

function fixture(nx, ny, backend, tile, threshold, dt, steps)
    reference_state = Toy.State(nx, ny)
    Toy.initialize!(reference_state)
    initial = copy(reference_state.q)
    reference_work = Toy.Workspace(reference_state;backend=:serial)
    Toy.simulate!(reference_state, reference_work, dt, steps)
    reference = copy(reference_state.q)
    state = Toy.State(nx, ny)
    Toy.initialize!(state)
    if VARIANT == "standard"
        work = Toy.Workspace(state;backend,tile,threshold,flux=:recompute)
        values = state.q
        trajectory! = () -> Toy.simulate!(state, work, dt, steps)
        warm! = () -> Toy.simulate!(state, work, dt, 2)
        reset! = () -> (copyto!(values, initial); state.t = 0.0; nothing)
        state_check! = () -> (state.q === values || error("state.q identity changed"); state.t)
        translation_check! = () -> Toy.translation_error(state)
        probe! = () -> begin
            reset!(); Toy.step!(state, work, dt)
            expected = copy(values); expected_time = state.t
            reset!(); Toy.track_workers!(work, true)
            recorded = try
                Toy.step!(state, work, dt)
                copy(work.worker_ids)
            finally
                Toy.track_workers!(work, false)
            end
            nt = length(work.tiles)
            pass_ids = [sort!(unique(filter(!iszero,
                @view recorded[(pass-1)*nt+1:pass*nt]))) for pass in 1:3]
            all(!iszero, @view recorded[1:3nt]) ||
                error("standard probe missed a tile pass")
            delta = maximum(abs, values .- expected)
            delta == 0 && state.t == expected_time ||
                error("standard worker probe changed state or clock")
            Dict{String,Any}(
                "active_worker_ids"=>sort!(unique(vcat(pass_ids...))),
                "compute_pass_worker_ids"=>pass_ids,
                "copyback_worker_ids"=>Int[],
                "copyback_executed"=>false,
                "probe_step_count"=>1,
                "probe_state_delta"=>delta,
                "active_worker_scope"=>"untimed actual standard sweep tile loops",
            )
        end
        bytes = Toy.memory_bytes(state, work)
    elseif startswith(VARIANT, "fused")
        work = FusedWorkspace(state;backend,tile,threshold)
        values = state.q
        simulate! = VARIANT == "fused_amortized" ? fused_simulate_amortized! : fused_simulate!
        trajectory! = () -> simulate!(state, work, dt, steps)
        warm! = () -> simulate!(state, work, dt, 2)
        reset! = () -> (copyto!(values, initial); state.t = 0.0; nothing)
        state_check! = () -> (state.q === values || error("state.q identity changed"); state.t)
        translation_check! = () -> Toy.translation_error(state)
        probe! = if VARIANT in ("fused_affine", "fused_amortized")
            () -> begin
                nprobe = VARIANT == "fused_amortized" ? steps : 1
                reset!(); simulate!(state, work, dt, nprobe)
                expected = copy(values); expected_time = state.t
                reset!()
                probe = WorkerProbe(work; nsteps=nprobe)
                if VARIANT == "fused_amortized"
                    fused_simulate_amortized!(state, work, dt, nprobe, probe)
                else
                    fused_step!(state, work, dt, probe)
                end
                pass_ids = [active_workers(probe, pass) for pass in 1:nprobe]
                copy_ids = active_copyback_workers(probe)
                all(!iszero, probe.compute) ||
                    error("affine probe missed a compute chunk")
                copy_expected = VARIANT == "fused_affine" || isodd(nprobe)
                if copy_expected
                    all(!iszero, probe.copyback) ||
                        error("affine probe missed a copy-back chunk")
                else
                    all(iszero, probe.copyback) ||
                        error("even amortized trajectory unexpectedly copied back")
                end
                delta = maximum(abs, values .- expected)
                delta == 0 && state.t == expected_time && state.q === values ||
                    error("affine worker probe changed state, clock, or q identity")
                all_ids = sort!(unique(vcat(copy_ids,
                    reduce(vcat, pass_ids; init=Int[]))))
                Dict{String,Any}(
                    "active_worker_ids"=>all_ids,
                    "compute_pass_worker_ids"=>pass_ids,
                    "copyback_worker_ids"=>copy_ids,
                    "copyback_executed"=>copy_expected,
                    "probe_step_count"=>nprobe,
                    "probe_state_delta"=>delta,
                    "active_worker_scope"=>"untimed actual affine compute and copy-back chunk loops",
                )
            end
        else
            () -> Dict{String,Any}(
                "active_worker_ids"=>Int[],
                "compute_pass_worker_ids"=>Vector{Vector{Int}}(),
                "copyback_worker_ids"=>Int[],
                "copyback_executed"=>false,
                "probe_step_count"=>0,
                "active_worker_scope"=>"legacy fused prototype not instrumented",
            )
        end
        bytes = fused_memory_bytes(state, work)
    else
        work = HaloWorkspace(PaddedState(state);backend,tile,threshold)
        values = halo_values(work)
        trajectory! = () -> halo_simulate!(work, dt, steps)
        warm! = () -> halo_simulate!(work, dt, 2)
        # Constant-carrier arithmetic does not consume the clock. Restore it
        # with the experiment's reset helper to keep metadata equally defined.
        reset! = () -> (copyto!(values, initial); work.state.t = 0.0; nothing)
        state_check! = () -> work.state.t
        translation_check! = () -> begin
            copyto!(state.q, values)
            state.t = work.state.t
            Toy.translation_error(state)
        end
        probe! = () -> Dict{String,Any}(
            "active_worker_ids"=>Int[],
            "compute_pass_worker_ids"=>Vector{Vector{Int}}(),
            "copyback_worker_ids"=>Int[],
            "copyback_executed"=>false,
            "probe_step_count"=>0,
            "active_worker_scope"=>"halo prototype not instrumented",
        )
        bytes = halo_memory_bytes(work)
    end
    values, reference, trajectory!, warm!, reset!, state_check!,
    translation_check!, probe!, bytes, dt, steps
end

function measure(values, reference, trajectory!, warm!, reset!, state_check!,
                 translation_check!, probe!, bytes, dt, steps)
    reset!(); warm!(); reset!(); warm!()
    reset!(); trajectory!() # Compile the exact timed closure before measuring.
    reset!(); GC.gc()
    sample = @timed trajectory!()
    observed_time = state_check!()
    abs(observed_time - steps*dt) <=
        128eps(Float64)*steps*max(1.0,abs(steps*dt)) ||
        error("layout trajectory ended at incorrect time: $observed_time")
    difference = maximum(abs, values .- reference)
    difference <= 512eps(Float64)*max(1.0, maximum(abs,reference)) ||
        error("layout changed the numerical result: $difference")
    mass_error = abs(sum(values)/length(values) - 1.0)
    mass_error <= 512eps(Float64) || error("mass changed: $mass_error")
    translation_l2 = translation_check!()
    isfinite(translation_l2) || error("nonfinite analytic translation error")
    result = Dict{String,Any}("time_seconds"=>sample.time, "allocation_bytes"=>sample.bytes,
        "gc_seconds"=>sample.gctime, "max_state_delta"=>difference,
        "mass_error"=>mass_error, "checksum"=>sum(values),
        "translation_l2"=>translation_l2,
        "observed_final_time"=>observed_time,
        "numerical_array_bytes"=>bytes, "peak_rss_bytes"=>Sys.maxrss())
    merge!(result, probe!())
    result
end

function main()
    affinity = pin_workers!()
    nx = parse(Int, option("nx", "1024"))
    ny = parse(Int, option("ny", string(nx)))
    steps = parse(Int, option("steps", "256"))
    backend = Symbol(option("backend", "polyester"))
    tile = (parse(Int,option("tile-x","1024")), parse(Int,option("tile-y","32")))
    threshold = parse(Int, option("threshold", "16384"))
    min(nx,ny,steps) > 0 || error("positive dimensions and step count required")
    dt = 0.35/(0.7nx+0.3ny)
    result = measure(fixture(nx,ny,backend,tile,threshold,dt,steps)...)
    merge!(result, Dict("kind"=>"exploratory_layout_screen", "variant"=>VARIANT,
        "nx"=>nx, "ny"=>ny, "steps"=>steps, "dt"=>dt, "final_time"=>steps*dt,
        "backend"=>string(backend), "tile"=>collect(tile), "threshold"=>threshold,
        "julia_version"=>string(VERSION), "workers"=>Threads.nthreads(:default),
        "blas_threads"=>BLAS.get_num_threads(), "affinity"=>affinity,
        "core_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"..","src","UniformLWAdvection.jl")))),
        "script_sha256"=>bytes2hex(sha256(read(@__FILE__))),
        "variant_sha256"=>VARIANT=="standard" ? "core" :
            bytes2hex(sha256(read(joinpath(@__DIR__,(VARIANT=="fused_amortized" ? "fused_affine" : VARIANT)*".jl")))),
        "fused_base_sha256"=>startswith(VARIANT,"fused") ?
            bytes2hex(sha256(read(joinpath(@__DIR__,"fused.jl")))) : "not_loaded",
        "affine_polyester_sha256"=>VARIANT in ("fused_affine","fused_amortized") ?
            bytes2hex(sha256(read(joinpath(@__DIR__,"fused_affine_polyester.jl")))) : "not_loaded"))
    open(option("output","/tmp/lw-layout-screen.toml"), "w") do io
        TOML.print(io,result)
    end
end
main()
