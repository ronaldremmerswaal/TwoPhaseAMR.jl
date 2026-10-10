# Matched, fresh-process benchmark of the unchanged standard LW toy kernel.
# All file IO, worker probes, and diagnostics are outside the timed trajectory.
using LinearAlgebra, TOML
BLAS.set_num_threads(1)
push!(LOAD_PATH, normpath(joinpath(@__DIR__, "..", "..", "..", "research", "UniformLWAdvection")))
using UniformLWAdvection, Polyester
const Toy = UniformLWAdvection

function options(args)
    result = Dict{String,String}()
    for arg in args
        pair = split(arg, '='; limit=2)
        length(pair) == 2 || error("expected name=value argument: $arg")
        result[pair[1]] = pair[2]
    end
    result
end

function pin_workers(cpus)
    length(cpus) == Threads.nthreads(:default) || error("one CPU per Julia worker required")
    length(unique(cpus)) == length(cpus) || error("CPU IDs must be distinct")
    Sys.islinux() || error("worker pinning requires Linux")
    observations = Vector{Dict{String,Int}}(undef, length(cpus))
    Threads.@threads :static for slot in eachindex(cpus)
        Threads.threadid() == slot || error("unexpected Julia default-pool mapping")
        cpu = cpus[slot]
        0 <= cpu < 1024 || error("CPU ID outside cpu_set_t")
        mask = zeros(UInt8, 128)
        mask[cpu ÷ 8 + 1] = UInt8(1) << (cpu % 8)
        ccall(:sched_setaffinity, Cint, (Cint,Csize_t,Ptr{UInt8}),
              0, length(mask), mask) == 0 || error("sched_setaffinity failed")
        actual = zeros(UInt8, 128)
        ccall(:sched_getaffinity, Cint, (Cint,Csize_t,Ptr{UInt8}),
              0, length(actual), actual) == 0 || error("sched_getaffinity failed")
        actual == mask || error("worker affinity mismatch")
        observations[slot] = Dict("worker" => slot, "cpu" => cpu,
                                  "observed_cpu" => ccall(:sched_getcpu,Cint,()),
                                  "linux_tid" => ccall(:gettid,Cint,()))
    end
    observations
end

function probe!(state, work, initial, dt)
    copyto!(state.q, initial)
    state.t = 0.0
    Toy.step!(state, work, dt)
    expected = copy(state.q)
    expected_t = state.t
    copyto!(state.q, initial)
    state.t = 0.0
    Toy.track_workers!(work, true)
    recorded = try
        Toy.step!(state, work, dt)
        copy(work.worker_ids)
    finally
        Toy.track_workers!(work, false)
    end
    state.q == expected && state.t == expected_t ||
        error("tracked and untracked one-step states differ")
    nt = length(work.tiles)
    pass_ids = [sort!(unique(@view recorded[(pass-1)*nt+1:pass*nt])) for pass in 1:3]
    all(ids -> all(!iszero, ids), pass_ids) || error("one or more tiles were not probed")
    copyto!(state.q, initial)
    state.t = 0.0
    pass_ids
end

function run(o)
    mode = get(o, "mode", "measure")
    nx = parse(Int, o["nx"])
    ny = parse(Int, o["ny"])
    steps = parse(Int, get(o, "steps", "1"))
    tx = parse(Int, get(o, "tile-x", "512"))
    ty = parse(Int, get(o, "tile-y", "16"))
    nx > 0 && ny > 0 && (mode == "init" || steps > 0) ||
        error("invalid dimensions or steps")
    state = Toy.State(nx, ny; u=0.7, v=-0.3)
    if mode == "init"
        Toy.initialize!(state)
        open(o["output"], "w") do io
            write(io, state.q)
        end
        return
    end
    open(o["input"], "r") do io
        read!(io, state.q)
        eof(io) || error("input has trailing bytes")
    end
    initial = copy(state.q)
    dt = 0.35 / (0.7nx + 0.3ny)
    backend = mode == "oracle" ? :serial : :polyester
    work = Toy.Workspace(state; backend, tile=(tx,ty), threshold=16384,
                         flux=:recompute, carrier=:constant)
    cpus = mode == "oracle" ? Int[] : parse.(Int, split(o["cpus"], ','))
    affinity = mode == "oracle" ? Any[] : pin_workers(cpus)
    # Force compilation and establish the exact untracked execution path.
    Toy.simulate!(state, work, dt, 2)
    copyto!(state.q, initial)
    state.t = 0.0
    pass_ids = mode == "oracle" ? Any[] : probe!(state, work, initial, dt)
    GC.gc()
    started = time_ns()
    Toy.simulate!(state, work, dt, steps)
    elapsed = (time_ns() - started) / 1e9
    # Keep the full final field for an independent outside-timing comparison.
    open(o["output"], "w") do io
        write(io, state.q)
    end
    record = Dict{String,Any}(
        "language" => "julia", "mode" => mode,
        "nx" => nx, "ny" => ny, "steps" => steps, "workers" => Threads.nthreads(:default),
        "tile_x" => tx, "tile_y" => ty, "dt" => dt,
        "time_seconds" => elapsed,
        "seconds_per_unknown_step" => elapsed / (nx * ny * steps),
        "final_time" => state.t,
        "mass" => sum(state.q)/(nx*ny), "checksum" => sum(state.q),
        "sum_squares" => sum(abs2,state.q),
        "active_worker_ids_per_pass" => pass_ids,
        "active_worker_counts" => length.(pass_ids),
        "affinity" => affinity,
        "julia_version" => string(VERSION), "blas_threads" => BLAS.get_num_threads(),
        "numerical_array_bytes" => Toy.memory_bytes(state, work))
    open(o["report"], "w") do io
        TOML.print(io, record)
    end
end

run(options(ARGS))
