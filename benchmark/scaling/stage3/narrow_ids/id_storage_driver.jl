# Fresh-process, source-specific regular face-ID storage diagnostic.
using LinearAlgebra, TOML, SHA, Profile
BLAS.set_num_threads(1)

function options(args)
    result = Dict{String,String}()
    for arg in args
        pair = split(arg, '='; limit=2)
        length(pair) == 2 || error("expected key=value: $arg")
        result[pair[1]] = pair[2]
    end
    result
end

const OPT = options(ARGS)
const PACKAGE_ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
const SOURCE_ROOT = abspath(get(OPT, "source", PACKAGE_ROOT))
Base.include(Main, joinpath(SOURCE_ROOT, "src", "TwoPhaseAMR.jl"))
using .TwoPhaseAMR

function pin_workers(cpus)
    length(cpus) == Threads.nthreads(:default) || error("one CPU per Julia worker required")
    length(unique(cpus)) == length(cpus) || error("CPU IDs must be distinct")
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

function scatter!(field, q)
    nx, ny = field.mesh.block_size
    for (b, block) in pairs(field.mesh.blocks)
        ix = round(Int, block.origin[1] * size(q,1)) + 1
        iy = round(Int, block.origin[2] * size(q,2)) + 1
        copyto!(TwoPhaseAMR.interior(field,b), @view q[ix:ix+nx-1,iy:iy+ny-1])
    end
    field
end

function gather(field, n)
    q = Matrix{Float64}(undef,n,n)
    nx, ny = field.mesh.block_size
    for (b, block) in pairs(field.mesh.blocks)
        ix = round(Int, block.origin[1] * n) + 1
        iy = round(Int, block.origin[2] * n) + 1
        copyto!(@view(q[ix:ix+nx-1,iy:iy+ny-1]), TwoPhaseAMR.interior(field,b))
    end
    q
end

function write_report(path, record)
    open(path,"w") do io
        TOML.print(io,record)
    end
end

function run(o)
    n = parse(Int,o["n"])
    block = parse(Int,get(o,"block",string(n)))
    n % block == 0 || error("block does not divide n")
    mode = get(o,"mode","setup")
    kernel = get(o,"kernel","upwind")
    kernel == "upwind" || error("this driver only measures general production upwind")
    t0 = time_ns()
    mesh = BlockTreeMesh(base_mesh=(n,n),block_size=(block,block),
                         periodic=(true,true))
    mesh_s = (time_ns()-t0)/1e9
    t0 = time_ns()
    field = CellField(mesh,Float64)
    velocity = FaceVelocity(mesh,(x,y,t,axis)->axis == 1 ? 0.7 : -0.3)
    workspace = AdvectionWorkspace(field)
    workspace_s = (time_ns()-t0)/1e9
    record = Dict{String,Any}(
        "mode"=>mode,"kernel"=>kernel,"n"=>n,"block"=>block,
        "blocks"=>length(mesh.blocks),"faces"=>length(mesh.faces),
        "halo_entries"=>length(mesh.halo_entries),
        "mesh_setup_seconds"=>mesh_s,"workspace_setup_seconds"=>workspace_s,
        "julia_version"=>string(VERSION),
        "threads"=>Threads.nthreads(:default),
        "source"=>SOURCE_ROOT,
    )
    if mode == "setup"
        xids, yids = workspace.x_face_ids, workspace.y_face_ids
        record["x_face_id_eltype"] = string(eltype(xids))
        record["y_face_id_eltype"] = string(eltype(yids))
        record["x_face_id_length"] = length(xids)
        record["y_face_id_length"] = length(yids)
        record["x_face_id_payload_bytes"] = sizeof(eltype(xids)) * length(xids)
        record["y_face_id_payload_bytes"] = sizeof(eltype(yids)) * length(yids)
        record["mesh_summary_bytes"] = Base.summarysize(mesh)
        record["workspace_summary_bytes"] = Base.summarysize(workspace)
        record["total_summary_bytes"] = Base.summarysize((mesh,field,velocity,workspace))
        write_report(o["report"],record)
        return
    end
    q0 = Matrix{Float64}(undef,n,n)
    open(o["input"],"r") do io
        read!(io,q0)
        eof(io) || error("input has trailing bytes")
    end
    scatter!(field,q0)
    gather(field,n) == q0 || error("initial scatter/gather differs")
    dt = 0.35 / n
    # The toy uses this constant-carrier CFL formula outside its timer.
    record["dt"] = dt
    cpus = parse.(Int,split(o["cpus"],','))
    record["affinity"] = pin_workers(cpus)
    stable_timestep(velocity,workspace;cfl=0.9)
    cfl_scan_seconds = Float64[]
    bound = 0.0
    for _ in 1:5
        t0 = time_ns()
        bound = stable_timestep(velocity,workspace;cfl=0.9)
        push!(cfl_scan_seconds,(time_ns()-t0)/1e9)
    end
    record["cfl_scan_seconds"] = sort(cfl_scan_seconds)[3]
    record["cfl_scan_samples_seconds"] = cfl_scan_seconds
    record["cfl_bound"] = bound
    record["cfl_query"] = 0.9
    dt <= bound*(1+1e-12) || error("CFL violation")
    steps = parse(Int,o["steps"])
    steps > 0 || error("steps must be positive")
    step!() = advect!(field,velocity,dt,workspace)
    # Compile and fault in all arrays before the measured trajectory.
    step!()
    if isdefined(TwoPhaseAMR,:_STAGE3_PROBE_FLUX)
        flux_seen = getfield(TwoPhaseAMR,:_STAGE3_PROBE_FLUX)
        gather_seen = getfield(TwoPhaseAMR,:_STAGE3_PROBE_GATHER)
        fill!(flux_seen,0)
        fill!(gather_seen,0)
        scatter!(field,q0)
        step!()
        record["active_flux_worker_ids"] = findall(!iszero,flux_seen)
        record["active_gather_worker_ids"] = findall(!iszero,gather_seen)
    end
    scatter!(field,q0)
    GC.gc()
    if mode == "profile"
        t0 = time_ns()
        for _ in 1:steps
            step!()
        end
        record["complete_steps_seconds"] = (time_ns()-t0)/1e9
        scatter!(field,q0)
        GC.gc()
        record["caller_allocated_bytes_one_step"] = @allocated step!()
        scatter!(field,q0)
        Profile.clear()
        Profile.@profile for _ in 1:steps
            step!()
        end
        open(o["profile"],"w") do io
            Profile.print(io;format=:flat,sortedby=:count)
        end
        record["steps"] = steps
        write_report(o["report"],record)
        return
    end
    t0 = time_ns()
    for _ in 1:steps
        step!()
    end
    elapsed = (time_ns()-t0)/1e9
    q = gather(field,n)
    open(o["output"],"w") do io
        write(io,q)
    end
    record["steps"] = steps
    record["time_seconds"] = elapsed
    record["ns_per_unknown_step"] = 1e9*elapsed/(n*n*steps)
    record["checksum"] = sum(q)
    record["sum_squares"] = sum(abs2,q)
    record["field_sha256"] = bytes2hex(sha256(reinterpret(UInt8,vec(q))))
    write_report(o["report"],record)
end

if haskey(OPT,"batch")
    cases = TOML.parsefile(OPT["batch"])["cases"]
    for case in cases
        settings = copy(OPT)
        delete!(settings,"batch")
        for (key,value) in case
            settings[key] = string(value)
        end
        run(settings)
        GC.gc()
    end
else
    run(OPT)
end
