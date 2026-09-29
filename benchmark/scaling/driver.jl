# One process, one independent synchronized sample. Results are machine-local TOML.
using LinearAlgebra, Statistics, TOML, SHA, Profile, Random
BLAS.set_num_threads(1)
const OPTIONS = Dict((split(argument, '='; limit=2)[1] => split(argument, '='; limit=2)[2]) for argument in ARGS)
option(key, fallback) = get(OPTIONS, key, fallback)
const CASE = option("case", "stream")
const BACKEND = option("backend", "threads")
const NX = parse(Int, option("nx", "512"))
const NY = parse(Int, option("ny", string(NX)))
const TILE = parse(Int, option("tile", "32"))
const TARGET = parse(Float64, option("seconds", "1.0"))
const NSTEPS = parse(Int, option("steps", "0"))
const OUTPUT = option("output", "/tmp/scaling-sample.toml")

if CASE in ("upwind", "upwind_complete", "wy_complete", "wy_kernel")
    @eval using TwoPhaseAMR
elseif CASE == "toy"
    toy_path = abspath(option("toy-path", normpath(joinpath(@__DIR__, "..", "..", "research", "UniformLWAdvection", "src", "UniformLWAdvection.jl"))))
    push!(LOAD_PATH, dirname(dirname(toy_path)))
    @eval using UniformLWAdvection
    if BACKEND == "polyester"
        @eval using Polyester
    end
end
if BACKEND in ("polyester_core", "polyester_thread")
    @eval using Polyester
end

function traverse(f, n)
    if BACKEND == "serial" || Threads.nthreads(:default) == 1
        for i in 1:n; f(i); end
    elseif BACKEND == "threads"
        Threads.@threads :static for i in 1:n; f(i); end
    elseif BACKEND == "polyester_core"
        polyester_core(f, n)
    elseif BACKEND == "polyester_thread"
        polyester_thread(f, n)
    else
        error("unsupported control backend $BACKEND")
    end
end
if BACKEND in ("polyester_core", "polyester_thread")
    @eval function polyester_core(f, n)
        Polyester.@batch per=core for i in 1:n; f(i); end
    end
    @eval function polyester_thread(f, n)
        Polyester.@batch per=thread for i in 1:n; f(i); end
    end
end

function workers_probe(work)
    # Each work item owns one Int entry; packed bits and thread-indexed scratch are avoided.
    observed = zeros(Int, work)
    traverse(work) do item; observed[item] = Threads.threadid(); end
    ids=sort(unique(observed))
    return ids, [count(==(id),observed) for id in ids]
end

function control_fixture()
    n = NX * NY
    source = reshape([1.0 + 0.2sinpi(2i / n) for i in 1:n], NX, NY)
    second = fill(0.3, NX, NY)
    output = zeros(NX, NY)
    # Fixed random permutation gives a synthetic destination-owned irregular
    # gather control. It is a proxy, not an AMR face/incidence schedule.
    indices = CASE == "gather" ? randperm(MersenneTwister(29092026), n) : Int[]
    tiles = cld(NY, TILE)
    function kernel!()
        traverse(tiles) do tile
            first = (tile - 1) * TILE + 1
            last = min(NY, tile * TILE)
            if CASE == "stream"
                @inbounds for j in first:last, i in 1:NX
                    output[i,j] = source[i,j] + 1.3second[i,j]
                end
            elseif CASE == "stencil"
                @inbounds for j in first:last, i in 1:NX
                    im = i == 1 ? NX : i - 1; ip = i == NX ? 1 : i + 1
                    jm = j == 1 ? NY : j - 1; jp = j == NY ? 1 : j + 1
                    output[i,j] = source[i,j] + 0.125 * (
                        source[im,j] + source[ip,j] + source[i,jm] + source[i,jp] - 4source[i,j])
                end
            elseif CASE == "gather"
                @inbounds for j in first:last, i in 1:NX
                    k = indices[i + NX * (j - 1)]
                    k1 = k == n ? 1 : k + 1
                    k2 = k > n - NX ? k + NX - n : k + NX
                    output[i,j] = 0.5source[k] + 0.25source[k1] + 0.25source[k2]
                end
            else
                error("unknown control $CASE")
            end
        end
        nothing
    end
    reset!() = nothing # Stationary inputs; each sweep completely overwrites output.
    trajectory!(steps) = (for _ in 1:steps; kernel!(); end; nothing)
    function diagnostics(steps)
        all(isfinite, output) || error("nonfinite control result")
        err = 0.0
        for j in 1:NY, i in 1:NX
            expected = if CASE == "stream"
                source[i,j] + 1.3second[i,j]
            elseif CASE == "stencil"
                source[i,j] + 0.125 * (source[mod1(i-1,NX),j] + source[mod1(i+1,NX),j] +
                    source[i,mod1(j-1,NY)] + source[i,mod1(j+1,NY)] - 4source[i,j])
            else
                k=indices[i+NX*(j-1)]
                0.5source[k]+0.25source[mod1(k+1,n)]+0.25source[mod1(k+NX,n)]
            end
            err=max(err,abs(output[i,j]-expected))
        end
        err<=1e-12 || error("control oracle mismatch: $err")
        Dict("max_error" => err, "checksum" => sum(output), "mass_error" => 0.0,
             "numerical_contract" => "stationary input, complete output overwrite")
    end
    traffic = CASE == "stream" ? 24 : CASE == "stencil" ? 48 : 40
    Dict("fixture"=>"stationary_control", "working_set_bytes"=>Base.summarysize((source,second,output,indices)),
         "traffic_bytes_per_cell_estimate"=>traffic, "traffic_model"=>"logical loads plus store; ignores cache reuse/write allocation",
         "cells"=>n, "tiles"=>tiles, "gather_index_policy"=>CASE=="gather" ? "fixed random permutation plus adjacent reads; synthetic AMR proxy" : "none"), reset!, trajectory!, diagnostics, ()->workers_probe(tiles)
end

function launch_fixture()
    items = parse(Int, option("launch-items", "128"))
    output = zeros(Int, items)
    function kernel!()
        traverse(items) do item; @inbounds output[item] = item; end
        nothing
    end
    trajectory!(steps) = (for _ in 1:steps; kernel!(); end; nothing)
    diagnostics(steps) = Dict("max_error"=>maximum(abs, output .- (1:items)), "mass_error"=>0.0,
                            "checksum"=>sum(output),"numerical_contract"=>"launch and complete barrier; one owned output per item")
    Dict("fixture"=>"synchronized_launch", "working_set_bytes"=>Base.summarysize(output),
         "traffic_bytes_per_cell_estimate"=>0,"cells"=>items,"tiles"=>items), ()->nothing, trajectory!, diagnostics, ()->workers_probe(items)
end

function package_fixture()
    block = parse(Int, option("block", "32"))
    # The declared large ladder divides this block; odd grids use a single root.
    bx, by = gcd(NX, block), gcd(NY, block)
    mesh = TwoPhaseAMR.BlockTreeMesh(base_mesh=(NX,NY), block_size=(bx,by),periodic=(true,true))
    exact(x,y,t=0.0) = 1 + 0.2sinpi(2(x - 0.7t)) * cospi(2(y + 0.3t))
    field = TwoPhaseAMR.CellField(mesh, (x,y)->exact(x,y))
    velocity = TwoPhaseAMR.FaceVelocity(mesh,(x,y,t,axis)->axis==1 ? 0.7 : -0.3)
    workspace = TwoPhaseAMR.AdvectionWorkspace(field)
    dt = min(0.35 / (0.7NX + 0.3NY), TwoPhaseAMR.stable_timestep(velocity,workspace;cfl=0.35))
    initial_mass = TwoPhaseAMR.integrate(field)
    initial_blocks = copy.(field.blocks)
    function reset!()
        for (values,initial) in zip(field.blocks, initial_blocks); copyto!(values,initial); end
        nothing
    end
    function trajectory!(steps)
        for s in 1:steps
            if CASE == "upwind_complete"
                selected_dt=TwoPhaseAMR.stable_timestep(velocity,workspace;cfl=0.35)
                selected_dt+1e-15>=dt || error("CFL contract changed")
            end
            TwoPhaseAMR.advect!(field,velocity,dt,workspace;time=(s-1)*dt)
        end
        nothing
    end
    function diagnostics(steps)
        error2 = 0.0; maximum_error = 0.0; checksum = 0.0
        for b in eachindex(mesh.blocks), j in 1:by, i in 1:bx
            x,y=TwoPhaseAMR.cellcenter(mesh,b,i,j)
            value=TwoPhaseAMR.interior(field,b)[i,j]
            err=abs(value-exact(x,y,steps*dt))
            error2 += mesh.blocks[b].volume * err^2
            maximum_error=max(maximum_error,err); checksum+=value
        end
        mass_error=abs(TwoPhaseAMR.integrate(field)-initial_mass)
        mass_error<=1e-10 || error("upwind conservation mismatch: $mass_error")
        Dict("mass_error"=>mass_error,"l2_error"=>sqrt(error2),
             "max_error"=>maximum_error,"checksum"=>checksum,"dt"=>dt,"final_time"=>steps*dt,
             "numerical_contract"=>"main-package first-order upwind/forward Euler; periodic smooth translation")
    end
    function active_workers()
        observed=zeros(Int, max(length(mesh.faces),NX*NY))
        TwoPhaseAMR._parallel_for(length(observed);grain=1) do k; observed[k]=Threads.threadid(); end
        ids=sort(unique(observed))
        ids,[count(==(id),observed) for id in ids]
    end
    Dict("fixture"=>"main_uniform_upwind", "cells"=>NX*NY,"blocks"=>length(mesh.blocks),
         "block_nx"=>bx,"block_ny"=>by,"faces"=>length(mesh.faces),
         "working_set_bytes"=>Base.summarysize((mesh,field,velocity,workspace,initial_blocks)),
         "traffic_bytes_per_cell_estimate"=>0, "traffic_model"=>"not modeled for face/incidence pipeline"),
         reset!,trajectory!,diagnostics,active_workers
end


function wy_fixture()
    block = parse(Int, option("block", "32"))
    bx, by = gcd(NX, block), gcd(NY, block)
    mesh = TwoPhaseAMR.BlockTreeMesh(base_mesh=(NX,NY),block_size=(bx,by),periodic=(false,false))
    shape(x,y) = hypot(x-0.5,y-0.75)-0.15
    field = TwoPhaseAMR.InterfaceField(mesh,shape;lipschitz=1,reconstruction_method=:lhf)
    scalar = TwoPhaseAMR.PhaseScalar(field;liquid=(x,y)->1+0.1sinpi(2x),
                                    gas=(x,y)->2+0.1cospi(2y))
    one = TwoPhaseAMR.PhaseScalar(field)
    velocity = TwoPhaseAMR.FaceVelocity(mesh)
    work = Ref(TwoPhaseAMR.WYAdvectionWorkspace(field))
    psi(x,y) = sinpi(x)^2*sinpi(y)^2/pi
    function setflow!(time)
        factor = cospi(time)
        for (k,face) in pairs(mesh.faces)
            x,y = face.center
            h = face.area/2
            value = face.axis==1 ?
                (psi(x,y+h)-psi(x,y-h))/face.area :
                -(psi(x+h,y)-psi(x-h,y))/face.area
            @inbounds velocity.interior[k] = factor*value
        end
        nothing
    end
    setflow!(0.0)
    peak_dt = TwoPhaseAMR.stable_timestep(velocity,work[];cfl=0.4)
    isfinite(peak_dt) && peak_dt>0 || error("invalid WY peak CFL")
    full_steps = ceil(Int,1/peak_dt)
    prescribed_steps = CASE=="wy_kernel" ? 1 : full_steps
    final_dt = CASE=="wy_kernel" ? peak_dt : 1-(full_steps-1)*peak_dt
    final_dt>0 && final_dt<=peak_dt*(1+64eps()) || error("invalid WY final step")
    diagnostic_mode = option("diagnostics","summary")
    diagnostic_mode in ("summary","sampled") || error("diagnostics must be summary or sampled")
    CASE=="wy_kernel" && diagnostic_mode!="summary" && error("WY kernel diagnostic uses summary diagnostics")
    interval = parse(Int,option("diagnostic-interval","10"))
    interval>0 || error("diagnostic interval must be positive")
    samples = zeros(Float64,cld(prescribed_steps,interval))
    initial_fraction = copy.(field.fraction.blocks)
    initial_liquid = copy.(scalar.liquid.blocks)
    initial_gas = copy.(scalar.gas.blocks)
    initial_one_liquid = copy.(one.liquid.blocks)
    initial_one_gas = copy.(one.gas.blocks)
    initial_volume = TwoPhaseAMR.integrate(field)
    initial_liquid_amount = TwoPhaseAMR.integrate(scalar,:liquid)
    initial_gas_amount = TwoPhaseAMR.integrate(scalar,:gas)
    function reset!()
        for (destination,source) in zip(field.fraction.blocks,initial_fraction);copyto!(destination,source);end
        for (destination,source) in zip(scalar.liquid.blocks,initial_liquid);copyto!(destination,source);end
        for (destination,source) in zip(scalar.gas.blocks,initial_gas);copyto!(destination,source);end
        for (destination,source) in zip(one.liquid.blocks,initial_one_liquid);copyto!(destination,source);end
        for (destination,source) in zip(one.gas.blocks,initial_one_gas);copyto!(destination,source);end
        field.geometry_is_valid[]=false
        field.phase_apertures_valid[]=false
        work[]=TwoPhaseAMR.WYAdvectionWorkspace(field)
        fill!(samples,0.0)
        setflow!(CASE=="wy_kernel" ? peak_dt/2 : 0.0)
        nothing
    end
    function trajectory!(steps)
        1<=steps<=prescribed_steps || error("WY trajectory must be a fixed T=1 prefix")
        if CASE=="wy_kernel"
            steps==1 || error("WY kernel diagnostic is exactly one reset step")
            TwoPhaseAMR.advect!(field,velocity,peak_dt,work[];scalars=(scalar,one),
                sweep_order=(1,2),scheme=TwoPhaseAMR.Fromm())
            return nothing
        end
        for k in 1:steps
            # Peak CFL is checked every step, independent of the reversing carrier.
            setflow!(0.0)
            limit=TwoPhaseAMR.stable_timestep(velocity,work[];cfl=0.4)
            dt=k==prescribed_steps ? final_dt : peak_dt
            dt<=limit*(1+64eps()) || error("WY CFL sequence changed")
            time=(k-1)*peak_dt
            setflow!(time+dt/2)
            TwoPhaseAMR.advect!(field,velocity,dt,work[];scalars=(scalar,one),
                sweep_order=iseven(k-1) ? (1,2) : (2,1),scheme=TwoPhaseAMR.Fromm())
            if diagnostic_mode=="sampled" && (k%interval==0 || k==prescribed_steps)
                samples[cld(k,interval)]=TwoPhaseAMR.integrate(field)
            end
        end
        nothing
    end
    function diagnostics(steps)
        volume_error=abs(TwoPhaseAMR.integrate(field)-initial_volume)
        liquid_error=abs(TwoPhaseAMR.integrate(scalar,:liquid)-initial_liquid_amount)
        gas_error=abs(TwoPhaseAMR.integrate(scalar,:gas)-initial_gas_amount)
        tolerance=16steps*length(mesh.faces)*eps()
        volume_error<=tolerance || error("WY volume conservation mismatch $volume_error > $tolerance")
        liquid_error<=tolerance || error("WY liquid amount mismatch $liquid_error > $tolerance")
        gas_error<=tolerance || error("WY gas amount mismatch $gas_error > $tolerance")
        reversal=0.0;centered=0.0;one_error=0.0
        minimum_fraction=Inf;maximum_fraction=-Inf
        minimum_intrinsic=Inf;maximum_intrinsic=-Inf
        h=mesh.halo
        for b in eachindex(mesh.blocks),j in 1:by,i in 1:bx
            current=field.fraction.blocks[b][i+h,j+h]
            original=initial_fraction[b][i+h,j+h]
            l=scalar.liquid.blocks[b][i+h,j+h]
            g=scalar.gas.blocks[b][i+h,j+h]
            isfinite(current) && isfinite(l) && isfinite(g) || error("nonfinite WY field")
            l0=initial_liquid[b][i+h,j+h]
            g0=initial_gas[b][i+h,j+h]
            one_error=max(one_error,abs(one.liquid.blocks[b][i+h,j+h]-1),
                          abs(one.gas.blocks[b][i+h,j+h]-1))
            volume=mesh.blocks[b].volume
            reversal+=volume*abs(current-original)
            centered+=volume*(original*abs(l-l0)+(1-original)*abs(g-g0))
            minimum_fraction=min(minimum_fraction,current)
            maximum_fraction=max(maximum_fraction,current)
            minimum_intrinsic=min(minimum_intrinsic,l,g)
            maximum_intrinsic=max(maximum_intrinsic,l,g)
        end
        one_error<=1e-11 || error("WY compatible constant-one identity mismatch $one_error")
        minimum_fraction>=0 && maximum_fraction<=1 || error("WY fraction out of bounds")
        checks=Dict("mass_error"=>volume_error,"volume_error"=>volume_error,
            "liquid_amount_error"=>liquid_error,"gas_amount_error"=>gas_error,
            "conservation_budget"=>tolerance,"constant_one_max_error"=>one_error,
            "fraction_l1_change"=>reversal,"centered_l1_change"=>centered,
            "fraction_min"=>minimum_fraction,"fraction_max"=>maximum_fraction,
            "intrinsic_min"=>minimum_intrinsic,"intrinsic_max"=>maximum_intrinsic,
            "sampled_volume_max_drift"=>diagnostic_mode=="sampled" ? maximum(abs.(samples .- initial_volume)) : 0.0,
            "sampled_volume_count"=>diagnostic_mode=="sampled" ? count(!iszero,samples) : 0,
            "checksum"=>TwoPhaseAMR.integrate(field),"dt"=>peak_dt,
            "final_time"=>CASE=="wy_complete" && steps==prescribed_steps ? 1.0 : steps*peak_dt,
            "numerical_contract"=>CASE=="wy_kernel" ?
                "one frozen peak-midpoint carrier WY+compatible phase scalar step; LHF, Fromm, CFL 0.4" :
                "closed-wall T=1 reversal; LHF/WY, Fromm centered phase scalar, one velocity, cfl 0.4 peak sequence")
        if CASE=="wy_complete"
            checks["reversal_l1_fraction"]=reversal
            checks["reversal_l1_centered"]=centered
        end
        checks
    end
    function active_workers()
        observed=zeros(Int,max(length(mesh.faces),NX*NY))
        TwoPhaseAMR._parallel_for(length(observed);grain=1) do k;observed[k]=Threads.threadid();end
        ids=sort(unique(observed))
        ids,[count(==(id),observed) for id in ids]
    end
    Dict("fixture"=>CASE=="wy_kernel" ? "uniform_wy_compatible_centered_one_step" : "uniform_wy_compatible_centered_reversal",
         "cells"=>NX*NY,"blocks"=>length(mesh.blocks),"block_nx"=>bx,"block_ny"=>by,
         "faces"=>length(mesh.faces),"working_set_bytes"=>Base.summarysize((mesh,field,scalar,one,velocity,work[],initial_fraction,initial_liquid,initial_gas,initial_one_liquid,initial_one_gas,samples)),
         "diagnostics_mode"=>diagnostic_mode,"diagnostic_interval"=>interval,
         "prescribed_steps"=>prescribed_steps,"peak_cfl_dt"=>peak_dt,"last_dt"=>final_dt,
         "traffic_bytes_per_cell_estimate"=>0,"traffic_model"=>"not modeled for geometric WY reconstruction"),
         reset!,trajectory!,diagnostics,active_workers
end

function toy_fixture()
    toy = UniformLWAdvection
    state = toy.State(NX,NY;u=0.7,v=-0.3)
    flux = Symbol(option("flux", "recompute"))
    workspace=toy.Workspace(state;flux,backend=Symbol(BACKEND),tile=(parse(Int,option("tile-x","128")),TILE),threshold=parse(Int,option("threshold","16384")))
    dt=0.35/(0.7NX+0.3NY)
    reset!()=toy.initialize!(state;t=0.0)
    reset!(); initial_mass=toy.mass(state)
    trajectory!(steps)=toy.simulate!(state,workspace,dt,steps)
    function diagnostics(steps)
        error=toy.translation_error(state,steps*dt;u=0.7,v=-0.3)
        mass_error=abs(toy.mass(state)-initial_mass)
        mass_error<=1e-10 || error("toy conservation mismatch: $mass_error")
        Dict("mass_error"=>mass_error,"l2_error"=>error,"checksum"=>sum(state.q),"dt"=>dt,"final_time"=>steps*dt,
             "numerical_contract"=>"periodic cell-average Strang directional Lax-Wendroff translation")
    end
    function active_workers()
        reset!()
        toy.track_workers!(workspace,true)
        toy.step!(state,workspace,dt)
        ids=toy.active_workers(workspace)
        visits=filter(!iszero,workspace.worker_ids)
        counts=[count(==(id),visits) for id in ids]
        toy.track_workers!(workspace,false)
        ids,counts
    end
    Dict("fixture"=>"standalone_uniform_lw", "cells"=>NX*NY,
         "working_set_bytes"=>Base.summarysize((state,workspace)), "flux"=>string(flux),
         "toy_source_sha256"=>bytes2hex(sha256(read(toy_path))),
         "toy_polyester_extension_sha256"=>BACKEND=="polyester" ? bytes2hex(sha256(read(joinpath(dirname(dirname(toy_path)),"ext","UniformLWAdvectionPolyesterExt.jl")))) : "not_loaded",
         "traffic_bytes_per_cell_estimate"=>(flux==:stored ? 192 : 144),
         "traffic_model"=>"constant-carrier logical references over 3 Strang sweeps: recompute 3q+2speed+1store each; stored (2q+1speed+1fluxstore)+(1q+2flux+1store) each; ignores cache reuse/write allocation"),
         reset!,trajectory!,diagnostics,active_workers
end


const WORKER_AFFINITY = Ref{Any}(nothing)

function linux_thread_affinity()
    Sys.islinux() || return Int[]
    mask=zeros(UInt8,128) # Linux cpu_set_t for CPU IDs below 1024.
    status=ccall(:sched_getaffinity,Cint,(Cint,Csize_t,Ptr{UInt8}),0,length(mask),mask)
    status==0 || error("sched_getaffinity failed with errno $(Libc.errno())")
    [cpu for cpu in 0:1023 if (mask[cpu÷8+1] & (UInt8(1) << (cpu%8))) != 0]
end

function pin_linux_thread!(cpu::Int)
    Sys.islinux() || error("--pin-workers requires Linux")
    0<=cpu<1024 || error("CPU ID outside 128-byte Linux affinity mask")
    mask=zeros(UInt8,128)
    mask[cpu÷8+1] |= UInt8(1) << (cpu%8)
    status=ccall(:sched_setaffinity,Cint,(Cint,Csize_t,Ptr{UInt8}),0,length(mask),mask)
    status==0 || error("sched_setaffinity CPU $cpu failed with errno $(Libc.errno())")
    nothing
end

function configure_worker_affinity!()
    pin=option("pin-workers","false")=="true"
    cpus=pin ? parse.(Int,split(option("pin-cpus",""),',')) : Int[]
    count=Threads.nthreads(:default)
    pin && length(cpus)!=count && error("need one distinct pinned CPU per default worker")
    pin && length(unique(cpus))!=count && error("pinned CPUs must be distinct")
    records=Vector{Dict{String,Any}}(undef,count)
    Threads.@threads :static for slot in 1:count
        worker=Threads.threadid()
        worker==slot || error("static worker mapping changed: slot $slot ran on $worker")
        pin && pin_linux_thread!(cpus[slot])
        allowed=linux_thread_affinity()
        pin && allowed!=[cpus[slot]] && error("pinning was not retained for Julia worker $slot")
        records[slot]=Dict("julia_worker_id"=>worker,
            "linux_tid"=>Sys.islinux() ? ccall(:gettid,Cint,()) : -1,
            "current_cpu"=>Sys.islinux() ? ccall(:sched_getcpu,Cint,()) : -1,
            "allowed_cpus"=>allowed)
    end
    WORKER_AFFINITY[]=records
    nothing
end

function runtime_metadata()
    gc_threads = isdefined(Threads, :ngcthreads) ? Threads.ngcthreads() : -1
    affinity = Sys.islinux() ? match(r"Cpus_allowed_list:\s*([^\n]+)",read("/proc/self/status",String)) : nothing
    Dict("julia_version"=>string(VERSION),"julia_executable"=>joinpath(Sys.BINDIR,Base.julia_exename()),
         "threads_default"=>Threads.nthreads(:default),"threads_interactive"=>Threads.nthreads(:interactive),
         "gc_threads"=>gc_threads,"blas_threads"=>BLAS.get_num_threads(),"blas_config"=>string(BLAS.get_config()),
         "process_affinity"=>isnothing(affinity) ? "unknown" : strip(affinity[1]),
         "process_affinity_scope"=>"calling Julia thread; other workers recorded separately",
         "per_worker_affinity"=>WORKER_AFFINITY[],
         "pin_workers"=>option("pin-workers","false")=="true",
         "cpu_model"=>Sys.cpu_info()[1].model,"total_memory"=>Sys.total_memory(),
         "gc_retained"=>true,"precision"=>"Float64","runtime_scheduler"=>BACKEND,
         "tile_columns"=>TILE,"manifest_sha256"=>isfile(joinpath(dirname(Base.active_project()),"Manifest.toml")) ?
             bytes2hex(sha256(read(joinpath(dirname(Base.active_project()),"Manifest.toml")))) : "missing")
end

function main()
    configure_worker_affinity!()
    startup=time_ns()
    fixture,reset!,trajectory!,diagnostics,active_workers = CASE=="launch" ? launch_fixture() : CASE in ("upwind","upwind_complete") ? package_fixture() : CASE in ("wy_complete","wy_kernel") ? wy_fixture() : CASE=="toy" ? toy_fixture() : control_fixture()
    construction_seconds=(time_ns()-startup)*1e-9
    reset!(); first=@timed trajectory!(1)
    reset!(); trajectory!(min(2,get(fixture,"prescribed_steps",2))) # Every execution path compiled and warmed.
    steps=NSTEPS
    if CASE in ("wy_complete","wy_kernel")
        expected=fixture["prescribed_steps"]
        steps==0 || steps==expected || error("WY must use the full fixed T=1 trajectory ($expected steps)")
        steps=expected
    end
    calibration_seconds=0.0
    if steps==0
        steps=1
        while true
            reset!(); sample=@timed trajectory!(steps)
            calibration_seconds += sample.time
            sample.time>=TARGET && break
            # Modest growth bounds temporary calibration trajectories.
            steps *= max(2,min(8,ceil(Int,TARGET/max(sample.time,1e-8))))
            steps<=100_000_000 || error("calibration exceeded trajectory cap")
        end
    end
    if option("profile", "false") == "true"
        reset!()
        Profile.clear()
        Profile.@profile trajectory!(steps)
        open(option("profile-output", OUTPUT * ".profile.txt"), "w") do io
            println(io,"Aggregate flat profile (sample counts, not wall-time fractions)")
            Profile.print(io;format=:flat,sortedby=:count)
            println(io,"\nPer-thread flat profile (actual sampled kernel stacks)")
            Profile.print(io;format=:flat,sortedby=:count,groupby=:thread)
        end
    end
    reset!()
    GC.gc() # GC remains enabled inside the measured application trajectory.
    timed=@timed trajectory!(steps)
    checks=diagnostics(steps)
    all(isfinite,values(filter(pair->pair.second isa Number,checks))) || error("nonfinite diagnostics")
    workers,worker_items=active_workers()
    reset!()
    record=Dict("schema_version"=>1,"case"=>CASE,"backend"=>BACKEND,"nx"=>NX,"ny"=>NY,"steps"=>steps,
                "time_seconds"=>timed.time,"allocation_bytes"=>timed.bytes,"gc_seconds"=>timed.gctime,"peak_rss_bytes_linux"=>Sys.islinux() ? Sys.maxrss() : -1,
                "first_call_seconds"=>first.time,"first_call_allocation_bytes"=>first.bytes,
                "construction_seconds"=>construction_seconds,"calibration_seconds"=>calibration_seconds,
                "active_worker_ids"=>workers,"active_workers"=>length(workers),
                "worker_probe_items"=>worker_items,
                "worker_probe_imbalance_ratio"=>maximum(worker_items)/max(1,Statistics.mean(worker_items)),
                "active_worker_scope"=>CASE in ("upwind","upwind_complete","wy_complete","wy_kernel") ? "package scheduler probe outside timed path; not full timestep instrumentation" : "same backend and work partition outside timing",
                "fixture"=>fixture,"diagnostics"=>checks,"runtime"=>runtime_metadata(),
                "timed_scope"=>CASE=="upwind" ? "fixed-dt advection kernel" : CASE=="upwind_complete" ? "stable timestep selection plus advection" : CASE=="toy" ? "complete Strang transport step" : CASE=="wy_complete" ? "full T=1 WY+compatible phase scalar reversal including peak CFL, carrier sampling, reconstruction and declared diagnostics" : CASE=="wy_kernel" ? "one frozen-carrier WY+compatible phase scalar step including reconstruction" : "synchronized control kernel")
    open(OUTPUT,"w") do io; TOML.print(io,record); end
    println("$CASE/$BACKEND $(NX)x$(NY): $steps steps in $(round(timed.time;digits=6)) seconds; active worker probe=$(length(workers))")
end
main()
