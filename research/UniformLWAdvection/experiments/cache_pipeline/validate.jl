using Test
push!(LOAD_PATH, normpath(joinpath(@__DIR__,"..","..")))
using UniformLWAdvection
include(joinpath(@__DIR__, "CachePipeline.jl"))
const CP = CachePipeline
const U = UniformLWAdvection

function pair(nx,ny,tile; u=0.7,v=-0.3)
    a = U.State(nx,ny;u,v)
    b = U.State(nx,ny;u,v)
    U.initialize!(a)
    U.initialize!(b)
    # Add nonseparable content so a periodic seam or local-halo mistake is
    # visible even on grids for which the smooth translation fixture aliases.
    for j in 1:ny, i in 1:nx
        a.q[i,j] += 0.031*sin(i*1.7+j*0.9)
    end
    copyto!(b.q,a.q)
    a,b,U.Workspace(a;backend=:serial,flux=:recompute),
        CP.PipelineWorkspace(b;tile,threaded=true)
end

@testset "full-step tile pipeline" begin
    cases = ((1,1,(1,1)),(1,7,(1,3)),(7,1,(3,1)),
             (3,5,(2,2)),(7,11,(3,4)),(17,13,(8,5)),
             (31,19,(7,6)),(64,49,(16,7)))
    for (nx,ny,tile) in cases
        for steps in (1,2,3,4,7,8,16,32)
            a,b,wa,wb = pair(nx,ny,tile)
            original_q = b.q
            initial_mass = U.mass(b)
            dt = U.stable_timestep(a;cfl=0.3)
            for _ in 1:steps
                U.step!(a,wa,dt)
                CP.step!(b,wb,dt)
                @test b.q === original_q
                @test b.q == a.q
                @test b.t == a.t
            end
            @test isapprox(U.mass(b),U.mass(a);atol=1e-14,rtol=0)
            @test isapprox(U.mass(b),initial_mass;atol=5e-14,rtol=0)
        end
    end

    a,b,wa,wb = pair(17,13,(7,5))
    @test_throws ArgumentError CP.step!(b,CP.PipelineWorkspace(b;tile=(7,5)),Inf)
    wb.next = b.q
    @test_throws ArgumentError CP.step!(b,wb,U.stable_timestep(b))

    a,b,wa,wb = pair(17,13,(7,5))
    c,d,wc,wd = pair(17,13,(7,5);u=-0.2,v=0.6)
    @test wb.next !== wd.next
    @test all(x !== y for x in wb.scratch for y in wd.scratch)
    dta, dtc = U.stable_timestep(a), U.stable_timestep(c)
    U.simulate!(a,wa,dta,4)
    U.simulate!(c,wc,dtc,4)
    t1 = Threads.@spawn CP.simulate!(b,wb,dta,4)
    t2 = Threads.@spawn CP.simulate!(d,wd,dtc,4)
    fetch(t1); fetch(t2)
    @test b.q == a.q
    @test d.q == c.q

    # A smooth periodic translation should retain second-order convergence.
    errors = Float64[]
    for n in (16,32,64)
        s = U.State(n,n)
        U.initialize!(s)
        w = CP.PipelineWorkspace(s;tile=(7,5))
        nsteps = ceil(Int,0.2/U.stable_timestep(s;cfl=0.3))
        CP.simulate!(s,w,0.2/nsteps,nsteps)
        push!(errors,U.translation_error(s))
    end
    @test errors[1]/errors[2] > 3
    @test errors[2]/errors[3] > 3
    println("smooth errors: ",errors)

    allocations = Int[]
    for n in (128,512)
        s = U.State(n,n)
        U.initialize!(s)
        w = CP.PipelineWorkspace(s;tile=(32,16))
        dt = U.stable_timestep(s)
        CP.step!(s,w,dt)
        bytes = @allocated CP.step!(s,w,dt)
        @test bytes < 100_000
        push!(allocations,bytes)
        println("n=$n warm-step allocation bytes: ",bytes,
                "; numerical array bytes: ",CP.memory_bytes(s,w))
    end
    @test abs(allocations[1]-allocations[2]) < 10_000

    # The copy pass is timed and must actually use the requested team too.
    s = U.State(1024,1024)
    U.initialize!(s)
    w = CP.PipelineWorkspace(s;tile=(128,32),batches_per_worker=4)
    CP.track_workers!(w)
    CP.step!(s,w,U.stable_timestep(s))
    compute = CP.active_workers(w,:compute)
    copyback = CP.active_workers(w,:copyback)
    @test length(compute) == Threads.nthreads(:default)
    @test length(copyback) == Threads.nthreads(:default)
    println("threads=$(Threads.nthreads(:default)) compute=$compute copyback=$copyback")
end
