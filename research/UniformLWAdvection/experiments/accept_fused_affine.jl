using Test
using UniformLWAdvection
using Polyester
using Base.Threads

include(joinpath(@__DIR__,"fused_affine.jl"))
using .UniformLWFusedAffineExperiment

const MASS_BUDGET = 512eps(Float64)
const TILE = (512,8)
const THRESHOLD = 16384

function fresh_pair(nx,ny,u,v)
    reference = State(nx,ny;u,v)
    candidate = State(nx,ny;u,v)
    initialize!(reference)
    initialize!(candidate)
    wr = Workspace(reference;backend=:serial,flux=:recompute)
    wc = FusedWorkspace(candidate;backend=:polyester,tile=TILE,
                        threshold=THRESHOLD)
    reference,candidate,wr,wc
end

function check_case(nx,ny,u,v,nsteps,amortized)
    reference,candidate,wr,wc = fresh_pair(nx,ny,u,v)
    identity = candidate.q
    m0 = mass(candidate)
    dt = min(0.006,stable_timestep(candidate))
    if amortized
        simulate!(reference,wr,dt,nsteps)
        fused_simulate_amortized!(candidate,wc,dt,nsteps)
        @test isequal(candidate.q,reference.q)
    else
        for _ in 1:nsteps
            step!(reference,wr,dt)
            fused_step!(candidate,wc,dt)
            @test isequal(candidate.q,reference.q)
        end
    end
    @test candidate.q === identity
    @test candidate.t == reference.t
    @test abs(mass(candidate)-m0) <= MASS_BUDGET
end

function check_order()
    errors = Float64[]
    for n in (32,64,128)
        s = State(n,n;u=0.67,v=-0.31)
        initialize!(s)
        w = FusedWorkspace(s;backend=:polyester,tile=TILE,
                           threshold=THRESHOLD)
        tfinal = 0.25
        nsteps = ceil(Int,tfinal/stable_timestep(s))
        fused_simulate_amortized!(s,w,tfinal/nsteps,nsteps)
        push!(errors,translation_error(s,tfinal))
        @test abs(mass(s)-1.0) <= MASS_BUDGET
    end
    rates = log2.(errors[1:2] ./ errors[2:3])
    @test all(>(1.8),rates)
    println("ORDER ",rates)
end

function check_probes(nsteps)
    reference,candidate,_,wc = fresh_pair(257,193,-0.65,0.35)
    plain = State(257,193;u=-0.65,v=0.35)
    initialize!(plain)
    wp = FusedWorkspace(plain;backend=:polyester,tile=TILE,
                        threshold=THRESHOLD)
    probe = WorkerProbe(wc;nsteps)
    dt = min(0.003,stable_timestep(candidate))
    identity = candidate.q
    if nsteps == 1
        fused_step!(plain,wp,dt)
        fused_step!(candidate,wc,dt,probe)
    else
        fused_simulate_amortized!(plain,wp,dt,nsteps)
        fused_simulate_amortized!(candidate,wc,dt,nsteps,probe)
    end
    @test isequal(candidate.q,plain.q)
    @test candidate.q === identity
    @test candidate.t == plain.t
    @test all(pass -> !isempty(active_workers(probe,pass)),1:nsteps)
    @test isempty(active_copyback_workers(probe)) == iseven(nsteps)
    @test all(id -> 1 <= id <= Threads.nthreads(:default),probe.compute)
    @test all(id -> 0 <= id <= Threads.nthreads(:default),probe.copyback)
    counts = [length(active_workers(probe,pass)) for pass in 1:nsteps]
    println("WORKERS nsteps=",nsteps," compute=",counts,
            " copyback=",length(active_copyback_workers(probe)))
end

function check_concurrent(amortized)
    configs = ((257,193,0.6,-0.2),(261,187,-0.4,0.3))
    states = [State(nx,ny;u,v) for (nx,ny,u,v) in configs]
    references = [State(nx,ny;u,v) for (nx,ny,u,v) in configs]
    foreach(initialize!,states)
    foreach(initialize!,references)
    work = [FusedWorkspace(s;backend=:polyester,tile=TILE,
                           threshold=THRESHOLD) for s in states]
    serial = [Workspace(s;backend=:serial) for s in references]
    identities = [s.q for s in states]
    dt = 0.003
    for k in eachindex(references)
        if amortized
            simulate!(references[k],serial[k],dt,4)
        else
            for _ in 1:4
                step!(references[k],serial[k],dt)
            end
        end
    end
    tasks = [Threads.@spawn begin
        if amortized
            fused_simulate_amortized!(states[k],work[k],dt,4)
        else
            for _ in 1:4
                fused_step!(states[k],work[k],dt)
            end
        end
    end for k in eachindex(states)]
    foreach(fetch,tasks)
    for k in eachindex(states)
        @test isequal(states[k].q,references[k].q)
        @test states[k].q === identities[k]
        @test states[k].t == references[k].t
    end
end

function allocation_sample(n)
    s = State(n,n)
    initialize!(s)
    w = FusedWorkspace(s;backend=:polyester,tile=TILE,
                       threshold=THRESHOLD)
    dt = min(0.001,stable_timestep(s))
    fused_step!(s,w,dt)
    fused_simulate_amortized!(s,w,dt,3)
    # Run the probes twice so their own method specialization is warm.
    @allocated fused_step!(s,w,dt)
    step_bytes = @allocated fused_step!(s,w,dt)
    @allocated fused_simulate_amortized!(s,w,dt,3)
    trajectory_bytes = @allocated fused_simulate_amortized!(s,w,dt,3)
    (step_bytes,trajectory_bytes)
end

if ARGS == ["--concurrent-only"]
    @testset "concurrent affine observable and amortized steps" begin
        for amortized in (false,true), _ in 1:2
            check_concurrent(amortized)
        end
    end
    println("CONCURRENT_ACCEPTED threads=",Threads.nthreads(:default),
            " Julia=",VERSION)
else
@testset "fresh-process affine acceptance" begin
    @test Threads.nthreads(:default) == parse(Int,ARGS[1])
    for (nx,ny) in ((1,1),(1,7),(7,1),(37,29),(257,193)),
        (u,v) in ((0.0,0.0),(0.7,-0.3),(-0.65,0.35))
        check_case(nx,ny,u,v,5,false)
        for nsteps in (0,1,2,3,4,5)
            check_case(nx,ny,u,v,nsteps,true)
        end
    end
    check_order()
    for nsteps in (1,3,4)
        check_probes(nsteps)
    end
    for amortized in (false,true), _ in 1:2
        check_concurrent(amortized)
    end
    small = allocation_sample(512)
    large = allocation_sample(2048)
    println("ALLOC 512=",small," 2048=",large)
    @test large[1] <= small[1]+4096
    @test large[2] <= small[2]+4096
end

println("ACCEPTED threads=",Threads.nthreads(:default),
        " repeat=",ARGS[2]," Julia=",VERSION)
end
