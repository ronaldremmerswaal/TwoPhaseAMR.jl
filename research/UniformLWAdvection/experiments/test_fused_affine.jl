using Test
using UniformLWAdvection
using Base.Threads

include(joinpath(@__DIR__,"fused_affine.jl"))
using .UniformLWFusedAffineExperiment

@testset "affine fused Strang tile parity" begin
    for (nx,ny) in ((1,1),(1,7),(7,1),(2,3),(37,29),(128,73)),
        (u,v) in ((0.0,0.0),(0.7,-0.3),(-0.65,0.35)),
        backend in (:serial,:threads,:spawn),
        tile in ((8,5),(17,7),(128,2))
        baseline = State(nx,ny;u,v)
        candidate = State(nx,ny;u,v)
        initialize!(baseline)
        initialize!(candidate)
        initial_array = candidate.q
        reference = Workspace(baseline;backend=:serial,flux=:recompute)
        fused = FusedWorkspace(candidate;backend,tile,threshold=0,
                               batches_per_worker=3)
        dt = min(0.006,stable_timestep(candidate))
        m0 = mass(candidate)
        for _ in 1:5
            step!(baseline,reference,dt)
            fused_step!(candidate,fused,dt)
            @test candidate.q === initial_array
            @test maximum(abs.(candidate.q .- baseline.q)) <= 8e-15
        end
        @test abs(mass(candidate)-m0) < 8e-15
        @test candidate.t == baseline.t
    end
end

@testset "affine fused independent concurrent workspaces" begin
    if Threads.nthreads(:default) > 1
        states = [State(97,71;u=0.6,v=-0.2),State(93,67;u=-0.4,v=0.3)]
        references = [State(97,71;u=0.6,v=-0.2),State(93,67;u=-0.4,v=0.3)]
        foreach(initialize!,states)
        foreach(initialize!,references)
        works = [FusedWorkspace(s;backend=:threads,tile=(11,7),threshold=0)
                 for s in states]
        serial = [Workspace(s;backend=:serial) for s in references]
        dt = 0.003
        for k in eachindex(references), _ in 1:4
            step!(references[k],serial[k],dt)
        end
        tasks = [Threads.@spawn fused_simulate_amortized!(states[k],works[k],dt,4)
                 for k in eachindex(states)]
        foreach(fetch,tasks)
        for k in eachindex(states)
            @test maximum(abs.(states[k].q .- references[k].q)) <= 8e-15
        end
    end
end

@testset "affine fused guards" begin
    s = State(33,21)
    initialize!(s)
    w = FusedWorkspace(s;backend=:threads,tile=(7,4),threshold=0)
    @test_throws ArgumentError fused_step!(s,w,-1)
    @test_throws ArgumentError fused_step!(s,w,10)
    @test_throws ArgumentError fused_simulate!(s,w,0.01,-1)
    @test_throws ArgumentError fused_step!(State(33,21),w,0.001)
    set_velocity!(s,fill(0.2,33,21),fill(0.1,33,21))
    @test_throws ArgumentError fused_step!(s,w,0.001)
end

@testset "affine fused amortized trajectory" begin
    for (nx,ny) in ((1,1),(1,7),(7,1),(37,29),(128,73)),
        (u,v) in ((0.0,0.0),(0.7,-0.3),(-0.65,0.35)),
        backend in (:serial,:threads,:spawn), nsteps in 0:5
        baseline = State(nx,ny;u,v)
        candidate = State(nx,ny;u,v)
        initialize!(baseline)
        initialize!(candidate)
        reference = Workspace(baseline;backend=:serial,flux=:recompute)
        fused = FusedWorkspace(candidate;backend,tile=(8,5),threshold=0)
        q_identity = candidate.q
        dt = min(0.006,stable_timestep(candidate))
        for _ in 1:nsteps
            step!(baseline,reference,dt)
        end
        fused_simulate_amortized!(candidate,fused,dt,nsteps)
        @test candidate.q === q_identity
        @test maximum(abs.(candidate.q .- baseline.q)) <= 8e-15
        @test abs(mass(candidate)-mass(baseline)) <= 8e-15
        @test candidate.t == baseline.t
    end
    s = State(33,21)
    initialize!(s)
    w = FusedWorkspace(s;backend=:threads,tile=(7,4),threshold=0)
    q_before = copy(s.q)
    @test_throws ArgumentError fused_simulate_amortized!(s,w,0.01,-1)
    @test_throws ArgumentError fused_simulate_amortized!(s,w,-1,2)
    @test_throws ArgumentError fused_simulate_amortized!(s,w,10,2)
    @test_throws ArgumentError fused_simulate_amortized!(State(33,21),w,0.001,2)
    @test s.q == q_before
    @test s.t == 0
end

@testset "affine fused worker probe parity" begin
    for backend in (:serial,:threads), nsteps in (1,2,3)
        plain = State(257,193;u=-0.65,v=0.35)
        tracked = State(257,193;u=-0.65,v=0.35)
        initialize!(plain)
        initialize!(tracked)
        w_plain = FusedWorkspace(plain;backend,tile=(512,8),threshold=16384)
        w_tracked = FusedWorkspace(tracked;backend,tile=(512,8),threshold=16384)
        probe = WorkerProbe(w_tracked;nsteps)
        dt = min(0.003,stable_timestep(plain))
        if nsteps == 1
            fused_step!(plain,w_plain,dt)
            fused_step!(tracked,w_tracked,dt,probe)
        else
            fused_simulate_amortized!(plain,w_plain,dt,nsteps)
            fused_simulate_amortized!(tracked,w_tracked,dt,nsteps,probe)
        end
        @test plain.q == tracked.q
        @test plain.t == tracked.t
        @test all(pass -> !isempty(active_workers(probe,pass)),1:nsteps)
        @test isempty(active_copyback_workers(probe)) == iseven(nsteps)
        @test all(id -> 1 <= id <= Threads.nthreads(:default),probe.compute)
        @test all(id -> 0 <= id <= Threads.nthreads(:default),probe.copyback)
    end
end
