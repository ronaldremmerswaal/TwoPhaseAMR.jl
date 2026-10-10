# Run from an environment that has Polyester installed and the toy package on
# LOAD_PATH, e.g. the repository environment stacked with the toy project.
using Test
using Polyester
using UniformLWAdvection
using Base.Threads

isempty(ARGS) || ARGS == ["--concurrent-only"] ||
    error("use --concurrent-only to run only the concurrency regression")

if isempty(ARGS)
@testset "optional Polyester extension" begin
    for carrier in (:constant,:variable), flux in (:recompute,:stored)
        a = State(53,31;u=0.57,v=-0.28)
        b = deepcopy(a)
        initialize!(a)
        initialize!(b)
        if carrier === :variable
            u = [0.1*sinpi(2i/53)*cospi(2j/31) for i in 1:53,j in 1:31]
            v = [0.1*cospi(2i/53)*sinpi(2j/31) for i in 1:53,j in 1:31]
            set_velocity!(a,u,v)
            set_velocity!(b,u,v)
        end
        wp = Workspace(a;backend=:polyester,carrier,flux,tile=(8,5),threshold=0)
        wr = Workspace(b;backend=:serial,carrier,flux)
        track_workers!(wp)
        step!(a,wp,0.002)
        step!(b,wr,0.002)
        @test a.q == b.q
        @test !isempty(active_workers(wp))
        @test length(active_workers(wp)) <= Threads.nthreads()
        track_workers!(wp,false)
        for _ in 1:4
            step!(a,wp,0.002)
            step!(b,wr,0.002)
        end
        @test a.q == b.q
    end
end
end

@testset "concurrent independent Polyester workspaces" begin
    # Uneven domains exceed the parallel threshold; each trajectory owns its
    # state and scratch. Run the two flux modes at the same time repeatedly.
    configs = (
        (257,193,0.61,-0.27,:recompute),
        (263,187,-0.43,0.29,:stored),
    )
    for _ in 1:3
        states = [State(nx,ny;u,v) for (nx,ny,u,v,_) in configs]
        foreach(initialize!,states)
        references = deepcopy(states)
        workspaces = [Workspace(states[k];backend=:polyester,
            flux=configs[k][5],tile=(32,8),threshold=16384)
            for k in eachindex(states)]
        serial = [Workspace(references[k];backend=:serial,
            flux=configs[k][5]) for k in eachindex(references)]
        identities = [s.q for s in states]
        dt = 0.002
        nsteps = 6
        for k in eachindex(references)
            simulate!(references[k],serial[k],dt,nsteps)
        end
        tasks = [Threads.@spawn simulate!(states[k],workspaces[k],dt,nsteps)
            for k in eachindex(states)]
        foreach(fetch,tasks)
        for k in eachindex(states)
            @test isequal(states[k].q,references[k].q)
            @test states[k].q === identities[k]
            @test states[k].t == references[k].t
        end
    end
end
