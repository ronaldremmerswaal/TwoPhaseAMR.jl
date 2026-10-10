using Test
using UniformLWAdvection
using Base.Threads

include(joinpath(@__DIR__,"fused.jl"))
using .UniformLWFusedExperiment

@testset "temporally fused Strang tile prototype" begin
    @test_throws ArgumentError FusedWorkspace(State(8,8);tile=(0,4))
    @test_throws ArgumentError FusedWorkspace(State(8,8);backend=:unknown)
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
    end
end

@testset "fused state and scheduling guards" begin
    s = State(33,21)
    initialize!(s)
    w = FusedWorkspace(s;backend=:threads,tile=(7,4),threshold=0)
    @test_throws ArgumentError fused_step!(s,w,-1)
    @test_throws ArgumentError fused_step!(s,w,10)
    @test_throws ArgumentError fused_simulate!(s,w,0.01,-1)
    other = State(33,21)
    @test_throws ArgumentError fused_step!(other,w,0.001)
    set_velocity!(s,fill(0.2,33,21),fill(0.1,33,21))
    @test_throws ArgumentError fused_step!(s,w,0.001)
end

@testset "fused smooth translation order" begin
    errors = Float64[]
    for n in (32,64,128)
        s = State(n,n;u=0.67,v=-0.31)
        initialize!(s)
        w = FusedWorkspace(s;backend=:threads,tile=(11,9),threshold=0)
        final_time = 0.25
        nsteps = ceil(Int,final_time/stable_timestep(s))
        fused_simulate!(s,w,final_time/nsteps,nsteps)
        push!(errors,translation_error(s,final_time))
        @test abs(mass(s)-1) < 3e-14
    end
    @test all(>(1.8),log2.(errors[1:2] ./ errors[2:3]))
end
