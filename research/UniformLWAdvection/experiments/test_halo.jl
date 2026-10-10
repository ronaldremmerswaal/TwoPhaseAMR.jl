using Test
using UniformLWAdvection

include(joinpath(@__DIR__,"halo.jl"))
using .UniformLWHaloExperiment

@testset "periodic halo-copy Strang prototype" begin
    for (nx,ny) in ((1,1),(1,9),(9,1),(2,3),(37,29),(128,73)),
        (u,v) in ((0.0,0.0),(0.7,-0.3),(-0.65,0.35)),
        backend in (:serial,:threads), tile in ((8,5),(17,7))
        core = State(nx,ny;u,v)
        initialize!(core)
        padded = PaddedState(core)
        reference = Workspace(core;backend=:serial,flux=:recompute)
        work = HaloWorkspace(padded;backend,tile,threshold=0)
        @test padded_memory_bytes(padded,work) == 5*(nx+2)*(ny+2)*sizeof(Float64)
        dt = min(0.006,stable_timestep(core))
        initial_mass = padded_mass(padded)
        q_identity = padded.q
        for _ in 1:5
            step!(core,reference,dt)
            padded_step!(padded,work,dt)
            @test padded.q === q_identity
            @test maximum(abs.(@view(padded.q[2:nx+1,2:ny+1]) .- core.q)) <= 8e-15
        end
        @test abs(padded_mass(padded)-initial_mass) < 8e-15
        @test padded.t == core.t
    end
end

@testset "halo prototype guards" begin
    s = State(33,21)
    initialize!(s)
    padded = PaddedState(s)
    @test_throws ArgumentError HaloWorkspace(padded;tile=(0,4))
    @test_throws ArgumentError HaloWorkspace(padded;backend=:unknown)
    work = HaloWorkspace(padded;backend=:threads,tile=(7,4),threshold=0)
    @test_throws ArgumentError padded_step!(padded,work,-1)
    @test_throws ArgumentError padded_step!(padded,work,10)
    @test_throws ArgumentError padded_simulate!(padded,work,0.01,-1)
    @test_throws ArgumentError padded_step!(PaddedState(s),work,0.001)
    padded.q = work.first
    @test_throws ArgumentError padded_step!(padded,work,0.001)
    set_velocity!(s,fill(0.2,33,21),fill(0.1,33,21))
    @test_throws ArgumentError PaddedState(s)
end
