using Test
using UniformLWAdvection
using Polyester

include(joinpath(@__DIR__,"halo.jl"))
using .UniformLWHaloExperiment

@testset "periodic halo-copy Polyester smoke" begin
    core = State(127,73;u=-0.65,v=0.35)
    initialize!(core)
    padded = PaddedState(core)
    reference = Workspace(core;backend=:serial)
    work = HaloWorkspace(padded;backend=:polyester,tile=(16,9),threshold=0)
    dt = min(0.004,stable_timestep(core))
    for _ in 1:7
        step!(core,reference,dt)
        padded_step!(padded,work,dt)
    end
    @test maximum(abs.(@view(padded.q[2:128,2:74]) .- core.q)) <= 8e-15
    @test abs(padded_mass(padded)-mass(core)) <= 8e-15
end
