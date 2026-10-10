using Test
using UniformLWAdvection
using Polyester

include(joinpath(@__DIR__,"fused_affine.jl"))
using .UniformLWFusedAffineExperiment

@testset "affine fused Polyester smoke" begin
    baseline = State(127,73;u=-0.65,v=0.35)
    candidate = State(127,73;u=-0.65,v=0.35)
    initialize!(baseline)
    initialize!(candidate)
    reference = Workspace(baseline;backend=:serial)
    fused = FusedWorkspace(candidate;backend=:polyester,tile=(16,9),threshold=0)
    dt = min(0.004,stable_timestep(candidate))
    for _ in 1:7
        step!(baseline,reference,dt)
        fused_step!(candidate,fused,dt)
    end
    @test maximum(abs.(candidate.q .- baseline.q)) <= 8e-15
    @test abs(mass(candidate)-mass(baseline)) <= 8e-15
    amortized = State(127,73;u=-0.65,v=0.35)
    initialize!(amortized)
    other = FusedWorkspace(amortized;backend=:polyester,tile=(16,9),threshold=0)
    q_identity = amortized.q
    fused_simulate_amortized!(amortized,other,dt,7)
    @test amortized.q === q_identity
    @test maximum(abs.(amortized.q .- baseline.q)) <= 8e-15
    tracked = State(127,73;u=-0.65,v=0.35)
    initialize!(tracked)
    wt = FusedWorkspace(tracked;backend=:polyester,tile=(16,9),threshold=0)
    probe = WorkerProbe(wt;nsteps=7)
    fused_simulate_amortized!(tracked,wt,dt,7,probe)
    @test tracked.q == amortized.q
    @test tracked.t == amortized.t
    @test all(pass -> !isempty(active_workers(probe,pass)),1:7)
    @test !isempty(active_copyback_workers(probe))
end
