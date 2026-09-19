using Test
using TwoPhaseAMR

@testset "Performance contracts" begin
    include("performance.jl")
    include("two_phase_performance.jl")
end
