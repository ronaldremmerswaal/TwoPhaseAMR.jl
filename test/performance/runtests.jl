using Test
using TwoPhaseAMR

@testset verbose = true showtiming = true "Performance contracts" begin
    include("performance.jl")
    include("two_phase_performance.jl")
end
