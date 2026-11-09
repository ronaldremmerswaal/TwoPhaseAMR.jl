# Symbolic group of the audit of eq:forces:stress-balance (and eq:forces:steady, eq:forces:cont-momentum) (Symbolics.jl).
#
#   julia --project=@claude-symbolics proof_review/eq-forces-continuous/scripts/run_symbolic.jl

using Test

@testset "eq-forces-continuous: symbolic checks" begin
    for script in (
            "step01_continuous_symbolic.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
