# Symbolic group of the audit of eq:forces:map (and eq:forces:map-gradient, eq:forces:conforming) (Symbolics.jl).
#
#   julia --project=@claude-symbolics proof_review/eq-forces-map/scripts/run_symbolic.jl

using Test

@testset "eq-forces-map: symbolic checks" begin
    for script in (
            "step01_map_symbolic.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
