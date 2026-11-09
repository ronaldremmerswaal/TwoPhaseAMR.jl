# Package group of the audit of eq:forces:map (and eq:forces:map-gradient, eq:forces:conforming) (needs TwoPhaseAMR; run from the repository root).
#
#   julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-map/scripts/run_code.jl
#
# Each script runs in its own testset; the outermost @testset throws if any check fails.

using Test

@testset "eq-forces-map: checks against the TwoPhaseAMR implementation" begin
    for script in (
            "code_01_conforming.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
