# Package group of the audit of eq:momentum:ctu (needs TwoPhaseAMR; run from the repo root).
#
#   julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-momentum-ctu/scripts/run_code.jl
#
# Each script runs in its own testset; the outermost @testset throws if any check fails.

using Test

@testset "eq-momentum-ctu: checks against the TwoPhaseAMR implementation" begin
    for script in (
            "code_01_ctu_formula.jl",
            "code_02_ctu_boundedness.jl",
            "sharp_01_hanging_overshoot.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
