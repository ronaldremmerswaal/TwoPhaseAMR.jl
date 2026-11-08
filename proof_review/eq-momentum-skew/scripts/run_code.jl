# Package group of the audit of eq:momentum:skew (needs TwoPhaseAMR; run from the repo root).
#
#   julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-momentum-skew/scripts/run_code.jl
#
# Each script runs in its own testset; the outermost @testset throws if any check fails.

using Test

@testset "eq-momentum-skew: checks against the TwoPhaseAMR implementation" begin
    for script in (
            "code_01_skew_periodic.jl",
            "code_02_skew_walls.jl",
            "sharp_01_arithmetic_mean.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
