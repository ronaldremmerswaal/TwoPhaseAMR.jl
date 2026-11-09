# Package group of the audit of eq:forces:balance (needs TwoPhaseAMR; run from the repository root).
#
#   julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-balance/scripts/run_code.jl
#
# Each script runs in its own testset; the outermost @testset throws if any check fails.

using Test

@testset "eq-forces-balance: checks against the TwoPhaseAMR implementation" begin
    for script in (
            "code_01_gradient_identity.jl",
            "code_02_projection_returns.jl",
            "code_03_iff_injectivity.jl",
            "sharp_01_operator_mismatch.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
