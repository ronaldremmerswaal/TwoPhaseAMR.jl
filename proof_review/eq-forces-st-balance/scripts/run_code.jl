# Package group of the audit of eq:forces:st-balance (and eq:forces:st-jump) (needs TwoPhaseAMR; run from the repository root).
#
#   julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-st-balance/scripts/run_code.jl
#
# Each script runs in its own testset; the outermost @testset throws if any check fails.

using Test

@testset "eq-forces-st-balance: checks against the TwoPhaseAMR implementation" begin
    for script in (
            "code_01_const_curvature.jl",
            "code_02_jump_formula_and_limits.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
