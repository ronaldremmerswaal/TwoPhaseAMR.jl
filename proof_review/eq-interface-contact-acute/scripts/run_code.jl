# Package group of the audit of eq:interface:contact-acute (needs TwoPhaseAMR; run from the repo root).
#
#   julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-interface-contact-acute/scripts/run_code.jl
#
# Exits non-zero if any check fails (the outermost @testset throws).

using Test

@testset "eq:interface:contact-acute: checks against the TwoPhaseAMR implementation" begin
    for script in (
            "code_01_acute_vs_package.jl",
            "sharp_01_evaluation_point.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
