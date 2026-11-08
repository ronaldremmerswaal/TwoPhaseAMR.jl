# Package group of the audit of eq:interface:contact-row (needs TwoPhaseAMR; run from the repo root).
#
#   julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-interface-contact-row/scripts/run_code.jl
#
# Exits non-zero if any check fails (the outermost @testset throws).

using Test

@testset "eq:interface:contact-row: checks against the TwoPhaseAMR implementation" begin
    for script in (
            "code_01_contact_row_vs_package.jl",
            "sharp_01_linearisation.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
