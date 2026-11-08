# Package group of the audit of dev:interface:signs (needs TwoPhaseAMR; run from the repo root).
#
#   julia --heap-size-hint=3G -t 1 --project=. proof_review/dev-interface-signs/scripts/run_code.jl
#
# Exits non-zero if any check fails (the outermost @testset throws).

using Test

@testset "dev:interface:signs: checks against the TwoPhaseAMR implementation" begin
    for script in (
            "code_01_signs_sessile.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
