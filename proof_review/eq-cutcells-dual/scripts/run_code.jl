# Package group of the audit of eq:cutcells:dual (needs TwoPhaseAMR; run from the repo root).
#
#   julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-cutcells-dual/scripts/run_code.jl
#
# Exits non-zero if any check fails (the outermost @testset throws).

using Test

@testset "eq:cutcells:dual: checks against the TwoPhaseAMR implementation" begin
    for script in (
            "code_01_dual_vs_package.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
