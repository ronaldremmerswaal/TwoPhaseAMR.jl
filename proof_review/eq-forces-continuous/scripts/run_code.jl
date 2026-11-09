# Package group of the audit of eq:forces:stress-balance (and eq:forces:steady, eq:forces:cont-momentum) (needs TwoPhaseAMR; run from the repository root).
#
#   julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-continuous/scripts/run_code.jl
#
# Each script runs in its own testset; the outermost @testset throws if any check fails.

using Test

@testset "eq-forces-continuous: checks against the TwoPhaseAMR implementation" begin
    for script in (
            "code_01_cell_curvature_sign.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
