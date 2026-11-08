# Package group of the audit of eq:interface:area (needs TwoPhaseAMR; run from the repo root).
#
#   julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-interface-area/scripts/run_code.jl
#
# Exits non-zero if any check fails (the outermost @testset throws).

using Test

@testset "eq:interface:area: checks against the TwoPhaseAMR implementation" begin
    for script in (
            "code_01_area_vs_reference.jl",
            "code_02_inverse.jl",
            "code_03_open_polygon_intercept.jl",
            "sharp_01_reflected_branch.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
