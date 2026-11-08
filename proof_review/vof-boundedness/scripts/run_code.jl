# Package group of the audit of (no label; suggested prop:vof:boundedness) paragraph "Boundedness." of Section 5.5 (needs TwoPhaseAMR; run from the repo root).
#
#   julia --heap-size-hint=3G -t 1 --project=. proof_review/vof-boundedness/scripts/run_code.jl
#
# Each script runs in its own testset; the outermost @testset throws if any check fails.

using Test

@testset "vof-boundedness: checks against the TwoPhaseAMR implementation" begin
    for script in (
            "code_01_chain_and_disjointness.jl",
            "code_02_triangle_depth.jl",
            "code_03_issue52_strip_limitation.jl",
            "code_04_overlap_search.jl",
            "code_05_abstract_overlap.jl",
            "code_06_counterexample.jl",
            "sharp_01_cfl_thresholds.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
