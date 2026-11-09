# Package group of the audit of eq:forces:two-velocity (issue #76) (needs TwoPhaseAMR; run from the repository root).
#
#   julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-two-velocity/scripts/run_code.jl
#
# Each script runs in its own testset; the outermost @testset throws if any check fails.

using Test

@testset "eq-forces-two-velocity: checks against the TwoPhaseAMR implementation" begin
    for script in (
            "code_01_map_relation.jl",
            "code_02_hydrostatic_projection.jl",
            "code_03_drop_tension.jl",
            "code_04_full_step.jl",
            "sharp_01_condition_cases.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
