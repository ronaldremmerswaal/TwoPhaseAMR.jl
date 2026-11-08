# Package group of the audit of eq:setting:patch-metric (needs TwoPhaseAMR; run from the repo root).
#
#   julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-setting-patch-metric/scripts/run_code.jl
#
# Exits non-zero if any check fails (the outermost @testset throws).

using Test

@testset "eq:setting:patch-metric: checks against the TwoPhaseAMR implementation" begin
    for script in (
            "code_01_square_meshes.jl",
            "code_02_stretched.jl",
            "code_03_assumptions.jl",
            "code_04_balance_defect.jl",
            "code_05_balance_diagnosis.jl",       # redefines two internal functions temporarily: keep last
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
