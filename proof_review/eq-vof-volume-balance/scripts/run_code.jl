# Package group of the audit of eq:vof:volume-balance (needs TwoPhaseAMR; run from the repo root).
#
#   julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-vof-volume-balance/scripts/run_code.jl
#
# Each script runs in its own testset; the outermost @testset throws if any check fails.

using Test

@testset "eq-vof-volume-balance: checks against the TwoPhaseAMR implementation" begin
    for script in (
            "code_01_volume_balance.jl",
            "code_02_dilation_cellwise.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
