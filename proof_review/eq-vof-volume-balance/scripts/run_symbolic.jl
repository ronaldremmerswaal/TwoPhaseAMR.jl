# Symbolic group of the audit of eq:vof:volume-balance (needs Symbolics.jl).
#
#   julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/eq-vof-volume-balance/scripts/run_symbolic.jl
#
# Exits non-zero if any check fails (the outermost @testset throws).

using Test

@testset "eq-vof-volume-balance: symbolic checks" begin
    for script in (
            "step01_volume_balance_symbolic.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
