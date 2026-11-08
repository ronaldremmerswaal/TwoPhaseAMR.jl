# Symbolic group of the audit of eq:vof:fluctuation (needs Symbolics.jl).
#
#   julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/eq-vof-fluctuation/scripts/run_symbolic.jl
#
# Exits non-zero if any check fails (the outermost @testset throws).

using Test

@testset "eq-vof-fluctuation: symbolic checks" begin
    for script in (
            "step01_fluctuation_symbolic.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
