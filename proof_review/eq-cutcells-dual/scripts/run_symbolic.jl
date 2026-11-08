# Symbolic group of the audit of eq:cutcells:dual (needs Symbolics.jl).
#
#   julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/eq-cutcells-dual/scripts/run_symbolic.jl
#
# Exits non-zero if any check fails (the outermost @testset throws).

using Test

@testset "eq:cutcells:dual: symbolic checks" begin
    for script in (
            "step01_dual_symbolic.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
