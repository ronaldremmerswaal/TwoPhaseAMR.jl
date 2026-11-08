# Symbolic group of the audit of dev:interface:signs (needs Symbolics.jl).
#
#   julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/dev-interface-signs/scripts/run_symbolic.jl
#
# Exits non-zero if any check fails (the outermost @testset throws).

using Test

@testset "dev:interface:signs: symbolic checks" begin
    for script in (
            "step01_signs_symbolic.jl",
            "step02_convention_matrix.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
