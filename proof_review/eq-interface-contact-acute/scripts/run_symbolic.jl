# Symbolic group of the audit of eq:interface:contact-acute (needs Symbolics.jl).
#
#   julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/eq-interface-contact-acute/scripts/run_symbolic.jl
#
# Exits non-zero if any check fails (the outermost @testset throws).

using Test

@testset "eq:interface:contact-acute: symbolic checks" begin
    for script in (
            "step01_root_symbolic.jl",
            "sharp_02_rationalisation.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
