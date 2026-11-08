# Symbolic group of the audit of (no label; suggested prop:vof:boundedness) paragraph "Boundedness." of Section 5.5 (needs Symbolics.jl).
#
#   julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/vof-boundedness/scripts/run_symbolic.jl
#
# Exits non-zero if any check fails (the outermost @testset throws).

using Test

@testset "vof-boundedness: symbolic checks" begin
    for script in (
            "step01_boundedness_symbolic.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
