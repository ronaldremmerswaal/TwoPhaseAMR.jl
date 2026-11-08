# Symbolic group of the audit of (no label; suggested prop:vof:frozen-dilation) frozen-carrier dilation vs mixture constraint (needs Symbolics.jl).
#
#   julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/vof-frozen-dilation/scripts/run_symbolic.jl
#
# Exits non-zero if any check fails (the outermost @testset throws).

using Test

@testset "vof-frozen-dilation: symbolic checks" begin
    for script in (
            "step01_mixture_symbolic.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
