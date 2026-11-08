# Symbolic group of the audit of (no label; suggested eq:vof:projected-dilation-bound) projected-carrier dilation bound (needs Symbolics.jl).
#
#   julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/vof-projected-dilation/scripts/run_symbolic.jl
#
# Exits non-zero if any check fails (the outermost @testset throws).

using Test

@testset "vof-projected-dilation: symbolic checks" begin
    for script in (
            "step01_projected_dilation_symbolic.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
