# Symbolic group of the audit of sec:momentum:cfl (needs Symbolics.jl).
#
#   julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/sec-momentum-cfl/scripts/run_symbolic.jl
#
# Exits non-zero if any check fails (the outermost @testset throws).

using Test

@testset "sec-momentum-cfl: symbolic checks" begin
    for script in (
            "step01_courant_symbolic.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
