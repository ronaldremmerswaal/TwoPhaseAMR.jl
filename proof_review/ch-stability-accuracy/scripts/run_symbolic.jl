# Symbolic and replica group of the audit of ch:stability-accuracy (needs Symbolics.jl for step01 only).
#
#   julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/ch-stability-accuracy/scripts/run_symbolic.jl
#
# Exits non-zero if any check fails (the outermost @testset throws).

using Test

@testset "ch-stability-accuracy: symbolic and replica checks" begin
    for script in (
            "step01_bounds_symbolic.jl",
            "step02_selection_replica.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
