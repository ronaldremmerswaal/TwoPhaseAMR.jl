# Package group of the audit of (no label; suggested prop:vof:frozen-dilation) frozen-carrier dilation vs mixture constraint (needs TwoPhaseAMR; run from the repo root).
#
#   julia --heap-size-hint=3G -t 1 --project=. proof_review/vof-frozen-dilation/scripts/run_code.jl
#
# Each script runs in its own testset; the outermost @testset throws if any check fails.

using Test

@testset "vof-frozen-dilation: checks against the TwoPhaseAMR implementation" begin
    for script in (
            "code_01_mixture_constraint.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
