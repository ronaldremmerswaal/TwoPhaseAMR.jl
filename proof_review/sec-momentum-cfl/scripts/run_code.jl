# Package group of the audit of sec:momentum:cfl (needs TwoPhaseAMR; run from the repo root).
#
#   julia --heap-size-hint=3G -t 1 --project=. proof_review/sec-momentum-cfl/scripts/run_code.jl
#
# Each script runs in its own testset; the outermost @testset throws if any check fails.

using Test

@testset "sec-momentum-cfl: checks against the TwoPhaseAMR implementation" begin
    for script in (
            "code_01_courant.jl",
            "sharp_01_courant_extremizer.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
