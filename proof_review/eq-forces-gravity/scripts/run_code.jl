# Package group of the audit of eq:forces:aperture-density (and eq:forces:gravity-jump, eq:forces:potential) (needs TwoPhaseAMR; run from the repository root).
#
#   julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-gravity/scripts/run_code.jl
#
# Each script runs in its own testset; the outermost @testset throws if any check fails.

using Test

@testset "eq-forces-gravity: checks against the TwoPhaseAMR implementation" begin
    for script in (
            "code_01_potential_density_jumps.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
