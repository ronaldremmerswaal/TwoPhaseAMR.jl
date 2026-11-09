# Package group of the audit of eq:forces:stages (and eq:forces:density, eq:forces:coefficient, eq:forces:body) (needs TwoPhaseAMR; run from the repository root).
#
#   julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-stages/scripts/run_code.jl
#
# Each script runs in its own testset; the outermost @testset throws if any check fails.

using Test

@testset "eq-forces-stages: checks against the TwoPhaseAMR implementation" begin
    for script in (
            "code_01_density_coefficient.jl",
            "code_02_stages_and_bounds.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
