# Package group of the audit of eq:forces:hydrostatic (needs TwoPhaseAMR; run from the repository root).
#
#   julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-hydrostatic/scripts/run_code.jl
#
# Each script runs in its own testset; the outermost @testset throws if any check fails.

using Test

@testset "eq-forces-hydrostatic: checks against the TwoPhaseAMR implementation" begin
    for script in (
            "code_01_planar_state.jl",
            "code_02_special_configurations.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
