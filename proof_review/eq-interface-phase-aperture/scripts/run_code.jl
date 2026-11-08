# Package group of the audit of eq:interface:phase-aperture (needs TwoPhaseAMR; run from the repo root).
#
#   julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-interface-phase-aperture/scripts/run_code.jl
#
# Exits non-zero if any check fails (the outermost @testset throws).

using Test

@testset "eq:interface:phase-aperture: checks against the TwoPhaseAMR implementation" begin
    for script in (
            "code_01_phase_apertures.jl",
            "code_02_covered_neighbour.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
