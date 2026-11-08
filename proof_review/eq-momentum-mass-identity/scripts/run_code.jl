# Package group of the audit of eq:momentum:mass-identity (needs TwoPhaseAMR; run from the repo root).
#
#   julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-momentum-mass-identity/scripts/run_code.jl
#
# Each script runs in its own testset; the outermost @testset throws if any check fails.

using Test

@testset "eq-momentum-mass-identity: checks against the TwoPhaseAMR implementation" begin
    for script in (
            "code_01_mass_identity.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
