# Code group of the audit of ch:stability-accuracy (needs TwoPhaseAMR; run with --project=<repository root> or docs/techref/figures).
#
#   julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/ch-stability-accuracy/scripts/run_code.jl
#
# Exits non-zero if any check fails (the outermost @testset throws).

using Test

include(joinpath(@__DIR__, "_prelude.jl"))
@testset "ch-stability-accuracy: TwoPhaseAMR implementation" begin
    for script in (
            "code_01_bounds.jl",
            "code_02_selection_real.jl",
            "code_03_reproduce_macros.jl",
            "sharp_01_margin_bisection.jl",
            "sharp_02_margin_caveats.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
