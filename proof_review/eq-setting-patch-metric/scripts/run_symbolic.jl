# Symbolic / exact-model group of the audit of eq:setting:patch-metric (needs Symbolics.jl).
#
#   julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/eq-setting-patch-metric/scripts/run_symbolic.jl
#
# Exits non-zero if any check fails (the outermost @testset throws).

using Test

@testset "eq:setting:patch-metric: symbolic and exact-model checks" begin
    for script in (
            "step01_configurations_symbolic.jl",
            "step02_telescoping_symbolic.jl",
            "step03_exhaustive_enumeration.jl",
            "step04_wall_vertices.jl",
            "step05_hanging_example.jl",
            "sharp_01_aspect_ratio.jl",
            "sharp_02_grading.jl",
        )
        @testset "$script" begin
            include(joinpath(@__DIR__, script))
        end
    end
end
