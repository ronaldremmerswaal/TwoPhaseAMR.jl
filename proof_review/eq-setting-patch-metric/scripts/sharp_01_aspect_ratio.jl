# Sharpness 01: is "square cells" needed? Exact asymmetry for stretched cells; PD thresholds.
#
# Label:  eq:setting:patch-metric
# Claim:  03_discrete_setting.tex:291-305 (stretched hx = 2 hy: asymmetric at refinement corners).
# Results (hx = r hy, hy = fine cell height, sigma = 1 for a coarse cell, 0 for a fine one):
#   T12 - T21 = (r^2 - 1)/4 hy^2 (sNE + sSW - sNW - sSE),
#   => asymmetric iff r != 1 AND the corner is CFFF, CFCF or CCCF (unequal number of coarse cells on
#      the two diagonals); symmetric for FFFF, CCFF and for every hanging vertex whatever r;
#   => for every r != 1 there is a 2:1-graded mesh (one refined block) with an asymmetric patch,
#      so "square" cannot be weakened to "all aspect ratios": the hypothesis is sharp for the class.
#   Positive definiteness of sym(T) (the guard `isposdef` in _local_inverse_metric): with F = 4T
#   in units hy^2 (F11 = F22 = a, F12 = F21 = b for the three asymmetric classes),
#      det sym(T) = -(hy^4/64) b (r^2 - c r + 1)(2 a r + b (r^2 + 1)),   c = 2a/b = 10, 6, 22,
#   so sym(T) is positive definite iff r in (c - sqrt(c^2-4), c + sqrt(c^2-4))/2:
#   CFFF (0.101, 9.899), CFCF (0.172, 5.828), CCCF (0.0456, 21.95); CCFF / hanging / FFFF: all r.
# Run:    julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/eq-setting-patch-metric/scripts/sharp_01_aspect_ratio.jl

using Symbolics
using Test
isdefined(Main, :PatchModel) || include(joinpath(@__DIR__, "model.jl"))
using .PatchModel

@variables r hy

is_zero(x) = (v = Symbolics.value(Symbolics.simplify(Symbolics.expand(Num(x)))); v isa Number ? iszero(v) : false)

@testset "sharp_01: aspect ratio" begin
    classes = [
        "FFFF" => corner_cells(1, 1, 1, 1),
        "CFFF" => corner_cells(2, 1, 1, 1),
        "CCFF" => corner_cells(2, 2, 1, 1),
        "CFCF" => corner_cells(2, 1, 2, 1),
        "CCCF" => corner_cells(2, 2, 2, 1),
        "hanging" => hanging_cells(:W),
    ]
    c_expected = Dict("CFFF" => 10, "CFCF" => 6, "CCCF" => 22)
    for (name, cells) in classes
        F = moment_integers(faces_at_vertex(cells, 0, 0))
        T = physical_moment(F, r * hy, hy)                     # hx = r hy
        asym = T[1, 2] - T[2, 1]
        println(rpad(name, 8), " F = 4T/hy^2 = ", F, "    T12 - T21 = ", Symbolics.simplify(asym))
        if haskey(c_expected, name)
            @test !is_zero(asym)
            a, b = F[1, 1], F[1, 2]
            @test F[2, 2] == a && F[2, 1] == b                 # structure used in the factorisation
            c = c_expected[name]
            @test 2a == c * b
            S12 = (T[1, 2] + T[2, 1]) / 2
            det_sym = T[1, 1] * T[2, 2] - S12^2
            @test is_zero(det_sym + hy^4 / 64 * b * (r^2 - c * r + 1) * (2a * r + b * (r^2 + 1)))
            @test is_zero(asym - b * (r^2 - 1) / 4 * hy^2)
            rlo, rhi = (c - sqrt(c^2 - 4)) / 2, (c + sqrt(c^2 - 4)) / 2
            println("         sym(T) positive definite  <=>  r in (", round(rlo; digits = 4), ", ", round(rhi; digits = 4), ")")
            @test rlo * rhi ≈ 1
        else
            @test is_zero(asym)                                # symmetric for every aspect ratio
        end
    end
    # the text's stretched fixture: r = 2 at a CFFF corner gives (r^2-1)/4 = 3/4 (hy^2)
    F = moment_integers(faces_at_vertex(corner_cells(2, 1, 1, 1), 0, 0))
    T = physical_moment(F, 2.0, 1.0)
    @test T[1, 2] - T[2, 1] ≈ 3 / 4
end
