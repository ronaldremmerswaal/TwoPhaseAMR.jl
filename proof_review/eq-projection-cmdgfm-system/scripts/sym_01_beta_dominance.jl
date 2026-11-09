# Symbolic/numeric check 01: the CMDGFM weight and diagonal dominance (eq:projection:beta, eq:projection:dominance, dev:projection:beta).
#
# Labels: eq:projection:beta, eq:projection:dominance, dev:projection:beta
# Source: docs/techref/chapters/10_projection.tex:296-326, 400-420 (dominance), 452-460 (deviation)
# Claims:
#   B1  β = (1-s)/(|R|-s) satisfies β|R| + (1-β)s = 1 (the dominance bound is attained) for |R| > s;
#   B2  if the inference faces are dominated by their phase (m^l/m <= 1/2 at i^g, m^g/m <= 1/2 at i^l) then 0 <= s <= 1, and for |R| > 1:
#       0 <= β < 1 and the denominator |R| - s > 0 (the code's error branch and the clamp never act);
#   B3  β ↦ β|R| + (1-β)s is increasing iff |R| > s, so β is the LARGEST β in [0,1] with β|R| + (1-β)s <= 1; for |R| <= 1, β = 1 gives |R| <= 1;
#       the right inequality of (dominance) is an equality exactly when |R| >= 1;
#   B4  thesis weight β_th = 3x² - 2x³, x = |n_f·n_k| = n_k: β_th|R| = (3x - 2x²) sqrt(1-x²) has supremum 0.873 (as printed) -- and the
#       thesis weight does NOT bound the inference part: with s = 1 the row sum β_th|R| + (1-β_th)s exceeds 1 (maximum computed).
# Run:    julia --project=@claude-symbolics proof_review/eq-projection-cmdgfm-system/scripts/sym_01_beta_dominance.jl

using Symbolics, Test, Random
@variables R s
β = (1 - s) / (R - s)
@testset "beta algebra" begin
    @test isequal(Symbolics.simplify_fractions(β * R + (1 - β) * s - 1), 0)                  # B1
    # B2: 1 - β = (R - 1)/(R - s), β = (1-s)/(R-s): both fractions have a positive numerator/denominator when R > 1 >= s
    @test isequal(Symbolics.simplify_fractions((1 - β) - (R - 1) / (R - s)), 0)
    rng = MersenneTwister(3)
    for _ in 1:10_000
        r = 1 + 10rand(rng)^2; sv = rand(rng)                     # |R| > 1, s in [0,1]
        b = (1 - sv) / (r - sv)
        @test 0 <= b < 1 && r - sv > 0
        @test b * r + (1 - b) * sv ≈ 1 atol = 1.0e-14
    end
    # B3: f(β) = β|R| + (1-β)s has slope |R| - s: increasing iff |R| > s; so for |R| > 1 >= s the bound is attained exactly at β = (1-s)/(|R|-s) and exceeded for larger β
    f(b, r, sv) = b * r + (1 - b) * sv
    for _ in 1:10_000
        r = 5rand(rng); sv = rand(rng)
        if r <= 1
            @test f(1.0, r, sv) <= 1 + 1.0e-15                    # β = 1 allowed; equality iff |R| = 1 (s-independent): f(1) = |R|
        else
            b = (1 - sv) / (r - sv)
            @test f(b, r, sv) ≈ 1 atol = 1.0e-13
            @test f(min(1.0, b + 1.0e-3), r, sv) > 1 - 1.0e-13 && (b >= 1 - 1.0e-3 || f(b + 1.0e-3, r, sv) > 1)   # a larger β violates the bound
        end
    end
    # equality of the right inequality of (dominance) exactly when |R| >= 1 (for non-aligned rows): |R| < 1 gives |R| < 1; |R| >= 1 gives 1
    for r in (0.2, 0.7, 0.999), sv in (0.0, 0.5, 1.0)
        @test f(1.0, r, sv) < 1
    end
    for r in (1.0, 1.5, 7.0), sv in (0.0, 0.5, 0.99)
        b = r == 1.0 ? 1.0 : (1 - sv) / (r - sv)
        @test f(b, r, sv) ≈ 1 atol = 1.0e-14
    end
end
# B4: thesis weight
thesis(x) = begin
    β_ = 3x^2 - 2x^3; (β_ * sqrt(1 - x^2) / x, β_)
end
xs = range(1.0e-6, 1 - 1.0e-9; length = 2_000_001)
vals = [
    begin
        (br, b) = thesis(x); (br, br + (1 - b) * 1.0)
    end for x in xs
]
supbr, i1 = findmax(first, vals)
supfull, i2 = findmax(last, vals)
println("sup_x β_th |R| = ", supbr, " at x = ", xs[i1], ";  sup_x (β_th |R| + (1-β_th)·1) = ", supfull, " at x = ", xs[i2])
@testset "thesis weight" begin
    @test 0.8725 < supbr < 0.8735                                # "0.873" of dev:projection:beta
    @test supfull > 1.0                                          # the inference coefficients are not controlled (s = 1)
    # smallest s for which the thesis weight breaks weak dominance somewhere
    s_break = minimum(
        let s0 = sv
            any(
                    begin
                        (br, b) = thesis(x); br + (1 - b) * s0 > 1
                    end for x in xs[1:2000:end]
                ) ? s0 : Inf
        end for sv in 0:0.01:1
    )
    println("thesis weight violates β|R| + (1-β)s <= 1 for s >= ", s_break)
    @test s_break < 1
end
println("sym_01_beta_dominance: OK")
