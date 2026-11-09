# Symbolic check 01: the face data and the phase decomposition (eq:projection:face-data, phase-gradient, phase-identities, decomposition),
# and the pointwise part of mixture-update.
#
# Labels: eq:projection:face-data, eq:projection:phase-gradient, eq:projection:phase-identities, eq:projection:decomposition,
#         eq:projection:mixture-update (the face-wise identity)
# Source: docs/techref/chapters/10_projection.tex:164-258
# Claims (per face; m^l, m^g > 0, a = a^l + a^g, ζ^l = -m^g, ζ^g = m^l, m = m^l + m^g, m^r = m^l m^g / m, apgfm = (a^l ζ^l + a^g ζ^g)/m):
#   F1  ζ^g - ζ^l = m ;  m^l ζ^l + m^g ζ^g = 0                                                      (used before phase-identities)
#   P1  S^φ = S + ζ^φ ξ / m  =>  S^g - S^l = ξ ;  m^l S^l + m^g S^g = m S                           (phase-identities)
#   D1  ū = (m^l u^l + m^g u^g)/m, [u] = u^g - u^l  =>  u^φ = ū + ζ^φ [u]/m                         (decomposition, line 1)
#   D2  m^l (u^l)² + m^g (u^g)² = m ū² + m^r [u]²                                                    (decomposition, line 1)
#   D3  a^l u^l + a^g u^g = a ū + apgfm [u]                                                          (decomposition, line 2)
#   D4  after the update u^φ ← u^φ - dt S^φ:  ū ← ū - dt S,  [u] ← [u] - dt ξ,
#       a^l u^l + a^g u^g ← … - dt (a S + apgfm ξ)                                                   (mixture-update, face-wise)
#   D5  without solid, a^l = ĉ, a^g = 1 - ĉ, m^l = ρ^l ĉ, m^g = ρ^g (1-ĉ):  apgfm = ĉ(1-ĉ)(ρ^l - ρ^g)/m   (stated after mixture-update)
#   D6  momentum identity of the correction: Σ_φ m^φ S^φ = m S  (cell sums of the momentum change are those of the common gradient)
# Run:    julia --project=@claude-symbolics proof_review/eq-projection-two-velocity/scripts/sym_01_phase_algebra.jl

using Symbolics, Test
@variables ml mg al ag ul ug S xi dt rl rg c
m = ml + mg
zl = -mg; zg = ml
# zero test: simplify_fractions (symbolic), and exact rational evaluation at 20 random points (Schwartz-Zippel)
using Random
const RNG = MersenneTwister(11)
const VARS = [ml, mg, al, ag, ul, ug, S, xi, dt, rl, rg, c]
function zero_(e)
    ok = isequal(Symbolics.simplify_fractions(e), 0)
    ok && return true
    for _ in 1:20
        d = Dict(v => rand(RNG, 1:97) // rand(RNG, 1:13) for v in VARS)
        val = Symbolics.substitute(e, d)
        isequal(val, 0) || return false
    end
    return true
end
mr = ml * mg / m
apg = (al * zl + ag * zg) / m
a = al + ag
ub = (ml * ul + mg * ug) / m
jump = ug - ul
Sl = S + zl / m * xi; Sg = S + zg / m * xi
@testset "two-velocity phase algebra" begin
    @test zero_((zg - zl) - m)                                    # F1
    @test zero_(ml * zl + mg * zg)                                # F1
    @test zero_((Sg - Sl) - xi)                                   # P1
    @test zero_((ml * Sl + mg * Sg) - m * S)                      # P1 / D6
    @test zero_((ub + zl / m * jump) - ul)                        # D1 (liquid)
    @test zero_((ub + zg / m * jump) - ug)                        # D1 (gas)
    @test zero_((ml * ul^2 + mg * ug^2) - (m * ub^2 + mr * jump^2))   # D2
    @test zero_((al * ul + ag * ug) - (a * ub + apg * jump))      # D3
    # D4
    ul1 = ul - dt * Sl; ug1 = ug - dt * Sg
    ub1 = (ml * ul1 + mg * ug1) / m
    @test zero_(ub1 - (ub - dt * S))
    @test zero_((ug1 - ul1) - (jump - dt * xi))
    @test zero_((al * ul1 + ag * ug1) - ((al * ul + ag * ug) - dt * (a * S + apg * xi)))
    # D5
    apg5 = apg |> x -> Symbolics.substitute(x, Dict(al => c, ag => 1 - c, ml => rl * c, mg => rg * (1 - c)))
    @test zero_(apg5 - c * (1 - c) * (rl - rg) / (rl * c + rg * (1 - c)))
end
println("sym_01_phase_algebra: OK")
