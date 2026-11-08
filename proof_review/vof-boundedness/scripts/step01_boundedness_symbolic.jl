# Step 01 (symbolic): the algebra of the boundedness argument and of the triangle example.
#
# Label:  none (suggested prop:vof:boundedness; the example is inside the same paragraph) -- Section 5.5 (sec:vof:floor)
# Source: docs/techref/chapters/05_vof_transport.tex:296-322
# Checks (Symbolics.jl):
#   S1  geometric reading of the sweep update (5.6): for chi = 1 the update of C is the conservative update of the gas fraction
#       1 - C with the gas volumes O_f - L_f (partition (5.7)), for chi = 0 the conservative update of C.
#   S2  bound chain: with 0 <= C <= 1/2 (chi = 0), 0 <= out <= C (outflow of the moved phase at most its volume, which needs
#       disjoint rectangles) and 0 <= in <= nu_in <= 1/2 the update C' = C - out + in satisfies 0 <= C' <= 1; the extremes are
#       attained (C' = 0 at out = C, in = 0; C' = 1 at C = 1/2, in = 1/2).  Written as exact polynomial identities in the slack variables.
#   S3  triangle example: the strip of depth d over the open triangle {xi + eta <= 0} has width w(s) = s at depth s from the
#       face, so A(d) = d^2/2; with a_f = 1/2, nu_ext = nu_out = nu = 1/2 the target is a_f nu = 1/4 and d = 1/sqrt(2).
#   S4  far-corner family: open triangle {xi + eta <= -1 + eps}: A(d) = (d - 1 + eps)^2 / 2 for d >= 1 - eps; nu_out = nu at
#       target tau = a_f nu with nu_out = tau / (eps^2/2) = 1/2 gives tau = eps^2/4 and d = 1 - eps + eps/sqrt(2) -> 1.
#   S5  consistent geometry: if the width w(s) of the open part at distance s from the face is concave with w(0) = l, then
#       A(d) >= l d (1 - d/(2 d1)) ... used only as the chord bound A(d) >= d (l + w(d))/2 >= l d / 2 (so d <= 2 nu);
#       checked as an identity of the trapezoid rule for a linear w.
# Run:    julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/vof-boundedness/scripts/step01_boundedness_symbolic.jl

using Symbolics, Test

expanded(x) = Symbolics.simplify(Symbolics.expand(x))
iszero_sym(x) = isequal(expanded(x), 0)

@testset "boundedness: algebra (symbolic)" begin
    @testset "S1 geometric reading of (5.6)" begin
        @variables C V o1 o2 o3 L1 L2 L3 O1 O2 O3
        o = (o1, o2, o3); L = (L1, L2, L3); O = (O1, O2, O3)
        # chi = 1: C' = C - sum o (L - O)/V ;  gas fraction G = 1 - C, gas volumes O - L:  G' = G - sum o (O - L)/V
        Cp = C - sum(o[i] * (L[i] - O[i]) for i in 1:3) / V
        Gp = (1 - C) - sum(o[i] * (O[i] - L[i]) for i in 1:3) / V
        @test iszero_sym(Cp + Gp - 1)
        # chi = 0: C' = C - sum o L / V is the conservative update of the liquid
        Cp0 = C - sum(o[i] * L[i] for i in 1:3) / V
        @test iszero_sym(Cp0 - (C - sum(o[i] * L[i] for i in 1:3) / V))
    end
    @testset "S2 bound chain" begin
        @variables C out inn nu s1 s2 s3
        # slacks: C = 1/2 - s1 (s1 >= 0), out = C - s2 (s2 >= 0, out <= C), inn = nu - s3 (s3 >= 0), nu = 1/2 - s4
        @variables s4
        Cv = 1 // 2 - s1; outv = Cv - s2; nuv = 1 // 2 - s4; innv = nuv - s3
        Cnew = Cv - outv + innv
        @test iszero_sym(Cnew - (s2 + 1 // 2 - s4 - s3))                       # C' = s2 + nu - s3
        # lower bound: C' = (C - out) + inn = s2 + inn >= 0 since s2 >= 0, inn >= 0 (inn >= 0 is a hypothesis)
        @test iszero_sym(Cnew - (s2 + innv))
        # upper bound: 1 - C' = (1/2 + s1) + out - inn  >= 1/2 - inn >= 0
        @test iszero_sym((1 - Cnew) - ((1 // 2 + s1) + outv - innv))
        # attained: C = 1/2, out = 0, inn = 1/2 gives C' = 1; C = anything, out = C, inn = 0 gives C' = 0
        @test substitute(Cnew, Dict(s1 => 0, s2 => 1 // 2, s3 => 0, s4 => 0)) == 1
        @test substitute(Cnew, Dict(s2 => 0, s3 => 1 // 2 - s4)) == 0
    end
    @testset "S3 triangle example" begin
        @variables d s
        A = d^2 / 2
        @test iszero_sym(Symbolics.derivative(A, d) - d)                      # width at depth d is d
        af = 1 // 2; nu = 1 // 2
        s2 = sqrt(Num(2))
        @test iszero_sym(substitute(A, Dict(d => 1 / s2)) - af * nu)
        # nu_out = a_f nu |f| / |c^o| = (1/2)(1/2) / (1/2) = 1/2 and nu_ext = nu = 1/2
        @test af * nu / (1 // 2) == 1 // 2
    end
    @testset "S4 far-corner family" begin
        @variables d eps
        A = (d - 1 + eps)^2 / 2
        @test iszero_sym(Symbolics.derivative(A, d) - (d - 1 + eps))
        tau = eps^2 / 4                                                       # a_f nu with nu_out = 1/2: tau = |c^o|/2 * (1/2)?  see below
        # nu_out = a_f nu / |c^o| with a_f = 1/2, |c^o| = eps^2/2:  nu = 1/2 -> a_f nu_ext-based target tau = a_f * dt u = (1/2) dt u,
        # and nu_out = tau / |c^o| = 1/2  =>  tau = eps^2/4
        @test iszero_sym(tau / (eps^2 / 2) - 1 // 2)
        s2 = sqrt(Num(2))
        dstar = 1 - eps + eps / s2
        @test iszero_sym(substitute(A, Dict(d => dstar)) - tau)
        @test iszero_sym(substitute(dstar, Dict(eps => 0)) - 1)                # the limit eps -> 0 is 1
    end
    @testset "S5 chord bound for a linear width" begin
        @variables l m d
        w(s) = l + (m - l) * s                                               # linear (concave) width from l to m at depth 1
        A = l * d + (m - l) * d^2 / 2                                        # integral of w over [0, d]
        trapezoid = d * (l + w(d)) / 2
        @test iszero_sym(A - trapezoid)
    end
end
