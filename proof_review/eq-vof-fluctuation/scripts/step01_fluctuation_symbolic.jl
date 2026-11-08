# Step 01 (symbolic): the algebra behind the fluctuation form (5.19) and its snap terms.
#
# Label:  eq:vof:fluctuation  (with eq:vof:scalar = (5.18), eq:vof:snap = (5.14))
# Source: docs/techref/chapters/05_vof_transport.tex:513-552
# Checks (Symbolics.jl; one cell, three faces of the swept axis, arbitrary orientations / volumes / face states).  The
# denominators C (the fraction in the denominator of (5.19)) are kept as symbols, so that every identity is a polynomial
# identity in the face data (and in 1/V) and the simplifier does not have to cancel sums.
#   S1  Without snapping (C = Cpre), (5.19) reproduces the conservative form (5.18):  Cpre * a1 - A518 = 0.
#   S2  With a snapped fraction C_s = Cpre + delta in the denominator, C_s * a1 - A518 = delta * a0:
#       the snapped volume carries the value a^{[j-1]}  (a snap to a positive fraction).
#   S3  A snap to zero (a1 := a0, C_s = 0) removes the amount A518 = Cpre*a0 + I of (5.18).
#   S4  Gas as the physical complement: with gas volume = open - liquid volume and gas indicator 1 - chi, the pre-clamp gas
#       fraction is 1 - Cpre and the snapped gas volume is -delta (the sign flips for the gas).
#   S5  A cell without the phase (C0 = 0, chi = 0) that only receives it: a1 is the volume-weighted mean of the face
#       states, independent of the stored value a0 (no leak of the stale value of an absent phase).
#   S6  A constant a0 = an = s_i gives I = 0 (constants are preserved).
# Run:    julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/eq-vof-fluctuation/scripts/step01_fluctuation_symbolic.jl

using Symbolics, Test

@variables V C0 a0 an chi delta Cp Cs Gp Gs
@variables o[1:3] up[1:3] uo[1:3] s[1:3]           # orientation, phase volume, open (carrier) volume, face state
@variables sg[1:3] ag0 agn                          # gas: face states, intrinsic value, frozen value

expanded(e) = Symbolics.simplify(Symbolics.expand(e))
iszero_sym(e) = isequal(expanded(e), 0)

Σ(f) = sum(f(i) for i in 1:3)

Cpre_def = C0 - Σ(i -> o[i] * (up[i] - chi * uo[i])) / V
A518 = C0 * a0 - Σ(i -> o[i] * (up[i] * s[i] - chi * an * uo[i])) / V
I = Σ(i -> o[i] * (up[i] * (a0 - s[i]) + chi * uo[i] * (an - a0))) / V

@testset "eq:vof:fluctuation symbolic" begin
    @testset "S1 no snap: (5.19) = (5.18)" begin
        a1 = a0 + I / Cp                                   # denominator = the fraction after the update, Cp = Cpre
        product = expanded(Cp * a1)                         # the symbol Cp cancels
        @test iszero_sym(product - (Cp * a0 + I))
        @test iszero_sym(substitute(product - A518, Dict(Cp => Cpre_def)))
    end
    @testset "S2 snap to a positive fraction leaves the snapped volume with a^{[j-1]}" begin
        a1 = a0 + I / Cs                                   # denominator = the snapped fraction
        product = expanded(Cs * a1)
        deviation = substitute(product - A518 - (Cs - Cp) * a0, Dict(Cp => Cpre_def))
        @test iszero_sym(deviation)                         # C_s a1 - A518 = (C_s - Cpre) a0
    end
    @testset "S3 snap to zero removes the amount of (5.18)" begin
        @test iszero_sym(substitute(A518 - (Cp * a0 + I), Dict(Cp => Cpre_def)))
    end
    @testset "S4 gas as the complement: pre-clamp fraction 1 - Cpre, snapped volume -delta" begin
        # gas phase volume := uo - up, gas indicator 1 - chi, gas fraction before: 1 - C0
        Gpre_def = (1 - C0) - Σ(i -> o[i] * ((uo[i] - up[i]) - (1 - chi) * uo[i])) / V
        @test iszero_sym(Gpre_def - (1 - Cpre_def))
        Ig = Σ(i -> o[i] * ((uo[i] - up[i]) * (ag0 - sg[i]) + (1 - chi) * uo[i] * (agn - ag0))) / V
        Ag = (1 - C0) * ag0 - Σ(i -> o[i] * ((uo[i] - up[i]) * sg[i] - (1 - chi) * agn * uo[i])) / V
        # snapped complement Gs = 1 - Cs = Gp - delta with Gp = 1 - Cp  (Cs = Cp + delta)
        product = expanded(Gs * (ag0 + Ig / Gs))
        deviation = substitute(product - Ag - (Gs - Gp) * ag0, Dict(Gp => Gpre_def))
        @test iszero_sym(deviation)                         # Gs a1 - A518g = (G_s - Gpre) a0 = -delta a0 for G_s - Gpre = -delta
        # the snapped gas fraction 1 - Cs, Cs = Cpre + delta, differs from the pre-clamp gas fraction by -delta
        @test iszero_sym(((1 - (Cpre_def + delta)) - Gpre_def) - (-delta))
    end
    @testset "S5 a phase that is absent and only received does not keep the stale value" begin
        @variables w1 w2 W
        w = (w1, w2, W - w1 - w2)                            # incoming phase volumes w_i = -o_i up_i, with sum W
        Cpr = W / V                                          # C0 = 0, chi = 0
        Ii = Σ(i -> w[i] * (s[i] - a0)) / V
        a1 = a0 + Ii / Cpr
        @test iszero_sym(a1 - Σ(i -> w[i] * s[i]) / W)
    end
    @testset "S6 constants give a zero increment" begin
        Iconst = Σ(i -> o[i] * (up[i] * (a0 - a0) + chi * uo[i] * (a0 - a0))) / V
        @test iszero_sym(Iconst)
        # and with the frozen value and the face states all equal to a0 the increment of (5.19) vanishes identically
        Isub = substitute(I, Dict(s[1] => a0, s[2] => a0, s[3] => a0, an => a0))
        @test iszero_sym(Isub)
    end
end
