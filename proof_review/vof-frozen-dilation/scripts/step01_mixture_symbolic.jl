# Step 01 (symbolic): the algebra of the frozen-carrier dilation and of the test velocities used in code_01.
#
# Label:  none (suggested prop:vof:frozen-dilation); Section 5.8.1 after (5.23), Section 5.9 first bullet
# Source: docs/techref/chapters/05_vof_transport.tex:596-620, 797-806
# Checks (Symbolics.jl):
#   S1  Streamfunction face fluxes (segment-end differences) are exactly solenoidal on a coarse cell with two hanging
#       segments and on its two fine neighbours: the hanging vertex value cancels (used to build mixture-solenoidal carriers).
#   S2  Splitting a mixture flux density a^o w into phase velocities u^l = w (1 + d_l), u^g = w (1 + d_g) with
#       a^l d_l + a^g d_g = 0 and a^l + a^g = a^o preserves a^l u^l + a^g u^g = a^o w, for any jump d_l (also u^l u^g < 0).
#   S3  The dilation of cell c in one step is chi_c * dt * m_c with m_c = sum_f o_{c,f} |f| (a^l_f u^l_f + a^g_f u^g_f):
#       for chi_c = 1 it vanishes iff m_c = 0; for chi_c = 0 it vanishes whatever m_c is.  (So "exactly when" holds cellwise
#       on {chi = 1} only.)
#   S4  The dilation of the frozen carrier equals chi_c dt |c| (div upsilon^open)_c with upsilon^open = sum_phi a^phi u^phi
#       (5.23), and with equal phase velocities and a^l + a^g = 1 it reduces to the one-velocity dilation chi_c dt |c| (div u)_c.
# Run:    julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/vof-frozen-dilation/scripts/step01_mixture_symbolic.jl

using Symbolics, Test

expanded(x) = Symbolics.simplify(Symbolics.expand(x))
iszero_sym(x) = isequal(expanded(x), 0)

@testset "frozen dilation (symbolic)" begin
    @testset "S1 streamfunction fluxes telescope around coarse and fine cells (hanging vertex)" begin
        @variables p00 p20 p22 p02 p21 p30 p31 p32 p01 p10     # psi at the vertices (x, y): p_{x y}
        xflux(top, bottom) = top - bottom                      # x-face from bottom to top end point
        yflux(right, left) = -(right - left)                   # y-face from left to right end point
        # coarse cell A = [0,2]^2, right side split at the hanging vertex (2,1)
        left = xflux(p02, p00); bottom = yflux(p20, p00); top = yflux(p22, p02)
        right1 = xflux(p22, p21); right2 = xflux(p21, p20)
        sum_A = (right1 + right2) - left + top - bottom
        @test iszero_sym(sum_A)
        # fine cell B1 = [2,3] x [1,2]
        r1 = xflux(p32, p31); bb = yflux(p31, p21); tt = yflux(p32, p22)
        sum_B1 = r1 - right1 - bb + tt
        @test iszero_sym(sum_B1)
        # fine cell B2 = [2,3] x [0,1]
        r2 = xflux(p31, p30); b2 = yflux(p30, p20); t2 = yflux(p31, p21)
        sum_B2 = r2 - right2 - b2 + t2
        @test iszero_sym(sum_B2)
    end
    @testset "S2 phase split preserves the mixture flux" begin
        @variables w al ag dl
        ao = al + ag
        dg = -al * dl / ag
        ul = w * (1 + dl); ug = w * (1 + dg)
        @test iszero_sym(expanded(Symbolics.simplify_fractions(al * ul + ag * ug - ao * w)))
    end
    @testset "S3 dilation vanishes iff the mixture divergence vanishes (chi = 1)" begin
        @variables chi dt m
        dilation = chi * dt * m
        @test isequal(substitute(dilation, Dict(chi => 0)), 0)               # chi = 0: zero for every m
        @test isequal(substitute(dilation, Dict(chi => 1, m => 0)), 0)        # chi = 1: zero if m = 0
        @test iszero_sym(substitute(dilation, Dict(chi => 1)) / dt - m)        # chi = 1: dilation / dt = m, zero only if m = 0
    end
    @testset "S4 carrier divergence form and the equal-velocity reduction" begin
        @variables o1 o2 o3 f1 f2 f3 l1 l2 l3 g1 g2 g3 u1 u2 u3 ulv1 ulv2 ulv3
        o = (o1, o2, o3); f = (f1, f2, f3); al = (l1, l2, l3); ag = (g1, g2, g3); ulv = (ulv1, ulv2, ulv3)
        ug = (u1, u2, u3)
        m = sum(o[i] * f[i] * (al[i] * ulv[i] + ag[i] * ug[i]) for i in 1:3)
        carrier_div = sum(o[i] * f[i] * (al[i] * ulv[i] + ag[i] * ug[i]) for i in 1:3)   # |c| (div upsilon^open)_c
        @test iszero_sym(m - carrier_div)
        # equal phase velocities, a^l + a^g = 1: one-velocity divergence
        one_velocity = sum(o[i] * f[i] * ug[i] for i in 1:3)
        equal = substitute(m, Dict(ulv1 => u1, ulv2 => u2, ulv3 => u3, g1 => 1 - l1, g2 => 1 - l2, g3 => 1 - l3))
        @test iszero_sym(equal - one_velocity)
    end
end
