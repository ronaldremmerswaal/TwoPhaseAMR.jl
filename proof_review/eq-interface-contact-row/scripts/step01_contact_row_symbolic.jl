# Step 01: the linearised contact row (4.20) of the constrained GHF fit.
#
# Label:  eq:interface:contact-row
# Claim:  04_interface.tex:500-523: with the solid normal into the fluid, n_w, and tau* the tangential
#         coordinate of the triple point, "the linearised condition (4.3) on the fitted normal is
#         -2 tau* (t.n_w) a2 - (t.n_w) a1 = cos(theta) - eta.n_w, added to (4.15) with weight 1e3; it is the
#         unnormalised fitted normal eta - (a1 + 2 a2 tau*) t inserted in (4.3)."
# Thesis: st_curvature.tex, "Triple point curvature for arbitrary geometries" (eq. 4.97):
#         [1; -B - 2 A S'(x_* - x_c^I)]' R' n_Gamma = cos(theta) with R = [eta S].
# Notation: eta = (e1, e2) the seed unit normal, t = (-e2, e1) the tangent used in (4.15), n_w = (w1, w2) the
#         wall normal, f' = 2 a2 tau* + a1 the slope of the fitted parabola at the triple point (in units of h:
#         tau, nu are divided by h, so f' is the physical slope).
# Checks (symbolic):
#   S1  the fitted curve nu = a2 tau^2 + a1 tau + a0 has the unit normal N^ = (eta - f' t)/sqrt(1+f'^2) (perpendicular
#       to its tangent t + f' eta); the row (4.20) is N.n_w = cos(theta) for the UNNORMALISED N = eta - f' t;
#   S2  the thesis' (4.97), [1, -B - 2 A tau*] R' n_Gamma = cos(theta) with R' n_Gamma = (eta.n_w, t.n_w), is (4.20);
#   S3  the exact condition N^.n_w = cos(theta) differs from the row by the factor sqrt(1+f'^2): the row is the first-order
#       (in f') linearisation, the enforced normal satisfies N^.n_w = cos(theta)/sqrt(1+f'^2) = cos(theta) (1 - f'^2/2 + ..),
#       i.e. the enforced contact angle is off by ~ (1/2) f'^2 cot(theta);
#   S4  exactness for a straight interface that meets the wall at theta: seed = exact normal => a1 = a2 = 0 solves the row;
#   S5  the row is a row of the least-squares matrix scaled by alpha: the weight in the functional (4.15)
#       sum_j w_j r_j^2 is alpha^2 (the code scales the row by alpha = 1e3, i.e. weight 1e6, not 1e3).
# Run:    julia --project=@claude-symbolics proof_review/eq-interface-contact-row/scripts/step01_contact_row_symbolic.jl

using Symbolics, Test, Random

@variables e1 e2 w1 w2 a1 a2 ts th f alpha
isz(e) = isequal(Symbolics.simplify(Symbolics.expand(e)), 0)

eta = (e1, e2)
tvec = (-e2, e1)
nw = (w1, w2)
dot2(a, b) = a[1] * b[1] + a[2] * b[2]
reduce_unit(ex) = Symbolics.simplify(Symbolics.expand(Symbolics.substitute(Symbolics.expand(ex), Dict(e2^2 => 1 - e1^2))))   # e1^2 + e2^2 = 1
iszu(e) = isequal(reduce_unit(e), 0)

@testset "S1 fitted normal and the row (4.20)" begin
    fp = 2a2 * ts + a1                                         # slope at the triple point
    N = (eta[1] - fp * tvec[1], eta[2] - fp * tvec[2])         # unnormalised fitted normal eta - f' t
    tang = (tvec[1] + fp * eta[1], tvec[2] + fp * eta[2])      # tangent of the fitted curve in the (t, eta) frame
    @test iszu(dot2(N, tang))                                  # N is perpendicular to the curve
    @test iszu(dot2(N, N) - (1 + fp^2))                        # |N|^2 = 1 + f'^2
    # N.n_w = cos(theta)  <=>  (4.20): -2 tau* (t.n_w) a2 - (t.n_w) a1 = cos(theta) - eta.n_w
    row_lhs = -2ts * dot2(tvec, nw) * a2 - dot2(tvec, nw) * a1
    row_rhs = th - dot2(eta, nw)                              # th stands for cos(theta)
    @test isz((dot2(N, nw) - th) - (row_lhs - row_rhs))
end

@testset "S2 the thesis' (4.97)" begin
    # R' n_Gamma = (eta.n_w, t.n_w);  [1, -B - 2 A tau*] . (eta.n_w, t.n_w) = cos(theta)
    lhs97 = 1 * dot2(eta, nw) + (-a1 - 2a2 * ts) * dot2(tvec, nw)
    fp = 2a2 * ts + a1
    N = (eta[1] - fp * tvec[1], eta[2] - fp * tvec[2])
    @test isz(lhs97 - dot2(N, nw))
end

@testset "S3 linearisation error: exact condition versus the row" begin
    # exact: cos(theta_enforced) = N.n_w / sqrt(1+f'^2).  The row makes N.n_w = cos(theta), so
    #   cos(theta_enforced) = cos(theta) / sqrt(1+f'^2).
    g(ff) = th / sqrt(1 + ff^2)
    g0 = Symbolics.substitute(g(f), Dict(f => 0))
    g1 = Symbolics.substitute(Symbolics.derivative(g(f), f), Dict(f => 0))
    g2 = Symbolics.substitute(Symbolics.derivative(Symbolics.derivative(g(f), f), f), Dict(f => 0))
    @test isz(g0 - th)
    @test isz(g1)                                              # no first-order error
    @test isz(g2 + th)                                         # g'' (0) = -cos(theta): cos(theta)(1 - f'^2/2 + ...)
    # the enforced angle: d(cos theta) = -sin(theta) d(theta)  =>  Delta theta = (1/2) f'^2 cot(theta) + O(f'^4)
    for theta in (0.3, 1.0, 2.0, 2.8), fv in (1.0e-2, 1.0e-1)
        enforced = acos(cos(theta) / sqrt(1 + fv^2))
        @test isapprox(enforced - theta, 0.5 * fv^2 * cos(theta) / sin(theta); rtol = 2fv, atol = 1.0e-12)   # leading term, relative error O(f')
    end
    # angle error at the largest slope offsets seen in practice (f' = 0.05 ... 0.3), theta = 60 degrees
    for fv in (0.05, 0.1, 0.3)
        println("  f' = $fv: enforced angle error at theta = 60 deg = ", round(rad2deg(acos(cos(pi / 3) / sqrt(1 + fv^2)) - pi / 3), digits = 4), " degrees")
    end
end

@testset "S4 a straight interface that meets the wall at the prescribed angle satisfies the row with a1 = a2 = 0" begin
    # seed = exact unit normal with eta.n_w = cos(theta): rhs = cos(theta) - eta.n_w = 0 and the row is homogeneous
    row_rhs = th - dot2(eta, nw)
    @test isz(Symbolics.substitute(row_rhs, Dict(th => dot2(eta, nw))))
    @test isz(Symbolics.substitute(-2ts * dot2(tvec, nw) * a2 - dot2(tvec, nw) * a1, Dict(a1 => 0, a2 => 0)))
end

@testset "S5 scaling of the row versus weight in the functional" begin
    # a row alpha * (g . a - r) in the stacked system minimises sum_j (sqrt(w_j) r_j)^2 + alpha^2 r_c^2:
    # the weight in sum_j w_j r_j^2 is alpha^2.
    @variables rc wc
    stacked = (alpha * rc)^2
    @test isz(stacked - alpha^2 * rc^2)
    println("  scaling alpha = 1e3 corresponds to weight alpha^2 = ", 1.0e3^2, " in the functional (4.15)")
end
