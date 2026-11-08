# Step 02: which printed thesis statement is consistent with which convention?  (typo or convention?)
#
# Label:  dev:interface:signs
# Claim:  04_interface.tex:526-537: "The thesis prints the opposite signs ... appear to be typographical errors".
# Thesis (st_curvature.tex, Sec. 4.7), the five printed statements, for the principal direction "+" (liquid below the graph
#         z = H, or y = H, eta with positive component along the principal direction) and n_Gamma the inward wall normal:
#   T1  (4.92)  kappa = + [(1+Hy^2)Hxx + (1+Hx^2)Hyy - 2 Hx Hy Hxy]/W^3                      (upper sign of "+-")
#   T2  right angle:  cos(theta) = eta.n_Gamma = - H_x / sqrt(1+H_x^2)      (H^{+y}, n_Gamma = e_x)
#   T3  right angle:  dH^{+y}/dx = 1/tan(theta)
#   T4  acute:        cos(theta) = eta.n_Gamma = + 1/sqrt(1+H_x^2)          (H^{+y}, n_Gamma = e_y)
#   T5  acute:        kappa = 2 A / sqrt(1+B^2)^3                           (H = A x^2 + B x + C, + orientation)
# Conventions:  G = eta points into the gas, kappa = div eta, theta measured in the liquid (thesis Sec. 2.1; the techref);
#               L = eta points into the liquid (eta_L = -eta_G), kappa_L = div eta_L, the printed contact condition
#                   eta_L.n_Gamma = cos(theta) with the same theta.
# Result: G satisfies T2, T4 (the contact-angle premises) and violates T1, T3, T5; L satisfies T1, T3, T5 and violates
#         T2, T4.  No single convention makes the printed set consistent, and G is the one the thesis declares (and the
#         one under which sessile drops have kappa = +1/R).  Hence T1, T3, T5 are the typographical errors.
# Run:    julia --project=@claude-symbolics proof_review/dev-interface-signs/scripts/step02_convention_matrix.jl

using ForwardDiff, Test, Random, LinearAlgebra

rng = MersenneTwister(77)

# height function H(x,y) = c0 + c1 x + c2 y + c3 x^2 + c4 x y + c5 y^2 (general 2-jet)
function height(c)
    H(x, y) = c[1] + c[2] * x + c[3] * y + c[4] * x^2 + c[5] * x * y + c[6] * y^2
    Hx(x, y) = c[2] + 2c[4] * x + c[5] * y
    Hy(x, y) = c[3] + c[5] * x + 2c[6] * y
    return H, Hx, Hy
end

# curvature = divergence of eta_conv = sg * sigma * (-Hx, -Hy, 1)/W, sg = +1 (G) or -1 (L); sigma = +1 here ("+")
function curvature(c, convention, x, y)
    _, Hx, Hy = height(c)
    sg = convention == :G ? 1.0 : -1.0
    eta(p) = begin
        hx, hy = Hx(p[1], p[2]), Hy(p[1], p[2])
        W = sqrt(1 + hx^2 + hy^2)
        sg .* [-hx / W, -hy / W, 1 / W]
    end
    return tr(ForwardDiff.jacobian(eta, [x, y, 0.0]))
end

printedE(c, x, y) = begin
    _, Hx, Hy = height(c)
    hx, hy = Hx(x, y), Hy(x, y)
    hxx, hyy, hxy = 2c[4], 2c[6], c[5]
    ((1 + hy^2) * hxx + (1 + hx^2) * hyy - 2hx * hy * hxy) / sqrt(1 + hx^2 + hy^2)^3
end

@testset "convention matrix for the five printed statements" begin
    results = Dict((t, conv) => true for t in (:T1, :T2, :T3, :T4, :T5), conv in (:G, :L))
    for _ in 1:200
        c = randn(rng, 6)
        x, y = randn(rng, 2)
        for conv in (:G, :L)
            # T1: kappa = + E
            results[(:T1, conv)] &= isapprox(curvature(c, conv, x, y), printedE(c, x, y); rtol = 1.0e-9, atol = 1.0e-12)
            sg = conv == :G ? 1.0 : -1.0
            # 2-D height function y = H(x) with principal direction +y, wall normal e_x (inward) and e_y
            hx = randn(rng)
            eta2 = sg .* [-hx, 1.0] ./ sqrt(1 + hx^2)                       # eta_conv for "+" orientation
            # T2: eta.e_x = -H_x/sqrt(1+H_x^2)
            results[(:T2, conv)] &= isapprox(eta2[1], -hx / sqrt(1 + hx^2); atol = 1.0e-13)
            # T3: with theta defined by cos(theta) = eta.e_x:  H_x = 1/tan(theta)
            theta = acos(clamp(eta2[1], -1, 1))
            results[(:T3, conv)] &= isapprox(hx, 1 / tan(theta); rtol = 1.0e-9, atol = 1.0e-12)
            # T4: eta.e_y = +1/sqrt(1+H_x^2)
            results[(:T4, conv)] &= isapprox(eta2[2], 1 / sqrt(1 + hx^2); atol = 1.0e-13)
            # T5: kappa = 2A/sqrt(1+B^2)^3 for H = A x^2 + B x + C
            A, B = randn(rng), randn(rng)
            kappa2 = ForwardDiff.derivative(t -> sg * (-(B + 2A * t) / sqrt(1 + (B + 2A * t)^2)), 0.0)   # d/dx of eta_x
            results[(:T5, conv)] &= isapprox(kappa2, 2A / sqrt(1 + B^2)^3; rtol = 1.0e-9, atol = 1.0e-12)
        end
    end
    println("  statement   G (eta -> gas, the thesis' declared convention)   L (eta -> liquid)")
    for t in (:T1, :T2, :T3, :T4, :T5)
        println("  ", t, "          ", results[(t, :G)] ? "consistent  " : "INCONSISTENT", "                                      ", results[(t, :L)] ? "consistent" : "INCONSISTENT")
    end
    @test results[(:T2, :G)] && results[(:T4, :G)]
    @test !results[(:T1, :G)] && !results[(:T3, :G)] && !results[(:T5, :G)]
    @test results[(:T1, :L)] && results[(:T3, :L)] && results[(:T5, :L)]
    @test !results[(:T2, :L)] && !results[(:T4, :L)]
end
