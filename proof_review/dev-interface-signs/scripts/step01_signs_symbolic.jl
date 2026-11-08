# Step 01: sign conventions of the height-function formulas (kappa = div(eta), contact angle).
#
# Label:  dev:interface:signs  (eq:interface:graph, eq:interface:contact-right, eq:interface:ghf-result)
# Claim:  04_interface.tex:20-45, 460-475, 524-537: with eta pointing into the gas, kappa = div(eta)
#         (convex liquid: kappa > 0), the wall normal n_w pointing into the fluid and theta measured in
#         the liquid, (4.2) carries "minus/plus" for the height function H^{+-k}, the right-angle branch has
#         dH/d(xi) = -cos(theta)/(s sin(theta)) at the wall (xi along the inward normal), the ghost value is
#         H_{-1} = H_0 + h cos(theta)/(s sin(theta)) and the acute branch has kappa = -2 s a2/(1+H'^2)^(3/2).
#         The thesis prints the opposite signs (eq. 4.92 and Sec. 4.7.2).
# Thesis: st_curvature.tex: "kappa = div eta = div( (+-1)/W [-Hx; -Hy; 1] ) = +- [(1+Hy^2)Hxx + (1+Hx^2)Hyy
#         - 2 Hx Hy Hxy]/W^3" (4.92, W = sqrt(1+Hx^2+Hy^2)); right angle: "cos theta = eta.n =
#         -H_x/sqrt(1+H_x^2), which yields H_x = 1/tan(theta)"; acute: "kappa = 2A/sqrt(1+B^2)^3".
# Checks (symbolic unless stated):
#   S1  div of eta = sigma (-Hx,-Hy,1)/W equals -sigma [..]/W^3  (chain rule, W^2 = 1+Hx^2+Hy^2), i.e. the
#       printed "+-" of (4.92) has the wrong sign; numerical spot check of the same identity with the
#       divergence computed by Symbolics.derivative of the explicit expression for a general quadratic H.
#   S2  2D reduction: kappa = -sigma H''/(1+H'^2)^(3/2) for eta = sigma (e_k - H' e_t)/sqrt(1+H'^2)  ((4.2)).
#   S3  convex disc: div(x/|x|) = 1/r > 0 ((4.1)); concave height function (H'' < 0, sigma = +) gives kappa > 0.
#   S4  right angle: eta.n_w = cos(theta) with eta = s (e_t - (dH/dx_k) e_k)/sqrt(..), n_w = psi e_k forces
#       dH/d(xi) = -cos(theta)/(s sin(theta)); the thesis' printed +1/tan(theta) is the solution for s = -1.
#   S5  ghost value: (H_0 - H_{-1})/h = -cos(theta)/(s sin(theta)); the normal built from this slope satisfies
#       eta.n_w = cos(theta) exactly.
#   S6  acute branch: cos(theta) = sigma/sqrt(1+H'^2) forces sigma = sgn(cos theta), H'^2 = tan^2(theta), and
#       kappa = -sigma 2 a2/(1+slope^2)^(3/2): the thesis' printed +2A/... is right only for sigma = -1.
# Logic of the verdict: each of the three printed signs conflicts with a premise printed in the SAME
#   paragraph of the thesis, so they are typographical errors whatever the convention:
#   P1  displayed eta = (+-)(-Hx,-Hy,1)/W and kappa = div eta   =>  kappa = (-+) E          (printed: +- E)   [S1, S1b]
#   P2  cos(theta) = eta.n_Gamma = -H_x/sqrt(1+H_x^2) (H^{+y}, n_Gamma = e_x) => H_x = -1/tan(theta) (printed: +1/tan) [S4]
#   P3  eta = (-H_x, 1)/sqrt(1+H_x^2) (H^{+y}, the acute example), kappa = div eta => kappa = -2A/(1+B^2)^(3/2)
#       (printed: +2A/...; the printed sign is right for the negative orientation)                       [S6]
# Run:    julia --project=@claude-symbolics proof_review/dev-interface-signs/scripts/step01_signs_symbolic.jl

using Symbolics, Test, Random

Random.seed!(20261108)   # deterministic

@variables hx hy hxx hyy hxy W sigma
isz(e) = isequal(Symbolics.simplify(Symbolics.expand(e)), 0)
# numerical value of a symbolic expression (constants folded, including sqrt)
ev(ex, vals) = Float64(Symbolics.value(Symbolics.substitute(ex, vals; fold = Val(true))))

@testset "S1 3D (eq. 4.92): div eta = -sigma E, E = [(1+Hy^2)Hxx + (1+Hx^2)Hyy - 2 Hx Hy Hxy]/W^3" begin
    # chain rule with W as an auxiliary symbol: dW/dx = (Hx Hxx + Hy Hxy)/W, dW/dy = (Hx Hxy + Hy Hyy)/W
    dWx = (hx * hxx + hy * hxy) / W
    dWy = (hx * hxy + hy * hyy) / W
    # d/dx (-sigma Hx / W) + d/dy (-sigma Hy / W)   (the z-component sigma/W does not depend on z)
    div = -sigma * (hxx / W - hx * dWx / W^2) - sigma * (hyy / W - hy * dWy / W^2)
    E = ((1 + hy^2) * hxx + (1 + hx^2) * hyy - 2hx * hy * hxy) / W^3
    # multiply by W^3 and reduce W^2 = 1 + hx^2 + hy^2
    ex = Symbolics.expand((div + sigma * E) * W^3)
    ex = Symbolics.substitute(ex, Dict(W^2 => 1 + hx^2 + hy^2))
    @test isz(ex)
    # ... and the printed formula (sign +-) is NOT div eta: div eta - sigma E = -2 sigma E != 0
    @test !isz(Symbolics.expand((div - sigma * E) * W^3) |> e -> Symbolics.substitute(e, Dict(W^2 => 1 + hx^2 + hy^2)))
end

@testset "S1b the same identity from the explicit divergence of a general quadratic height function (numerical spot)" begin
    @variables x y z c0 c1 c2 c3 c4 c5
    H = c0 + c1 * x + c2 * y + c3 * x^2 + c4 * x * y + c5 * y^2
    Hx, Hy = Symbolics.derivative(H, x), Symbolics.derivative(H, y)
    Hxx, Hyy, Hxy = Symbolics.derivative(Hx, x), Symbolics.derivative(Hy, y), Symbolics.derivative(Hx, y)
    Wexpr = sqrt(1 + Hx^2 + Hy^2)
    eta = sigma .* (-Hx / Wexpr, -Hy / Wexpr, 1 / Wexpr)
    div = Symbolics.derivative(eta[1], x) + Symbolics.derivative(eta[2], y) + Symbolics.derivative(eta[3], z)
    E = ((1 + Hy^2) * Hxx + (1 + Hx^2) * Hyy - 2Hx * Hy * Hxy) / Wexpr^3
    rng = MersenneTwister(1)
    for _ in 1:40
        vals = Dict(
            x => randn(rng), y => randn(rng), z => randn(rng), c0 => randn(rng), c1 => randn(rng), c2 => randn(rng),
            c3 => randn(rng), c4 => randn(rng), c5 => randn(rng), sigma => rand(rng) < 0.5 ? 1.0 : -1.0
        )
        d = ev(div, vals)
        e = ev(sigma * E, vals)
        @test isapprox(d, -e; rtol = 1.0e-12, atol = 1.0e-13)         # div eta = -sigma E
        @test !isapprox(d, e; rtol = 1.0e-3, atol = 1.0e-3) || abs(e) < 1.0e-3   # not +sigma E
    end
end

@testset "S2/S3 two dimensions: (4.2), and a convex disc has positive curvature" begin
    @variables u up           # u = H', up = H''
    # eta_t = -sigma u / sqrt(1+u^2) depends on x_t only through u: d/dx_t = (d/du) * up
    eta_t = -sigma * u / sqrt(1 + u^2)
    kappa = Symbolics.derivative(eta_t, u) * up
    for sv in (1.0, -1.0), uv in (-2.0, -0.3, 0.0, 0.8), upv in (-1.7, 0.4)
        # kappa = -sigma H''/(1+H'^2)^(3/2)   ((4.2): "minus" for H^{+k}, "plus" for H^{-k})
        @test isapprox(ev(kappa, Dict(u => uv, up => upv, sigma => sv)), -sv * upv / (1 + uv^2)^1.5; rtol = 1.0e-13, atol = 1.0e-14)
    end
    # convex disc: eta = (x,y)/sqrt(x^2+y^2), kappa = div eta = 1/r
    @variables xx yy
    rr = sqrt(xx^2 + yy^2)
    divdisc = Symbolics.derivative(xx / rr, xx) + Symbolics.derivative(yy / rr, yy)
    for _ in 1:5
        vals = Dict(xx => 0.3 + rand(), yy => 0.3 + rand())
        @test isapprox(ev(divdisc, vals), 1 / sqrt(vals[xx]^2 + vals[yy]^2); rtol = 1.0e-13)
    end
    # the top of a convex liquid drop: sigma = +1 (eta points up), H'' < 0 -> kappa = -H''/(1+H'^2)^(3/2) > 0
    @test ev(kappa, Dict(u => 0.0, up => -1.3, sigma => 1.0)) > 0
end

@testset "S4 right-angle slope at the wall: eta.n_w = cos(theta)  =>  dH/d(xi) = -cos(theta)/(s sin(theta))" begin
    @variables c sn          # c = cos(theta), sn = sin(theta) > 0; c^2 + sn^2 = 1
    red(ex) = Symbolics.simplify(Symbolics.expand(Symbolics.substitute(Symbolics.expand(ex), Dict(c^2 => 1 - sn^2))))
    for sv in (1, -1)       # orientation s of the principal direction (s^2 = 1, 1/s = s)
        # work with v = u sin(theta) (no division): the claimed solution u = -cos/(s sin) has v = -s c
        v = -sv * c
        # rho^2 sn^2 = (1 + u^2) sn^2 = sn^2 + v^2 = 1, i.e. rho = sqrt(1+u^2) = 1/sn (sn > 0)
        @test isz(red(sn^2 + v^2 - 1))
        # eta.n_w = -s u / rho = -s u sn = -s v = c
        @test isz(red(-sv * v - c))
        # the thesis' printed solution u = 1/tan(theta) = c/sn has v = c: eta.n_w = -s c, which equals cos(theta)
        # only for s = -1
        vtheirs = c
        residual = red(-sv * vtheirs - c)
        @test sv == -1 ? isz(residual) : !isz(residual)
    end
    for sv in (1.0, -1.0), theta in (0.4, 1.2, 2.0, 2.9)
        cth, sth = cos(theta), sin(theta)
        slope = -cth / (sv * sth)
        @test isapprox(-sv * slope / sqrt(1 + slope^2), cth; atol = 1.0e-14)          # the doc
        thesis = cth / sth
        @test isapprox(-sv * thesis / sqrt(1 + thesis^2), -sv * cth; atol = 1.0e-14)  # the thesis' slope gives -s cos(theta)
    end
end

@testset "S5 ghost value (4.18) imposes eta.n_w = cos(theta) at the wall face" begin
    @variables H0 hh c sn
    s = sigma
    ghost = H0 + hh * c / (s * sn)                              # (4.18): H_{-1} = H_0 + h cos(theta)/(s sin(theta))
    slope_wall = (H0 - ghost) / hh                              # face-centred difference of (ghost, first cell)
    @test isz(slope_wall + c / (s * sn))                        # = -cos(theta)/(s sin(theta))
    # numerical: the normal built from this slope has eta.n_w = cos(theta)
    for sv in (1.0, -1.0), theta in (0.3, 1.1, 1.9, 2.8), psi in (1.0, -1.0), H0v in (0.2, 3.0), hv in (0.1, 1.7)
        ghostv = H0v + hv * cos(theta) / (sv * sin(theta))
        dHdxi = (H0v - ghostv) / hv                               # derivative along the inward direction
        dHdxk = psi * dHdxi                                       # x_k = x_wall + psi xi
        eta_t = sv / sqrt(1 + dHdxk^2); eta_k = -sv * dHdxk / sqrt(1 + dHdxk^2)
        @test isapprox(psi * eta_k, cos(theta); atol = 1.0e-13)   # eta . n_w with n_w = psi e_k
    end
end

@testset "S6 acute/obtuse branch: sigma = sgn(cos theta), H'^2 = tan^2 theta, kappa = -sigma H''/(...)" begin
    for theta in (0.2, 0.6, 2.6, 3.0)
        s = sign(cos(theta))
        slope2 = tan(theta)^2
        @test isapprox(s / sqrt(1 + slope2), cos(theta); atol = 1.0e-14)               # eta.n_w with eta_xi = sigma/sqrt(1+H'^2)
        @test isapprox(-s * 2 * 0.3 / (1 + slope2)^1.5, -s * 2 * 0.3 * abs(cos(theta))^3; atol = 1.0e-14)
    end
    # the printed thesis curvature +2A/sqrt(1+B^2)^3 equals -sigma 2A/(..) only for sigma = -1 (gas wedge, obtuse theta)
    A, B = -0.31, 0.8
    @test isapprox(-(+1) * 2A / (1 + B^2)^1.5, -2A / (1 + B^2)^1.5)                  # sigma = +1: doc says -2A/(..), thesis +2A/(..)
    @test !isapprox(2A / (1 + B^2)^1.5, -2A / (1 + B^2)^1.5)
    @test isapprox(-(-1) * 2A / (1 + B^2)^1.5, 2A / (1 + B^2)^1.5)                   # sigma = -1: both agree
end
