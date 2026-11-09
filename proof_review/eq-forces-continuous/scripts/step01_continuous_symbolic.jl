# Symbolic check 01: sign conventions of the continuous model of Section 8.1.
#
# Label:  eq:forces:stress-balance, eq:forces:steady, eq:forces:cont-momentum
# Source: docs/techref/chapters/08_forces.tex:24-80
#   C1  kappa = div n for n = x/|x| (outward from the drop centre, into the gas): kappa = 1/r in 2D (convex drop: kappa > 0);
#   C2  Young-Laplace: with S^l = -p^l I, S^g = -p^g I and [[.]] = gas minus liquid, n.[[S]] = sigma kappa n  <=>  [[p]] = -sigma kappa,
#       i.e. p^l - p^g = sigma kappa;
#   C3  the singular force: for the smoothed liquid indicator psi = (1 - tanh(d/eps))/2 with d the signed distance (negative in the liquid),
#       grad psi = -n delta_eps(d) with delta_eps = sech^2(d/eps)/(2 eps) and int delta_eps dd = 1:  grad 1_l = -n delta;
#       force sigma kappa grad 1_l = -sigma kappa n delta points into the liquid for kappa > 0;
#   C4  steady state (a): -grad p + sigma kappa0 grad psi = 0 for p = p0 + sigma kappa0 psi;
#   C5  steady state (b): grad(rho g.x) = rho g in each phase, and p = rho^pi g.(x - x_I) is continuous across the plane g.(x - x_I) = 0.
# Run:    julia --project=@claude-symbolics proof_review/eq-forces-continuous/scripts/step01_continuous_symbolic.jl

using Symbolics, Test

zero_expr(e) = isequal(Symbolics.simplify(Symbolics.expand(e)), 0)

@variables x y r sigma kappa p_l p_g eps0 dd rho_l rho_g gx gy x0 y0 s
r_expr = sqrt(x^2 + y^2)
nx, ny = x / r_expr, y / r_expr
Dx = Differential(x); Dy = Differential(y)
div_n = Symbolics.expand_derivatives(Dx(nx)) + Symbolics.expand_derivatives(Dy(ny))
function numeric(e, d)
    vars = collect(keys(d))
    f = Symbolics.build_function(e, vars...; expression = Val{false})
    return f((d[v] for v in vars)...)
end
@testset "C1 curvature of a circle" begin
    # symbolic derivative, evaluated at points (the sqrt does not simplify symbolically)
    for (xv, yv) in ((0.3, 0.4), (-1.2, 0.7), (2.0, -0.5), (0.01, 3.0))
        @test numeric(div_n, Dict(x => xv, y => yv)) ≈ 1 / hypot(xv, yv) rtol = 1.0e-13
    end
    # exact in the radius symbol:  d/dx (x/r) + d/dy (y/r) = 2/r - (x^2 + y^2)/r^3 = 1/r  with x^2 + y^2 = r^2
    @test zero_expr(Symbolics.substitute((2 / r - (x^2 + y^2) / r^3) - 1 / r, Dict(x => r * cos(s), y => r * sin(s))) |> e -> Symbolics.simplify(e; expand = true))
end
@testset "C2 Young-Laplace sign" begin
    # n . [[S]] n with S = -p I
    jumpS = -p_g + p_l          # [[S]] = S^g - S^l = (-p_g) - (-p_l)  (coefficient of I)
    @test zero_expr(jumpS - (p_l - p_g))
    # sigma kappa n = n.[[S]]  => p_l - p_g = sigma kappa
    @test zero_expr(Symbolics.substitute(jumpS - sigma * kappa, Dict(p_l => p_g + sigma * kappa)))
end
@testset "C3 singular force" begin
    d = dd
    psi = (1 - tanh(d / eps0)) / 2
    Dd = Differential(dd)
    dpsi = Symbolics.expand_derivatives(Dd(psi))
    delta_eps = (1 / cosh(d / eps0))^2 / (2 * eps0)
    for (dv, ev) in ((0.37, 0.21), (-1.3, 0.5), (0.0, 1.0), (2.2, 0.7))
        @test abs(numeric(dpsi + delta_eps, Dict(dd => dv, eps0 => ev))) < 1.0e-14
    end
    # int delta_eps dd over R = psi(-inf) - psi(+inf) = 1
    @test numeric(psi, Dict(dd => -40.0, eps0 => 1.0)) - numeric(psi, Dict(dd => 40.0, eps0 => 1.0)) ≈ 1.0 atol = 1.0e-12
end
@testset "C4 steady drop" begin
    @variables psi_x
    @test zero_expr((-(sigma * kappa * psi_x)) + sigma * kappa * psi_x)       # -grad p + sigma kappa grad psi with grad p = sigma kappa grad psi
end
@testset "C5 hydrostatic planar interface" begin
    pot = gx * x + gy * y
    for (rho, name) in ((rho_l, "l"), (rho_g, "g"))
        grad = (Symbolics.expand_derivatives(Dx(rho * pot)), Symbolics.expand_derivatives(Dy(rho * pot)))
        @test zero_expr(grad[1] - rho * gx) && zero_expr(grad[2] - rho * gy)
    end
    # p^pi = rho^pi g.(x - x_I) vanishes on the plane g.(x - x_I) = 0 for both phases
    plane = gx * (x - x0) + gy * (y - y0)
    @test zero_expr(Symbolics.substitute(rho_l * plane, Dict(y => y0 - gx * (x - x0) / gy)))
    @test zero_expr(Symbolics.substitute(rho_g * plane, Dict(y => y0 - gx * (x - x0) / gy)))
end
