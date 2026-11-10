# Symbolic check 01: the continuous pressure-work split (eq:projection:cont-energy) and the sign of thesis eq. (7.24).
#
# Label:  eq:projection:cont-energy   (thesis eq. (7.24), pt2_gradient_mimetic.tex:5-7)
# Source: docs/techref/chapters/10_projection.tex:41-58
# Claim (chapter): for u.n = 0 on the boundary and [[p]] = 0,
#       -∫_Ω u.∇p dV = ∫_Ω p ∇.u dV + ∫_I p [[u.η]] dS,   [[·]] = gas - liquid, η = outward normal of the liquid = into the gas;
#       the thesis prints "- ∫_I p [[u_η]] dS" (same definitions: basic_navier_stokes.tex:62-66: "[[a]] = a^g - a^l", η "points into the gas phase").
# Method: exact polynomial integration (rational arithmetic via Symbolics) on Ω = [0,1]² with the interface y = y0 + s x, liquid below,
#       p one polynomial (continuous), u^l, u^g different polynomial fields with u.n = 0 on ∂Ω; random rational coefficients (seeded).
#       The residual of the chapter's identity must be exactly 0; that of the thesis' sign equals 2 ∫_I p [[u.η]] dS ≠ 0.
# Run:    julia --project=@claude-symbolics proof_review/eq-projection-continuous/scripts/sym_01_energy_split.jl

using Symbolics, Random, Test
@variables x y
rng = MersenneTwister(7242)

# Exact integral of a polynomial of x only / of (x, y) over y from lo(x) to hi(x), then x over [0,1].
function poly_integral(expr, v, lo, hi)
    e = Symbolics.expand(expr)
    deg = 0
    # collect coefficients in v by repeated differentiation at v = 0 (polynomial of low degree)
    coeffs = Any[]
    cur = e
    k = 0
    while true
        c = Symbolics.substitute(cur, Dict(v => 0)) / factorial(k)
        push!(coeffs, c)
        cur = Symbolics.expand(Symbolics.expand_derivatives(Differential(v)(cur)))
        k += 1
        (isequal(cur, 0) || k > 12) && break
    end
    anti(t) = sum(coeffs[i] * t^i / i for i in 1:length(coeffs))   # ∫_0^t Σ c_k v^k dv = Σ c_k t^{k+1}/(k+1), index i = k+1
    return Symbolics.expand(anti(hi) - anti(lo))
end
dx(e) = Symbolics.expand(Symbolics.expand_derivatives(Differential(x)(e)))
dy(e) = Symbolics.expand(Symbolics.expand_derivatives(Differential(y)(e)))

randpoly(deg) = sum((rand(rng, -5:5) // 3) * x^i * y^j for i in 0:deg for j in 0:deg if i + j <= deg)
s = 1 // 2; y0 = 1 // 4
yI = y0 + s * x
p = randpoly(3)
# velocity: u_x = x(1-x) q_x, u_y = y(1-y) q_y   =>  u.n = 0 on all four walls
ul = (x * (1 - x) * randpoly(2), y * (1 - y) * randpoly(2))
ug = (x * (1 - x) * randpoly(2), y * (1 - y) * randpoly(2))

region_integral(f, lo, hi) = poly_integral(poly_integral(f, y, lo, hi), x, 0, 1)
lhs = -(region_integral(ul[1] * dx(p) + ul[2] * dy(p), 0, yI) + region_integral(ug[1] * dx(p) + ug[2] * dy(p), yI, 1))
t1 = region_integral(p * (dx(ul[1]) + dy(ul[2])), 0, yI) + region_integral(p * (dx(ug[1]) + dy(ug[2])), yI, 1)
# interface term: η = (-s, 1)/sqrt(1+s²), dS = sqrt(1+s²) dx   =>  ∫_I p [[u.η]] dS = ∫_0^1 p (-s [[u_x]] + [[u_y]]) dx  on y = yI
on_I(e) = Symbolics.substitute(e, Dict(y => yI))
jump_eta = on_I((ug[1] - ul[1]) * (-s) + (ug[2] - ul[2]))
t2 = poly_integral(Symbolics.expand(on_I(p) * jump_eta), x, 0, 1)
println("-∫u.∇p = ", lhs, "\n ∫p div u = ", t1, "\n ∫_I p[[u.η]] dS = ", t2)
println("residual of the chapter's identity  lhs - t1 - t2 = ", Symbolics.expand(lhs - t1 - t2))
println("residual of the thesis' printed sign lhs - t1 + t2 = ", Symbolics.expand(lhs - t1 + t2), "   (= 2 t2)")
@testset "cont-energy sign" begin
    @test isequal(Symbolics.expand(lhs - t1 - t2), 0)
    @test !isequal(Symbolics.expand(lhs - t1 + t2), 0)
    @test isequal(Symbolics.expand((lhs - t1 + t2) - 2 * t2), 0)
    # consequence stated in the chapter: no work on a solenoidal field with continuous normal velocity: pick ug = ul + w with w.η = 0
end

println("sym_01_energy_split: OK")
