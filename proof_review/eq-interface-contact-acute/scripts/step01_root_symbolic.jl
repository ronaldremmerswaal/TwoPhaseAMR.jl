# Step 01: the acute/obtuse contact parabola (4.19) is the thesis root selected by B*tan(theta) > 0.
#
# Label:  eq:interface:contact-acute   (and dev:interface:contact-acute)
# Claim:  04_interface.tex:477-499: the parabola H(x) = a2 x^2 + a1 x + h1 through (0,h1), (w,h2)
#         (heights from the wall) that vanishes with slope T = |tan(theta)| at an unknown point has
#         a2 = ((h2-h1)^2 - w^2 T^2) / (w^2 (h1+h2+sqrt(4 h1 h2 + w^2 T^2))), a1 = (h2-h1)/w - a2 w:
#         "the thesis' root with a1 tan(theta) > 0 [thesis (4.94)] in a rationalised form"; the slope
#         is evaluated at x = -w (the wall cell) instead of at the first column.
# Thesis: /phd_project/thesis/st_curvature.tex, section "Extremely acute and obtuse contact angles in
#         2D": D = 4 (H_i-H_*)(H_{i+1}-H_*) + h^2 tan^2(theta), A^pm = (H_i - 2H_* + H_{i+1} pm sqrt D)/h^2,
#         B^pm = (-2 (H_i-H_*) mp sqrt D)/h, B^+ B^- = -4 (H_i-H_*)(H_{i+1}-H_i)/h^2 - tan^2(theta) < 0.
# Scaling: x -> x/w removes the cell width. With p = h1, q = h2 (heights above the wall, > 0),
#         t = w T (T = |tan(theta)|), a = a2 w^2, b = a1 w, xi = x*/w (contact point in units of w),
#         r = sqrt(D), D = 4 p q + t^2, the parabola is H = a xi^2 + b xi + p with H(1) = q,
#         H(xi*) = 0, H'(xi*) = t.  (a2 = a/w^2, a1 = b/w, tan^2(theta) w^2 = t^2.)
# Run:    julia --project=@claude-symbolics proof_review/eq-interface-contact-acute/scripts/step01_root_symbolic.jl

using Symbolics, Test, Random

@variables p q t r a b xi

D = 4p * q + t^2
bof(aa) = (q - p) - aa                                       # H(1) = a + b + p = q

# reduce a polynomial in r modulo r^2 = D and test whether it vanishes identically
function reduce_r(ex)
    ex = Symbolics.expand(ex)
    coeffs, rest = Symbolics.polynomial_coeffs(ex, [r])
    out = rest
    for (mono, cf) in coeffs
        # degree of the monomial in r: evaluate it at r = 2 (mono = r^k  =>  2^k)
        k = round(Int, log2(Float64(Symbolics.value(Symbolics.substitute(mono, Dict(r => 2))))))
        out += cf * D^(k ÷ 2) * (isodd(k) ? r : 1)
    end
    return Symbolics.Num(Symbolics.simplify(Symbolics.expand(out)))
end
vanishes(ex) = isequal(reduce_r(ex), 0)

aminus = p + q - r                                           # = a2^- w^2 of the thesis
aplus = p + q + r
Q(aa) = aa^2 - 2 * (p + q) * aa + (q - p)^2 - t^2            # quadratic of the thesis (times w^2)

@testset "elimination of the contact point: b^2 = t^2 + 4 a p" begin
    # a xi^2 + b xi + p = 0 and 2 a xi + b = t  =>  (2 a xi + b)^2 = b^2 - 4 a p  (so t^2 = b^2 - 4 a p)
    ex = (2a * xi + b)^2 - (b^2 - 4a * p) - 4a * (a * xi^2 + b * xi + p)
    @test isequal(Symbolics.simplify(Symbolics.expand(ex)), 0)
    # with b = (q-p) - a this is the quadratic Q(a) = 0
    @test isequal(Symbolics.simplify(Symbolics.expand(bof(a)^2 - t^2 - 4a * p - Q(a))), 0)
end

@testset "roots of the quadratic and the thesis' formulas" begin
    @test vanishes(Q(aminus))
    @test vanishes(Q(aplus))
    # b^pm of the thesis: b^pm = -(2p pm r)   (B^pm = -(2 (H_i-H_*) pm sqrt D)/h)
    @test vanishes(bof(aminus) - (r - 2p))
    @test vanishes(bof(aplus) + (2p + r))
    # b^+ b^- = -4 p (q-p) - t^2   (the thesis: B^+ B^- = -4 (H_i-H_*)(H_{i+1}-H_i)/h^2 - tan^2(theta) < 0)
    @test vanishes(bof(aplus) * bof(aminus) + 4p * (q - p) + t^2)
    # b^- (r + 2p) = r^2 - 4p^2 = 4 p (q-p) + t^2: b^- > 0  <=>  4 p (q-p) + t^2 > 0
    @test vanishes(bof(aminus) * (r + 2p) - (4p * (q - p) + t^2))
end

@testset "(4.19) is the minus root A^-" begin
    # a_code = ((q-p)^2 - t^2)/(p+q+r) = p+q-r  <=>  (q-p)^2 - t^2 - (p+q-r)(p+q+r) = 0   (mod r^2 = D)
    @test vanishes(((q - p)^2 - t^2) - (p + q - r) * (p + q + r))
    # the printed a1 = (h2-h1)/w - a2 w is b/w with b = (q-p) - a, i.e. B^- = (r - 2p)/w for this root;
    # the sign condition a1 tan(theta) > 0 (= b^- > 0 for T > 0) holds iff 4 p (q-p) + t^2 > 0, in
    # particular whenever q > p (the source requires h2 > h1 > 0).
end

@testset "which root is physical: the contact point lies before the first column only for b > 0" begin
    # contact point: xi = (t - b)/(2a); from t^2 = b^2 - 4 a p:   xi = -2p/(b + t)   (both roots)
    bm, bp = bof(aminus), bof(aplus)
    @test vanishes((t - bm) * (bm + t) + 4p * aminus)          # t^2 - b^2 = -4 a p
    @test vanishes((t - bp) * (bp + t) + 4p * aplus)
    # minus root: b^- + t > 0 (b^- > 0, t > 0)  =>  xi^- = -2p/(b^- + t) < 0: the contact point lies before the
    # first column.  Plus root: b^+ + t = t - (2p + r) < 0 because r > t (r^2 - t^2 = 4pq > 0)  =>  xi^+ > 0:
    # the "contact point" would lie beyond the first column although H(0) = h1 > 0 -- not a wall contact.
    @test vanishes(r^2 - t^2 - 4p * q)
    @test vanishes((bp + t) - (t - (2p + r)))
    # numerical confirmation with random admissible heights (including q < p where b^- > 0 still holds if t is large)
    rng = MersenneTwister(11)
    nvalid = 0
    for _ in 1:2000
        pv = 0.05 + rand(rng); qv = pv + 3rand(rng); tv = 0.05 + 3rand(rng)
        rv = sqrt(4pv * qv + tv^2)
        am, ap = pv + qv - rv, pv + qv + rv
        bmv, bpv = (qv - pv) - am, (qv - pv) - ap
        abs(am) > 1.0e-6 || continue                            # straight interface: xi = -p/t, 0/0 in this form
        xm, xp = (tv - bmv) / (2am), (tv - bpv) / (2ap)
        @test bmv > 0 && bpv < 0
        @test xm < 0 < xp
        @test isapprox(xm, -2pv / (bmv + tv); rtol = 1.0e-9)
        @test abs(am * xm^2 + bmv * xm + pv) <= 1.0e-9 && isapprox(2am * xm + bmv, tv; rtol = 1.0e-9)
        @test abs(ap * xp^2 + bpv * xp + pv) <= 1.0e-9 && isapprox(2ap * xp + bpv, tv; rtol = 1.0e-9)
        nvalid += 1
    end
    @test nvalid > 1900
    # the condition h2 > h1 of the source is sufficient, not necessary, for b^- > 0:
    #   b^- > 0  <=>  4 h1 (h2 - h1) + w^2 T^2 > 0   (t = w T)
    pv, qv, tv = 1.0, 0.9, 0.5                                   # h2 < h1 but 4*1*(-0.1) + 0.25 < 0: b^- < 0
    @test (qv - pv) - (pv + qv - sqrt(4pv * qv + tv^2)) < 0
    pv, qv, tv = 1.0, 0.95, 0.5                                  # 4*1*(-0.05) + 0.25 > 0: b^- > 0 although h2 < h1
    @test (qv - pv) - (pv + qv - sqrt(4pv * qv + tv^2)) > 0
end

@testset "random 512-bit check of the rationalisation (independent of the symbolic reduction)" begin
    setprecision(BigFloat, 512) do
        rng = MersenneTwister(5)
        worst = BigFloat(0)
        for _ in 1:2000
            pv, qv, wv, Tv = [BigFloat(rand(rng)) + BigFloat(1) / 100 for _ in 1:4]
            rv = sqrt(4pv * qv + wv^2 * Tv^2)
            a_direct = (pv + qv - rv) / wv^2
            a_rationalised = ((qv - pv)^2 - wv^2 * Tv^2) / (wv^2 * (pv + qv + rv))
            worst = max(worst, abs(a_direct - a_rationalised))
        end
        @test worst < BigFloat(10)^(-120)
    end
end

# numerical spot evaluation of derivative identities with fractional powers
@variables A B w s xe
@testset "evaluation point of the slope: x = 0 (thesis) versus x = -w (code)" begin
    # curvature of the parabola with slope sigma(x_e) = B + 2 A x_e (orientation s = +-1)
    kappa(xx) = -2s * A / (1 + (B + 2A * xx)^2)^(3 // 2)
    dk = Symbolics.derivative(kappa(xe), xe)
    closed = 12s * A^2 * (B + 2A * xe) / (1 + (B + 2A * xe)^2)^(5 // 2)
    rng = MersenneTwister(3)
    ev(ex, vals) = Float64(Symbolics.value(Symbolics.substitute(ex, vals)))
    for _ in 1:50
        vals = Dict(A => -(0.1 + rand(rng)), B => 0.1 + rand(rng), w => 0.2 + rand(rng), s => rand(rng) < 0.5 ? 1.0 : -1.0, xe => randn(rng))
        @test isapprox(ev(dk, vals), ev(closed, vals); rtol = 1.0e-12)
        # first-order relative change between the two evaluation points: kappa(-w)/kappa(0) - 1 ~ 6 w A B / (1+B^2)
        vals0 = merge(vals, Dict(xe => 0.0))
        rel = -ev(w * closed, vals0) / ev(kappa(xe), vals0)
        @test isapprox(rel, ev(6w * A * B / (1 + B^2), vals); rtol = 1.0e-12)
    end
    # for a drop of curvature kappa0 > 0 (A = -kappa0 (1+B^2)^(3/2)/2, s = +1) the relative change is
    # 6 w A B/(1+B^2) = -3 w kappa0 B sqrt(1+B^2)
    for _ in 1:20
        Bv, wv, k0 = 0.1 + rand(rng), 0.2 + rand(rng), 0.1 + rand(rng)
        Av = -k0 * (1 + Bv^2)^1.5 / 2
        @test isapprox(6wv * Av * Bv / (1 + Bv^2), -3wv * k0 * Bv * sqrt(1 + Bv^2); rtol = 1.0e-13)
    end
end
