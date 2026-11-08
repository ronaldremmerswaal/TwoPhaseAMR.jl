# Label: eq:interface:area   (also eq:interface:intercept, the plane-constant inverse)
# Steps: S1 regime areas = closed form (4.8), S2 reflected branch / symmetry of the exact areas,
#        S3 continuity and C^1 matching, S4 closed-form inverse, S5 conditioning.
# Source: docs/techref/chapters/04_interface.tex:136-161 (eqs. 4.8, 4.9).
# Run:    julia --project=@claude-symbolics proof_review/eq-interface-area/scripts/step01_area_symbolic.jl
#
# Notation (as in the source): w1,w2 are the projected widths |n_k|(r_hi,k - r_lo,k); with
# w_< = min, w_> = max, the area fraction of the rectangle below the line, as a function of the
# plane constant z measured from the lowest corner, is F(z) = P(w1 U1 + w2 U2 <= z), U_k uniform
# on [0,1] (change of variables U_k = |n_k|(xi_k - lowest corner_k)).  We take w1 = w_< <= w2 = w_>
# without loss of generality (the source only uses min and max).

using Symbolics, Test

@variables w1 w2 z c

shoel(pts) = sum(
    pts[k][1] * pts[mod1(k + 1, length(pts))][2] - pts[mod1(k + 1, length(pts))][1] * pts[k][2]
        for k in eachindex(pts)
) / 2
issym0(e) = isequal(Symbolics.simplify(Symbolics.expand(e)), 0)

# The two closed-form pieces of the source: C(z) for z < w_<  and  C(z) for z >= w_<.
Ccorner(zz) = zz^2 / (2 * w1 * w2)
Clin(zz) = (zz - w1 / 2) / w2

# Exact areas of the liquid polygon {w1 U1 + w2 U2 <= z} in [0,1]^2 (counter-clockwise vertices):
area_I(zz) = shoel([(0, 0), (zz / w1, 0), (0, zz / w2)])                                      # 0 < z <= w1
area_II(zz) = shoel([(0, 0), (1, 0), (1, (zz - w1) / w2), (0, zz / w2)])                      # w1 <= z <= w2
area_III(zz) = shoel([(0, 0), (1, 0), (1, (zz - w1) / w2), ((zz - w2) / w1, 1), (0, 1)])      # w2 <= z <= w1+w2

@testset "S1 regime areas vs (4.8)" begin
    # Which branch of (4.8) is active in each regime.  The case conditions are monotone in z, so it
    # suffices to compare the regime ends with (w1+w2)/2:
    #   I  (z <= w1):      z <= w1 <= (w1+w2)/2, i.e. (w1+w2)/2 - w1 = (w2-w1)/2 >= 0  -> C(z), z < w_<
    #   II (w1 <= z <= w2): C(z) with z >= w_< for z <= (w1+w2)/2, and 1 - C(w1+w2-z) with
    #                       z' = w1+w2-z in [w1,(w1+w2)/2) otherwise (both are the exact area)
    #   III (w2 <= z):     z >= w2 >= (w1+w2)/2 (since w2 - (w1+w2)/2 = (w2-w1)/2 >= 0) -> reflected,
    #                       z' = w1+w2-z <= w1 -> C(z') is the corner formula
    @test issym0((w1 + w2) / 2 - w1 - (w2 - w1) / 2)
    @test issym0(area_I(z) - Ccorner(z))
    @test issym0(area_II(z) - Clin(z))
    @test issym0(area_II(z) - (1 - Clin(w1 + w2 - z)))
    @test issym0(w2 - (w1 + w2) / 2 - (w2 - w1) / 2)
    @test issym0(area_III(z) - (1 - Ccorner(w1 + w2 - z)))
    println("  regime III in closed form: ", Symbolics.simplify(area_III(z)))
end

@testset "S2 the reflected branch is exact: F(z) + F(w1+w2-z) = 1 for the exact areas" begin
    @test issym0(area_I(z) + area_III(w1 + w2 - z) - 1)        # z in I  <-> w1+w2-z in III
    @test issym0(area_III(z) + area_I(w1 + w2 - z) - 1)
    @test issym0(area_II(z) + area_II(w1 + w2 - z) - 1)        # II is mapped to itself
end

@testset "S3 continuity and C^1 matching of the closed form" begin
    # exact regimes join C^1 at z = w1 and z = w2
    @test issym0(area_I(w1) - area_II(w1))
    @test issym0(area_II(w2) - area_III(w2))
    d(f, zz) = Symbolics.substitute(Symbolics.derivative(f(z), z), Dict(z => zz))
    @test issym0(d(area_I, w1) - d(area_II, w1))
    @test issym0(d(area_II, w2) - d(area_III, w2))
    # the two sides of the cut z = (w1+w2)/2 of (4.8) (C(z) versus 1 - C(w1+w2-z)) agree in value 1/2
    # and in slope
    mid = (w1 + w2) / 2
    @test issym0(Clin(mid) - 1 // 2)
    @test issym0(1 - Clin(w1 + w2 - mid) - 1 // 2)
    left = Symbolics.substitute(Symbolics.derivative(Clin(z), z), Dict(z => mid))
    right = Symbolics.substitute(Symbolics.derivative(1 - Clin(w1 + w2 - z), z), Dict(z => mid))
    @test issym0(left - right)
    # F is the CDF of w1 U1 + w2 U2: total mass 1 at z = w1+w2, density 1/w2 on [w1,w2]
    @test issym0(1 - Ccorner(w1 + w2 - (w1 + w2)) - 1)
    @test issym0(Symbolics.derivative(Clin(z), z) - 1 / w2)
end

@testset "S4 closed-form inverse (Scardovelli--Zaleski) and its branch limits" begin
    zI(cc) = sqrt(2 * w1 * w2 * cc)
    zII(cc) = w2 * cc + w1 / 2
    zIII(cc) = w1 + w2 - sqrt(2 * w1 * w2 * (1 - cc))
    @test issym0(Ccorner(zI(c)) - c)
    @test issym0(Clin(zII(c)) - c)
    @test issym0(1 - Ccorner(w1 + w2 - zIII(c)) - c)
    c1 = w1 / (2 * w2)                                          # F(w_<)
    @test issym0(Ccorner(w1) - c1)
    @test issym0(zII(c1) - w1)
    @test issym0(zII(1 - c1) - w2)
    # z_I(c1) = sqrt(2 w1 w2 c1) = sqrt(w1^2) = w1 and z_III(1-c1) = w1+w2 - sqrt(w1^2) = w2 for w1 > 0;
    # the radicand identity is the symbolic content:
    @test issym0((2 * w1 * w2 * c1) - w1^2)
    @test issym0(zIII(1) - (w1 + w2))
    @test issym0(zI(0))
end

@testset "S5 conditioning: dz/dc ~ c^(-1/2) in the corner regime, constant in the linear regime" begin
    zI(cc) = sqrt(2 * w1 * w2 * cc)
    dz = Symbolics.derivative(zI(c), c)
    @test issym0(dz - w1 * w2 / zI(c))
    @test issym0(dz^2 * 2 * c - w1 * w2)                         # (dz/dc)^2 = w1 w2 / (2c)
    @test issym0(Symbolics.derivative(w2 * c + w1 / 2, c) - w2)
    println("  dz/dc (corner regime) = ", Symbolics.simplify(dz), ";  (linear regime) = ", w2)
end
