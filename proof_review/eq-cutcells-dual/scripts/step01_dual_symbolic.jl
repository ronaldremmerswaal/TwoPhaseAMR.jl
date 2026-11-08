# Step 01: the open staggered volume (7.5) against its definition (3.7), symbolically.
#
# Label:  eq:cutcells:dual   (definition: eq:setting:open-dual (3.7); no-solid limit: eq:setting:dual-volume (3.3))
# Claim:  07_cutcells.tex:134-160 (eq. 7.5): V^o_f = (1/2) sum_{c in C(f)} |c| (r_hi - r_lo) (1 - Phi(n_s, a_s; [-1/2,1/2] x [r_lo, r_hi]))
#         "with Phi = 0 for uncut and the contribution zero for covered cells"; "without a solid V^o_f = V_f"; "in general
#         V^o_f != a_f V_f".   Definition (03_discrete_setting.tex:105-109): V^o_f = (1/2) sum_c |S(c,f) ∩ c^o|, S(c,f) the
#         strip of c over the full width of c normal to f and the tangential extent of f.
# Checks (symbolic):
#   S1  affine map x = x_c + diag(h) xi: the physical area of a polygon is |c| times its local area (Jacobian h_1 h_2 = |c|);
#   S2  the strip S(c,f) has local image [-1/2,1/2]_k x [r_lo, r_hi]_t, physical area |c| (r_hi - r_lo);
#       the physical half-plane of the solid, n_s.xi >= a_s, is (n_s1/h_1) x_1 + (n_s2/h_2) x_2 >= a_s in coordinates relative to x_c;
#   S3  no solid: (1/2) sum_c |c| (r_hi,c - r_lo,c) = |f| (1/2)(h_{c-,k} + h_{c+,k}) = V_f, since |f| = h_{c,t} (r_hi,c - r_lo,c);
#   S4  V^o_f != a_f V_f in general: a cell cut by a plane parallel to the face (counterexample, exact numbers);
#   S5  Phi-formula for the complement: |R ∩ {n.xi >= a}| = |R| - |R ∩ {n.xi <= a}| (the line has measure zero) -- with the
#       corner formula (4.8) in the corner regime this is the strip formula of (7.5).
# Run:    julia --project=@claude-symbolics proof_review/eq-cutcells-dual/scripts/step01_dual_symbolic.jl

using Symbolics, Test

@variables h1 h2 hk ht rlo1 rhi1 rlo2 rhi2 x1 x2 x3 x4 y1 y2 y3 y4
isz(e) = isequal(Symbolics.simplify(Symbolics.expand(e)), 0)
shoel(pts) = sum(
    pts[k][1] * pts[mod1(k + 1, length(pts))][2] - pts[mod1(k + 1, length(pts))][1] * pts[k][2] for k in eachindex(pts)
) / 2

@testset "S1 affine map: area scales with the Jacobian |c| = h1 h2" begin
    local_pts = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    phys_pts = [(h1 * p[1], h2 * p[2]) for p in local_pts]
    @test isz(shoel(phys_pts) - h1 * h2 * shoel(local_pts))
end

@testset "S2 strip and half-plane in local versus physical coordinates" begin
    # strip of a cell with face normal along axis 1: local [-1/2,1/2] x [rlo, rhi]; physical widths h1 and h2 (rhi-rlo)
    strip_local = [(-1 // 2, rlo2), (1 // 2, rlo2), (1 // 2, rhi2), (-1 // 2, rhi2)]
    strip_phys = [(h1 * p[1], h2 * p[2]) for p in strip_local]
    @test isz(shoel(strip_phys) - h1 * h2 * (rhi2 - rlo2))                       # |c| (r_hi - r_lo)
    # physical face length |f| = h2 (rhi - rlo) -> |S| = h1 |f|
    @test isz(shoel(strip_phys) - h1 * (h2 * (rhi2 - rlo2)))
    # half-plane n.xi >= a with xi = x/h (componentwise): (n1/h1) x1 + (n2/h2) x2 >= a
    @variables n1 n2 a
    @test isz((n1 * (x1 / h1) + n2 * (x2 / h2)) - ((n1 / h1) * x1 + (n2 / h2) * x2))
end

@testset "S3 without a solid the open dual volume is the extruded face (3.3)" begin
    # face normal along axis k with cell widths hk (normal) and ht (tangential) in the cell c; |f| = ht (rhi - rlo) for both cells
    # minus cell: normal width hkm, tangential width htm, extent [rlo_m, rhi_m]; plus cell likewise; |f| common
    @variables hkm hkp htm htp rlom rhim rlop rhip
    # |f| = htm (rhim - rlom) = htp (rhip - rlop): impose by solving for rhim, rhip
    f = Symbolics.Num(Symbolics.variable(:f))
    rhim_val = rlom + f / htm
    rhip_val = rlop + f / htp
    V = (1 // 2) * (hkm * htm * (rhim_val - rlom) + hkp * htp * (rhip_val - rlop))   # (1/2) sum |c| (rhi - rlo)
    @test isz(V - f * (hkm + hkp) / 2)                                                 # = |f| (1/2)(h_{c-,k} + h_{c+,k}) = V_f of (3.3)
end

@testset "S4 V^o_f != a_f V_f: cell cut by a plane parallel to the face" begin
    # minus cell cut by xi_1 <= a_s (solid on the left, open for xi_1 >= a_s), n_s = (1,0); plus cell uncut; square unit cells,
    # face at xi_1 = +1/2 of the minus cell (full side: rlo = -1/2, rhi = 1/2).
    @variables as
    # open fraction of the minus strip: area of [-1/2,1/2]^2 with xi_1 >= as = 1/2 - as
    open_minus = (1 // 2 - as) * 1
    open_plus = 1
    Vopen = (1 // 2) * (open_minus + open_plus)
    # traces of the face xi_1 = 1/2 (plus side of the minus cell): open for as < 1/2: trace 1; plus cell uncut: 1
    af = (1 + 1) // 2
    Vfull = (1 // 2) * (1 + 1)
    @test isz(Vopen - (3 // 4 - as / 2))
    @test isz(Symbolics.substitute(Vopen, Dict(as => 0)) - 3 // 4)         # half the cell solid: V^o = 3/4 V
    @test isz(Symbolics.substitute(af * Vfull, Dict(as => 0)) - 1)         # a_f V_f = 1
    @test !isz(Symbolics.substitute(Vopen - af * Vfull, Dict(as => 0)))
end

@testset "S5 complement formula |R ∩ {n.xi >= a}| = |R| - |R ∩ {n.xi <= a}| in the corner regime" begin
    # rectangle R = [0,w1] x [0,w2] (lowest corner at the origin), line w-weighted: u1 + u2 <= z with u_k in [0, w_k]
    @variables w1 w2 z
    corner = z^2 / (2 * w1 * w2)                                  # area fraction of {<= z} for z < min(w1,w2), (4.8)
    area_le = w1 * w2 * corner                                    # = z^2 / 2
    area_ge = w1 * w2 - area_le
    # direct shoelace of the polygon above the line: vertices (z,0),(w1,0),(w1,w2),(0,w2),(0,z) with w1,w2 >= z
    above = shoel([(z, 0), (w1, 0), (w1, w2), (0, w2), (0, z)])
    @test isz(above - area_ge)
end
