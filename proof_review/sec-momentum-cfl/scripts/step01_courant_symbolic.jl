# Step 01 (symbolic): certificates for the dual Courant bound of the two-phase transport and the convexity threshold of the upwind update.
#
# Label:  sec:momentum:cfl  (last paragraph: nu_g <= nu^dagger on a uniform mesh without solid; eq:momentum:cfl and its convexity sentence)
# Source: docs/techref/chapters/06_momentum_transport.tex:535-575; docs/techref/chapters/05_vof_transport.tex (eq:vof:cfl)
# Checks (Symbolics.jl).  nu = nu^dagger, |c| the cell volume, Phi_h = dt |h| v_h the time-integrated volume of a primal face; the WY bound
# gives, for every cell, in_c <= nu |c| and out_c <= nu |c| (in_c, out_c = volume entering / leaving c through all its faces):
#   S1  Uniform mesh, dual face through the centre of a cell c (rule (N)): Upsilon = (a + b)/2, a through the low face and b through the high
#       face of c, positive along e_k; R_in, R_out the volumes through the other faces of c (>= 0).  For each sign pattern of (a, b) an explicit
#       certificate  nu |c| - |Upsilon| = (sum of non-negative terms)  is verified, so |Upsilon| <= nu |c| and nu_g <= nu on V_f = |c|.
#       Transverse dual faces (rule (T)): Upsilon = (h + h')/2 with |h|, |h'| <= nu |c| (a single face volume is at most the in or out volume).
#   S2  Composite mesh, the hanging fine segment f_1 of a coarse cell c (fine neighbour c', |c'| = |c|/4, V_{f_1} = 3/8 |c|): the dual face to the
#       opposite coarse face has Upsilon = (1/2) Phi_{f_1} + (1/4) Phi_opp, with |Phi_opp| <= nu |c| and |Phi_{f_1}| <= nu |c'| = nu |c|/4 (the in/out
#       cap of the SMALLER cell): the supremum of |Upsilon| is exactly nu V_{f_1}, so the ratio is 1 (not more): the small control volume is
#       compensated by the tighter cap.  All dual-face types are covered by the exact linear programs of sharp_01.
#   S3  Convexity of the unsplit upwind update at an unknown with signed outflows s_g = sigma_g U_g, sum_g s_g = 0 (solenoidal): Out = In =
#       (1/2) sum_g |U_g|, so with dt = c dt^stag(cfl = 1) the diagonal coefficient 1 - (dt/V) Out equals 1 - c/2 at the control volume of the
#       largest rate: non-negative exactly for c <= 2 (the text's c <= 1 is conservative by the factor 2).
# Run:    julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/sec-momentum-cfl/scripts/step01_courant_symbolic.jl

using Symbolics, Test, Random

expanded(e) = Symbolics.simplify(Symbolics.expand(e))
iszero_sym(e) = isequal(expanded(e), 0)

@testset "sec:momentum:cfl symbolic" begin
    @testset "S1 certificates on the uniform mesh (all sign patterns)" begin
        @variables nuc A B Rin Rout                         # nu |c|, |a|, |b| >= 0 and the volumes through the other faces
        # orientations: low face o = -1, high face o = +1; a, b positive along e_k.  The volume LEAVING c: a < 0 (low face), b > 0 (high face).
        for (sa, sb) in ((1, 1), (-1, -1), (1, -1), (-1, 1))
            a_out = sa < 0 ? A : 0
            a_in = sa > 0 ? A : 0
            b_out = sb > 0 ? B : 0
            b_in = sb < 0 ? B : 0
            out_c = a_out + b_out + Rout
            in_c = a_in + b_in + Rin
            if sa == sb
                # one face carries inflow, the other outflow: |Upsilon| = (A + B)/2 and in_c + out_c = A + B + Rin + Rout
                certificate = (nuc - in_c) / 2 + (nuc - out_c) / 2 + Rin / 2 + Rout / 2
                @test iszero_sym((nuc - (A + B) / 2) - certificate)
            else
                # both faces carry inflow (sa = +1, sb = -1) or both outflow: X = in_c or out_c contains A + B
                X = sa > 0 ? in_c : out_c
                R = sa > 0 ? Rin : Rout
                # A >= B: |Upsilon| = (A - B)/2 ;  B >= A: (B - A)/2
                @test iszero_sym((nuc - (A - B) / 2) - ((nuc - X) + (A + 3B) / 2 + R))
                @test iszero_sym((nuc - (B - A) / 2) - ((nuc - X) + (3A + B) / 2 + R))
            end
        end
        @variables h hp
        @test iszero_sym((nuc - (h + hp) / 2) - ((nuc - h) / 2 + (nuc - hp) / 2))              # transverse dual face: h, hp <= nu |c|
    end
    @testset "S2 the hanging segment compensates its small volume" begin
        @variables nuc phi1 phiopp
        volume = 3 // 8                                      # V_{f_1} / |c|
        extremal = Dict(phi1 => nuc / 4, phiopp => nuc)      # caps: nu |c'| = nu |c| / 4 and nu |c|
        @test iszero_sym(substitute(phi1 / 2 + phiopp / 4, extremal) - volume * nuc)
        # the other dual face of the same segment, to the other fine segment of the side: Upsilon = (Phi_{f_2} - Phi_{f_1})/4, same volumes
        @variables phi2
        @test iszero_sym(substitute((phi2 - phi1) / 4, Dict(phi2 => nuc / 4, phi1 => -nuc / 4)) - nuc / 8)
        @test (1 // 8) / volume == 1 // 3                    # ratio 1/3 of nu: far from the bound
    end
    @testset "S3 convexity threshold of the unsplit upwind update" begin
        rng = MersenneTwister(3)
        @variables c
        for trial in 1:40
            m = rand(rng, 3:7)
            eps = rand(rng, [-1, 1], m)                      # sign of the signed outflow s_g = sigma_g U_g
            mags = [Symbolics.variable(Symbol("m", trial, "_", i)) for i in 1:m]
            Out = sum(eps[i] > 0 ? mags[i] : 0 for i in 1:m)
            In = sum(eps[i] < 0 ? mags[i] : 0 for i in 1:m)
            balance = sum(eps[i] * mags[i] for i in 1:m)     # = sum_g s_g
            abs_sum = sum(mags)
            @test iszero_sym((Out - In) - balance)
            @test iszero_sym((2 * Out - abs_sum) - balance)  # solenoidal (balance = 0): Out = (1/2) sum |U_g|
            @test iszero_sym((2 * In - abs_sum) + balance)
        end
        @variables Out_ abs_sum_ V
        # control volume of the largest rate: dt = c V / sum|U|;  with Out = sum|U| / 2
        coefficient = 1 - (c * V / abs_sum_) / V * Out_
        @test iszero_sym(substitute(coefficient, Dict(Out_ => abs_sum_ / 2)) - (1 - c / 2))
        @test iszero_sym(substitute(coefficient, Dict(Out_ => abs_sum_ / 2, c => 2)))      # zero at c = 2: convex exactly up to 2
    end
end
