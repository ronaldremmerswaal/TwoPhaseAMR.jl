# Step 01 (symbolic): the divergence identity of the dual faces, rules (N) and (T) of Chapter 6.
#
# Label:  eq:momentum:identity  (with eq:momentum:weights = rules (N), (T); eq:momentum:dual-divergence)
# Source: docs/techref/chapters/06_momentum_transport.tex:132-147 (weights), 189-215 (divergence identity)
# Checks (Symbolics.jl).  A local patch around one unknown f of axis k: its two cells c^-, c^+ with arbitrary side partitions
# (one to three face segments per side), the transverse faces of both cells (one or two on each of top and bottom) and the
# cell across each of them, again with arbitrary side partitions.  The dual faces at f are built by the two rules from the
# printed formulas, the dual divergence at f is accumulated with the orientations sigma_{f,g} = +1 (tail) / -1 (head), and
# compared with the printed right-hand side  sum_c (varpi_{c,f}/2) sum_{h in F(c) cap F^(t)} o_{c,h} Phi_h.
#   S1  With free (unconstrained) fractions the residual of the axial part (t = k) is exactly
#           -(1/2) Phi_f ( o_{c^-,f} S_{c^-} + o_{c^+,f} S_{c^+} ) = -(1/2) Phi_f ( S_{c^-} - S_{c^+} ),
#       S_c = sum of the fractions of ALL axial faces of c (both sides).
#   S2  With free fractions the residual of the transverse part (t != k) is exactly
#           sum_{c in C(f)} sum_{h in F(c) cap F^(t)} (1/2) varpi_{c,f} o_{c,h} Phi_h ( S^{s(c)}_{c'(h)} - 1 ),
#       c'(h) the cell across the transverse face h, s(c) the side of c on which f lies, S^s_{c'} the fractions on side s of c'.
#   S3  If every side partition sums to one (the printed hypothesis, and for the neighbours across the transverse faces too)
#       both residuals vanish identically: the identity of eq:momentum:identity.
#   S4  Emptying the side s(c) of the neighbours across the transverse faces of ONE cell (a wall side without unknown)
#       loses exactly the flux terms predicted by S2 (the mechanism of issue 60); emptying the opposite side changes nothing.
#   S5  A wall side in one of the two cells of f (S_c = 1) reproduces S1: the residual is (1/2) Phi_f, which is what the
#       boundary dual face of eq:momentum:boundary supplies (checked against the package in code_02).
# Run:    julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/eq-momentum-identity/scripts/step01_identity_symbolic.jl

using Symbolics, Test, Random
isdefined(Main, :MomentumSymbolic) || include(joinpath(@__DIR__, "..", "..", "_shared", "momentum_symbolic.jl"))
using .MomentumSymbolic

@testset "eq:momentum:identity symbolic" begin
    @testset "S1/S2 closed-form residuals for free fractions (random topologies)" begin
        rng = MersenneTwister(20261108)
        for trial in 1:25
            patch = build_patch(rng; constrained = false)
            Phi = all_fluxes(patch)
            div_ax, div_tr = divergence_at_f(patch, Phi)
            rhs_ax, rhs_tr = printed_rhs(patch, Phi)
            pred_ax, pred_tr = predicted_residuals(patch, Phi)
            @test iszero_sym((div_ax - rhs_ax) - pred_ax)
            @test iszero_sym((div_tr - rhs_tr) - pred_tr)
        end
    end
    @testset "S3 the identity for side partitions summing to one" begin
        rng = MersenneTwister(77)
        for trial in 1:25
            patch = build_patch(rng; constrained = true)
            Phi = all_fluxes(patch)
            div_ax, div_tr = divergence_at_f(patch, Phi)
            rhs_ax, rhs_tr = printed_rhs(patch, Phi)
            @test iszero_sym(div_ax - rhs_ax)
            @test iszero_sym(div_tr - rhs_tr)
        end
    end
    @testset "S4 a wall side on the neighbours across the transverse faces" begin
        for trial in 1:10
            # (a) empty the neighbours' side OPPOSITE to the side of f in cm: nothing changes
            patch = build_patch(MersenneTwister(trial); constrained = true)
            for (u, h, o) in neighbours_of(patch, :cm)
                patch.sides[(u, -1)] = EMPTY             # f lies on the high side of cm
            end
            Phi = all_fluxes(patch)
            div_ax, div_tr = divergence_at_f(patch, Phi)
            rhs_ax, rhs_tr = printed_rhs(patch, Phi)
            @test iszero_sym(div_ax - rhs_ax)
            @test iszero_sym(div_tr - rhs_tr)
            # (b) empty the side of the neighbours on which f lies in cm: the fluxes of the transverse faces of cm are lost
            patch = build_patch(MersenneTwister(100 + trial); constrained = true)
            nb = neighbours_of(patch, :cm)
            for (u, h, o) in nb
                patch.sides[(u, +1)] = EMPTY
            end
            Phi = all_fluxes(patch)
            div_ax, div_tr = divergence_at_f(patch, Phi)
            rhs_ax, rhs_tr = printed_rhs(patch, Phi)
            varpi = frac_of(patch, :cm, +1, :f)
            lost = sum(varpi / 2 * o * Phi[h] for (u, h, o) in nb; init = 0)
            @test iszero_sym(div_ax - rhs_ax)
            @test iszero_sym((div_tr - rhs_tr) + lost)      # exactly the lost terms
            @test !iszero_sym(div_tr - rhs_tr)
            pred_ax, pred_tr = predicted_residuals(patch, Phi)
            @test iszero_sym((div_tr - rhs_tr) - pred_tr)
        end
    end
    @testset "S5 a wall side on a cell of f: residual (1/2) Phi_f, supplied by the boundary dual face" begin
        for trial in 1:10
            patch = build_patch(MersenneTwister(300 + trial); constrained = true)
            patch.sides[(:cm, -1)] = EMPTY                 # a wall normal to e_k behind cm; f stays on its high side
            Phi = all_fluxes(patch)
            div_ax, div_tr = divergence_at_f(patch, Phi)
            rhs_ax, rhs_tr = printed_rhs(patch, Phi)
            # -(1/2) Phi_f (o_cm S_cm + o_cp S_cp) with S_cm = 1, S_cp = 2
            @test iszero_sym((div_ax - rhs_ax) - Phi[:f] / 2)
            @test iszero_sym(div_tr - rhs_tr)
        end
    end
end
