# Step 01 (symbolic): the algebra behind the dilation bound of the projected carrier.
#
# Label:  none (suggested eq:vof:projected-dilation-bound) -- Section 5.9, sentence after the two bullets
# Source: docs/techref/chapters/05_vof_transport.tex:797-813
# Checks (Symbolics.jl):
#   S1  Gauss on the composite toy mesh (coarse A, fine B1, B2): sum_c |c^o| (div^o v)_c = 0 for face-owned fluxes with zero
#       boundary flux, hence the dilation sum over chi = 1 cells equals minus the sum over chi = 0 cells: the signed sum over
#       chi = 1 is the net flux through the boundary of the chi = 1 region, which the stopping rule does not control.
#   S2  The dilation of one step is dt sum_{chi_c = 1} |c^o| (div^o v)_c (definition (5.8)/(3.x) with a_f v_f as carrier).
#   S3  Chain: with |x_c| <= tau |c^o| on the chi = 1 cells, |sum x_c| <= sum |x_c| <= tau sum |c^o|; equality iff all x_c = +tau |c^o|
#       (or all -tau |c^o|): written as an identity for the slack variables.
#   S4  Equality example: x_c = tau |c^o| on chi = 1 cells gives dilation = dt tau sum_{chi=1} |c^o| exactly.
#   S5  A solver target t = max(projtol, floor) replaces projtol: for projtol < floor the printed bound dt projtol sum is smaller than the
#       bound dt floor sum by the factor projtol/floor.
# Run:    julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/vof-projected-dilation/scripts/step01_projected_dilation_symbolic.jl

using Symbolics, Test

expanded(x) = Symbolics.simplify(Symbolics.expand(x))
iszero_sym(x) = isequal(expanded(x), 0)

@testset "projected dilation bound: algebra (symbolic)" begin
    @testset "S1 Gauss on the toy mesh" begin
        @variables a1 a2 a3 v1 v2 v3 f1 f2 f3
        # faces: f1 = A|B1, f2 = A|B2 (x faces), f3 = B2|B1 (y face); orientation +1 for the minus cell
        flux = (f1 * a1 * v1, f2 * a2 * v2, f3 * a3 * v3)
        divA = flux[1] + flux[2]
        divB1 = -flux[1] - flux[3]
        divB2 = -flux[2] + flux[3]
        @test iszero_sym(divA + divB1 + divB2)
    end
    @testset "S2/S4 dilation and the equality example" begin
        @variables dt tau VA VB1 VB2 chiA chiB1 chiB2 xA xB1 xB2
        V = (VA, VB1, VB2); chi = (chiA, chiB1, chiB2)
        x = (tau * VA, tau * VB1, tau * VB2)                                 # |c^o| div_c = tau |c^o|
        dilation = dt * sum(chi[i] * x[i] for i in 1:3)
        @test iszero_sym(dilation - dt * tau * sum(chi[i] * V[i] for i in 1:3))
    end
    @testset "S3 the chain |sum| <= sum |.| <= tau sum |c^o| (slack form)" begin
        @variables tau V1 V2 s1 s2 e1 e2
        # x_i = tau V_i (1 - s_i) e_i with slack s_i >= 0 and sign e_i in {-1, +1}; |x_i| = tau V_i (1 - s_i) <= tau V_i
        for e in ((1, 1), (1, -1), (-1, 1), (-1, -1))
            x1 = tau * V1 * (1 - s1) * e[1]; x2 = tau * V2 * (1 - s2) * e[2]
            absum = tau * V1 * (1 - s1) + tau * V2 * (1 - s2)                # sum |x_i|
            bound = tau * (V1 + V2)
            @test iszero_sym(bound - absum - tau * (V1 * s1 + V2 * s2))      # bound - sum|x| = tau sum V s >= 0
            if e[1] == e[2]
                @test iszero_sym((x1 + x2)^2 - absum^2)                    # |sum| = sum |.| iff the signs agree
            end
        end
    end
    @testset "S5 solver target replaces projtol" begin
        @variables dt projtol floor Vsum
        printed = dt * projtol * Vsum; correct = dt * floor * Vsum
        @test iszero_sym(printed - correct * (projtol / floor))
    end
end
