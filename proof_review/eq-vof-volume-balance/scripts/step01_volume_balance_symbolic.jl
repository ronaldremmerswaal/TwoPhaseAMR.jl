# Step 01 (symbolic): the cell sum of the sweep update (5.6) gives the volume balance (5.30).
#
# Label:  eq:vof:volume-balance  (with eq:vof:sweep = (5.6), eq:vof:step = (5.8))
# Source: docs/techref/chapters/05_vof_transport.tex:178-186 (5.6), 197-203 (5.8), 778-796 (5.30)
# Model:  the smallest composite configuration with both sweep axes and a hanging face: a coarse cell A with two fine
#         neighbours B1 (upper) and B2 (lower) on its right, so that A has two x-segments f1, f2 and B1, B2 share the y-face
#         f3.  Orientation o_{c,f} = +1 if c is the minus cell of f.  Boundary faces carry no volume and are omitted.
#         Sweep 1 moves f1, f2 (axis 1), sweep 2 moves f3 (axis 2), the carrier O_f is the same in both sweeps,
#         the indicator chi_c is frozen, snap changes delta_{j,c} enter as in (5.14).
# Checks: S1 the sum over cells of (5.6) over the two sweeps equals
#            (5.30):  sum_c |c^o| (C^[2]_c - C^[0]_c) = sum_c chi_c * sum_{f in F(c)} o_{c,f} O_f + sum_j sum_c |c^o| delta_{j,c};
#         S2 summing (5.8) cell by cell gives the same (the two-sweep form (5.8) = the sum of the sweeps (5.6));
#         S3 the gas volume sum_c |c^o| (1 - C_c) changes by the opposite amount.
# Run:    julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/eq-vof-volume-balance/scripts/step01_volume_balance_symbolic.jl

using Symbolics, Test

zero_expr(e) = isequal(Symbolics.simplify(Symbolics.expand(e)), 0)

@variables VA VB1 VB2                         # open cell volumes |c^o|
@variables CA CB1 CB2                          # fractions before the step
@variables chiA chiB1 chiB2                    # frozen indicators
@variables L1 L2 L3 O1 O2 O3                   # liquid and carrier volumes of f1, f2, f3
@variables d1A d1B1 d1B2 d2A d2B1 d2B2         # snap changes of sweeps 1 and 2

# incidences (cell, face, orientation)
cells = (:A => (VA, CA, chiA, d1A, d2A), :B1 => (VB1, CB1, chiB1, d1B1, d2B1), :B2 => (VB2, CB2, chiB2, d1B2, d2B2))
incidence = Dict(
    :A => [(1, +1, 1), (2, +1, 1)],           # (face number, orientation, axis): A is the minus cell of f1, f2
    :B1 => [(1, -1, 1), (3, -1, 2)],
    :B2 => [(2, -1, 1), (3, +1, 2)],
)
Lf = (L1, L2, L3); Of = (O1, O2, O3)

function sweep(name, V, C, chi, delta, axis)
    net = sum((o * (Lf[f] - chi * Of[f]) for (f, o, a) in incidence[name] if a == axis); init = 0)
    return C - net / V + delta                 # (5.6) divided by |c^o|
end

@testset "eq:vof:volume-balance symbolic" begin
    total_after = 0; total_before = 0; dilation = 0; snaps = 0; gas_change = 0
    step_after = 0
    for (name, (V, C, chi, d1, d2)) in cells
        C1 = sweep(name, V, C, chi, d1, 1)
        C2 = sweep(name, V, C1, chi, d2, 2)
        total_after += V * C2; total_before += V * C
        dilation += chi * sum((o * Of[f] for (f, o, a) in incidence[name]); init = 0)
        snaps += V * (d1 + d2)
        gas_change += V * (1 - C2) - V * (1 - C)
        # (5.8) in one line
        step_after += V * C - sum((o * Lf[f] for (f, o, a) in incidence[name]); init = 0) +
            chi * sum((o * Of[f] for (f, o, a) in incidence[name]); init = 0) + V * (d1 + d2)
    end
    @testset "S1 sum of the sweeps = (5.30)" begin
        @test zero_expr((total_after - total_before) - (dilation + snaps))
    end
    @testset "S2 (5.8) cell by cell sums to the same" begin
        @test zero_expr(step_after - total_after)
        @test zero_expr((step_after - total_before) - (dilation + snaps))
    end
    @testset "S3 the gas volume changes by the opposite amount" begin
        @test zero_expr(gas_change + (total_after - total_before))
    end
end
