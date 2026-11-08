# Step 01 (symbolic): the cell sum of the scalar update (5.18) gives the scalar balance (5.32); snap and reset terms.
#
# Label:  eq:vof:scalar-balance  (with eq:vof:scalar = (5.18), eq:vof:fluctuation = (5.19), eq:vof:reset = (5.29))
# Source: docs/techref/chapters/05_vof_transport.tex:513-526 (5.18), 816-837 (5.31 and the sentence on snaps and resets)
# Model:  the composite configuration of eq-vof-volume-balance/step01 (coarse A, fine B1, B2; faces f1, f2 in sweep 1, f3 in
#         sweep 2), phase volumes P_f, carrier volumes O_f and FACE-OWNED states s_f (one state per face, seen by both cells).
# Checks: S1 summing (5.18) over cells and sweeps gives (5.32):
#            sum_c |c^o| (C a)^{n+1} - sum_c |c^o| (C a)^{n} = sum_c chi_c a^n_c sum_{f in F(c)} o_{c,f} O_f ,
#            i.e. dt sum_c chi_c a^n_c |c| (div upsilon^open)_c, when no fraction is clamped or snapped;
#         S2 the frozen weights chi_c a^n_c (not the current a^[j-1]) are what makes both sweeps add up:
#            with the current value in the second sweep the sum would differ by sum_c chi_c (a^[1]_c - a^n_c) * (sweep-2 carrier divergence);
#         S3 with snapping, (5.32) acquires sum_{j,c} |c^o| * dev_{j,c} where dev = C_s a_new - A518 is (C_s - Cpre) a^[j-1]
#            for a snap to a positive fraction and -A518 for a snap to zero (eq-vof-fluctuation S2, S3);
#         S4 projected carrier, gas: the auxiliary carrier has no dilation term; the physical amount after the step differs
#            from the sum of the auxiliary amounts by the reset terms of (5.29):
#            P_2 - P_0 = sum_j sum_c |c^o| (1 - C^[j] - G*^[j]) a^[j]  (no snaps).
# Run:    julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/eq-vof-scalar-balance/scripts/step01_scalar_balance_symbolic.jl

using Symbolics, Test

zero_expr(e) = isequal(Symbolics.simplify(Symbolics.expand(e)), 0)

@variables VA VB1 VB2                          # open volumes
@variables CA CB1 CB2                           # phase fractions before the step
@variables aA aB1 aB2                           # intrinsic values before the step (= frozen dilation values a^n)
@variables chiA chiB1 chiB2
@variables P1 P2 P3 O1 O2 O3 s1 s2 s3           # phase volumes, carrier volumes, face states of f1, f2, f3
@variables dA1 dB11 dB21 dA2 dB12 dB22          # deviations dev_{j,c} (snaps), times the open volume below

names = (:A, :B1, :B2)
V = Dict(:A => VA, :B1 => VB1, :B2 => VB2)
C0 = Dict(:A => CA, :B1 => CB1, :B2 => CB2)
a0 = Dict(:A => aA, :B1 => aB1, :B2 => aB2)
chi = Dict(:A => chiA, :B1 => chiB1, :B2 => chiB2)
incidence = Dict(:A => [(1, +1, 1), (2, +1, 1)], :B1 => [(1, -1, 1), (3, -1, 2)], :B2 => [(2, -1, 1), (3, +1, 2)])
Pf = (P1, P2, P3); Of = (O1, O2, O3); sf = (s1, s2, s3)
dev = Dict((:A, 1) => dA1, (:B1, 1) => dB11, (:B2, 1) => dB21, (:A, 2) => dA2, (:B1, 2) => dB12, (:B2, 2) => dB22)

# amount update (5.18) per sweep, frozen weights chi a^n
function amount_after(name, amount_before, axis; frozen = true, a_current = nothing)
    weight = frozen ? chi[name] * a0[name] : chi[name] * a_current
    flux = sum((o * (Pf[f] * sf[f] - weight * Of[f]) for (f, o, ax) in incidence[name] if ax == axis); init = 0)
    return amount_before - flux / V[name]
end

@testset "eq:vof:scalar-balance symbolic" begin
    total0 = sum(V[n] * C0[n] * a0[n] for n in names)
    amount1 = Dict(n => amount_after(n, C0[n] * a0[n], 1) for n in names)           # per open volume, C a after sweep 1
    amount2 = Dict(n => amount_after(n, amount1[n], 2) for n in names)
    total2 = sum(V[n] * amount2[n] for n in names)
    dilation = sum(chi[n] * a0[n] * sum((o * Of[f] for (f, o, ax) in incidence[n]); init = 0) for n in names)
    @testset "S1 (5.32)" begin
        @test zero_expr((total2 - total0) - dilation)
    end
    @testset "S2 the dilation weight must be frozen" begin
        @variables a1A a1B1 a1B2                     # values after sweep 1 (differ from a^n)
        a1 = Dict(:A => a1A, :B1 => a1B1, :B2 => a1B2)
        amount2_current = Dict(n => amount_after(n, amount1[n], 2; frozen = false, a_current = a1[n]) for n in names)
        total2_current = sum(V[n] * amount2_current[n] for n in names)
        defect = (total2_current - total0) - dilation
        expected = sum(chi[n] * (a1[n] - a0[n]) * sum((o * Of[f] for (f, o, ax) in incidence[n] if ax == 2); init = 0)
            for n in names)
        @test !zero_expr(defect)
        @test zero_expr(defect - expected)
    end
    @testset "S3 snaps add the deviations" begin
        amount1s = Dict(n => amount1[n] + dev[(n, 1)] for n in names)
        amount2s = Dict(n => amount_after(n, amount1s[n], 2) + dev[(n, 2)] for n in names)
        total2s = sum(V[n] * amount2s[n] for n in names)
        snaps = sum(V[n] * (dev[(n, 1)] + dev[(n, 2)]) for n in names)
        @test zero_expr((total2s - total0) - dilation - snaps)
    end
    @testset "S4 auxiliary gas carrier plus resets (5.29), no dilation for the gas" begin
        @variables G0A G0B1 G0B2 R1A R1B1 R1B2 R2A R2B1 R2B2        # reset defects (1 - C^[j] - G*^[j]) times a^[j]
        # physical amount P_0 = sum |c^o| (1 - C0) a0 ; aux amounts have no dilation: chi = 0
        gas0 = Dict(:A => 1 - CA, :B1 => 1 - CB1, :B2 => 1 - CB2)
        function aux_sweep(name, amount, axis)
            flux = sum((o * Pf[f] * sf[f] for (f, o, ax) in incidence[name] if ax == axis); init = 0)
            return amount - flux / V[name]
        end
        P0 = sum(V[n] * gas0[n] * a0[n] for n in names)
        reset1 = Dict(:A => R1A, :B1 => R1B1, :B2 => R1B2); reset2 = Dict(:A => R2A, :B1 => R2B1, :B2 => R2B2)
        M1 = Dict(n => aux_sweep(n, gas0[n] * a0[n], 1) for n in names)               # aux amount after sweep 1
        P1 = Dict(n => M1[n] + reset1[n] for n in names)                              # physical amount after the reset
        M2 = Dict(n => aux_sweep(n, P1[n], 2) for n in names)
        P2 = Dict(n => M2[n] + reset2[n] for n in names)
        @test zero_expr(sum(V[n] * P2[n] for n in names) - P0 - sum(V[n] * (reset1[n] + reset2[n]) for n in names))
    end
end
