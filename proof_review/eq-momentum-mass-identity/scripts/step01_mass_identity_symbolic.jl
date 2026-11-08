# Step 01 (symbolic): the staggered mass identity of one sweep and the exact defect when conditions (a), (b) fail.
#
# Label:  eq:momentum:mass-identity  (with eq:momentum:identity, eq:momentum:predicted, eq:momentum:sweep-dilation, eq:vof:sweep)
# Source: docs/techref/chapters/06_momentum_transport.tex:586-650 (mass identity and conditions (a), (b))
# Setting (Symbolics.jl): the local patch of step01 of eq:momentum:identity (unknown f, its two cells, arbitrary hanging side
# partitions, the transverse faces and neighbours), one sweep along the axis xi.  Primal liquid / carrier volumes Pl_h, Po_h on the
# faces of axis xi (gas volume Po - Pl); per cell c of f: open volume O_c, open strip area L_c = |Lambda_{c,f} cap c^o|, previous
# conditional fraction C_c, frozen indicator chi_c, clamp change Delta_c (predicted fraction = pre-clamp + Delta_c):
#     sigma_c  = O_c^{-1} sum_h o_{c,h} Po_h                        (eq:momentum:sweep-dilation, directional carrier dilation)
#     dC_c     = O_c^{-1} sum_h o_{c,h} (chi_c Po_h - Pl_h)         (liquid fraction update of the sweep, eq:vof:sweep)
#     rho(C)   = rg + (rl - rg) C,   V_f rho_f^{[j]} = (1/2) sum_c L_c rho(C_c + dC_c + Delta_c)        (eq:momentum:predicted)
# The cell relations are imposed by eliminating one unshared face per cell, so sigma_c and dC_c are independent symbols.  The
# defect of eq:momentum:mass-identity is
#     d_f = V_f (rho^{[j]} - rho^{[j-1]}) + sum_pi rho^pi D[Phi^pi]_f - V_f H[rho chi sigma]_f ,
# D the dual divergence of the axis-xi dual faces built from rules (N), (T) (V_f = (1/2) sum_c L_c).
#   S1  d_f = -(1/2) sum_c ( L_c - varpi_{c,f} O_c ) mu_c + (1/2) sum_c L_c (rl - rg) Delta_c,
#       mu_c = rho^chi_c sigma_c - (rl - rg) dC_c = sum_pi rho^pi (div Phi^pi)_c,  rho^chi_c = rg + (rl - rg) chi_c,
#       for the component along the sweep (axial rule (N)) and across it (transverse rule (T)), arbitrary hanging partitions;
#   S2  under (a) Delta_c = 0 and (b) L_c = varpi_{c,f} O_c in both cells the defect vanishes identically (conditions of the text);
#   S3  (b) and (a) are necessary: with L_c != varpi O_c in one cell, or a clamp, the defect is generically non-zero; a pure-liquid
#       cell has mu_c = rl sigma_c, so even a pure-liquid cut coarse cell with two segments on a side gives the defect
#       -(1/2)(L - varpi O) rl sigma_c;
#   S4  the old state: with the stored (snapped) fraction of the previous sweep S1 holds; reading rho^{[j-1]} as the pre-snap predicted
#       density of the previous sweep adds -(1/2) sum_c L_c (rl - rg) snap_c (snap_c the change by the snap).
# Run:    julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/eq-momentum-mass-identity/scripts/step01_mass_identity_symbolic.jl

using Symbolics, Test, Random
isdefined(Main, :MomentumSymbolic) || include(joinpath(@__DIR__, "..", "..", "_shared", "momentum_symbolic.jl"))
using .MomentumSymbolic

@variables rl rg
const CELLS = (:cm, :cp)

"""Sweep data of a patch for the component along (`axial = true`) or across (`axial = false`) the sweep axis."""
function mass_defect(patch; axial, with_clamp = true, old_extra = (c -> 0))
    (; orient, trans_of) = patch
    faces_of(cell) = axial ? [(fh, orient[(cell, fh)]) for fh in axial_faces(patch, cell)] : trans_of[cell]
    names = unique(vcat([first(x) for c in CELLS for x in faces_of(c)]))
    sym(prefix, name) = Symbolics.variable(Symbol(prefix, name))
    Pl = Dict(name => sym("Pl_", name) for name in names)
    Po = Dict(name => sym("Po_", name) for name in names)
    O = Dict(c => sym("O_", c) for c in CELLS)
    L = Dict(c => sym("L_", c) for c in CELLS)
    C = Dict(c => sym("C_", c) for c in CELLS)
    chi = Dict(c => sym("chi_", c) for c in CELLS)
    Delta = Dict(c => sym("Delta_", c) for c in CELLS)
    sig = Dict(c => sym("sigma_", c) for c in CELLS)
    dC = Dict(c => sym("dC_", c) for c in CELLS)
    # cell relations:  sum_h o Po_h = O sigma   and   sum_h o Pl_h = chi O sigma - O dC   (eliminate one unshared face)
    rules = Dict{Any, Any}()
    for c in CELLS
        list = faces_of(c)
        shared(h) = count(cc -> any(y -> y[1] == h, faces_of(cc)), CELLS) > 1
        (e, oe) = first(x for x in list if x[1] != :f && !shared(x[1]))
        others = [x for x in list if x[1] != e]
        rules[Po[e]] = oe * (O[c] * sig[c] - sum(o * Po[h] for (h, o) in others; init = 0))
        rules[Pl[e]] = oe * (chi[c] * O[c] * sig[c] - O[c] * dC[c] - sum(o * Pl[h] for (h, o) in others; init = 0))
    end
    # dual divergence of the liquid and gas volumes (gas = Po - Pl) at f; only faces of axis xi carry volume
    Pg = Dict(name => Po[name] - Pl[name] for name in names)
    allnames = collect(keys(all_fluxes(patch)))
    div_of(P) = begin
        da, dt = divergence_at_f(patch, Dict{Symbol, Any}(name => get(P, name, 0) for name in allnames))
        axial ? da : dt
    end
    Dl, Dg = div_of(Pl), div_of(Pg)
    varpi(c) = frac_of(patch, c, orient[(c, :f)], :f)
    rho(Cx) = rg + (rl - rg) * Cx
    Vrho_new = sum(L[c] * rho(C[c] + dC[c] + (with_clamp ? Delta[c] : 0)) for c in CELLS) / 2
    Vrho_old = sum(L[c] * rho(C[c] + old_extra(c)) for c in CELLS) / 2
    rhochi(c) = rg + (rl - rg) * chi[c]
    VHrhochisigma = sum(L[c] * rhochi(c) * sig[c] for c in CELLS) / 2
    defect = substitute((Vrho_new - Vrho_old) + (rl * Dl + rg * Dg) - VHrhochisigma, rules)
    mu(c) = rhochi(c) * sig[c] - (rl - rg) * dC[c]
    predicted = -sum((L[c] - varpi(c) * O[c]) * mu(c) for c in CELLS) / 2 +
        (with_clamp ? sum(L[c] * (rl - rg) * Delta[c] for c in CELLS) / 2 : 0)
    return (; defect, predicted, L, O, varpi, Delta, chi, sig, dC, mu, C)
end

@testset "eq:momentum:mass-identity symbolic" begin
    @testset "S1 the exact defect (axial and transverse component, random hanging topologies)" begin
        for (seed, axial) in ((1, true), (2, true), (3, true), (4, false), (5, false), (6, false), (7, true), (8, false), (9, true), (10, false))
            patch = build_patch(MersenneTwister(seed); constrained = true)
            r = mass_defect(patch; axial)
            @test iszero_sym(r.defect - r.predicted)
        end
    end
    @testset "S2 conditions (a) and (b): the defect vanishes" begin
        for (seed, axial) in ((11, true), (12, false), (13, true), (14, false))
            patch = build_patch(MersenneTwister(seed); constrained = true)
            r = mass_defect(patch; axial, with_clamp = false)
            cond_b = Dict{Any, Any}(r.L[c] => r.varpi(c) * r.O[c] for c in CELLS)
            @test iszero_sym(substitute(r.defect, cond_b))
        end
    end
    @testset "S3 necessity of (a) and (b)" begin
        for axial in (true, false)
            patch = build_patch(MersenneTwister(21); constrained = true)
            r = mass_defect(patch; axial, with_clamp = false)
            @test !iszero_sym(r.defect)                                  # (b) not imposed
            cond_b = Dict{Any, Any}(r.L[c] => r.varpi(c) * r.O[c] for c in CELLS)
            rc = mass_defect(patch; axial, with_clamp = true)
            @test !iszero_sym(substitute(rc.defect, cond_b))             # (b) holds, (a) violated: the clamp term remains
            @test iszero_sym(substitute(rc.defect, cond_b) - sum(rc.varpi(c) * rc.O[c] * (rl - rg) * rc.Delta[c] for c in CELLS) / 2)
            # pure-liquid cells (chi = 1, no fraction change): defect = -(1/2) sum_c (L - varpi O) rl sigma_c
            pure = Dict{Any, Any}()
            for c in CELLS
                pure[r.chi[c]] = 1
                pure[r.dC[c]] = 0
            end
            @test iszero_sym(substitute(r.defect, pure) + sum((r.L[c] - r.varpi(c) * r.O[c]) * rl * r.sig[c] for c in CELLS) / 2)
        end
    end
    @testset "S4 snapped versus pre-snap old state in the next sweep" begin
        @variables snap_cm snap_cp
        snap = Dict(:cm => snap_cm, :cp => snap_cp)
        patch = build_patch(MersenneTwister(31); constrained = true)
        stored = mass_defect(patch; axial = false, with_clamp = false)
        literal = mass_defect(patch; axial = false, with_clamp = false, old_extra = c -> snap[c])
        @test iszero_sym(literal.defect - (stored.defect - sum(stored.L[c] * (rl - rg) * snap[c] for c in CELLS) / 2))
        cond_b = Dict{Any, Any}(stored.L[c] => stored.varpi(c) * stored.O[c] for c in CELLS)
        @test iszero_sym(substitute(stored.defect, cond_b))              # stored (snapped) old state: identity holds under (b)
        @test !iszero_sym(substitute(literal.defect, cond_b))            # pre-snap old state: a defect of the size of the snap
    end
end
