# Code check 01: the staggered mass identity of the one-velocity sweeps, conditions (a), (b), the old state, momentum.
#
# Label:  eq:momentum:mass-identity  (eq:momentum:predicted, eq:momentum:one-velocity, eq:momentum:conservative)
# Source: docs/techref/chapters/06_momentum_transport.tex:586-650;
#         src/two_phase/momentum_workspace.jl (_advect_momentum_component!), src/vof/fluxes.jl (_advect_wy!)
# Setting: random composite meshes (periodic / walled, with and without a solid disc or sloped wall), random interface, smooth
# NON-solenoidal carrier, one `advect!` of the one-velocity model (LaxWendroff and MonotonizedCentral, both sweep orders, default
# fraction floor and a large floor 0.06 that forces snaps).  The code's primal phase volumes are replayed through the independent
# phase schedule (docs/techref/figures/momentum.jl) and the identity is evaluated unknown by unknown and sweep by sweep.
#   M1  (hard)  no clamp in the cells of f and condition (b)  =>  mass-identity defect = 0 to rounding (all unknown-sweeps);
#   M2  (hard)  (b) holds, a clamp acts: the defect is exactly the clamp term (1/V_f) sum_c w_c (rho^l - rho^g) Delta_c;
#   M3  (hard)  the old density of the identity is the density of the STORED (snapped) fraction at the start of the sweep: with
#               snaps in sweep 1 the sweep-2 defect is zero under (a), (b); with the pre-snap predicted density of sweep 1 as old
#               state ("rho^{[j-1]}" read literally from eq:momentum:predicted) it is not (size reported);
#   M4  (hard)  momentum: when every unknown-sweep satisfies (a), (b) and nothing snaps, the step changes the momentum
#               sum_f V_f rho_f v_f of each component by exactly sum_j sum_f V_f R^{[j]}_f v^n_f (the frozen-dilation term),
#               up to rounding;
#   M4a (hard)  the exact momentum bookkeeping of the replayed update, in EVERY fixture (with snaps, with failures of (b)):
#                 sum_f V rho^{[j]}_pred v^{[j]} - sum_f V rho^{[j-1]}_stored v^{[j-1]} = sum_f V d_f v^{[j-1]}_f + sum_f V R_f v^n_f
#               per sweep (d_f the mass defect, so a defect drifts the momentum by sum V d v), and the replica reproduces the code's
#               velocity after advect!;
#   M4c (hard)  a carrier that is solenoidal in every cell (potential part switched off): the momentum of each component is conserved up to
#               rounding when (a), (b) hold and nothing snaps (the two frozen dilations of a cell cancel);
#   M5  (excluded) (b) violated (cut coarse cell with two segments on a side, issue 63): the defect is non-zero; pre-fix it equals
#               -(1/V_f) sum_c (w_c - varpi O_c / 2) mu_c exactly (reported, not asserted).
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-momentum-mass-identity/scripts/code_01_mass_identity.jl

using Test, Random, LinearAlgebra
isdefined(Main, :MomentumAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "momentum_audit.jl"))
using .MomentumAudit
using TwoPhaseAMR
const MA = MomentumAudit
const D = MA.Docs

"""One fixture and step; returns `nothing` if the package refuses the state (reported as skipped)."""
function run_fixture(rng; periodic, solid_kind, levels, fraction_floor, order, density, scheme, potential = 0.4)
    field, velocity, solid = MA.random_two_phase(rng; periodic, solid_kind, levels, fraction_floor, potential)
    reconstruct!(field)
    work = WYAdvectionWorkspace(field)
    limit = stable_timestep(velocity, work)
    isfinite(limit) || return nothing
    dt = limit * (0.5 + 0.5 * rand(rng))
    mesh = field.mesh
    C0 = Dict(ref => D.cond(field, ref) for ref in D.all_cells(mesh))
    destination = FaceVelocity(mesh)
    interface_scheme = default_amr_interface_scheme(scheme)
    try
        advect!(
            field, velocity, dt, work; momentum = destination, density,
            momentum_workspace = MomentumAdvectionWorkspace(field), sweep_order = order,
            scheme, amr_interface_scheme = interface_scheme,
        )
    catch err
        (err isa ErrorException || err isa ArgumentError) || rethrow()
        return nothing
    end
    sweeps, Cfinal = MA.replay_mass_identity(field, work, C0, order, density; velocity, scheme)
    return (; field, velocity, destination, sweeps, C0, Cfinal, density, dt)
end

@testset "eq:momentum:mass-identity: one-velocity sweeps against the package" begin
    rng = MersenneTwister(63)
    stats = Dict(
        :fixtures => 0, :skipped => 0, :unknown_sweeps => 0, :a_and_b => 0, :b_only => 0, :not_b => 0, :not_b_nonzero => 0,
        :snap_unknowns => 0, :snap_literal_nonzero => 0, :momentum_checked => 0,
    )
    worst = Dict(:a_and_b => 0.0, :clamp_term => 0.0, :snap_stored => 0.0, :momentum => 0.0, :bookkeeping => 0.0, :replica => 0.0)
    literal_max = 0.0
    not_b_max = 0.0
    not_b_formula_gap = 0.0
    cases = Any[]
    for (periodic, solid_kind) in (((true, true), nothing), ((false, false), nothing), (((true, true)), :disc), ((false, false), :wall), ((false, false), :disc))
        for (floor, order, scheme, density) in (
                (nothing, (1, 2), LaxWendroff(), (1000.0, 1.0)), (0.06, (2, 1), MonotonizedCentral(), (1000.0, 1.0)),
                (nothing, (2, 1), LaxWendroff(), (3.7, 1.1)), (0.06, (1, 2), LaxWendroff(), (50.0, 1.0)),
            )
            push!(cases, (periodic, solid_kind, floor, order, scheme, density))
        end
    end
    for (periodic, solid_kind, floor, order, scheme, density) in cases, rep in 1:3
        r = run_fixture(rng; periodic, solid_kind, levels = 1, fraction_floor = floor, order, density, scheme)
        if r === nothing
            stats[:skipped] += 1
            continue
        end
        stats[:fixtures] += 1
        rho_l, rho_g = density
        all_ab = true
        any_snap = false
        for (j, sweep) in enumerate(r.sweeps), comp in sweep.comps
            n = comp.n
            for i in 1:n
                stats[:unknown_sweeps] += 1
                tol = 1.0e-12 * ((rho_l + rho_g) + abs(comp.mass[i]) + abs(comp.R[i]))      # rounding scale per unit volume
                a = !comp.clamp_flag[i]
                b = comp.b_flag[i]
                if b && a
                    stats[:a_and_b] += 1
                    worst[:a_and_b] = max(worst[:a_and_b], abs(comp.defect[i]) / rho_l)
                    @test abs(comp.defect[i]) <= tol                                         # M1
                    if j == 2 && comp.snap_flag[i]
                        stats[:snap_unknowns] += 1
                        worst[:snap_stored] = max(worst[:snap_stored], abs(comp.defect[i]) / rho_l)
                        abs(comp.defect_literal[i]) > 1.0e-9 * rho_l && (stats[:snap_literal_nonzero] += 1)
                        literal_max = max(literal_max, abs(comp.defect_literal[i]) / rho_l)
                    end
                elseif b
                    stats[:b_only] += 1
                    gap = abs(comp.defect[i] - comp.clamp_term[i])
                    worst[:clamp_term] = max(worst[:clamp_term], gap / rho_l)
                    @test gap <= max(tol, 1.0e-9 * abs(comp.clamp_term[i]))                   # M2
                else
                    stats[:not_b] += 1
                    abs(comp.defect[i]) > 1.0e-9 * rho_l && (stats[:not_b_nonzero] += 1)
                    not_b_max = max(not_b_max, abs(comp.defect[i]) / rho_l)
                    not_b_formula_gap = max(not_b_formula_gap, abs(comp.defect[i] - comp.predicted[i]) / rho_l)
                end
                (a && b) || (all_ab = false)
            end
        end
        # M4a exact bookkeeping per sweep and component, and the replica against the code
        for (j, sweep) in enumerate(r.sweeps), comp in sweep.comps
            sch = comp.schedule
            mix(x) = rho_g .+ (rho_l - rho_g) .* x
            lhs = sum(sch.V .* mix(comp.new) .* comp.v_after) - sum(sch.V .* mix(comp.old) .* comp.v_before)
            v0 = D.component_values(r.velocity, sch.dofs)
            rhs = sum(sch.V .* comp.defect .* comp.v_before) + sum(sch.V .* comp.R .* v0)
            size = sum(sch.V .* mix(comp.new) .* abs.(comp.v_after)) + sum(sch.V .* mix(comp.old) .* abs.(comp.v_before))
            worst[:bookkeeping] = max(worst[:bookkeeping], abs(lhs - rhs) / size)
            @test abs(lhs - rhs) <= 1.0e-12 * size
        end
        for k in (1, 2)
            sch = r.sweeps[1].comps[k].schedule
            code_v = D.component_values(r.destination, sch.dofs)
            replica = r.sweeps[end].comps[k].v_after
            scale_v = maximum(abs, D.component_values(r.velocity, sch.dofs))
            worst[:replica] = max(worst[:replica], maximum(abs, code_v .- replica) / scale_v)
            @test maximum(abs, code_v .- replica) <= 1.0e-12 * scale_v
        end
        # M4 momentum
        snaps = any(any(sweep.snapped[ref] != sweep.pred[ref] for ref in keys(sweep.pred)) for sweep in r.sweeps)
        if all_ab && !snaps
            mesh = r.field.mesh
            before = after = predicted = 0.0
            scale = 0.0
            for k in (1, 2)
                s = r.sweeps[1].comps[k].schedule
                v0 = D.component_values(r.velocity, s.dofs)
                v1 = D.component_values(r.destination, s.dofs)
                mix(C) = rho_g .+ (rho_l - rho_g) .* D.independent_interpolation(s.interpolation, ref -> C[ref])
                before += sum(s.V .* mix(r.C0) .* v0)
                after += sum(s.V .* mix(r.Cfinal) .* v1)
                scale += sum(s.V .* mix(r.C0) .* abs.(v0)) + sum(s.V .* mix(r.Cfinal) .* abs.(v1))
                for sweep in r.sweeps
                    comp = sweep.comps[k]
                    predicted += sum(s.V .* comp.R .* v0)
                end
            end
            gap = abs((after - before) - predicted) / scale
            worst[:momentum] = max(worst[:momentum], gap)
            stats[:momentum_checked] += 1
            @test gap <= 1.0e-12
        end
    end
    println("mass identity audit: ", stats)
    println("worst (hard): ", worst, "; literal pre-snap old state after a snap: max |defect|/rho_l = ", literal_max)
    MA.excluded(
        "M5 condition (b) fails (issue 63)",
        (; unknown_sweeps = stats[:not_b], nonzero = stats[:not_b_nonzero], max_defect_over_rho_l = not_b_max, formula_gap = not_b_formula_gap),
        note = "pre-fix: defect = -(1/V_f) sum_c (w_c - varpi O_c/2) mu_c exactly",
    )
    @test stats[:fixtures] >= 20
    @test stats[:a_and_b] > 5000
    @test stats[:snap_unknowns] > 0 && stats[:snap_literal_nonzero] > 0       # the literal reading of rho^{[j-1]} is a different quantity
    @test stats[:momentum_checked] >= 5
end

@testset "eq:momentum:mass-identity: solenoidal carrier conserves the momentum (M4c)" begin
    rng = MersenneTwister(631)
    checked = 0
    worst_change = 0.0
    for (periodic, solid_kind) in (((true, true), nothing), ((false, false), nothing), ((true, true), :disc)), rep in 1:6
        density = rep % 2 == 0 ? (1000.0, 1.0) : (4.5, 1.2)
        r = run_fixture(
            rng; periodic, solid_kind, levels = 1, fraction_floor = nothing, order = rep % 3 == 0 ? (2, 1) : (1, 2), density,
            scheme = rep % 2 == 0 ? LaxWendroff() : MonotonizedCentral(), potential = 0.0,
        )
        r === nothing && continue
        rho_l, rho_g = density
        all_ab = all(comp.b_flag[i] && !comp.clamp_flag[i] for sweep in r.sweeps for comp in sweep.comps for i in 1:comp.n)
        snaps = any(any(sweep.snapped[ref] != sweep.pred[ref] for ref in keys(sweep.pred)) for sweep in r.sweeps)
        (all_ab && !snaps) || continue
        before = after = scale = 0.0
        for k in (1, 2)
            s = r.sweeps[1].comps[k].schedule
            v0 = D.component_values(r.velocity, s.dofs)
            v1 = D.component_values(r.destination, s.dofs)
            mix(C) = rho_g .+ (rho_l - rho_g) .* D.independent_interpolation(s.interpolation, ref -> C[ref])
            before += sum(s.V .* mix(r.C0) .* v0)
            after += sum(s.V .* mix(r.Cfinal) .* v1)
            scale += sum(s.V .* mix(r.C0) .* abs.(v0))
        end
        change = abs(after - before) / scale
        worst_change = max(worst_change, change)
        checked += 1
        @test change <= 1.0e-13
    end
    println("M4c: ", checked, " solenoidal fixtures; largest relative momentum change ", worst_change)
    @test checked >= 5
end
