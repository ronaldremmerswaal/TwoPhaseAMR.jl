# Code check 01: the scalar balance (5.32), its snap terms and the gas reset, against the package.
#
# Label:  eq:vof:scalar-balance
# Source: docs/techref/chapters/05_vof_transport.tex:816-837 (eq. 5.31 and the sentence "A snap to a positive fraction adds
#         |c^o| delta C^[j] a^[j-1], a snap to zero removes the cell's amount, and the projected carrier adds the reset of
#         the gas amounts")
# Claims: (i)   no fraction clamped or snapped:  A^{n+1} - A^n = dt sum_c chi^phi_c a^{phi,n}_c |c| (div upsilon^open)_c ,
#               A = sum_c |c^o| C^phi_c a^phi_c, for the liquid and the gas scalar and ANY face state (all eight schemes);
#         (ii)  with snaps, the change is the dilation plus  sum |c^o| (C_s - Cpre) a^[j-1]  (snap to a positive fraction)
#               and  - A518  (snap to zero); for the gas the snapped volume is -delta C (the sign of the printed
#               formula holds for the liquid only);
#         (iii) projected carrier, gas scalar: no dilation term; the amount after the step differs from the sum of the
#               auxiliary amounts by the resets sum_j sum_c |c^o| (1 - C^[j] - G*^[j]) a^[j].
# Method: random fixtures as in eq-vof-fluctuation (graded AMR meshes, walls, anisotropic cells), random non-solenoidal
#         velocities (so the dilation is non-zero), the three carriers, both orders.  Left-hand side: the package
#         (`integrate(q, phase)` before / after `advect!`); right-hand side: carrier volumes from the definitions and the
#         independent scalar replica of VofAudit (for the snap and reset terms).
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-vof-scalar-balance/scripts/code_01_scalar_balance.jl
#         (from the repository root)

using TwoPhaseAMR
using Test, Random, Printf
const AMR = TwoPhaseAMR
isdefined(Main, :VofAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "vof_audit.jl"))
using .VofAudit
const D = VofAudit.D

const ALL_SCHEMES = Dict(
    :upwind => Upwind(), :centred => Centred(), :lw => LaxWendroff(), :fromm => Fromm(),
    :minmod => Minmod(), :mc => MonotonizedCentral(), :vanleer => VanLeer(), :superbee => Superbee(),
)

"""Carrier volume of every face (both phases of a frozen carrier summed), from the definitions of Chapter 5."""
function carrier_volumes(field, carrier, dt)
    mesh = field.mesh
    volumes = zeros(length(mesh.faces))
    for (index, face) in pairs(mesh.faces)
        if carrier.kind == :frozen
            for phase in 1:2
                a = carrier.apertures[phase][index]
                iszero(a) && continue
                volumes[index] += dt * carrier.velocity[phase].interior[index] * face.area * a
            end
        else
            volumes[index] = dt * carrier.velocity.interior[index] * face.area * D.aperture_of(field, index)
        end
    end
    return volumes
end

function run_case(rng; kind, carrier_kind, scheme, floor, order, with_replica, tolerance = 1.0e-3)
    field, _ = VofAudit.random_field(rng; cut = kind == :cut, levels = 1, fraction_floor = floor)
    mesh = field.mesh
    velocity = carrier_kind == :one ? VofAudit.random_velocity(rng, field; noise = 0.5) :
        VofAudit.random_phase_velocity(rng, field; noise = 0.5, opposite = 0.3)
    prepared = VofAudit.prepare_carrier(carrier_kind, field, velocity; tolerance)
    dt = prepared.dt
    isfinite(dt) || return (; skipped = "infinite dt")
    q = PhaseScalar(field; liquid = (x, y) -> 2 + sin(0.7x) * cos(0.4y), gas = (x, y) -> -1.5 + cos(0.5x + 0.3y))
    q0 = VofAudit.scalar_values(q, mesh)
    reconstruct!(field)
    cells = D.all_cells(mesh)
    chi = Dict(ref => D.cond(field, ref) > 0.5 ? 1.0 : 0.0 for ref in cells)
    aux = carrier_kind == :projected
    snaps = 0
    replica = nothing
    if with_replica
        replica = try
            VofAudit.scalar_replica(field, prepared.carrier, dt, order, q0; scheme)
        catch err
            err isa ErrorException || rethrow()
            return (; skipped = "replica: " * first(split(sprint(showerror, err), '\n')))
        end
        snaps = count(e -> abs(e.snap) > 1.0e-12, replica.events)
    else
        r = try
            D.wy_replica(field, prepared.carrier, dt, order)
        catch err
            (err isa AssertionError || err isa ErrorException) || rethrow()
            return (; skipped = "replica: " * first(split(sprint(showerror, err), '\n')))
        end
        snaps = r.snaps
    end
    before_l, before_g = integrate(q, :liquid), integrate(q, :gas)
    try
        advect!(
            field, velocity, dt, prepared.work; scalars = (q,), sweep_order = order,
            scheme = ALL_SCHEMES[scheme],
            amr_interface_scheme = scheme in (:upwind, :centred, :lw) ? ALL_SCHEMES[scheme] :
                default_amr_interface_scheme(ALL_SCHEMES[scheme])
        )
    catch err
        err isa ErrorException || rethrow()
        return (; skipped = "code: " * first(split(sprint(showerror, err), '\n')))
    end
    after_l, after_g = integrate(q, :liquid), integrate(q, :gas)
    # dilation of (5.32): dt sum_c chi^phi a^n |c| (div upsilon^open)_c = sum_c chi^phi a^n sum_f o V_f
    volumes = carrier_volumes(field, prepared.carrier, dt)
    incidence = D.cell_faces(mesh)
    div = Dict(
        ref => sum((o * volumes[index] for (index, o) in get(incidence, ref, Tuple{Int, Int}[])); init = 0.0)
            for ref in cells
    )
    dil_l = sum(chi[ref] * q0.liquid[ref] * div[ref] for ref in cells)
    dil_g = aux ? 0.0 : sum((1 - chi[ref]) * q0.gas[ref] * div[ref] for ref in cells)
    scale = sum(D.open_volume_of(field, ref) * (abs(q0.liquid[ref]) + abs(q0.gas[ref])) for ref in cells)
    out = (; dl = after_l - before_l, dg = after_g - before_g, dil_l, dil_g, scale, snaps, aux)
    if replica !== nothing
        dev_l = sum(e.open * e.dev for e in replica.events if e.phase == 1)
        dev_g = sum(e.open * e.dev for e in replica.events if e.phase == 2)
        # printed formula read literally with the LIQUID snap change delta C for both phases
        literal_g = sum(
            e.open * (-e.snap) * e.qprev for e in replica.events
                if e.phase == 2 && !aux && !e.snapped_to_zero && abs(e.snap) > 1.0e-12; init = 0.0
        )
        correct_g = sum(
            e.open * e.snap * e.qprev for e in replica.events
                if e.phase == 2 && !aux && !e.snapped_to_zero && abs(e.snap) > 1.0e-12; init = 0.0
        )
        # resets (5.29)-type terms of the auxiliary gas carrier
        resets = 0.0
        if aux
            liquid_fraction = Dict((e.ref, e.sweep) => e.cphase for e in replica.events if e.phase == 1)
            for e in replica.events
                e.phase == 2 || continue
                resets += e.open * (1 - liquid_fraction[(e.ref, e.sweep)] - e.cphase) * e.after
            end
        end
        out = merge(out, (; dev_l, dev_g, resets, literal_g, correct_g))
    end
    return out
end

@testset "eq:vof:scalar-balance code" begin
    rng = MersenneTwister(2026110801)
    @testset "(i) no snaps: all eight face-state schemes, three carriers, liquid and gas" begin
        worst = Dict(:liquid => 0.0, :gas => 0.0, :dilation => 0.0)
        cases = skipped = 0
        reasons = String[]
        for kind in (:plain, :cut), carrier in (:one, :frozen), scheme in keys(ALL_SCHEMES)
            res = run_case(
                rng; kind, carrier_kind = carrier, scheme, floor = sqrt(eps()),
                order = rand(rng) < 0.5 ? (1, 2) : (2, 1), with_replica = false
            )
            cases += 1
            if hasproperty(res, :skipped)
                skipped += 1; push!(reasons, res.skipped); continue
            end
            res.snaps == 0 || (skipped += 1; push!(reasons, "snaps present"); continue)
            worst[:liquid] = max(worst[:liquid], abs(res.dl - res.dil_l) / res.scale)
            worst[:gas] = max(worst[:gas], abs(res.dg - res.dil_g) / res.scale)
            worst[:dilation] = max(worst[:dilation], abs(res.dil_l) / res.scale, abs(res.dil_g) / res.scale)
        end
        println("(i) cases ", cases, ", skipped ", skipped, " ", unique(reasons), "; worst relative residual ", worst)
        @test cases - skipped >= 24
        @test worst[:liquid] <= 1.0e-13 && worst[:gas] <= 1.0e-13
        @test worst[:dilation] > 1.0e-4                            # the dilation is not negligible
    end
    @testset "(i') projected carrier, liquid scalar (gas has resets, see (iii))" begin
        worst = 0.0; dil = 0.0; cases = 0
        for kind in (:plain, :cut), scheme in (:upwind, :lw, :mc, :superbee)
            res = run_case(
                rng; kind, carrier_kind = :projected, scheme, floor = sqrt(eps()),
                order = (1, 2), with_replica = false, tolerance = 1.0e-3
            )
            hasproperty(res, :skipped) && continue
            res.snaps == 0 || continue
            cases += 1
            worst = max(worst, abs(res.dl - res.dil_l) / res.scale)
            dil = max(dil, abs(res.dil_l) / res.scale)
        end
        println("(i') projected carrier, liquid scalar: ", cases, " cases, residual ", worst, ", dilation ", dil)
        @test cases >= 4
        @test worst <= 1.0e-13
    end
    @testset "(ii) snaps: positive-snap and zero-snap terms; sign for the gas" begin
        worst = Dict(:liquid => 0.0, :gas => 0.0); literal_gap = 0.0; correct_gap = 0.0
        cases = 0; snapped_cases = 0
        for kind in (:plain, :cut), carrier in (:one, :frozen), scheme in (:upwind, :centred, :lw), rep in 1:2
            res = run_case(
                rng; kind, carrier_kind = carrier, scheme, floor = 0.06,
                order = rep == 1 ? (1, 2) : (2, 1), with_replica = true
            )
            hasproperty(res, :skipped) && continue
            cases += 1
            res.snaps > 0 && (snapped_cases += 1)
            worst[:liquid] = max(worst[:liquid], abs(res.dl - res.dil_l - res.dev_l) / res.scale)
            worst[:gas] = max(worst[:gas], abs(res.dg - res.dil_g - res.dev_g) / res.scale)
            # gas, positive snaps only: the amount gained is correct_g, the literal reading gives literal_g
            literal_gap = max(literal_gap, abs(res.literal_g - res.correct_g) / res.scale)
            correct_gap = max(correct_gap, abs(res.correct_g) / res.scale)
        end
        println(
            "(ii) cases ", cases, " (with snaps ", snapped_cases, "); residual with the snap terms ", worst,
            "; literal-vs-correct gas snap term ", literal_gap, " (term itself ", correct_gap, ")"
        )
        @test cases >= 20 && snapped_cases >= 15
        @test worst[:liquid] <= 1.0e-13 && worst[:gas] <= 1.0e-13
        @test literal_gap > 1.0e-6                                  # the printed sign is wrong for the gas
    end
    @testset "(iii) projected carrier, gas scalar: resets" begin
        worst_reset = 0.0; worst_noreset = 0.0; cases = 0; reset_size = 0.0
        for kind in (:plain, :cut), scheme in (:upwind, :centred, :lw), order in ((1, 2), (2, 1))
            res = run_case(
                rng; kind, carrier_kind = :projected, scheme, floor = sqrt(eps()), order,
                with_replica = true, tolerance = 1.0e-3
            )
            hasproperty(res, :skipped) && continue
            cases += 1
            worst_reset = max(worst_reset, abs(res.dg - res.resets - res.dev_g) / res.scale)
            worst_noreset = max(worst_noreset, abs(res.dg - res.dev_g) / res.scale)
            reset_size = max(reset_size, abs(res.resets) / res.scale)
        end
        println(
            "(iii) cases ", cases, "; residual with the resets ", worst_reset, ", without ", worst_noreset,
            ", reset size ", reset_size
        )
        @test cases >= 6
        @test worst_reset <= 1.0e-13
    end
end
