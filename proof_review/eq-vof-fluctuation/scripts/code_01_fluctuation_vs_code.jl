# Code check 01: the fluctuation form (5.19) and its snap terms against the package.
#
# Label:  eq:vof:fluctuation  (with eq:vof:scalar, eq:vof:snap)
# Source: docs/techref/chapters/05_vof_transport.tex:520-552 (eq. 5.18 at 521-526, eq. 5.19 at 531-537, snap sentences 539-545)
# Claims: (i)  the code's phase scalars after a full step equal an independent evaluation of (5.19) with the snapped
#              fraction in the denominator (fluxes from independent polygon clipping, face states of (5.20) for the
#              schemes Upwind / Centred / LaxWendroff);
#         (ii) without snapping (5.19) equals the conservative form (5.18) (deviation = 0);
#         (iii) a snap to a positive phase fraction leaves the snapped phase volume with the value a^{[j-1]}
#              (deviation = (C_s - Cpre) a^{[j-1]}), for the gas the snapped volume is -delta C;
#         (iv) a snap to zero removes the amount of the cell;
#         (v)  a constant scalar is reproduced bitwise, also when fractions snap;
#         (vi) each snap changes the phase volume by at most the floor, each clamp by at most 128 eps.
# Method: random fixtures (graded AMR meshes with hanging faces, with and without a random sloped wall, anisotropic
#         cells), one-velocity, frozen-subface and projected-liquid carriers, both sweep orders, default fraction floor and a
#         large floor (0.06) that forces many snaps.  Replica = Docs.replica_fluxes + VofAudit.scalar_replica.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-vof-fluctuation/scripts/code_01_fluctuation_vs_code.jl
#         (from the repository root)

using TwoPhaseAMR
using Test, Random, Printf
const AMR = TwoPhaseAMR
isdefined(Main, :VofAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "vof_audit.jl"))
using .VofAudit
const D = VofAudit.D

const SCHEMES = Dict(:upwind => Upwind(), :centred => Centred(), :lw => LaxWendroff())

function run_case(rng; kind, carrier_kind, scheme, floor, order, aniso = false)
    field, _ = VofAudit.random_field(rng; cut = kind == :cut, levels = 1, aniso, fraction_floor = floor)
    mesh = field.mesh
    velocity = carrier_kind == :one ? VofAudit.random_velocity(rng, field; noise = 0.5) :
        VofAudit.random_phase_velocity(rng, field; noise = 0.5, opposite = 0.3)
    prepared = VofAudit.prepare_carrier(carrier_kind, field, velocity; tolerance = 1.0e-10)
    dt = prepared.dt
    isfinite(dt) || return nothing
    q = PhaseScalar(field;
        liquid = (x, y) -> 2 + sin(0.7x) * cos(0.4y) + rand(rng) * 0.0,
        gas = (x, y) -> -1.5 + cos(0.5x + 0.3y))
    q0 = VofAudit.scalar_values(q, mesh)
    replica = try
        VofAudit.scalar_replica(field, prepared.carrier, dt, order, q0; scheme)
    catch err
        err isa ErrorException || rethrow()
        return (; skipped = sprint(showerror, err))
    end
    code_error = nothing
    try
        advect!(field, velocity, dt, prepared.work; scalars = (q,), sweep_order = order,
            scheme = SCHEMES[scheme], amr_interface_scheme = SCHEMES[scheme])
    catch err
        err isa ErrorException || rethrow()
        code_error = sprint(showerror, err)
    end
    code_error === nothing || return (; skipped = "code: " * code_error)
    dl, dg = VofAudit.scalar_difference(q, replica, mesh)
    # amount-weighted difference (well conditioned where the fraction is small)
    # (events of the last sweep carry the final intrinsic values and the final phase fractions)
    amount = 0.0
    for e in replica.events
        e.sweep == 2 || continue
        qa = e.phase == 1 ? AMR._get(q.liquid, e.ref) : AMR._get(q.gas, e.ref)
        amount = max(amount, abs(e.cphase * (qa - e.after)))
    end
    return (; replica, dl, dg, amount, snaps = count(e -> abs(e.snap) > 1.0e-12, replica.events))
end

@testset "eq:vof:fluctuation code" begin
    rng = MersenneTwister(20261108)
    totals = Dict(:cases => 0, :skipped => 0, :events => 0, :nosnap => 0, :posnap => 0, :zerosnap => 0,
        :posnap_gas => 0, :clamps => 0)
    worst = Dict(:dl => 0.0, :dg => 0.0, :amount => 0.0, :dev_nosnap => 0.0, :dev_posnap => 0.0,
        :snap_max => 0.0, :clamp_max => 0.0, :removed_scaled => 0.0)
    # (kind, carrier, scheme, floor, order)
    plan = Tuple[]
    for kind in (:plain, :cut), carrier in (:one, :frozen, :projected), scheme in (:upwind, :centred, :lw)
        for (floor, order) in ((sqrt(eps()), (1, 2)), (0.06, (2, 1)))
            push!(plan, (kind, carrier, scheme, floor, order))
        end
    end
    for (index, (kind, carrier, scheme, floor, order)) in enumerate(plan)
        for rep in 1:(carrier == :projected ? 1 : 2)
            res = run_case(rng; kind, carrier_kind = carrier, scheme, floor, order, aniso = rep == 2)
            totals[:cases] += 1
            if res === nothing || hasproperty(res, :skipped)
                totals[:skipped] += 1
                res === nothing || println("  skipped: ", res.skipped)
                continue
            end
            worst[:dl] = max(worst[:dl], res.dl); worst[:dg] = max(worst[:dg], res.dg)
            worst[:amount] = max(worst[:amount], res.amount)
            for e in res.replica.events
                totals[:events] += 1
                magnitude = abs(e.snap)
                if magnitude <= 256eps()
                    totals[:nosnap] += 1
                    e.updated && (worst[:dev_nosnap] = max(worst[:dev_nosnap], abs(e.dev) / (1 + abs(e.A518))))
                    magnitude > 0 && (worst[:clamp_max] = max(worst[:clamp_max], magnitude))
                elseif e.snapped_to_zero
                    totals[:zerosnap] += 1
                    # the removed amount is the whole conservative amount A518 (dev = -A518 exactly)
                    @test abs(e.dev + e.A518) <= 1.0e-13 * (1 + abs(e.A518))
                    worst[:snap_max] = max(worst[:snap_max], magnitude)
                    worst[:removed_scaled] = max(worst[:removed_scaled], abs(e.A518) / (floor * (1 + abs(e.qprev))))
                else
                    totals[:posnap] += 1
                    e.phase == 2 && (totals[:posnap_gas] += 1)
                    @test e.updated
                    worst[:dev_posnap] = max(worst[:dev_posnap],
                        abs(e.dev - e.snap * e.qprev) / (1 + abs(e.qprev)))
                    worst[:snap_max] = max(worst[:snap_max], magnitude)
                end
            end
        end
    end
    println("fluctuation form vs code: ", totals)
    println("worst deviations:         ", worst)
    @test totals[:cases] - totals[:skipped] >= 40
    # (i) code = independent evaluation of (5.19); intrinsic values to the conditioning of 1/C_s, amounts to round-off
    @test worst[:amount] <= 1.0e-12
    # (ii) no snap: (5.19) = (5.18)
    @test worst[:dev_nosnap] <= 1.0e-12
    # (iii) snap to a positive fraction: deviation = (C_s - Cpre) a^{[j-1]}
    @test totals[:posnap] > 100 && totals[:posnap_gas] > 20
    @test worst[:dev_posnap] <= 1.0e-12
    # (iv) snap to zero: removed amount
    @test totals[:zerosnap] > 100
    # (vi) volume changed by a snap: at most the floor
    @test worst[:snap_max] <= 0.06 + 1.0e-12
    @test worst[:clamp_max] <= 128eps() * 1.01
end

@testset "eq:vof:fluctuation constants are bitwise preserved, also with snaps" begin
    rng = MersenneTwister(7)
    for kind in (:plain, :cut), carrier_kind in (:one, :frozen, :projected), floor in (sqrt(eps()), 0.08)
        field, _ = VofAudit.random_field(rng; cut = kind == :cut, levels = 1, fraction_floor = floor)
        mesh = field.mesh
        velocity = carrier_kind == :one ? VofAudit.random_velocity(rng, field; noise = 0.5) :
            VofAudit.random_phase_velocity(rng, field; noise = 0.5, opposite = 0.3)
        prepared = VofAudit.prepare_carrier(carrier_kind, field, velocity; tolerance = 1.0e-10)
        q = PhaseScalar(field; liquid = 3.0, gas = 7.0)
        advect!(field, velocity, prepared.dt, prepared.work; scalars = (q,), scheme = LaxWendroff())
        ok = all(AMR._get(q.liquid, ref) == 3.0 && AMR._get(q.gas, ref) == 7.0 for ref in D.all_cells(mesh))
        @test ok
    end
end
