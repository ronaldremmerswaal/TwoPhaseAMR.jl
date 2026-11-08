# Code check 01: the volume balance (5.30) for the three carriers, with a non-zero dilation and with snaps.
#
# Label:  eq:vof:volume-balance
# Source: docs/techref/chapters/05_vof_transport.tex:778-796 (eq. 5.30; "holds for all three carriers, up to rounding";
#         "the gas volume changes by the opposite amount"; boundary faces carry none)
# Claim:  V^{liq,n+1} - V^{liq,n} = dt sum_c chi_c |c| (div upsilon^open)_c + sum_j sum_c |c^o| delta C^[j]_c,
#         V = sum_c |c^o| C_c = sum_c |c| phi_c, with upsilon^open the carrier of the path.
# Method: random fixtures (graded AMR meshes with hanging faces, anisotropic cells, with and without a random wall),
#         random NON-solenoidal velocities (so the dilation is far from zero), carriers one-velocity / frozen subfaces /
#         projected liquid (loose projection tolerance 1e-3 so that its residual divergence is visible), both sweep
#         orders, default floor and a large floor 0.06 (forces snaps).  The right-hand side is evaluated by an independent
#         replica (Docs.wy_replica: independent polygon fluxes, update of the text, clamp/snap, package reconstruction);
#         the left-hand side is the package's `advect!` (liquid volume read from the `fraction` field).
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-vof-volume-balance/scripts/code_01_volume_balance.jl
#         (from the repository root)

using TwoPhaseAMR
using Test, Random, Printf
const AMR = TwoPhaseAMR
isdefined(Main, :VofAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "vof_audit.jl"))
using .VofAudit
const D = VofAudit.D

total_open_volume(field) = sum(D.open_volume_of(field, ref) for ref in D.all_cells(field.mesh))
gas_volume(field) = total_open_volume(field) - D.liquid_volume(field)

function run_case(rng; kind, carrier_kind, floor, order, aniso, tolerance = 1.0e-3)
    field, _ = VofAudit.random_field(rng; cut = kind == :cut, levels = 1, aniso, fraction_floor = floor)
    velocity = carrier_kind == :one ? VofAudit.random_velocity(rng, field; noise = 0.5) :
        VofAudit.random_phase_velocity(rng, field; noise = 0.5, opposite = 0.3)
    prepared = VofAudit.prepare_carrier(carrier_kind, field, velocity; tolerance)
    dt = prepared.dt
    isfinite(dt) || return (; skipped = "infinite dt")
    replica = try
        D.wy_replica(field, prepared.carrier, dt, order)
    catch err
        (err isa AssertionError || err isa ErrorException) || rethrow()
        return (; skipped = "replica: " * first(split(sprint(showerror, err), '\n')))
    end
    before, gas_before = D.liquid_volume(field), gas_volume(field)
    try
        advect!(field, velocity, dt, prepared.work; sweep_order = order)
    catch err
        err isa ErrorException || rethrow()
        return (; skipped = "code: " * first(split(sprint(showerror, err), '\n')))
    end
    after, gas_after = D.liquid_volume(field), gas_volume(field)
    change = after - before
    residual = abs(change - replica.dilation_total - replica.snapped)
    return (;
        change, dilation = replica.dilation_total, snapped = replica.snapped, snaps = replica.snaps,
        residual, gas_residual = abs((gas_after - gas_before) + change),
        scale = total_open_volume(field), difference = D.fraction_difference(field, replica.field),
    )
end

@testset "eq:vof:volume-balance code" begin
    rng = MersenneTwister(20261109)
    stats = Dict(:cases => 0, :skipped => 0, :snap_cases => 0)
    worst = Dict(
        :residual => 0.0, :gas_residual => 0.0, :difference => 0.0, :max_dilation => 0.0,
        :max_snapped => 0.0
    )
    by_kind = Dict{Tuple{Symbol, Symbol}, Float64}()
    skipped = String[]
    for kind in (:plain, :cut), carrier in (:one, :frozen, :projected)
        for (floor, order) in ((sqrt(eps()), (1, 2)), (0.06, (2, 1)), (sqrt(eps()), (2, 1)), (0.06, (1, 2)))
            for rep in 1:3
                res = run_case(rng; kind, carrier_kind = carrier, floor, order, aniso = rep == 3)
                stats[:cases] += 1
                if hasproperty(res, :skipped)
                    stats[:skipped] += 1; push!(skipped, "$kind/$carrier: " * res.skipped)
                    continue
                end
                worst[:residual] = max(worst[:residual], res.residual / max(1, res.scale))
                worst[:gas_residual] = max(worst[:gas_residual], res.gas_residual / max(1, res.scale))
                worst[:difference] = max(worst[:difference], res.difference)
                worst[:max_dilation] = max(worst[:max_dilation], abs(res.dilation))
                worst[:max_snapped] = max(worst[:max_snapped], abs(res.snapped))
                res.snaps > 0 && (stats[:snap_cases] += 1)
                key = (kind, carrier)
                by_kind[key] = max(get(by_kind, key, 0.0), res.residual / max(1, res.scale))
            end
        end
    end
    println("cases: ", stats)
    println("worst: ", worst)
    println("worst relative residual per (kind, carrier): ", by_kind)
    isempty(skipped) || println("skipped (code/replica refused the step at the stable dt): ", unique(skipped))
    @test stats[:cases] - stats[:skipped] >= 40
    # (5.30) holds to round-off
    @test worst[:residual] <= 1.0e-13
    @test worst[:gas_residual] <= 1.0e-13
    # the fractions after the step equal the independent sweep sequence
    @test worst[:difference] <= 1.0e-12
    # the test is not vacuous: large dilation and snapped volumes occur
    @test worst[:max_dilation] > 1.0e-2
    @test worst[:max_snapped] > 1.0e-3 && stats[:snap_cases] >= 10
end

@testset "eq:vof:volume-balance: boundary faces carry no volume (zero boundary velocity is enforced)" begin
    rng = MersenneTwister(5)
    field, _ = VofAudit.random_field(rng; cut = true, levels = 1)
    velocity = VofAudit.random_velocity(rng, field)
    work = WYAdvectionWorkspace(field)
    @test all(iszero, velocity.boundary)
    @test isfinite(stable_timestep(velocity, work))
    velocity.boundary[1] = 0.1
    @test_throws ArgumentError stable_timestep(velocity, work)
end
