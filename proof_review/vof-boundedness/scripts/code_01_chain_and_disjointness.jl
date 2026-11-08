# Code check 01: the chain of inequalities of the boundedness argument of Section 5.5, class by class.
#
# Label:  none (suggested prop:vof:boundedness) -- paragraph "Boundedness." of Section 5.5 (sec:vof:floor)
# Source: docs/techref/chapters/05_vof_transport.tex:296-322 (the argument), 286-295 ("The run-time check is the only guarantee
#         in general") and the "Limitation at level transitions" paragraph 232-258; thesis Lemma 3.3 (int_advection_split.tex).
# Claims (verified here, class by class; donor class = (uncut | cut, hanging side or not) of the swept axis):
#   (a) the rectangles through which a cell donates are pairwise disjoint in UNCUT donors (any level structure) and in CUT
#       donors whose two sides normal to the sweep are single faces, for the one-velocity and projected-liquid carriers
#       (full-width rectangles); the frozen carrier is covered in uncut donors only (depth <= nu_ext <= 1/2);
#   (b) under pairwise disjointness the outflow of the conservatively moved phase (liquid if chi = 0, gas if chi = 1) through
#       the rectangles is at most the volume of that phase in the cell;
#   (c) its inflow over the sweeps so far is at most the incoming carrier volume, hence
#           chi = 0:  0 <= C <= 1/2 + nu_in   (+ eps after a snap),      chi = 1:  1/2 - nu_in <= C <= 1;
#   (d) with nu_in <= 1/2 (the CFL bound) this gives 0 <= C <= 1 before clamping, up to rounding and the floor eps added by a
#       snap after the first sweep;
#   (e) in the classes NOT covered (cut donors with a hanging side; the frozen carrier in cut donors, whose subface rectangles
#       are not full width) the proof does not apply -- this script reports counts, it does not assert disjointness there
#       (code_04 searches for violations there).
# Method: random fixtures on graded meshes (0, 1, 2 levels; isotropic and anisotropic cells), periodic without solid and walled
#         with a random wall; random NON-solenoidal velocities (face-wise noise), time step = stable_timestep (the CFL bound is
#         attained somewhere); independent replica sweeps (polygon clipping, VofAudit.sweep_records) record pre-clamp fractions,
#         the outflow/inflow of the conservative phase, rectangle overlaps and depths per donor and sweep; the package's
#         `advect!` is run at the same time step and its outcome is compared with the replica's.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/vof-boundedness/scripts/code_01_chain_and_disjointness.jl
#         (from the repository root)

using TwoPhaseAMR
using Test, Random, Printf
const AMR = TwoPhaseAMR
isdefined(Main, :VofAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "vof_audit.jl"))
using .VofAudit
const D = VofAudit.D

const TOL = 1.0e-12

function classify_error(message)
    occursin("out of bounds", message) && return :bounds
    (
        occursin("exceeds the donor's open subcell volume", message) || occursin("frozen phase-subface flux exceeds", message) ||
            occursin("swept volume exceeds the open strip", message)
    ) && return :strip
    return :other
end

"""Run one case; returns (records or nothing, replica outcome, code outcome)."""
function run_case(rng; kind, carrier_kind, levels, aniso, order)
    field, _ = VofAudit.random_field(rng; cut = kind == :cut, levels, aniso)
    velocity = carrier_kind == :one ? VofAudit.random_velocity(rng, field; noise = 1.0) :
        VofAudit.random_phase_velocity(rng, field; noise = 1.0, opposite = 0.4)
    prepared = VofAudit.prepare_carrier(carrier_kind, field, velocity; tolerance = 1.0e-8)
    dt = prepared.dt
    isfinite(dt) || return nothing
    records = nothing; replica_outcome = :ok
    try
        records = VofAudit.sweep_records(field, prepared.carrier, dt, order)
    catch err
        err isa ErrorException || rethrow()
        replica_outcome = classify_error(sprint(showerror, err))
    end
    code_outcome = :ok
    trial = copy(field)
    trial_prepared = VofAudit.prepare_carrier(carrier_kind, trial, velocity; tolerance = 1.0e-8)
    try
        advect!(trial, velocity, dt, trial_prepared.work; sweep_order = order)
    catch err
        err isa ErrorException || rethrow()
        code_outcome = classify_error(sprint(showerror, err))
    end
    return (; records, replica_outcome, code_outcome, kind, carrier_kind, levels, dt)
end

mutable struct ClassStats
    records::Int            # (cell, sweep) with at least one donating rectangle
    multi::Int              # with >= 2 rectangles
    overlap::Int            # rectangles overlap geometrically
    overlap_phase::Int      # overlap contains open conservative phase
    i1_violation::Float64   # max (out_cons - v_cons) over records with disjoint conservative phase
    i1_violation_all::Float64   # same over all records
    cmin::Float64
    cmax::Float64
    chain_violation::Float64    # max violation of the chain (c)
    max_depth::Float64
end
ClassStats() = ClassStats(0, 0, 0, 0, 0.0, 0.0, Inf, -Inf, 0.0, 0.0)

function accumulate!(stats::ClassStats, recs, floor)
    for r in recs
        r.cls == :covered && continue
        stats.cmin = min(stats.cmin, r.cpre); stats.cmax = max(stats.cmax, r.cpre)
        # chain (c): chi = 0: C <= 1/2 + nu_in + floor ; chi = 1: C >= 1/2 - nu_in - floor ; always 0 <= C <= 1 (+ floor)
        chain = r.chi == 0 ? r.cpre - (0.5 + r.nu_in + floor) : (0.5 - r.nu_in - floor) - r.cpre
        stats.chain_violation = max(stats.chain_violation, chain)
        r.ndr >= 1 || continue
        stats.records += 1
        r.ndr >= 2 && (stats.multi += 1)
        r.overlap > TOL && (stats.overlap += 1)
        r.overlap_phase > TOL && (stats.overlap_phase += 1)
        excess = r.out_cons - r.v_cons
        stats.i1_violation_all = max(stats.i1_violation_all, excess)
        r.overlap_phase <= TOL && (stats.i1_violation = max(stats.i1_violation, excess))
        stats.max_depth = max(stats.max_depth, r.depth)
    end
    return
end

@testset "vof boundedness: chain of inequalities per donor class" begin
    rng = MersenneTwister(20261113)
    stats = Dict{Tuple{Symbol, Symbol, Bool}, ClassStats}()      # (carrier, cls, hanging)
    outcomes = Dict{Tuple{Symbol, Symbol, Symbol, Symbol}, Int}()
    mismatches = String[]
    runs = 0
    plan = Tuple[]
    for carrier in (:one, :frozen, :projected)
        for levels in (1, 2), rep in 1:4
            push!(plan, (:plain, carrier, levels, rep % 2 == 0))
        end
        for levels in (0, 1, 2), rep in 1:6
            push!(plan, (:cut, carrier, levels, rep % 2 == 0))
        end
    end
    for (kind, carrier, levels, aniso) in plan
        order = rand(rng) < 0.5 ? (1, 2) : (2, 1)
        res = run_case(rng; kind, carrier_kind = carrier, levels, aniso, order)
        res === nothing && continue
        runs += 1
        key = (kind, carrier, res.replica_outcome, res.code_outcome)
        outcomes[key] = get(outcomes, key, 0) + 1
        if res.replica_outcome != res.code_outcome
            push!(mismatches, "$kind/$carrier/levels=$levels: replica $(res.replica_outcome), code $(res.code_outcome)")
        end
        res.records === nothing && continue
        floor = sqrt(eps())
        for recs in res.records.records, r in recs
            key = (carrier, r.cls, r.hanging)
            haskey(stats, key) || (stats[key] = ClassStats())
        end
        for recs in res.records.records
            for key in unique((carrier, r.cls, r.hanging) for r in recs)
                accumulate!(stats[key], [r for r in recs if (carrier, r.cls, r.hanging) == key], floor)
            end
        end
    end
    println("runs: ", runs)
    println("outcomes (kind, carrier, replica, code) => count:")
    for (k, v) in sort(collect(outcomes); by = x -> string(x[1]))
        println("   ", k, " => ", v)
    end
    isempty(mismatches) || println("replica/code outcome mismatches (pure donors are not checked by the replica): ", mismatches)
    println("class statistics (carrier, donor class, hanging side):")
    for (k, s) in sort(collect(stats); by = x -> string(x[1]))
        @printf(
            "   %-8s %-7s hanging=%-5s records=%5d multi=%4d overlap=%3d overlap_phase=%3d  I1viol(disjoint)=%+.1e I1viol(all)=%+.1e  C in [%+.2e, %.6f]  chain=%+.1e  max depth=%.3f\n",
            k[1], k[2], k[3], s.records, s.multi, s.overlap, s.overlap_phase, s.i1_violation, s.i1_violation_all, s.cmin, s.cmax,
            s.chain_violation, s.max_depth
        )
    end
    # (a) disjointness where claimed (one-velocity carrier)
    for carrier in (:one, :projected), hanging in (false, true)
        s = get(stats, (carrier, :uncut, hanging), nothing)
        s === nothing && continue
        @test s.overlap == 0
    end
    for carrier in (:one, :projected)
        s = get(stats, (carrier, :cut, false), nothing)
        @test s !== nothing && s.records > 30 && s.overlap == 0
    end
    s = get(stats, (:frozen, :uncut, false), nothing)
    @test s !== nothing && s.records > 1000 && s.overlap == 0           # frozen carrier in uncut donors: depth <= nu_ext <= 1/2
    # (b) outflow <= phase volume under disjointness; (c)/(d) the chain, in every class where (a) is proven
    for (k, s) in stats
        (k[1] in (:one, :projected) && (k[2] == :uncut || (k[2] == :cut && !k[3]))) || (k[1] == :frozen && k[2] == :uncut) || continue
        @test s.i1_violation <= 1.0e-12
        @test s.chain_violation <= 1.0e-12
        @test s.cmin >= -1.0e-12 && s.cmax <= 1 + sqrt(eps()) + 1.0e-12
    end
    # the code never reports a bound violation in these runs, and agrees with the replica except for pure-donor strips
    @test all(k -> k[4] != :bounds, keys(outcomes))
    @test all(k -> k[4] != :other, keys(outcomes))
end
