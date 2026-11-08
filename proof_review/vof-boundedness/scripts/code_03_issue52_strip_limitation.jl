# Code check 03: the "Limitation at level transitions" of Section 5.4 / GitHub issue #52 -- a coarse cut donor whose half-strip
# cannot hold the carrier volume of a segment -- and what it implies for the hypotheses of the boundedness argument.
#
# Label:  none (paragraph "Limitation at level transitions", Section 5.4); issue #52 (open at the time of the audit)
# Source: docs/techref/chapters/05_vof_transport.tex:232-258; docs/techref/figures/vof.jl `strip_limitation_check`; issue #52 reproducer
# Claims: (i)   the fixture of the text: a coarse cut cell (open fraction 0.113) with a segment of a_f = 0.90 that must sweep 0.0267
#               of the coarse volume at dt = dtwy = 0.0594 while its half-strip has only 0.0253 open: `advect!` stops with the
#               error at the step `stable_timestep` returns; the failing donor is mixed;
#         (ii)  the same geometry with a pure liquid or pure gas interface runs at dtwy (pure donors need no rectangle) in the
#               one-velocity path;
#         (iii) the critical step is dt_crit = dtwy * (strip open area)/(target), here about 0.95 dtwy: the failure is a property of
#               the segment (a per-segment limit), not of the cell-wise bounds (5.15), which are satisfied;
#         (iv)  for UNCUT donors and cut donors with single-face sides no such failure occurs at dtwy (many random cases, see
#               code_01 of this folder); for the frozen carrier the strip is the strip over the phase subface, and the error can occur
#               for single-face cut donors as well (counted below).
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/vof-boundedness/scripts/code_03_issue52_strip_limitation.jl
#         (from the repository root)

using TwoPhaseAMR
using Test, Random, Printf
const AMR = TwoPhaseAMR
isdefined(Main, :VofAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "vof_audit.jl"))
using .VofAudit
const D = VofAudit.D

function issue_mesh()
    return BlockTreeMesh(;
        base_mesh = (12, 12), block_size = (4, 4), extent = (12.0, 12.0), periodic = (false, false),
        max_level = 1, refine = (k, o, e) -> k == BlockKey(0, 1, 1)
    )
end

function issue_fields(; interface = (x, y) -> y - 2.65 - 0.2x)
    mesh = issue_mesh()
    geometry = GeometryField(mesh, (x, y) -> y - 2.2625 - 0.25x; lipschitz = 1.04)
    field = InterfaceField(mesh, interface; geometry, lipschitz = 1.02)
    velocity = FaceVelocity(mesh, (x, y, t, axis) -> axis == 2 ? 1.0 : 0.0)
    enforce_no_permeation!(velocity)
    return field, velocity
end

function try_advect(field, velocity, dt; kind = :one)
    trial = copy(field)
    two = kind == :one ? nothing : (velocity isa Tuple ? velocity : (velocity, copy(velocity)))
    work = kind == :one ? WYAdvectionWorkspace(trial) :
        WYAdvectionWorkspace(trial, two; interface_transport = FrozenInterfaceFluxes())
    try
        advect!(trial, kind == :one ? velocity : two, dt, work; check_timestep = false)
        return :ok
    catch err
        err isa ErrorException || rethrow()
        return occursin("exceeds", sprint(showerror, err)) ? :strip : :other
    end
end

@testset "issue 52 / strip limitation" begin
    @testset "(i) the reproducer of the issue" begin
        field, velocity = issue_fields()
        work = WYAdvectionWorkspace(field)
        dtwy = stable_timestep(velocity, work)
        @test isapprox(dtwy, 0.0594; atol = 5.0e-5)
        err = nothing
        trial = copy(field)
        try
            advect!(trial, velocity, dtwy, WYAdvectionWorkspace(trial))
        catch e
            err = e
        end
        @test err isa ErrorException && occursin("exceeds the donor's open subcell volume", sprint(showerror, err))
        # numbers of the text (docs fixture, same geometry with the refinement of vof_amr_mesh)
        strip = D.strip_limitation_check()
        println(
            "(i) text fixture: dt = ", strip.dt, ", half-strip open area ", strip.strip, ", segment target ", strip.target,
            ", aperture ", strip.aperture, ", donor open fraction ", strip.open, ", donor liquid fraction ", strip.fraction
        )
        @test strip.target > strip.strip
        @test 0 < strip.fraction < 1
        @test isapprox(strip.strip, 0.0253; atol = 5.0e-5) && isapprox(strip.target, 0.0267; atol = 5.0e-5)
        @test isapprox(strip.aperture, 0.9; atol = 5.0e-3) && isapprox(strip.open, 0.113; atol = 5.0e-4)
        # (iii) critical step by bisection on the success of advect!
        lo, hi = 0.0, 1.0
        for _ in 1:40
            mid = (lo + hi) / 2
            try_advect(field, velocity, mid * dtwy) == :ok ? (lo = mid) : (hi = mid)
        end
        println("(iii) critical step dt_crit / dtwy = ", lo, "   (strip / target = ", strip.strip / strip.target, ")")
        @test isapprox(lo, strip.strip / strip.target; atol = 1.0e-3)
        @test lo < 1 && lo > 0.9
    end
    @testset "(ii) pure donors need no rectangle (one-velocity path)" begin
        for (name, phi) in ((:liquid, (x, y) -> -1.0), (:gas, (x, y) -> 1.0))
            field, velocity = issue_fields(; interface = phi)
            dtwy = stable_timestep(velocity, WYAdvectionWorkspace(field))
            outcome = try_advect(field, velocity, dtwy)
            frozen = try_advect(field, velocity, dtwy; kind = :frozen)
            println("(ii) pure ", name, ": one-velocity ", outcome, ", frozen carrier (same velocity for both phases) ", frozen)
            @test outcome == :ok
        end
    end
    @testset "(iv) frequency of the failure at the stable step, by donor class" begin
        rng = MersenneTwister(5252)
        counts = Dict{Tuple{Symbol, Int, Symbol}, Vector{Int}}()                # (carrier, levels, kind) => [runs, failures]
        for carrier in (:one, :frozen), levels in (0, 1, 2), rep in 1:10
            field, _ = VofAudit.random_field(rng; cut = true, levels, aniso = rep % 2 == 0)
            velocity = carrier == :one ? VofAudit.random_velocity(rng, field; noise = 0.6) :
                VofAudit.random_phase_velocity(rng, field; noise = 0.6, opposite = 0.4)
            prepared = VofAudit.prepare_carrier(carrier, field, velocity)
            isfinite(prepared.dt) || continue
            outcome = try_advect(field, velocity, prepared.dt; kind = carrier)
            c = get!(counts, (carrier, levels, :cut), [0, 0])
            c[1] += 1; outcome == :strip && (c[2] += 1)
        end
        for (k, v) in sort(collect(counts); by = x -> string(x[1]))
            println("(iv) ", k, ": ", v[2], " of ", v[1], " runs fail with the strip error at the stable step")
        end
        # single level (levels = 0): every donor side is a single face; the one-velocity path cannot fail there
        @test counts[(:one, 0, :cut)][2] == 0
    end
end
