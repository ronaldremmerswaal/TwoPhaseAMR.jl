# Sharpness experiment 01: how tight are the CFL hypotheses of the boundedness argument (nu_in, nu_out <= nu^dagger <= 1/2)?
#
# Label:  none (paragraph "Boundedness." of Section 5.5; eq:vof:cfl, eq:vof:dtwy)
# Source: docs/techref/chapters/05_vof_transport.tex:296-322, 330-345; thesis Lemma 3.3
# Experiments on a uniform periodic mesh (one-velocity carrier, no solid), run with `check_timestep = false` beyond the stable step:
#   (A) upper bound: a cell with C = 1/2 (chi = 0) between two pure-liquid cells with converging face velocities receives the carrier
#       volume nu_in; the fraction after the sweep is C = 1/2 + nu_in exactly, so nu_in = 1/2 reaches 1 and any nu_in > 1/2 leaves [0,1]
#       (the package raises its out-of-bounds error): nu_in <= 1/2 is SHARP for the upper bound;
#   (B) lower bound: a cell with C = 1/2 and a diagonal interface donating to both sides with depth s each: C after the sweep is
#       0.5 - |L cap R_left| - |L cap R_right|; it is >= 0 iff the two rectangles do not overlap in the liquid, s <= 1/2 (nu_in of the receivers = s):
#       (B1) s = 0.4: nu_out = 0.8 > 1/2 violates the printed outgoing bound, yet C stays in [0,1]  (nu_out <= 1/2 is not needed here);
#       (B2) s = 0.55: C < 0, the package raises the error;
#   (C) both experiments agree with the independent polygon replica.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/vof-boundedness/scripts/sharp_01_cfl_thresholds.jl
#         (from the repository root)

using TwoPhaseAMR
using Test, Random, Printf
const AMR = TwoPhaseAMR
isdefined(Main, :VofAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "vof_audit.jl"))
using .VofAudit
const D = VofAudit.D

function row_fixture(; cell = (6, 6))
    mesh = D.vof_uniform_mesh()
    field = InterfaceField(mesh, (x, y) -> -1.0; lipschitz = 1)                  # everything liquid ...
    center = CellRef(1, cell...)
    AMR._set!(field.fraction, 0.5, center)                                       # ... except the centre cell: C = 1/2
    reconstruct!(field)
    return mesh, field, center
end

function x_faces_of(mesh, cell)
    right = findfirst(f -> f.axis == 1 && f.minus == cell, mesh.faces)
    left = findfirst(f -> f.axis == 1 && f.plus == cell, mesh.faces)
    return left, right
end

function step_outcome(field, velocity, dt)
    trial = copy(field)
    work = WYAdvectionWorkspace(trial)
    try
        advect!(trial, velocity, dt, work; check_timestep = false)
        return (:ok, trial)
    catch err
        err isa ErrorException || rethrow()
        return (occursin("out of bounds", sprint(showerror, err)) ? :out_of_bounds : :other, nothing)
    end
end

@testset "CFL thresholds of the boundedness argument" begin
    @testset "(A) upper bound: C = 1/2 + nu_in, sharp at nu_in = 1/2" begin
        mesh, field, center = row_fixture()
        left, right = x_faces_of(mesh, center)
        velocity = FaceVelocity(mesh)
        velocity.interior[left] = 1.0          # liquid flows into the centre cell from the left ...
        velocity.interior[right] = -1.0        # ... and from the right
        dtwy = stable_timestep(velocity, WYAdvectionWorkspace(field))              # nu_in = 2 dt  ->  dt = 1/4
        @test isapprox(dtwy, 0.25; atol = 1.0e-15)
        rows = Tuple[]
        for factor in (0.5, 0.9, 1.0, 1.0 + 1.0e-6, 1.1, 1.5)
            dt = factor * dtwy
            outcome, after = step_outcome(field, velocity, dt)
            c = outcome == :ok ? D.cond(after, center) : NaN
            push!(rows, (factor, 2dt, outcome, c))
            println("(A) nu_in = ", 2dt, ": ", outcome, ", C after = ", c, "   (1/2 + nu_in = ", 0.5 + 2dt, ")")
            if factor <= 1.0
                @test outcome == :ok
                @test isapprox(c, 0.5 + 2dt; atol = 1.0e-14)
            else
                @test outcome == :out_of_bounds
            end
        end
    end
    @testset "(B) lower bound: overlap of the two donating rectangles" begin
        mesh = D.vof_uniform_mesh()
        center_xy = (5.5, 5.5)
        # diagonal interface through the centre of the cell [5,6]x[5,6]: liquid = {x + y <= 11}, C = 1/2 exactly
        field = InterfaceField(mesh, (x, y) -> x + y - 11.0; lipschitz = sqrt(2))
        center = CellRef(1, 6, 6)
        reconstruct!(field)
        @test isapprox(D.cond(field, center), 0.5; atol = 1.0e-12)
        left, right = x_faces_of(mesh, center)
        velocity = FaceVelocity(mesh)
        velocity.interior[left] = -1.0         # the centre cell donates to the left ...
        velocity.interior[right] = 1.0         # ... and to the right
        for (s, expect_ok) in ((0.4, true), (0.5, true), (0.55, false), (0.7, false))
            dt = s                                          # depth s on each side (|u| = 1, unit cells)
            nu_out = 2s
            outcome, after = step_outcome(field, velocity, dt)
            # independent replica of the donor
            volumes, liquids, _, _ = D.replica_fluxes(copy(field), D.one_velocity_carrier(velocity), dt, 1)
            cpre = 0.5 + (0.0 - sum(o * liquids[i] for (i, o) in D.cell_faces(mesh)[center] if mesh.faces[i].axis == 1)) / 1.0
            println("(B) s = ", s, " (nu_out = ", nu_out, ", nu_in(receivers) = ", s, "): ", outcome, ", C after (replica, pre-clamp) = ", cpre)
            @test (outcome == :ok) == expect_ok
            @test (cpre >= -1.0e-12) == expect_ok
        end
    end
end
