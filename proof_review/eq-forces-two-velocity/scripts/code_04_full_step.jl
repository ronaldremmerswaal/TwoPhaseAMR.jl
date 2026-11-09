# Code check 04: one full inviscid step (FlowSimulation, simulate!) from the hydrostatic rest state with the code AFTER the fix of #76
# (commit 497a401, PR #81): the forces enter the predictor through the projection's pressure map.
#
# Label:  eq:forces:two-velocity  (issue #76)
# Source: docs/techref/chapters/08_forces.tex (two-velocity section); src/two_phase/two_velocity/stepping.jl (_phase_force_substep!)
# Cases: planar heavy liquid (interface y = 4.6, g = (0,-1), densities (1000, 1), dt = 0.01, tolerance 1e-10), uniform / composite mesh x
# {no solid, solid in the gas, submerged solid (in the liquid)}; velocity models one velocity, CMDGFM, mimetic (default).
#   F1  one velocity: <= 1e-6 of dt |g| (solver tolerance);
#   F2  CMDGFM: <= 1e-8 in every case (the cases that failed before the fix: composite with a solid in either phase, 0.029 / 0.030);
#   F3  mimetic: > 1e-3 in every case (still open, issue #76), reported.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-two-velocity/scripts/code_04_full_step.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ForcesAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "forces_audit.jl"))
using .ForcesAudit
using TwoPhaseAMR
const FA = ForcesAudit

function hydrostatic_step(kind, model, solid; dt = 0.01)
    field = FA.plane_field(kind; gravity = (0.0, -1.0), solid)
    physics = TwoPhase(field; density = FA.DENSITY, dynamic_viscosity = (0.0, 0.0), gravity = (0.0, -1.0))
    flow = FlowSimulation(field.mesh, physics; velocity_model = model, dt)
    simulate!(flow, dt; tolerance = 1.0e-10)
    return FA.maxabs(flow.velocity) / dt
end
models = (:one => OneVelocityModel(), :cmdgfm => TwoVelocityModel(pressure_projection = CMDGFMProjection()),
    :mimetic => TwoVelocityModel(pressure_projection = MimeticProjection()))
rows = []
for kind in (:uniform, :amr), solid in (:none, :gas, :liquid), (name, model) in models
    r = hydrostatic_step(kind, model, solid)
    push!(rows, (; kind, solid, name, r))
    println(rpad("$kind $solid $name", 26), " leftover/(dt|g|) = ", r)
end
@testset "full hydrostatic step after the fix of #76" begin
    for r in rows
        r.name == :one && @test r.r <= 1.0e-6
        r.name == :cmdgfm && @test r.r <= 1.0e-8
        r.name == :mimetic && @test r.r > 1.0e-3
    end
end
