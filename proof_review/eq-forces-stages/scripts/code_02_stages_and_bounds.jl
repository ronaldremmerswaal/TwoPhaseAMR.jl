# Code check 02: the stage structure and the time-step bounds of eq:forces:stages and the body acceleration (eq:forces:body).
#
# Label:  eq:forces:stages, eq:forces:body, eq:vof:dt-forces (as used in 8.7)
# Source: docs/techref/chapters/08_forces.tex:495-519 (body), 575-624 (stages); src/navier_stokes/stages.jl (_add_body_force!), src/single_phase/stepping.jl (step!),
#         src/two_phase/stepping.jl, src/two_phase/two_velocity/stepping.jl; thesis (4.1), (7.4)
# Checks:
#   B1  body acceleration: v_f += dt * b(x_f, y_f, t^n + dt/2, k_f) at the face centres, interior faces only, boundary values untouched; the callback is
#       evaluated at the midpoint time only (recorded); in the two-velocity model absent (NaN) values are skipped and the callback is not evaluated there;
#   S1  from rest, a planar interface with gravity and a LARGE viscosity (mu = (5000, 500), nu dt/h^2 = 0.2): one step leaves the velocity at the solver
#       tolerance, exactly as with mu = 0 -- the gravity increment is applied after the viscous step and is not diffused (Remark 7.1);
#   S2  the one-velocity step needs no refresh/force when sigma = 0 and g = 0: velocity stays exactly zero from rest;
#   T1  stable_timestep at rest equals min(dt_cap, dt_grav) of eq:vof:dt-forces with h the smallest cell width (sigma > 0, g != 0);
#   T2  step! rejects dt just above (1 + 32 eps) of the bound and accepts dt at the bound, for the capillary and the gravity limit, and names the bound.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-stages/scripts/code_02_stages_and_bounds.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ForcesAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "forces_audit.jl"))
using .ForcesAudit
using TwoPhaseAMR
const FA = ForcesAudit

# B1
mesh = FA.mesh_of(:amr)
calls = Tuple{Float64, Float64, Float64, Int}[]
b(x, y, t, k) = (push!(calls, (x, y, t, k)); 1.0 + 2x - 3y + 0.5t + 10k)
v = FaceVelocity(mesh); v.interior .= 0.25; v.boundary .= -7.0
AMR._add_body_force!(v, 0.02, 1.5, b)
b1 = maximum(abs(v.interior[i] - (0.25 + 0.02 * (1.0 + 2f.center[1] - 3f.center[2] + 0.5 * 1.5 + 10f.axis))) for (i, f) in pairs(mesh.faces))
b1_boundary = all(==(-7.0), v.boundary)
b1_times = all(c -> c[3] == 1.5, calls) && length(calls) == length(mesh.faces)
phases = (FaceVelocity(mesh), FaceVelocity(mesh))
for p in phases
    p.interior .= 0.0
end
phases[2].interior[1:3:end] .= NaN
empty!(calls)
AMR._add_body_force!(phases, 0.02, 1.5, b)
absent_ok = all(isnan, phases[2].interior[1:3:end]) && length(calls) == length(mesh.faces) + count(!isnan, phases[2].interior)
println("B1 body force: max deviation ", b1, ", boundary untouched ", b1_boundary, ", callback at midpoint only, once per face ", b1_times, "; two-velocity absent values skipped ", absent_ok)

# S1, S2
function one_step(model_physics, field; dt = 0.01)
    flow = FlowSimulation(field.mesh, model_physics; dt)
    simulate!(flow, dt; tolerance = 1.0e-10)
    return FA.maxabs(flow.velocity) / dt
end
field = FA.plane_field(:uniform; solid = :none)
rest_viscous = one_step(TwoPhase(field; density = FA.DENSITY, dynamic_viscosity = (5000.0, 500.0), gravity = (0.0, -1.0)), field)
field = FA.plane_field(:uniform; solid = :none)
rest_inviscid = one_step(TwoPhase(field; density = FA.DENSITY, dynamic_viscosity = (0.0, 0.0), gravity = (0.0, -1.0)), field)
field = FA.plane_field(:uniform; solid = :none)
rest_zero = one_step(TwoPhase(field; density = FA.DENSITY, dynamic_viscosity = (0.0, 0.0)), field)
println("S1 one step from rest, gravity: mu = (5000, 500): ", rest_viscous, "; mu = 0: ", rest_inviscid, "; S2 no forces: ", rest_zero)

# T1, T2
field = FA.plane_field(:amr; solid = :none)
sigma = 0.07; g = (0.0, -1.0)
physics = TwoPhase(field; density = FA.DENSITY, dynamic_viscosity = (0.0, 0.0), surface_tension = SurfaceTension(sigma), gravity = g)
flow = FlowSimulation(field.mesh, physics; dt = 1.0e-4)
h = minimum(minimum(blk.spacing) for blk in field.mesh.blocks)
rho = FA.DENSITY
dt_cap = sqrt(sum(rho) * h^3 / (4pi * sigma))
dt_grav = sqrt(sum(rho) / abs(rho[1] - rho[2]) * h / (2 * hypot(g...)))
bound = stable_timestep(flow)
println("T1 h = ", h, ": dt_cap = ", dt_cap, ", dt_grav = ", dt_grav, ", stable_timestep at rest = ", bound)
function rejects(sigma, g, dt)
    f = FA.plane_field(:amr; solid = :none)
    ph = TwoPhase(f; density = FA.DENSITY, dynamic_viscosity = (0.0, 0.0), surface_tension = SurfaceTension(sigma), gravity = g)
    fl = FlowSimulation(f.mesh, ph; dt)
    try
        AMR.step!(fl.velocity, fl.pressure, fl.system, dt, fl.workspace)
        return nothing
    catch err
        return err isa ArgumentError ? err.msg : rethrow()
    end
end
cap_only = rejects(0.07, (0.0, 0.0), 1.01 * dt_cap)
cap_ok = rejects(0.07, (0.0, 0.0), dt_cap)
grav_only = rejects(0.0, (0.0, -1.0), 1.01 * dt_grav)
grav_ok = rejects(0.0, (0.0, -1.0), dt_grav)
println("T2 capillary: dt = 1.01 bound -> ", cap_only, "; dt = bound -> ", cap_ok, "\n   gravity: dt = 1.01 bound -> ", grav_only, "; dt = bound -> ", grav_ok)

@testset "stages, body acceleration, bounds" begin
    @test b1 <= 1.0e-15 && b1_boundary && b1_times && absent_ok
    @test rest_viscous <= 1.0e-6 && rest_inviscid <= 1.0e-6 && rest_zero == 0
    @test abs(bound - min(dt_cap, dt_grav)) <= 1.0e-14 * bound
    @test cap_only isa String && occursin("capillary", cap_only)
    @test cap_ok === nothing
    @test grav_only isa String && occursin("gravity", grav_only)
    @test grav_ok === nothing
end
