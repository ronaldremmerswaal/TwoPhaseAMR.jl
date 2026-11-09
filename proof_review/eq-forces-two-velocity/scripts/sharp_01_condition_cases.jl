# Sharpness 01: why the CMDGFM balance condition is exactly  "R^1 J_g = R^tv D p  for a cell function p"  (the jump unknown xi cannot help).
#
# Label:  eq:forces:two-velocity
# Source: src/two_phase/projection/common.jl (_projection_gradient_core!): phase gradients = common + zeta^pi/m * xi with
#         zeta^l = -m_g, zeta^g = m_l, m = m_l + m_g, so  grad^g - grad^l = xi  (zeta^g - zeta^l = m).
# Both phase velocities start equal (the common increment), hence after the projection  v^l - v^g = +/- dt xi  on every face and
# v^l = v^g where there is no xi (non-mixed faces).  A rest state (v^l = v^g = 0) therefore requires xi = 0, and then
# R^1 J = R^tv D p for the pressure p of the solve (the pressure part is common to both phases).  Checked on the projection outputs:
#   X1  max | |v^l - v^g| - dt |xi| | over the faces where both phases exist (identity of the code),
#   X2  max |xi|: ZERO in every hydrostatic case, also in the failing ones (composite + solid) -- the jump rows are homogeneous, xi = 0 solves them;
#   X3  consequently the CMDGFM phase velocities equal the ONE-velocity projection (cut-cell project!) with the two-velocity operator (coefficient a/rho)
#       of the common increment R^1 J: max |v^pi_CMDGFM - v_one-velocity(R^tv)| / (dt max|R^1 J|)  at the level of the solver tolerance;
#       so the leftover of the failing cases is  (I - P_tv) R^1 J,  P_tv the projection of the map R^tv.
# NOTE (after the fix of #76, commit 497a401): the code now adds R^tv J; this script deliberately keeps the FORMER increment R^1 J (one-velocity operator)
# as the counterfactual that shows the mechanism; the code's own path is checked in code_02 (:code) and code_04 (full steps).
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-two-velocity/scripts/sharp_01_condition_cases.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ForcesAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "forces_audit.jl"))
using .ForcesAudit
using TwoPhaseAMR
const FA = ForcesAudit

dt = 0.01
g = (0.0, -1.0)
rows = []
for (kind, solid) in ((:uniform, :gas), (:amr, :none), (:amr, :gas), (:amr, :liquid))
    field = FA.plane_field(kind; gravity = g, solid)
    mesh = field.mesh
    op = FA.one_velocity_operator(field)
    acc = FaceVelocity(mesh)
    gravity!(acc, field, op, EllipticWorkspace(op); density = FA.DENSITY, gravity = g)
    phases = (FaceVelocity(mesh), FaceVelocity(mesh))
    for ph in phases
        ph.interior .= dt .* acc.interior
    end
    AMR._mark_inactive_phase_velocities!(phases, field)
    solver = CMDGFMProjectionSolver(field; density = FA.DENSITY)
    workspace = CMDGFMProjectionWorkspace(solver)
    stats = project!(phases, CellField(mesh), solver, dt, workspace; tolerance = 1.0e-12, max_iterations = 4000)
    stats.converged || error("CMDGFM did not converge")
    xi = workspace.jump
    both = [i for i in eachindex(xi) if !isnan(phases[1].interior[i]) && !isnan(phases[2].interior[i])]
    diff = [abs(phases[1].interior[i] - phases[2].interior[i]) for i in both]
    x1 = maximum(abs(diff[k] - dt * abs(xi[i])) for (k, i) in enumerate(both))
    x2 = maximum(abs, xi)
    nonmixed_equal = maximum(abs(phases[1].interior[i] - phases[2].interior[i]) for i in both if xi[i] == 0; init = 0.0)
    leftover = FA.maxabs(phases) / dt
    # X3: one-velocity projection with the two-velocity operator
    optv = FA.two_velocity_operator(field)
    v1 = FaceVelocity(mesh); v1.interior .= dt .* acc.interior
    FA.project_velocity!(field, optv, v1; dt)
    x3 = maximum(abs(phases[1].interior[i] - v1.interior[i]) for i in eachindex(v1.interior) if !isnan(phases[1].interior[i])) / (dt * maximum(abs, acc.interior))
    push!(rows, (; kind, solid, x1, x2, nonmixed_equal, leftover, x3))
    println(
        rpad("$kind $solid", 14), " X1 ||v^l - v^g| - dt|xi|| = ", round(x1, sigdigits = 2), "   X2 max|xi| = ", round(x2, sigdigits = 3),
        "   v^l - v^g at faces without xi: ", round(nonmixed_equal, sigdigits = 2), "   leftover = ", round(leftover, sigdigits = 3), "   X3 = ", round(x3, sigdigits = 2)
    )
end
@testset "xi is zero: CMDGFM = one-velocity projection with the two-velocity map" begin
    for r in rows
        @test r.x1 <= 1.0e-12
        @test r.nonmixed_equal <= 1.0e-12
        @test r.x2 == 0                                      # xi = 0 also where the rest state is lost
        @test r.x3 <= 1.0e-8
    end
end
