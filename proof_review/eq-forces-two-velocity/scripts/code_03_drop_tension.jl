# Code check 03: drop with prescribed constant curvature, its force added to both phases and projected by CMDGFM (issue #76, third bullet:
# "balanced without a solid but not on the cut mesh, although the solid does not touch the drop -- unexplained").
#
# Label:  eq:forces:two-velocity  (issue #76)
# Source: docs/techref/chapters/08_forces.tex:545-574; docs/techref/figures/forces.jl (two_velocity_tension uses CSF)
# Checks (drop of radius 2.23, sigma = 0.7, kappa0 = 1/2.23, CSF and CGFM; uniform, composite, composite + sloped solid):
#   D1  CMDGFM leftover / (dt max|R J|) with the former one-velocity increment (before the fix of #76) and with the code's increment (after the fix,
#       the projection's A/rho map): only (cut, CSF, former) fails (1.3e-2); the code after the fix is balanced everywhere (<= 1e-8); CGFM always was;
#   D2  the cause: E = R^1 J - R^tv J (J = D(sigma kappa0 psi)) is nonzero at ONE face, a cut face with a = 0.8 and J = 0 that is coupled by the
#       metric K at the level transition x = 4 to faces carrying a CSF jump (the CSF smear reaches the row y = 1.5, one cell above the solid);
#       CGFM puts the jump on the faces where the indicator changes, which are not coupled to a cut face.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-two-velocity/scripts/code_03_drop_tension.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ForcesAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "forces_audit.jl"))
using .ForcesAudit
using TwoPhaseAMR
const FA = ForcesAudit

dt = 0.01
kappa0 = 1 / FA.DROP.radius
rows = []
for kind in (:uniform, :amr, :cut), formulation in (CSF(), CGFM())
    field = FA.drop_field(kind); mesh = field.mesh
    name = nameof(typeof(formulation))
    leftovers = Dict()
    for which in (:one, :code)
        op = if which == :one
            FA.one_velocity_operator(field)
        elseif isnothing(field.geometry)
            FA.one_velocity_operator(field)
        else
            cs = CMDGFMProjectionSolver(field; density = FA.DENSITY)      # the operator of the code after the fix of #76
            AMR._phase_force_pressure_operator!(cs, CMDGFMProjectionWorkspace(cs))
        end
        force = FaceVelocity(mesh)
        surface_tension!(force, field, op, EllipticWorkspace(op); sigma = FA.SIGMA, formulation, curvature = kappa0)
        scale = maximum(abs, force.interior)
        phases = (FaceVelocity(mesh), FaceVelocity(mesh))
        for ph in phases
            ph.interior .= dt .* force.interior
        end
        AMR._mark_inactive_phase_velocities!(phases, field)
        solver = CMDGFMProjectionSolver(field; density = FA.DENSITY)
        stats = project!(phases, CellField(mesh), solver, dt, CMDGFMProjectionWorkspace(solver); tolerance = 1.0e-12, max_iterations = 4000)
        stats.converged || error("CMDGFM did not converge")
        leftovers[which] = FA.maxabs(phases) / (dt * scale)
    end
    # D2
    a = FA.apertures(field)
    isolid = !isnothing(field.geometry)
    rho = AMR.mixture_density(field, FA.DENSITY)
    op1 = FA.one_velocity_operator(field); optv = FA.two_velocity_operator(field)
    D = FA.jump_matrix(mesh)
    psi = FA.cell_vector(ref -> (c = AMR._conditional_fraction(field, ref); formulation isa CSF ? c : Float64(c > 0.5)), mesh)
    J = FA.SIGMA * kappa0 .* (D * psi) .* (a .> 0)
    R1 = FA.force_map(FA.metric_matrix(op1), isolid ? a .^ 2 ./ rho : 1 ./ rho, a, J, isolid)
    Rtv = FA.force_map(FA.metric_matrix(optv), isolid ? a ./ rho : 1 ./ rho, a, J, isolid)
    E = R1 .- Rtv
    bad = findall(i -> abs(E[i]) > 1.0e-10 * maximum(abs, R1), eachindex(E))
    push!(
        rows, (;
            kind, name, one = leftovers[:one], code = leftovers[:code], nE = length(bad),
            E_faces = [(mesh.faces[i].axis, round.(mesh.faces[i].center, digits = 3), a[i], J[i]) for i in bad],
        )
    )
    println(
        rpad("$kind $name", 12), " CMDGFM leftover: former increment ", round(leftovers[:one], sigdigits = 3), ", code after the fix ", round(leftovers[:code], sigdigits = 3),
        "   faces with E != 0: ", length(bad), " ", [(mesh.faces[i].axis, round.(mesh.faces[i].center, digits = 3), round(a[i], digits = 2), J[i]) for i in bad]
    )
end
@testset "drop with constant curvature through CMDGFM" begin
    for r in rows
        failing = r.kind == :cut && r.name == :CSF          # the former increment (before the fix of #76)
        @test (r.one > 1.0e-3) == failing
        @test r.code <= 1.0e-8                              # the code after the fix balances every case, also the cut CSF drop
        @test (r.nE > 0) == failing
        failing && @test r.nE == 1 && r.E_faces[1][3] < 1 && r.E_faces[1][4] == 0
    end
end
