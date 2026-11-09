# Code check 01: sign of the discrete curvature and of the Laplace pressure -- a convex drop has kappa > 0 (cell curvature of the package ~ 1/R) and
# the one-velocity projection of the surface-tension force gives the higher pressure inside (eq:forces:stress-balance, eq:forces:steady (a)).
#
# Label:  eq:forces:stress-balance, eq:forces:steady
# Source: docs/techref/chapters/08_forces.tex:24-80; src/interface (cell curvature), src/two_phase/forces.jl
# Checks: mean cell curvature of the mixed cells of the drop (uniform, composite) equals 1/R to 5 %; the Laplace pressure p_core - p_far of the
# projected CGFM force is within 5 % of sigma/R and positive; the static-drop pressure of a force with the exactly constant curvature is sigma/R to 1e-6.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-continuous/scripts/code_01_cell_curvature_sign.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ForcesAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "forces_audit.jl"))
using .ForcesAudit
using TwoPhaseAMR
const FA = ForcesAudit

dt = 0.01
rows = []
for kind in (:uniform, :amr)
    field = FA.drop_field(kind); mesh = field.mesh
    kappas = [AMR._get(field.curvature, ref) for ref in AMR._cell_references(mesh) if 0 < AMR._conditional_fraction(field, ref) < 1]
    kappas = filter(isfinite, kappas)
    mean_kappa = sum(kappas) / length(kappas)
    op = FA.one_velocity_operator(field)
    pressures = Float64[]
    for curvature in (nothing, 1 / FA.DROP.radius)
        force = FaceVelocity(mesh)
        surface_tension!(force, field, op, EllipticWorkspace(op); sigma = FA.SIGMA, formulation = CGFM(), curvature)
        v = FaceVelocity(mesh); v.interior .= dt .* force.interior
        p, _ = FA.project_velocity!(field, op, v; dt)
        pv = FA.cell_vector(ref -> AMR._get(p, ref), mesh)
        cx = FA.cell_vector(ref -> cellcenter(mesh, ref.block, ref.i, ref.j)[1], mesh)
        cy = FA.cell_vector(ref -> cellcenter(mesh, ref.block, ref.i, ref.j)[2], mesh)
        dist = hypot.(cx .- FA.DROP.centre[1], cy .- FA.DROP.centre[2])
        core = dist .< FA.DROP.radius - 1.0; far = dist .> FA.DROP.radius + 1.0
        push!(pressures, sum(pv[core]) / count(core) - sum(pv[far]) / count(far))
    end
    push!(rows, (; kind, mean_kappa, laplace = pressures[1], laplace_const = pressures[2]))
    println(rpad("$kind", 8), " mean cell curvature ", mean_kappa, " (1/R = ", 1 / FA.DROP.radius, "), Laplace pressure real curvature ", pressures[1], ", constant curvature ", pressures[2], " (sigma/R = ", FA.SIGMA / FA.DROP.radius, ")")
end
@testset "sign of curvature and Laplace pressure" begin
    for r in rows
        @test abs(r.mean_kappa * FA.DROP.radius - 1) <= 0.05
        @test r.laplace > 0 && abs(r.laplace * FA.DROP.radius / FA.SIGMA - 1) <= 0.05
        @test abs(r.laplace_const * FA.DROP.radius / FA.SIGMA - 1) <= 1.0e-6
    end
end
