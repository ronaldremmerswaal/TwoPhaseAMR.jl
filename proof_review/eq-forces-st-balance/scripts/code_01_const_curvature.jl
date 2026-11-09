# Code check 01: balanced surface tension, eq:forces:st-balance -- with a constant curvature the force is the weighted gradient of
# sigma*kappa0*psi, the one-velocity projection leaves a solenoidal predictor unchanged, and the pressure jumps by sigma*kappa0
# (CGFM) or follows the conditional fraction (CSF).
#
# Label:  eq:forces:st-balance  (eq:forces:st-jump, eq:forces:stress-balance, eq:forces:steady (a))
# Source: docs/techref/chapters/08_forces.tex:338-366; src/two_phase/forces.jl (surface_tension!, _surface_tension_jump)
# Checks (drop of radius 2.23, sigma = 0.7, densities (1000, 1), kappa0 = 1/2.23; uniform, composite, composite+sloped solid; CSF, CGFM):
#   S1  the jumps written by surface_tension! are J = sigma kappa0 D psi (psi = C or the indicator C > 1/2) on every face with a > 0, zero elsewhere;
#   S2  the acceleration equals the weighted gradient of sigma kappa0 psi (code's weighted_gradient!/_cut_cell_weighted_gradient!);
#   S3  projection of  v0 + dt * acceleration  (v0 nonzero solenoidal) returns v0, the pressure p satisfies S(p - sigma kappa0 psi) = 0;
#   S4  CGFM: p is constant in the liquid core and in the gas away from the interface and p_liquid - p_gas = sigma kappa0
#       (discrete Young-Laplace, kappa0 > 0 for a convex drop); CSF: p - sigma kappa0 C is constant (min/max over cells of the component).
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-st-balance/scripts/code_01_const_curvature.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ForcesAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "forces_audit.jl"))
using .ForcesAudit
using TwoPhaseAMR
const FA = ForcesAudit

rng = MersenneTwister(20261114)
dt = 0.01
kappa0 = 1 / FA.DROP.radius
results = []
for kind in (:uniform, :amr, :cut), formulation in (CSF(), CGFM())
    field = FA.drop_field(kind)
    mesh = field.mesh
    op = FA.one_velocity_operator(field)
    work = EllipticWorkspace(op)
    a = FA.apertures(field)
    D = FA.jump_matrix(mesh)
    psi = FA.cell_vector(mesh) do ref
        c = AMR._conditional_fraction(field, ref)
        formulation isa CSF ? c : Float64(c > 0.5)
    end
    force = FaceVelocity(mesh)
    surface_tension!(force, field, op, work; sigma = FA.SIGMA, formulation, curvature = kappa0)
    J = copy(op.gradient_operator.pressure_jump)
    Jexp = FA.SIGMA * kappa0 .* (D * psi) .* (a .> 0)
    s1 = maximum(abs, J .- Jexp) / maximum(abs, Jexp)
    S = FA.weighted_gradient_of(field, op, FA.SIGMA * kappa0 .* psi)
    s2 = maximum(abs, force.interior .- S.interior) / maximum(abs, S.interior)
    w = FaceVelocity(mesh); w.interior .= randn(rng, length(w.interior))
    FA.project_velocity!(field, op, w; dt)
    v0 = copy(w.interior)
    v = FaceVelocity(mesh); v.interior .= v0 .+ dt .* force.interior
    p, _ = FA.project_velocity!(field, op, v; dt)
    s3 = maximum(abs, v.interior .- v0) / (dt * maximum(abs, force.interior))
    pv = FA.cell_vector(ref -> AMR._get(p, ref), mesh)
    Sres = FA.weighted_gradient_of(field, op, pv .- FA.SIGMA * kappa0 .* psi)
    s3p = maximum(abs, Sres.interior) / maximum(abs, S.interior)
    # S4: pressure levels (active cells only)
    active = isnothing(field.geometry) ? trues(length(pv)) :
        FA.cell_vector(ref -> field.geometry.active_cells[AMR._cell_index(ref, mesh.block_size...)] ? 1.0 : 0.0, mesh) .> 0
    cx = FA.cell_vector(ref -> cellcenter(mesh, ref.block, ref.i, ref.j)[1], mesh)
    cy = FA.cell_vector(ref -> cellcenter(mesh, ref.block, ref.i, ref.j)[2], mesh)
    dist = hypot.(cx .- FA.DROP.centre[1], cy .- FA.DROP.centre[2])
    fraction = FA.cell_vector(ref -> AMR._conditional_fraction(field, ref), mesh)
    if formulation isa CGFM
        liquid = active .& (fraction .== 1) .& (dist .< FA.DROP.radius - 1.0)
        gas = active .& (fraction .== 0) .& (dist .> FA.DROP.radius + 1.0)
        spread = max(-(-)(extrema(pv[liquid])...), -(-)(extrema(pv[gas])...))
        level = sum(pv[liquid]) / count(liquid) - sum(pv[gas]) / count(gas)
        s4 = (; spread, level)
    else
        resid = (pv .- FA.SIGMA * kappa0 .* fraction)[active]
        s4 = (; spread = maximum(resid) - minimum(resid), level = NaN)
    end
    push!(results, (; kind, name = nameof(typeof(formulation)), s1, s2, s3, s3p, s4))
    println(
        rpad("$kind $(nameof(typeof(formulation)))", 12), " S1=", round(s1, sigdigits = 2), " S2=", round(s2, sigdigits = 2), " S3(velocity)=", round(s3, sigdigits = 2),
        " S3(S(p-psi))=", round(s3p, sigdigits = 2), " S4 spread=", round(s4.spread, sigdigits = 2), " level jump=", round(s4.level, sigdigits = 6), " (sigma kappa0 = ", round(FA.SIGMA * kappa0, sigdigits = 6), ")"
    )
end
@testset "balanced surface tension (constant curvature)" begin
    for r in results
        @test r.s1 <= 1.0e-14
        @test r.s2 <= 1.0e-12
        @test r.s3 <= 1.0e-5          # leftover of the linear solve (relative defect 1e-12 at density ratio 1000), not of the map
        @test r.s3p <= 1.0e-5
        @test r.s4.spread <= 1.0e-6 * FA.SIGMA * kappa0
        r.name == :CGFM && @test abs(r.s4.level - FA.SIGMA * kappa0) <= 1.0e-6
    end
end
