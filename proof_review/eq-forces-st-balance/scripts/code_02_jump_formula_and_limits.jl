# Code check 02: the surface-tension jump (eq:forces:st-jump), its sign, the zero rules, the level-transition remark, and the
# limits of the balanced statement for the real (face-to-face varying) curvature.
#
# Label:  eq:forces:st-jump, eq:forces:st-balance
# Source: docs/techref/chapters/08_forces.tex:308-366; src/two_phase/forces.jl (surface_tension!)
# Checks (drop fixtures; sigma = 0.7):
#   J1  the code's jumps equal sigma * kappa_f * psi-jump * |f| with psi = C (CSF) or [C > 1/2] (CGFM), kappa_f the face curvature of the code,
#       zero if sigma = 0, psi-jump = 0, a_f = 0 or the face is unresolved (independent re-evaluation from the conditional fractions);
#   J2  sign: at a face whose low cell is liquid and high cell gas the CGFM jump is -sigma kappa_f |f| (< 0 for kappa_f > 0);
#   J3  R J = sigma kappa_f (S psi) exactly at the faces whose row of K is diagonal (conforming), and not at the coupled faces of a
#       level transition when kappa_f varies;
#   J4  real curvature: the force is NOT balanced -- consistency residual of J (|J_S - D_S a| over a) is O(1) of the jump scale, the
#       leftover velocity after one projection is reported; the Laplace pressure (liquid core minus far gas) of the projected force is
#       reported against sigma / R (CGFM).
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-st-balance/scripts/code_02_jump_formula_and_limits.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ForcesAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "forces_audit.jl"))
using .ForcesAudit
using TwoPhaseAMR
const FA = ForcesAudit

dt = 0.01
rows = []
for kind in (:uniform, :amr, :cut), formulation in (CSF(), CGFM())
    field = FA.drop_field(kind)
    mesh = field.mesh
    op = FA.one_velocity_operator(field)
    work = EllipticWorkspace(op)
    a = FA.apertures(field)
    nx, ny = mesh.block_size
    force = FaceVelocity(mesh)
    count = Ref(0)
    surface_tension!(force, field, op, work; sigma = FA.SIGMA, formulation, unresolved_curvature_faces = count)
    J = copy(op.gradient_operator.pressure_jump)
    # J1: independent jumps (curvature from the code's face curvature function, audited in eq-forces-curvature)
    Jexp = zeros(length(J)); unresolved = 0; sign_ok = true; nlow = 0
    for (i, face) in pairs(mesh.faces)
        a[i] > 0 || continue
        cm, cp = AMR._conditional_fraction(field, face.minus), AMR._conditional_fraction(field, face.plus)
        pj = formulation isa CSF ? cp - cm : Float64((cp > 0.5) - (cm > 0.5))
        pj == 0 && continue
        kappa = AMR._face_curvature(field, face)
        if kappa === nothing
            unresolved += 1; continue
        end
        Jexp[i] = FA.SIGMA * kappa * pj * face.area
        if formulation isa CGFM && cm > 0.5 && cp <= 0.5
            nlow += 1
            sign_ok &= Jexp[i] < 0 && J[i] == -FA.SIGMA * kappa * face.area
        end
    end
    j1 = maximum(abs, J .- Jexp) / maximum(abs, Jexp)
    # J3: diagonal-row faces
    K = FA.metric_matrix(op)
    psi = FA.cell_vector(ref -> (c = AMR._conditional_fraction(field, ref); formulation isa CSF ? c : Float64(c > 0.5)), mesh)
    S = FA.weighted_gradient_of(field, op, psi)
    kappa_f = [
        begin
            k = AMR._face_curvature(field, f); k === nothing ? 0.0 : k
        end for f in mesh.faces
    ]
    rowcount = [length(nzrange(K, i)) for i in 1:size(K, 1)]       # CSC of a symmetric matrix: column i = row i
    diagonal_rows = findall(==(1), rowcount)
    coupled = findall(>(1), rowcount)
    rhs = FA.SIGMA .* kappa_f .* S.interior
    d_diag = maximum(abs, (force.interior .- rhs)[diagonal_rows]) / maximum(abs, rhs)
    d_coupled = isempty(coupled) ? 0.0 : maximum(abs, (force.interior .- rhs)[coupled]) / maximum(abs, rhs)
    # J4
    D = FA.jump_matrix(mesh)
    support = findall(>(0), a)
    cons = FA.jump_consistency(D, J, support)
    left = FA.project_force(field, op, force; dt) / maximum(abs, force.interior)
    push!(rows, (; kind, name = nameof(typeof(formulation)), j1, unresolved, count = count[], sign_ok, nlow, d_diag, d_coupled, ndiag = length(diagonal_rows), ncoupled = length(coupled), cons, left))
    println(
        rpad("$kind $(nameof(typeof(formulation)))", 12), " J1=", j1, " unresolved ", unresolved, "/", count[], " J2 ok ", sign_ok, " (", nlow, " faces)",
        " J3 diag-rows ", round(d_diag, sigdigits = 2), " (", length(diagonal_rows), " faces), coupled rows ", round(d_coupled, sigdigits = 2), " (", length(coupled), " faces)",
        " J4 consistency ", round(cons, sigdigits = 3), " leftover/|force| ", round(left, sigdigits = 3)
    )
end
# sigma = 0 and the Laplace pressure of the real force
field = FA.drop_field(:amr)
op = FA.one_velocity_operator(field)
zero_force = FaceVelocity(field.mesh)
surface_tension!(zero_force, field, op, EllipticWorkspace(op); sigma = 0.0)
sigma0_zero = maximum(abs, zero_force.interior) == 0 && all(iszero, op.gradient_operator.pressure_jump)
force = FaceVelocity(field.mesh)
surface_tension!(force, field, op, EllipticWorkspace(op); sigma = FA.SIGMA, formulation = CGFM())
v = FaceVelocity(field.mesh); v.interior .= dt .* force.interior
p, _ = FA.project_velocity!(field, op, v; dt)
pv = FA.cell_vector(ref -> AMR._get(p, ref), field.mesh)
cx = FA.cell_vector(ref -> cellcenter(field.mesh, ref.block, ref.i, ref.j)[1], field.mesh)
cy = FA.cell_vector(ref -> cellcenter(field.mesh, ref.block, ref.i, ref.j)[2], field.mesh)
dist = hypot.(cx .- FA.DROP.centre[1], cy .- FA.DROP.centre[2])
core = dist .< FA.DROP.radius - 1.0
far = dist .> FA.DROP.radius + 1.0
laplace = sum(pv[core]) / count(core) - sum(pv[far]) / count(far)
println(
    "sigma = 0 gives zero jumps and force: ", sigma0_zero, "; Laplace pressure of the real CGFM force on the composite mesh: ", laplace,
    " (sigma/R = ", FA.SIGMA / FA.DROP.radius, ")"
)
@testset "surface-tension jump" begin
    @test sigma0_zero
    for r in rows
        @test r.j1 <= 1.0e-15
        @test r.count == r.unresolved
        r.name == :CGFM && @test r.sign_ok && r.nlow > 0
        @test r.d_diag <= 1.0e-12
        r.ncoupled > 0 && @test r.d_coupled > 1.0e-6
        @test r.cons > 1.0e-3
    end
    @test abs(laplace - FA.SIGMA / FA.DROP.radius) <= 0.05 * FA.SIGMA / FA.DROP.radius
end
