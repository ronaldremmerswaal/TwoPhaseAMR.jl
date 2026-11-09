# Code check 02: one-velocity projection of  v + dt R J  with  J = D a  returns v and the pressure a (eq:forces:balance, consequence).
#
# Label:  eq:forces:balance
# Source: docs/techref/chapters/08_forces.tex:169-212; src/operators/elliptic.jl (project!), src/embedded/operators.jl (project!)
# Checks (5 fixtures of the balance audit; v discretely (open-)solenoidal and nonzero, obtained by projecting a random velocity;
# random cell function a):
#   P1  the projection of v itself changes v by at most the solver tolerance (v is solenoidal);
#   P2  the projection of v + dt R(D a) returns v: max |v_new - v| / (dt max|R(D a)|);
#   P3  the pressure returned satisfies S p = S a, i.e. p = a + (constant per connected component): max |S(p - a)| / max|S a|.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-balance/scripts/code_02_projection_returns.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ForcesAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "forces_audit.jl"))
using .ForcesAudit
using TwoPhaseAMR
const FA = ForcesAudit

rng = MersenneTwister(20261110)
dt = 0.01
worst1 = worst2 = worst3 = 0.0
for (name, field) in FA.balance_fixtures()
    mesh = field.mesh
    op = FA.one_velocity_operator(field)
    work = EllipticWorkspace(op)
    D = FA.jump_matrix(mesh)
    w = FaceVelocity(mesh); w.interior .= randn(rng, length(w.interior))
    FA.project_velocity!(field, op, w; dt)                       # v0 = P w
    v0 = copy(w.interior)
    again = FaceVelocity(mesh); again.interior .= v0
    FA.project_velocity!(field, op, again; dt)
    e1 = maximum(abs, again.interior .- v0) / maximum(abs, v0)
    avals = randn(rng, size(D, 2))
    J = D * avals
    acc = FaceVelocity(mesh)
    AMR._weighted_jump_force!(acc, op, work, J, field.geometry)
    v = FaceVelocity(mesh); v.interior .= v0 .+ dt .* acc.interior
    p, _ = FA.project_velocity!(field, op, v; dt)
    e2 = maximum(abs, v.interior .- v0) / (dt * maximum(abs, acc.interior))
    # P3: S (p - a)
    pvals = FA.cell_vector(ref -> AMR._get(p, ref), mesh)
    Sd = FA.weighted_gradient_of(field, op, pvals .- avals)
    Sa = FA.weighted_gradient_of(field, op, avals)
    e3 = maximum(abs, Sd.interior) / maximum(abs, Sa.interior)
    println(rpad("fixture $name", 20), " P1=", round(e1, sigdigits = 3), " P2=", round(e2, sigdigits = 3), " P3=", round(e3, sigdigits = 3))
    global worst1 = max(worst1, e1); global worst2 = max(worst2, e2); global worst3 = max(worst3, e3)
end
@testset "projection returns v and a" begin
    @test worst1 <= 1.0e-9
    @test worst2 <= 1.0e-9
    @test worst3 <= 1.0e-9
end
