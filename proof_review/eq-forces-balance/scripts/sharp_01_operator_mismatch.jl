# Sharpness 01: the balance statement needs BOTH conditions of the text -- the jumps of one cell function, and the projection
# with the operator of the force map.  Each is violated in turn; the leftover velocity after one projection is reported.
#
# Label:  eq:forces:balance
# Source: docs/techref/chapters/08_forces.tex:169-212
# Cases (composite meshes, dt = 0.01, v = 0, J = D a for a random cell function a, force R_gamma J):
#   M1  projection operator with a stale coefficient (density of a drop displaced by 0.3, same mesh and metric): leftover O(1);
#   M2  projection operator = the two-velocity operator (coefficient a/rho instead of a^2/rho) on the cut mesh: leftover > 0 only
#       with a solid (without solid the two coefficients coincide: leftover = solver tolerance);
#   M3  J perturbed on ONE face with gamma > 0 (relative 1e-2): the leftover equals the best-approximation error, nonzero.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-balance/scripts/sharp_01_operator_mismatch.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ForcesAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "forces_audit.jl"))
using .ForcesAudit
using TwoPhaseAMR
const FA = ForcesAudit

rng = MersenneTwister(20261112)
dt = 0.01
function leftover(field, force_operator, projection_operator, J)
    acc = FaceVelocity(field.mesh)
    AMR._weighted_jump_force!(acc, force_operator, EllipticWorkspace(force_operator), J, field.geometry)
    return FA.project_force(field, projection_operator, acc; dt) / maximum(abs, acc.interior)
end

# M1 -- stale coefficient
field = FA.drop_field(:amr)
shifted = InterfaceField(field.mesh, (x, y) -> hypot(x - 4.6, y - 4.1) - FA.DROP.radius; lipschitz = 1)
op_now = FA.one_velocity_operator(field)
op_stale = FA.one_velocity_operator(shifted)
D = FA.jump_matrix(field.mesh)
avals = randn(rng, size(D, 2))
m1_fresh = leftover(field, op_now, FA.one_velocity_operator(field), D * avals)
m1_stale = leftover(field, op_now, op_stale, D * avals)
println("M1 projection with the operator of the force map: ", m1_fresh, "; with a stale coefficient: ", m1_stale)

# M2 -- two-velocity coefficient
fcut = FA.drop_field(:cut)
Dc = FA.jump_matrix(fcut.mesh)
ac = randn(rng, size(Dc, 2))
m2_same = leftover(fcut, FA.one_velocity_operator(fcut), FA.one_velocity_operator(fcut), Dc * ac)
m2_tv = leftover(fcut, FA.one_velocity_operator(fcut), FA.two_velocity_operator(fcut), Dc * ac)
fnos = FA.drop_field(:amr)
Dn = FA.jump_matrix(fnos.mesh)
an = randn(rng, size(Dn, 2))
m2_nosolid = leftover(fnos, FA.one_velocity_operator(fnos), FA.two_velocity_operator(fnos), Dn * an)
println("M2 cut mesh: one-velocity projection ", m2_same, ", two-velocity operator ", m2_tv, "; no solid, two-velocity operator ", m2_nosolid)

# M3 -- one wrong face
J = Dn * an
J[div(length(J), 2)] += 1.0e-2 * maximum(abs, J)
m3 = leftover(fnos, FA.one_velocity_operator(fnos), FA.one_velocity_operator(fnos), J)
println("M3 one face wrong by 1e-2: leftover ", m3)

@testset "sharpness: both conditions of eq:forces:balance are needed" begin
    @test m1_fresh <= 1.0e-8
    @test m1_stale >= 1.0e-3
    @test m2_same <= 1.0e-8
    @test m2_tv >= 1.0e-3
    @test m2_nosolid <= 1.0e-8
    @test m3 >= 1.0e-5
end
