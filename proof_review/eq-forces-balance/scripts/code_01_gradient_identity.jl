# Code check 01: the force map of a jump field that is the jump of a cell function is the weighted gradient
# (eq:forces:balance, eq:forces:map-gradient), faces with gamma_f = 0 do not enter.
#
# Label:  eq:forces:balance, eq:forces:map-gradient
# Source: docs/techref/chapters/08_forces.tex:137-212; src/two_phase/forces.jl (_weighted_jump_force!),
#         src/operators/elliptic.jl (weighted_gradient!), src/embedded/operators.jl (_cut_cell_weighted_gradient!)
# Checks, on uniform/composite meshes without solid, with the sloped solid and with a solid disc (5 fixtures, random cell
# functions a, seeded):
#   B1  R(D a) from the code (_weighted_jump_force!) equals the weighted gradient S a of the code (relative gap);
#   B2  the same with arbitrary garbage added to J on the faces with gamma_f = 0 (only a_f = 0 faces with a solid);
#   B3  R J replicated independently from K (read from the code), gamma = 1/rho or a^2/rho, W-scaling inside K.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-balance/scripts/code_01_gradient_identity.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ForcesAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "forces_audit.jl"))
using .ForcesAudit
using TwoPhaseAMR
const FA = ForcesAudit

rng = MersenneTwister(20261109)
worst_gradient = 0.0; worst_garbage = 0.0; worst_replica = 0.0
for (name, field) in FA.balance_fixtures()
    mesh = field.mesh
    op = FA.one_velocity_operator(field)
    work = EllipticWorkspace(op)
    solid = !isnothing(field.geometry)
    a = FA.apertures(field)
    D = FA.jump_matrix(mesh)
    K = FA.metric_matrix(op)
    rho = AMR.mixture_density(field, FA.DENSITY)
    gamma = solid ? a .^ 2 ./ rho : 1 ./ rho
    for draw in 1:3
        avals = randn(rng, size(D, 2))
        J = D * avals
        dest = FaceVelocity(mesh)
        unused = FaceVelocity(mesh)
        op.gradient_operator.pressure_jump .= J      # (the code's scratch is written by the force routines)
        AMR._weighted_jump_force!(dest, op, work, copy(J), field.geometry)
        S = FA.weighted_gradient_of(field, op, avals)
        scale = maximum(abs, S.interior)
        global worst_gradient = max(worst_gradient, maximum(abs, dest.interior .- S.interior) / scale)
        # B2: garbage on the faces with gamma = 0
        Jg = copy(J)
        dead = findall(iszero, gamma)
        Jg[dead] .= 1.0e3 .* randn(rng, length(dead))
        dest2 = FaceVelocity(mesh)
        AMR._weighted_jump_force!(dest2, op, work, Jg, field.geometry)
        global worst_garbage = max(worst_garbage, maximum(abs, dest2.interior .- S.interior) / scale)
        # B3: independent replica
        replica = FA.force_map(K, gamma, a, J, solid)
        global worst_replica = max(worst_replica, maximum(abs, replica .- dest.interior) / scale)
    end
    println(
        rpad("fixture $name", 20), " faces=", length(mesh.faces), " gamma=0 faces=", count(iszero, gamma),
        " nondiagonal K: ", !(op.gradient_operator.inverse_metric isa AMR.DiagonalInverseFaceMetric)
    )
end
println("B1 max |R(Da) - S a| / max|S a|          = ", worst_gradient)
println("B2 same with garbage on gamma = 0 faces  = ", worst_garbage)
println("B3 max |R J (replica) - R J (code)| rel. = ", worst_replica)
@testset "gradient identity" begin
    @test worst_gradient <= 1.0e-14
    @test worst_garbage <= 1.0e-14
    @test worst_replica <= 1.0e-13
end
