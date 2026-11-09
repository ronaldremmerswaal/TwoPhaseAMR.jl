# Code check 01: conforming reduction of the force map (eq:forces:conforming), the metric entries 1/V_f and 1/Vo_f, the coupling at level
# transitions, and the independent replica of eq:forces:map with an independent staggered density.
#
# Label:  eq:forces:map, eq:forces:conforming, eq:forces:map-gradient (see also eq-forces-balance), eq:forces:density
# Source: docs/techref/chapters/08_forces.tex:137-168; src/two_phase/forces.jl (_weighted_jump_force!), src/embedded/operators.jl (GradientOperator(geometry))
# Checks (uniform / composite meshes without solid, with the sloped solid, with a disc; random J):
#   C1  faces whose row of K is diagonal: K_ff = 1/V_f (no solid) and 1/Vo_f (solid, W^2/V_f);
#   C2  at those faces (R J)_f = J_f/(rho_f V_f) (no solid) and a_f J_f/(rho_f Vo_f) (solid), rho_f an independent staggered density;
#   C3  the faces with a coupled row lie in the vertex patches at the level transition x = 4 (the composite fixtures only); there a face
#       without a jump carries an acceleration;
#   C4  independent replica of eq:forces:map (K of the code, independent rho, gamma = 1/rho or a^2/rho) equals the code's force for the surface
#       tension (CSF and CGFM jumps of the drop) and gravity of the planar interface on every fixture.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-map/scripts/code_01_conforming.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ForcesAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "forces_audit.jl"))
using .ForcesAudit
using TwoPhaseAMR
const FA = ForcesAudit

rng = MersenneTwister(20261115)
w1 = w2 = 0.0; w3 = 0.0; coupled_x = (Inf, -Inf); no_jump_acc = 0
for (name, field) in FA.balance_fixtures()
    mesh = field.mesh
    op = FA.one_velocity_operator(field)
    solid = !isnothing(field.geometry)
    a = FA.apertures(field)
    K = FA.metric_matrix(op)
    rho = [FA.staggered_density_independent(field, f) for f in mesh.faces]
    V = Float64.(mesh.face_dual_volumes)
    Vo = solid ? Float64.(field.geometry.face_dual_volumes) : V
    rowcount = [length(nzrange(K, i)) for i in 1:size(K, 1)]
    diag = [i for i in findall(==(1), rowcount) if a[i] > 0 && Vo[i] > 0]
    d1 = maximum(abs(K[i, i] * Vo[i] - 1) for i in diag)
    J = randn(rng, length(a))
    out = FaceVelocity(mesh)
    AMR._weighted_jump_force!(out, op, EllipticWorkspace(op), J, field.geometry)
    pred = solid ? a .* J ./ (rho .* Vo) : J ./ (rho .* V)
    d2 = maximum(abs(out.interior[i] - pred[i]) / abs(pred[i]) for i in diag)
    global w1 = max(w1, d1); global w2 = max(w2, d2)
    coupled = [i for i in findall(>(1), rowcount) if a[i] > 0]
    if !isempty(coupled)
        xs = [mesh.faces[i].center[1] for i in coupled]
        global coupled_x = (min(coupled_x[1], minimum(xs)), max(coupled_x[2], maximum(xs)))
        # a face without a jump carrying an acceleration: J = 0 at one face of a coupled patch, nonzero neighbours
        Jz = randn(rng, length(a)); f0 = coupled[1]; Jz[f0] = 0
        o2 = FaceVelocity(mesh); AMR._weighted_jump_force!(o2, op, EllipticWorkspace(op), Jz, field.geometry)
        global no_jump_acc += abs(o2.interior[f0]) > 1.0e-8 ? 1 : 0
    end
    println(rpad("fixture $name", 16), " diagonal rows ", length(diag), ", coupled rows ", length(coupled), ": C1 ", round(d1, sigdigits = 2), "  C2 ", round(d2, sigdigits = 2))
end
# C4: independent replica with surface tension and gravity
for kind in (:uniform, :amr, :cut), formulation in (CSF(), CGFM())
    field = FA.drop_field(kind); mesh = field.mesh
    op = FA.one_velocity_operator(field)
    force = FaceVelocity(mesh)
    surface_tension!(force, field, op, EllipticWorkspace(op); sigma = FA.SIGMA, formulation)
    J = copy(op.gradient_operator.pressure_jump)
    solid = !isnothing(field.geometry)
    a = FA.apertures(field)
    rho = [FA.staggered_density_independent(field, f) for f in mesh.faces]
    replica = FA.force_map(FA.metric_matrix(op), solid ? a .^ 2 ./ rho : 1 ./ rho, a, J, solid)
    global w3 = max(w3, maximum(abs, force.interior .- replica) / maximum(abs, replica))
end
for kind in (:uniform, :amr), solid in (:none, :gas, :liquid)
    g = (0.0, -1.0)
    field = FA.plane_field(kind; gravity = g, solid); mesh = field.mesh
    op = FA.one_velocity_operator(field)
    force = FaceVelocity(mesh)
    gravity!(force, field, op, EllipticWorkspace(op); density = FA.DENSITY, gravity = g)
    J = copy(op.gradient_operator.pressure_jump)
    isolid = !isnothing(field.geometry)
    a = FA.apertures(field)
    rho = [FA.staggered_density_independent(field, f) for f in mesh.faces]
    replica = FA.force_map(FA.metric_matrix(op), isolid ? a .^ 2 ./ rho : 1 ./ rho, a, J, isolid)
    global w3 = max(w3, maximum(abs, force.interior .- replica) / maximum(abs, replica))
end
println("C1 max |K_ff Vo - 1| = ", w1, "; C2 max relative deviation of (R J)_f = a J/(rho Vo) = ", w2, "; C3 coupled faces x in ", coupled_x, ", faces without jump with acceleration: ", no_jump_acc,
    "; C4 replica gap = ", w3)
@testset "force map: conforming reduction and independent replica" begin
    @test w1 <= 1.0e-13
    @test w2 <= 1.0e-12
    @test 3.0 <= coupled_x[1] && coupled_x[2] <= 5.0
    @test no_jump_acc >= 3
    @test w3 <= 1.0e-12
end
