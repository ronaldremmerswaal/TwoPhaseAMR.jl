# Code check 01: gravity model of Chapter 8 -- potential point (eq:forces:potential), aperture density (eq:forces:aperture-density), integrated jumps
# (eq:forces:gravity-jump), the reductions stated below them, and the periodic-gravity rejection.
#
# Label:  eq:forces:potential, eq:forces:aperture-density, eq:forces:gravity-jump
# Source: docs/techref/chapters/08_forces.tex:399-467; src/two_phase/forces.jl (gravity!), src/interface/height_functions.jl (_interface_centroid)
# Checks:
#   G1  _interface_centroid(n, a) = midpoint of the segment {n.l = a} in the whole reference cell [-1/2,1/2]^2 (200 random lines, seeded);
#   G2  the cell potential the code forms (the scratch vector work.input after gravity!) equals g.(x_c + h .* segmid) for 0 < C < 1 and g.x_c otherwise
#       (also in a pure cut cell and in a triple cell), on the drop fixtures (incl. the sloped solid) and the planar fixtures (with solids, oblique g);
#   G3  phase apertures: a^l + a^g = a_f on every face of every fixture (max deviation), a = 1 without solid; on a face in one phase rho_a is that phase's
#       density, also on cut faces; always rho_g <= rho_a <= rho_l;
#   G4  the code's integrated jumps equal rho_a |f| (phi_+ - phi_-) with independent phi and the package's phase apertures (a-priori audited in
#       eq:interface:phase-aperture), also with gravity not normal to an interface (drop), on every fixture; thesis (7.64): no solid, rho_a = rho_l a_l + rho_g a_g;
#   G5  conforming rows: (R J)_f = J_f/(rho_f V_f) (no solid), and in a pure phase on a uniform mesh the acceleration is g.e_k exactly;
#   G6  gravity along a periodic direction is rejected (ArgumentError), a periodic mesh with g normal to the periodic direction is accepted.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-gravity/scripts/code_01_potential_density_jumps.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ForcesAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "forces_audit.jl"))
using .ForcesAudit
using TwoPhaseAMR
const FA = ForcesAudit

rng = MersenneTwister(20261116)
# G1
g1 = 0.0
for _ in 1:200
    theta = 2pi * rand(rng)
    n = (cos(theta), sin(theta))
    # an intercept for which the line crosses the square
    a = (rand(rng) - 0.5) * (abs(n[1]) + abs(n[2])) * 0.98
    p, q = FA.segment_in(FA.SQUARE, n, a)
    expected = ((p[1] + q[1]) / 2, (p[2] + q[2]) / 2)
    got = AMR._interface_centroid(n[1], n[2], a)
    global g1 = max(g1, hypot(got[1] - expected[1], got[2] - expected[2]))
end
println("G1 max |centroid - segment midpoint| = ", g1)

fixtures = Any[]
for kind in (:uniform, :amr, :cut)
    push!(fixtures, ("drop $kind", FA.drop_field(kind), (0.3, -0.9)))
end
for kind in (:uniform, :amr), solid in (:none, :gas, :liquid)
    push!(fixtures, ("plane $kind $solid", FA.plane_field(kind; gravity = (0.0, -1.0), solid), (0.0, -1.0)))
end
push!(fixtures, ("plane oblique amr none", FA.plane_field(:amr; gravity = (0.6, -0.8), solid = :none), (0.6, -0.8)))
push!(fixtures, ("plane oblique uniform gas", FA.plane_field(:uniform; gravity = (0.6, -0.8), solid = :gas), (0.6, -0.8)))

worst_pot = worst_sum = worst_jump = worst_one_phase = 0.0
range_ok = true
ncut_single = 0
for (name, field, g) in fixtures
    mesh = field.mesh
    op = FA.one_velocity_operator(field)
    work = EllipticWorkspace(op)
    force = FaceVelocity(mesh)
    gravity!(force, field, op, work; density = FA.DENSITY, gravity = g)
    J = copy(op.gradient_operator.pressure_jump)
    # G2
    phi = FA.gravity_potential_independent(field, g)
    pot = copy(work.input)
    global worst_pot = max(worst_pot, maximum(abs, pot .- phi) / maximum(abs, phi))
    # G3
    liquid, gas = AMR._current_phase_face_apertures!(field)
    a = FA.apertures(field)
    global worst_sum = max(worst_sum, maximum(abs, liquid .+ gas .- a))
    for i in eachindex(a)
        a[i] > 0 || continue
        rho = (FA.DENSITY[1] * liquid[i] + FA.DENSITY[2] * gas[i]) / a[i]
        global range_ok &= FA.DENSITY[2] - 1.0e-9 <= rho <= FA.DENSITY[1] + 1.0e-9
        if gas[i] == 0 && liquid[i] > 0
            global worst_one_phase = max(worst_one_phase, abs(rho - FA.DENSITY[1]) / FA.DENSITY[1])
            0 < a[i] < 1 && (global ncut_single += 1)
        elseif liquid[i] == 0 && gas[i] > 0
            global worst_one_phase = max(worst_one_phase, abs(rho - FA.DENSITY[2]) / FA.DENSITY[2])
            0 < a[i] < 1 && (global ncut_single += 1)
        end
    end
    # G4
    Jind = FA.gravity_jumps_independent(field, g)
    global worst_jump = max(worst_jump, maximum(abs, J .- Jind) / maximum(abs, Jind))
    println(
        rpad(name, 28), " potential gap ", round(maximum(abs, pot .- phi) / maximum(abs, phi), sigdigits = 2), "  |a_l + a_g - a| ",
        round(maximum(abs, liquid .+ gas .- a), sigdigits = 2), "  jump gap ", round(maximum(abs, J .- Jind) / maximum(abs, Jind), sigdigits = 2)
    )
end
println("G1 ", g1, "; G2 worst ", worst_pot, "; G3 worst |a_l + a_g - a| ", worst_sum, ", one-phase density deviation ", worst_one_phase, " (", ncut_single, " cut one-phase faces), range ", range_ok, "; G4 worst ", worst_jump)

# G5: conforming reductions on the uniform mesh without solid, pure phases
g5_density = 0.0; g5_accel = 0.0; npure = 0
for g in ((0.0, -1.0), (0.6, -0.8))
    field = FA.plane_field(:uniform; gravity = g, solid = :none)
    mesh = field.mesh
    op = FA.one_velocity_operator(field)
    force = FaceVelocity(mesh)
    gravity!(force, field, op, EllipticWorkspace(op); density = FA.DENSITY, gravity = g)
    J = copy(op.gradient_operator.pressure_jump)
    rho = AMR.mixture_density(field, FA.DENSITY)
    V = Float64.(mesh.face_dual_volumes)
    global g5_density = max(g5_density, maximum(abs.(force.interior .- J ./ (rho .* V)) ./ max.(abs.(force.interior), 1.0e-12)))
    for (i, f) in pairs(mesh.faces)
        cm, cp = AMR._conditional_fraction(field, f.minus), AMR._conditional_fraction(field, f.plus)
        (cm == cp && (cm == 0 || cm == 1)) || continue
        global npure += 1
        global g5_accel = max(g5_accel, abs(force.interior[i] - g[f.axis]))
    end
end
println("G5 (R J)_f = J/(rho V): ", g5_density, "; pure-phase acceleration - g.e_k: ", g5_accel, " (", npure, " faces)")

# G6: periodic rejection
periodic_mesh = BlockTreeMesh(; base_mesh = (8, 8), block_size = (4, 4), extent = (8.0, 8.0), periodic = (true, false), max_level = 0)
pf = InterfaceField(periodic_mesh, (x, y) -> y - 4.6; lipschitz = 1)
pop = PoissonOperator(pf; density = FA.DENSITY)
pwork = EllipticWorkspace(pop)
rejected = try
    gravity!(FaceVelocity(periodic_mesh), pf, pop, pwork; density = FA.DENSITY, gravity = (1.0, 0.0)); false
catch err
    err isa ArgumentError
end
accepted = try
    gravity!(FaceVelocity(periodic_mesh), pf, pop, pwork; density = FA.DENSITY, gravity = (0.0, -1.0)); true
catch
    false
end
println("G6 gravity along a periodic direction rejected: ", rejected, "; normal to it accepted: ", accepted)

@testset "gravity potential, aperture density, jumps" begin
    @test g1 <= 1.0e-14
    @test worst_pot <= 1.0e-14
    @test worst_sum <= 1.0e-12
    @test range_ok
    @test worst_one_phase <= 1.0e-12
    @test ncut_single > 0
    @test worst_jump <= 1.0e-13
    @test g5_density <= 1.0e-12
    @test npure > 100 && g5_accel <= 1.0e-12
    @test rejected && accepted
end
