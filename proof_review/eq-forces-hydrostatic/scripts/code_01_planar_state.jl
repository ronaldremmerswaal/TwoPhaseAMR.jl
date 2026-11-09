# Code check 01: hydrostatic condition (eq:forces:hydrostatic) for the planar interface normal to g -- the explicit pressure of
# Lemma 7.1 balances the gravity jumps of the code on every face, without solid and next to a sloped solid in either phase.
#
# Label:  eq:forces:hydrostatic  (eq:forces:steady (b), eq:forces:potential, eq:forces:aperture-density, eq:forces:gravity-jump)
# Source: docs/techref/chapters/08_forces.tex:440-467; src/two_phase/forces.jl (gravity!); thesis Lemma 7.1 (7.65), App. 7.A
# Cell function: a_c = rho^pi g.(x_c - x0) in a pure cell of phase pi, a_c = 0 in a mixed cell (conditional fraction in (0,1)).
# Checks (uniform and composite meshes of the drop fixtures, densities (1000, 1), planar interface through (4.3, 4.6)):
#   H1  vertical g, no solid / solid in the gas / solid in the liquid: max_f |J_f - (D a)_f| / (rho_a |f| |g| h) over the faces with a_f > 0,
#       with J the jumps the code writes (gravity!), in units of the acceleration error;
#   H2  the same without solid for random tilts of g (angles in (0, 35 deg), planes meeting only the vertical walls) and random offsets;
#   H5  a plane that meets the bottom wall: PLIC not exact in the corner, centroid offset and consistency residual both > 0;
#   H3  least-squares consistency of J (relative residual of  J_S = D_S a  over a), independent of the explicit a;
#   H4  oblique g and a solid: the claim holds exactly when the interface cells' centroids lie on the plane (q = g.(x^I - x0) = 0);
#       where the PLIC is not exact (cells whose height stencil meets the solid) both the centroid offset and the residual are > 0.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-hydrostatic/scripts/code_01_planar_state.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ForcesAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "forces_audit.jl"))
using .ForcesAudit
using TwoPhaseAMR
const FA = ForcesAudit

function code_jumps(field, g)
    op = FA.one_velocity_operator(field)
    gravity!(FaceVelocity(field.mesh), field, op, EllipticWorkspace(op); density = FA.DENSITY, gravity = g)
    return copy(op.gradient_operator.pressure_jump)
end

"""Acceleration-scaled error of J against D a on the faces with aperture > 0."""
function acceleration_error(field, J, avals, g)
    mesh = field.mesh
    D = FA.jump_matrix(mesh)
    a = FA.apertures(field)
    liquid, gas = AMR._current_phase_face_apertures!(field)
    worst = 0.0
    r = J .- D * avals
    for (i, face) in pairs(mesh.faces)
        a[i] > 0 || continue
        rho = (FA.DENSITY[1] * liquid[i] + FA.DENSITY[2] * gas[i]) / a[i]
        h = max(mesh.blocks[face.minus.block].spacing[face.axis], mesh.blocks[face.plus.block].spacing[face.axis])
        worst = max(worst, abs(r[i]) / (rho * face.area * hypot(g...) * h))
    end
    return worst
end

centroid_offset(field, g, point) = begin
    phi = FA.gravity_potential_independent(field, g)
    mesh = field.mesh
    worst = 0.0
    for ref in AMR._cell_references(mesh)
        c = AMR._conditional_fraction(field, ref)
        0 < c < 1 || continue
        worst = max(worst, abs(phi[AMR._cell_index(ref, mesh.block_size...)] - (g[1] * point[1] + g[2] * point[2])) / hypot(g...))
    end
    worst
end

rows = []
# H1: vertical gravity, solids
for kind in (:uniform, :amr), solid in (:none, :gas, :liquid)
    g = (0.0, -1.0); point = (4.3, 4.6)
    field = FA.plane_field(kind; gravity = g, solid, point)
    J = code_jumps(field, g)
    avals = FA.hydrostatic_cell_function(field, g, point)
    err = acceleration_error(field, J, avals, g)
    D = FA.jump_matrix(field.mesh)
    support = findall(>(0), FA.apertures(field))
    cons = FA.jump_consistency(D, J, support)
    off = centroid_offset(field, g, point)
    push!(rows, (; kind, solid, g, err, cons, off))
    println(rpad("H1 $kind $solid vertical", 28), " acceleration error ", round(err, sigdigits = 3), "  consistency ", round(cons, sigdigits = 3), "  centroid offset/h-units ", round(off, sigdigits = 3))
end
# H2: random tilts, no solid; planes that meet a horizontal wall are excluded (see H5)
rng = MersenneTwister(20261113)
worst_tilt = 0.0; worst_tilt_cons = 0.0; accepted = 0
for kind in (:uniform, :amr), draw in 1:6
    theta = deg2rad(35 * rand(rng)); g = (sin(theta), -cos(theta))
    point = (3.0 + 2rand(rng), 3.5 + 2rand(rng))
    y0 = point[2] - tan(theta) * point[1]; y8 = point[2] + tan(theta) * (8 - point[1])
    (1 <= y0 <= 7 && 1 <= y8 <= 7) || continue
    global accepted += 1
    field = FA.plane_field(kind; gravity = g, solid = :none, point)
    J = code_jumps(field, g)
    avals = FA.hydrostatic_cell_function(field, g, point)
    err = acceleration_error(field, J, avals, g)
    D = FA.jump_matrix(field.mesh)
    cons = FA.jump_consistency(D, J, collect(1:length(J)))
    global worst_tilt = max(worst_tilt, err); global worst_tilt_cons = max(worst_tilt_cons, cons)
end
println("H2 random tilts (", accepted, " planes, no solid): max acceleration error ", worst_tilt, ", consistency ", worst_tilt_cons)
# H5: a plane through the corner region meets the bottom wall: PLIC not exact there, the state is not balanced
corner = let theta = deg2rad(37.8), g = (sin(deg2rad(37.8)), -cos(deg2rad(37.8))), point = (4.9, 3.79)
    field = FA.plane_field(:uniform; gravity = g, solid = :none, point)
    J = code_jumps(field, g)
    cons = FA.jump_consistency(FA.jump_matrix(field.mesh), J, collect(1:length(J)))
    off = centroid_offset(field, g, point)
    println("H5 plane meeting the bottom wall at x = 0: consistency ", cons, ", centroid offset ", off)
    (; cons, off)
end
# H4: oblique gravity with a solid
oblique = []
for kind in (:uniform, :amr), solid in (:gas, :liquid)
    g = (0.6, -0.8); point = (4.3, 4.6)
    field = FA.plane_field(kind; gravity = g, solid, point)
    J = code_jumps(field, g)
    D = FA.jump_matrix(field.mesh)
    cons = FA.jump_consistency(D, J, findall(>(0), FA.apertures(field)))
    off = centroid_offset(field, g, point)
    push!(oblique, (; kind, solid, cons, off))
    println(rpad("H4 $kind $solid oblique", 28), " consistency ", round(cons, sigdigits = 3), "  max centroid offset ", round(off, sigdigits = 3))
end

@testset "hydrostatic state of the planar interface" begin
    for r in rows
        @test r.err <= 1.0e-12
        @test r.cons <= 1.0e-12
    end
    @test accepted >= 6
    @test worst_tilt <= 1.0e-12
    @test worst_tilt_cons <= 1.0e-12
    @test (corner.off <= 1.0e-12) == (corner.cons <= 1.0e-12) && corner.cons > 1.0e-6
    # Lemma 7.1 as stated needs the interface centroids on the plane: exact <=> offset zero (both directions over all cases)
    for r in oblique
        @test (r.off <= 1.0e-12) == (r.cons <= 1.0e-12)
    end
    @test any(r -> r.off > 1.0e-6 && r.cons > 1.0e-6, oblique)      # the failure exists and is tied to the PLIC offset
end
