# Code check 02: configurations at the edge of the hydrostatic statement -- interface on a grid line, interface meeting the sloped
# solid (triple cells), and the gravity density of the code before issue #74.
#
# Label:  eq:forces:hydrostatic  (eq:forces:aperture-density; "Until issue 74 the code used ...")
# Source: docs/techref/chapters/08_forces.tex:399-467; thesis Lemma 7.1, proof case 2 (App. 7.A)
# Cases (vertical g, densities (1000, 1)):
#   G1  interface exactly on the grid line y = 4.5 (uniform and composite): faces between a pure liquid and a pure gas cell
#       (the thesis proof's case 2 does not cover them); consistency residual and the phase apertures of those faces;
#   G2  interface on a grid line that is a horizontal level transition (coarse liquid cell below, fine gas cells above): still balanced (no closed
#       loop crosses the interface twice with different jumps), but the half-half phase apertures give a jump 0.750375 of the one of the continuous p^dagger
#       ((rho_l + rho_g)(h1 + h2)/2 against rho_l h1 + rho_g h2 over 2): the pressure has a step at the interface; with equal widths the ratio is exactly 1;
#   T1  horizontal interface y = 1.2 meeting the sloped solid below (solid in the liquid): triple cells; embedded contact angle pi/2
#       (default) and the geometrically matching angle atan(0.1): consistency residual and centroid offset;
#   O1  the pre-#74 density rho_l a_l + rho_g (1 - a_l): consistency residual on the liquid-solid fixtures (reproduces the
#       failure the text attributes to it), against the open-face mixture (consistent);
#   P1  the potential point is free: any point of a pure cell and any point of the PLIC segment in a mixed cell give a balanced planar state
#       (deviation box dev:forces:gravity-cut: "does not depend on this choice");
#   O2  the code's jumps equal the independent jumps of eq:forces:gravity-jump (mode :open) on every fixture used here.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-hydrostatic/scripts/code_02_special_configurations.jl

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
consistency(field, J) = FA.jump_consistency(FA.jump_matrix(field.mesh), J, findall(>(0), FA.apertures(field)))
function centroid_offset(field, g, point)
    phi = FA.gravity_potential_independent(field, g)
    mesh = field.mesh
    return maximum(abs(phi[AMR._cell_index(ref, mesh.block_size...)] - (g[1] * point[1] + g[2] * point[2])) / hypot(g...)
        for ref in AMR._cell_references(mesh) if 0 < AMR._conditional_fraction(field, ref) < 1; init = 0.0)
end

g = (0.0, -1.0)
# G1
grid = Dict()
for kind in (:uniform, :amr)
    point = (4.3, 4.5)
    field = FA.plane_field(kind; gravity = g, solid = :none, point)
    J = code_jumps(field, g)
    cons = consistency(field, J)
    liquid, gas = AMR._current_phase_face_apertures!(field)
    pure_pairs = [i for (i, f) in pairs(field.mesh.faces) if
        AMR._conditional_fraction(field, f.minus) == 1 && AMR._conditional_fraction(field, f.plus) == 0]
    apertures = isempty(pure_pairs) ? nothing : (extrema(liquid[pure_pairs]), extrema(gas[pure_pairs]))
    grid[kind] = (; cons, n = length(pure_pairs), apertures)
    println("G1 $kind: interface on y = 4.5: pure-liquid/pure-gas faces ", length(pure_pairs), "; (liquid, gas) aperture ranges ", apertures,
        "; consistency residual ", cons)
end
# G2: interface on a grid line that is a HORIZONTAL level transition (upper half refined): pure liquid cell below (width h1), pure gas cell above (h2 = h1/2)
refine_upper = (key, origin, extent) -> key.level == 0 && key.j >= 2
mesh_transition = BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (FA.BOX, FA.BOX), periodic = (false, false), max_level = 1, refine = refine_upper)
transition = Dict()
for y0 in (4.0, 4.5)           # y0 = 4.0: on the transition line; 4.5: on a fine-mesh grid line (control, equal cells)
    field = FA.plane_field(:amr; gravity = g, solid = :none, point = (4.3, y0), mesh = mesh_transition)
    J = code_jumps(field, g)
    cons = consistency(field, J)
    liquid, gas = AMR._current_phase_face_apertures!(field)
    pure_pairs = [i for (i, f) in pairs(field.mesh.faces) if
        AMR._conditional_fraction(field, f.minus) == 1 && AMR._conditional_fraction(field, f.plus) == 0]
    widths = unique(sort([(field.mesh.blocks[field.mesh.faces[i].minus.block].spacing[2], field.mesh.blocks[field.mesh.faces[i].plus.block].spacing[2]) for i in pure_pairs]))
    # ratio of the code's jump to the jump that p^dagger (continuous at the interface) would need: rho_l h1/2 + rho_g h2/2
    ratios = [J[i] / (field.mesh.faces[i].area * (FA.DENSITY[1] * field.mesh.blocks[field.mesh.faces[i].minus.block].spacing[2] / 2 +
        FA.DENSITY[2] * field.mesh.blocks[field.mesh.faces[i].plus.block].spacing[2] / 2)) for i in pure_pairs]
    transition[y0] = (; cons, n = length(pure_pairs), widths, ratio = extrema(abs.(ratios)), apertures = (extrema(liquid[pure_pairs]), extrema(gas[pure_pairs])))
    println("G2 interface on y = $y0 (horizontal level transition at y = 4): pure-liquid/pure-gas faces ", length(pure_pairs), ", (h_below, h_above) ", widths,
        ", (liquid, gas) apertures ", transition[y0].apertures, ", consistency residual ", cons, ", jump / jump of the continuous p^dagger ", transition[y0].ratio)
end
# T1
triple = Dict()
for (label, contact) in (("pi/2", pi / 2), ("atan(0.1)", atan(0.1)), ("pi-atan(0.1)", pi - atan(0.1)))
    point = (4.3, 1.2)
    field = FA.plane_field(:uniform; gravity = g, solid = :liquid, point, contact)
    J = code_jumps(field, g)
    triple[label] = (; cons = consistency(field, J), off = centroid_offset(field, g, point))
    println("T1 horizontal interface y = 1.2 meeting the sloped solid, embedded contact angle ", label, ": consistency ", triple[label].cons,
        ", centroid offset ", triple[label].off)
end
# O1, O2
old = Dict(); agree = 0.0
for kind in (:uniform, :amr)
    field = FA.plane_field(kind; gravity = g, solid = :liquid)
    Jc = code_jumps(field, g)
    Jo = FA.gravity_jumps_independent(field, g)
    Jold = FA.gravity_jumps_independent(field, g; mode = :old)
    global agree = max(agree, maximum(abs, Jc .- Jo) / maximum(abs, Jo))
    liquid, gas = AMR._current_phase_face_apertures!(field)
    a = FA.apertures(field)
    old[kind] = (; open = consistency(field, Jo), old = consistency(field, Jold),
        differing = count(i -> abs(Jo[i] - Jold[i]) > 1.0e-9 * max(abs(Jo[i]), 1.0), eachindex(Jo)),
        cut_liquid = count(i -> 0 < a[i] < 1 && gas[i] == 0 && liquid[i] > 0, eachindex(Jo)))
    println("O1 $kind solid in liquid: consistency with open-face mixture ", old[kind].open, ", pre-#74 density ", old[kind].old,
        "; faces whose jump differs: ", old[kind].differing, " (cut liquid faces ", old[kind].cut_liquid, ")")
end
println("O2 max |J_code - J_independent| / max|J| = ", agree)
# P1: the potential point is free (deviation box dev:forces:gravity-cut): any point of a pure cell, any point of the PLIC segment in a mixed cell
rng = MersenneTwister(20261118)
free_point = Dict()
for kind in (:uniform, :amr), solid in (:none, :gas, :liquid)
    field = FA.plane_field(kind; gravity = g, solid)
    worst = 0.0
    for draw in 1:3
        phi = FA.gravity_potential_random_points(field, g, rng)
        worst = max(worst, consistency(field, FA.gravity_jumps_for_potential(field, phi)))
    end
    free_point[(kind, solid)] = worst
    println("P1 $kind $solid: random potential points, consistency residual ", worst)
end

@testset "special hydrostatic configurations" begin
    @test agree <= 1.0e-13
    for kind in (:uniform, :amr)
        @test old[kind].open <= 1.0e-12
    end
    @test old[:amr].old > 1.0e-6                    # the pre-#74 density is not balanced next to the sloped solid (composite mesh)
    @test all(v -> v <= 1.0e-12, values(free_point))      # the balance does not depend on the potential point (planar state)
    @test all(t -> t.n > 0, values(grid))
    @test transition[4.5].cons <= 1.0e-12                    # control: equal cells on both sides
    @test transition[4.0].cons <= 1.0e-12                    # still balanced at a horizontal level transition ...
    expected_ratio = sum(FA.DENSITY) * (0.5 + 0.25) / (2 * (FA.DENSITY[1] * 0.5 + FA.DENSITY[2] * 0.25))     # (rho_l + rho_g)(h1 + h2) / (2 (rho_l h1 + rho_g h2)) = 0.750375
    @test all(r -> abs(r - expected_ratio) <= 1.0e-12, transition[4.0].ratio) && all(r -> abs(r - 1) <= 1.0e-12, transition[4.5].ratio)   # ... but the pressure is not the continuous p^dagger
    @test all(t -> t.cons <= 1.0e-12, values(grid))          # interface on a grid line: balanced (phase apertures 1/2, 1/2)
    @test all(t -> t.cons > 1.0e-6 && t.off > 1.0e-6, values(triple))    # triple cells: PLIC off the plane, not balanced
    @test length(unique(round(t.cons, sigdigits = 8) for t in values(triple))) == 1   # independent of the embedded angle
end
