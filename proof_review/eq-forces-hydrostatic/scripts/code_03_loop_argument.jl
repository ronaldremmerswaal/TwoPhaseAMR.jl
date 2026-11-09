# Code check 03: the closed-path (loop) argument of the hydrostatic paragraph of chapter 8 (new text after the audit of 2026-11-09).
#
# Label:  eq:forces:hydrostatic  (loop argument: "the jumps divided by |f| add up, along a closed path of faces from a mixed cell c through liquid
#         cells to a mixed cell c' and back through gas cells, to (rho_l - rho_g)(phi_c' - phi_c)")
# Source: docs/techref/chapters/08_forces.tex:498-504
# Premise P: every open face with a pure neighbour lies in one phase (phase aperture zero for the other), so rho_a_f is that phase's density.
# (the two paths must each contain at least one pure cell; a direct face between two mixed cells is not part of the argument)
# Then, with J_f the code's gravity jumps, on any path of open faces c = c_0, c_1, ..., c_m = c' whose interior cells are pure liquid,
#     sum_k J_{f_k}/|f_k| (oriented) = rho_l (phi_c' - phi_c),   and on one with pure gas cells  rho_g (phi_c' - phi_c);
# the closed path liquid-out / gas-back therefore sums to (rho_l - rho_g)(phi_c' - phi_c).  For the jumps of a cell function the sum is zero.
# Checks, on the fixtures of code_01/code_02 (balanced and not balanced; 40 all pairs of mixed cells (at most 4000 random ones), seeded):
#   L1  premise P holds on every fixture used (count of open faces with a pure neighbour and both phase apertures positive);
#   L2  the loop sum of the code's J equals (rho_l - rho_g)(phi_c' - phi_c) to rounding (the telescoping identity itself);
#   L2' loops out through the pure cells of one phase and back through the direct face between two adjacent mixed cells: the sum is
#       (rho_pi - rho_a,f)(phi_c' - phi_c)  (rho_a,f the mixture density of that mixed-mixed face);
#   L3  balanced fixtures: all loop sums vanish and all mixed-cell potentials joined by paths coincide; unbalanced fixtures: the loops of the text vanish (they do not
#       reach the offending cells); triple cells: a loop through a mixed-mixed face is nonzero (proof of non-existence, independent of any solver); oblique
#       planes next to a solid: premise P fails at one face each;
#   L4  the converse direction is NOT implied: equal potentials of the mixed cells is what the loop argument gives; it is the plane x_0 only through
#       the explicit a (the PLIC need not lie on the plane g.(x - x_0) = 0 itself, any common potential level balances).
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-hydrostatic/scripts/code_03_loop_argument.jl

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

function loop_analysis(field, g, rng; npairs = 4000)
    mesh = field.mesh
    nx, ny = mesh.block_size
    J = code_jumps(field, g)
    a = FA.apertures(field)
    liquid, gas = AMR._current_phase_face_apertures!(field)
    phi = FA.gravity_potential_independent(field, g)
    ncell = length(phi)
    C = FA.cell_vector(ref -> AMR._conditional_fraction(field, ref), mesh)
    kind(c) = C[c] >= 1 ? :liquid : C[c] <= 0 ? :gas : :mixed
    # premise P and adjacency of open faces
    violations = 0; wrong_liquid = 0; wrong_gas = 0
    usable = trues(length(a))
    adj = [Tuple{Int, Int, Float64}[] for _ in 1:ncell]       # (neighbour, face, orientation sign of moving into neighbour)
    for (i, f) in pairs(mesh.faces)
        a[i] > 0 || continue
        m, p = AMR._cell_index(f.minus, nx, ny), AMR._cell_index(f.plus, nx, ny)
        km, kp = kind(m), kind(p)
        tol = 1.0e-12 * a[i]
        # premise P: a face with a pure LIQUID neighbour has no gas aperture, with a pure GAS neighbour no liquid aperture
        # (faces between a pure liquid and a pure gas cell, the grid-line case, are not on the paths)
        if (km == :liquid || kp == :liquid) && !(km == :gas || kp == :gas) && gas[i] > tol
            violations += 1; wrong_liquid += 1; usable[i] = false
        elseif (km == :gas || kp == :gas) && !(km == :liquid || kp == :liquid) && liquid[i] > tol
            violations += 1; wrong_gas += 1; usable[i] = false
        end
        usable[i] && (push!(adj[m], (p, i, +1.0)); push!(adj[p], (m, i, -1.0)))   # faces violating P are not used by the argument
    end
    function path_sum(src, dst, phase)
        # BFS over open faces; interior cells must be of `phase`; returns sum of oriented J/|f| or nothing
        prev = Dict{Int, Tuple{Int, Int, Float64}}(src => (0, 0, 0.0))
        queue = [src]
        while !isempty(queue)
            u = popfirst!(queue)
            for (v, i, s) in adj[u]
                haskey(prev, v) && continue
                (v == dst && u != src) || kind(v) == phase || continue      # at least one interior pure cell: no direct face between c and c'
                prev[v] = (u, i, s)
                v == dst && (queue = Int[]; break)
                push!(queue, v)
            end
            haskey(prev, dst) && break
        end
        haskey(prev, dst) || return nothing
        total = 0.0; v = dst
        while v != src
            u, i, s = prev[v]
            total += s * J[i] / mesh.faces[i].area
            v = u
        end
        return total
    end
    mixed = findall(c -> kind(c) == :mixed, 1:ncell)
    worst_identity = 0.0; max_loop = 0.0; n_loops = 0; max_dphi = 0.0
    scale = FA.DENSITY[1] * hypot(g...) * 0.25
    pairs_all = [(c, c2) for c in mixed for c2 in mixed if c < c2]
    length(pairs_all) > npairs && (pairs_all = pairs_all[randperm(rng, length(pairs_all))[1:npairs]])
    for (c, c2) in pairs_all
        sl = path_sum(c, c2, :liquid); sg = path_sum(c, c2, :gas)
        (sl === nothing || sg === nothing) && continue
        n_loops += 1
        loop = sl - sg
        predicted = (FA.DENSITY[1] - FA.DENSITY[2]) * (phi[c2] - phi[c])
        worst_identity = max(worst_identity, abs(loop - predicted) / scale)
        max_loop = max(max_loop, abs(loop) / scale)
        max_dphi = max(max_dphi, abs(phi[c2] - phi[c]))
    end
    # second kind of closed path: out through the pure cells of one phase, back through the DIRECT face between the two adjacent mixed cells;
    # that face carries rho_a (its phase mixture), so the sum is (rho_pi - rho_a)(phi_c' - phi_c)
    n2 = 0; identity2 = 0.0; max_loop2 = 0.0
    for u in mixed, (v, i, s) in adj[u]
        (kind(v) == :mixed && u < v) || continue
        rho_a = (FA.DENSITY[1] * liquid[i] + FA.DENSITY[2] * gas[i]) / a[i]
        direct = s * J[i] / mesh.faces[i].area            # oriented u -> v
        for (phase, rho) in ((:liquid, FA.DENSITY[1]), (:gas, FA.DENSITY[2]))
            sp = path_sum(u, v, phase)
            sp === nothing && continue
            n2 += 1
            loop = sp - direct
            identity2 = max(identity2, abs(loop - (rho - rho_a) * (phi[v] - phi[u])) / scale)
            max_loop2 = max(max_loop2, abs(loop) / scale)
        end
    end
    return (; violations, wrong_liquid, wrong_gas, n_loops, worst_identity, max_loop, max_dphi, n2, identity2, max_loop2)
end

rng = MersenneTwister(20261120)
g0 = (0.0, -1.0); g1 = (0.6, -0.8)
cases = Any[
    ("uniform none vertical", FA.plane_field(:uniform; gravity = g0, solid = :none), g0, true),
    ("amr gas vertical", FA.plane_field(:amr; gravity = g0, solid = :gas), g0, true),
    ("amr liquid (submerged solid) vertical", FA.plane_field(:amr; gravity = g0, solid = :liquid), g0, true),
    ("amr none oblique", FA.plane_field(:amr; gravity = g1, solid = :none), g1, true),
    ("uniform liquid oblique + solid", FA.plane_field(:uniform; gravity = g1, solid = :liquid), g1, false),
    ("amr liquid oblique + solid", FA.plane_field(:amr; gravity = g1, solid = :liquid), g1, false),
    ("uniform gas oblique + solid", FA.plane_field(:uniform; gravity = g1, solid = :gas), g1, false),
    ("triple cells (y = 1.2 meets the sloped solid)", FA.plane_field(:uniform; gravity = g0, solid = :liquid, point = (4.3, 1.2)), g0, false),
]
results = []
for (name, field, g, balanced) in cases
    r = loop_analysis(field, g, rng)
    push!(results, (; name, balanced, r...))
    println(rpad(name, 48), " premise violations ", r.violations, " (gas aperture next to pure liquid ", r.wrong_liquid, ", liquid aperture next to pure gas ", r.wrong_gas, "), loops ", r.n_loops, ", identity gap ", round(r.worst_identity, sigdigits = 2),
        ", max |loop| ", round(r.max_loop, sigdigits = 2), ", max |phi_c' - phi_c| ", round(r.max_dphi, sigdigits = 2),
        "\n    loops through a mixed-mixed face: ", r.n2, ", identity gap ", round(r.identity2, sigdigits = 2), ", max |loop| ", round(r.max_loop2, sigdigits = 2))
end
# L4: a PLIC displaced coherently (all mixed cells of the horizontal interface moved by the same physical distance, still inside their cells):
# the points of the mixed cells lie at one common potential level, not on the plane through x_0 -- the state is still balanced
shifted = let
    g = g0
    field = FA.plane_field(:uniform; gravity = g, solid = :none)
    delta = 0.05
    for ref in AMR._cell_references(field.mesh)
        c = AMR._conditional_fraction(field, ref)
        0 < c < 1 || continue
        h = field.mesh.blocks[ref.block].spacing[2]
        AMR._set!(field.intercept, AMR._get(field.intercept, ref) + delta / h * sign(AMR._get(field.normal_y, ref)), ref)
    end
    J = code_jumps(field, g)
    cons = FA.jump_consistency(FA.jump_matrix(field.mesh), J, findall(>(0), FA.apertures(field)))
    phi = FA.gravity_potential_independent(field, g)
    mixed_phi = [phi[AMR._cell_index(ref, field.mesh.block_size...)] for ref in AMR._cell_references(field.mesh) if 0 < AMR._conditional_fraction(field, ref) < 1]
    offset_from_plane = maximum(abs.(mixed_phi .- (g[1] * 4.3 + g[2] * 4.6)))
    (; cons, spread = maximum(mixed_phi) - minimum(mixed_phi), offset_from_plane)
end
println("L4 coherently displaced PLIC (0.05): consistency ", shifted.cons, ", spread of the mixed-cell potentials ", shifted.spread, ", offset from the plane through x_0 ", shifted.offset_from_plane)

@testset "loop argument of the hydrostatic paragraph" begin
    @test shifted.cons <= 1.0e-12 && shifted.spread <= 1.0e-12 && shifted.offset_from_plane > 1.0e-3
    for r in results
        r.balanced && @test r.violations == 0                       # L1 (the unbalanced oblique cases have one face that violates P; it is not used)
        @test r.n_loops > 0
        @test r.worst_identity <= 1.0e-10             # L2
        @test r.identity2 <= 1.0e-10                  # L2': (rho_pi - rho_a)(phi_c' - phi_c) for the loops through a mixed-mixed face
        if r.balanced
            @test r.max_loop <= 1.0e-10 && r.max_dphi <= 1.0e-12     # L3
        else
            # L3: the loops of the text (liquid out, gas back) are ALL ZERO here -- the offending cells are not joined by them. Non-existence of a cell
            # function is proved either by a nonzero loop through a mixed-mixed face (triple cells: 0.05) or, for the oblique planes next to a solid, the
            # failure sits at the one face that violates P (a gas aperture next to a pure liquid cell), which the argument cannot use.
            @test r.max_loop <= 1.0e-10
            @test r.max_loop2 > 1.0e-6 || r.violations >= 1
        end
    end
end
