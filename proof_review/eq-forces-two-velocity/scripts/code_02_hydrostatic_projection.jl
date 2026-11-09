# Code check 02: hydrostatic rest state through the two-velocity projections (issue #76), projection only: both phase velocities are set to
# dt * R J_g (the increment of eq:forces:two-velocity) and projected; the maximum remaining phase velocity / (dt |g|) is reported.
#
# Label:  eq:forces:two-velocity  (issue #76)
# Source: docs/techref/chapters/08_forces.tex:520-574; src/two_phase/projection/cmdgfm_projection.jl, mimetic_projection.jl, common.jl
# Cases: uniform / composite mesh x {no solid, solid in the gas, solid in the liquid}, planar interface through (4.3, 4.6), g = (0, -1), densities (1000, 1).
#   map = :one   the increment of the code (one-velocity map R^1, coefficient a^2/rho);
#   map = :tv    counterfactual: the same jumps J_g through the two-velocity map R^tv (coefficient a/rho), i.e. gravity!/surface_tension! called
#                with the two-velocity pressure operator -- NOT what the code does, it isolates the cause.
#   P1  CMDGFM, map one: balanced (<= 1e-8) iff E = R^1 J - R^tv(D p*) = 0 (code_01), fails by 3e-2 on the composite mesh with a solid;
#   P2  CMDGFM, map tv: balanced to solver tolerance in every case, so the coefficient mismatch is the only cause for CMDGFM;
#   P3  mimetic, default :joint_smoothed: never balanced, with either map, even without solid (0.0175 uniform, 0.66 composite; on the uniform mesh
#       all residual faces touch a mixed cell); variant :joint_face: balanced to solver tolerance on the uniform mesh without solid, 0.09 on the composite one;
#   P4  the same projections for an interface exactly on the grid line y = 4.5 (no mixed cell): CMDGFM balanced, default mimetic NOT (~1 and ~1.8).
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-two-velocity/scripts/code_02_hydrostatic_projection.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ForcesAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "forces_audit.jl"))
using .ForcesAudit
using TwoPhaseAMR
const FA = ForcesAudit

dt = 0.01
g = (0.0, -1.0)

function phase_increment(field, which)
    mesh = field.mesh
    op = which == :one ? FA.one_velocity_operator(field) : FA.two_velocity_operator(field)
    acc = FaceVelocity(mesh)
    gravity!(acc, field, op, EllipticWorkspace(op); density = FA.DENSITY, gravity = g)
    phases = (FaceVelocity(mesh), FaceVelocity(mesh))
    for ph in phases
        ph.interior .= dt .* acc.interior
    end
    AMR._mark_inactive_phase_velocities!(phases, field)
    return phases
end

function project_two_velocity(field, phases, projection; variant = :joint_smoothed)
    mesh = field.mesh
    if projection == :cmdgfm
        solver = CMDGFMProjectionSolver(field; density = FA.DENSITY)
        stats = project!(phases, CellField(mesh), solver, dt, CMDGFMProjectionWorkspace(solver); tolerance = 1.0e-12, max_iterations = 4000)
    else
        solver = MimeticProjectionSolver(field; density = FA.DENSITY, variant)
        stats = project!(phases, CellField(mesh), solver, dt, MimeticProjectionWorkspace(solver); tolerance = 1.0e-12, max_iterations = 4000)
    end
    stats.converged || error("two-velocity projection did not converge")
    return phases
end

leftover(phases) = FA.maxabs(phases) / dt

function mixed_adjacent(field, i)
    f = field.mesh.faces[i]
    return any(ref -> 0 < AMR._conditional_fraction(field, ref) < 1, (f.minus, f.plus))
end

function localisation(field, phases)
    worst = leftover(phases)
    big = Int[]
    for ph in 1:2, i in eachindex(phases[ph].interior)
        v = phases[ph].interior[i]
        isnan(v) && continue
        abs(v) / dt > 1.0e-3 * worst && push!(big, i)
    end
    big = unique(big)
    return (; n = length(big), mixed = count(i -> mixed_adjacent(field, i), big))
end

rows = []
for kind in (:uniform, :amr), solid in (:none, :gas, :liquid), projection in (:cmdgfm, :mimetic), which in (:one, :tv)
    solid == :none && which == :tv && continue
    field = FA.plane_field(kind; gravity = g, solid)
    phases = phase_increment(field, which)
    project_two_velocity(field, phases, projection)
    r = leftover(phases)
    loc = r > 1.0e-3 ? localisation(field, phases) : (; n = 0, mixed = 0)
    push!(rows, (; kind, solid, projection, which, r, loc))
    println(rpad("$kind $solid $projection $which", 30), " leftover/(dt|g|) = ", round(r, sigdigits = 3), "   residual faces (>1e-3 max): ", loc.n, ", adjacent to a mixed cell: ", loc.mixed)
end
# P3 variant :joint_face
variants = []
for kind in (:uniform, :amr)
    field = FA.plane_field(kind; gravity = g, solid = :none)
    phases = phase_increment(field, :one)
    project_two_velocity(field, phases, :mimetic; variant = :joint_face)
    push!(variants, (; kind, r = leftover(phases)))
    println("P3 mimetic :joint_face, $kind, no solid: ", variants[end].r)
end
# P4 grid-aligned interface
aligned = []
for kind in (:uniform, :amr), projection in (:cmdgfm, :mimetic)
    field = FA.plane_field(kind; gravity = g, solid = :none, point = (4.3, 4.5))
    phases = phase_increment(field, :one)
    project_two_velocity(field, phases, projection)
    push!(aligned, (; kind, projection, r = leftover(phases)))
    println("P4 interface on the grid line y = 4.5, $kind, $projection: ", aligned[end].r)
end

@testset "hydrostatic state through the two-velocity projections" begin
    for r in rows
        if r.projection == :cmdgfm && r.which == :tv
            @test r.r <= 1.0e-8                                           # P2
        elseif r.projection == :cmdgfm && r.which == :one
            expected_failure = r.kind == :amr && r.solid != :none          # P1 (issue #76)
            @test (r.r > 1.0e-3) == expected_failure
            expected_failure || @test r.r <= 1.0e-8
        else
            @test r.r > 1.0e-3                                            # P3: mimetic never balanced
        end
    end
    for v in variants                                                 # P3: :joint_face is balanced on the uniform mesh, not on the composite one
        @test (v.r > 1.0e-3) == (v.kind == :amr)
    end
    for a in aligned                                                  # P4: the default mimetic projection is worse on a grid-aligned interface
        a.projection == :mimetic && @test a.r > 0.1
    end
    for a in aligned
        a.projection == :cmdgfm && @test a.r <= 1.0e-8
    end
end
