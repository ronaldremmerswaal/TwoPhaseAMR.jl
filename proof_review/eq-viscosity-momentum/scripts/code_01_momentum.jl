# Code check 01: eq:viscosity:momentum -- conservation of momentum of the two-phase midpoint solve.
#
# Label:  eq:viscosity:momentum
# Source: docs/techref/chapters/09_viscosity.tex:389-407; src/two_phase/diffusion/{variable_diffusion_geometry,variable_diffusion_solver}.jl
# Checks, on random periodic 2:1 composite meshes with random interfaces (levels 1 and 2, five draws each, mu = (0.1,0.002)):
#   M1  S 1_k = 0 for both components (the claim "a constant component has no strain"), to round-off relative to max|S|;
#   M2  sum_f rho_f V_f (u2 - u1)_f = 1_k' r   with r = A u2 - rhs (loose and exact solve), to round-off,
#   M3  the same for the two-velocity combined momentum is in step03_two_velocity (not here).
# Scope of the hypothesis "periodic mesh without solid" (sharpness experiments, same file):
#   M4  with a solid S 1_k != 0 (momentum exchanged with the solid): size,
#   M5  walls: x-periodic channel with y walls: S 1_1 != 0 for NoSlip but = 0 for FreeSlip (tangential momentum conserved),
#       S 1_2 != 0 for both (boundary normal faces are omitted from the normal rows),
#   M6  fully walled box: S 1_k != 0 for both wall types.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-viscosity-momentum/scripts/code_01_momentum.jl
using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ViscosityAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "viscosity_audit.jl"))
using .ViscosityAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR
const MA = ViscosityAudit.MA

component_ones(mesh, k) = [face.axis == k ? 1.0 : 0.0 for face in mesh.faces]
relmax(S, v) = maximum(abs, S * v) / maximum(abs, nonzeros(S))

@testset "M1/M2: periodic composite meshes" begin
    rng = MersenneTwister(5150)
    worst_const = 0.0; worst_mom = 0.0; ncase = 0
    for levels in (1, 2), trial in 1:5
        field, _, _ = MA.random_two_phase(rng; periodic = (true, true), levels)
        mesh = field.mesh
        op = StaggeredDiffusionOperator(field; viscosity = FIX_VISC)
        AMR._update_dynamic_viscosity!(op, field, FIX_VISC)
        density = mixture_density(field, FIX_DENSITY)
        M = density .* op.volumes
        dt = 0.05
        solver = AMR.VariableStaggeredDiffusionSolver(op, density, dt)
        u1 = randn(rng, length(mesh.faces))
        src = FaceVelocity(mesh); src.interior .= u1
        for tol in (1.0e-12, 1.0e-3)
            dst = FaceVelocity(mesh)
            implicit_midpoint_diffusion!(dst, src, op, dt, solver; density, tolerance = tol, max_cycles = 4000)
            u2 = dst.interior
            r = M .* (u2 .- u1) .+ dt / 2 .* (op.stiffness * (u1 .+ u2))
            for k in 1:2
                one_k = component_ones(mesh, k)
                worst_const = max(worst_const, relmax(op.stiffness, one_k))
                change = dot(one_k, M .* (u2 .- u1))
                scale = dot(one_k, M .* abs.(u1))
                worst_mom = max(worst_mom, abs(change - dot(one_k, r)) / scale)
                ncase += 1
            end
        end
    end
    println("periodic: $ncase cases, max |S 1_k|/max|S| = ", worst_const, ", max momentum identity defect (rel.) = ", worst_mom)
    @test worst_const <= 1.0e-13
    @test worst_mom <= 1.0e-12
end

@testset "M4: a solid exchanges momentum" begin
    rng = MersenneTwister(77)
    field, _, _ = MA.random_two_phase(rng; periodic = (true, true), solid_kind = :disc, levels = 1)
    op = StaggeredDiffusionOperator(field; viscosity = FIX_VISC)
    sizes = [relmax(op.stiffness, component_ones(field.mesh, k)) for k in 1:2]
    println("periodic + solid: max |S 1_k|/max|S| = ", sizes)
    @test maximum(sizes) > 1.0e-3
end

@testset "M5/M6: walls" begin
    rng = MersenneTwister(88)
    for (periodic, label) in (((true, false), "x-periodic, y walls"), ((false, false), "box"))
        field, _, _ = MA.random_two_phase(rng; periodic, levels = 1)
        for boundary in (NoSlip(), FreeSlip())
            op = StaggeredDiffusionOperator(field; viscosity = FIX_VISC, boundary)
            s = [relmax(op.stiffness, component_ones(field.mesh, k)) for k in 1:2]
            println(rpad(label, 22), rpad(typeof(boundary).name.name, 9), " max |S 1_k|/max|S| = ", s)
            if periodic == (true, false) && boundary isa FreeSlip
                @test s[1] <= 1.0e-13          # tangential momentum is conserved by free-slip walls
                @test s[2] > 1.0e-3
            elseif periodic == (true, false)
                @test s[1] > 1.0e-3 && s[2] > 1.0e-3
            else
                @test s[1] > 1.0e-3 && s[2] > 1.0e-3
            end
        end
    end
end
