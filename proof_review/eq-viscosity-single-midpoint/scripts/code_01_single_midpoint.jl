# Code check 01: eq:viscosity:single-midpoint -- the single-phase Crank--Nicolson step with the explicit half at u^n and the wall data
# at the midpoint; solver choice and stopping criterion.
#
# Label:  eq:viscosity:single-midpoint
# Source: docs/techref/chapters/09_viscosity.tex:332-372; src/navier_stokes/implicit_diffusion.jl (implicit_midpoint_diffusion!,
#         _add_wall_source!, StaggeredEllipticSolver), src/single_phase/stepping.jl (_diffusion_substep!)
# Checks (walled composite mesh with a time-dependent moving wall, uniform and composite periodic meshes):
#   M1  (I - a L) u2 = u1 + a L u^n + dt nu V^-1 sum_w cond_w u_w(t + dt/2),  a = dt nu/2, solved independently with a sparse direct solve,
#       equals the package solve to the tolerance; u^n and u1 differ;
#   M2  the wall data are taken at the midpoint t + dt/2 (the alternatives t, t + dt change the answer by O(dt) relative);
#   M3  CG is chosen iff V L is symmetric for every shift (uniform periodic and uniform walled mesh), BiCGSTAB otherwise (composite);
#   M4  momentum: sum_f V_f (u2 - u1)_f = 0 on the uniform periodic mesh (although u^n != u1), != 0 on the periodic composite mesh (band);
#   M5  the two-phase energy identity does NOT hold for the single-phase step: size of E2 - E1 + (dt nu/4) w' S w with S the core stiffness;
#   M6  stopping criterion: final strong residual max_f |r_f|/V_f <= tol max_f |rhs_f|/V_f (or the round-off floor); max_cycles too small
#       raises an error.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-viscosity-single-midpoint/scripts/code_01_single_midpoint.jl
using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ViscosityAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "viscosity_audit.jl"))
using .ViscosityAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR

composite(periodic) = BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic, max_level = 1, refine = (key, o, e) -> key.level == 0 && key.i >= 2)

function independent_step(op, mesh, u1, un, dt, time, boundary)
    nu = op.viscosity; a = dt * nu / 2
    result = zeros(length(mesh.faces))
    for (component, axis) in ((op.x, 1), (op.y, 2))
        g = component.local_to_global; n = length(g)
        b = zeros(n)
        for c in component.wall_connections
            AMR._include_wall_connection(boundary, c) || continue
            b[c.degree_of_freedom] += c.conductance * AMR._wall_value(boundary, c, time + dt / 2, axis, Float64) / component.volumes[c.degree_of_freedom]
        end
        L = component.laplacian
        rhs = u1[g] .+ a .* (L * un[g]) .+ dt * nu .* b
        result[g] .= (sparse(I, n, n) - a * L) \ rhs
    end
    return result
end

@testset "M1/M2: the step" begin
    rng = MersenneTwister(1234)
    mesh = composite((false, false))
    nu = 0.05; dt = 0.2; time = 0.3
    lid(x, y, t, axis) = axis == 1 ? 1.0 + 2sin(3t) + 0.2x : 0.1 * cos(t)
    boundary = NoSlip(lid)
    op = StaggeredDiffusionOperator(mesh; viscosity = nu, boundary)
    solver = StaggeredEllipticSolver(StaggeredEllipticOperator(op; diffusion = dt * nu / 2))
    u1 = FaceVelocity(mesh); un = FaceVelocity(mesh)
    u1.interior .= randn(rng, length(mesh.faces)); un.interior .= u1.interior .+ 0.3 .* randn(rng, length(mesh.faces))
    result = FaceVelocity(mesh); work = FaceVelocity(mesh)
    implicit_midpoint_diffusion!(result, u1, un, op, dt, solver, work; time, tolerance = 1.0e-13, max_cycles = 500)
    ref = independent_step(op, mesh, u1.interior, un.interior, dt, time, boundary)
    err = maximum(abs, result.interior .- ref) / maximum(abs, ref)
    println("M1: package vs independent direct solve: ", err)
    @test err <= 1.0e-10
    # M2: alternatives for the wall time level
    for shift in (-dt / 2, +dt / 2)
        alt = independent_step(op, mesh, u1.interior, un.interior, dt, time + shift, boundary)
        d = maximum(abs, alt .- ref) / maximum(abs, ref)
        println("   wall data at t + dt/2 ", shift > 0 ? "+" : "-", " dt/2 would change the result by ", d)
        @test d > 1.0e-4
    end
end

@testset "M3: solver choice" begin
    mk(mesh) = begin
        op = StaggeredDiffusionOperator(mesh; viscosity = 1.0)
        StaggeredEllipticSolver(StaggeredEllipticOperator(op; diffusion = 0.1))
    end
    for (name, mesh, expected) in (
            ("uniform periodic", BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (true, true)), "CgWorkspace"),
            ("uniform walled", BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (false, false)), "CgWorkspace"),
            ("composite walled", composite((false, false)), "BicgstabWorkspace"),
            ("composite periodic", composite((true, true)), "BicgstabWorkspace"),
        )
        s = mk(mesh)
        println(rpad(name, 20), typeof(s.x.krylov).name.name)
        @test string(typeof(s.x.krylov).name.name) == expected
    end
end

@testset "M4/M5: momentum and energy" begin
    rng = MersenneTwister(5)
    for (name, mesh) in (("uniform periodic", BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (true, true))), ("composite periodic", composite((true, true))))
        nu = 1.0; dt = 0.05
        op = StaggeredDiffusionOperator(mesh; viscosity = nu)
        solver = StaggeredEllipticSolver(StaggeredEllipticOperator(op; diffusion = dt * nu / 2))
        u1 = FaceVelocity(mesh); un = FaceVelocity(mesh)
        u1.interior .= randn(rng, length(mesh.faces)); un.interior .= randn(rng, length(mesh.faces))
        result = FaceVelocity(mesh); work = FaceVelocity(mesh)
        implicit_midpoint_diffusion!(result, u1, un, op, dt, solver, work; tolerance = 1.0e-13, max_cycles = 500)
        rels = Float64[]
        for (component, axis) in ((op.x, 1), (op.y, 2))
            g = component.local_to_global; V = component.volumes
            change = dot(V, result.interior[g] .- u1.interior[g])
            push!(rels, abs(change) / dot(V, abs.(result.interior[g] .- u1.interior[g])))
        end
        println(rpad(name, 20), "relative momentum change |sum V du| / sum V |du| = ", rels)
        name == "uniform periodic" && @test maximum(rels) <= 1.0e-12
        name == "composite periodic" && @test maximum(rels) > 1.0e-5
    end
    # energy: the identity of the two-phase step with the core stiffness fails
    mesh = BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (true, true))
    nu = 1.0; dt = 0.05
    op = StaggeredDiffusionOperator(mesh; viscosity = nu)
    solver = StaggeredEllipticSolver(StaggeredEllipticOperator(op; diffusion = dt * nu / 2))
    u1 = FaceVelocity(mesh); un = FaceVelocity(mesh)
    u1.interior .= randn(rng, length(mesh.faces)); un.interior .= u1.interior .+ 0.5 .* randn(rng, length(mesh.faces))
    result = FaceVelocity(mesh); work = FaceVelocity(mesh)
    implicit_midpoint_diffusion!(result, u1, un, op, dt, solver, work; tolerance = 1.0e-13, max_cycles = 500)
    defect = 0.0; scale = 0.0
    for component in (op.x, op.y)
        g = component.local_to_global; V = component.volumes; S = component.stiffness
        u1c, u2c = u1.interior[g], result.interior[g]
        w = u1c .+ u2c
        dE = (dot(u2c, V .* u2c) - dot(u1c, V .* u1c)) / 2
        diss = dt * nu / 4 * dot(w, S * w)
        defect += abs(dE + diss); scale += abs(dE) + diss
    end
    println("energy-identity defect |E2 - E1 + (dt nu/4) w'Sw| / (|dE| + dissipation) = ", defect / scale, " (explicit half at u^n != u^(1))")
    @test defect / scale > 1.0e-3
    # with u^n = u^(1) on the uniform periodic mesh the identity does hold
    implicit_midpoint_diffusion!(result, u1, u1, op, dt, solver, work; tolerance = 1.0e-13, max_cycles = 500)
    defect2 = 0.0
    for component in (op.x, op.y)
        g = component.local_to_global; V = component.volumes; S = component.stiffness
        u1c, u2c = u1.interior[g], result.interior[g]; w = u1c .+ u2c
        dE = (dot(u2c, V .* u2c) - dot(u1c, V .* u1c)) / 2
        defect2 = max(defect2, abs(dE + dt * nu / 4 * dot(w, S * w)) / abs(dE))
    end
    println("with u^n = u^(1) on the uniform periodic mesh the energy identity holds: defect ", defect2)
    @test defect2 <= 1.0e-9
end

@testset "M6: stopping criterion" begin
    rng = MersenneTwister(6)
    mesh = composite((false, false))
    nu = 1.0; dt = 5.0
    op = StaggeredDiffusionOperator(mesh; viscosity = nu)
    solver = StaggeredEllipticSolver(StaggeredEllipticOperator(op; diffusion = dt * nu / 2))
    u1 = FaceVelocity(mesh); u1.interior .= randn(rng, length(mesh.faces))
    result = FaceVelocity(mesh); work = FaceVelocity(mesh)
    for tol in (1.0e-3, 1.0e-6, 1.0e-10)
        implicit_midpoint_diffusion!(result, u1, u1, op, dt, solver, work; tolerance = tol, max_cycles = 500)
        worst = 0.0
        for (component, axis) in ((op.x, 1), (op.y, 2))
            g = component.local_to_global; V = component.volumes
            a = dt * nu / 2
            rhs = u1.interior[g] .+ a .* (component.laplacian * u1.interior[g])
            res = rhs .- (result.interior[g] .- a .* (component.laplacian * result.interior[g]))
            worst = max(worst, maximum(abs, res) / maximum(abs, rhs))
        end
        println("tol ", tol, ": strong residual / max|rhs| = ", worst)
        @test worst <= 2 * tol + 1.0e-12
    end
    @test_throws ErrorException implicit_midpoint_diffusion!(result, u1, u1, op, dt, solver, work; tolerance = 1.0e-13, max_cycles = 1)
end
