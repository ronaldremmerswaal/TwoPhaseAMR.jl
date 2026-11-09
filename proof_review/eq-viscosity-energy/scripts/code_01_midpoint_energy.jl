# Code check 01: eq:viscosity:energy on the package's two-phase midpoint solve (walled composite, cut, periodic composite drop).
#
# Label:  eq:viscosity:energy (also :phase-midpoint)
# Source: docs/techref/chapters/09_viscosity.tex:376-407; src/two_phase/diffusion/variable_diffusion_solver.jl
#         (implicit_midpoint_diffusion!), src/solvers/multigrid/elliptic_solver.jl (_true_residual!: residual = b - A x)
# Checks (random velocities, densities (1000,1), mu = (0.1, 0.002), dt = 0.05 and 50):
#   E1  the solve satisfies (M + dt/2 S) u2 = (M - dt/2 S) u1 to the tolerance (strong residual),
#   E2  with a deliberately loose tolerance (1e-2) the identity eq:viscosity:energy holds EXACTLY with r = A u2 - rhs
#       (the solver's own residual is rhs - A u2 = -r, checked against solver.residual); the opposite sign fails,
#   E3  the exact-solve energy change is <= 0 and equals the dissipation,
#   E4  S is the matrix of the operator (S = E' W E, independent strain map of _shared/viscosity_audit.jl).
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-viscosity-energy/scripts/code_01_midpoint_energy.jl
using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ViscosityAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "viscosity_audit.jl"))
using .ViscosityAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR

function solve_case(field, u1vec, dt; tolerance, max_cycles = 2000)
    mesh = field.mesh
    op = StaggeredDiffusionOperator(field; viscosity = FIX_VISC)
    AMR._update_dynamic_viscosity!(op, field, FIX_VISC)
    density = mixture_density(field, FIX_DENSITY)
    solver = AMR.VariableStaggeredDiffusionSolver(op, density, dt)
    source = FaceVelocity(mesh); source.interior .= u1vec
    result = FaceVelocity(mesh)
    implicit_midpoint_diffusion!(result, source, op, dt, solver; density, tolerance, max_cycles)
    return op, density, solver, source, result
end

@testset "eq:viscosity:energy on the package solve" begin
    rng = MersenneTwister(1109)
    for (name, fx) in ((:walled, fixture_walled), (:cut, fixture_cut), (:periodic, fixture_periodic)), dt in (0.05, 50.0)
        field = fx()
        nf = length(field.mesh.faces)
        u1 = randn(rng, nf)
        op, density, solver, source, exact = solve_case(field, u1, dt; tolerance = 1.0e-13)
        S = op.stiffness
        M = density .* op.volumes
        u2 = copy(exact.interior)
        # E1 strong residual of the exact solve
        r = M .* (u2 .- u1) .+ dt / 2 .* (S * (u1 .+ u2))
        strong = maximum(abs, r ./ M) / maximum(abs, u1)
        # E3 energy change vs dissipation
        w = u1 .+ u2
        dE = (dot(u2, M .* u2) - dot(u1, M .* u1)) / 2
        diss = dt / 4 * dot(w, S * w)
        @test dE <= 0
        # E2 loose solve
        _, _, solver2, _, loose = solve_case(field, u1, dt; tolerance = 1.0e-2)
        v2 = copy(loose.interior)
        rl = M .* (v2 .- u1) .+ dt / 2 .* (S * (u1 .+ v2))            # A v2 - rhs
        wl = u1 .+ v2
        dEl = (dot(v2, M .* v2) - dot(u1, M .* u1)) / 2
        right = -dt / 4 * dot(wl, S * wl) + dot(wl, rl) / 2
        wrong = -dt / 4 * dot(wl, S * wl) - dot(wl, rl) / 2
        scale = abs(dot(u1, M .* u1)) / 2 + abs(dt / 4 * dot(wl, S * wl))
        sign_gap = maximum(abs, (rl .+ solver2.residual) ./ M) / maximum(abs, u1)    # solver.residual = rhs - A v2 = -rl (velocity-scaled)
        println(rpad("$name dt=$dt", 18), " strong(exact)=", round(strong, sigdigits = 3), "  |dE+diss|/diss=", round(abs(dE + diss) / diss, sigdigits = 3),
            "  loose: |r|/|rhs|=", round(maximum(abs, rl ./ M) / maximum(abs, u1), sigdigits = 3),
            "  identity defect=", round(abs(dEl - right) / scale, sigdigits = 3), "  wrong-sign defect=", round(abs(dEl - wrong) / scale, sigdigits = 3),
            "  solver.residual vs -r: ", round(sign_gap, sigdigits = 3))
        @test strong <= 1.0e-10
        @test abs(dE + diss) / diss <= 1.0e-9
        @test abs(dEl - right) / scale <= 1.0e-12
        significant = maximum(abs, rl ./ M) / maximum(abs, u1) > 1.0e-4      # the loose solve is only inexact at the stiff step
        significant && @test abs(dEl - wrong) / scale > 1.0e-6
        @test sign_gap <= 1.0e-10
    end
end

@testset "E4: stiffness equals E'WE of the independent strain map" begin
    for fx in (fixture_walled, fixture_periodic)
        field = fx()
        op = StaggeredDiffusionOperator(field; viscosity = FIX_VISC)
        E, meas, logmu, _ = independent_strain(field, FIX_VISC)
        S = E' * Diagonal(meas .* exp.(logmu)) * E
        @test maximum(abs, S - op.stiffness) <= 1.0e-12 * maximum(abs, S)
    end
end
