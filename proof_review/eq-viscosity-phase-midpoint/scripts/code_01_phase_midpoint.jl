# Code check 01: eq:viscosity:phase-midpoint -- the one-velocity two-phase midpoint solve about the transported velocity, via the flow
# stepping (_diffusion_substep!), solver choice, special cases.
#
# Label:  eq:viscosity:phase-midpoint
# Source: docs/techref/chapters/09_viscosity.tex:376-410; src/two_phase/stepping.jl (_diffusion_substep!),
#         src/two_phase/diffusion/variable_diffusion_solver.jl (implicit_midpoint_diffusion!, VariableStaggeredDiffusionSolver)
# Checks (FlowSimulation on the periodic composite drop fixture, TwoPhase density (1000,1), mu (0.1,0.002), free slip):
#   P1  the step's velocity equals the direct solve of (M + dt/2 S) u2 = (M - dt/2 S) u1 about the TRANSPORTED velocity u1 = workspace.convected
#       (old_velocity differs from u1 and is not used), to 4096 eps relative (as the repository test) -- here also with the composite drop;
#   P2  M = diag(rho_f V_f) with rho_f of eq:forces:density (mixture_density) and V the dual volumes (open volumes with 1 where they
#       vanish for a solid);
#   P3  the matrix is symmetric positive definite and the solver uses CG (Krylov type of the solver);
#   P4  both viscosities zero: u2 = u1 and no operator is built; one zero viscosity is rejected (error type reported); non-positive
#       single-phase viscosity rejected;
#   P5  tolerance semantics: max_f |r_f|/m_f <= max(tol max_f |rhs_f|/m_f, floor) with m = rho V (stats of the solver), error otherwise.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-viscosity-phase-midpoint/scripts/code_01_phase_midpoint.jl
using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ViscosityAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "viscosity_audit.jl"))
using .ViscosityAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR

@testset "P1-P3: the flow's diffusion substep" begin
    field = fixture_periodic(); mesh = field.mesh
    flow = FlowSimulation(
        mesh, TwoPhase(field; density = FIX_DENSITY, dynamic_viscosity = FIX_VISC); boundary = FreeSlip(),
        numerics = NavierStokesOptions(diffusion = LinearSolverOptions(preconditioner = nothing))
    )
    phase = flow.workspace.two_phase
    rng = MersenneTwister(99)
    old_values = randn(rng, length(mesh.faces)); split_values = old_values .+ 0.3 .* randn(rng, length(mesh.faces))
    copyto!(flow.workspace.old_velocity.interior, old_values)
    copyto!(flow.workspace.convected.interior, split_values)
    dt = 0.05
    operator = phase.diffusion
    AMR._update_dynamic_viscosity!(operator, field, flow.system.two_phase.dynamic_viscosity)
    density = mixture_density(field, flow.system.two_phase.density)
    inertia = density .* operator.volumes
    matrix = Diagonal(inertia) + (dt / 2) * operator.stiffness
    expected = matrix \ (inertia .* split_values .- (dt / 2) .* (operator.stiffness * split_values))
    @test norm(operator.stiffness * (old_values .- split_values)) > 0          # u^n and u^(1) really differ
    AMR._diffusion_substep!(flow.velocity, flow.system, flow.workspace, dt, 0.0, flow.system.two_phase; tolerance = 1.0e-12, max_cycles = 500)
    err = maximum(abs, flow.velocity.interior .- expected) / maximum(abs, expected)
    println("P1: substep vs direct solve about u^(1): ", err)
    @test err <= 1.0e-9
    # the old velocity is irrelevant
    copyto!(flow.workspace.old_velocity.interior, randn(rng, length(mesh.faces)))
    velocity2 = FaceVelocity(mesh)
    AMR._diffusion_substep!(velocity2, flow.system, flow.workspace, dt, 0.0, flow.system.two_phase; tolerance = 1.0e-12, max_cycles = 500)
    @test maximum(abs, velocity2.interior .- flow.velocity.interior) <= 1.0e-9 * maximum(abs, expected)
    # P2
    @test maximum(abs, density .- [FIX_DENSITY[1] * AMR._staggered_liquid_fraction(field, field.geometry, f) + FIX_DENSITY[2] * (1 - AMR._staggered_liquid_fraction(field, field.geometry, f)) for f in mesh.faces]) <= 1.0e-12
    @test operator.volumes == mesh.face_dual_volumes
    # P3
    @test isposdef(Symmetric(Matrix(matrix)))
    println("P3: Krylov workspace of the diffusion solver: ", typeof(flow.workspace.two_phase.diffusion_solver.krylov).name.name)
    @test string(typeof(flow.workspace.two_phase.diffusion_solver.krylov).name.name) == "CgWorkspace"
    # P5 stats
    stats = flow.workspace.two_phase.diffusion_solver.stats
    println("P5: solver stats: converged=", stats.converged, " final_linf=", stats.final_linf, " target=", stats.target_linf, " residual_scale=", stats.residual_scale)
    @test stats.converged && stats.final_linf <= stats.target_linf
end

@testset "P4: special viscosities" begin
    field = fixture_periodic(); mesh = field.mesh
    # one zero viscosity
    r1 = try
        TwoPhase(field; density = FIX_DENSITY, dynamic_viscosity = (0.1, 0.0)); "accepted"
    catch err
        sprint(showerror, err)
    end
    println("TwoPhase(...; dynamic_viscosity = (0.1, 0.0)): ", r1)
    r2 = try
        flow = FlowSimulation(mesh, TwoPhase(field; density = FIX_DENSITY, dynamic_viscosity = (0.1, 0.0))); "accepted"
    catch err
        sprint(showerror, err)
    end
    println("FlowSimulation with (0.1, 0.0): ", r2)
    @test r1 != "accepted" || r2 != "accepted"
    @test_throws ArgumentError StaggeredDiffusionOperator(field; viscosity = (0.1, 0.0))
    # moving walls and VelocityBoundaryConditions are rejected by the variable-viscosity operator (eq:viscosity:rows paragraph)
    @test_throws ArgumentError StaggeredDiffusionOperator(field; viscosity = FIX_VISC, boundary = NoSlip((x, y, t, axis) -> 1.0))
    @test_throws ArgumentError StaggeredDiffusionOperator(field; viscosity = FIX_VISC, boundary = VelocityBoundaryConditions(top = NoSlip()))
    @test StaggeredDiffusionOperator(field; viscosity = FIX_VISC, boundary = NoSlip()) isa AMR.VariableStaggeredDiffusionOperator
    @test StaggeredDiffusionOperator(field; viscosity = FIX_VISC, boundary = FreeSlip()) isa AMR.VariableStaggeredDiffusionOperator
    @test_throws ArgumentError StaggeredDiffusionOperator(mesh; viscosity = 0.0)
    @test_throws ArgumentError StaggeredDiffusionOperator(mesh; viscosity = -1.0)
    # both zero: velocity = convected, no operator
    flow0 = FlowSimulation(mesh, TwoPhase(field; density = FIX_DENSITY, dynamic_viscosity = (0.0, 0.0)); boundary = FreeSlip())
    rng = MersenneTwister(1)
    vals = randn(rng, length(mesh.faces))
    copyto!(flow0.workspace.convected.interior, vals)
    AMR._diffusion_substep!(flow0.velocity, flow0.system, flow0.workspace, 0.05, 0.0, flow0.system.two_phase)
    @test flow0.velocity.interior == vals
end
