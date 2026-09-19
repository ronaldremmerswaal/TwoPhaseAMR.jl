isdefined(@__MODULE__, :test_cell_values) || include("test_utils.jl")

@testset "optional extensions are inactive without their weak dependency" begin
    @test isnothing(
        Base.get_extension(
            TwoPhaseAMR, :TwoPhaseAMRAlgebraicMultigridExt,
        )
    )
    @test isnothing(
        Base.get_extension(
            TwoPhaseAMR, :TwoPhaseAMRLimitedLDLFactorizationsExt,
        )
    )
    @test isnothing(Base.get_extension(TwoPhaseAMR, :TwoPhaseAMRMakieExt))
end

import IncompleteLU
using Krylov

_optional_linear_solver_dependencies = all(
    package -> !isnothing(Base.find_package(package)),
    ("AlgebraicMultigrid", "LimitedLDLFactorizations"),
)

if !_optional_linear_solver_dependencies
    @testset "optional linear-solver extensions require test extras" begin
        @test_skip _optional_linear_solver_dependencies
    end
else
    import AlgebraicMultigrid
    import LimitedLDLFactorizations

@testset "optional linear-solver extensions load on demand" begin
    @test !isnothing(
        Base.get_extension(
            TwoPhaseAMR, :TwoPhaseAMRAlgebraicMultigridExt,
        )
    )
    @test !isnothing(
        Base.get_extension(
            TwoPhaseAMR, :TwoPhaseAMRLimitedLDLFactorizationsExt,
        )
    )
end

@testset "optional public preconditioners" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 1,
        refine = (key, origin, extent) -> true,
    )
    pressure_operator = PoissonOperator(mesh)
    pressure_rhs = CellField(mesh)
    initialize!(
        pressure_rhs, (x, y) ->
        8pi^2 * sinpi(2x) * cospi(2y)
    )

    pressure_cases = (
        (
            name = :algebraic_multigrid, krylov = Krylov.CgWorkspace,
            preconditioner = :amg, options = (;),
        ),
        (
            name = :incomplete_cholesky, krylov = Krylov.CgWorkspace,
            preconditioner = :ichol,
            options = (; memory = 4, shift = 1.0e-8),
        ),
        # ILU is not guaranteed symmetric positive definite, so pair it with
        # BiCGSTAB instead of violating CG's preconditioner requirements.
        (
            name = :ilu, krylov = Krylov.BicgstabWorkspace,
            preconditioner = :ilu,
            options = (; drop_tolerance = 1.0e-3),
        ),
    )
    for case in pressure_cases
        solver = EllipticSolver(
            pressure_operator; krylov = case.krylov,
            preconditioner = case.preconditioner,
            preconditioner_options = case.options
        )
        pressure = CellField(mesh)
        stats = solve!(
            pressure, solver, pressure_rhs; rtol = 1.0e-9, max_cycles = 200
        )
        @test stats.converged
        @test stats.preconditioner === case.name
        @test size(solver.preconditioner) == size(pressure_operator.matrix)
        @test eltype(solver.preconditioner) === Float64
        @test all(isfinite, test_cell_values(pressure))
    end

    diffusion = StaggeredDiffusionOperator(
        mesh; viscosity = 0.1, boundary = FreeSlip()
    )
    diffusion_operator = StaggeredEllipticOperator(
        diffusion; diffusion = 0.05
    )
    velocity_rhs = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y)
    )
    for (name, specification, options) in (
            (:algebraic_multigrid, :amg, (;)),
            (
                :incomplete_cholesky, :ichol,
                (; memory = 4, shift = 1.0e-8),
            ),
            (:ilu, :ilu, (; drop_tolerance = 1.0e-3)),
        )
        solver = StaggeredEllipticSolver(
            diffusion_operator; preconditioner = specification,
            preconditioner_options = options
        )
        velocity = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
        stats = solve!(
            velocity, solver, velocity_rhs;
            tolerance = 1.0e-9, max_cycles = 200
        )
        @test stats.converged
        @test stats.x.preconditioner === name
        @test stats.y.preconditioner === name
        @test all(isfinite, velocity.interior)
    end

    navier_mesh = BlockTreeMesh(block_size = (4, 4))
    navier_solver = NavierStokesSolver(
        navier_mesh; density = 1.0, dynamic_viscosity = 0.01,
        boundary = FreeSlip(),
        pressure_preconditioner = :amg,
        diffusion_preconditioner = :ilu,
        diffusion_preconditioner_options = (; drop_tolerance = 1.0e-3),
        coarse_size = 8
    )
    navier_workspace = NavierStokesWorkspace(navier_solver)
    velocity = FaceVelocity(navier_mesh, (x, y, time, axis) -> 0.0)
    pressure = CellField(navier_mesh)
    stats = step!(
        velocity, pressure, navier_solver, 0.1, navier_workspace;
        force = (x, y, time, axis) ->
        axis == 1 ? 0.1sinpi(2y) : 0.0,
        tolerance = 1.0e-9, max_cycles = 100
    )
    @test stats.converged
    @test stats.preconditioner === :algebraic_multigrid
    @test navier_workspace.elliptic.x.stats.preconditioner === :ilu
end
end
