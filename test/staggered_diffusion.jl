using LinearAlgebra: Symmetric, dot, eigvals
using SparseArrays: nnz
using Krylov

function _face_component_energy(velocity, tendency, component)
    result = 0.0
    for local_index in eachindex(component.local_to_global)
        global_index = component.local_to_global[local_index]
        result += component.volumes[local_index] *
                  velocity.interior[global_index] *
                  tendency.interior[global_index]
    end
    return result
end

@testset "staggered diffusion stable core on uniform and AMR meshes" begin
    meshes = (
        BlockTreeMesh(block_size=(8, 8)),
        BlockTreeMesh(block_size=(8, 8), periodic=(false, false)),
        BlockTreeMesh(
            block_size=(4, 4), periodic=(false, false), max_level=2,
            refine=(key, origin, extent) ->
                key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
        ),
    )
    for mesh in meshes, boundary in (NoSlip(), FreeSlip())
        operator = StaggeredDiffusionOperator(mesh; viscosity=0.2, boundary)
        velocity = FaceVelocity(
            mesh, (x, y, time, axis) ->
                axis == 1 ? sinpi(2x) + 0.3cospi(2y) :
                            cospi(2x) - 0.2sinpi(4y))
        tendency = FaceVelocity(
            mesh, (x, y, time, axis) -> 0.0)
        diffusion!(tendency, velocity, operator)
        energy_rate = _face_component_energy(velocity, tendency, operator.x) +
                      _face_component_energy(velocity, tendency, operator.y)
        @test energy_rate <= 256eps() * length(velocity.interior)
        @test operator.x.stiffness ≈ transpose(operator.x.stiffness)
        @test operator.y.stiffness ≈ transpose(operator.y.stiffness)
        @test minimum(eigvals(Symmetric(Matrix(operator.x.stiffness)))) >= -2e-12
        @test minimum(eigvals(Symmetric(Matrix(operator.y.stiffness)))) >= -2e-12
        @test stable_timestep(operator) > 0
    end
end

@testset "corrected AMR Laplacian converges and has stable spectrum" begin
    function corrected_error(nx; spectrum=false)
        mesh = BlockTreeMesh(
            block_size=(nx, nx), max_level=2,
            refine=(key, origin, extent) ->
                key.level == 0 ||
                (key.level == 1 && key.i == 0 && key.j == 0),
        )
        velocity = FaceVelocity(
            mesh, (x, y, time, axis) -> axis == 1 ?
                sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y))
        tendency = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
        operator = StaggeredDiffusionOperator(mesh; boundary=FreeSlip())
        diffusion!(tendency, velocity, operator)
        squared = 0.0
        total_volume = 0.0
        for component in (operator.x, operator.y)
            if spectrum
                eigenvalues = eigvals(Matrix(component.laplacian))
                @test maximum(real, eigenvalues) <=
                      1e-10 * maximum(abs, eigenvalues)
            end
            for local_index in eachindex(component.local_to_global)
                global_index = component.local_to_global[local_index]
                face = mesh.faces[global_index]
                x, y = face.center
                exact = -8pi^2 * (face.axis == 1 ?
                    sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y))
                squared += component.volumes[local_index] *
                           (tendency.interior[global_index] - exact)^2
                total_volume += component.volumes[local_index]
            end
        end
        return sqrt(squared / total_volume)
    end
    corrected_error(4; spectrum=true)
    errors = corrected_error.((8, 16, 32))
    @test log2(errors[1] / errors[2]) > 1.8
    @test log2(errors[2] / errors[3]) > 1.5
end

@testset "wall slip and no-slip closures" begin
    mesh = BlockTreeMesh(
        block_size=(12, 12), periodic=(true, false), extent=(1.0, 1.0))
    couette = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? y : 0.0)
    tendency = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    moving_lid = NoSlip((x, y, time, axis) ->
        axis == 1 && y > 0.5 ? 1.0 : 0.0)
    no_slip = StaggeredDiffusionOperator(mesh; boundary=moving_lid)
    diffusion!(tendency, couette, no_slip)
    @test maximum(abs, tendency.interior) <= 2e-12

    constant_tangent = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? 2.0 : 0.0)
    free_slip = StaggeredDiffusionOperator(mesh; boundary=FreeSlip())
    diffusion!(tendency, constant_tangent, free_slip)
    @test maximum(abs, tendency.interior) <= 2e-12

    stationary = StaggeredDiffusionOperator(mesh)
    diffusion!(tendency, constant_tangent, stationary)
    @test minimum(tendency.interior) < 0
end

@testset "uniform periodic staggered Laplacian convergence" begin
    function error(nx)
        mesh = BlockTreeMesh(block_size=(nx, nx))
        velocity = FaceVelocity(
            mesh, (x, y, time, axis) -> axis == 1 ?
                sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y))
        tendency = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
        operator = StaggeredDiffusionOperator(mesh; boundary=FreeSlip())
        diffusion!(tendency, velocity, operator)
        squared = 0.0
        volume = 0.0
        for component in (operator.x, operator.y)
            for local_index in eachindex(component.local_to_global)
                global_index = component.local_to_global[local_index]
                face = mesh.faces[global_index]
                x, y = face.center
                exact = -8pi^2 * (face.axis == 1 ?
                    sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y))
                squared += component.volumes[local_index] *
                           (tendency.interior[global_index] - exact)^2
                volume += component.volumes[local_index]
            end
        end
        return sqrt(squared / volume)
    end
    errors = error.((8, 16, 32))
    @test log2(errors[1] / errors[2]) > 1.9
    @test log2(errors[2] / errors[3]) > 1.9
end

@testset "implicit midpoint staggered elliptic solve" begin
    mesh = BlockTreeMesh(block_size=(16, 16))
    viscosity = 0.1
    diffusion_operator = StaggeredDiffusionOperator(
        mesh; viscosity, boundary=FreeSlip())
    elliptic_operator = StaggeredEllipticOperator(
        diffusion_operator; diffusion=0.05 * viscosity)
    elliptic_solver = StaggeredEllipticSolver(elliptic_operator)
    old_velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? sinpi(2x) : cospi(2y))
    convected = copy(old_velocity)
    numerical = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    work = copy(numerical)
    dt = 0.1
    implicit_midpoint_diffusion!(
        numerical, convected, old_velocity, diffusion_operator,
        dt, elliptic_solver, work)

    dx = mesh.blocks[1].spacing[1]
    discrete_eigenvalue = -4sinpi(1 / 16)^2 / dx^2
    amplification = (1 + dt * viscosity * discrete_eigenvalue / 2) /
                    (1 - dt * viscosity * discrete_eigenvalue / 2)
    for (index, face) in pairs(mesh.faces)
        initial = face.axis == 1 ? sinpi(2face.center[1]) :
                                  cospi(2face.center[2])
        @test numerical.interior[index] ≈ amplification * initial atol=2e-13 rtol=2e-13
    end
    @test elliptic_solver.operator.diffusion ≈ dt * viscosity / 2
    @test elliptic_solver isa StaggeredEllipticSolver
    @test elliptic_solver.x.krylov isa Krylov.BicgstabWorkspace
    @test length(elliptic_solver.x.preconditioner.levels) > 1
    @test length(elliptic_solver.y.preconditioner.levels) > 1
    @test size(last(elliptic_solver.x.preconditioner.levels).matrix, 1) <
          size(first(elliptic_solver.x.preconditioner.levels).matrix, 1)
    @test elliptic_solver.x.stats.converged
    @test elliptic_solver.y.stats.converged

    x_hierarchy = elliptic_solver.x.preconditioner
    y_hierarchy = elliptic_solver.y.preconditioner
    implicit_midpoint_diffusion!(
        numerical, convected, old_velocity, diffusion_operator,
        dt, elliptic_solver, work)
    @test elliptic_solver.x.preconditioner === x_hierarchy
    @test elliptic_solver.y.preconditioner === y_hierarchy
end

@testset "external BiCGSTAB and staggered preconditioners" begin
    mesh = BlockTreeMesh(
        block_size=(8, 8), max_level=2,
        refine=(key, origin, extent) ->
            key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 0),
    )
    diffusion = StaggeredDiffusionOperator(
        mesh; viscosity=0.1, boundary=FreeSlip())
    operator = StaggeredEllipticOperator(diffusion; diffusion=0.1)
    solver = StaggeredEllipticSolver(operator; coarse_size=16)
    right_hand_side = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y))
    solution = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    stats = solve!(solution, solver, right_hand_side;
                   rtol=1e-10, max_cycles=80)
    @test stats.converged
    @test stats.x.method === :bicgstab
    @test stats.y.method === :bicgstab
    @test stats.x.preconditioner === :geometric_multigrid
    @test stats.y.preconditioner === :geometric_multigrid
    @test stats.x.cycles < 80
    @test stats.y.cycles < 80
    @test length(solver.x.preconditioner.levels) > 2
    @test length(solver.y.preconditioner.levels) > 2
    for (hierarchy, component) in (
            (solver.x.preconditioner, operator.x),
            (solver.y.preconditioner, operator.y),
        )
        levels = hierarchy.levels
        @test first(levels).matrix === component.matrix
        @test all(level -> level.prolongation isa TwoPhaseAMR.WeightedTransfer,
                  levels)
        @test all(level -> nnz(level.prolongation) <=
                           3size(level.prolongation, 1), levels)
        @test all(level -> isapprox(
                      level.matrix, transpose(level.matrix);
                      rtol=1e-13, atol=1e-14), levels[2:end])
        @test sum(size(level.matrix, 1) for level in levels) /
              size(first(levels).matrix, 1) < 1.7

        transfer = first(levels).prolongation
        fine = collect(range(0.1, 1.0; length=size(transfer, 1)))
        coarse = collect(range(-0.2, 0.7; length=size(transfer, 2)))
        prolonged = similar(fine)
        restricted = similar(coarse)
        TwoPhaseAMR._prolongate!(prolonged, transfer, coarse)
        TwoPhaseAMR._restrict!(restricted, transfer, fine)
        @test dot(fine, prolonged) ≈ dot(restricted, coarse)
    end
    residuals = residual_norms(solver)
    @test residuals.x[2] < 1e-9
    @test residuals.y[2] < 1e-9

    plain_solution = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    plain_solver = StaggeredEllipticSolver(
        operator; preconditioner=nothing)
    plain_stats = solve!(
        plain_solution, plain_solver, right_hand_side;
        rtol=1e-10, max_cycles=500)
    @test plain_stats.converged
    @test plain_stats.x.preconditioner === :none
    @test maximum(abs, solution.interior - plain_solution.interior) < 2e-9

    for (method, workspace) in (
            (:gmres, Krylov.GmresWorkspace),
            (:fgmres, Krylov.FgmresWorkspace),
        )
        gmres_solution = FaceVelocity(
            mesh, (x, y, time, axis) -> 0.0)
        gmres_solver = StaggeredEllipticSolver(
            operator;
            krylov=(matrix, rhs) -> workspace(
                matrix, rhs; memory=20),
            krylov_options=(; restart=true, reorthogonalization=true),
            coarse_size=16)
        gmres_stats = solve!(
            gmres_solution, gmres_solver, right_hand_side;
            rtol=1e-10, max_cycles=80)
        @test gmres_stats.converged
        @test gmres_stats.x.method === method
        @test maximum(abs,
            gmres_solution.interior - plain_solution.interior) < 2e-9
    end

    fill!(solution.interior, 0)
    solve!(solution, solver, right_hand_side; rtol=1e-10, max_cycles=80)
    fill!(solution.interior, 0)
    @test @allocated(solve!(
        solution, solver, right_hand_side;
        rtol=1e-10, max_cycles=80)) == 0

    @test_throws ArgumentError StaggeredEllipticSolver(
        operator; preconditioner=:unsupported)
    @test_throws ArgumentError StaggeredGeometricMultigridPreconditioner(
        operator; max_transfer_entries=1)
end
