using LinearAlgebra: Symmetric, eigvals

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
    @test elliptic_solver isa StaggeredGeometricMultigridSolver
    @test length(elliptic_solver.x.levels) > 1
    @test length(elliptic_solver.y.levels) > 1
    @test size(last(elliptic_solver.x.levels).matrix, 1) <
          size(first(elliptic_solver.x.levels).matrix, 1)
    @test elliptic_solver.x.stats.converged
    @test elliptic_solver.y.stats.converged

    x_hierarchy = elliptic_solver.x
    y_hierarchy = elliptic_solver.y
    implicit_midpoint_diffusion!(
        numerical, convected, old_velocity, diffusion_operator,
        dt, elliptic_solver, work)
    @test elliptic_solver.x === x_hierarchy
    @test elliptic_solver.y === y_hierarchy
end

@testset "mixed-level staggered geometric multigrid" begin
    mesh = BlockTreeMesh(
        block_size=(8, 8), max_level=2,
        refine=(key, origin, extent) ->
            key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 0),
    )
    diffusion = StaggeredDiffusionOperator(
        mesh; viscosity=0.1, boundary=FreeSlip())
    operator = StaggeredEllipticOperator(diffusion; diffusion=0.1)
    solver = StaggeredGeometricMultigridSolver(operator; coarse_size=16)
    right_hand_side = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y))
    solution = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    stats = solve!(solution, solver, right_hand_side;
                   rtol=1e-10, max_cycles=80)
    @test stats.converged
    @test stats.x.method === :bicgstab
    @test stats.y.method === :bicgstab
    @test stats.x.cycles < 80
    @test stats.y.cycles < 80
    @test length(solver.x.levels) > 2
    @test length(solver.y.levels) > 2
    residuals = residual_norms(solver)
    @test residuals.x[2] < 1e-9
    @test residuals.y[2] < 1e-9

    standalone_solution = FaceVelocity(
        mesh, (x, y, time, axis) -> 0.0)
    standalone_solver = StaggeredGeometricMultigridSolver(
        operator; coarse_size=16, method=:multigrid)
    standalone_stats = solve!(
        standalone_solution, standalone_solver, right_hand_side;
        rtol=1e-10, max_cycles=80)
    @test standalone_stats.converged
    @test standalone_stats.x.method === :multigrid
    @test stats.x.cycles < standalone_stats.x.cycles
    @test stats.y.cycles < standalone_stats.y.cycles
    @test maximum(abs,
        solution.interior - standalone_solution.interior) < 2e-9

    fill!(solution.interior, 0)
    solve!(solution, solver, right_hand_side; rtol=1e-10, max_cycles=80)
    fill!(solution.interior, 0)
    @test @allocated(solve!(
        solution, solver, right_hand_side;
        rtol=1e-10, max_cycles=80)) == 0

    maximum_rate = max(diffusion.x.maximum_row_sum,
                       diffusion.y.maximum_row_sum)
    weak_operator = StaggeredEllipticOperator(
        diffusion; diffusion=0.45 / maximum_rate)
    weak_solver = StaggeredGeometricMultigridSolver(
        weak_operator; coarse_size=16)
    weak_solution = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    weak_stats = solve!(
        weak_solution, weak_solver, right_hand_side;
        rtol=1e-10, max_cycles=80)
    @test weak_stats.converged
    @test weak_stats.x.method === :multigrid
    @test weak_stats.y.method === :multigrid

    @test_throws ArgumentError StaggeredGeometricMultigridSolver(
        operator; method=:unsupported)
end
