using LinearAlgebra: Symmetric, dot, eigvals
using SparseArrays: SparseMatrixCSC, nnz
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

@testset "staggered diffusion viscosity dispatch" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    boundary = FreeSlip()
    scalar = StaggeredDiffusionOperator(mesh; viscosity = 0.2, boundary)
    scalar_compat = TwoPhaseAMR._staggered_diffusion_operator(
        mesh; viscosity = 0.2, boundary, T = Float64,
        construction_cache = nothing, reuse = nothing, cache_layer = 1,
        geometry = nothing,
    )
    @test scalar isa StaggeredDiffusionOperator
    @test scalar_compat.viscosity == scalar.viscosity
    @test scalar_compat.x.stiffness == scalar.x.stiffness
    @test scalar_compat.y.stiffness == scalar.y.stiffness
    @test scalar_compat.x.laplacian == scalar.x.laplacian
    @test scalar_compat.y.laplacian == scalar.y.laplacian

    mesh32 = BlockTreeMesh(
        block_size = (4, 4), origin = (0.0f0, 0.0f0), extent = (1.0f0, 1.0f0),
    )
    scalar32 = StaggeredDiffusionOperator(
        mesh32; viscosity = 0.2f0, boundary, T = Float32,
    )
    viscosity32 = CellField(mesh32; init = 0.7f0)
    variable32 = TwoPhaseAMR._staggered_diffusion_operator(
        mesh32; viscosity = viscosity32, boundary, T = Float32,
        construction_cache = TwoPhaseAMR.VariableDiffusionConstructionCache(Float32),
    )
    @test eltype(scalar32.x.laplacian) === Float32
    @test eltype(variable32.stiffness) === Float32

    viscosity = CellField(mesh, (x, y) -> 1 + 0.2sinpi(2x) * cospi(2y))
    variable = StaggeredDiffusionOperator(mesh; viscosity, boundary)
    cache = TwoPhaseAMR.VariableDiffusionConstructionCache(Float64)
    cached = TwoPhaseAMR._staggered_diffusion_operator(
        mesh; viscosity, boundary, T = Float64, construction_cache = cache,
        cache_layer = 2, geometry = nothing,
    )
    reused = TwoPhaseAMR._staggered_diffusion_operator(
        mesh; viscosity, boundary, T = Float64, construction_cache = cache,
        reuse = cached, cache_layer = 2, geometry = nothing,
    )
    @test variable isa VariableStaggeredDiffusionOperator
    @test reused isa VariableStaggeredDiffusionOperator
    @test reused.strain == variable.strain
    @test reused.stress_interpolation == variable.stress_interpolation
    @test reused.stress_weights == variable.stress_weights
    @test reused.stiffness == variable.stiffness
    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? sinpi(2y) : cospi(2x),
    )
    expected = FaceVelocity(mesh)
    actual = FaceVelocity(mesh)
    diffusion!(expected, velocity, variable)
    diffusion!(actual, velocity, reused)
    @test isequal(actual.interior, expected.interior)
    @test isequal(actual.boundary, expected.boundary)

    geometry = GeometryField(mesh, (x, y) -> y - 0.18 - 0.1x)
    interface = InterfaceField(
        mesh, (x, y) -> x + 0.2y - 0.47; geometry, lipschitz = hypot(1, 0.2),
    )
    phases = (2.0, 0.4)
    interface_operator = StaggeredDiffusionOperator(
        interface; viscosity = phases, boundary,
    )
    interface_cache = TwoPhaseAMR.VariableDiffusionConstructionCache(Float64)
    cached_interface = TwoPhaseAMR._staggered_diffusion_operator(
        interface; viscosity = phases, boundary,
        construction_cache = interface_cache, cache_layer = 2,
    )
    reused_interface = TwoPhaseAMR._staggered_diffusion_operator(
        interface; viscosity = phases, boundary,
        construction_cache = interface_cache, reuse = cached_interface,
        cache_layer = 2,
    )
    @test reused_interface.volumes == interface_operator.volumes
    @test reused_interface.strain == interface_operator.strain
    @test reused_interface.stress_weights == interface_operator.stress_weights
    @test reused_interface.stiffness == interface_operator.stiffness

    for (invalid, message) in (
            (0.0, "constant viscosity must be strictly positive"),
            (NaN, "constant viscosity must be strictly positive"),
            (
                CellField(mesh; init = 0.0),
                "cell dynamic viscosity must be finite and strictly positive",
            ),
            (
                CellField(mesh; init = Inf),
                "cell dynamic viscosity must be finite and strictly positive",
            ),
        )
        exception = try
            StaggeredDiffusionOperator(mesh; viscosity = invalid)
            nothing
        catch error
            error
        end
        @test exception isa ArgumentError
        @test exception isa ArgumentError && exception.msg == message
    end
    @test StaggeredDiffusionOperator(mesh; viscosity = Inf).viscosity == Inf
    for invalid_options in (
            (; geometry = :invalid), (; construction_cache = cache),
            (; reuse = reused), (; cache_layer = 2),
        )
        @test_throws ArgumentError TwoPhaseAMR._staggered_diffusion_operator(
            mesh; viscosity = 0.2, invalid_options...,
        )
    end
    other_mesh = BlockTreeMesh(block_size = (4, 4))
    @test_throws ArgumentError StaggeredDiffusionOperator(
        mesh; viscosity = CellField(other_mesh; init = 1.0),
    )
    @test_throws ArgumentError TwoPhaseAMR._staggered_diffusion_operator(
        mesh; viscosity,
        construction_cache = TwoPhaseAMR.VariableDiffusionConstructionCache(Float32),
    )
    @test_throws ArgumentError TwoPhaseAMR._staggered_diffusion_operator(
        mesh; viscosity, reuse = :invalid,
    )
end

@testset "staggered diffusion stable core on uniform and AMR meshes" begin
    meshes = (
        BlockTreeMesh(block_size = (8, 8)),
        BlockTreeMesh(block_size = (8, 8), periodic = (false, false)),
        BlockTreeMesh(
            block_size = (4, 4), periodic = (false, false), max_level = 2,
            refine = (key, origin, extent) ->
            key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
        ),
    )
    for mesh in meshes, boundary in (NoSlip(), FreeSlip())
        operator = StaggeredDiffusionOperator(mesh; viscosity = 0.2, boundary)
        velocity = FaceVelocity(
            mesh, (x, y, time, axis) ->
            axis == 1 ? sinpi(2x) + 0.3cospi(2y) :
                cospi(2x) - 0.2sinpi(4y)
        )
        tendency = FaceVelocity(
            mesh, (x, y, time, axis) -> 0.0
        )
        diffusion!(tendency, velocity, operator)
        energy_rate = _face_component_energy(velocity, tendency, operator.x) +
            _face_component_energy(velocity, tendency, operator.y)
        @test energy_rate <= 256eps() * length(velocity.interior)
        @test operator.x.stiffness ≈ transpose(operator.x.stiffness)
        @test operator.y.stiffness ≈ transpose(operator.y.stiffness)
        @test minimum(eigvals(Symmetric(Matrix(operator.x.stiffness)))) >= -2.0e-12
        @test minimum(eigvals(Symmetric(Matrix(operator.y.stiffness)))) >= -2.0e-12
        @test stable_timestep(operator) > 0
    end
end

@testset "corrected AMR Laplacian converges and has stable spectrum" begin
    function corrected_error(nx; spectrum = false)
        mesh = BlockTreeMesh(
            block_size = (nx, nx), max_level = 2,
            refine = (key, origin, extent) ->
            key.level == 0 ||
                (key.level == 1 && key.i == 0 && key.j == 0),
        )
        velocity = FaceVelocity(
            mesh, (x, y, time, axis) -> axis == 1 ?
                sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y)
        )
        tendency = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
        operator = StaggeredDiffusionOperator(mesh; boundary = FreeSlip())
        diffusion!(tendency, velocity, operator)
        squared = 0.0
        total_volume = 0.0
        for component in (operator.x, operator.y)
            if spectrum
                eigenvalues = eigvals(Matrix(component.laplacian))
                @test maximum(real, eigenvalues) <=
                    1.0e-10 * maximum(abs, eigenvalues)
            end
            for local_index in eachindex(component.local_to_global)
                global_index = component.local_to_global[local_index]
                face = mesh.faces[global_index]
                x, y = face.center
                exact = -8pi^2 * (
                    face.axis == 1 ?
                        sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y)
                )
                squared += component.volumes[local_index] *
                    (tendency.interior[global_index] - exact)^2
                total_volume += component.volumes[local_index]
            end
        end
        return sqrt(squared / total_volume)
    end
    corrected_error(4; spectrum = true)
    errors = corrected_error.((8, 16, 32))
    @test log2(errors[1] / errors[2]) > 1.8
    @test log2(errors[2] / errors[3]) > 1.5
end

@testset "wall slip and no-slip closures" begin
    mesh = BlockTreeMesh(
        block_size = (12, 12), periodic = (true, false), extent = (1.0, 1.0)
    )
    couette = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? y : 0.0
    )
    tendency = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    moving_lid = NoSlip(
        (x, y, time, axis) ->
        axis == 1 && y > 0.5 ? 1.0 : 0.0
    )
    no_slip = StaggeredDiffusionOperator(mesh; boundary = moving_lid)
    diffusion!(tendency, couette, no_slip)
    @test maximum(abs, tendency.interior) <= 2.0e-12

    constant_tangent = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? 2.0 : 0.0
    )
    free_slip = StaggeredDiffusionOperator(mesh; boundary = FreeSlip())
    diffusion!(tendency, constant_tangent, free_slip)
    @test maximum(abs, tendency.interior) <= 2.0e-12

    stationary = StaggeredDiffusionOperator(mesh)
    diffusion!(tendency, constant_tangent, stationary)
    @test minimum(tendency.interior) < 0
end

@testset "uniform periodic staggered Laplacian convergence" begin
    function error(nx)
        mesh = BlockTreeMesh(block_size = (nx, nx))
        velocity = FaceVelocity(
            mesh, (x, y, time, axis) -> axis == 1 ?
                sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y)
        )
        tendency = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
        operator = StaggeredDiffusionOperator(mesh; boundary = FreeSlip())
        diffusion!(tendency, velocity, operator)
        squared = 0.0
        volume = 0.0
        for component in (operator.x, operator.y)
            for local_index in eachindex(component.local_to_global)
                global_index = component.local_to_global[local_index]
                face = mesh.faces[global_index]
                x, y = face.center
                exact = -8pi^2 * (
                    face.axis == 1 ?
                        sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y)
                )
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
    mesh = BlockTreeMesh(block_size = (16, 16))
    viscosity = 0.1
    diffusion_operator = StaggeredDiffusionOperator(
        mesh; viscosity, boundary = FreeSlip()
    )
    elliptic_operator = StaggeredEllipticOperator(
        diffusion_operator; diffusion = 0.05 * viscosity
    )
    elliptic_solver = StaggeredEllipticSolver(elliptic_operator)
    old_velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? sinpi(2x) : cospi(2y)
    )
    convected = copy(old_velocity)
    numerical = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    work = copy(numerical)
    dt = 0.1
    implicit_midpoint_diffusion!(
        numerical, convected, old_velocity, diffusion_operator,
        dt, elliptic_solver, work
    )

    dx = mesh.blocks[1].spacing[1]
    discrete_eigenvalue = -4sinpi(1 / 16)^2 / dx^2
    amplification = (1 + dt * viscosity * discrete_eigenvalue / 2) /
        (1 - dt * viscosity * discrete_eigenvalue / 2)
    for (index, face) in pairs(mesh.faces)
        initial = face.axis == 1 ? sinpi(2face.center[1]) :
            cospi(2face.center[2])
        @test numerical.interior[index] ≈ amplification * initial atol = 2.0e-13 rtol = 2.0e-13
    end
    @test elliptic_solver.operator.diffusion ≈ dt * viscosity / 2
    @test elliptic_solver isa StaggeredEllipticSolver
    @test elliptic_solver.x.krylov isa Krylov.CgWorkspace
    @test elliptic_solver.x.preconditioner isa
        TwoPhaseAMR.TreeStaggeredMultigridComponent
    @test elliptic_solver.y.preconditioner isa
        TwoPhaseAMR.TreeStaggeredMultigridComponent
    # A uniform root block has no block-tree level to prune.
    @test length(elliptic_solver.x.preconditioner.levels) == 1
    @test length(elliptic_solver.y.preconditioner.levels) == 1
    @test elliptic_solver.x.stats.converged
    @test elliptic_solver.y.stats.converged
    @test elliptic_solver.x.stats.initial_linf >=
        elliptic_solver.x.stats.final_linf
    @test elliptic_solver.x.stats.target_linf >=
        sqrt(eps()) * elliptic_solver.x.stats.residual_scale

    x_hierarchy = elliptic_solver.x.preconditioner
    y_hierarchy = elliptic_solver.y.preconditioner
    implicit_midpoint_diffusion!(
        numerical, convected, old_velocity, diffusion_operator,
        dt, elliptic_solver, work
    )
    @test elliptic_solver.x.preconditioner === x_hierarchy
    @test elliptic_solver.y.preconditioner === y_hierarchy
end

@testset "staggered base-mesh multigrid" begin
    for periodic in ((true, true), (true, false), (false, false))
        mesh = BlockTreeMesh(
            base_mesh = (16, 16), block_size = (4, 4),
            periodic = periodic, max_level = 1,
        )
        diffusion = StaggeredDiffusionOperator(
            mesh; viscosity = 0.2, boundary = FreeSlip(),
        )
        operator = StaggeredEllipticOperator(diffusion; diffusion = 0.03)
        solver = StaggeredEllipticSolver(operator; coarse_size = 16)
        hierarchy = solver.x.preconditioner
        @test [op.mesh.base_mesh for op in hierarchy.diffusion_operators] ==
            [(16, 16), (8, 8), (4, 4)]
        @test all(op -> op.mesh.periodic == periodic, hierarchy.diffusion_operators)
        @test length(solver.y.preconditioner.levels) == 3
        for index in 1:2
            fine = hierarchy.diffusion_operators[index].mesh
            coarse = hierarchy.diffusion_operators[index + 1].mesh
            transfer = hierarchy.face_prolongations[index]
            @test size(transfer) == (length(fine.faces), length(coarse.faces))
            if periodic == (true, true)
                @test transfer * ones(length(coarse.faces)) ≈
                    ones(length(fine.faces)) atol = 64eps()
                @test transpose(transfer) * fine.face_dual_volumes ≈
                    coarse.face_dual_volumes atol = 64eps()
            end
        end
        source = FaceVelocity(
            mesh, (x, y, time, axis) -> axis == 1 ?
                sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y),
        )
        destination = FaceVelocity(mesh)
        stats = solve!(destination, solver, source; tolerance = 1.0e-9, max_cycles = 100)
        @test stats.converged
        if periodic == (true, true)
            old_mesh = last(hierarchy.diffusion_operators).mesh
            old = TwoPhaseAMR.TreeStaggeredGeometricMultigridPreconditioner(
                operator, solver.x.preconditioner, solver.y.preconditioner,
            )
            marker = CellField(mesh)
            adaptation = AMRWorkspace(mesh, marker)
            labels = [
                block.key == BlockKey(0, 0, 0) ? :refine : :keep
                    for block in mesh.blocks
            ]
            @test adapt!(adaptation, labels; step = 1, force = true).changed
            new_diffusion = StaggeredDiffusionOperator(
                mesh; viscosity = 0.2, boundary = FreeSlip(),
            )
            new_operator = StaggeredEllipticOperator(new_diffusion; diffusion = 0.03)
            rebuilt = TwoPhaseAMR.TreeStaggeredGeometricMultigridPreconditioner(
                new_operator; coarse_size = 16, reuse = old,
            )
            @test rebuilt.x.reused_levels > 0
            @test rebuilt.y.reused_levels > 0
            @test last(rebuilt.x.diffusion_operators).mesh === old_mesh
        end
    end
    # Physical walls remove staggered unknowns. The base stopping test is
    # still in cells, not in component-face counts.
    wall_mesh = BlockTreeMesh(
        base_mesh = (8, 8), block_size = (4, 4), periodic = (false, false),
    )
    wall_diffusion = StaggeredDiffusionOperator(
        wall_mesh; viscosity = 0.2, boundary = FreeSlip(),
    )
    wall_operator = StaggeredEllipticOperator(wall_diffusion; diffusion = 0.03)
    @test length(wall_operator.x.volumes) < 60 < ncells(wall_mesh)
    wall_solver = StaggeredEllipticSolver(wall_operator; coarse_size = 60)
    @test [op.mesh.base_mesh for op in wall_solver.x.preconditioner.diffusion_operators] ==
        [(8, 8), (4, 4)]
    @test length(wall_solver.y.preconditioner.levels) == 2

    periodic_mesh = BlockTreeMesh(
        base_mesh = (4, 4), block_size = (2, 2), periodic = (true, true),
    )
    periodic_diffusion = StaggeredDiffusionOperator(
        periodic_mesh; viscosity = 0.2, boundary = FreeSlip(),
    )
    periodic_solver = StaggeredEllipticSolver(
        StaggeredEllipticOperator(periodic_diffusion; diffusion = 0.03);
        coarse_size = 1,
    )
    @test length(periodic_solver.x.preconditioner.levels) == 1
    @test length(periodic_solver.y.preconditioner.levels) == 1
end

@testset "dimensionless diffusion tolerance and failure report" begin
    mesh = BlockTreeMesh(block_size = (8, 8))
    diffusion = StaggeredDiffusionOperator(
        mesh; viscosity = 0.1, boundary = FreeSlip(),
    )
    operator = StaggeredEllipticOperator(diffusion; diffusion = 0.01)
    solver = StaggeredEllipticSolver(
        operator; multigrid = MultigridOptions(coarse_size = 8),
    )
    right_hand_side = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ? sinpi(2y) : cospi(2x),
    )
    solution = FaceVelocity(mesh)
    tolerance = 1.0e-3
    stats = solve!(
        solution, solver, right_hand_side; tolerance, max_cycles = 0,
    )
    @test !stats.converged
    @test stats.x.target_linf ≈ tolerance * stats.x.residual_scale
    @test stats.y.target_linf ≈ tolerance * stats.y.residual_scale

    destination = FaceVelocity(mesh)
    work = FaceVelocity(mesh)
    exception = try
        implicit_midpoint_diffusion!(
            destination, right_hand_side, right_hand_side,
            diffusion, 0.1, solver, work;
            tolerance = 1.0e-12, max_cycles = 0,
        )
        nothing
    catch caught
        caught
    end
    @test exception isa ErrorException
    message = sprint(showerror, exception)
    for field in (
            "tolerance=", "dimensionless_residual=", "max_cycles=0",
            "cycles=", "initial_l2=", "initial_linf=", "final_l2=",
            "final_linf=", "target_linf=", "residual_scale=",
            "removed_rhs_mean=", "method=", "preconditioner=",
        )
        @test occursin(field, message)
    end
end

@testset "external BiCGSTAB and staggered preconditioners" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 0),
    )
    diffusion = StaggeredDiffusionOperator(
        mesh; viscosity = 0.1, boundary = FreeSlip()
    )
    operator = StaggeredEllipticOperator(diffusion; diffusion = 0.1)
    solver = StaggeredEllipticSolver(operator; coarse_size = 16)
    right_hand_side = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y)
    )
    solution = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    stats = solve!(
        solution, solver, right_hand_side;
        tolerance = 1.0e-10, max_cycles = 80
    )
    @test stats.converged
    @test stats.x.method === :bicgstab
    @test stats.y.method === :bicgstab
    @test stats.x.preconditioner === :tree_staggered_geometric_multigrid
    @test stats.y.preconditioner === :tree_staggered_geometric_multigrid
    @test stats.x.cycles < 80
    @test stats.y.cycles < 80
    @test length(solver.x.preconditioner.levels) > 2
    @test length(solver.y.preconditioner.levels) > 2
    for (hierarchy, component, axis) in (
            (solver.x.preconditioner, solver.operator.x, 1),
            (solver.y.preconditioner, solver.operator.y, 2),
        )
        levels = hierarchy.levels
        @test first(levels).matrix === component.matrix
        @test all(
            level -> level.prolongation isa SparseMatrixCSC,
            levels
        )
        @test all(
            level -> nnz(level.prolongation) <=
                2size(level.prolongation, 1), levels
        )
        @test all(
            levels[index].matrix === hierarchy.operators[index].matrix
                for index in eachindex(levels)
        )
        for index in 2:length(levels)
            rediscretized = StaggeredEllipticOperator(
                hierarchy.diffusion_operators[index];
                reaction = operator.reaction,
                diffusion = operator.diffusion,
            )
            reference = axis == 1 ? rediscretized.x : rediscretized.y
            @test levels[index].matrix == reference.matrix
        end
        @test all(diff(length.(getproperty.(hierarchy.operators, :volumes))) .< 0)
        @test nblocks(last(hierarchy.diffusion_operators).mesh) == 1

        transfer = first(levels).prolongation
        fine = collect(range(0.1, 1.0; length = size(transfer, 1)))
        coarse = collect(range(-0.2, 0.7; length = size(transfer, 2)))
        prolonged = similar(fine)
        restricted = similar(coarse)
        TwoPhaseAMR._prolongate!(prolonged, transfer, coarse)
        TwoPhaseAMR._restrict!(restricted, transfer, fine)
        @test dot(fine, prolonged) ≈ dot(restricted, coarse)
    end
    residuals = residual_norms(solver)
    @test residuals.x[2] < 1.0e-9
    @test residuals.y[2] < 1.0e-9

    plain_solution = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    plain_solver = StaggeredEllipticSolver(
        operator; preconditioner = nothing
    )
    plain_stats = solve!(
        plain_solution, plain_solver, right_hand_side;
        tolerance = 1.0e-10, max_cycles = 500
    )
    @test plain_stats.converged
    @test plain_stats.x.preconditioner === :none
    @test maximum(abs, solution.interior - plain_solution.interior) < 2.0e-9

    for (method, workspace) in (
            (:gmres, Krylov.GmresWorkspace),
            (:fgmres, Krylov.FgmresWorkspace),
        )
        gmres_solution = FaceVelocity(
            mesh, (x, y, time, axis) -> 0.0
        )
        gmres_solver = StaggeredEllipticSolver(
            operator;
            krylov = (matrix, rhs) -> workspace(
                matrix, rhs; memory = 20
            ),
            krylov_options = (; restart = true, reorthogonalization = true),
            coarse_size = 16
        )
        gmres_stats = solve!(
            gmres_solution, gmres_solver, right_hand_side;
            tolerance = 1.0e-10, max_cycles = 80
        )
        @test gmres_stats.converged
        @test gmres_stats.x.method === method
        @test maximum(
            abs,
            gmres_solution.interior - plain_solution.interior
        ) < 2.0e-9
    end

    @test_throws ArgumentError StaggeredEllipticSolver(
        operator; preconditioner = :unsupported
    )
    @test_throws ArgumentError TreeStaggeredGeometricMultigridPreconditioner(
        operator; pruning = :unsupported
    )
end

@testset "boundary extrapolation cache follows mesh identity and epoch" begin
    boundary = VelocityBoundaryConditions(
        left = Inflow((x, y, t, axis) -> axis == 1 ? 1 : 0),
        right = Outflow(), bottom = FreeSlip(), top = FreeSlip(),
    )
    mesh = BlockTreeMesh(block_size = (4, 4), periodic = (false, false))
    sources = TwoPhaseAMR._boundary_extrapolation_schedule!(boundary, mesh)
    @test sources === TwoPhaseAMR._boundary_extrapolation_schedule!(boundary, mesh)
    @test all(index isa Int for face_sources in sources for index in face_sources)
    generation = mesh.topology_cache.generation
    TwoPhaseAMR._invalidate_mesh_topology_cache!(mesh)
    refreshed = TwoPhaseAMR._boundary_extrapolation_schedule!(boundary, mesh)
    @test refreshed !== sources
    @test refreshed == sources
    @test boundary.cache[].generation == generation + 1

    first_mesh = BlockTreeMesh(
        block_size = (4, 4), periodic = (false, false),
        extent = (2.0, 1.0)
    )
    second_mesh = BlockTreeMesh(
        block_size = (4, 4), periodic = (false, false),
        extent = (3.0, 1.0)
    )
    @test first_mesh.topology_cache.generation == second_mesh.topology_cache.generation
    first_sources = TwoPhaseAMR._boundary_extrapolation_schedule!(boundary, first_mesh)
    second_sources = TwoPhaseAMR._boundary_extrapolation_schedule!(boundary, second_mesh)
    @test first_sources !== second_sources
    @test boundary.cache[].topology_identity === second_mesh.topology_cache.identity

    mesh32 = BlockTreeMesh(
        block_size = (4, 4), periodic = (false, false),
        origin = (0.0f0, 0.0f0), extent = (2.0f0, 1.0f0)
    )
    other = TwoPhaseAMR._boundary_extrapolation_schedule!(boundary, mesh32)
    @test other !== second_sources
    @test boundary.cache[].topology_identity === mesh32.topology_cache.identity
    velocity = FaceVelocity(mesh32, (x, y, t, axis) -> axis == 1 ? x + 1 : 0)
    enforce_velocity_boundary!(velocity, boundary)
    net_flux = sum(
        face.side * face.area * velocity.boundary[index]
            for (index, face) in pairs(mesh32.boundary_faces)
    )
    @test abs(net_flux) <= 256eps(Float32)
end
