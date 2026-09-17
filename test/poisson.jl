using LinearAlgebra: Symmetric, dot, eigvals, ldiv!, mul!, opnorm
using SparseArrays: rowvals
using Krylov

function _cell_values(field::CellField)
    nx, ny = field.mesh.block_size
    return [
        interior(field, block)[i, j]
            for block in eachindex(field.mesh.blocks) for j in 1:ny for i in 1:nx
    ]
end

@testset "Factorized pressure action matches assembled operator" begin
    for topology in (:uniform, :mixed, :adapted)
        mesh = BlockTreeMesh(
            block_size = (4, 4), max_level = 2,
            refine = (key, origin, extent) -> topology == :mixed &&
                (
                key.level == 0 ||
                    (key.level == 1 && key.i == 0 && key.j == 0)
            ),
        )
        if topology == :adapted
            marker = CellField(mesh, (x, y) -> sinpi(2x) * sinpi(2y))
            adaptation = AMRWorkspace(mesh, marker)
            adapt!(adaptation, [:refine]; step = 1, force = true)
            labels = [
                block.key.level == 1 && block.key.i == 0 &&
                    block.key.j == 0 ? :refine : :keep
                    for block in mesh.blocks
            ]
            adapt!(adaptation, labels; step = 2, force = true)
        end
        coefficient = [
            1 + 0.3sin(index) for index in eachindex(mesh.faces)
        ]
        operator = EllipticOperator(
            mesh; reaction = 0.2, diffusion = 0.7,
            face_coefficient = coefficient,
        )
        action = TwoPhaseAMR.FactorizedEllipticOperator(operator)
        x = [sin(0.7index) for index in eachindex(operator.volumes)]
        y = [cos(0.3index) for index in eachindex(operator.volumes)]
        action_x = similar(x)
        action_y = similar(y)
        mul!(action_x, action, x)
        @test action_x ≈ operator.matrix * x rtol = 64eps() atol = 64eps()
        mul!(action_y, action, y)
        @test dot(x, action_y) ≈ dot(action_x, y) rtol = 128eps()
        @test dot(x, action_x) > 0
        mul!(action_x, action, x)
        @test @allocated(mul!(action_x, action, x)) == 0

        poisson = PoissonOperator(mesh; face_coefficient = coefficient)
        poisson_action = TwoPhaseAMR.FactorizedEllipticOperator(poisson)
        mul!(action_x, poisson_action, ones(length(x)))
        @test maximum(abs, action_x) <= 64eps()
    end
end

@testset "Rediscretized pressure hierarchy on pruned trees" begin
    @test GeometricMultigridPreconditioner ===
        TreeGeometricMultigridPreconditioner
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 3,
        refine = (key, origin, extent) ->
            key.level == 0 ||
                (key.level < 3 && origin[1] < 0.5 && origin[2] < 0.5),
    )
    coefficient = [
        exp(0.4sin(index)) for index in eachindex(mesh.faces)
    ]
    operator = PoissonOperator(mesh; face_coefficient = coefficient)
    all_terminal = TwoPhaseAMR.TreeGeometricMultigridPreconditioner(
        operator; coarse_size = 64, pruning = :all,
    )
    deepest = TwoPhaseAMR.TreeGeometricMultigridPreconditioner(
        operator; coarse_size = 64, pruning = :deepest,
    )
    @test length(all_terminal.levels) > 2
    @test length(all_terminal.operators[2].volumes) <
        length(deepest.operators[2].volumes)
    @test all(diff(length.(getproperty.(all_terminal.operators, :volumes))) .< 0)

    for index in 1:(length(all_terminal.operators) - 1)
        fine = all_terminal.operators[index]
        coarse = all_terminal.operators[index + 1]
        expected_coefficient = similar(coarse.face_coefficient)
        TwoPhaseAMR._restrict_face_coefficients!(
            expected_coefficient, fine.face_coefficient,
            all_terminal.coefficient_restrictions[index],
        )
        @test coarse.face_coefficient ≈ expected_coefficient rtol = 16eps()
        reference = PoissonOperator(
            coarse.mesh; face_coefficient = expected_coefficient,
        )
        @test coarse.matrix ≈ reference.matrix rtol = 64eps()
        @test maximum(abs, coarse.matrix * ones(length(coarse.volumes))) <=
            1024eps() * max(opnorm(Matrix(coarse.matrix), Inf), 1.0)
    end

    refreshed = TwoPhaseAMR._replace_face_coefficient(
        operator,
        [exp(0.6cos(index)) for index in eachindex(mesh.faces)],
    )
    identity = all_terminal
    TwoPhaseAMR._refresh_geometric_multigrid!(all_terminal, refreshed)
    @test all_terminal === identity
    right_hand_side = [
        sin(0.2index) for index in eachindex(operator.volumes)
    ]
    destination = similar(right_hand_side)
    ldiv!(destination, all_terminal, right_hand_side)
    @test @allocated(
        ldiv!(destination, all_terminal, right_hand_side)
    ) == 0

    large_block = BlockTreeMesh(block_size = (8, 8))
    within_block = TwoPhaseAMR.TreeGeometricMultigridPreconditioner(
        PoissonOperator(large_block); coarse_size = 16,
    )
    @test within_block isa TreeGeometricMultigridPreconditioner
    @test length(within_block.levels) == 1
    @test first(within_block.operators).mesh === large_block
end

function _poisson_error(block_size; mixed = true, rtol = 1.0e-10)
    mesh = mixed ? BlockTreeMesh(
            block_size = (block_size, block_size), max_level = 2,
            refine = (key, origin, extent) ->
            key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
        ) : BlockTreeMesh(block_size = (block_size, block_size))
    exact_function = (x, y) -> sinpi(2x) * cospi(2y)
    rhs = CellField(mesh)
    initialize!(rhs, (x, y) -> 8pi^2 * exact_function(x, y))
    pressure = CellField(mesh)
    solver = EllipticSolver(PoissonOperator(mesh); coarse_size = 8)
    stats = solve!(pressure, solver, rhs; rtol, max_cycles = 60)

    squared_error = 0.0
    maximum_error = 0.0
    total_volume = 0.0
    nx, ny = mesh.block_size
    for (block_index, block) in pairs(mesh.blocks), j in 1:ny, i in 1:nx
        x, y = cellcenter(mesh, block_index, i, j)
        error = abs(interior(pressure, block_index)[i, j] - exact_function(x, y))
        squared_error += block.volume * error^2
        maximum_error = max(maximum_error, error)
        total_volume += block.volume
    end
    return (
        l2 = sqrt(squared_error / total_volume), linf = maximum_error,
        stats, levels = length(solver.preconditioner.levels),
    )
end

@testset "elliptic operator preserves gradient-divergence duality" begin
    for mesh in (
            BlockTreeMesh(block_size = (8, 8)),
            BlockTreeMesh(
                block_size = (4, 4), max_level = 2,
                refine = (key, origin, extent) ->
                key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
            ),
        )
        pressure = CellField(mesh)
        initialize!(
            pressure,
            (x, y) -> sinpi(2x) * cospi(2y) + 0.1cospi(4x - 2y)
        )
        operator = PoissonOperator(mesh)
        workspace = EllipticWorkspace(operator)
        applied = CellField(mesh)
        apply!(applied, operator, pressure, workspace)

        face_gradient = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
        gradient!(face_gradient, pressure, operator.gradient_operator)
        divergence = CellField(mesh)
        divergence!(divergence, face_gradient)
        @test _cell_values(applied) ≈ -_cell_values(divergence) atol = 8.0e-13 rtol = 8.0e-13

        @test operator.matrix ≈ transpose(operator.matrix) atol = 4.0e-14 rtol = 4.0e-14
        @test maximum(abs, operator.matrix * ones(ncells(mesh))) <= 2.0e-13
    end
end

@testset "variable-coefficient elliptic operator" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
    )
    cell_coefficient = CellField(mesh)
    initialize!(cell_coefficient, (x, y) -> x < 0.5 ? 1.0 : 25.0)
    face_coefficient = harmonic_face_coefficients(cell_coefficient)
    @test all(>(0), face_coefficient)
    @test minimum(face_coefficient) >= 1
    @test maximum(face_coefficient) <= 25

    operator = PoissonOperator(mesh; face_coefficient)
    pressure = CellField(mesh)
    initialize!(pressure, (x, y) -> sinpi(2x) * cospi(2y))
    workspace = EllipticWorkspace(operator)
    applied = CellField(mesh)
    apply!(applied, operator, pressure, workspace)
    weighted = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    weighted_gradient!(weighted, pressure, operator, workspace)
    divergence = CellField(mesh)
    divergence!(divergence, weighted)
    @test _cell_values(applied) ≈ -_cell_values(divergence) atol = 2.0e-11 rtol = 2.0e-13
    @test operator.matrix ≈ transpose(operator.matrix) atol = 2.0e-13 rtol = 2.0e-13

    eigenvalues = eigvals(Symmetric(Matrix(operator.matrix)))
    tolerance = 5.0e-12 * maximum(eigenvalues)
    @test first(eigenvalues) >= -tolerance
    @test count(value -> abs(value) <= tolerance, eigenvalues) == 1
end

@testset "geometric multigrid Poisson convergence" begin
    results = _poisson_error.((4, 8, 16))
    @test all(result -> result.stats.converged, results)
    @test maximum(getproperty.(getproperty.(results, :stats), :cycles)) -
        minimum(getproperty.(getproperty.(results, :stats), :cycles)) <= 3
    for norm_name in (:l2, :linf)
        errors = collect(getproperty.(results, norm_name))
        rates = log2.(errors[1:2] ./ errors[2:3])
        @test all(diff(errors) .< 0)
        @test minimum(rates) > 1.7
    end
end

@testset "external Krylov solver and independent preconditioner" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 4,
        refine = (key, origin, extent) ->
        key.level == 0 ||
            (key.level < 4 && origin[1] < 0.5 && origin[2] < 0.5),
    )
    rhs = CellField(mesh)
    initialize!(
        rhs, (x, y) ->
        8pi^2 * sinpi(2x) * cospi(2y) +
            2pi^2 * sinpi(2x + 2y)
    )
    operator = PoissonOperator(mesh)

    preconditioner = TreeGeometricMultigridPreconditioner(
        operator; coarse_size = 32
    )
    krylov = Krylov.CgWorkspace(operator.matrix, zeros(ncells(mesh)))
    solver = EllipticSolver(operator, krylov, preconditioner)
    solution = CellField(mesh)
    stats = solve!(solution, solver, rhs; rtol = 1.0e-10, max_cycles = 100)

    @test stats.converged
    @test stats.method === :cg
    @test stats.preconditioner === :tree_geometric_multigrid
    @test solver.krylov === krylov
    @test solver.preconditioner === preconditioner

    default_solver = EllipticSolver(operator; coarse_size = 32)
    default_solution = CellField(mesh)
    default_stats = solve!(
        default_solution, default_solver, rhs; rtol = 1.0e-10, max_cycles = 100
    )
    @test default_stats.converged
    @test default_stats.method === :cg
    @test maximum(
        abs,
        _cell_values(default_solution) - _cell_values(solution)
    ) < 2.0e-9

    no_preconditioner = EllipticSolver(operator; preconditioner = nothing)
    plain_solution = CellField(mesh)
    plain_stats = solve!(
        plain_solution, no_preconditioner, rhs; rtol = 1.0e-8, max_cycles = 500
    )
    @test plain_stats.converged
    @test plain_stats.preconditioner === :none
    @test stats.cycles < plain_stats.cycles
end

@testset "Poisson solve is independent of uniform block partitioning" begin
    one_block = BlockTreeMesh(block_size = (16, 16))
    four_blocks = BlockTreeMesh(
        block_size = (8, 8), max_level = 1,
        refine = (key, origin, extent) -> true,
    )
    function solve_partition(mesh)
        rhs = CellField(mesh)
        initialize!(rhs, (x, y) -> 8pi^2 * sinpi(2x) * cospi(2y))
        pressure = CellField(mesh)
        solver = EllipticSolver(PoissonOperator(mesh); coarse_size = 8)
        stats = solve!(pressure, solver, rhs; rtol = 2.0e-12)
        values = Dict{Tuple{Float64, Float64}, Float64}()
        nx, ny = mesh.block_size
        for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
            values[cellcenter(mesh, block, i, j)] = interior(pressure, block)[i, j]
        end
        return values, stats
    end
    first_values, first_stats = solve_partition(one_block)
    second_values, second_stats = solve_partition(four_blocks)
    @test first_stats.converged && second_stats.converged
    @test keys(first_values) == keys(second_values)
    @test maximum(
        abs(first_values[key] - second_values[key])
            for key in keys(first_values)
    ) <= 2.0e-12
end

@testset "periodic nullspace convention" begin
    mesh = BlockTreeMesh(block_size = (8, 8))
    rhs = CellField(mesh; init = 3.0)
    pressure = CellField(mesh; init = 7.0)
    solver = EllipticSolver(PoissonOperator(mesh); coarse_size = 4)
    @test_throws ArgumentError solve!(pressure, solver, rhs; compatibility = :error)
    stats = solve!(pressure, solver, rhs; compatibility = :project)
    @test stats.converged
    @test stats.removed_rhs_mean ≈ 3
    @test abs(integrate(pressure)) <= 16eps()
    @test maximum(abs, _cell_values(pressure)) <= 16eps()
end

@testset "shifted operator supports implicit diffusion" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 1,
        refine = (key, origin, extent) -> true,
    )
    operator = EllipticOperator(mesh; reaction = 1, diffusion = 0.02)
    workspace = EllipticWorkspace(operator)
    exact = CellField(mesh)
    initialize!(exact, (x, y) -> 0.4 + sinpi(2x) * cospi(2y))
    rhs = CellField(mesh)
    apply!(rhs, operator, exact, workspace)
    numerical = CellField(mesh)
    solver = EllipticSolver(operator; coarse_size = 8)
    stats = solve!(numerical, solver, rhs; rtol = 2.0e-12, max_cycles = 60)
    @test stats.converged
    @test maximum(abs, _cell_values(numerical) - _cell_values(exact)) <= 2.0e-11
end

@testset "pressure projection" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
    )
    initial_velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y),
    )
    divergence = CellField(mesh)
    divergence!(divergence, initial_velocity)
    initial_divergence = maximum(abs, _cell_values(divergence))

    coefficient = CellField(mesh)
    initialize!(coefficient, (x, y) -> x < 0.5 ? 1.0 : 100.0)
    for face_coefficient in (
            ones(length(mesh.faces)),
            harmonic_face_coefficients(coefficient),
        )
        velocity = copy(initial_velocity)
        pressure = CellField(mesh)
        solver = EllipticSolver(
            PoissonOperator(mesh; face_coefficient); coarse_size = 8
        )
        projection_workspace = ProjectionWorkspace(solver)
        stats = project!(
            velocity, pressure, solver, 0.1,
            projection_workspace; tolerance = 1.0e-11
        )
        divergence!(divergence, velocity)
        final_divergence = maximum(abs, _cell_values(divergence))
        @test stats.converged
        @test stats.initial_linf >= stats.final_linf
        @test stats.residual_scale ≈ 100
        @test stats.final_linf / stats.residual_scale <= 1.0e-11
        @test 0.1 * final_divergence ≈
            stats.final_linf / stats.residual_scale atol = eps() rtol = 0
        @test final_divergence <= 2.0e-11 * initial_divergence
        @test abs(integrate(pressure)) <= 2.0e-12
    end
end

@testset "dimensionless pressure tolerance validation" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    velocity = FaceVelocity(mesh)
    pressure = CellField(mesh)
    solver = EllipticSolver(PoissonOperator(mesh); coarse_size = 4)
    workspace = ProjectionWorkspace(solver)
    for invalid in (0.0, -eps(), 1.0, Inf, NaN)
        @test_throws ArgumentError project!(
            velocity, pressure, solver, 0.1, workspace;
            tolerance = invalid,
        )
    end
end

@testset "mixed-level no-permeation projection" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), periodic = (false, false), max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 0),
    )
    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ?
            0.3 + sinpi(2x) * cospi(2y) :
            -0.2 + cospi(2x) * sinpi(2y)
    )
    pressure = CellField(mesh)
    solver = EllipticSolver(
        PoissonOperator(mesh); coarse_size = 8
    )
    workspace = ProjectionWorkspace(solver)
    stats = project!(
        velocity, pressure, solver, 0.05, workspace;
        tolerance = 1.0e-10, max_cycles = 80
    )
    divergence = CellField(mesh)
    divergence!(divergence, velocity)
    @test stats.converged
    @test maximum(abs, _cell_values(divergence)) <= 1.0e-8
    @test all(iszero, velocity.boundary)
    @test abs(integrate(pressure)) <= 2.0e-11
end

@testset "elliptic apply allocation" begin
    mesh = BlockTreeMesh(block_size = (8, 8))
    operator = PoissonOperator(mesh)
    workspace = EllipticWorkspace(operator)
    source = CellField(mesh; init = 1.0)
    destination = CellField(mesh)
    apply!(destination, operator, source, workspace)
    @test @allocated(apply!(destination, operator, source, workspace)) == 0

    solver = EllipticSolver(operator; coarse_size = 4)
    initialize!(source, (x, y) -> sinpi(2x) * cospi(2y))
    solve!(destination, solver, source; rtol = 1.0e-8)
    fill!(interior(destination, 1), 0)
    @test @allocated(solve!(destination, solver, source; rtol = 1.0e-8)) == 0
end
