using LinearAlgebra: Symmetric, eigvals

function _cell_values(field::CellField)
    nx, ny = field.mesh.block_size
    return [interior(field, block)[i, j]
            for block in eachindex(field.mesh.blocks) for j in 1:ny for i in 1:nx]
end

function _poisson_error(block_size; mixed=true, rtol=1e-10)
    mesh = mixed ? BlockTreeMesh(
        block_size=(block_size, block_size), max_level=2,
        refine=(key, origin, extent) ->
            key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
    ) : BlockTreeMesh(block_size=(block_size, block_size))
    exact_function = (x, y) -> sinpi(2x) * cospi(2y)
    rhs = CellField(mesh)
    initialize!(rhs, (x, y) -> 8pi^2 * exact_function(x, y))
    pressure = CellField(mesh)
    solver = GeometricMultigridSolver(PoissonOperator(mesh); coarse_size=8)
    stats = solve!(pressure, solver, rhs; rtol, max_cycles=60)

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
    return (l2=sqrt(squared_error / total_volume), linf=maximum_error,
            stats, levels=length(solver.levels))
end

@testset "elliptic operator preserves gradient-divergence duality" begin
    for mesh in (
        BlockTreeMesh(block_size=(8, 8)),
        BlockTreeMesh(
            block_size=(4, 4), max_level=2,
            refine=(key, origin, extent) ->
                key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
        ),
    )
        pressure = CellField(mesh)
        initialize!(pressure,
                    (x, y) -> sinpi(2x) * cospi(2y) + 0.1cospi(4x - 2y))
        operator = PoissonOperator(mesh)
        workspace = EllipticWorkspace(operator)
        applied = CellField(mesh)
        apply!(applied, operator, pressure, workspace)

        face_gradient = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
        gradient!(face_gradient, pressure, operator.gradient_operator)
        divergence = CellField(mesh)
        divergence!(divergence, face_gradient)
        @test _cell_values(applied) ≈ -_cell_values(divergence) atol=8e-13 rtol=8e-13

        @test operator.matrix ≈ transpose(operator.matrix) atol=4e-14 rtol=4e-14
        @test maximum(abs, operator.matrix * ones(ncells(mesh))) <= 2e-13
    end
end

@testset "variable-coefficient elliptic operator" begin
    mesh = BlockTreeMesh(
        block_size=(4, 4), max_level=2,
        refine=(key, origin, extent) ->
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
    @test _cell_values(applied) ≈ -_cell_values(divergence) atol=2e-11 rtol=2e-13
    @test operator.matrix ≈ transpose(operator.matrix) atol=2e-13 rtol=2e-13

    eigenvalues = eigvals(Symmetric(Matrix(operator.matrix)))
    tolerance = 5e-12 * maximum(eigenvalues)
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

@testset "Poisson solve is independent of uniform block partitioning" begin
    one_block = BlockTreeMesh(block_size=(16, 16))
    four_blocks = BlockTreeMesh(
        block_size=(8, 8), max_level=1,
        refine=(key, origin, extent) -> true,
    )
    function solve_partition(mesh)
        rhs = CellField(mesh)
        initialize!(rhs, (x, y) -> 8pi^2 * sinpi(2x) * cospi(2y))
        pressure = CellField(mesh)
        solver = GeometricMultigridSolver(PoissonOperator(mesh); coarse_size=8)
        stats = solve!(pressure, solver, rhs; rtol=2e-12)
        values = Dict{Tuple{Float64,Float64},Float64}()
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
    @test maximum(abs(first_values[key] - second_values[key])
                  for key in keys(first_values)) <= 2e-12
end

@testset "periodic nullspace convention" begin
    mesh = BlockTreeMesh(block_size=(8, 8))
    rhs = CellField(mesh; init=3.0)
    pressure = CellField(mesh; init=7.0)
    solver = GeometricMultigridSolver(PoissonOperator(mesh); coarse_size=4)
    @test_throws ArgumentError solve!(pressure, solver, rhs; compatibility=:error)
    stats = solve!(pressure, solver, rhs; compatibility=:project)
    @test stats.converged
    @test stats.removed_rhs_mean ≈ 3
    @test abs(integrate(pressure)) <= 16eps()
    @test maximum(abs, _cell_values(pressure)) <= 16eps()
end

@testset "shifted operator supports implicit diffusion" begin
    mesh = BlockTreeMesh(
        block_size=(8, 8), max_level=1,
        refine=(key, origin, extent) -> true,
    )
    operator = EllipticOperator(mesh; reaction=1, diffusion=0.02)
    workspace = EllipticWorkspace(operator)
    exact = CellField(mesh)
    initialize!(exact, (x, y) -> 0.4 + sinpi(2x) * cospi(2y))
    rhs = CellField(mesh)
    apply!(rhs, operator, exact, workspace)
    numerical = CellField(mesh)
    solver = GeometricMultigridSolver(operator; coarse_size=8)
    stats = solve!(numerical, solver, rhs; rtol=2e-12, max_cycles=60)
    @test stats.converged
    @test maximum(abs, _cell_values(numerical) - _cell_values(exact)) <= 2e-11
end

@testset "pressure projection" begin
    mesh = BlockTreeMesh(
        block_size=(8, 8), max_level=2,
        refine=(key, origin, extent) ->
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
    for face_coefficient in (ones(length(mesh.faces)),
                             harmonic_face_coefficients(coefficient))
        velocity = copy(initial_velocity)
        pressure = CellField(mesh)
        solver = GeometricMultigridSolver(
            PoissonOperator(mesh; face_coefficient); coarse_size=8)
        projection_workspace = ProjectionWorkspace(solver)
        stats = project!(velocity, pressure, solver, 0.1,
                         projection_workspace; rtol=1e-11)
        divergence!(divergence, velocity)
        final_divergence = maximum(abs, _cell_values(divergence))
        @test stats.converged
        @test final_divergence <= 2e-11 * initial_divergence
        @test abs(integrate(pressure)) <= 2e-12
    end
end

@testset "mixed-level no-permeation projection" begin
    mesh = BlockTreeMesh(
        block_size=(4, 4), periodic=(false, false), max_level=2,
        refine=(key, origin, extent) ->
            key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 0),
    )
    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ?
            0.3 + sinpi(2x) * cospi(2y) :
            -0.2 + cospi(2x) * sinpi(2y))
    pressure = CellField(mesh)
    solver = GeometricMultigridSolver(
        PoissonOperator(mesh); coarse_size=8)
    workspace = ProjectionWorkspace(solver)
    stats = project!(velocity, pressure, solver, 0.05, workspace;
                     rtol=1e-10, max_cycles=80)
    divergence = CellField(mesh)
    divergence!(divergence, velocity)
    @test stats.converged
    @test maximum(abs, _cell_values(divergence)) <= 1e-8
    @test all(iszero, velocity.boundary)
    @test abs(integrate(pressure)) <= 2e-11
end

@testset "elliptic apply allocation" begin
    mesh = BlockTreeMesh(block_size=(8, 8))
    operator = PoissonOperator(mesh)
    workspace = EllipticWorkspace(operator)
    source = CellField(mesh; init=1.0)
    destination = CellField(mesh)
    apply!(destination, operator, source, workspace)
    @test @allocated(apply!(destination, operator, source, workspace)) == 0

    solver = GeometricMultigridSolver(operator; coarse_size=4)
    initialize!(source, (x, y) -> sinpi(2x) * cospi(2y))
    solve!(destination, solver, source; rtol=1e-8)
    fill!(interior(destination, 1), 0)
    @test @allocated(solve!(destination, solver, source; rtol=1e-8)) == 0
end
