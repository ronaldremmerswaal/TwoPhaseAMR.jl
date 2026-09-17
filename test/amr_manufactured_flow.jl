include("amr_manufactured_problem.jl")

@testset "localized manufactured vortex demonstrates AMR benefit" begin
    cases = [_localized_vortex_case(level) for level in 1:3]
    equal_cell_uniform = _uniform_localized_vortex_case(48)
    errors = first.(cases)

    @test all(diff(errors) .< 0)
    @test all(log2.(errors[1:2] ./ errors[2:3]) .> 0.8)
    @test cases[3][3] < 0.6 * (8 * 2^3)^2
    @test abs(cases[3][3] - equal_cell_uniform[2]) /
        equal_cell_uniform[2] < 0.05
    @test cases[3][1] < equal_cell_uniform[1]
    @test cases[3][5] == 3
    @test all(case -> case[2] <= 2.0e-8, cases)
    @test all(case -> case[6].pressure.converged, cases)
    @test all(case -> iszero(case[6].mesh_updates), cases)
    @test cases[3][7]
end


@testset "Navier--Stokes rebuilds from uniform to mixed topology" begin
    problem = LocalizedManufacturedVortex(
        (0.37, 0.53), 8.0, 0.05, 1.0, 2.0e-3
    )
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 3,
        adaptation = MeshAdaptationOptions(
            update_interval = 1, coarsen_delay = typemax(Int)
        )
    )
    marker = CellField(mesh)
    initialize!(marker, (x, y) -> _localized_marker(problem, x, y))
    indicator = InterpolationErrorIndicator(
        marker; E_refine = 0.025, E_coarsen = 0.0, relative_floor = 0.02
    )
    velocity = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    pressure = CellField(mesh)
    solver = NavierStokesSolver(
        mesh; density = 1.0, dynamic_viscosity = problem.viscosity,
        boundary = FreeSlip(),
        coarse_size = 16, refinement_indicators = (indicator,)
    )
    workspace = NavierStokesWorkspace(solver)

    result = simulate!(
        velocity, pressure, solver, 3.0e-3, workspace;
        dt = 1.0e-3, rtol = 2.0e-10, max_cycles = 60
    )

    @test result.mesh_updates == 3
    @test sort!(unique(block.key.level for block in mesh.blocks)) == [2, 3]
    @test isbalanced(mesh)
    @test result.pressure.converged
end
