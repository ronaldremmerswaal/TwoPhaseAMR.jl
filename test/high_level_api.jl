@testset "high-level FlowSimulation API" begin
    mesh = BlockTreeMesh(block_size = (4, 4))

    zero_velocity = FaceVelocity(mesh)
    @test zero_velocity isa AbstractMeshField{Float64}
    @test all(iszero, zero_velocity.interior)
    @test all(iszero, zero_velocity.boundary)

    sampled_pressure = CellField(mesh, (x, y) -> x + 2y)
    for block in eachindex(mesh.blocks), j in 1:4, i in 1:4
        x, y = cellcenter(mesh, block, i, j)
        @test interior(sampled_pressure, block)[i, j] == x + 2y
    end

    flow = FlowSimulation(
        mesh; density = 2.0, dynamic_viscosity = 0.2,
        velocity = (0.1, -0.2), pressure = 2,
        boundary = FreeSlip(), coarse_size = 8, dt = 0.01
    )
    @test isnothing(flow.interface)
    @test flow.diffusion_operator === flow.solver.diffusion
    @test flow isa NavierStokesSimulation
    @test flow.velocity.mesh === mesh
    @test flow.pressure.mesh === mesh
    @test flow.solver.mesh === mesh
    @test flow.solver.pressure_preconditioner === :tree_geometric
    @test flow.solver.diffusion_preconditioner === :tree_geometric
    @test flow.solver.density === 2.0
    @test flow.solver.dynamic_viscosity === 0.2
    @test flow.solver.kinematic_viscosity === 0.1
    @test all(==(0.5), flow.solver.pressure_solver.operator.face_coefficient)
    @test flow.workspace.old_velocity.mesh === mesh
    @test flow.solver.amr_interface_scheme isa MC
    @test flow.default_dt == 0.01
    @test flow.workspace.elliptic.operator.diffusion == 0.0005
    @test all(==(2.0), first(flow.pressure.blocks))
    @test occursin("FlowSimulation(time=0.0", sprint(show, flow))
    @test stable_timestep(flow; cfl = 0.5) > 0

    first_result = simulate!(
        flow, 0.01; rtol = 1.0e-10, max_cycles = 20
    )
    @test first_result.time == flow.time == 0.01
    @test first_result.steps == 1
    @test first_result.total_steps == flow.steps == 1
    @test first_result.total_mesh_updates == flow.mesh_updates == 0
    @test first_result.pressure.converged

    second_result = simulate!(
        flow, 0.02; rtol = 1.0e-10, max_cycles = 20
    )
    @test second_result.steps == 1
    @test second_result.total_steps == flow.steps == 2
    @test flow.time == 0.02

    velocity = FaceVelocity(mesh, Float32; init = 0.0f0)
    pressure = CellField(mesh, Float32)
    solver = NavierStokesSolver(
        mesh; density = 1.0f0, dynamic_viscosity = 0.1f0,
        T = Float32, coarse_size = 8
    )
    workspace = NavierStokesWorkspace(solver)
    advanced = FlowSimulation(
        velocity, pressure, solver; workspace, time = 1, steps = 3,
        mesh_updates = 1
    )
    @test advanced.velocity === velocity
    @test advanced.pressure === pressure
    @test advanced.solver === solver
    @test advanced.workspace === workspace
    @test advanced.time === 1.0f0
    @test advanced.steps == 3
    @test advanced.mesh_updates == 1
    @test isnothing(advanced.default_dt)

    inferred = FlowSimulation(
        mesh; density = 1.0, dynamic_viscosity = 0.1,
        velocity, pressure = CellField(mesh, Float32),
        coarse_size = 8
    )
    @test eltype(inferred.velocity) === Float32
    @test typeof(inferred.solver.kinematic_viscosity) === Float32

    other_mesh = BlockTreeMesh(block_size = (4, 4))
    @test_throws ArgumentError FlowSimulation(
        mesh; density = 1.0, dynamic_viscosity = 0.1,
        velocity = (1.0,), coarse_size = 8
    )
    @test_throws ArgumentError FlowSimulation(
        mesh; density = 1.0, dynamic_viscosity = 0.1,
        velocity = :invalid, coarse_size = 8
    )
    @test_throws ArgumentError FlowSimulation(
        mesh; density = 1.0, dynamic_viscosity = 0.1,
        pressure = :invalid, coarse_size = 8
    )
    @test_throws ArgumentError FlowSimulation(
        mesh; density = 1.0, dynamic_viscosity = 0.1,
        velocity = FaceVelocity(other_mesh),
        coarse_size = 8
    )
    interface = InterfaceField(mesh, (x, y) -> x - 0.5; lipschitz = 1)
    @test_throws ArgumentError FlowSimulation(
        mesh; density = (1.0, 1.0), dynamic_viscosity = (0.02, 0.001)
    )
    @test_throws ArgumentError FlowSimulation(
        mesh; density = 1.0, dynamic_viscosity = (0.02, 0.001)
    )
    @test_throws ArgumentError FlowSimulation(
        mesh; interface, density = 1.0,
        dynamic_viscosity = (0.02, 0.001)
    )
    @test_throws ArgumentError FlowSimulation(
        mesh; density = 1.0, dynamic_viscosity = 0.1,
        velocity = FaceVelocity(mesh, Float32),
        pressure = CellField(mesh, Float64), coarse_size = 8
    )
    @test_throws ArgumentError FlowSimulation(
        mesh; density = 1.0, dynamic_viscosity = 0.1,
        T = Int, coarse_size = 8
    )
    @test_throws ArgumentError FlowSimulation(
        mesh; density = 1.0, dynamic_viscosity = 0.1,
        time = Inf, coarse_size = 8
    )
    @test_throws ArgumentError FlowSimulation(
        mesh; density = 1.0, dynamic_viscosity = 0.1,
        dt = 0, coarse_size = 8
    )
    @test_throws ArgumentError FlowSimulation(
        mesh; density = 1.0, dynamic_viscosity = 0.1,
        dt = Inf, coarse_size = 8
    )
    @test_throws ArgumentError FlowSimulation(
        mesh; density = 1.0, dynamic_viscosity = 0.1,
        dt = "invalid", coarse_size = 8
    )
    @test_throws ArgumentError FlowSimulation(
        velocity, pressure, solver; dt = -0.01
    )
    @test_throws ArgumentError NavierStokesWorkspace(solver; dt = 0)
    @test_throws ArgumentError NavierStokesWorkspace(solver; dt = Inf)
    @test_throws ArgumentError NavierStokesWorkspace(solver; dt = "invalid")
    @test_throws ArgumentError FlowSimulation(
        velocity, pressure, solver; steps = -1
    )
    @test_throws ArgumentError FlowSimulation(
        velocity, pressure, solver; steps = 0.5
    )
    @test_throws ArgumentError FlowSimulation(
        velocity, pressure, solver; steps = true
    )
    @test_throws ArgumentError FlowSimulation(
        velocity, pressure, solver; mesh_updates = -1
    )
    @test_throws ArgumentError FlowSimulation(
        velocity, pressure, solver;
        workspace = NavierStokesWorkspace(
            NavierStokesSolver(
                other_mesh; density = 1.0f0, dynamic_viscosity = 0.1f0,
                T = Float32, coarse_size = 8
            )
        )
    )
    @test_throws ArgumentError simulate!(flow, 0.01)
    @test_throws ArgumentError simulate!(flow, 0.03; time = 0.0)
    @test_throws ArgumentError simulate!(inferred, Inf; dt = 0.01)
    @test_throws ArgumentError simulate!(inferred, "invalid"; dt = 0.01)
    @test_throws ArgumentError step!(
        velocity, pressure, solver, Inf, workspace
    )
end

@testset "FlowSimulation preserves Float32 through a regrid" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 1,
        adaptation = MeshAdaptationOptions(coarsen_delay = 0)
    )
    marker = CellField(mesh, Float32)
    initialize!(marker, (x, y) -> 1.0f0 + 0.2f0 * sinpi(4x) * sinpi(4y))
    criterion = InterpolationErrorIndicator(
        marker; E_refine = 1.0f-3, E_coarsen = 1.0f-4
    )
    flow = FlowSimulation(
        mesh; density = 1.0f0, dynamic_viscosity = 0.01f0,
        T = Float32, dt = 0.01f0,
        boundary = FreeSlip(), coarse_size = 8,
        refinement_indicators = criterion
    )

    result = simulate!(flow, 0.01f0; rtol = 1.0f-5, max_cycles = 40)
    @test result.mesh_updates == 1
    @test nblocks(mesh) == 4
    @test eltype(flow.velocity) === Float32
    @test eltype(flow.pressure) === Float32
    @test typeof(flow.solver.kinematic_viscosity) === Float32
    @test eltype(flow.workspace.old_velocity) === Float32
    @test flow.workspace.elliptic.operator.diffusion === 5.0f-5
end
