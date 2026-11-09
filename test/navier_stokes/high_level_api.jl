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

    numerics = NavierStokesOptions(multigrid = MultigridOptions(coarse_size = 8))
    flow = FlowSimulation(
        mesh, SinglePhase(density = 2.0, dynamic_viscosity = 0.2);
        velocity = (0.1, -0.2), pressure = 2,
        boundary = FreeSlip(), numerics, dt = 0.01,
    )
    @test isnothing(flow.interface)
    @test flow.diffusion_operator === flow.solver.diffusion
    @test flow isa FlowSimulation
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
    @test flow.solver.amr_interface_scheme isa MonotonizedCentral
    @test flow.default_dt == 0.01
    @test flow.workspace.elliptic.operator.diffusion == 0.0005
    @test all(==(2.0), first(flow.pressure.blocks))
    @test stable_timestep(flow; cfl = 0.5) > 0

    first_result = simulate!(
        flow, 0.01; tolerance = 1.0e-10, max_cycles = 20
    )
    @test first_result.time == flow.time == 0.01
    @test first_result.steps == 1
    @test first_result.total_steps == flow.steps == 1
    @test first_result.total_mesh_updates == flow.mesh_updates == 0
    @test first_result.pressure.converged

    second_result = simulate!(
        flow, 0.02; tolerance = 1.0e-10, max_cycles = 20
    )
    @test second_result.steps == 1
    @test second_result.total_steps == flow.steps == 2
    @test flow.time == 0.02

    velocity = FaceVelocity(mesh, Float32; init = 0.0f0)
    pressure = CellField(mesh, Float32)
    solver = IncompressibleNavierStokesSolver(
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

    single = SinglePhase(density = 1.0, dynamic_viscosity = 0.1)
    inferred = FlowSimulation(
        mesh, single; velocity, pressure = CellField(mesh, Float32), numerics,
    )
    @test eltype(inferred.velocity) === Float32
    @test typeof(inferred.solver.kinematic_viscosity) === Float32

    other_mesh = BlockTreeMesh(block_size = (4, 4))
    @test_throws ArgumentError FlowSimulation(
        mesh, single; velocity = (1.0,), numerics,
    )
    @test_throws ArgumentError FlowSimulation(
        mesh, single; velocity = :invalid, numerics,
    )
    @test_throws ArgumentError FlowSimulation(
        mesh, single; pressure = :invalid, numerics,
    )
    @test_throws ArgumentError FlowSimulation(
        mesh, single; velocity = FaceVelocity(other_mesh), numerics,
    )
    interface = InterfaceField(mesh, (x, y) -> x - 0.5; lipschitz = 1)
    @test_throws ArgumentError SinglePhase(
        density = (1.0, 1.0), dynamic_viscosity = (0.02, 0.001),
    )
    @test_throws ArgumentError SinglePhase(
        density = 1.0, dynamic_viscosity = (0.02, 0.001),
    )
    @test_throws ArgumentError TwoPhase(
        interface; density = 1.0, dynamic_viscosity = (0.02, 0.001),
    )
    @test_throws ArgumentError FlowSimulation(
        mesh, single; velocity = FaceVelocity(mesh, Float32),
        pressure = CellField(mesh, Float64), numerics,
    )
    @test_throws ArgumentError FlowSimulation(
        mesh, single; T = Int, numerics,
    )
    @test_throws ArgumentError FlowSimulation(
        mesh, single; time = Inf, numerics,
    )
    @test_throws ArgumentError FlowSimulation(
        mesh, single; dt = 0, numerics,
    )
    @test_throws ArgumentError FlowSimulation(
        mesh, single; dt = Inf, numerics,
    )
    @test_throws ArgumentError FlowSimulation(
        mesh, single; dt = "invalid", numerics,
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
            IncompressibleNavierStokesSolver(
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
        mesh, SinglePhase(density = 2.0f0, dynamic_viscosity = 0.02f0);
        T = Float32, dt = 0.01f0, boundary = FreeSlip(),
        numerics = NavierStokesOptions(multigrid = MultigridOptions(coarse_size = 8)),
        adaptivity = Adaptivity(criteria = criterion),
    )

    result = simulate!(flow, 0.01f0; tolerance = 1.0f-5, max_cycles = 40)
    @test result.mesh_updates == 1
    @test nblocks(mesh) == 4
    @test eltype(flow.velocity) === Float32
    @test eltype(flow.pressure) === Float32
    @test typeof(flow.solver.kinematic_viscosity) === Float32
    @test all(
        ==(0.5f0), flow.solver.pressure_solver.operator.face_coefficient,
    )
    @test eltype(flow.workspace.old_velocity) === Float32
    @test flow.workspace.elliptic.operator.diffusion === 5.0f-5
end

@testset "time-step safety" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    physics = SinglePhase(density = 1.0, dynamic_viscosity = 0.01)
    moving = FlowSimulation(mesh, physics; velocity = (1.0, 0.0))
    bound = stable_timestep(moving; cfl = 0.5)
    @test_throws ArgumentError step!(
        moving.velocity, moving.pressure, moving.solver, 1.01bound,
        moving.workspace,
    )
    @test_throws ArgumentError simulate!(moving, 1.01bound; dt = 1.01bound)
    @test step!(
        moving.velocity, moving.pressure, moving.solver, 0.99bound,
        moving.workspace,
    ).converged

    rest = FlowSimulation(
        BlockTreeMesh(block_size = (4, 4), periodic = (false, false)),
        physics; boundary = NoSlip((x, y, t, axis) -> axis == 1 && y > 0.5 ? 1.0 : 0.0),
    )
    @test simulate!(rest, 0.01).steps > 100
end

function _resume_pressure_force(state)
    geometry = state.geometry
    mesh = geometry.mesh
    drag = lift = 0.0
    nx, ny = mesh.block_size
    for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        cell = CellRef(block, i, j)
        segment = TwoPhaseAMR._local_plic_segment(geometry.solid, cell)
        isnothing(segment) && continue
        first, last = segment
        spacing = mesh.blocks[block].spacing
        length = hypot(
            (last[1] - first[1]) * spacing[1],
            (last[2] - first[2]) * spacing[2],
        )
        normal_x = TwoPhaseAMR._get(geometry.solid.normal_x, cell) / spacing[1]
        normal_y = TwoPhaseAMR._get(geometry.solid.normal_y, cell) / spacing[2]
        norm = hypot(normal_x, normal_y)
        iszero(norm) && continue
        factor = -2 * TwoPhaseAMR._get(state.pressure, cell) * length / norm
        drag += factor * normal_x
        lift += factor * normal_y
    end
    return (drag, lift)
end

function _resume_cylinder_flow()
    mesh = BlockTreeMesh(
        block_size = (14, 8), origin = (-4.0, -4.0),
        extent = (14.0, 8.0), periodic = (false, false), max_level = 2,
        refine = (key, origin, extent) -> key.level < 1,
        adaptation = MeshAdaptationOptions(
            update_interval = 2, coarsen_delay = 2, minimum_level = 1,
        ),
    )
    geometry = GeometryField(
        mesh, (x, y) -> hypot(x, y) - 0.5;
        lipschitz = 1, atol = 1.0e-7, maxdepth = 24,
    )
    inflow = (x, y, time, axis) -> axis == 1 ? 1.0 : 0.0
    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? 1.0 :
            0.005exp(-((x - 0.75)^2 + y^2)),
    )
    boundary = VelocityBoundaryConditions(
        left = Inflow(inflow), right = Outflow(),
        bottom = ZeroGradientVelocity(), top = ZeroGradientVelocity(),
    )
    criteria = (
        GeometryRefinementIndicator(geometry; level = 2),
        VorticityRefinementIndicator(
            velocity; refine_tolerance = 0.2, coarsen_tolerance = 0.2 / 3,
            maximum_level = 2,
        ),
    )
    return FlowSimulation(
        mesh, SinglePhase(density = 1.0, dynamic_viscosity = 1 / 200);
        velocity, geometry, boundary, adaptivity = Adaptivity(; criteria),
        diagnostics = (Diagnostic(:pressure_force, _resume_pressure_force),),
    )
end

# Permit accumulated projection error and force integration roundoff: the
# physical solves below use a dimensionless tolerance of 1e-9.
function _check_resume_forces(reference, resumed; tolerance = 2.0e-8)
    expected = diagnostic_history(reference, :pressure_force)
    actual = diagnostic_history(resumed, :pressure_force)
    @test actual.steps == expected.steps
    @test actual.times ≈ expected.times rtol = 0 atol = 1.0e-14
    @test all(
        isapprox(a[1], b[1]; rtol = tolerance, atol = tolerance)
            for (a, b) in zip(actual.values, expected.values)
    )
    @test all(
        isapprox(a[2], b[2]; rtol = tolerance, atol = tolerance)
            for (a, b) in zip(actual.values, expected.values)
    )
    return nothing
end

@testset "automatic timestep resumes on adaptive cylinder" begin
    options = (; cfl = 0.25, tolerance = 1.0e-9, max_cycles = 1000)
    probe = _resume_cylinder_flow()
    simulate!(probe, 0.04; options...)
    times = diagnostic_history(probe, :pressure_force).times
    split_time = times[length(times) ÷ 2]
    target = 2split_time

    reference = _resume_cylinder_flow()
    simulate!(reference, target; options...)
    resumed = _resume_cylinder_flow()
    first_run = simulate!(resumed, split_time; options...)
    pressure = [copy(interior(resumed.pressure, b)) for b in eachindex(resumed.pressure.blocks)]
    pressure_solver = resumed.solver.pressure_solver
    diffusion_solver = resumed.workspace.elliptic
    history = diagnostic_history(resumed, :pressure_force)
    steps = resumed.solver.mesh.adaptation.completed_steps
    simulate!(resumed, split_time; options...)
    @test resumed.solver.pressure_solver === pressure_solver
    @test resumed.workspace.elliptic === diffusion_solver
    @test resumed.solver.mesh.adaptation.completed_steps == steps
    @test diagnostic_history(resumed, :pressure_force).values == history.values
    @test all(
        pressure[b] == interior(resumed.pressure, b)
            for b in eachindex(pressure)
    )
    second_run = simulate!(resumed, target; options...)

    @test first_run.mesh_updates > 0
    @test first_run.steps + second_run.steps == reference.steps == resumed.steps
    @test resumed.mesh_updates == reference.mesh_updates
    @test [b.key for b in resumed.solver.mesh.blocks] ==
        [b.key for b in reference.solver.mesh.blocks]
    @test resumed.velocity.interior ≈ reference.velocity.interior rtol = 2.0e-8 atol = 2.0e-8
    @test all(
        isapprox(
            interior(resumed.pressure, b), interior(reference.pressure, b);
            rtol = 2.0e-8, atol = 2.0e-8,
        ) for b in eachindex(reference.pressure.blocks)
    )
    _check_resume_forces(reference, resumed)

    # A cut immediately after regridding creates a large genuine short-step
    # projection pressure. Resuming with the retained proposal must match a
    # control which explicitly uses the pre-cut proposal, not that tiny step.
    proposal = times[3] - times[2]
    off_grid = times[2] + 0.01proposal
    stop_time = off_grid + 2.5proposal
    automatic = _resume_cylinder_flow()
    fixed = _resume_cylinder_flow()
    simulate!(automatic, off_grid; options...)
    simulate!(fixed, off_grid; options...)
    simulate!(automatic, stop_time; options...)
    simulate!(fixed, stop_time; dt = proposal, options...)
    @test automatic.steps == fixed.steps
    @test automatic.velocity.interior ≈ fixed.velocity.interior rtol = 2.0e-8 atol = 2.0e-8
    _check_resume_forces(fixed, automatic)
end

@testset "infinite initial bound retains the chosen proposal" begin
    flow = FlowSimulation(
        BlockTreeMesh(block_size = (4, 4)),
        SinglePhase(density = 1.0, dynamic_viscosity = 0.01),
    )
    first_run = simulate!(flow, 0.008)
    second_run = simulate!(flow, 0.009)
    @test first_run.steps == 128
    @test second_run.steps == 16

    # An explicit physical step ends the automatic selection history.
    simulate!(flow, 0.01; dt = 0.001)
    @test simulate!(flow, 0.011).steps == 128

    # Reusing the low-level workspace for different fields is a new run,
    # even when the caller supplies the previous run's end time.
    other_velocity = FaceVelocity(flow.solver.mesh)
    other_pressure = CellField(flow.solver.mesh)
    other_run = simulate!(
        other_velocity, other_pressure, flow.solver, 0.012, flow.workspace;
        time = 0.011,
    )
    @test other_run.steps == 128
end

@testset "Float32 flow retains the precision of a Float64 CFL proposal" begin
    make_flow() = FlowSimulation(
        BlockTreeMesh(block_size = (4, 4)),
        SinglePhase(density = 1.0f0, dynamic_viscosity = 0.01f0);
        T = Float32, velocity = (1.0f0, 0.0f0),
        diagnostics = (KineticEnergy(),),
    )
    force = (x, y, time, axis) -> axis == 1 ? -0.001f0 : 0.0f0
    options = (; cfl = 0.4, force, tolerance = 1.0f-5, max_cycles = 40)
    reference = make_flow()
    simulate!(reference, 8.0f0; options...)
    resumed = make_flow()
    simulate!(resumed, 4.0f0; options...)
    simulate!(resumed, 8.0f0; options...)
    expected = diagnostic_history(reference, :kinetic_energy)
    actual = diagnostic_history(resumed, :kinetic_energy)
    @test resumed.steps == reference.steps
    @test actual.times == expected.times
    @test actual.values == expected.values
    @test resumed.velocity.interior == reference.velocity.interior
end
