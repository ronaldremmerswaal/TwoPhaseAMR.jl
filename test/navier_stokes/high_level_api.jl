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
    @test_throws ArgumentError step!(
        moving.velocity, moving.pressure, moving.solver, 0.01,
        moving.workspace; _reference_dt = 0.005,
    )
    @test_throws ArgumentError step!(
        moving.velocity, moving.pressure, moving.solver, 0.01,
        moving.workspace; _reference_dt = 0.0,
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

@testset "adaptive cylinder short final step (issue #90)" begin
    options = (; cfl = 0.25, tolerance = 1.0e-9, max_cycles = 1000)
    reference = _resume_cylinder_flow()
    simulate!(reference, 0.08; options...)
    history = diagnostic_history(reference, :pressure_force)
    k = length(history.times) - 10
    time = history.times[k]
    proposal = history.times[k + 1] - time
    before, after = history.values[k], history.values[k + 1]
    # Falling CFL proposals use the preceding actual step as their defect
    # reference, so compare the normal-step form with a separate full-pressure
    # trajectory rather than hard-coding the pre-policy adaptive forces.
    # Resolve both linear systems more tightly than the 1e-10 force check.
    incremental = deepcopy(reference)
    absolute = deepcopy(reference)
    for iteration in 1:3
        for block in absolute.pressure.blocks
            fill!(block, 0.0)
        end
        step_time = reference.time + (iteration - 1) * proposal
        incremental_stats = step!(
            incremental.velocity, incremental.pressure, incremental.solver,
            proposal, incremental.workspace; time = step_time,
            tolerance = 1.0e-15, max_cycles = 5000,
        )
        absolute_stats = step!(
            absolute.velocity, absolute.pressure, absolute.solver,
            proposal, absolute.workspace; time = step_time,
            tolerance = 1.0e-15, max_cycles = 5000,
        )
        @test incremental_stats.converged && absolute_stats.converged
        incremental_force = _resume_pressure_force(
            (
                geometry = incremental.solver.geometry, pressure = incremental.pressure,
            )
        )
        absolute_force = _resume_pressure_force(
            (
                geometry = absolute.solver.geometry, pressure = absolute.pressure,
            )
        )
        @test incremental_force[1] ≈ absolute_force[1] atol = 1.0e-10 rtol = 0
        @test incremental_force[2] ≈ absolute_force[2] atol = 1.0e-10 rtol = 0
    end
    for fraction in (0.1, 0.5)
        shortened = _resume_cylinder_flow()
        target = time + fraction * proposal
        result = simulate!(shortened, target; options...)
        short_history = diagnostic_history(shortened, :pressure_force)
        short_force = last(short_history.values)
        @test result.pressure.converged
        @test short_history.times[end - 1] ≈ time atol = 1.0e-14 rtol = 0
        @test short_history.times[end] ≈ target atol = 1.0e-14 rtol = 0
        @test short_history.times[end] - short_history.times[end - 1] ≈
            fraction * proposal atol = 1.0e-14 rtol = 0
        for axis in 1:2
            variation = abs(after[axis] - before[axis])
            excursion = abs(short_force[axis] - before[axis])
            @test variation > 0
            # Five nearby steps gave maxima 8.50 (10%) and 4.89 (50%)
            # over both force components; 12 allows trajectory variation.
            @test excursion <= 12variation
        end
    end
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

    # A cut immediately after regridding must retain the unshortened proposal
    # both for the reference-step correction and when the run resumes.
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

@testset "tiny final timestep is absorbed" begin
    make_flow(; velocity = (0.0, 0.0)) = FlowSimulation(
        BlockTreeMesh(block_size = (4, 4), periodic = (true, true)),
        SinglePhase(density = 1.0, dynamic_viscosity = 0.01);
        velocity, diagnostics = (KineticEnergy(),),
    )

    proposal = 0.01
    for (remainder, expected_steps) in ((0.5e-6, 2), (2.0e-6, 3))
        flow = make_flow()
        target = proposal * (2 + remainder)
        result = simulate!(flow, target; dt = proposal)
        times = diagnostic_history(flow, :kinetic_energy).times
        @test result.steps == expected_steps
        @test result.time == flow.time == target
        @test times[end] == target
        if expected_steps == 2
            @test times[end - 1] == proposal
            # The diffusion shift records the actual, stretched physical step.
            @test flow.workspace.elliptic.operator.diffusion ≈
                0.01 * (target - proposal) / 2
        end
    end

    # This absolute remainder exceeds the loop's 16eps time tolerance but
    # is far below the relative threshold.
    flow = make_flow()
    target = 2proposal + 1.0e-14
    @test target - 2proposal > 16eps()
    result = simulate!(flow, target; dt = proposal)
    times = diagnostic_history(flow, :kinetic_energy).times
    @test result.steps == 2
    @test result.time == flow.time == times[end] == target
    @test flow.workspace.elliptic.operator.diffusion ≈
        0.01 * (target - proposal) / 2

    # An automatically chosen proposal survives a stretched endpoint and is
    # reused on the next call rather than replaced by the stretched step.
    flow = make_flow()
    simulate!(flow, 1.28)
    retained = flow.workspace.simulation_step.step
    target = flow.time + 2retained + 0.5e-6retained
    result = simulate!(flow, target)
    @test result.steps == 2
    @test result.time == target
    @test flow.workspace.simulation_step.step == retained
    resumed = simulate!(flow, target + 2.5retained)
    @test resumed.steps == 3
    @test flow.workspace.simulation_step.step == retained

    # A proposal at the hard CFL bound can still absorb the tiny remainder.
    limited = make_flow(; velocity = (1.0, 0.0))
    bound = stable_timestep(limited; cfl = 0.5)
    target = bound * (1 + 0.5e-6)
    result = simulate!(limited, target; cfl = 0.5)
    times = diagnostic_history(limited, :kinetic_energy).times
    @test result.steps == 1
    @test result.time == target
    @test times[end] == target

    # Capillary restrictions do not scale with the requested convective CFL.
    for fixed in (false, true)
        mesh = BlockTreeMesh(block_size = (4, 4), periodic = (true, false))
        interface = InterfaceField(mesh, (x, y) -> y - 0.43)
        capillary = FlowSimulation(
            mesh,
            TwoPhase(
                interface; density = (1.0, 1.0),
                dynamic_viscosity = (0.0, 0.0), surface_tension = 1.0,
            );
            boundary = FreeSlip(),
        )
        bound = stable_timestep(capillary)
        target = bound * (1 + 0.5e-6)
        result = fixed ? simulate!(capillary, target; dt = bound) :
            simulate!(capillary, target)
        @test result.steps == 1
        @test result.time == target
    end
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

@testset "standalone flow projection retains physical pressure" begin
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, true))
    flow = FlowSimulation(
        mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.01);
        velocity = (x, y, time, axis) -> axis == 1 ? sinpi(2x) : 0.0,
        pressure = (x, y) -> 2 + cospi(2y),
    )
    pressure_before = copy(flow.pressure)
    stats = project!(flow; dt = 0.1, tolerance = 1.0e-12, max_cycles = 100)
    divergence = CellField(mesh)
    divergence!(divergence, flow.velocity)
    @test stats.converged
    @test maximum(abs, interior(divergence, 1)) < 1.0e-9
    @test all(
        interior(flow.pressure, block) == interior(pressure_before, block)
            for block in eachindex(flow.pressure.blocks)
    )
end

mutable struct ReferenceStepObserver <: AbstractVisualization
    procedure::Any
    times::Vector{Float64}
    references::Vector{Float64}
end

ReferenceStepObserver() = ReferenceStepObserver(nothing, Float64[], Float64[])

function TwoPhaseAMR.initialize_visualization!(
        observer::ReferenceStepObserver; step_procedure = nothing, kwargs...
    )
    observer.procedure = step_procedure
    return observer
end

function TwoPhaseAMR.update_visualization!(
        observer::ReferenceStepObserver; event, time, kwargs...
    )
    if event === :step
        push!(observer.times, time)
        push!(observer.references, observer.procedure.reference_dt)
    end
    return observer
end

function _reference_defect_flow(; max_level = 0)
    mesh = BlockTreeMesh(
        block_size = (8, 8), periodic = (true, true), max_level = max_level,
    )
    velocity = (x, y, time, axis) ->
    axis == 1 ? 1.0 + 1.0e-3sinpi(2x) : 0.0
    return FlowSimulation(
        mesh, SinglePhase(density = 1.0, dynamic_viscosity = 1.0e-3);
        velocity,
        numerics = NavierStokesOptions(multigrid = MultigridOptions(coarse_size = 8)),
    )
end

function _reference_step_copy(flow, dt, reference_dt)
    velocity = copy(flow.velocity)
    pressure = copy(flow.pressure)
    workspace = NavierStokesWorkspace(flow.solver)
    workspace.advection.sweep_forward[] = flow.workspace.advection.sweep_forward[]
    stats = step!(
        velocity, pressure, flow.solver, dt, workspace;
        time = flow.time, _reference_dt = reference_dt,
        tolerance = 1.0e-12, max_cycles = 100,
    )
    @test stats.converged
    return velocity, pressure
end

@testset "abrupt proposal drops retain the actual preceding step" begin
    for start in (:fixed, :adaptive), mode in (:fixed, :adaptive)
        flow = _reference_defect_flow()
        # Leave a resolved inherited defect using a clipped first step.
        first = if start === :fixed
            simulate!(
                flow, 0.01; dt = 0.1, tolerance = 1.0e-12, max_cycles = 100,
            )
        else
            simulate!(
                flow, 0.01; cfl = 0.4, tolerance = 1.0e-12, max_cycles = 100,
            )
        end
        @test first.steps == 1
        @test flow.workspace.simulation_step.last_dt ≈ 0.01
        small_dt = mode === :fixed ? 0.001 : stable_timestep(flow; cfl = 0.0008)
        @test small_dt < 0.2 * 0.01
        expected_velocity, expected_pressure = _reference_step_copy(flow, small_dt, 0.01)
        _, amplified_pressure = _reference_step_copy(flow, small_dt, small_dt)
        @test maximum(
            abs, interior(expected_pressure, 1) .- interior(amplified_pressure, 1),
        ) > 1.0e-4
        options = (; tolerance = 1.0e-12, max_cycles = 100)
        if mode === :fixed
            result = simulate!(flow, flow.time + small_dt; dt = small_dt, options...)
        else
            result = simulate!(flow, flow.time + small_dt; cfl = 0.0008, options...)
        end
        @test result.steps == 1
        @test flow.velocity.interior ≈ expected_velocity.interior atol = 1.0e-10
        @test interior(flow.pressure, 1) ≈ interior(expected_pressure, 1) atol = 1.0e-9
    end

    # A force-induced CFL collapse inside one call must also use the previous
    # actual step, rather than treating the new, smaller bound as the gauge.
    flow = FlowSimulation(
        BlockTreeMesh(block_size = (4, 4), periodic = (true, true)),
        SinglePhase(density = 1.0, dynamic_viscosity = 0.001);
        velocity = (1.0, 0.0),
    )
    initial_bound = stable_timestep(flow; cfl = 0.4)
    force = (x, y, time, axis) -> axis == 1 ? 400.0 : 0.0
    observer = ReferenceStepObserver()
    simulate!(
        flow, 1.04initial_bound; cfl = 0.4, force, visualization = observer,
        tolerance = 1.0e-10, max_cycles = 100,
    )
    @test length(observer.times) >= 2
    @test observer.references[1] ≈ initial_bound
    @test observer.times[2] - observer.times[1] < 0.1initial_bound
    @test observer.references[2] ≈ initial_bound
end

@testset "checkpoint uses the dropped-step defect reference" begin
    flow = _reference_defect_flow()
    simulate!(flow, 0.01; dt = 0.1, tolerance = 1.0e-12, max_cycles = 100)
    checkpoint_dt = 0.0005
    _, expected_pressure = _reference_step_copy(flow, checkpoint_dt, 0.01)
    _, wrong_pressure = _reference_step_copy(flow, checkpoint_dt, 0.001)
    @test maximum(
        abs, interior(expected_pressure, 1) .- interior(wrong_pressure, 1),
    ) > 1.0e-4
    registry = VisualizationFieldRegistry()
    register_field!(registry, :pressure, flow.pressure)
    session = VisualizationSession(
        registry; checkpoint_times = [flow.time + checkpoint_dt],
        exact_checkpoints = true,
    )
    result = simulate!(
        flow, flow.time + 0.001; dt = 0.001, visualization = session,
        tolerance = 1.0e-12, max_cycles = 100,
    )
    @test result.steps == 1
    checkpoint = only(visualization_checkpoints(session)).frame
    @test checkpoint[:pressure] ≈ visualization_values(expected_pressure) atol = 1.0e-9
    @test session.step_procedure.reference_dt ≈ 0.01
end

@testset "previous defect time is not reused for a different state" begin
    for change in (
            :time, :step, :projection, :workspace, :solver,
            :velocity_field, :pressure_field,
        )
        flow = _reference_defect_flow()
        simulate!(flow, 0.01; dt = 0.1, tolerance = 1.0e-12, max_cycles = 100)
        if change === :time
            flow.time = 0.5
        elseif change === :step
            @test step!(
                flow.velocity, flow.pressure, flow.solver, 0.001,
                flow.workspace; time = flow.time,
                tolerance = 1.0e-12, max_cycles = 100,
            ).converged
        elseif change === :workspace
            TwoPhaseAMR._replace_workspace!(
                flow.workspace, NavierStokesWorkspace(flow.solver),
            )
        elseif change === :projection
            old_velocity = copy(flow.velocity.interior)
            @test project!(
                flow; dt = 0.01, tolerance = 1.0e-12, max_cycles = 100,
            ).converged
            @test maximum(abs, flow.velocity.interior .- old_velocity) > 1.0e-8
            @test isnothing(flow.workspace.simulation_step.velocity)
            @test iszero(flow.workspace.simulation_step.last_dt)
            @test isinf(flow.workspace.simulation_step.step)
        end
        observer = ReferenceStepObserver()
        if change === :solver
            solver = IncompressibleNavierStokesSolver(
                flow.solver.mesh; density = 1.0, dynamic_viscosity = 1.0e-3,
                coarse_size = 8,
            )
            simulate!(
                flow.velocity, flow.pressure, solver, flow.time + 0.001,
                flow.workspace; time = flow.time, dt = 0.001,
                visualization = observer, tolerance = 1.0e-12, max_cycles = 100,
            )
        elseif change in (:velocity_field, :pressure_field)
            velocity = change === :velocity_field ? copy(flow.velocity) : flow.velocity
            pressure = change === :pressure_field ? copy(flow.pressure) : flow.pressure
            simulate!(
                velocity, pressure, flow.solver, flow.time + 0.001,
                flow.workspace; time = flow.time, dt = 0.001,
                visualization = observer, tolerance = 1.0e-12, max_cycles = 100,
            )
        else
            simulate!(
                flow, flow.time + 0.001; dt = 0.001,
                visualization = observer, tolerance = 1.0e-12, max_cycles = 100,
            )
        end
        @test only(observer.references) ≈ 0.001
    end
end

@testset "regrid defects keep the transfer reference" begin
    for two_velocity in (false, true)
        mesh = BlockTreeMesh(
            block_size = (4, 4), max_level = 1,
            adaptation = MeshAdaptationOptions(coarsen_delay = 0),
        )
        marker = CellField(mesh)
        initialize!(marker, (x, y) -> 1 + 0.2sinpi(4x) * sinpi(4y))
        criterion = InterpolationErrorIndicator(
            marker; E_refine = 1.0e-3, E_coarsen = 1.0e-4,
        )
        adaptivity = Adaptivity(criteria = criterion)
        numerics = NavierStokesOptions(multigrid = MultigridOptions(coarse_size = 8))
        flow = if two_velocity
            interface = InterfaceField(mesh, (x, y) -> x - 0.43; lipschitz = 1)
            FlowSimulation(
                mesh,
                TwoPhase(
                    interface; density = (2.0, 1.0),
                    dynamic_viscosity = (0.0, 0.0)
                );
                velocity = (0.0, 0.0), boundary = FreeSlip(), adaptivity, numerics,
                velocity_model = TwoVelocityModel(
                    pressure_projection = CMDGFMProjection(),
                ),
            )
        else
            FlowSimulation(
                mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.01);
                velocity = (x, y, time, axis) ->
                axis == 1 ? 0.05 + 0.001sinpi(2x) : 0.0,
                boundary = FreeSlip(), adaptivity, numerics,
            )
        end
        first = simulate!(
            flow, 0.01; dt = 0.1, tolerance = 1.0e-10, max_cycles = 100,
        )
        @test first.mesh_updates == 1
        @test flow.workspace.simulation_step.last_dt ≈ 0.01
        @test flow.workspace.simulation_step.defect_dt ≈ 0.1
        if !two_velocity
            expected_velocity, expected_pressure = _reference_step_copy(flow, 0.001, 0.1)
            _, amplified_pressure = _reference_step_copy(flow, 0.001, 0.001)
            @test maximum(
                maximum(
                    abs, interior(expected_pressure, block) .-
                        interior(amplified_pressure, block),
                ) for block in eachindex(flow.pressure.blocks)
            ) > 1.0e-4
        end
        observer = ReferenceStepObserver()
        result = simulate!(
            flow, 0.011; dt = 0.001, visualization = observer,
            tolerance = 1.0e-12, max_cycles = 100,
        )
        @test result.steps == 1
        @test only(observer.references) ≈ 0.1
        @test flow.workspace.simulation_step.last_dt ≈ 0.001
        @test flow.workspace.simulation_step.defect_dt ≈ 0.001
        if !two_velocity
            @test flow.velocity.interior ≈ expected_velocity.interior atol = 1.0e-9
            @test all(
                isapprox(
                    interior(flow.pressure, block), interior(expected_pressure, block);
                    atol = 1.0e-8,
                ) for block in eachindex(flow.pressure.blocks)
            )
        end
    end

    # A regrid following an abrupt drop creates a new defect: do not carry
    # the older step's larger reference past that topology change.
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 1,
        adaptation = MeshAdaptationOptions(coarsen_delay = 0),
    )
    marker = CellField(mesh)
    initialize!(marker, (x, y) -> 1 + 0.2sinpi(4x) * sinpi(4y))
    criterion = InterpolationErrorIndicator(
        marker; E_refine = 1.0e-3, E_coarsen = 1.0e-4, t_init = 0.0105,
    )
    flow = FlowSimulation(
        mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.01);
        adaptivity = Adaptivity(criteria = criterion),
        numerics = NavierStokesOptions(multigrid = MultigridOptions(coarse_size = 8)),
    )
    first = simulate!(flow, 0.01; dt = 0.01)
    @test first.mesh_updates == 0
    before_regrid = ReferenceStepObserver()
    second = simulate!(flow, 0.011; dt = 0.001, visualization = before_regrid)
    @test second.mesh_updates == 1
    @test only(before_regrid.references) ≈ 0.01
    @test flow.workspace.simulation_step.defect_dt ≈ 0.001
    after_regrid = ReferenceStepObserver()
    simulate!(flow, 0.0111; dt = 0.0001, visualization = after_regrid)
    @test only(after_regrid.references) ≈ 0.001

    # Regridding independently of simulate! invalidates the old state even
    # when field objects, solver and time are unchanged.
    flow = _reference_defect_flow(max_level = 1)
    simulate!(flow, 0.01; dt = 0.1, tolerance = 1.0e-12, max_cycles = 100)
    amr = AMRWorkspace(flow.solver.mesh, flow.velocity, flow.pressure)
    @test adapt!(amr, [:refine]; step = flow.steps, force = true).changed
    TwoPhaseAMR._rebuild_topology_dependent_state!(
        flow.solver, flow.workspace; dt = 0.001,
    )
    observer = ReferenceStepObserver()
    simulate!(
        flow, 0.011; dt = 0.001, visualization = observer,
        tolerance = 1.0e-12, max_cycles = 100,
    )
    @test only(observer.references) ≈ 0.001

    # The automatic proposal is also invalid after an external regrid, even
    # if a larger new CFL would otherwise expose the old retained proposal.
    flow = _reference_defect_flow(max_level = 1)
    old_bound = stable_timestep(flow; cfl = 0.01)
    first = simulate!(
        flow, old_bound; cfl = 0.01, tolerance = 1.0e-12, max_cycles = 100,
    )
    @test first.steps == 1
    old_proposal = flow.workspace.simulation_step.step
    amr = AMRWorkspace(flow.solver.mesh, flow.velocity, flow.pressure)
    @test adapt!(amr, [:refine]; step = flow.steps, force = true).changed
    TwoPhaseAMR._rebuild_topology_dependent_state!(
        flow.solver, flow.workspace; dt = old_proposal,
    )
    new_proposal = stable_timestep(flow; cfl = 0.4)
    @test new_proposal > 2old_proposal
    observer = ReferenceStepObserver()
    result = simulate!(
        flow, flow.time + old_proposal / 2; cfl = 0.4, visualization = observer,
        tolerance = 1.0e-12, max_cycles = 100,
    )
    @test result.steps == 1
    @test only(observer.references) ≈ new_proposal
end

@testset "standalone projection invalidates the resumed automatic selection" begin
    flow = _reference_defect_flow()
    simulate!(flow, 0.01; cfl = 0.4, tolerance = 1.0e-12, max_cycles = 100)
    @test isfinite(flow.workspace.simulation_step.step)
    @test flow.workspace.simulation_step.last_dt ≈ 0.01
    @test project!(
        flow; dt = 0.01, tolerance = 1.0e-12, max_cycles = 100,
    ).converged
    @test isnothing(flow.workspace.simulation_step.velocity)
    @test isinf(flow.workspace.simulation_step.step)
    @test iszero(flow.workspace.simulation_step.last_dt)
    cfl = 0.0008
    proposal = stable_timestep(flow; cfl)
    @test proposal < 0.2 * 0.01
    observer = ReferenceStepObserver()
    simulate!(
        flow, flow.time + proposal / 2; cfl, visualization = observer,
        tolerance = 1.0e-12, max_cycles = 100,
    )
    @test only(observer.references) ≈ proposal
end
