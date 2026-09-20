using LinearAlgebra: norm, rank
using SparseArrays: nonzeros, spdiagm

const _LOCAL_RADIUS = 0.3
const _LOCAL_CENTRE = (0.5, 0.5)
const _LOCAL_PRESSURE_SCALE = 40.0

function _local_test_velocity(mesh)
    return (
        FaceVelocity(
            mesh,
            (x, y, time, axis) -> axis == 1 ?
                sinpi(2y) + 0.1cospi(2x) : -0.2sinpi(2x),
        ),
        FaceVelocity(
            mesh,
            (x, y, time, axis) -> axis == 1 ?
                0.3cospi(2y) : sinpi(2x),
        ),
    )
end

function _local_manufactured_final_velocity(x, y, phase)
    phase == 2 && return (0.0, 0.0)
    dx = x - _LOCAL_CENTRE[1]
    dy = y - _LOCAL_CENTRE[2]
    factor = cospi(2(dx^2 + dy^2))
    return (2dy * factor, -2dx * factor)
end

function _local_manufactured_gradient(x, y, phase, density)
    gx = 2pi * cospi(2x)
    gy = -pi * sinpi(2y)
    if phase == 1
        dx = x - _LOCAL_CENTRE[1]
        dy = y - _LOCAL_CENTRE[2]
        level = dx^2 + dy^2 - _LOCAL_RADIUS^2
        factor = x - y
        gx += _LOCAL_PRESSURE_SCALE * (2dx * factor + level)
        gy += _LOCAL_PRESSURE_SCALE * (2dy * factor - level)
    end
    return (gx / density[phase], gy / density[phase])
end

function _local_manufactured_error(
        resolution; density = (10.0, 1.0), constraint_location = :cell,
        interpolation_weights = :c1mc,
    )
    mesh = BlockTreeMesh(
        block_size = (resolution, resolution), periodic = (true, true),
    )
    interface = InterfaceField(
        mesh,
        (x, y) -> hypot(
            x - _LOCAL_CENTRE[1], y - _LOCAL_CENTRE[2],
        ) - _LOCAL_RADIUS;
        lipschitz = 1,
    )
    velocity = ntuple(2) do phase
        FaceVelocity(
            mesh,
            (x, y, time, axis) ->
                _local_manufactured_final_velocity(x, y, phase)[axis] +
                _local_manufactured_gradient(
                    x, y, phase, density,
                )[axis],
        )
    end
    solver = LocalMimeticProjectionSolver(
        interface; density, constraint_location, interpolation_weights,
    )
    workspace = LocalMimeticProjectionWorkspace(solver)
    stats = project!(
        velocity, CellField(mesh), solver, 1.0, workspace;
        tolerance = 1.0e-9, max_iterations = 5000,
    )
    data = TwoPhaseAMR._mimetic_face_data(solver, workspace)
    dual_volumes = TwoPhaseAMR._uncut_face_dual_volumes(mesh, Float64)
    error = weight = 0.0
    for (face_index, face) in pairs(mesh.faces)
        fractions = (
            data.liquid_fraction[face_index],
            1 - data.liquid_fraction[face_index],
        )
        for phase in 1:2
            local_weight = fractions[phase] * dual_volumes[face_index]
            exact = _local_manufactured_final_velocity(
                face.center..., phase,
            )[face.axis]
            error += local_weight * abs(
                velocity[phase].interior[face_index] - exact,
            )
            weight += local_weight
        end
    end
    return error / weight, stats, workspace
end

@testset "Local energy-compatible projection" begin
    coarse_error, coarse_stats, coarse_workspace =
        _local_manufactured_error(8)
    fine_error, fine_stats, fine_workspace =
        _local_manufactured_error(16)

    for (stats, workspace) in (
            (coarse_stats, coarse_workspace),
            (fine_stats, fine_workspace),
        )
        @test stats.converged
        @test stats.matrix_symmetry_error == 0
        @test stats.mixture_volume_defect <= stats.target_defect
        @test stats.normal_velocity_defect <= stats.target_defect
        @test stats.tangential_defect <= stats.target_defect
        @test stats.energy_identity_error <= 64stats.target_defect
        @test stats.orthogonality_error <= 64stats.target_defect
        @test rank(Matrix(workspace.constraint_matrix)) ==
            size(workspace.constraint_matrix, 1)
        @test stats.independent_constraint_rows ==
            size(workspace.constraint_matrix, 1)
        @test stats.candidate_constraint_rows ==
            stats.independent_constraint_rows +
            stats.dependent_constraint_rows
        # H stores the interface-length-scaled rows; it is not row-normalised.
        @test maximum(abs, nonzeros(workspace.constraint_matrix)) < 0.5
    end
    @test fine_error < 0.6coarse_error
    @test coarse_stats.jump_unknowns >
        2coarse_stats.independent_constraint_rows
    @test coarse_stats.constraint_location == :cell
    @test coarse_stats.interpolation_weights == :c1mc

    geometric_coarse_error, geometric_coarse_stats, _ =
        _local_manufactured_error(8; interpolation_weights = :geometric)
    geometric_fine_error, geometric_fine_stats, _ =
        _local_manufactured_error(16; interpolation_weights = :geometric)
    @test geometric_coarse_stats.converged
    @test geometric_fine_stats.converged
    @test geometric_fine_error < 0.7geometric_coarse_error
    @test geometric_coarse_stats.interpolation_weights == :geometric

    face_coarse_error, face_coarse_stats, _ = _local_manufactured_error(
        8; constraint_location = :face,
    )
    face_fine_error, face_fine_stats, _ = _local_manufactured_error(
        16; constraint_location = :face,
    )
    @test face_coarse_stats.converged
    @test face_fine_stats.converged
    @test face_fine_error < 0.4face_coarse_error
    @test face_coarse_stats.dependent_constraint_rows > 0
    @test face_coarse_stats.constraint_location == :face

    # The third preconditioner block is precisely H K_xi^{-1} H'.
    mesh = coarse_workspace.liquid_gradient.mesh
    interface = InterfaceField(
        mesh,
        (x, y) -> hypot(x - 0.5, y - 0.5) - _LOCAL_RADIUS;
        lipschitz = 1,
    )
    solver = LocalMimeticProjectionSolver(interface; density = (10.0, 1.0))
    workspace = LocalMimeticProjectionWorkspace(solver)
    assembled = TwoPhaseAMR._assemble_local_mimetic_system!(
        solver, workspace, 1.0, _local_test_velocity(mesh),
    )
    np = length(assembled.pressure_indices)
    nx = length(assembled.active_faces)
    nl = size(assembled.constraint, 1)
    multiplier_block = assembled.block_preconditioner[
        (np + nx + 1):(np + nx + nl),
        (np + nx + 1):(np + nx + nl),
    ]
    expected = assembled.constraint *
        spdiagm(0 => inv.(assembled.relative_diagonal)) *
        transpose(assembled.constraint)
    @test multiplier_block ≈ expected
end

@testset "Local projection active set, AMR, and embedded geometry" begin
    active_mesh = BlockTreeMesh(
        block_size = (8, 8), periodic = (true, true),
    )
    active_interface = InterfaceField(
        active_mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.27;
        lipschitz = 1,
    )
    active_solver = LocalMimeticProjectionSolver(
        active_interface; density = (1000.0, 1.0),
        active_mass_fraction = 0.01,
    )
    active_velocity = _local_test_velocity(active_mesh)
    active_workspace = LocalMimeticProjectionWorkspace(active_solver)
    active_stats = project!(
        active_velocity, CellField(active_mesh), active_solver, 0.02,
        active_workspace; tolerance = 1.0e-8, max_iterations = 4000,
    )
    @test active_stats.converged
    @test active_stats.mass_deactivated_faces > 0
    @test active_stats.energy_identity_error <= 1.0e-8
    @test active_stats.normal_velocity_defect <= 1.0e-8

    amr_mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 2,
        periodic = (true, true),
        refine = (key, origin, extent) -> key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 0),
    )
    amr_interface = InterfaceField(
        amr_mesh, (x, y) -> hypot(x - 0.4, y - 0.4) - 0.23;
        lipschitz = 1,
    )
    amr_solver = LocalMimeticProjectionSolver(
        amr_interface; density = (10.0, 1.0),
    )
    amr_stats = project!(
        _local_test_velocity(amr_mesh), CellField(amr_mesh), amr_solver,
        0.02, LocalMimeticProjectionWorkspace(amr_solver);
        tolerance = 1.0e-8, max_iterations = 4000,
    )
    @test amr_stats.converged
    @test amr_stats.matrix_symmetry_error == 0
    @test amr_stats.energy_identity_error <= 1.0e-8

    embedded_mesh = BlockTreeMesh(block_size = (8, 8))
    geometry = GeometryField(
        embedded_mesh, (x, y) -> y - 0.16; lipschitz = 1,
    )
    embedded_interface = InterfaceField(
        embedded_mesh, (x, y) -> x - 0.52;
        geometry, lipschitz = 1,
    )
    embedded_velocity = _local_test_velocity(embedded_mesh)
    enforce_no_permeation!(embedded_velocity)
    embedded_solver = LocalMimeticProjectionSolver(
        embedded_interface; density = (10.0, 1.0),
    )
    embedded_stats = project!(
        embedded_velocity, CellField(embedded_mesh), embedded_solver,
        0.02, LocalMimeticProjectionWorkspace(embedded_solver);
        tolerance = 1.0e-8, max_iterations = 4000,
    )
    @test embedded_stats.converged
    @test embedded_stats.embedded_deactivated_faces > 0
    @test embedded_stats.energy_identity_error <= 1.0e-8
    @test all(isfinite, embedded_velocity[1].interior)
    @test all(isfinite, embedded_velocity[2].interior)
end

@testset "Local projection time-dependent integration" begin
    mesh = BlockTreeMesh(
        block_size = (6, 6), periodic = (true, true),
    )
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.22;
        lipschitz = 1,
    )
    liquid = (x, y, time, axis) -> axis == 1 ?
        -0.05(y - 0.5) : 0.05(x - 0.5)
    gas = (x, y, time, axis) -> zero(x)
    flow = FlowSimulation(
        mesh; density = (10.0, 1.0), dynamic_viscosity = (0.0, 0.0),
        interface, velocity = (liquid, gas), dt = 5.0e-4,
        interface_transport = FrozenInterfaceFluxes(),
        two_velocity_projection = LocalMimeticProjection(),
    )
    result = simulate!(
        flow, 1.0e-3; tolerance = 1.0e-7, max_cycles = 2000,
    )
    @test result.steps == 2
    @test result.pressure isa LocalMimeticProjectionStats
    @test result.pressure.converged
    @test result.pressure.energy_identity_error <= 1.0e-7
    @test all(isfinite, flow.velocity[1].interior)
    @test all(isfinite, flow.velocity[2].interior)
end

function _local_rotation_time_error(dt; final_time = 0.015625)
    mesh = BlockTreeMesh(block_size = (8, 8))
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.2;
        lipschitz = 1,
    )
    initial_fraction = copy(interface.fraction)
    velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
    transport = WYAdvectionWorkspace(
        interface, velocity; interface_transport = FrozenInterfaceFluxes(),
    )
    solver = LocalMimeticProjectionSolver(
        interface; density = (1.0, 1.0),
    )
    workspace = LocalMimeticProjectionWorkspace(solver)
    pressure = CellField(mesh)
    time = 0.0
    step = 0
    maximum_energy_defect = 0.0
    while time < final_time - 16eps()
        step_size = min(dt, final_time - time)
        set_velocities!(
            velocity,
            (
                (x, y, time, axis) -> axis == 1 ?
                    -2pi * (y - 0.5) : 2pi * (x - 0.5),
                (x, y, time, axis) -> zero(x),
            ),
        )
        enforce_no_permeation!(velocity)
        stats = project!(
            velocity, pressure, solver, step_size, workspace;
            tolerance = 1.0e-8, max_iterations = 2000,
        )
        stats.converged || error("local rotation projection did not converge")
        maximum_energy_defect = max(
            maximum_energy_defect, stats.energy_identity_error,
        )
        advect!(
            interface, velocity, step_size, transport;
            sweep_order = iseven(step) ? (1, 2) : (2, 1),
        )
        time += step_size
        step += 1
    end
    error = sum(eachindex(mesh.blocks)) do block
        mesh.blocks[block].volume * sum(
            abs,
            interior(interface.fraction, block) .-
                interior(initial_fraction, block),
        )
    end
    return (; error, maximum_energy_defect, steps = step)
end

@testset "Local projection time-step refinement" begin
    results = map(
        dt -> _local_rotation_time_error(dt),
        (0.0078125, 0.00390625, 0.001953125),
    )
    errors = collect(getproperty.(results, :error))
    @test getproperty.(results, :steps) == (2, 4, 8)
    @test all(diff(errors) .< 0)
    @test errors[end] < 0.7errors[1]
    @test maximum(getproperty.(results, :maximum_energy_defect)) <= 1.0e-8
end

@testset "Local projection configuration validation" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    interface = InterfaceField(mesh, (x, y) -> x - 0.5; lipschitz = 1)
    @test LocalMimeticProjection() isa AbstractTwoVelocityProjection
    @test LocalMimeticProjection().interpolation_weights == :c1mc
    @test CellLocalMimeticProjection().constraint_location == :cell
    @test FaceLocalMimeticProjection().constraint_location == :face
    @test_throws ArgumentError LocalMimeticProjectionSolver(
        interface; active_mass_fraction = 0.5,
    )
    @test_throws ArgumentError LocalMimeticProjectionSolver(
        interface; rank_tolerance = 0.0,
    )
    @test_throws ArgumentError LocalMimeticProjectionSolver(
        interface; constraint_location = :vertex,
    )
    @test_throws ArgumentError LocalMimeticProjectionSolver(
        interface; interpolation_weights = :uniform,
    )
end
