using LinearAlgebra: ldiv!, norm, rank
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
        resolution; density = (10.0, 1.0), jump_interpolant = :cell,
        interpolation_weights = nothing, infer_face_selection = :paired,
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
    solver = MimeticProjectionSolver(
        interface; density, jump_interpolant, interpolation_weights,
        infer_face_selection,
    )
    workspace = MimeticProjectionWorkspace(solver)
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
    @test coarse_stats.jump_interpolant == :cell
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
        8; jump_interpolant = :face,
    )
    face_fine_error, face_fine_stats, _ = _local_manufactured_error(
        16; jump_interpolant = :face,
    )
    @test face_coarse_stats.converged
    @test face_fine_stats.converged
    @test face_fine_error < 0.4face_coarse_error
    @test face_coarse_stats.dependent_constraint_rows > 0
    @test face_coarse_stats.jump_interpolant == :face

    # The commuting model changes only the adjoint placement of the same
    # aperture-normal cell equation. Its hard moment identity must therefore
    # reproduce the baseline on constant Cartesian jump fields, irrespective
    # of how many tangential rows pass certification.
    commuting_mesh = BlockTreeMesh(
        block_size = (8, 8), periodic = (true, true),
    )
    commuting_interface = InterfaceField(
        commuting_mesh,
        (x, y) -> hypot(x - 0.5, y - 0.5) - _LOCAL_RADIUS;
        lipschitz = 1,
    )
    commuting_solver = MimeticProjectionSolver(
        commuting_interface; density = (10.0, 1.0),
        jump_interpolant = :commuting, commuting_sliver_epsilon = 0,
    )
    commuting_workspace = MimeticProjectionWorkspace(commuting_solver)
    commuting_assembled = TwoPhaseAMR._assemble_mimetic_system!(
        commuting_solver, commuting_workspace, 1.0,
        _local_test_velocity(commuting_mesh),
    )
    reconstruction =
        TwoPhaseAMR._local_aperture_normal_tangent_reconstruction(
            commuting_solver, commuting_assembled.data,
            commuting_assembled.active_faces,
        )
    constants = zeros(
        Float64, length(commuting_assembled.active_faces), 2,
    )
    for (local_face, global_face) in
            pairs(commuting_assembled.active_faces)
        constants[local_face, commuting_mesh.faces[global_face].axis] = 1
    end
    @test commuting_assembled.candidate_constraint * constants ≈
        reconstruction.normal * constants atol = 1.0e-12
    @test commuting_assembled.commuting_count > 0
    @test commuting_assembled.exact_commuting_count +
        commuting_assembled.approximate_commuting_count ==
        commuting_assembled.commuting_count
    @test commuting_assembled.fallback_count == 0
    @test commuting_assembled.commuting_residual <= 1.0e-10
    @test commuting_assembled.pressure_commuting_count ==
        commuting_assembled.candidate_count
    @test commuting_assembled.pressure_fallback_count == 0
    @test commuting_assembled.pressure_commuting_residual <=
        maximum(commuting_mesh.blocks[1].spacing)
    @test commuting_assembled.sliver_extrapolated == 0
    @test commuting_assembled.sliver_zeroed == 0
    @test commuting_assembled.full_active_faces ==
        commuting_assembled.active_faces
    @test iszero(norm(
        commuting_assembled.fixed_active_jump_gradient, Inf,
    ))

    # The production default removes sliver unknowns through xi=E*xi_hat.
    # Every donor is an immediate same-axis face sharing a tangential vertex,
    # and the closest staggered fraction to 1/2 wins deterministically.
    sliver_solver = MimeticProjectionSolver(
        commuting_interface; density = (10.0, 1.0),
        jump_interpolant = :commuting,
    )
    sliver_workspace = MimeticProjectionWorkspace(sliver_solver)
    sliver_data = TwoPhaseAMR._mimetic_face_data(
        sliver_solver, sliver_workspace,
    )
    sliver_active, _, _ = TwoPhaseAMR._local_active_faces(
        sliver_solver, sliver_data,
    )
    sliver_dual = TwoPhaseAMR._uncut_face_dual_volumes(
        commuting_mesh, Float64,
    )[sliver_active]
    sliver_relative = sliver_data.relative_mass[sliver_active] .* sliver_dual
    sliver_space = TwoPhaseAMR._local_commuting_jump_space(
        sliver_solver, sliver_data, sliver_active, sliver_relative,
        sliver_dual,
    )
    @test sliver_space.extrapolated_count > 0
    @test sliver_space.extrapolated_count + sliver_space.zeroed_count ==
        length(sliver_space.sliver_faces)
    @test length(sliver_space.active_faces) < length(sliver_active)
    @test transpose(sliver_space.extension) *
        spdiagm(0 => sliver_relative) * sliver_space.extension ≈
        spdiagm(0 => sliver_space.relative_diagonal)
    active_set = Set(sliver_active)
    patches = TwoPhaseAMR._vertex_patches(commuting_mesh, Float64)
    epsilon = sliver_solver.commuting_sliver_epsilon
    for (sliver, donor) in zip(
            sliver_space.sliver_faces, sliver_space.sliver_donor_faces,
        )
        face = commuting_mesh.faces[sliver]
        candidates = TwoPhaseAMR._cmdgfm_vertex_candidates(
            patches, commuting_mesh, face, sliver, face.axis,
        )
        filter!(candidates) do candidate
            fraction = sliver_data.liquid_fraction[candidate]
            candidate in active_set &&
                epsilon <= fraction <= 1 - epsilon
        end
        sort!(candidates; by = candidate -> (
            abs(sliver_data.liquid_fraction[candidate] - 0.5), candidate,
        ))
        @test donor == (isempty(candidates) ? 0 : first(candidates))
    end

    # With no valid donor the extension has no columns, so every active jump
    # is prescribed to vanish rather than being extrapolated recursively.
    zero_donor_solver = MimeticProjectionSolver(
        commuting_interface; density = (10.0, 1.0),
        jump_interpolant = :commuting, commuting_sliver_epsilon = 0.49,
    )
    zero_donor_space = TwoPhaseAMR._local_commuting_jump_space(
        zero_donor_solver, sliver_data, sliver_active, sliver_relative,
        sliver_dual,
    )
    @test isempty(zero_donor_space.active_faces)
    @test zero_donor_space.extrapolated_count == 0
    @test zero_donor_space.zeroed_count == length(sliver_active)

    commuting_error, commuting_stats, commuting_solution_workspace =
        _local_manufactured_error(8; jump_interpolant = :commuting)
    @test isfinite(commuting_error)
    @test commuting_stats.converged
    @test commuting_stats.matrix_symmetry_error == 0
    @test commuting_stats.energy_identity_error <= 1.0e-8
    @test commuting_stats.momentum_identity_error <= 1.0e-8
    @test commuting_stats.sliver_extrapolated_faces > 0
    @test commuting_stats.sliver_extrapolated_faces +
        commuting_stats.sliver_zeroed_faces > 0
    for (sliver, donor) in zip(
            sliver_space.sliver_faces, sliver_space.sliver_donor_faces,
        )
        expected = iszero(donor) ? 0.0 :
            commuting_solution_workspace.jump[donor]
        @test commuting_solution_workspace.jump[sliver] ≈ expected
    end

    # A deliberately zero-radius patch is unable to certify the curved
    # interface. The method must fall back to N0 without losing its moments.
    fallback_solver = MimeticProjectionSolver(
        commuting_interface; density = (10.0, 1.0),
        jump_interpolant = :commuting, commuting_patch_radius = 0,
        commuting_max_patch_radius = 0,
        commuting_accuracy_factor = 1.0e-12,
        commuting_sliver_epsilon = 0,
    )
    fallback_workspace = MimeticProjectionWorkspace(fallback_solver)
    fallback_assembled = TwoPhaseAMR._assemble_mimetic_system!(
        fallback_solver, fallback_workspace, 1.0,
        _local_test_velocity(commuting_mesh),
    )
    @test fallback_assembled.fallback_count > 0
    @test fallback_assembled.commuting_count +
        fallback_assembled.fallback_count ==
        fallback_assembled.candidate_count
    @test fallback_assembled.pressure_fallback_count > 0
    @test fallback_assembled.pressure_commuting_count +
        fallback_assembled.pressure_fallback_count ==
        fallback_assembled.candidate_count
    @test fallback_assembled.candidate_constraint * constants ≈
        reconstruction.normal * constants atol = 1.0e-12

    # The third preconditioner block is precisely H K_xi^{-1} H'.
    mesh = coarse_workspace.liquid_gradient.mesh
    interface = InterfaceField(
        mesh,
        (x, y) -> hypot(x - 0.5, y - 0.5) - _LOCAL_RADIUS;
        lipschitz = 1,
    )
    solver = MimeticProjectionSolver(
        interface; density = (10.0, 1.0), jump_interpolant = :cell,
    )
    workspace = MimeticProjectionWorkspace(solver)
    assembled = TwoPhaseAMR._assemble_mimetic_system!(
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
    active_solver = MimeticProjectionSolver(
        active_interface; density = (1000.0, 1.0),
        active_mass_fraction = 0.01, jump_interpolant = :cell,
    )
    active_velocity = _local_test_velocity(active_mesh)
    active_workspace = MimeticProjectionWorkspace(active_solver)
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
    amr_solver = MimeticProjectionSolver(
        amr_interface; density = (10.0, 1.0), jump_interpolant = :cell,
    )
    amr_stats = project!(
        _local_test_velocity(amr_mesh), CellField(amr_mesh), amr_solver,
        0.02, MimeticProjectionWorkspace(amr_solver);
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
    embedded_solver = MimeticProjectionSolver(
        embedded_interface; density = (10.0, 1.0),
        jump_interpolant = :cell,
    )
    embedded_stats = project!(
        embedded_velocity, CellField(embedded_mesh), embedded_solver,
        0.02, MimeticProjectionWorkspace(embedded_solver);
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
        two_velocity_projection = MimeticProjection(jump_interpolant = :cell),
    )
    result = simulate!(
        flow, 1.0e-3; tolerance = 1.0e-7, max_cycles = 2000,
    )
    @test result.steps == 2
    @test result.pressure isa MimeticProjectionStats
    @test result.pressure.converged
    @test result.pressure.energy_identity_error <= 1.0e-7
    @test all(isfinite, flow.velocity[1].interior)
    @test all(isfinite, flow.velocity[2].interior)
end

function _local_rotation_time_error(
        dt; final_time = 0.015625, jump_interpolant = :face,
    )
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
    solver = MimeticProjectionSolver(
        interface; density = (1.0, 1.0), jump_interpolant,
    )
    workspace = MimeticProjectionWorkspace(solver)
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

    # Keep the cell-centred alternative covered without claiming temporal
    # convergence on this deliberately short, coarse rotation problem.  Its
    # KH behaviour is assessed separately by the vortex-sheet diagnostics.
    cell_results = map(
        dt -> _local_rotation_time_error(dt; jump_interpolant = :cell),
        (0.0078125, 0.00390625, 0.001953125),
    )
    @test all(isfinite, getproperty.(cell_results, :error))
    @test maximum(
        getproperty.(cell_results, :maximum_energy_defect),
    ) <= 1.0e-8
end

@testset "Inferred jump projection and face selection" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), periodic = (true, true),
    )
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.3;
        lipschitz = 1,
    )
    for selection in (
            :paired, :smoothed, :plic, :stencil, :smoothed_stencil,
        )
        solver = MimeticProjectionSolver(
            interface; density = (10.0, 1.0),
            jump_interpolant = :infer, infer_face_selection = selection,
        )
        workspace = MimeticProjectionWorkspace(solver)
        velocity = _local_test_velocity(mesh)
        assembled = TwoPhaseAMR._assemble_mimetic_system!(
            solver, workspace, 1.0, velocity,
        )
        @test assembled.matrix ≈ transpose(assembled.matrix)
        @test rank(Matrix(assembled.constraint)) ==
            size(assembled.constraint, 1)
        @test all(>(0), assembled.relative_diagonal)
        preconditioner = TwoPhaseAMR._local_mimetic_preconditioner(
            solver, assembled,
        )
        rhs = sin.(collect(1:size(assembled.matrix, 1)))
        expected = assembled.block_preconditioner \ rhs
        preconditioned = similar(rhs)
        ldiv!(preconditioned, preconditioner, rhs)
        @test preconditioned ≈ expected atol = 1.0e-9
        # Krylov workspaces may reuse their input as the output buffer.
        aliased = copy(rhs)
        ldiv!(aliased, preconditioner, aliased)
        @test aliased ≈ expected atol = 1.0e-9
        if !(selection in (:stencil, :smoothed_stencil))
            metric = transpose(assembled.jump_extension) *
                spdiagm(0 => assembled.full_relative_diagonal) *
                assembled.jump_extension
            @test Matrix(metric) ≈ Matrix(
                spdiagm(0 => assembled.relative_diagonal),
            ) atol = 1.0e-12
        end
        stats = project!(
            velocity, CellField(mesh), solver, 1.0, workspace;
            tolerance = 1.0e-9, max_iterations = 5000,
        )
        @test stats.converged
        @test stats.mixture_volume_defect <= stats.target_defect
        @test stats.normal_velocity_defect <= stats.target_defect
        @test stats.energy_identity_error <= 64stats.target_defect
        @test maximum(abs, stats.momentum_error) <=
            64stats.target_defect
        @test stats.jump_interpolant == :infer
    end
end

@testset "Local projection configuration validation" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    interface = InterfaceField(mesh, (x, y) -> x - 0.5; lipschitz = 1)
    @test MimeticProjection() isa AbstractTwoVelocityProjection
    @test MimeticProjection().jump_interpolant == :divergence
    @test MimeticProjection(
        jump_interpolant = :cell,
    ).interpolation_weights == :c1mc
    @test MimeticProjection(jump_interpolant = :cell).jump_interpolant == :cell
    @test MimeticProjection(jump_interpolant = :face).jump_interpolant == :face
    @test MimeticProjection(
        jump_interpolant = :commuting,
    ).commuting_patch_radius == 2
    @test MimeticProjection(
        jump_interpolant = :commuting,
    ).interpolation_weights == :geometric
    @test MimeticProjection(
        jump_interpolant = :commuting,
    ).commuting_sliver_epsilon == 0.05
    @test MimeticProjection(
        jump_interpolant = :infer,
    ).infer_face_selection == :paired
    @test MimeticProjection(
        jump_interpolant = :infer, infer_face_selection = :stencil,
    ).interpolation_weights == :geometric
    @test MimeticProjection(
        jump_interpolant = :infer,
        infer_face_selection = :smoothed_stencil,
    ).interpolation_weights == :geometric
    @test_throws ArgumentError MimeticProjection(
        jump_interpolant = :infer, infer_face_selection = :invalid,
    )
    @test_throws ArgumentError MimeticProjectionSolver(
        interface; jump_interpolant = :infer,
        interpolation_weights = :c1mc,
    )
    @test_throws ArgumentError MimeticProjection(
        jump_interpolant = :commuting, interpolation_weights = :c1mc,
    )
    @test_throws ArgumentError MimeticProjectionSolver(
        interface; active_mass_fraction = 0.5,
    )
    @test_throws ArgumentError MimeticProjectionSolver(
        interface; rank_tolerance = 0.0,
    )
    @test_throws ArgumentError MimeticProjectionSolver(
        interface; jump_interpolant = :vertex,
    )
    @test_throws ArgumentError MimeticProjectionSolver(
        interface; interpolation_weights = :uniform,
    )
    @test_throws ArgumentError MimeticProjectionSolver(
        interface; jump_interpolant = :commuting,
        commuting_patch_radius = 3, commuting_max_patch_radius = 2,
    )
    @test_throws ArgumentError MimeticProjectionSolver(
        interface; jump_interpolant = :commuting,
        commuting_amplification_limit = 0,
    )
    @test_throws ArgumentError MimeticProjectionSolver(
        interface; jump_interpolant = :commuting,
        commuting_accuracy_factor = 0,
    )
    @test_throws ArgumentError MimeticProjectionSolver(
        interface; jump_interpolant = :commuting,
        commuting_sliver_epsilon = -eps(),
    )
    @test_throws ArgumentError MimeticProjectionSolver(
        interface; jump_interpolant = :commuting,
        commuting_sliver_epsilon = 0.5,
    )
end
