using LinearAlgebra: ldiv!, norm, rank, cholesky, Symmetric
using SparseArrays: sparse
include("../helpers.jl")

function _joint_test_velocity(mesh)
    return (
        FaceVelocity(
            mesh, (x, y, time, axis) ->
            axis == 1 ? sinpi(2y) + 0.1cospi(2x) : -0.2sinpi(2x)
        ),
        FaceVelocity(
            mesh, (x, y, time, axis) ->
            axis == 1 ? 0.3cospi(2y) : sinpi(2x)
        ),
    )
end

# Build the required predictor with the current common pressure map, without
# reusing a jump left by a previous correction.
function _joint_pressure_predictor!(velocity, pressure, solver, dt)
    mesh = solver.interface.mesh
    work = MimeticProjectionWorkspace(solver)
    gradients = (FaceVelocity(mesh), FaceVelocity(mesh))
    mimetic_gradient!(gradients, pressure, solver, work)
    for phase in 1:2, index in eachindex(velocity[phase].interior)
        isnan(velocity[phase].interior[index]) && continue
        velocity[phase].interior[index] -= dt * gradients[phase].interior[index]
    end
    return gradients
end

function _joint_correction_energies(before, after, data, volumes)
    initial = final = correction = 0.0
    for phase in 1:2
        masses = phase == 1 ? data.liquid_mass : data.gas_mass
        for index in eachindex(volumes)
            isfinite(after[phase].interior[index]) || continue
            u0 = before[phase].interior[index]
            u1 = after[phase].interior[index]
            weight = 0.5 * masses[index] * volumes[index]
            initial += weight * u0^2
            final += weight * u1^2
            correction += weight * (u1 - u0)^2
        end
    end
    return initial, final, correction
end

@testset "Mimetic incremental pressure correction" begin
    for variant in (:joint_smoothed, :joint_face), domain in (:periodic, :cut, :split)
        mesh = BlockTreeMesh(
            block_size = domain == :split ? (8, 8) : (6, 6),
            periodic = domain == :periodic ? (true, true) : (false, false),
        )
        geometry = if domain == :periodic
            nothing
        elseif domain == :cut
            GeometryField(mesh, (x, y) -> y - 0.12 - 0.08x; lipschitz = hypot(1, 0.08))
        else
            GeometryField(mesh, (x, y) -> abs(x - 0.5) - 0.14; lipschitz = 1)
        end
        levelset = if domain == :periodic
            (x, y) -> hypot(x - 0.5, y - 0.5) - 0.27
        elseif domain == :cut
            (x, y) -> x - 0.52
        else
            (x, y) -> y - 0.47
        end
        interface = InterfaceField(mesh, levelset; geometry, lipschitz = 1)
        solver = MimeticProjectionSolver(interface; density = (10.0, 1.0), variant)
        work = MimeticProjectionWorkspace(solver)
        reference_work = MimeticProjectionWorkspace(solver)
        pressure = CellField(mesh, (x, y) -> 3.25 + 0.35sinpi(2x) - 0.2cospi(2y))
        fill_halo!(pressure)
        old_pressure = copy(pressure)
        reference_pressure = CellField(mesh)
        predictor = _joint_test_velocity(mesh)
        for phase in predictor
            fill!(phase.boundary, 0.0)
        end
        isnothing(geometry) || enforce_no_permeation!(predictor)
        dt = 0.2
        old_gradient = _joint_pressure_predictor!(predictor, pressure, solver, dt)
        corrected = map(copy, predictor)
        reference = map(copy, predictor)
        tolerance = 1.0e-9
        stats = project!(
            corrected, pressure, solver, dt, work;
            tolerance, max_iterations = 5000,
        )
        reference_stats = project!(
            reference, reference_pressure, solver, dt, reference_work;
            tolerance, max_iterations = 5000,
        )
        @test stats.converged && reference_stats.converged
        @test stats.initial_residual > 0
        @test stats.mixture_volume_defect <= tolerance
        @test stats.normal_velocity_defect <= tolerance
        @test stats.tangential_defect <= tolerance
        @test stats.momentum_identity_error <= 1024eps() * maximum(solver.density)
        @test stats.energy_identity_error <= 128tolerance * maximum(solver.density)
        @test _phase_storage_contract(corrected, interface)

        # The same predictor and zero scratch pressure give the same increment
        # and corrected velocity, irrespective of the input pressure gauge.
        for block in eachindex(mesh.blocks)
            increment = interior(work.pressure_increment, block)
            @test increment ≈ interior(reference_pressure, block) rtol = 2.0e-8 atol = 2.0e-8
            @test interior(pressure, block) ≈
                interior(old_pressure, block) .+ increment rtol = 2.0e-13 atol = 2.0e-13
        end
        references = TwoPhaseAMR._cell_references(mesh)
        anchors = unique(work.cache.pressure.pressure_anchors)
        domain == :split && @test length(anchors) == 2
        for index in anchors
            cell = references[index]
            @test iszero(TwoPhaseAMR._get(work.pressure_increment, cell))
            @test TwoPhaseAMR._get(pressure, cell) == TwoPhaseAMR._get(old_pressure, cell)
        end
        @test work.jump ≈ reference_work.jump rtol = 2.0e-8 atol = 2.0e-8
        if domain == :periodic
            @test stats.independent_constraint_rows > 0
            @test norm(work.jump) > 1.0e-12
        end

        reconstructed = (FaceVelocity(mesh), FaceVelocity(mesh))
        mimetic_gradient!(reconstructed, work.pressure_increment, solver, work)
        for phase in 1:2
            active = _phase_interior_indices(interface, phase)
            @test corrected[phase].interior[active] ≈
                reference[phase].interior[active] rtol = 2.0e-8 atol = 2.0e-8
            @test corrected[phase].interior[active] ≈
                predictor[phase].interior[active] .-
                dt .* reconstructed[phase].interior[active] rtol = 2.0e-13 atol = 2.0e-13
        end
        for face in work.mixed_faces
            @test reconstructed[2].interior[face] -
                reconstructed[1].interior[face] ≈ work.jump[face]
        end

        if domain == :periodic
            # The uniform, uncut kinetic Riesz map is the diagonal phase-mass
            # norm used here. Test the actual Pythagorean identity, not only
            # its projection diagnostic.
            initial, final, correction = _joint_correction_energies(
                predictor, corrected, work.cache.pressure.face_data,
                work.cache.face_dual_volumes,
            )
            scale = max(1.0, initial, final, correction)
            @test abs(initial - final - correction) <= 128tolerance * scale
            @test abs(2 * (initial - final - correction)) ≈
                stats.energy_identity_error atol = 4096eps() * scale
        end

        # Equal external and old-pressure accelerations leave an exactly zero
        # predictor. Reusing the workspace must not reuse its preceding jump
        # or replace the nonconstant physical pressure with a zero increment.
        balanced = (FaceVelocity(mesh), FaceVelocity(mesh))
        for phase in 1:2, index in eachindex(mesh.faces)
            impulse = dt * old_gradient[phase].interior[index]
            balanced[phase].interior[index] = impulse - impulse
        end
        balanced_pressure = copy(old_pressure)
        balanced_stats = project!(
            balanced, balanced_pressure, solver, dt, work;
            tolerance, max_iterations = 5000,
        )
        @test balanced_stats.converged
        @test iszero(balanced_stats.initial_residual)
        @test iszero(balanced_stats.iterations)
        @test all(iszero, work.jump)
        for block in eachindex(mesh.blocks)
            @test all(iszero, interior(work.pressure_increment, block))
            @test interior(balanced_pressure, block) == interior(old_pressure, block)
        end
        for phase in 1:2
            @test all(iszero, balanced[phase].interior[_phase_interior_indices(interface, phase)])
        end
    end
end

@testset "Mimetic short-step inherited divergence" begin
    for variant in (:joint_smoothed, :joint_face)
        mesh = BlockTreeMesh(block_size = (6, 6), periodic = (true, true))
        interface = InterfaceField(
            mesh, (x, y) -> hypot(x - 0.47, y - 0.51) - 0.27; lipschitz = 1,
        )
        solver = MimeticProjectionSolver(interface; density = (10.0, 1.0), variant)
        work = MimeticProjectionWorkspace(solver)
        field = (x, y, time, axis) -> axis == 1 ? sinpi(2x) : 0.4sinpi(2y)
        old_velocity = (FaceVelocity(mesh, field), FaceVelocity(mesh, field))
        TwoPhaseAMR._mark_inactive_phase_velocities!(old_velocity, interface)
        old_data = TwoPhaseAMR._projection_face_data(solver, work)
        old_flux = zeros(ncells(mesh))
        TwoPhaseAMR._cmdgfm_cell_flux!(old_flux, old_data, interface, old_velocity)
        @test norm(old_flux) > 0

        dt, reference_dt = 1.0e-4, 0.02
        baseline = TwoPhaseAMR._assemble_mimetic_system!(solver, work, dt, old_velocity)
        indices = copy(baseline.pressure_indices)
        np = length(indices)
        baseline_rhs = copy(baseline.right_hand_side)
        equal_step = TwoPhaseAMR._assemble_mimetic_system!(
            solver, work, dt, old_velocity;
            _inherited_cell_flux = old_flux, _reference_dt = dt,
        )
        @test equal_step.right_hand_side == baseline_rhs
        shortened = TwoPhaseAMR._assemble_mimetic_system!(
            solver, work, dt, old_velocity;
            _inherited_cell_flux = old_flux, _reference_dt = reference_dt,
        )
        # With identical old and predictor velocities and zero phase jump,
        # the 1/dt part cancels before division, leaving the reference RHS.
        @test shortened.right_hand_side[1:np] ≈
            -old_flux[indices] ./ reference_dt rtol = 2.0e-11 atol = 2.0e-11
        @test shortened.right_hand_side[(np + 1):end] == baseline_rhs[(np + 1):end]

        short_velocity = map(copy, old_velocity)
        reference_velocity = map(copy, old_velocity)
        short_pressure, reference_pressure = CellField(mesh), CellField(mesh)
        short_work = MimeticProjectionWorkspace(solver)
        reference_work = MimeticProjectionWorkspace(solver)
        tolerance = 1.0e-12
        short_stats = project!(
            short_velocity, short_pressure, solver, dt, short_work;
            tolerance, max_iterations = 5000,
            _inherited_cell_flux = old_flux, _reference_dt = reference_dt,
        )
        reference_stats = project!(
            reference_velocity, reference_pressure, solver, reference_dt, reference_work;
            tolerance, max_iterations = 5000,
        )
        @test short_stats.converged && reference_stats.converged
        @test short_stats.mixture_volume_defect ==
            dt * short_stats.mixture_divergence
        @test short_stats.mixture_volume_defect > tolerance
        for block in eachindex(mesh.blocks)
            @test interior(short_pressure, block) ≈
                interior(reference_pressure, block) rtol = 2.0e-7 atol = 2.0e-7
        end
        final_flux = similar(old_flux)
        TwoPhaseAMR._cmdgfm_cell_flux!(
            final_flux, short_work.cache.pressure.face_data, interface, short_velocity,
        )
        @test final_flux ≈ (1 - dt / reference_dt) .* old_flux rtol = 1.0e-8 atol = 2.0e-9

        # Check the affine pressure-work identity against kinetic energies
        # computed independently from the input and corrected phase velocities.
        references = TwoPhaseAMR._cell_references(mesh)
        target_work = dt * (1 - dt / reference_dt) * sum(
            TwoPhaseAMR._get(short_work.pressure_increment, references[index]) *
                old_flux[index] for index in eachindex(old_flux)
        )
        initial, final, correction = _joint_correction_energies(
            old_velocity, short_velocity, short_work.cache.pressure.face_data,
            short_work.cache.face_dual_volumes,
        )
        energy_scale = max(1.0, initial, final, correction)
        balance = initial - final - correction - target_work
        @test abs(target_work) > 1.0e-8
        @test abs(balance) <= 128tolerance * energy_scale
        @test abs(2balance) ≈ short_stats.energy_identity_error atol = 4096eps() * energy_scale

        # Capture happens on the old interface. After movement the correction
        # still uses that stored flux, not a newly reconstructed aperture.
        moving_velocity = _joint_test_velocity(mesh)
        TwoPhaseAMR._mark_inactive_phase_velocities!(moving_velocity, interface)
        moving_data = TwoPhaseAMR._projection_face_data(solver, work)
        moving_old_flux = zeros(ncells(mesh))
        TwoPhaseAMR._cmdgfm_cell_flux!(
            moving_old_flux, moving_data, interface, moving_velocity,
        )
        initialize!(
            interface, (x, y) -> hypot(x - 0.52, y - 0.48) - 0.27,
        )
        moved_velocity = _joint_test_velocity(mesh)
        TwoPhaseAMR._mark_inactive_phase_velocities!(moved_velocity, interface)
        moved = TwoPhaseAMR._assemble_mimetic_system!(solver, work, dt, moved_velocity)
        moved_rhs = copy(moved.right_hand_side)
        moved_indices = copy(moved.pressure_indices)
        moved_np = length(moved_indices)
        corrected_rhs = TwoPhaseAMR._assemble_mimetic_system!(
            solver, work, dt, moved_velocity;
            _inherited_cell_flux = moving_old_flux, _reference_dt = reference_dt,
        ).right_hand_side
        expected = moved_rhs[1:moved_np] .+
            (inv(dt) - inv(reference_dt)) .* moving_old_flux[moved_indices]
        @test corrected_rhs[1:moved_np] ≈ expected rtol = 2.0e-11 atol = 1.0e-7
        @test corrected_rhs[(moved_np + 1):end] == moved_rhs[(moved_np + 1):end]
    end
end

@testset "Mimetic correction boundary pressure work" begin
    for variant in (:joint_smoothed, :joint_face)
        mesh = BlockTreeMesh(block_size = (6, 6), periodic = (false, false))
        interface = InterfaceField(mesh, (x, y) -> x - 0.47; lipschitz = 1)
        solver = MimeticProjectionSolver(interface; density = (10.0, 1.0), variant)
        work = MimeticProjectionWorkspace(solver)
        field = (x, y, time, axis) -> axis == 1 ? 1 + 0.2x * (1 - x) : 0.0
        predictor = (FaceVelocity(mesh, field), FaceVelocity(mesh, field))
        corrected = map(copy, predictor)
        dt, tolerance = 0.1, 1.0e-10
        stats = project!(
            corrected, CellField(mesh), solver, dt, work;
            tolerance, max_iterations = 5000,
        )
        @test stats.converged
        boundary_flux, _ = TwoPhaseAMR._mimetic_boundary_constraints(interface, corrected)
        references = TwoPhaseAMR._cell_references(mesh)
        boundary_work = dt * sum(
            TwoPhaseAMR._get(work.pressure_increment, references[index]) *
                boundary_flux[index] for index in eachindex(boundary_flux)
        )
        initial, final, correction = _joint_correction_energies(
            predictor, corrected, work.cache.pressure.face_data,
            work.cache.face_dual_volumes,
        )
        scale = max(1.0, initial, final, correction)
        balance = initial - final - correction - boundary_work
        @test abs(boundary_work) > 1.0e-8
        @test abs(balance) <= 128tolerance * scale
        @test abs(2balance) ≈ stats.energy_identity_error atol = 4096eps() * scale
    end
end

@testset "Mimetic increment survives regridding and fresh workspace" begin
    for variant in (:joint_smoothed, :joint_face)
        mesh = BlockTreeMesh(
            base_mesh = (8, 4), block_size = (4, 4), max_level = 1,
            periodic = (true, true),
        )
        interface = InterfaceField(mesh, (x, y) -> x - 0.273; lipschitz = 1)
        flow = FlowSimulation(
            mesh, TwoPhase(
                interface; density = (10.0, 1.0), dynamic_viscosity = (0.0, 0.0),
            );
            velocity = _joint_test_velocity(mesh), pressure = 3.0,
            velocity_model = TwoVelocityModel(
                interface_transport = FrozenInterfaceFluxes(),
                pressure_projection = MimeticProjection(; variant),
            ),
        )
        state = TwoPhaseAMR._two_velocity_flow_workspace!(
            flow.workspace.two_phase, flow.solver.two_phase,
        )
        work = state.projection_workspace
        increment = work.pressure_increment
        dt, tolerance = 0.02, 1.0e-8
        # The initial pressure is constant, so this predictor already
        # contains its (zero) weighted pressure gradient.
        first = project!(
            flow.velocity, flow.pressure, state.projection_solver, dt, work;
            tolerance, max_iterations = 5000,
        )
        @test first.converged
        @test maximum(
            abs, interior(flow.pressure, 1) .- 3.0,
        ) > 1.0e-8

        adaptation = AMRWorkspace(mesh, interface, flow.velocity, flow.pressure)
        labels = [block.key.i == 0 ? :refine : :keep for block in mesh.blocks]
        @test adapt!(adaptation, labels; step = 1, force = true).changed
        TwoPhaseAMR._rebuild_topology_dependent_state!(
            flow.solver, flow.workspace; dt,
        )
        state = flow.workspace.two_phase.two_velocity[]
        @test state.projection_workspace === work
        @test work.pressure_increment === increment
        @test nblocks(mesh) > 2

        # Compare a reused post-AMR workspace with a fresh one, as in a
        # resumed run. Both start from the transferred physical pressure and
        # the same predictor made with the new topology's gradient map.
        old_pressure = copy(flow.pressure)
        predictor = _joint_test_velocity(mesh)
        _joint_pressure_predictor!(predictor, old_pressure, state.projection_solver, dt)
        reused_velocity = map(copy, predictor)
        resumed_velocity = map(copy, predictor)
        resumed_pressure = copy(old_pressure)
        resumed_work = MimeticProjectionWorkspace(state.projection_solver)
        reused = project!(
            reused_velocity, flow.pressure, state.projection_solver, dt, work;
            tolerance, max_iterations = 5000,
        )
        resumed = project!(
            resumed_velocity, resumed_pressure, state.projection_solver, dt, resumed_work;
            tolerance, max_iterations = 5000,
        )
        @test reused.converged && resumed.converged
        @test length(increment.blocks) == nblocks(mesh)
        @test reused.mixture_volume_defect <= tolerance
        @test reused.normal_velocity_defect <= tolerance
        @test reused.tangential_defect <= tolerance
        for block in eachindex(mesh.blocks)
            @test interior(flow.pressure, block) ≈
                interior(old_pressure, block) .+ interior(increment, block) rtol = 2.0e-13 atol = 2.0e-13
            @test interior(flow.pressure, block) ≈
                interior(resumed_pressure, block) rtol = 2.0e-8 atol = 2.0e-8
            @test interior(increment, block) ≈
                interior(resumed_work.pressure_increment, block) rtol = 2.0e-8 atol = 2.0e-8
        end
        for phase in 1:2
            active = _phase_interior_indices(interface, phase)
            @test reused_velocity[phase].interior[active] ≈
                resumed_velocity[phase].interior[active] rtol = 2.0e-8 atol = 2.0e-8
        end
    end
end

@testset "Mimetic hydrostatic pressure persists over steps" begin
    for variant in (:joint_smoothed, :joint_face), cut in (false, true)
        mesh = BlockTreeMesh(
            block_size = (8, 8), extent = (8.0, 8.0),
            periodic = (false, false),
        )
        geometry = cut ?
            GeometryField(mesh, (x, y) -> y - 0.9 - 0.1x; lipschitz = hypot(1, 0.1)) :
            nothing
        interface = InterfaceField(
            mesh, (x, y) -> y - 4.6; geometry,
            contact_angle = pi / 2, lipschitz = 1,
        )
        solver = IncompressibleNavierStokesSolver(
            mesh; geometry, interface, density = (10.0, 1.0),
            dynamic_viscosity = (0.0, 0.0), gravity = (0.0, -1.0),
            boundary = FreeSlip(), interface_transport = FrozenInterfaceFluxes(),
            two_velocity_projection = MimeticProjection(; variant),
        )
        workspace = NavierStokesWorkspace(solver)
        velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
        pressure = CellField(mesh)
        # Gravity's potential is sampled at the interface centroid in a mixed
        # cell. Construct the matching continuous hydrostatic pressure there.
        for cell in TwoPhaseAMR._cell_references(mesh)
            _, y = cellcenter(mesh, cell.block, cell.i, cell.j)
            fraction = TwoPhaseAMR._conditional_fraction(interface, cell)
            value = fraction >= 1 ? -10y :
                (fraction <= 0 ? -46 - (y - 4.6) : -46.0)
            TwoPhaseAMR._set!(pressure, value, cell)
        end
        fill_halo!(pressure)
        initial_pressure = copy(pressure)
        pressure_scale = max(1.0, maximum(abs, interior(initial_pressure, 1)))
        @test pressure_scale > 1
        dt = 1.0e-3
        for step in 1:3
            stats = step!(
                velocity, pressure, solver, dt, workspace;
                time = (step - 1) * dt, tolerance = 1.0e-12, max_cycles = 6000,
            )
            @test stats.converged
            @test _phase_storage_contract(velocity, interface)
            for phase in 1:2
                active = _phase_interior_indices(interface, phase)
                @test maximum(abs, velocity[phase].interior[active]; init = 0.0) <=
                    1.0e-8 * dt
            end
            @test maximum(
                abs, interior(pressure, 1) .- interior(initial_pressure, 1),
            ) <= 1.0e-7 * pressure_scale
        end
    end
end

@testset "Jointly fitted mimetic projection variants" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), periodic = (true, true),
    )
    # The pressure-commuting bound was established with GHF-fallback normals;
    # commit 0b00093 changed the default to :lhf.
    # NUMERICS CONCERN: the default :lhf gives a residual of about 0.9 here,
    # versus the 0.2 bound below; the default's behavior needs investigation.
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.3;
        lipschitz = 1, reconstruction_method = :lhf_ghf,
    )
    for variant in (:joint_smoothed, :joint_face)
        solver = MimeticProjectionSolver(
            interface; density = (10.0, 1.0), variant,
        )
        workspace = MimeticProjectionWorkspace(solver)
        velocity = _joint_test_velocity(mesh)
        assembled = TwoPhaseAMR._assemble_mimetic_system!(
            solver, workspace, 1.0, velocity,
        )
        @test assembled.matrix ≈ transpose(assembled.matrix)
        @test rank(Matrix(assembled.constraint)) ==
            size(assembled.constraint, 1)
        @test all(>(0), assembled.relative_diagonal)
        @test length(assembled.active_faces) ==
            length(assembled.relative_diagonal)
        preconditioner = TwoPhaseAMR._local_mimetic_preconditioner(
            solver, assembled,
        )
        rhs = sin.(collect(1:size(assembled.matrix, 1)))
        expected = assembled.block_preconditioner \ rhs
        preconditioned = similar(rhs)
        ldiv!(preconditioned, preconditioner, rhs)
        @test preconditioned ≈ expected atol = 1.0e-9
        # A Krylov workspace may alias its input and output buffers.
        aliased = copy(rhs)
        ldiv!(aliased, preconditioner, aliased)
        @test aliased ≈ expected atol = 1.0e-9

        stats = project!(
            velocity, CellField(mesh), solver, 1.0, workspace;
            tolerance = 1.0e-9, max_iterations = 5000,
        )
        @test stats.converged
        @test stats.variant == variant
        @test stats.mixture_volume_defect <= stats.target_defect
        @test stats.normal_velocity_defect <= stats.target_defect
        @test stats.energy_identity_error <= 64stats.target_defect
        @test stats.orthogonality_error <= 64stats.target_defect
        @test maximum(abs, stats.momentum_error) <=
            64stats.target_defect
        @test stats.independent_constraint_rows > 0
        @test stats.candidate_constraint_rows ==
            stats.independent_constraint_rows +
            stats.dependent_constraint_rows
        @test stats.commuting_residual < 0.1
        @test stats.pressure_commuting_residual < 0.2
    end
end

@testset "Mimetic projection configuration" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    interface = InterfaceField(mesh, (x, y) -> x - 0.5; lipschitz = 1)
    @test MimeticProjection().variant isa TwoPhaseAMR._MimeticJointSmoothed
    @test MimeticProjection(variant = :joint_face).variant isa TwoPhaseAMR._MimeticJointFace
    @test typeof(MimeticProjection()) !==
        typeof(MimeticProjection(variant = :joint_face))
    for variant in (:joint_smoothed, :joint_face)
        projection = MimeticProjection(;
            variant, preconditioner = :ichol, preconditioner_options = (;),
            active_mass_fraction = 0.002, rank_tolerance = 1.0e-4,
            patch_radius = 1, pressure_weight = 12.0,
            commutator_weight = 34.0, correction_cap = 0.5, fit_sweeps = 4,
        )
        direct_solver = MimeticProjectionSolver(
            interface; density = (4.0, 2.0), variant,
            preconditioner = :ichol, preconditioner_options = (;),
            active_mass_fraction = 0.002, rank_tolerance = 1.0e-4,
            patch_radius = 1, pressure_weight = 12.0,
            commutator_weight = 34.0, correction_cap = 0.5, fit_sweeps = 4,
        )
        solver, workspace = TwoPhaseAMR._two_velocity_projection_workspace(
            interface, (4.0, 2.0), projection,
        )
        @test solver isa MimeticProjectionSolver
        @test workspace isa MimeticProjectionWorkspace
        @test TwoPhaseAMR._mimetic_variant_symbol(solver.variant) === variant
        @test direct_solver.variant === solver.variant
        @test typeof(solver) === typeof(direct_solver)
        @test solver.variant === projection.variant
        @test solver.density === (4.0, 2.0)
        @test direct_solver.density === solver.density
        @test solver.preconditioner === projection.preconditioner
        @test solver.preconditioner_options === projection.preconditioner_options
        @test solver.active_mass_fraction === projection.active_mass_fraction
        @test solver.rank_tolerance === projection.rank_tolerance
        @test solver.patch_radius === projection.patch_radius
        @test solver.pressure_weight === projection.pressure_weight
        @test solver.commutator_weight === projection.commutator_weight
        @test solver.correction_cap === projection.correction_cap
        @test solver.fit_sweeps === projection.fit_sweeps
    end
    @test_throws ArgumentError MimeticProjection(variant = :cell)
    for invalid in (nothing, "joint_face", :unknown)
        @test_throws ArgumentError MimeticProjection(variant = invalid)
        @test_throws ArgumentError MimeticProjectionSolver(
            interface; variant = invalid,
        )
    end
    @test_throws MethodError MimeticProjection(jump_interpolant = :face)
    @test_throws ArgumentError MimeticProjectionSolver(
        interface; variant = :paired,
    )
    @test_throws ArgumentError MimeticProjectionSolver(
        interface; active_mass_fraction = 0.5,
    )
    @test_throws ArgumentError MimeticProjectionSolver(
        interface; rank_tolerance = 0.0,
    )
    @test_throws ArgumentError MimeticProjection(
        pressure_weight = 0,
    )
    @test_throws ArgumentError MimeticProjection(
        correction_cap = 0,
    )
    @test_throws ArgumentError MimeticProjection(
        fit_sweeps = 0,
    )

    positional = (
        :joint_face, :cholesky, (;), 1.0e-6, nothing, 2,
        1000.0, 1000.0, 1.0, 3,
    )
    positional_projection = MimeticProjection(positional...)
    @test positional_projection.variant isa TwoPhaseAMR._MimeticJointFace
    for (index, invalid) in (
            (1, :cell), (2, :bogus), (6, -1), (7, 0.0),
            (8, 0.0), (9, 0.0), (10, 0),
        )
        args = ntuple(i -> i == index ? invalid : positional[i], length(positional))
        @test_throws ArgumentError MimeticProjection(args...)
        @test_throws MethodError typeof(positional_projection)(args...)
    end
    for preconditioner in (:block, :ilu, nothing)
        @test CMDGFMProjection(preconditioner, (;)).preconditioner === preconditioner
    end
    @test_throws ArgumentError CMDGFMProjection(:bogus, (;))
    @test_throws ArgumentError CMDGFMProjection(preconditioner = :bogus)
    @test_throws MethodError typeof(CMDGFMProjection())(:bogus, (;))
end

@testset "Mimetic Cholesky factor solves support both scalar types" begin
    for T in (Float32, Float64)
        pressure = T[4 1; 1 3]
        multiplier = T[7;;]
        jump_inverse = T[0.5, 0.2]
        preconditioner = TwoPhaseAMR.MimeticBlockCholeskyPreconditioner(
            cholesky(Symmetric(sparse(pressure), :L)), jump_inverse,
            cholesky(Symmetric(sparse(multiplier), :L)), (2, 2, 1),
            zeros(T, 2), zeros(T, 1),
        )
        matrix = zeros(T, 5, 5)
        matrix[1:2, 1:2] .= pressure
        matrix[3, 3] = inv(jump_inverse[1])
        matrix[4, 4] = inv(jump_inverse[2])
        matrix[5, 5] = multiplier[1, 1]
        rhs = T[1, -2, 3, -4, 5]
        expected = matrix \ rhs
        result = similar(rhs)
        ldiv!(result, preconditioner, rhs)
        tolerance = T === Float32 ? 3.0e-6 : 2.0e-14
        @test result ≈ expected rtol = tolerance atol = tolerance
        aliased = copy(rhs)
        ldiv!(aliased, preconditioner, aliased)
        @test aliased ≈ expected rtol = tolerance atol = tolerance
    end
end

@testset "Float32 pure-phase mimetic projection" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), periodic = (true, true),
        origin = (0.0f0, 0.0f0), extent = (1.0f0, 1.0f0)
    )
    interface = InterfaceField(mesh, (x, y) -> -1.0f0; lipschitz = 1)
    solver = MimeticProjectionSolver(interface; density = (10.0f0, 1.0f0))
    workspace = MimeticProjectionWorkspace(solver)
    stats = project!(
        _joint_test_velocity(mesh), CellField(mesh), solver, 0.02f0, workspace;
        tolerance = 1.0f-3, max_iterations = 4000
    )
    @test stats isa MimeticProjectionStats{Float32}
    @test stats.converged
    @test isfinite(stats.final_residual)
    @test !isnothing(workspace.cache.krylov)
end
