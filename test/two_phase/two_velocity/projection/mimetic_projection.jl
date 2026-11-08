using LinearAlgebra: norm, rank
include("../helpers.jl")

function _mimetic_test_velocity(mesh)
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

@testset "Three-block mimetic pressure projection" begin
    @testset "projection statistics and summaries" begin
        linear = LinearSolverStats(
            Float64; method = :cg, preconditioner = :linear_pc,
        )
        linear.converged = false
        linear.cycles = 3
        linear.initial_l2 = 111.0
        linear.initial_linf = 10.5
        linear.final_l2 = 222.0
        linear.final_linf = 2.75

        mesh = BlockTreeMesh(block_size = (4, 4))
        flow = FlowSimulation(
            mesh, SinglePhase(density = 2.0, dynamic_viscosity = 0.1),
        )
        result = (;
            time = 1.25, steps = 5, pressure = linear, mesh_updates = 2,
            total_steps = 8, total_mesh_updates = 3,
        )

        cmdgfm = CMDGFMProjectionStats(
            true, 7, 11.25, 0.75, 0.2, 1.5, 0.25, 0.125,
            0.001, (2.0, 3.0), 0.875, 0.9, 5, 6, 7, :cmd_pc, true, false,
        )
        mimetic_values = (;
            converged = false, iterations = 13,
            initial_residual = 17.25, final_residual = 3.125,
            matrix_symmetry_error = 0.125, mixture_divergence = 1.25,
            mixture_volume_defect = 0.25, normal_velocity_defect = 0.375,
            tangential_defect = 0.5, target_defect = 0.75,
            momentum_error = (4.0, 5.0), momentum_identity_error = 0.875,
            energy_identity_error = 1.125, orthogonality_error = 1.375,
            pressure_unknowns = 10, jump_unknowns = 11,
            candidate_constraint_rows = 12, independent_constraint_rows = 13,
            dependent_constraint_rows = 14, underresolved_constraint_rows = 15,
            commuting_constraint_rows = 16, fallback_constraint_rows = 17,
            commuting_residual = 2.125, pressure_commuting_constraint_rows = 18,
            pressure_fallback_constraint_rows = 19,
            pressure_commuting_residual = 2.375, mass_deactivated_faces = 20,
            embedded_deactivated_faces = 21, preconditioner = :mimetic_pc,
            variant = :joint_smoothed,
        )
        mimetic = MimeticProjectionStats(; mimetic_values...)
        @test fieldnames(typeof(mimetic)) == keys(mimetic_values)
        for name in keys(mimetic_values)
            @test getfield(mimetic, name) === getfield(mimetic_values, name)
        end
        for name in (
                :normal_jump_divergence, :normal_volume_defect,
                :multiplier_unknowns,
            )
            @test name ∉ propertynames(mimetic)
        end
        expected_show = "MimeticProjectionStats(" *
            join(
                ("$(name)=$(repr(getfield(mimetic, name)))"
                    for name in fieldnames(typeof(mimetic))),
                ", ",
            ) * ")"
        @test repr(mimetic) == expected_show
        unchecked_values = merge(mimetic_values, (; matrix_symmetry_error = nothing))
        unchecked = MimeticProjectionStats(; unchecked_values...)
        @test typeof(unchecked) === typeof(mimetic)
        @test isnothing(unchecked.matrix_symmetry_error)
        @test occursin("matrix_symmetry_error=nothing", repr(unchecked))

        for name in (:converged, :iterations, :initial_residual, :final_residual)
            @test name ∉ names(TwoPhaseAMR)
        end
        for (stats, expected_converged, expected_iterations, initial, final) in (
                (linear, false, 3, 10.5, 2.75),
                (cmdgfm, true, 7, 11.25, 0.75),
                (mimetic, false, 13, 17.25, 3.125),
            )
            @test TwoPhaseAMR.converged(stats) === expected_converged
            @test TwoPhaseAMR.iterations(stats) == expected_iterations
            @test TwoPhaseAMR.initial_residual(stats) == initial
            @test TwoPhaseAMR.final_residual(stats) == final

            full_result = merge(result, (; pressure = stats))
            text = sprint(io -> print_simulation_summary(io, flow, full_result))
            @test occursin("    converged: $(expected_converged)\n", text)
            @test occursin("    iterations: $(expected_iterations)\n", text)
            @test occursin("    residual: $(initial) -> $(final)\n", text)
            @test occursin("  final time: $(result.time)\n", text)
            @test occursin("  steps this run: $(result.steps)\n", text)
            @test occursin("  mesh updates this run: $(result.mesh_updates)\n", text)
            @test occursin("  total steps: $(result.total_steps)\n", text)
            @test occursin(
                "  total mesh updates: $(result.total_mesh_updates)\n", text,
            )
            @test occursin("    preconditioner: $(stats.preconditioner)\n", text)
            @test !occursin("normal volume defect", text)
            @test !occursin("normal jump divergence", text)
            if stats isa LinearSolverStats
                @test occursin("    method: cg\n", text)
                @test !occursin("    normal velocity defect:", text)
                @test !occursin("    tangential defect:", text)
            else
                for name in (
                        :mixture_divergence, :mixture_volume_defect,
                        :normal_velocity_defect, :momentum_identity_error,
                    )
                    label = replace(string(name), '_' => ' ')
                    @test occursin("    $(label): $(getfield(stats, name))\n", text)
                end
                if stats isa MimeticProjectionStats
                    @test occursin("    tangential defect: 0.5\n", text)
                else
                    @test !occursin("    tangential defect:", text)
                end
            end

            legacy = (;
                time = result.time, steps = result.steps,
                pressure = stats, mesh_updates = result.mesh_updates,
            )
            for (legacy_result, has_steps, has_updates) in (
                    (legacy, false, false),
                    (merge(legacy, (; total_steps = 9)), true, false),
                    (merge(legacy, (; total_mesh_updates = 4)), false, true),
                )
                legacy_text = sprint(
                    io -> print_simulation_summary(io, flow, legacy_result),
                )
                @test occursin("    residual: $(initial) -> $(final)\n", legacy_text)
                @test occursin("  total steps:", legacy_text) == has_steps
                @test occursin("  total mesh updates:", legacy_text) == has_updates
                if has_steps
                    @test occursin("  total steps: 9\n", legacy_text)
                end
                if has_updates
                    @test occursin("  total mesh updates: 4\n", legacy_text)
                end
            end
        end
    end
    for density in ((1.0, 1.0), (1000.0, 1.0))
        mesh = BlockTreeMesh(
            block_size = (6, 6), periodic = (true, true),
        )
        interface = InterfaceField(
            mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.27;
            lipschitz = 1,
        )
        velocity = _mimetic_test_velocity(mesh)
        pressure = CellField(mesh)
        solver = MimeticProjectionSolver(
            interface; density, preconditioner = :cholesky,
        )
        workspace = MimeticProjectionWorkspace(solver)
        tolerance = 1.0e-9
        stats = project!(
            velocity, pressure, solver, 0.02, workspace;
            tolerance, max_iterations = 2000, check_symmetry = true,
        )

        @test stats.converged
        @test stats.pressure_unknowns == ncells(mesh) - 1
        @test stats.jump_unknowns > 0
        @test stats.jump_unknowns == length(workspace.active_faces)
        @test stats.independent_constraint_rows > 0
        @test stats.independent_constraint_rows ==
            size(workspace.constraint_matrix, 1)
        @test stats.target_defect == tolerance
        @test stats.converged == (
            stats.mixture_volume_defect <= stats.target_defect &&
                stats.normal_velocity_defect <= stats.target_defect &&
                stats.tangential_defect <= stats.target_defect
        )
        @test TwoPhaseAMR._dimensionless_residual(stats) == max(
            stats.mixture_volume_defect, stats.normal_velocity_defect,
            stats.tangential_defect,
        )
        @test stats.matrix_symmetry_error == 0
        @test rank(Matrix(workspace.matrix)) == size(workspace.matrix, 1)
        @test stats.mixture_volume_defect <= tolerance
        @test stats.normal_velocity_defect <= tolerance
        @test stats.tangential_defect <= tolerance

        momentum_scale = maximum(density)
        @test maximum(abs, stats.momentum_error) <=
            256eps() * momentum_scale
        @test stats.momentum_identity_error <=
            256eps() * momentum_scale
        @test stats.orthogonality_error <=
            64tolerance * momentum_scale
        @test stats.energy_identity_error <=
            64tolerance * momentum_scale

        # The public reconstruction uses the pressure and jump retained by
        # the solve, and its phase difference is exactly xi on mixed faces.
        gradient = (FaceVelocity(mesh), FaceVelocity(mesh))
        mimetic_gradient!(gradient, pressure, solver, workspace)
        for face in workspace.mixed_faces
            @test gradient[2].interior[face] -
                gradient[1].interior[face] ≈ workspace.jump[face]
        end
    end
end

@testset "Mimetic pressure projection on an AMR interface" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 2,
        periodic = (true, true),
        refine = (key, origin, extent) -> key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 0),
    )
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.4, y - 0.4) - 0.23;
        lipschitz = 1,
    )
    velocity = _mimetic_test_velocity(mesh)
    pressure = CellField(mesh)
    solver = MimeticProjectionSolver(
        interface; density = (10.0, 1.0), preconditioner = :cholesky,
    )
    workspace = MimeticProjectionWorkspace(solver)
    tolerance = 1.0e-8
    stats = project!(
        velocity, pressure, solver, 0.02, workspace;
        tolerance, max_iterations = 4000, check_symmetry = true,
    )

    @test stats.converged
    @test stats.matrix_symmetry_error == 0
    @test stats.mixture_volume_defect <= tolerance
    @test stats.normal_velocity_defect <= tolerance
    @test stats.tangential_defect <= tolerance
    @test solver.interface.mesh ===
        solver.interface.fraction.mesh === pressure.mesh
end

@testset "Mimetic projection preserves prescribed through-flow" begin
    mesh = BlockTreeMesh(block_size = (8, 8))
    interface = InterfaceField(
        mesh, (x, y) -> x - 0.47; lipschitz = 1,
    )
    through_flow = (x, y, time, axis) -> axis == 1 ? 1.0 : 0.0
    velocity = (
        FaceVelocity(mesh, through_flow),
        FaceVelocity(mesh, through_flow),
    )
    boundary_before = map(field -> copy(field.boundary), velocity)
    solver = MimeticProjectionSolver(interface; density = (10.0, 1.0))
    stats = project!(
        velocity, CellField(mesh), solver, 0.02,
        MimeticProjectionWorkspace(solver);
        tolerance = 1.0e-9, max_iterations = 2000,
    )
    @test stats.converged
    @test isnothing(stats.matrix_symmetry_error)
    @test stats.mixture_volume_defect <= 1.0e-9
    @test stats.normal_velocity_defect <= 1.0e-9
    for phase in 1:2
        boundary_indices = _phase_boundary_indices(interface, phase)
        @test velocity[phase].boundary[boundary_indices] == boundary_before[phase][boundary_indices]
        error = maximum(_phase_interior_indices(interface, phase)) do index
            expected = mesh.faces[index].axis == 1 ? 1.0 : 0.0
            abs(velocity[phase].interior[index] - expected)
        end
        @test error <= 64eps()
    end
end

@testset "Mimetic projection with embedded geometry" begin
    meshes = (
        BlockTreeMesh(block_size = (6, 6)),
        BlockTreeMesh(
            block_size = (4, 4), max_level = 2,
            refine = (key, origin, extent) -> key.level == 0 ||
                (key.level == 1 && key.i == 0 && key.j == 1),
        ),
    )
    for mesh in meshes
        geometry = GeometryField(
            mesh, (x, y) -> y - 0.12 - 0.08x; lipschitz = hypot(1, 0.08),
        )
        interface = InterfaceField(
            mesh, (x, y) -> x - 0.52;
            geometry, lipschitz = 1,
        )
        for (index, face) in pairs(mesh.faces)
            minus = TwoPhaseAMR._get(
                interface.conditional_fraction, face.minus,
            )
            plus = TwoPhaseAMR._get(
                interface.conditional_fraction, face.plus,
            )
            if minus == plus == 1 && geometry.aperture[index] > 0
                liquid, gas = TwoPhaseAMR._phase_face_apertures(
                    interface, geometry, index,
                )
                @test liquid ≈ geometry.aperture[index] atol = 128eps()
                @test gas <= 128eps()
            end
        end
        velocity = _mimetic_test_velocity(mesh)
        enforce_no_permeation!(velocity)
        pressure = CellField(mesh)
        solver = MimeticProjectionSolver(
            interface; density = (1000.0, 1.0),
            preconditioner = :cholesky,
        )
        workspace = MimeticProjectionWorkspace(solver)
        tolerance = 1.0e-8
        stats = project!(
            velocity, pressure, solver, 0.02, workspace;
            tolerance, max_iterations = 4000, check_symmetry = true,
        )

        @test stats.converged
        @test stats.matrix_symmetry_error == 0
        @test stats.mixture_volume_defect <= tolerance
        @test stats.normal_velocity_defect <= tolerance
        @test stats.tangential_defect <= tolerance
        @test _phase_storage_contract(velocity, interface)
        if nblocks(mesh) == 1
            @test stats.momentum_identity_error <= 512eps() * 1000
            @test stats.energy_identity_error <= 64tolerance * 1000
            @test stats.orthogonality_error <= 64tolerance * 1000
        end
    end
end

@testset "Two-velocity Navier--Stokes integration" begin
    mesh = BlockTreeMesh(
        block_size = (6, 6), periodic = (true, true),
    )
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.23;
        lipschitz = 1,
    )
    liquid_velocity = (x, y, time, axis) -> axis == 1 ?
        -0.1(y - 0.5) : 0.1(x - 0.5)
    gas_velocity = (x, y, time, axis) -> zero(x)
    flow = FlowSimulation(
        mesh, TwoPhase(interface; density = (1.0, 1.0),
            dynamic_viscosity = (0.0, 0.0));
        velocity = (liquid_velocity, gas_velocity), dt = 1.0e-3,
        velocity_model = TwoVelocityModel(
            interface_transport = FrozenInterfaceFluxes(),
            pressure_projection = MimeticProjection(),
        ),
    )
    @test flow.velocity isa TwoPhaseVelocity
    @test isnothing(flow.workspace.two_phase.two_velocity[])
    result = simulate!(
        flow, 1.0e-3; tolerance = 1.0e-7, max_cycles = 1000,
    )
    @test result.steps == 1
    @test result.pressure isa MimeticProjectionStats
    @test result.pressure.converged
    @test !isnothing(flow.workspace.two_phase.two_velocity[])
    @test flow.time == 1.0e-3

    force_mesh = BlockTreeMesh(
        block_size = (12, 12), periodic = (true, true),
    )
    force_interface = InterfaceField(
        force_mesh,
        (x, y) -> hypot(x - 0.5, y - 0.5) - 0.22;
        lipschitz = 1,
    )
    force_solver = IncompressibleNavierStokesSolver(
        force_mesh; density = (10.0, 1.0),
        dynamic_viscosity = (0.0, 0.0), interface = force_interface,
        surface_tension = 0.01, surface_tension_formulation = CGFM(),
        interface_transport = FrozenInterfaceFluxes(),
    )
    force_workspace = NavierStokesWorkspace(force_solver)
    force_stats = step!(
        (FaceVelocity(force_mesh), FaceVelocity(force_mesh)),
        CellField(force_mesh), force_solver, 1.0e-4, force_workspace;
        tolerance = 1.0e-7, max_cycles = 2000,
    )
    @test force_stats.converged
    @test force_stats.initial_residual > 0

    viscous_interface = copy(interface)
    viscous_solver = IncompressibleNavierStokesSolver(
        mesh; density = (1.0, 1.0),
        dynamic_viscosity = (1.0e-3, 1.0e-3),
        interface = viscous_interface,
    )
    viscous_workspace = NavierStokesWorkspace(viscous_solver)
    viscous_velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
    viscous_stats = step!(
        viscous_velocity, CellField(mesh), viscous_solver, 1.0e-3,
        viscous_workspace; tolerance = 1.0e-10, max_cycles = 1000,
    )
    @test viscous_stats.converged
    @test viscous_workspace.two_phase.diffusion_solver.stats.converged

    adaptive_mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 1,
        periodic = (true, true),
    )
    adaptive_interface = InterfaceField(
        adaptive_mesh,
        (x, y) -> x - 0.47;
        lipschitz = 1,
    )
    adaptive_flow = FlowSimulation(
        adaptive_mesh, TwoPhase(adaptive_interface; density = (10.0, 1.0),
            dynamic_viscosity = (0.01, 0.001));
        velocity_model = TwoVelocityModel(),
        velocity = (
            (x, y, time, axis) -> 0.0,
            (x, y, time, axis) -> 0.0,
        ),
        dt = 1.0e-3,
        adaptivity = Adaptivity(criteria = (
            InterfaceRefinementIndicator(adaptive_interface; level = 1),
        )),
    )
    adaptive_result = simulate!(
        adaptive_flow, 2.0e-3;
        tolerance = 1.0e-7, max_cycles = 1000,
    )
    @test adaptive_result.steps == 2
    @test adaptive_result.mesh_updates == 1
    @test adaptive_result.pressure.converged
    @test adaptive_flow.workspace.two_phase.diffusion_solver.stats.converged
end
