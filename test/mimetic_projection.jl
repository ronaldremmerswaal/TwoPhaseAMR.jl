using LinearAlgebra: norm, rank

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
            tolerance, max_iterations = 2000,
        )

        @test stats.converged
        @test stats.pressure_unknowns == ncells(mesh) - 1
        @test stats.jump_unknowns > 0
        @test stats.multiplier_unknowns > 0
        @test stats.matrix_symmetry_error == 0
        @test rank(Matrix(workspace.matrix)) == size(workspace.matrix, 1)
        @test stats.mixture_volume_defect <= tolerance
        @test stats.normal_volume_defect <= tolerance
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
        tolerance, max_iterations = 4000,
    )

    @test stats.converged
    @test stats.matrix_symmetry_error == 0
    @test stats.mixture_volume_defect <= tolerance
    @test stats.normal_volume_defect <= tolerance
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
    @test stats.mixture_volume_defect <= 1.0e-9
    @test stats.normal_volume_defect <= 1.0e-9
    @test velocity[1].boundary == boundary_before[1]
    @test velocity[2].boundary == boundary_before[2]
    for phase in 1:2
        error = maximum(eachindex(mesh.faces)) do index
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
            tolerance, max_iterations = 4000,
        )

        @test stats.converged
        @test stats.matrix_symmetry_error == 0
        @test stats.mixture_volume_defect <= tolerance
        @test stats.normal_volume_defect <= tolerance
        @test stats.tangential_defect <= tolerance
        @test all(isfinite, velocity[1].interior)
        @test all(isfinite, velocity[2].interior)
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
        mesh; density = (1.0, 1.0),
        dynamic_viscosity = (0.0, 0.0), interface,
        velocity = (liquid_velocity, gas_velocity), dt = 1.0e-3,
        interface_method = FrozenInterfaceFluxes(),
        two_velocity_projection = MimeticProjection(),
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
    force_solver = NavierStokesSolver(
        force_mesh; density = (10.0, 1.0),
        dynamic_viscosity = (0.0, 0.0), interface = force_interface,
        surface_tension = 0.01, surface_tension_formulation = GFM(),
        interface_method = FrozenInterfaceFluxes(),
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
    viscous_solver = NavierStokesSolver(
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
        adaptive_mesh; density = (10.0, 1.0),
        dynamic_viscosity = (0.01, 0.001), interface = adaptive_interface,
        velocity = (
            (x, y, time, axis) -> 0.0,
            (x, y, time, axis) -> 0.0,
        ),
        dt = 1.0e-3,
        refinement_indicators = (
            InterfaceRefinementIndicator(adaptive_interface; level = 1),
        ),
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
