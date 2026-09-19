using Test, TwoPhaseAMR, LinearAlgebra

const CMD = TwoPhaseAMR

function _cmdgfm_test_velocity(mesh; jump = (0.37, -0.21), affine = false)
    liquid = FaceVelocity(
        mesh, (x, y, time, axis) -> begin
            base = affine ? (axis == 1 ? 0.2 + 0.3x - 0.1y : -0.4 + 0.2x + 0.25y) :
                (axis == 1 ? 0.2 : -0.4)
            base
        end
    )
    gas = FaceVelocity(
        mesh, (x, y, time, axis) -> begin
            base = affine ? (axis == 1 ? 0.2 + 0.3x - 0.1y : -0.4 + 0.2x + 0.25y) :
                (axis == 1 ? 0.2 : -0.4)
            base + jump[axis]
        end
    )
    return (liquid, gas)
end

function _cmdgfm_stencils(interface; density = (10.0, 1.0))
    solver = CMDGFMProjectionSolver(interface; density, preconditioner = nothing)
    workspace = CMDGFMProjectionWorkspace(solver)
    data = CMD._mimetic_face_data(solver, workspace)
    return solver, workspace, data, CMD._cmdgfm_jump_stencils(solver, data)
end

function _cmdgfm_projection_velocity(mesh)
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

@testset "CMDGFM jump interpolant consistency" begin
    configurations = (
        BlockTreeMesh(block_size = (12, 12)),
        BlockTreeMesh(base_mesh = (12, 12), block_size = (6, 6)),
        BlockTreeMesh(
            block_size = (4, 4), max_level = 2,
            refine = (key, origin, extent) -> key.level == 0 ||
                (key.level == 1 && key.i == 0 && key.j == 0),
        ),
    )
    normal = (cospi(1 / 3), sinpi(1 / 3))
    jump = (0.37, -0.21)
    for mesh in configurations
        interface = InterfaceField(
            mesh,
            (x, y) -> normal[1] * (x - 0.5) + normal[2] * (y - 0.5);
            lipschitz = 1,
        )
        _, _, _, stencils = _cmdgfm_stencils(interface)
        velocity = _cmdgfm_test_velocity(mesh; jump)
        checked = 0
        for stencil in stencils
            stencil.resolved || continue
            stencil.beta > sqrt(eps()) || continue
            interpolated = cmdgfm_jump_interpolant(velocity, stencil)
            if stencil.full_remove
                axis = mesh.faces[stencil.face].axis
                @test interpolated[axis] ≈ jump[axis] atol = 256eps()
                checked += 1
                continue
            end
            @test interpolated[1] ≈ jump[1] atol = 256eps()
            @test interpolated[2] ≈ jump[2] atol = 256eps()
            @test sum(stencil.transverse_weights) ≈ 1 atol = 64eps()
            checked += 1
        end
        @test checked > 0
    end

    # The cross-face inference term is first-order consistent for a smooth
    # common velocity plus a constant phase jump.
    errors = map((8, 16, 32)) do resolution
        mesh = BlockTreeMesh(
            block_size = (resolution, resolution),
            periodic = (false, false),
        )
        interface = InterfaceField(
            mesh,
            (x, y) -> normal[1] * (x - 0.5) + normal[2] * (y - 0.5);
            lipschitz = 1,
        )
        _, _, _, stencils = _cmdgfm_stencils(interface)
        velocity = _cmdgfm_test_velocity(mesh; jump, affine = true)
        values = [
            max(
                abs(cmdgfm_jump_interpolant(velocity, stencil)[1] - jump[1]),
                abs(cmdgfm_jump_interpolant(velocity, stencil)[2] - jump[2]),
            ) for stencil in stencils if
                stencil.resolved && !stencil.full_remove &&
                stencil.beta >= 0.1
        ]
        sum(values) / length(values)
    end
    @test all(<(0), diff(collect(errors)))
    @test log2(errors[1] / errors[end]) / 2 >= 0.8
end

@testset "CMDGFM revised beta and diagonal dominance" begin
    mesh = BlockTreeMesh(
        block_size = (16, 16), periodic = (true, true),
    )
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.27;
        lipschitz = 1,
    )
    solver, workspace, data, stencils = _cmdgfm_stencils(interface)
    @test any(stencil -> stencil.resolved && stencil.beta < 1, stencils)
    for stencil in stencils
        stencil.resolved || continue
        @test 0 <= stencil.beta <= 1
        abs(stencil.ratio) <= 1 && @test stencil.beta == 1
        if abs(stencil.ratio) > 1
            gas_face = stencil.gas_inference_face
            liquid_face = stencil.liquid_inference_face
            gamma_liquid_gas =
                data.liquid_mass[gas_face] / data.total_mass[gas_face]
            gamma_gas_liquid =
                data.gas_mass[liquid_face] / data.total_mass[liquid_face]
            expected = (1 - gamma_liquid_gas - gamma_gas_liquid) /
                (abs(stencil.ratio) - gamma_liquid_gas - gamma_gas_liquid)
            @test stencil.beta ≈ expected
        end
    end
    velocity = _cmdgfm_test_velocity(mesh; affine = true)
    assembled = CMD._assemble_cmdgfm_system!(
        solver, workspace, 0.02, velocity,
    )
    @test CMD._cmdgfm_jump_diagonal_margin(assembled.jump_block) >=
        -512eps()
    @test all(diag(Matrix(assembled.jump_block)) .≈ 1)
end

@testset "Equal-density CMDGFM phase-exchange symmetry" begin
    mesh = BlockTreeMesh(
        base_mesh = (16, 48), block_size = (8, 8),
        origin = (0.0, -1.5), extent = (1.0, 3.0),
        periodic = (true, false),
    )
    levelset = (x, y) -> y - 0.25 - 0.01cospi(2x)
    liquid = InterfaceField(mesh, levelset; lipschitz = 1)
    gas = InterfaceField(
        mesh, (x, y) -> -levelset(x, y); lipschitz = 1,
    )
    _, _, liquid_data, liquid_stencils = _cmdgfm_stencils(
        liquid; density = (1.0, 1.0),
    )
    _, _, gas_data, gas_stencils = _cmdgfm_stencils(
        gas; density = (1.0, 1.0),
    )

    @test liquid_data.liquid_mass ≈ gas_data.gas_mass atol = 64eps()
    @test liquid_data.gas_mass ≈ gas_data.liquid_mass atol = 64eps()
    liquid_by_face = Dict(stencil.face => stencil for stencil in liquid_stencils)
    gas_by_face = Dict(stencil.face => stencil for stencil in gas_stencils)
    @test keys(liquid_by_face) == keys(gas_by_face)
    for face in keys(liquid_by_face)
        first = liquid_by_face[face]
        second = gas_by_face[face]
        @test first.resolved == second.resolved
        @test first.full_remove == second.full_remove
        @test first.ratio ≈ second.ratio atol = 256eps() rtol = 256eps()
        @test first.beta ≈ second.beta atol = 256eps()
        @test first.transverse_faces == second.transverse_faces
        @test first.transverse_weights ≈ second.transverse_weights atol = 256eps()
        @test first.gas_inference_face == second.liquid_inference_face
        @test first.liquid_inference_face == second.gas_inference_face
    end
end

@testset "CMDGFM reference impose/infer eligibility" begin
    mesh = BlockTreeMesh(
        block_size = (32, 32), periodic = (true, false),
    )
    interface = InterfaceField(
        mesh, (x, y) -> 0.5 + 0.02cospi(2x) - y;
        lipschitz = 1.01,
    )
    _, _, data, stencils = _cmdgfm_stencils(interface; density = (3.0, 1.0))
    inferred = 0
    for stencil in stencils
        stencil.resolved || continue
        face = mesh.faces[stencil.face]
        if stencil.full_remove
            @test abs(stencil.ratio) <= 128eps()
            @test isempty(stencil.transverse_faces)
        elseif stencil.beta < 1
            gas_face = mesh.faces[stencil.gas_inference_face]
            liquid_face = mesh.faces[stencil.liquid_inference_face]
            # The reference uses the same-axis faces at the two tangential
            # endpoints of f, rather than the faces across its adjacent cells.
            @test gas_face.axis == liquid_face.axis == face.axis
            @test gas_face.center[face.axis] ≈ face.center[face.axis]
            @test liquid_face.center[face.axis] ≈ face.center[face.axis]
            @test data.gas_mass[stencil.gas_inference_face] >=
                data.liquid_mass[stencil.gas_inference_face]
            @test data.liquid_mass[stencil.liquid_inference_face] >=
                data.gas_mass[stencil.liquid_inference_face]
            inferred += 1
        end
    end
    @test inferred > 0

    flat_interface = InterfaceField(
        mesh, (x, y) -> 0.503 - y; lipschitz = 1,
    )
    _, _, _, flat_stencils = _cmdgfm_stencils(
        flat_interface; density = (3.0, 1.0),
    )
    @test any(stencil -> stencil.resolved && stencil.full_remove, flat_stencils)

    solver = CMDGFMProjectionSolver(interface; density = (3.0, 1.0))
    workspace = CMDGFMProjectionWorkspace(solver)
    assembled = CMD._assemble_cmdgfm_system!(
        solver, workspace, 0.01, _cmdgfm_test_velocity(mesh),
    )
    @test all(stencil -> stencil.resolved, assembled.stencils)
    @test assembled.underresolved > 0
end

@testset "CMDGFM projection invariants" begin
    meshes = (
        (
            BlockTreeMesh(
                block_size = (8, 8), periodic = (true, true),
            ),
            nothing,
        ),
        let mesh = BlockTreeMesh(
                block_size = (4, 4), max_level = 2,
                refine = (key, origin, extent) -> key.level == 0 ||
                    (key.level == 1 && key.i == 0 && key.j == 1),
            )
            geometry = GeometryField(
                mesh, (x, y) -> y - 0.1 - 0.05x;
                lipschitz = hypot(1, 0.05),
            )
            (mesh, geometry)
        end,
    )
    for (mesh, geometry) in meshes, density in ((1.0, 1.0), (1000.0, 1.0))
        interface = InterfaceField(
            mesh, (x, y) -> hypot(x - 0.45, y - 0.43) - 0.25;
            geometry, lipschitz = 1,
        )
        velocity = _cmdgfm_projection_velocity(mesh)
        isnothing(geometry) || enforce_no_permeation!(velocity)
        pressure = CellField(mesh)
        solver = CMDGFMProjectionSolver(interface; density)
        workspace = CMDGFMProjectionWorkspace(solver)
        tolerance = 1.0e-9
        stats = project!(
            velocity, pressure, solver, 0.02, workspace;
            tolerance, max_iterations = 4000,
        )
        @test stats.converged
        @test stats.matrix_asymmetry > 0
        @test stats.mixture_volume_defect <= tolerance
        @test stats.normal_velocity_defect <= tolerance
        @test stats.minimum_jump_diagonal_margin >= -512eps()
        @test stats.momentum_identity_error <= 1024eps() * maximum(density)
        if isnothing(geometry)
            @test maximum(abs, stats.momentum_error) <=
                1024eps() * maximum(density)
        end
        @test all(isfinite, velocity[1].interior)
        @test all(isfinite, velocity[2].interior)
    end
end

@testset "CMDGFM block-preconditioned BiCGSTAB" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), periodic = (true, true),
    )
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.27;
        lipschitz = 1,
    )
    solver = CMDGFMProjectionSolver(interface; density = (1000.0, 1.0))
    workspace = CMDGFMProjectionWorkspace(solver)
    stats = project!(
        _cmdgfm_projection_velocity(mesh), CellField(mesh), solver, 0.02,
        workspace; tolerance = 1.0e-8, max_iterations = 2000,
    )
    @test stats.converged
    @test stats.preconditioner == :cmdgfm_block
    @test stats.iterations < 100
    block = workspace.block_preconditioner
    @test block.pressure_preconditioner.multigrid isa
        TreeGeometricMultigridPreconditioner
    @test CMD._preconditioner_name(block.jump_preconditioner) == :ilu
    floor = interface.fraction_floor
    data = CMD._mimetic_face_data(solver, workspace)
    @test all(workspace.mixed_faces) do face
        fraction = data.liquid_fraction[face]
        fraction > floor && fraction < 1 - floor
    end

    repeated = project!(
        _cmdgfm_projection_velocity(mesh), CellField(mesh), solver, 0.02,
        workspace; tolerance = 1.0e-8, max_iterations = 2000,
    )
    @test repeated.converged
    @test repeated.pressure_hierarchy_reused
    @test repeated.jump_factor_reused

    pure_interface = InterfaceField(mesh, (x, y) -> -1.0; lipschitz = 1)
    pure_solver = CMDGFMProjectionSolver(pure_interface)
    pure_stats = project!(
        _cmdgfm_projection_velocity(mesh), CellField(mesh), pure_solver, 0.02,
        CMDGFMProjectionWorkspace(pure_solver);
        tolerance = 1.0e-8, max_iterations = 2000,
    )
    @test pure_stats.converged
    @test pure_stats.jump_unknowns == 0
end


@testset "Default CMDGFM Navier--Stokes policy and AMR rebuild" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 1,
        periodic = (true, true),
    )
    interface = InterfaceField(
        mesh, (x, y) -> x - 0.47; lipschitz = 1,
    )
    flow = FlowSimulation(
        mesh; density = (10.0, 1.0),
        dynamic_viscosity = (0.0, 0.0), interface,
        velocity = (
            (x, y, time, axis) -> 0.0,
            (x, y, time, axis) -> 0.0,
        ),
        dt = 1.0e-3,
        refinement_indicators = (
            InterfaceRefinementIndicator(interface; level = 1),
        ),
    )
    result = simulate!(
        flow, 2.0e-3; tolerance = 1.0e-7, max_cycles = 1000,
    )
    @test result.steps == 2
    @test result.mesh_updates == 1
    @test result.pressure isa CMDGFMProjectionStats
    @test result.pressure.converged
    @test flow.solver.two_phase.pressure_projection isa CMDGFMProjection

    force_mesh = BlockTreeMesh(
        block_size = (8, 8), periodic = (true, false),
    )
    force_interface = InterfaceField(
        force_mesh, (x, y) -> 0.5 + 0.02cospi(2x) - y;
        lipschitz = 1.01,
    )
    force_solver = NavierStokesSolver(
        force_mesh; density = (10.0, 1.0),
        dynamic_viscosity = (0.0, 0.0), interface = force_interface,
        surface_tension = 0.01, surface_tension_formulation = GFM(),
        gravity = (0.0, -1.0), boundary = FreeSlip(),
        two_velocity_projection = CMDGFMProjection(),
    )
    force_stats = step!(
        (FaceVelocity(force_mesh), FaceVelocity(force_mesh)),
        CellField(force_mesh), force_solver, 1.0e-4,
        NavierStokesWorkspace(force_solver);
        tolerance = 1.0e-7, max_cycles = 2000,
    )
    @test force_stats.converged
    @test force_stats.initial_residual > 0
end
