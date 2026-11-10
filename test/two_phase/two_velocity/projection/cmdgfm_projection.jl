using Test, TwoPhaseAMR, LinearAlgebra, SparseArrays
include("../helpers.jl")

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
    data = CMD._projection_face_data(solver, workspace)
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
        @test _phase_storage_contract(velocity, interface)
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
    data = CMD._projection_face_data(solver, workspace)
    @test all(workspace.mixed_faces) do face
        fraction = data.liquid_fraction[face]
        fraction > floor && fraction < 1 - floor
    end
    first_krylov = workspace.krylov

    repeated = project!(
        _cmdgfm_projection_velocity(mesh), CellField(mesh), solver, 0.02,
        workspace; tolerance = 1.0e-8, max_iterations = 2000,
    )
    @test repeated.converged
    @test repeated.pressure_hierarchy_reused
    @test repeated.jump_factor_reused
    @test workspace.krylov === first_krylov
    changed_jump = copy(workspace.jump_block)
    if !isempty(nonzeros(changed_jump))
        nonzeros(changed_jump)[1] += 0.01
        @test !CMD._cmdgfm_same_jump_matrix(
            workspace.block_preconditioner,
            (mixed_faces = workspace.mixed_faces, jump_block = changed_jump),
        )
    end

    previous_coefficients = copy(
        workspace.block_preconditioner.pressure_preconditioner.multigrid.
        operator.face_coefficient,
    )
    initialize!(
        interface, (x, y) -> hypot(x - 0.53, y - 0.5) - 0.27,
    )
    moved = project!(
        _cmdgfm_projection_velocity(mesh), CellField(mesh), solver, 0.02,
        workspace; tolerance = 1.0e-8, max_iterations = 2000,
    )
    @test moved.converged
    @test moved.pressure_hierarchy_reused
    current_coefficients = CMD._projection_pressure_operator(solver).
    face_coefficient
    @test previous_coefficients != current_coefficients
    @test workspace.block_preconditioner.pressure_preconditioner.multigrid.
    operator.face_coefficient == current_coefficients

    # The same cache must accommodate both an empty jump block and a later
    # nonempty one without discarding its pressure and Krylov workspaces.
    initialize!(interface, (x, y) -> -1.0)
    pure_reused = project!(
        _cmdgfm_projection_velocity(mesh), CellField(mesh), solver, 0.02,
        workspace; tolerance = 1.0e-8, max_iterations = 2000,
    )
    @test pure_reused.converged
    @test pure_reused.jump_unknowns == 0
    @test isnothing(workspace.block_preconditioner.jump_preconditioner)
    @test workspace.krylov === first_krylov
    initialize!(interface, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.27)
    mixed_again = project!(
        _cmdgfm_projection_velocity(mesh), CellField(mesh), solver, 0.02,
        workspace; tolerance = 1.0e-8, max_iterations = 2000,
    )
    @test mixed_again.converged
    @test mixed_again.jump_unknowns > 0
    @test CMD._preconditioner_name(workspace.block_preconditioner.jump_preconditioner) == :ilu
    @test workspace.krylov === first_krylov

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

@testset "CMDGFM AMR whole-system ILU regression (issue #126)" begin
    # Match the Chapter 11 drop and predictor, including the two-level band.
    centre = (0.513, 0.468)
    radius, band = 0.2371, 0.02
    function near_circle(key, origin, extent)
        xs = (origin[1], origin[1] + extent[1])
        ys = (origin[2], origin[2] + extent[2])
        cx, cy = centre
        nearest = hypot(clamp(cx, xs...) - cx, clamp(cy, ys...) - cy)
        farthest = maximum(hypot(x - cx, y - cy) for x in xs, y in ys)
        return nearest - band <= radius <= farthest + band
    end
    mesh = BlockTreeMesh(
        base_mesh = (32, 32), block_size = (8, 8), periodic = (false, false),
        max_level = 2, refine = near_circle,
    )
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - centre[1], y - centre[2]) - radius;
        lipschitz = 1,
    )
    common(x, y, axis) = axis == 1 ?
        0.4 + sinpi(2x) * cospi(y) : -0.2 + cospi(x) * sinpi(2y)
    jump(x, y, axis) = axis == 1 ? 0.37 + 0.1sinpi(2y) : -0.21 + 0.1cospi(2x)
    function predictor()
        liquid = FaceVelocity(mesh, (x, y, t, axis) -> common(x, y, axis))
        gas = FaceVelocity(mesh, (x, y, t, axis) -> common(x, y, axis) + jump(x, y, axis))
        velocity = (liquid, gas)
        enforce_no_permeation!(velocity)
        CMD._mark_inactive_phase_velocities!(velocity, interface)
        return velocity
    end

    dt, tolerance, budget = 0.01, 1.0e-10, 20_000
    for preconditioner in (:block, :ilu)
        solver = CMDGFMProjectionSolver(interface; density = (1000.0, 1.0), preconditioner)
        work = CMDGFMProjectionWorkspace(solver)
        stats = project!(
            predictor(), CellField(mesh), solver, dt, work;
            tolerance, max_iterations = budget, warm_start = false,
        )
        @test stats.pressure_unknowns + stats.jump_unknowns == 7933
        @test stats.jump_unknowns == 382
        @test stats.preconditioner == (preconditioner === :block ? :cmdgfm_block : :ilu)
        @test stats.converged
        @test stats.mixture_volume_defect <= tolerance
        @test stats.normal_velocity_defect <= tolerance
        @test 0 < stats.iterations <= budget
        if preconditioner === :block
            # On the original main fixture the block solve takes 16 iterations.
            @test 15 <= stats.iterations <= 17
        end
        residual = work.right_hand_side - work.matrix * work.solution
        volumes = work.pressure_cache.operator.volumes
        np = length(work.pressure_indices)
        @test maximum(
            dt^2 * abs(residual[index]) / volumes[cell]
                for (index, cell) in pairs(work.pressure_indices)
        ) <= tolerance
        @test maximum(
            dt^2 * abs(residual[np + index]) / CMD._projection_face_spacing(mesh, face)
                for (index, face) in pairs(work.mixed_faces)
        ) <= tolerance
    end
end

@testset "CMDGFM omitted gauge row and iteration budget" begin
    for geometry_case in (:periodic, :split)
        mesh = geometry_case === :periodic ?
            BlockTreeMesh(block_size = (8, 8), periodic = (true, true)) :
            BlockTreeMesh(block_size = (16, 8), periodic = (false, false))
        geometry = geometry_case === :periodic ? nothing :
            GeometryField(mesh, (x, y) -> abs(x - 0.5) - 0.09; lipschitz = 1)
        interface = InterfaceField(
            mesh, (x, y) -> hypot(x - 0.45, y - 0.43) - 0.25;
            geometry, lipschitz = 1,
        )
        solver = CMDGFMProjectionSolver(interface; density = (10.0, 1.0), preconditioner = :ilu)
        work = CMDGFMProjectionWorkspace(solver)
        velocity = _cmdgfm_projection_velocity(mesh)
        enforce_no_permeation!(velocity)
        CMD._mark_inactive_phase_velocities!(velocity, interface)
        dt, reference_dt = 0.02, 0.04
        inherited = [1.0e-3 * cos(index) for index in 1:ncells(mesh)]
        assembled = CMD._assemble_cmdgfm_system!(
            solver, work, dt, velocity;
            _inherited_cell_flux = inherited, _reference_dt = reference_dt,
        )
        np = length(assembled.pressure_indices)
        resize!(work.solution, length(assembled.right_hand_side))
        for index in eachindex(work.solution)
            work.solution[index] = sin(index) / 10
        end
        anchor_defect = CMD._cmdgfm_anchor_residual_defect(work, assembled, dt)
        CMD._copy_projection_cell_solution!(
            work.pressure_increment,
            zip(assembled.pressure_indices, view(work.solution, 1:np)),
        )
        fill!(work.jump, 0)
        for (index, face) in pairs(assembled.mixed_faces)
            work.jump[face] = work.solution[np + index]
        end
        gradients = (work.liquid_gradient, work.gas_gradient)
        cmdgfm_gradient!(
            gradients, work.pressure_increment, solver, work,
            assembled.pressure_operator, assembled.data,
        )
        CMD._projection_correct_phase_velocities!(velocity, gradients, dt, Float64)
        flux = zeros(ncells(mesh))
        CMD._cmdgfm_cell_flux!(flux, assembled.data, interface, velocity)
        volumes = assembled.pressure_operator.volumes
        anchors = findall(eachindex(volumes)) do cell
            volumes[cell] > 0 && iszero(work.pressure_cache.pressure_lookup[cell])
        end
        @test length(anchors) == (geometry_case === :periodic ? 1 : 2)
        target = (1 - dt / reference_dt) .* inherited
        expected = maximum(dt * abs(flux[cell] - target[cell]) / volumes[cell] for cell in anchors)
        @test expected > 0
        @test anchor_defect ≈ expected rtol = 2.0e-12 atol = 1.0e-14

        # A nonconverged solve must not exceed its budget, including any retry.
        limited = project!(
            _cmdgfm_projection_velocity(mesh), CellField(mesh), solver, dt, work;
            tolerance = 1.0e-14, max_iterations = 1, warm_start = false,
        )
        @test !limited.converged
        @test limited.iterations <= 1
        @test all(isfinite, work.solution)
    end
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
        mesh, TwoPhase(
            interface; density = (10.0, 1.0),
            dynamic_viscosity = (0.0, 0.0)
        );
        velocity_model = TwoVelocityModel(),
        velocity = (
            (x, y, time, axis) -> 0.0,
            (x, y, time, axis) -> 0.0,
        ),
        dt = 1.0e-3,
        adaptivity = Adaptivity(
            criteria = (
                InterfaceRefinementIndicator(interface; level = 1),
            )
        ),
    )
    old_state = CMD._two_velocity_flow_workspace!(flow.workspace.two_phase, flow.system.two_phase)
    old_work = old_state.projection_workspace
    result = simulate!(
        flow, 2.0e-3; tolerance = 1.0e-7, max_cycles = 1000,
    )
    @test result.steps == 2
    @test result.mesh_updates == 1
    @test result.pressure isa CMDGFMProjectionStats
    @test result.pressure.converged
    @test flow.system.two_phase.pressure_projection isa CMDGFMProjection
    @test flow.workspace.two_phase.two_velocity[].projection_workspace === old_work
    @test CMD._cmdgfm_same_topology(old_work.pressure_cache.keys, mesh)

    force_mesh = BlockTreeMesh(
        block_size = (8, 8), periodic = (true, false),
    )
    force_interface = InterfaceField(
        force_mesh, (x, y) -> 0.5 + 0.02cospi(2x) - y;
        lipschitz = 1.01,
    )
    force_solver = NavierStokesSystem(
        force_mesh; density = (10.0, 1.0),
        dynamic_viscosity = (0.0, 0.0), interface = force_interface,
        surface_tension = 0.01, surface_tension_formulation = CGFM(),
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

@testset "CMDGFM hydrostatic rest with embedded solids (issue #76)" begin
    # Composite planar-interface fixture from proof_review/_shared/forces_audit.jl:
    # the right half is refined, with a sloped solid entirely in either phase.
    gravity = (0.0, -1.0)
    dt = 1.0e-3
    for solid in (:liquid, :gas)
        mesh = BlockTreeMesh(
            base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0),
            periodic = (false, false), max_level = 1,
            refine = (key, origin, extent) -> key.level == 0 && key.i >= 2,
        )
        shape = solid == :liquid ?
            ((x, y) -> y - 0.9 - 0.1x) :
            ((x, y) -> 7.1 - 0.1x - y)
        geometry = GeometryField(mesh, shape; lipschitz = hypot(1, 0.1))
        interface = InterfaceField(
            mesh, (x, y) -> y - 4.6;
            geometry, contact_angle = pi / 2, lipschitz = 1,
        )
        # This is a submerged solid, not a liquid-gas-solid triple-cell test.
        @test all(CMD._cell_references(mesh)) do ref
            c = CMD._conditional_fraction(interface, ref)
            open = CMD._get(geometry.open_fraction, ref)
            !(0 < c < 1 && 0 < open < 1)
        end
        solver = NavierStokesSystem(
            mesh; geometry, interface, density = (1000.0, 1.0),
            dynamic_viscosity = (0.0, 0.0), gravity, boundary = FreeSlip(),
            two_velocity_projection = CMDGFMProjection(),
        )
        velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
        pressure = CellField(mesh)
        work = NavierStokesWorkspace(solver)
        stats = step!(
            velocity, pressure, solver, dt, work;
            tolerance = 1.0e-12, max_cycles = 6000,
        )
        @test stats.converged
        @test _phase_storage_contract(velocity, interface)
        # The exact discrete hydrostatic state is at rest; allow 1e-8 of
        # the gravity impulse for the iterative projection and round-off.
        for phase in 1:2
            interior = _phase_interior_indices(interface, phase)
            boundary = _phase_boundary_indices(interface, phase)
            @test !isempty(interior)
            @test maximum(abs, velocity[phase].interior[interior]) <=
                1.0e-8 * dt * hypot(gravity...)
            @test maximum(abs, velocity[phase].boundary[boundary]; init = 0.0) <=
                1.0e-8 * dt * hypot(gravity...)
        end
        first_pressure = copy(pressure)
        pressure_scale = maximum(
            maximum(abs, interior(first_pressure, block))
                for block in eachindex(first_pressure.blocks)
        )
        @test pressure_scale > 1
        for step in 2:3
            # A fresh workspace on the last step models resuming with the
            # physical pressure but without a cached pressure increment.
            step == 3 && (work = NavierStokesWorkspace(solver))
            stats = step!(
                velocity, pressure, solver, dt, work; time = (step - 1) * dt,
                tolerance = 1.0e-12, max_cycles = 6000,
            )
            @test stats.converged
            @test _phase_storage_contract(velocity, interface)
            for phase in 1:2
                active = _phase_interior_indices(interface, phase)
                @test maximum(abs, velocity[phase].interior[active]) <=
                    1.0e-7 * dt * hypot(gravity...)
            end
            for block in eachindex(pressure.blocks)
                @test maximum(
                    abs, interior(pressure, block) .- interior(first_pressure, block)
                ) <= 1.0e-5 * pressure_scale
            end
        end
    end
end

@testset "CMDGFM dimensionless jump and residual defects" begin
    mesh = BlockTreeMesh(
        base_mesh = (8, 4), block_size = (4, 4), extent = (4.0, 1.0),
        max_level = 1, periodic = (false, false),
        refine = (key, origin, extent) -> key.level == 0 && key.i == 0,
    )
    split_face = findfirst(mesh.faces) do face
        face.axis == 1 &&
            mesh.blocks[face.minus.block].key.level != mesh.blocks[face.plus.block].key.level
    end
    fine_y_face = findfirst(mesh.faces) do face
        face.axis == 2 && mesh.blocks[face.minus.block].key.level == 1 &&
            mesh.blocks[face.plus.block].key.level == 1
    end
    @test !isnothing(split_face) && !isnothing(fine_y_face)
    faces = [split_face, fine_y_face]
    # Both anisotropic face orientations use the adjacent fine cell's
    # smallest edge, not the normal width or center-to-center distance.
    h = [0.125, 0.125]
    coarse_x_face = findfirst(mesh.faces) do face
        face.axis == 1 && mesh.blocks[face.minus.block].key.level == 0 &&
            mesh.blocks[face.plus.block].key.level == 0
    end
    @test !isnothing(coarse_x_face)
    @test CMD._projection_face_spacing(mesh, coarse_x_face) == 0.25
    @test [CMD._projection_face_spacing(mesh, face) for face in faces] == h
    velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
    velocity[2].interior[faces] .= (2.0, 0.3)
    stencils = [
        CMD.CMDGFMJumpStencil(face, Int[], Float64[], 0.0, 0, 0, 1.0, true, true)
            for face in faces
    ]
    dt = 0.04
    for index in eachindex(faces)
        @test CMD._cmdgfm_normal_velocity_defect(velocity, [stencils[index]], dt) ≈
            dt * abs(velocity[2].interior[faces[index]]) / h[index]
    end
    @test CMD._cmdgfm_normal_velocity_defect(velocity, stencils, dt) ≈ dt * 2.0 / h[1]
    @test iszero(CMD._cmdgfm_normal_velocity_defect(velocity, stencils[1:0], dt))

    assembled = (;
        pressure_indices = [1, 2], mixed_faces = faces,
        pressure_operator = (; mesh, volumes = [0.25, 0.5]),
    )
    residual = [4.0, -1.0, 3.0, -2.0]
    defects = CMD._cmdgfm_residual_defects(residual, assembled, dt)
    @test defects.pressure ≈ dt^2 * 16.0
    @test defects.jump ≈ dt^2 * max(3.0 / h[1], 2.0 / h[2])
    empty_rows = merge(assembled, (; mixed_faces = Int[]))
    @test iszero(CMD._cmdgfm_residual_defects(residual[1:2], empty_rows, dt).jump)

    # Rescale the same AMR state and algebraic rows, independently of any
    # Krylov stopping differences. Pressure rows carry L²/T², jump rows L/T².
    reference_normal = CMD._cmdgfm_normal_velocity_defect(velocity, stencils, dt)
    for (length_unit, time_unit) in ((2.0^-20, 2.0^-10), (4.0, 0.5))
        scaled_mesh = BlockTreeMesh(
            base_mesh = (8, 4), block_size = (4, 4),
            extent = (4.0length_unit, length_unit),
            max_level = 1, periodic = (false, false),
            refine = (key, origin, extent) -> key.level == 0 && key.i == 0,
        )
        scaled_velocity = (FaceVelocity(scaled_mesh), FaceVelocity(scaled_mesh))
        scaled_velocity[2].interior[faces] .= velocity[2].interior[faces] .* (length_unit / time_unit)
        scaled_dt = dt * time_unit
        @test CMD._cmdgfm_normal_velocity_defect(scaled_velocity, stencils, scaled_dt) ≈
            reference_normal rtol = 1.0e-8
        scaled_assembled = merge(
            assembled, (;
                pressure_operator = (;
                    mesh = scaled_mesh, volumes = assembled.pressure_operator.volumes .* length_unit^2,
                ),
            )
        )
        scaled_residual = [
            residual[1:2] .* (length_unit^2 / time_unit^2);
            residual[3:4] .* (length_unit / time_unit^2);
        ]
        scaled_defects = CMD._cmdgfm_residual_defects(scaled_residual, scaled_assembled, scaled_dt)
        @test scaled_defects.pressure ≈ defects.pressure rtol = 1.0e-8
        @test scaled_defects.jump ≈ defects.jump rtol = 1.0e-8
    end
end

@testset "CMDGFM statistics measure a known normal jump" begin
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, true))
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.45, y - 0.47) - 0.26; lipschitz = 1,
    )
    jump = (0.37, -0.21)
    velocity = _cmdgfm_test_velocity(mesh; jump)
    solver = CMDGFMProjectionSolver(interface; density = (10.0, 1.0))
    work = CMDGFMProjectionWorkspace(solver)
    dt = 0.02
    stats = project!(
        velocity, CellField(mesh), solver, dt, work;
        tolerance = 10.0, max_iterations = 20, warm_start = false,
    )
    @test stats.converged
    @test iszero(stats.iterations)
    @test !isempty(work.stencils)
    expected = maximum(work.stencils) do stencil
        face = mesh.faces[stencil.face]
        h = min(
            minimum(mesh.blocks[face.minus.block].spacing),
            minimum(mesh.blocks[face.plus.block].spacing),
        )
        transverse = isempty(stencil.transverse_faces) ? 0.0 :
            stencil.ratio * jump[3 - face.axis]
        dt * abs(stencil.beta * (jump[face.axis] + transverse)) / h
    end
    @test expected > 0
    @test stats.normal_velocity_defect ≈ expected rtol = 2.0e-13 atol = 2.0e-13
end

function _unit_scaled_two_velocity_projection(
        method, length_unit, time_unit, density_unit;
        tolerance, max_iterations, preconditioner, warm_start = false
    )
    mesh = BlockTreeMesh(
        block_size = (6, 6), extent = (length_unit, length_unit),
        periodic = (true, true),
    )
    interface = InterfaceField(
        mesh,
        (x, y) -> hypot(x - 0.45length_unit, y - 0.47length_unit) - 0.26length_unit;
        lipschitz = 1,
    )
    speed = length_unit / time_unit
    liquid = FaceVelocity(
        mesh, (x, y, time, axis) -> speed * (
            axis == 1 ? sinpi(2y / length_unit) + 0.1cospi(2x / length_unit) :
                -0.2sinpi(2x / length_unit)
        )
    )
    gas = FaceVelocity(
        mesh, (x, y, time, axis) -> speed * (
            axis == 1 ? 0.3cospi(2y / length_unit) : sinpi(2x / length_unit)
        )
    )
    density = density_unit .* (10.0, 1.0)
    pressure = CellField(
        mesh, (x, y) -> density_unit * speed^2 * (
            0.3sinpi(2x / length_unit) - 0.2cospi(2y / length_unit)
        )
    )
    velocity = (liquid, gas)
    stats = if method == :cmdgfm
        solver = CMDGFMProjectionSolver(interface; density, preconditioner)
        work = CMDGFMProjectionWorkspace(solver)
        fill_halo!(pressure)
        gradients = (FaceVelocity(mesh), FaceVelocity(mesh))
        cmdgfm_gradient!(gradients, pressure, solver, work)
        CMD._projection_correct_phase_velocities!(
            velocity, gradients, 0.125time_unit, eltype(pressure),
        )
        project!(
            velocity, pressure, solver, 0.125time_unit, work;
            tolerance, max_iterations, warm_start,
        )
    else
        solver = MimeticProjectionSolver(interface; density, variant = method, preconditioner)
        work = MimeticProjectionWorkspace(solver)
        project!(
            velocity, pressure, solver, 0.125time_unit, work;
            tolerance, max_iterations,
        )
    end
    return (; stats, velocity, pressure, speed, pressure_unit = density_unit * speed^2)
end

@testset "Two-velocity projection convergence is independent of physical units" begin
    # Power-of-two changes preserve the dimensionless mesh and data exactly,
    # while exercising micrometre-sized cells and independent time/mass units.
    units = ((1.0, 1.0, 1.0), (2.0^-20, 2.0^-10, 2.0), (4.0, 0.5, 8.0))
    methods = (
        (:cmdgfm, false, :block), (:cmdgfm, true, :block),
        (:cmdgfm, false, :ilu), (:cmdgfm, false, nothing),
        (:joint_face, false, :cholesky), (:joint_face, false, nothing),
        (:joint_smoothed, false, :cholesky),
    )
    for (method, warm_start, preconditioner) in methods,
            (tolerance, max_iterations) in ((1.0e-6, 2000), (1.0e-12, 1))
        runs = [
            _unit_scaled_two_velocity_projection(
                method, length_unit, time_unit, density_unit;
                tolerance, max_iterations, preconditioner, warm_start,
            ) for (length_unit, time_unit, density_unit) in units
        ]
        first_run = first(runs)
        reference = first_run.stats
        @test reference.jump_unknowns > 0
        @test reference.iterations > 0
        if max_iterations == 1
            @test !reference.converged
        else
            @test reference.converged
        end
        for run in runs[2:end]
            result = run.stats
            # Unpreconditioned Krylov iterations can change under roundoff
            # in an equivalent matrix, even though both solves meet the same
            # stopping criterion. The default and ILU/Cholesky configurations
            # are required to retain identical iteration counts.
            if !isnothing(preconditioner)
                @test result.iterations == reference.iterations
            end
            @test result.converged == reference.converged
            defect_atol = reference.converged ? tolerance : 1.0e-12
            @test result.mixture_volume_defect ≈ reference.mixture_volume_defect rtol = 1.0e-8 atol = defect_atol
            @test result.normal_velocity_defect ≈ reference.normal_velocity_defect rtol = 1.0e-8 atol = defect_atol
            if method != :cmdgfm
                @test result.tangential_defect ≈ reference.tangential_defect rtol = 1.0e-8 atol = defect_atol
            end
            if result.converged
                @test result.mixture_volume_defect <= tolerance
                @test result.normal_velocity_defect <= tolerance
                if method != :cmdgfm
                    @test result.tangential_defect <= tolerance
                end
            end
            for phase in 1:2
                reference_values = first_run.velocity[phase].interior
                values = run.velocity[phase].interior
                active = findall(isfinite, reference_values)
                @test findall(isfinite, values) == active
                @test values[active] ./ run.speed ≈
                    reference_values[active] ./ first_run.speed rtol = 2.0e-5 atol = 2.0e-8
            end
            for block in eachindex(run.pressure.blocks)
                @test interior(run.pressure, block) ./ run.pressure_unit ≈
                    interior(first_run.pressure, block) ./ first_run.pressure_unit rtol = 2.0e-5 atol = 2.0e-8
            end
        end
    end
end
