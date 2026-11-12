using Test, TwoPhaseAMR

@testset "Conservative surface-tension method" begin
    @test CGFM() isa AbstractSurfaceTensionMethod
    @test isconcretetype(CGFM)
    @test fieldcount(CGFM) == 0
    @test SurfaceTension().method isa CGFM
    @test_throws MethodError CGFM(false)
    @test_throws MethodError CGFM(; conservative = false)

    mesh = BlockTreeMesh(block_size = (4, 4))
    interface = InterfaceField(mesh, (x, y) -> x - 0.37)
    geometry = GeometryField(mesh, (x, y) -> y - 0.12)
    @test_throws MethodError PoissonOperator(interface; formulation = CGFM())
    @test_throws MethodError PoissonOperator(
        interface, geometry; formulation = CGFM(),
    )
    operator = PoissonOperator(interface)
    @test_throws ArgumentError surface_tension!(
        FaceVelocity(mesh), interface, operator, EllipticWorkspace(operator);
        sigma = 1.0, formulation = nothing,
    )
    invalid = try
        surface_tension!(
            FaceVelocity(mesh), interface, operator, EllipticWorkspace(operator);
            sigma = -1.0, formulation = nothing,
        )
        nothing
    catch err
        err
    end
    @test invalid isa ArgumentError
    @test sprint(showerror, invalid) ==
        "ArgumentError: formulation must be CSF() or CGFM()"
end

@testset "Unresolved surface-tension curvature" begin
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (false, false))
    film = InterfaceField(mesh, (x, y) -> 1.0)
    values = interior(film.fraction, 1)
    fill!(values, 0)
    values[1, 2:6] .= (0.0, 0.167, 0.961, 0.994, 1.0)
    reconstruct!(film; force = true)
    op = PoissonOperator(film)
    count = Ref(0)
    force = FaceVelocity(mesh)
    for formulation in (CGFM(), CSF())
        surface_tension!(
            force, film, op, EllipticWorkspace(op);
            sigma = 1.0, formulation, unresolved_curvature_faces = count,
        )
        missing = findall(mesh.faces) do face
            cm = TwoPhaseAMR._conditional_fraction(film, face.minus)
            cp = TwoPhaseAMR._conditional_fraction(film, face.plus)
            !iszero(TwoPhaseAMR._surface_tension_jump(formulation, cm, cp)) &&
                isnothing(TwoPhaseAMR._face_curvature(film, face))
        end
        @test count[] == length(missing) > 0
        @test all(iszero, op.gradient_operator.pressure_jump[missing])
        @test all(iszero, force.interior[missing])
    end
    @test_throws ArgumentError surface_tension!(
        force, film, op, EllipticWorkspace(op); sigma = Inf,
    )

    flow = FlowSimulation(
        mesh, TwoPhase(
            film; density = (1.0, 1.0), dynamic_viscosity = (0.0, 0.0),
            surface_tension = 1.0,
        );
        dt = 1.0e-5,
        diagnostics = (
            Diagnostic(:unresolved, state -> state.workspace.unresolved_curvature_faces[]),
        ),
    )
    first_result = simulate!(flow, 1.0e-5)
    first_count = last(diagnostic_history(flow, :unresolved).values)
    @test first_count > 0
    @test first_result.unresolved_curvature_faces == first_count
    @test first_result.total_unresolved_curvature_faces == first_count
    @test occursin(
        "unresolved curvature faces this run", sprint(
            io -> print_simulation_summary(io, flow, first_result),
        )
    )
    second_result = simulate!(flow, 2.0e-5)
    second_count = last(diagnostic_history(flow, :unresolved).values)
    @test second_result.unresolved_curvature_faces == second_count
    @test second_result.total_unresolved_curvature_faces == first_count + second_count

    droplet = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.3,
    )
    op = PoissonOperator(droplet)
    surface_tension!(
        force, droplet, op, EllipticWorkspace(op);
        sigma = 1.0, unresolved_curvature_faces = count,
    )
    @test count[] == 0
    for (i, face) in pairs(mesh.faces)
        cm = TwoPhaseAMR._conditional_fraction(droplet, face.minus)
        cp = TwoPhaseAMR._conditional_fraction(droplet, face.plus)
        jump = (cp > 0.5) - (cm > 0.5)
        expected = iszero(jump) ? 0.0 :
            TwoPhaseAMR._face_curvature(droplet, face) * jump * face.area
        @test op.gradient_operator.pressure_jump[i] == expected
    end
end

@testset "Variable-density balanced CSF and CGFM projection" begin
    for kind in 1:3,dr in (1.0, 1000.0),formulation in (CSF(), CGFM())
        mesh = BlockTreeMesh(
            block_size = (8, 8), max_level = 2,
            refine = (k, o, e) -> kind == 2 && (k.level == 0 || (k.level == 1 && k.i == 0 && k.j == 0))
        )
        f = InterfaceField(mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.2)
        if kind == 3
            amr = AMRWorkspace(mesh, f)
            adapt!(amr, [:refine]; step = 1, force = true)
        end
        op = PoissonOperator(f; density = (dr, 1.0))
        @test reinterpret(UInt64, op.face_coefficient) ==
            reinterpret(UInt64, inv.(mixture_density(f, (dr, 1.0))))
        u = FaceVelocity(mesh)
        surface_tension!(u, f, op, EllipticWorkspace(op); sigma = 1.0, formulation, curvature = 5.0)
        expected_jumps = map(mesh.faces) do face
            cm = TwoPhaseAMR._conditional_fraction(f, face.minus)
            cp = TwoPhaseAMR._conditional_fraction(f, face.plus)
            jump = formulation isa CSF ? cp - cm : (cp > 0.5) - (cm > 0.5)
            jump == 0 ? 0.0 : 1.0 * 5.0 * jump * face.area
        end
        @test reinterpret(UInt64, op.gradient_operator.pressure_jump) ==
            reinterpret(UInt64, expected_jumps)
        if formulation isa CGFM
            default = FaceVelocity(mesh)
            surface_tension!(
                default, f, op, EllipticWorkspace(op); sigma = 1.0,
                curvature = 5.0,
            )
            @test reinterpret(UInt64, default.interior) ==
                reinterpret(UInt64, u.interior)
        end
        potential = copy(f.fraction)
        for ref in TwoPhaseAMR._cell_references(mesh)
            c = TwoPhaseAMR._get(f.fraction, ref)
            TwoPhaseAMR._set!(potential, 5 * (formulation isa CSF ? c : c > 0.5), ref)
        end
        expected = FaceVelocity(mesh)
        weighted_gradient!(expected, potential, op, EllipticWorkspace(op))
        @test u.interior ≈ expected.interior atol = 5.0e-13
        u.interior .*= 0.01
        solver = EllipticSolver(op; preconditioner = :tree_geometric)
        pressure = CellField(mesh)
        project!(u, pressure, solver, 0.01, ProjectionWorkspace(solver); tolerance = 1.0e-11)
        @test maximum(abs, u.interior) < 1.0e-9
        @test all(>(0), mixture_density(f, (dr, 1.0)))
    end
end
@testset "CMDGFM cut composite constant-curvature drop force map (#76)" begin
    # The audit fixture has a refined right half and a sloped solid clear of
    # the drop. The composite inverse metric nevertheless couples some drop
    # jumps to partially open solid faces.
    mesh = BlockTreeMesh(
        base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0),
        periodic = (false, false), max_level = 1,
        refine = (key, origin, extent) -> key.level == 0 && key.i >= 2,
    )
    geometry = GeometryField(
        mesh, (x, y) -> y - 0.9 - 0.1x; lipschitz = hypot(1, 0.1),
    )
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 4.3, y - 4.1) - 2.23;
        geometry, contact_angle = pi / 2, lipschitz = 1,
    )
    density, sigma, curvature, dt = (1000.0, 1.0), 0.7, inv(2.23), 0.01
    @test any(a -> 0 < a < 1, geometry.aperture)
    # The force substep has no prescribed-curvature argument. Giving every
    # curvature sample the same value makes its interpolation constant too.
    for block in eachindex(mesh.blocks)
        fill!(interior(interface.curvature, block), curvature)
    end

    for formulation in (CSF(), CGFM())
        flow = FlowSimulation(
            mesh,
            TwoPhase(
                interface; density, dynamic_viscosity = (0.0, 0.0),
                surface_tension = SurfaceTension(sigma; method = formulation),
            );
            geometry, boundary = FreeSlip(),
            velocity_model = TwoVelocityModel(
                pressure_projection = CMDGFMProjection(),
            ),
        )
        state = TwoPhaseAMR._two_velocity_flow_workspace!(
            flow.workspace.two_phase, flow.system.two_phase,
        )
        operator = TwoPhaseAMR._projection_pressure_operator(state.projection_solver)
        expected = FaceVelocity(mesh)
        surface_tension!(
            expected, interface, operator, EllipticWorkspace(operator);
            sigma, formulation, curvature,
        )
        @test maximum(abs, expected.interior) > 0
        potential = CellField(mesh)
        for ref in TwoPhaseAMR._cell_references(mesh)
            c = TwoPhaseAMR._conditional_fraction(interface, ref)
            value = formulation isa CSF ? c : (c > 0.5)
            TwoPhaseAMR._set!(potential, sigma * curvature * value, ref)
        end
        gradient = FaceVelocity(mesh)
        TwoPhaseAMR._cut_cell_weighted_gradient!(
            gradient, potential, operator, EllipticWorkspace(operator), geometry,
        )
        # Both paths apply the same pressure gradient, differing only in the
        # floating-point order of scalar multiplications.
        @test maximum(abs, expected.interior .- gradient.interior) <=
            2048eps() * max(1.0, maximum(abs, expected.interior))

        # Ensure this fixture actually distinguishes the old one-velocity map.
        if formulation isa CSF
            old_operator = PoissonOperator(interface, geometry; density)
            old_force = FaceVelocity(mesh)
            surface_tension!(
                old_force, interface, old_operator, EllipticWorkspace(old_operator);
                sigma, formulation, curvature,
            )
            @test maximum(abs, old_force.interior .- expected.interior) > 1.0e-10
        end

        TwoPhaseAMR._phase_force_substep!(
            flow.velocity, flow.workspace, dt, flow.system.two_phase, operator,
        )
        # Interpolating a constant introduces only round-off; this bound is
        # scaled to the force and well below the cut-face mapping discrepancy.
        scale = max(1.0, maximum(abs, expected.interior))
        @test maximum(
            abs, flow.workspace.two_phase.acceleration.interior .-
                expected.interior
        ) <= 2048eps() * scale
        for phase in flow.velocity
            active = findall(isfinite, phase.interior)
            @test !isempty(active)
            @test maximum(
                abs, phase.interior[active] .-
                    dt .* expected.interior[active]
            ) <= 2048eps() * scale
        end

        pressure = CellField(mesh)
        tolerance = 1.0e-11
        stats = TwoPhaseAMR._project_two_velocity!(
            flow.velocity, pressure, flow.system, flow.workspace, state, dt;
            tolerance, max_iterations = 4000,
        )
        @test stats.converged
        @test stats.mixture_volume_defect <= tolerance
        @test stats.normal_velocity_defect <= tolerance
        @test state.projection_workspace.pressure_cache.operator.face_coefficient ≈
            operator.face_coefficient rtol = 128eps()
        # The projection removes gradients of its own map, so the constant
        # curvature is balanced to the iterative solver's error.
        force_scale = dt * maximum(abs, expected.interior)
        for phase in flow.velocity
            active = filter(isfinite, phase.interior)
            @test maximum(abs, active) <= 1.0e-9 * force_scale
        end
        if formulation isa CSF
            # Counterfactual of issue #76: the same jump through the
            # one-velocity coefficient a^2/rho leaves a velocity at the cut
            # face (aperture 0.8) that the composite metric couples to the
            # drop's jumps, although the solid does not touch the drop.
            phases = (FaceVelocity(mesh), FaceVelocity(mesh))
            for phase in phases
                phase.interior .= dt .* old_force.interior
            end
            TwoPhaseAMR._mark_inactive_phase_velocities!(phases, interface)
            former = project!(
                phases, CellField(mesh), state.projection_solver, dt,
                CMDGFMProjectionWorkspace(state.projection_solver);
                tolerance = 1.0e-12, max_iterations = 4000,
            )
            @test former.converged
            worst = 0.0
            for phase in phases, index in eachindex(phase.interior)
                value = phase.interior[index]
                isnan(value) || (worst = max(worst, abs(value)))
            end
            @test worst > 1.0e-3 * dt * maximum(abs, old_force.interior)
        end
    end
end

@testset "Surface tension precedes gravity for each velocity model" begin
    for velocity_model in (
                OneVelocityModel(),
                TwoVelocityModel(pressure_projection = CMDGFMProjection()),
            ), formulation in (CSF(), CGFM())
        mesh = BlockTreeMesh(block_size = (16, 16), periodic = (false, false))
        interface = InterfaceField(
            mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.27;
            lipschitz = 1,
        )
        flow = FlowSimulation(
            mesh,
            TwoPhase(
                interface; density = (9.0, 1.0), dynamic_viscosity = (0.0, 0.0),
                surface_tension = SurfaceTension(0.7; method = formulation),
                gravity = (0.7, -3.1),
            );
            velocity_model, boundary = FreeSlip(),
            velocity = (x, y, time, axis) -> 0.317 + sin(3x + 2y + axis),
        )
        model = flow.system.two_phase
        workspace = flow.workspace
        is_shared = flow.velocity isa FaceVelocity
        phases = is_shared ? (flow.velocity,) : flow.velocity
        initial = map(phase -> copy(phase.interior), phases)
        initial_boundary = map(phase -> copy(phase.boundary), phases)
        operator = if is_shared
            TwoPhaseAMR._one_velocity_pressure_solver!(flow.system, workspace).operator
        else
            state = TwoPhaseAMR._two_velocity_flow_workspace!(workspace.two_phase, model)
            TwoPhaseAMR._two_velocity_pressure_operator!(flow.system, workspace, state, model)
        end
        tension = FaceVelocity(mesh)
        gravity = FaceVelocity(mesh)
        surface_tension!(
            tension, interface, operator, workspace.projection.elliptic;
            sigma = model.sigma, formulation = model.formulation,
        )
        gravity!(
            gravity, interface, operator, workspace.projection.elliptic;
            density = model.density, gravity = model.gravity,
        )
        @test any(!iszero, tension.interior)
        @test any(!iszero, gravity.interior)
        dt = 0.0171
        if is_shared
            TwoPhaseAMR._phase_force_substep!(flow.velocity, flow.system, workspace, dt, model)
        else
            TwoPhaseAMR._phase_force_substep!(flow.velocity, workspace, dt, model, operator)
        end
        distinguishes_order = false
        for phase in eachindex(phases)
            @test isequal(phases[phase].boundary, initial_boundary[phase])
            for index in eachindex(initial[phase])
                old = initial[phase][index]
                if isnan(old)
                    @test isnan(phases[phase].interior[index])
                    continue
                end
                expected = (old + dt * tension.interior[index]) + dt * gravity.interior[index]
                reversed = (old + dt * gravity.interior[index]) + dt * tension.interior[index]
                distinguishes_order |= !isequal(expected, reversed)
                @test isequal(phases[phase].interior[index], expected)
            end
        end
        @test distinguishes_order
    end
end

@testset "Surface-tension strength is linear in sigma" begin
    mesh = BlockTreeMesh(
        block_size = (64, 64), origin = (-0.5, 0.0),
        extent = (1.0, 1.0), periodic = (false, false),
    )
    theta = deg2rad(55)
    radius = 0.3
    interface = InterfaceField(
        mesh,
        (x, y) -> hypot(x, y + radius * cos(theta)) - radius;
        lipschitz = 1,
        contact_angle = (nothing, nothing, theta, nothing),
    )
    operator = PoissonOperator(interface; density = (10.0, 1.0))
    first = FaceVelocity(mesh)
    second = FaceVelocity(mesh)
    surface_tension!(
        first, interface, operator, EllipticWorkspace(operator);
        sigma = 0.25,
    )
    surface_tension!(
        second, interface, operator, EllipticWorkspace(operator);
        sigma = 0.75,
    )
    @test second.interior ≈ 3 .* first.interior rtol = 256eps()
end

@testset "Curvature extension agrees with exhaustive search" begin
    for periodic in ((false, false), (true, true))
        mesh = BlockTreeMesh(
            base_mesh = (16, 16), block_size = (4, 4), max_level = 1,
            periodic = periodic, refine = (key, origin, extent) -> key.level == 0 && key.i == 0,
        )
        field = InterfaceField(mesh, (x, y) -> hypot(x - 0.15, y - 0.5) - 0.24)
        samples = [
            (
                cellcenter(mesh, r.block, r.i, r.j),
                TwoPhaseAMR._get(field.curvature, r),
                TwoPhaseAMR._conditional_fraction(field, r) *
                    (1 - TwoPhaseAMR._conditional_fraction(field, r)),
                maximum(mesh.blocks[r.block].spacing),
            )
                for r in TwoPhaseAMR._cell_references(mesh)
                if 0 < TwoPhaseAMR._conditional_fraction(field, r) < 1 &&
                isfinite(TwoPhaseAMR._get(field.curvature, r))
        ]
        checked = 0
        for face in mesh.faces
            all(
                r -> TwoPhaseAMR._conditional_fraction(field, r) in (0, 1),
                (face.minus, face.plus)
            ) || continue
            radius = 3max(
                maximum(mesh.blocks[face.minus.block].spacing),
                maximum(mesh.blocks[face.plus.block].spacing)
            )
            total = weight = 0.0
            for (point, kappa, w, h) in samples
                dx = TwoPhaseAMR._wrapped_delta(point[1] - face.center[1], mesh.extent[1], periodic[1])
                dy = TwoPhaseAMR._wrapped_delta(point[2] - face.center[2], mesh.extent[2], periodic[2])
                distance = hypot(dx, dy)
                distance <= radius + h || continue
                gw = w / max(distance^2, (h / 4)^2)
                total += gw * kappa
                weight += gw
            end
            expected = weight > 0 ? total / weight : nothing
            @test isequal(TwoPhaseAMR._face_curvature(field, face), expected)
            checked += weight > 0
        end
        @test checked > 0
    end
end
@testset "Hydrostatic balance" begin
    for kind in 1:3, gravity in ((0.0, -1.0), (0.6, -0.8))
        mesh = BlockTreeMesh(
            block_size = (8, 8), max_level = 2, periodic = (false, false),
            refine = (k, o, e) -> kind == 2 && (k.level == 0 || (k.level == 1 && k.i == 0 && k.j == 0))
        )
        levelset = gravity == (0.0, -1.0) ?
            ((x, y) -> y - 0.43) :
            (
                (x, y) -> gravity[1] * (x - 0.5) +
                gravity[2] * (y - 0.5)
            )
        f = InterfaceField(mesh, levelset; lipschitz = 1)
        if kind == 3
            amr = AMRWorkspace(mesh, f);adapt!(amr, [:refine]; step = 1, force = true)
        end
        op = PoissonOperator(f; density = (1000.0, 1.0))
        u = FaceVelocity(mesh)
        gravity!(
            u, f, op, EllipticWorkspace(op);
            density = (1000.0, 1.0), gravity
        )
        u.interior .*= 0.01
        solver = EllipticSolver(op; preconditioner = :tree_geometric)
        p = CellField(mesh)
        project!(
            u, p, solver, 0.01, ProjectionWorkspace(solver);
            tolerance = 512eps(),
        )
        @test maximum(abs, u.interior) <= 2048eps()
    end
end

@testset "Gravity uses the open-face phase mixture" begin
    mesh = BlockTreeMesh(block_size = (16, 16), periodic = (false, false))
    geometry = GeometryField(
        mesh, (x, y) -> y - 0.22 - 0.36x; lipschitz = hypot(1, 0.36),
    )
    interface = InterfaceField(
        mesh, (x, y) -> y - 0.43;
        geometry, embedded_contact_angle = nothing, lipschitz = 1,
    )
    gravity = (0.6, -0.8)
    liquid, gas = TwoPhaseAMR._current_phase_face_apertures!(interface)
    partial = findall(a -> 0 < a < 1, geometry.aperture)
    @test any(i -> liquid[i] > 0 && iszero(gas[i]), partial)
    @test any(i -> gas[i] > 0 && iszero(liquid[i]), partial)
    @test any(i -> liquid[i] > 0 && gas[i] > 0, partial)
    @test any(iszero, geometry.aperture)

    for density in ((1000.0, 1.0), (1.0, 1000.0), (3.0, 3.0))
        operator = PoissonOperator(interface, geometry; density)
        work = EllipticWorkspace(operator)
        force = FaceVelocity(mesh)
        fill!(force.interior, NaN)
        fill!(force.boundary, NaN)
        gravity!(force, interface, operator, work; density, gravity)
        jumps = operator.gradient_operator.pressure_jump
        nx, ny = mesh.block_size
        for (i, face) in pairs(mesh.faces)
            open = geometry.aperture[i]
            if iszero(open)
                @test iszero(jumps[i])
                @test iszero(force.interior[i])
            else
                minus = TwoPhaseAMR._cell_index(face.minus, nx, ny)
                plus = TwoPhaseAMR._cell_index(face.plus, nx, ny)
                rho = (density[1] * liquid[i] + density[2] * gas[i]) / open
                expected = rho * face.area * (work.input[plus] - work.input[minus])
                @test jumps[i] ≈ expected rtol = 256eps() atol = 64eps()
                @test isfinite(force.interior[i])
            end
        end
        @test all(iszero, force.boundary)
    end
end

# The returned dimensionless speed also makes it possible to measure this
# identical complete-step fixture with another velocity model.
function _gravity_wall_rest_step(velocity_model = OneVelocityModel())
    mesh = BlockTreeMesh(
        base_mesh = (16, 16), block_size = (4, 4), max_level = 1,
        extent = (8.0, 8.0), periodic = (false, false),
        refine = (key, origin, extent) -> key.level == 0 && key.i >= 2,
    )
    geometry = GeometryField(
        mesh, (x, y) -> y - 0.9 - 0.1x; lipschitz = hypot(1, 0.1),
    )
    interface = InterfaceField(
        mesh, (x, y) -> y - 4.6; geometry, contact_angle = pi / 2,
        lipschitz = 1,
    )
    density = (1000.0, 1.0)
    gravity = (0.0, -1.0)
    dt = 0.01
    flow = FlowSimulation(
        mesh, TwoPhase(
            interface; density, dynamic_viscosity = (0.0, 0.0), gravity,
        ); velocity_model,
    )
    stats = step!(
        flow.velocity, flow.pressure, flow.system, dt, flow.workspace;
        tolerance = 1.0e-12, max_cycles = 1000,
    )
    velocity = flow.velocity
    maximum_speed = if velocity isa FaceVelocity
        maximum(abs, velocity.interior)
    else
        maximum(1:2) do phase
            name = phase == 1 ? :liquid : :gas
            maximum(
                abs(velocity[phase].interior[i])
                    for i in eachindex(mesh.faces)
                    if TwoPhaseAMR._phase_face_is_active(
                        interface, i, mesh.faces[i], name,
                    )
            )
        end
    end
    relative_speed = maximum_speed / (dt * hypot(gravity...))
    return (; flow, stats, relative_speed)
end

@testset "Hydrostatic liquid at a sloped solid on a composite mesh" begin
    result = _gravity_wall_rest_step()
    mesh = result.flow.system.mesh
    interface = result.flow.system.two_phase.interface
    geometry = interface.geometry
    @test Set(block.key.level for block in mesh.blocks) == Set((0, 1))
    @test any(mesh.faces) do face
        mesh.blocks[face.minus.block].key.level !=
            mesh.blocks[face.plus.block].key.level
    end
    liquid, gas = TwoPhaseAMR._current_phase_face_apertures!(interface)
    @test any(eachindex(mesh.faces)) do i
        0 < geometry.aperture[i] < 1 && liquid[i] > 0 && iszero(gas[i])
    end
    @test result.stats.converged
    # The pressure solve tolerance is much tighter than the requested
    # 1e-8 bound on the residual velocity relative to one gravity impulse.
    @test result.relative_speed <= 1.0e-8
end
