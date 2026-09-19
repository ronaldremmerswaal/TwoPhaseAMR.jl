using Test, TwoPhaseAMR
const TV = TwoPhaseAMR
phase_cell_fraction_allocations(interface, geometry, ref) =
    @allocated TV._phase_cell_fractions(interface, geometry, ref)
phase_staggered_fraction_allocations(interface, geometry, fraction, index) =
    @allocated TV._phase_staggered_fractions(
    interface, geometry, fraction, index,
)
phase_face_aperture_allocations(interface, geometry, index) =
    @allocated TV._phase_face_apertures(interface, geometry, index)

@testset "TwoPhaseVelocity tuple API" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), periodic = (false, false),
    )
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.55) - 0.24;
        lipschitz = 1,
    )
    liquid = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? 0.15 : -0.1,
    )
    gas = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? -0.3 : 0.2,
    )
    velocity = (liquid, gas)
    @test velocity isa TwoPhaseVelocity
    enforce_no_permeation!(velocity)

    workspace = WYAdvectionWorkspace(interface, velocity)
    @test workspace.interface_method isa FrozenInterfaceFluxes
    @test WYAdvectionWorkspace(
        interface, velocity; interface_method = FrozenInterfaceFluxes(),
    ).interface_method isa FrozenInterfaceFluxes
    @test_throws ArgumentError WYAdvectionWorkspace(
        interface, velocity; interface_method = :projected,
    )
    projected_workspace = WYAdvectionWorkspace(
        interface, velocity; interface_method = ProjectedLiquidVelocity(),
    )
    projected = TV._project_liquid_velocity!(projected_workspace, velocity)
    expected = min(
        stable_timestep(projected, projected_workspace; cfl = 0.3),
        stable_timestep(gas, projected_workspace; cfl = 0.3),
    )
    @test stable_timestep(
        velocity, projected_workspace; cfl = 0.3,
    ) == expected

    set_velocities!(
        velocity,
        (
            (x, y, time, axis) -> axis == 1 ? x + time : y,
            (x, y, time, axis) -> axis == 1 ? -x : time - y,
        );
        time = 0.25,
    )
    @test any(!iszero, liquid.interior)
    @test any(value -> value < 0, gas.interior)
    enforce_no_permeation!(velocity)
    @test all(iszero, liquid.boundary)
    @test all(iszero, gas.boundary)

    other_mesh = BlockTreeMesh(block_size = (8, 8))
    wrong_mesh = (liquid, FaceVelocity(other_mesh))
    @test_throws ArgumentError WYAdvectionWorkspace(interface, wrong_mesh)
    wrong_type = (liquid, FaceVelocity(mesh, Float32))
    @test_throws ArgumentError WYAdvectionWorkspace(interface, wrong_type)

    floored = InterfaceField(mesh; fraction_floor = 1.0e-6)
    floored_values = interior(floored.fraction, 1)
    floored_values[1, 1] = floored.fraction_floor / 2
    floored_values[2, 1] = 1 - floored.fraction_floor / 2
    reconstruct!(floored)
    @test !TV._phase_cell_active(
        floored, CellRef(1, 1, 1), Val(:liquid),
    )
    @test !TV._phase_cell_active(
        floored, CellRef(1, 2, 1), Val(:gas),
    )

    # Equal-valued but distinct phase fields still exercise the two-velocity
    # frozen-interface equations. Their phase fluxes recover the common
    # one-velocity flux for this planar translation.
    direct_interface = InterfaceField(
        mesh, (x, y) -> x - 0.43; lipschitz = 1,
    )
    tuple_interface = copy(direct_interface)
    direct_scalar = PhaseScalar(direct_interface; liquid = 3, gas = 7)
    tuple_scalar = PhaseScalar(tuple_interface; liquid = 3, gas = 7)
    common = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? 0.1 : 0.0,
    )
    enforce_no_permeation!(common)
    common_copy = copy(common)
    direct_work = WYAdvectionWorkspace(direct_interface)
    tuple_work = WYAdvectionWorkspace(
        tuple_interface, (common, common_copy);
        interface_method = FrozenInterfaceFluxes(),
    )
    dt = stable_timestep(common, direct_work; cfl = 0.2)
    advect!(
        direct_interface, common, dt, direct_work;
        scalars = (direct_scalar,),
    )
    advect!(
        tuple_interface, (common, common_copy), dt, tuple_work;
        scalars = (tuple_scalar,),
    )
    for block in eachindex(mesh.blocks)
        @test interior(tuple_interface, block) ==
            interior(direct_interface, block)
        @test interior(tuple_scalar.liquid, block) ==
            interior(direct_scalar.liquid, block)
        @test interior(tuple_scalar.gas, block) ==
            interior(direct_scalar.gas, block)
    end
    @test tuple_work.liquid_flux == direct_work.liquid_flux
    @test tuple_work.gas_flux == direct_work.gas_flux
    @test tuple_work.open_flux == direct_work.open_flux
    diagnostics = phase_flux_diagnostics(tuple_work)
    @test diagnostics.maximum_partition_defect <= eps()
    @test diagnostics.absolute_partition_defect <=
        length(mesh.faces) * eps()
    @test diagnostics.gas_reset_discrepancy == 0

    unequal = (common, copy(common))
    unequal[2].interior[1] += 0.01
    @test_throws ArgumentError advect!(
        copy(direct_interface), unequal, dt,
        WYAdvectionWorkspace(direct_interface, unequal),
    )
end

@testset "Projected liquid-velocity extension" begin
    mesh = BlockTreeMesh(
        block_size = (12, 12), periodic = (false, false),
    )
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.25;
        lipschitz = 1,
    )
    velocity = (
        FaceVelocity(
            mesh,
            (x, y, time, axis) -> axis == 1 ? x - 0.5 : y - 0.5,
        ),
        FaceVelocity(mesh),
    )
    enforce_no_permeation!(velocity)
    work = WYAdvectionWorkspace(
        interface, velocity;
        interface_method = ProjectedLiquidVelocity(
            tolerance = 1.0e-10, max_cycles = 500,
        ),
    )
    projected = TV._project_liquid_velocity!(work, velocity)
    diagnostics = phase_flux_diagnostics(work)
    @test diagnostics.projection_initial_divergence > 1
    @test diagnostics.projection_final_divergence <= 1.0e-10
    @test 0 < diagnostics.projection_iterations < 500
    @test all(isfinite, projected.interior)

    # An unequal tangential jump leaves a planar interface stationary, also
    # when the phase is clipped by a fixed solid.
    tangent_mesh = BlockTreeMesh(
        block_size = (8, 8), periodic = (true, false),
    )
    geometry = GeometryField(
        tangent_mesh, (x, y) -> 0.1 - y; lipschitz = 1,
    )
    tangent_interface = InterfaceField(
        tangent_mesh, (x, y) -> y - 0.47; geometry, lipschitz = 1,
    )
    tangent_velocity = (
        FaceVelocity(
            tangent_mesh,
            (x, y, time, axis) -> axis == 1 ? 0.15 : 0.0,
        ),
        FaceVelocity(
            tangent_mesh,
            (x, y, time, axis) -> axis == 1 ? -0.08 : 0.0,
        ),
    )
    enforce_no_permeation!(tangent_velocity)
    tangent_work = WYAdvectionWorkspace(
        tangent_interface, tangent_velocity;
        interface_method = ProjectedLiquidVelocity(),
    )
    constants = PhaseScalar(
        tangent_interface; liquid = 3.0, gas = 7.0,
    )
    tracer = PhaseScalar(
        tangent_interface;
        liquid = (x, y) -> 1 + 0.1sinpi(2x),
        gas = (x, y) -> 2 + 0.2cospi(2x),
    )
    initial = copy(tangent_interface.fraction)
    initial_mass = (
        integrate(tracer, :liquid), integrate(tracer, :gas),
    )
    dt = stable_timestep(tangent_velocity, tangent_work; cfl = 0.2)
    advect!(
        tangent_interface, tangent_velocity, dt, tangent_work;
        scalars = (constants, tracer),
    )
    for block in eachindex(tangent_mesh.blocks)
        @test interior(tangent_interface.fraction, block) ==
            interior(initial, block)
        @test interior(constants.liquid, block) ≈
            fill(3.0, tangent_mesh.block_size)
        @test interior(constants.gas, block) ≈
            fill(7.0, tangent_mesh.block_size)
    end
    @test integrate(tracer, :liquid) ≈ initial_mass[1]
    @test integrate(tracer, :gas) ≈ initial_mass[2]
    @test phase_flux_diagnostics(tangent_work).gas_reset_discrepancy ≈ 0

    # The gas carrier is deliberately compressible here. The physical
    # interface remains fixed, while the auxiliary carrier records the local
    # discrepancy and still transports a constant intrinsic scalar exactly.
    reset_mesh = BlockTreeMesh(
        block_size = (12, 12), periodic = (false, false),
    )
    reset_interface = InterfaceField(
        reset_mesh, (x, y) -> x - 0.5; lipschitz = 1,
    )
    reset_velocity = (
        FaceVelocity(reset_mesh),
        FaceVelocity(
            reset_mesh,
            (x, y, time, axis) -> axis == 1 ? 0.1sinpi(2x) : 0.0,
        ),
    )
    enforce_no_permeation!(reset_velocity)
    reset_scalar = PhaseScalar(
        reset_interface; liquid = 2.0, gas = 5.0,
    )
    reset_work = WYAdvectionWorkspace(
        reset_interface, reset_velocity;
        interface_method = ProjectedLiquidVelocity(),
    )
    reset_destination = (
        FaceVelocity(reset_mesh), FaceVelocity(reset_mesh),
    )
    reset_momentum = MomentumAdvectionWorkspace(
        reset_interface, reset_velocity,
    )
    reset_initial = copy(reset_interface.fraction)
    reset_dt = stable_timestep(
        reset_velocity, reset_work; cfl = 0.2,
    )
    advect!(
        reset_interface, reset_velocity, reset_dt, reset_work;
        scalars = (reset_scalar,),
        momentum = reset_destination, density = (1000.0, 1.0),
        momentum_workspace = reset_momentum,
    )
    for block in eachindex(reset_mesh.blocks)
        @test interior(reset_interface.fraction, block) ==
            interior(reset_initial, block)
        @test interior(reset_scalar.liquid, block) ≈
            fill(2.0, reset_mesh.block_size)
        @test interior(reset_scalar.gas, block) ≈
            fill(5.0, reset_mesh.block_size)
    end
    reset_diagnostics = phase_flux_diagnostics(reset_work)
    @test reset_diagnostics.absolute_gas_reset_discrepancy > 0
    @test abs(reset_diagnostics.gas_reset_discrepancy) <=
        reset_diagnostics.absolute_gas_reset_discrepancy
    @test reset_diagnostics.absolute_gas_momentum_reset_discrepancy_x > 0
    @test reset_diagnostics.absolute_gas_momentum_reset_discrepancy_y == 0

    # The projected correction must use the cut-cell gradient paired with the
    # open-volume divergence. This case exercises partial solid apertures and
    # the zero-pressure condition in phase-empty cells.
    cut_mesh = BlockTreeMesh(
        block_size = (8, 8), periodic = (false, false),
    )
    cut_geometry = GeometryField(
        cut_mesh, (x, y) -> 0.18 + 0.08x - y; lipschitz = 1,
    )
    cut_interface = InterfaceField(
        cut_mesh, (x, y) -> hypot(x - 0.5, y - 0.38) - 0.27;
        geometry = cut_geometry, lipschitz = 1,
    )
    cut_velocity = (
        FaceVelocity(
            cut_mesh,
            (x, y, time, axis) -> axis == 1 ? x - 0.5 : y - 0.38,
        ),
        FaceVelocity(cut_mesh),
    )
    enforce_no_permeation!(cut_velocity)
    cut_work = WYAdvectionWorkspace(
        cut_interface, cut_velocity;
        interface_method = ProjectedLiquidVelocity(
            tolerance = 1.0e-10, max_cycles = 500,
        ),
    )
    TV._project_liquid_velocity!(cut_work, cut_velocity)
    cut_diagnostics = phase_flux_diagnostics(cut_work)
    @test cut_diagnostics.projection_initial_divergence > 1
    @test cut_diagnostics.projection_final_divergence <= 1.0e-10

    # A vanished liquid phase has no auxiliary Poisson unknowns and is a
    # valid limiting state, rather than a failed solve with empty statistics.
    empty_liquid = InterfaceField(
        cut_mesh, (x, y) -> one(x); geometry = cut_geometry,
        lipschitz = 1,
    )
    empty_velocity = (FaceVelocity(cut_mesh), FaceVelocity(cut_mesh))
    empty_velocity[2].interior .= cut_velocity[1].interior
    empty_work = WYAdvectionWorkspace(
        empty_liquid, empty_velocity;
        interface_method = ProjectedLiquidVelocity(),
    )
    projected_empty = TV._project_liquid_velocity!(
        empty_work, empty_velocity,
    )
    @test all(iszero, projected_empty.interior)
    @test phase_flux_diagnostics(empty_work).projection_iterations == 0
end

@testset "Compatible separate phase-momentum transport" begin
    schemes = (
        Upwind(), Centred(), LW(), Fromm(),
        Minmod(), MC(), VanLeer(), Superbee(),
    )
    methods = (ProjectedLiquidVelocity(), FrozenInterfaceFluxes())
    for mixed in (false, true), method in methods, scheme in schemes
        mesh = mixed ? BlockTreeMesh(
                block_size = (8, 8), max_level = 2,
                periodic = (true, false),
                refine = (key, origin, extent) -> key.level == 0 ||
                (key.level == 1 && key.i == 0 && key.j == 0),
            ) : BlockTreeMesh(
                block_size = (16, 16), periodic = (true, false),
            )
        geometry = mixed ? GeometryField(
                mesh, (x, y) -> 0.1 - y; lipschitz = 1,
            ) : nothing
        interface = InterfaceField(
            mesh, (x, y) -> y - 0.53; geometry, lipschitz = 1,
        )
        velocity = (
            FaceVelocity(
                mesh,
                (x, y, time, axis) -> axis == 1 ? 0.15 : 0.0,
            ),
            FaceVelocity(
                mesh,
                (x, y, time, axis) -> axis == 1 ? -0.08 : 0.0,
            ),
        )
        enforce_no_permeation!(velocity)
        destination = (FaceVelocity(mesh), FaceVelocity(mesh))
        work = WYAdvectionWorkspace(
            interface, velocity; interface_method = method,
        )
        momentum = MomentumAdvectionWorkspace(interface, velocity)
        dt = stable_timestep(velocity, work; cfl = 0.2)
        advect!(
            interface, velocity, dt, work;
            momentum = destination, density = (1000.0, 1.0),
            momentum_workspace = momentum, scheme,
            amr_interface_scheme = default_amr_interface_scheme(scheme),
        )
        for phase in 1:2
            @test destination[phase].interior ≈
                velocity[phase].interior atol = 512eps()
            @test destination[phase].boundary ≈
                velocity[phase].boundary atol = 512eps()
            @test all(isfinite, destination[phase].interior)
            @test all(isfinite, destination[phase].boundary)
        end
    end

    # In-place self-advection is defined by the workspace snapshots.
    mesh = BlockTreeMesh(
        block_size = (12, 12), periodic = (true, false),
    )
    interface = InterfaceField(
        mesh, (x, y) -> y - 0.5; lipschitz = 1,
    )
    velocity = (
        FaceVelocity(
            mesh, (x, y, time, axis) -> axis == 1 ? 0.1 : 0.0,
        ),
        FaceVelocity(
            mesh, (x, y, time, axis) -> axis == 1 ? -0.05 : 0.0,
        ),
    )
    enforce_no_permeation!(velocity)
    initial = map(copy, velocity)
    work = WYAdvectionWorkspace(
        interface, velocity; interface_method = FrozenInterfaceFluxes(),
    )
    momentum = MomentumAdvectionWorkspace(interface, velocity)
    advect!(
        interface, velocity,
        stable_timestep(velocity, work; cfl = 0.2), work;
        momentum = velocity, density = (2.0, 3.0),
        momentum_workspace = momentum,
    )
    for phase in 1:2
        @test velocity[phase].interior ≈
            initial[phase].interior atol = 256eps()
        @test velocity[phase].boundary ≈
            initial[phase].boundary atol = 256eps()
    end

    wrong_workspace = MomentumAdvectionWorkspace(interface)
    @test_throws ArgumentError advect!(
        interface, velocity,
        stable_timestep(velocity, work; cfl = 0.2), work;
        momentum = (FaceVelocity(mesh), FaceVelocity(mesh)),
        density = (1.0, 1.0), momentum_workspace = wrong_workspace,
    )
end

@testset "Two-phase cell, staggered and face geometry partitions" begin
    meshes = (
        BlockTreeMesh(
            block_size = (8, 8), periodic = (false, false),
        ),
        BlockTreeMesh(
            block_size = (8, 8), max_level = 2,
            periodic = (false, false),
            refine = (key, origin, extent) -> key.level == 0 ||
                (key.level == 1 && key.i == 0 && key.j == 0),
        ),
    )
    for mesh in meshes
        geometry = GeometryField(
            mesh, (x, y) -> y - 0.12 - 0.18x; lipschitz = 1,
        )
        interface = InterfaceField(
            mesh, (x, y) -> hypot(x - 0.48, y - 0.46) - 0.28;
            geometry, lipschitz = 1,
        )
        reconstruct!(interface)

        nx, ny = mesh.block_size
        for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
            ref = CellRef(block, i, j)
            liquid, gas = TV._phase_cell_fractions(interface, ref)
            open = TV._get(geometry.open_fraction, ref)
            @test liquid >= 0
            @test gas >= -128eps()
            @test liquid + gas == open
            @test TV._phase_cell_active(
                interface, ref, Val(:liquid),
            ) == (liquid > 0)
            @test TV._phase_cell_active(
                interface, ref, Val(:gas),
            ) == (gas > 0)
        end

        for (index, _) in pairs(mesh.faces)
            liquid, gas = TV._phase_face_apertures(interface, index)
            @test liquid >= 0
            @test gas >= 0
            @test liquid + gas == geometry.aperture[index]
        end
        for (index, _) in pairs(mesh.boundary_faces)
            liquid, gas = TV._phase_boundary_face_apertures(
                interface, index,
            )
            @test liquid >= 0
            @test gas >= 0
            @test liquid + gas == geometry.boundary_aperture[index]
        end

        for Axis in (1, 2)
            fraction = TV._phase_face_average!(
                FaceField{Axis}(mesh), interface.conditional_fraction,
                geometry,
            )
            for index in eachindex(fraction.interior)
                liquid, gas = TV._phase_staggered_fractions(
                    interface, fraction, index,
                )
                open = geometry.dual_volumes[Axis][index]
                @test liquid + gas == (open > 0 ? 1.0 : 0.0)
                @test TV._phase_staggered_active(
                    interface, fraction, index, Val(:liquid),
                ) == (liquid > 0)
                @test TV._phase_staggered_active(
                    interface, fraction, index, Val(:gas),
                ) == (gas > 0)
            end
            for index in eachindex(fraction.boundary)
                liquid, gas = TV._phase_staggered_fractions(
                    interface, fraction, index; boundary = true,
                )
                open = geometry.boundary_dual_volumes[Axis][index]
                @test liquid + gas == (open > 0 ? 1.0 : 0.0)
            end
        end

        ref = CellRef(1, 1, 1)
        phase_cell_fraction_allocations(interface, geometry, ref)
        @test phase_cell_fraction_allocations(interface, geometry, ref) == 0
        fraction = TV._phase_face_average!(
            FaceField{1}(mesh), interface.conditional_fraction, geometry,
        )
        phase_staggered_fraction_allocations(
            interface, geometry, fraction, 1,
        )
        @test phase_staggered_fraction_allocations(
            interface, geometry, fraction, 1,
        ) == 0
        phase_face_aperture_allocations(interface, geometry, 1)
        @test phase_face_aperture_allocations(
            interface, geometry, 1,
        ) == 0
    end
end

@testset "Frozen-subface WY tangential transport" begin
    meshes = (
        BlockTreeMesh(
            block_size = (8, 8), periodic = (true, false),
        ),
        BlockTreeMesh(
            block_size = (8, 8), max_level = 2,
            periodic = (true, false),
            refine = (key, origin, extent) -> key.level == 0 ||
                (key.level == 1 && key.i == 0 && key.j == 0),
        ),
    )
    for mesh in meshes, with_geometry in (false, true)
        geometry = with_geometry ? GeometryField(
                mesh, (x, y) -> 0.1 - y; lipschitz = 1,
            ) : nothing
        interface = InterfaceField(
            mesh, (x, y) -> y - 0.47; geometry, lipschitz = 1,
        )
        velocity = (
            FaceVelocity(
                mesh, (x, y, time, axis) -> axis == 1 ? 0.15 : 0.0,
            ),
            FaceVelocity(
                mesh, (x, y, time, axis) -> axis == 1 ? -0.08 : 0.0,
            ),
        )
        enforce_no_permeation!(velocity)
        constants = PhaseScalar(interface; liquid = 3.0, gas = 7.0)
        tracer = PhaseScalar(
            interface;
            liquid = (x, y) -> 1 + 0.1sinpi(2x),
            gas = (x, y) -> 2 + 0.2cospi(2x),
        )
        staggered = PhaseFaceScalar{2}(
            interface; liquid = 4.0, gas = 9.0,
        )
        initial_fraction = copy(interface.fraction)
        initial_mass = (
            integrate(tracer, :liquid), integrate(tracer, :gas),
        )
        work = WYAdvectionWorkspace(
            interface, velocity;
            interface_method = FrozenInterfaceFluxes(),
        )
        dt = stable_timestep(velocity, work; cfl = 0.25)
        # `advect!` owns the time-level aperture refresh.  A preceding CFL
        # query is optional and must not be the source of its frozen geometry.
        fill!(work.liquid_aperture, NaN)
        fill!(work.gas_aperture, NaN)
        advect!(
            interface, velocity, dt, work;
            scalars = (constants, tracer),
            staggered_scalars = (staggered,),
        )
        for block in eachindex(mesh.blocks)
            @test interior(interface.fraction, block) ≈
                interior(initial_fraction, block) atol = 64eps()
            @test interior(constants.liquid, block) ≈
                fill(3.0, mesh.block_size) atol = 64eps()
            @test interior(constants.gas, block) ≈
                fill(7.0, mesh.block_size) atol = 128eps()
        end
        @test integrate(tracer, :liquid) ≈ initial_mass[1] atol = 2.0e-14
        @test integrate(tracer, :gas) ≈ initial_mass[2] atol = 2.0e-14
        @test staggered.liquid.interior ≈
            fill(4.0, length(staggered.liquid.interior)) atol = 128eps()
        @test staggered.gas.interior ≈
            fill(9.0, length(staggered.gas.interior)) atol = 256eps()
        diagnostics = phase_flux_diagnostics(work)
        @test diagnostics.maximum_partition_defect <= eps()
        @test diagnostics.absolute_partition_defect <=
            length(mesh.faces) * eps()
    end
end

@testset "Frozen-subface WY common-normal translation" begin
    for mesh in (
            BlockTreeMesh(
                block_size = (16, 16), periodic = (true, true),
            ),
            BlockTreeMesh(
                block_size = (8, 8), max_level = 2,
                periodic = (true, true),
                refine = (key, origin, extent) -> key.level == 0 ||
                    (key.level == 1 && key.i == 0 && key.j == 0),
            ),
        )
        periodic_distance(x, center) = mod(x - center + 0.5, 1) - 0.5
        levelset(center) = (x, y) -> abs(periodic_distance(x, center)) - 0.2
        interface = InterfaceField(mesh, levelset(0.5); lipschitz = 1)
        velocity = (
            FaceVelocity(
                mesh, (x, y, time, axis) -> axis == 1 ? 0.1 : 0.15,
            ),
            FaceVelocity(
                mesh, (x, y, time, axis) -> axis == 1 ? 0.1 : -0.08,
            ),
        )
        constants = PhaseScalar(interface; liquid = 2.0, gas = 5.0)
        work = WYAdvectionWorkspace(
            interface, velocity;
            interface_method = FrozenInterfaceFluxes(),
        )
        dt = stable_timestep(velocity, work; cfl = 0.2)
        advect!(interface, velocity, dt, work; scalars = (constants,))
        reference = InterfaceField(
            mesh, levelset(0.5 + 0.1dt); lipschitz = 1,
        )
        for block in eachindex(mesh.blocks)
            translation_error = maximum(
                abs.(
                    interior(interface.fraction, block) .-
                        interior(reference.fraction, block)
                )
            )
            @test translation_error <= 512eps()
            @test interior(constants.liquid, block) ≈
                fill(2.0, mesh.block_size) atol = 128eps()
            @test interior(constants.gas, block) ≈
                fill(5.0, mesh.block_size) atol = 256eps()
        end
        @test phase_flux_diagnostics(work).maximum_partition_defect <= eps()
    end
end

@testset "Intrinsic two-phase velocity transfer on AMR with geometry" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 1,
        periodic = (false, false),
        adaptation = MeshAdaptationOptions(coarsen_delay = 0),
    )
    geometry = GeometryField(
        mesh, (x, y) -> y - 0.11 - 0.13x; lipschitz = 1,
    )
    interface = InterfaceField(
        mesh, (x, y) -> x - 0.48;
        geometry, lipschitz = 1,
    )
    liquid_function = (x, y, time, axis) ->
    axis == 1 ? 1 + 0.2y : -0.3 + 0.2x
    gas_function = (x, y, time, axis) ->
    axis == 1 ? -0.5 + 0.1y : 0.8 + 0.1x
    velocity = (
        FaceVelocity(mesh, liquid_function),
        FaceVelocity(mesh, gas_function),
    )
    wy = WYAdvectionWorkspace(
        interface, velocity; interface_method = FrozenInterfaceFluxes(),
    )
    workspace = AMRWorkspace(mesh, interface, geometry, velocity)
    @test adapt!(workspace, [:refine]; step = 1, force = true).changed
    reconstruct!(interface)
    @test TV._reconfigure_wy_workspace!(wy, interface) === wy
    @test wy.interface_method isa FrozenInterfaceFluxes
    @test length(wy.open_flux) == length(mesh.faces)
    @test length(wy.liquid_flux) == length(mesh.faces)
    @test length(wy.gas_flux) == length(mesh.faces)
    @test length(wy.liquid_aperture) == length(mesh.faces)
    @test length(wy.gas_aperture) == length(mesh.faces)
    @test length(wy.gas_carrier.blocks) == nblocks(mesh)
    for component in velocity
        @test length(component.interior) == length(mesh.faces)
        @test length(component.boundary) == length(mesh.boundary_faces)
    end
    for (component, exact) in zip(
            velocity, (liquid_function, gas_function),
        )
        for (index, face) in pairs(mesh.faces)
            @test component.interior[index] ≈ exact(
                face.center..., 0.0, face.axis,
            ) atol = 64eps()
        end
        for (index, face) in pairs(mesh.boundary_faces)
            @test component.boundary[index] ≈ exact(
                face.center..., 0.0, face.axis,
            ) atol = 64eps()
        end
    end
    @test all(eachindex(mesh.blocks)) do block
        liquid = interior(interface.fraction, block)
        open = interior(geometry.open_fraction, block)
        all((-128eps() .<= liquid) .& (liquid .<= open .+ 128eps()))
    end
end

@testset "Two-velocity interface advection after dynamic cut-cell AMR" begin
    for method in (ProjectedLiquidVelocity(), FrozenInterfaceFluxes())
        mesh = BlockTreeMesh(
            block_size = (8, 8), max_level = 2,
            periodic = (true, false),
            adaptation = MeshAdaptationOptions(coarsen_delay = 0),
            refine = (key, origin, extent) -> key.level == 0,
        )
        geometry = GeometryField(
            mesh, (x, y) -> 0.1 - y; lipschitz = 1,
        )
        interface = InterfaceField(
            mesh, (x, y) -> y - 0.47; geometry, lipschitz = 1,
        )
        velocity = (
            FaceVelocity(
                mesh,
                (x, y, time, axis) -> axis == 1 ? 0.15 : 0.0,
            ),
            FaceVelocity(
                mesh,
                (x, y, time, axis) -> axis == 1 ? -0.08 : 0.0,
            ),
        )
        constants = PhaseScalar(interface; liquid = 3.0, gas = 7.0)
        staggered = (
            PhaseFaceScalar{1}(interface; liquid = 2.0, gas = 5.0),
            PhaseFaceScalar{2}(interface; liquid = 2.0, gas = 5.0),
        )
        adaptation = AMRWorkspace(
            mesh, interface, geometry, velocity, constants, staggered...,
        )
        work = WYAdvectionWorkspace(
            interface, velocity; interface_method = method,
        )
        momentum = MomentumAdvectionWorkspace(interface, velocity)
        old_work = work
        labels = [
            block.key.i == 0 && block.key.j == 0 ? :refine : :keep
                for block in mesh.blocks
        ]
        @test adapt!(
            adaptation, labels;
            step = 1, force = true,
        ).changed
        @test length(unique(block.key.level for block in mesh.blocks)) > 1
        work = TV._reconfigure_wy_workspace!(work, interface)
        momentum = TV._reconfigure_momentum_workspace!(
            momentum, interface,
        )
        if method isa ProjectedLiquidVelocity
            @test work !== old_work
        else
            @test work === old_work
        end
        initial = copy(interface.fraction)
        destination = (FaceVelocity(mesh), FaceVelocity(mesh))
        dt = stable_timestep(velocity, work; cfl = 0.2)
        advect!(
            interface, velocity, dt, work; scalars = (constants,),
            staggered_scalars = staggered,
            momentum = destination, density = (1000.0, 1.0),
            momentum_workspace = momentum,
        )
        for block in eachindex(mesh.blocks)
            @test interior(interface.fraction, block) ≈
                interior(initial, block)
            @test interior(constants.liquid, block) ≈
                fill(3.0, mesh.block_size)
            @test interior(constants.gas, block) ≈
                fill(7.0, mesh.block_size)
        end
        for scalar in staggered
            @test scalar.liquid.interior ≈
                fill(2.0, length(scalar.liquid.interior)) atol = 512eps()
            @test scalar.gas.interior ≈
                fill(5.0, length(scalar.gas.interior)) atol = 512eps()
        end
        for phase in 1:2
            @test destination[phase].interior ≈
                velocity[phase].interior atol = 512eps()
        end

        # Complete a prescribed refine/coarsen history. Both interface
        # policies see the same tree transitions, and all phase-face schedules
        # must be rebuilt against the coarsened topology before the next step.
        coarsen_labels = [
            block.key.level == 2 ? :coarsen : :keep
                for block in mesh.blocks
        ]
        @test adapt!(
            adaptation, coarsen_labels;
            step = 2, force = true,
        ).changed
        reconstruct!(interface)
        work = TV._reconfigure_wy_workspace!(work, interface)
        momentum = TV._reconfigure_momentum_workspace!(
            momentum, interface,
        )
        initial = copy(interface.fraction)
        destination = (FaceVelocity(mesh), FaceVelocity(mesh))
        dt = stable_timestep(velocity, work; cfl = 0.2)
        advect!(
            interface, velocity, dt, work; scalars = (constants,),
            staggered_scalars = staggered,
            momentum = destination, density = (1000.0, 1.0),
            momentum_workspace = momentum,
        )
        for block in eachindex(mesh.blocks)
            @test interior(interface.fraction, block) ≈
                interior(initial, block)
            @test interior(constants.liquid, block) ≈
                fill(3.0, mesh.block_size)
            @test interior(constants.gas, block) ≈
                fill(7.0, mesh.block_size)
        end
        for scalar in staggered
            @test scalar.liquid.interior ≈
                fill(2.0, length(scalar.liquid.interior)) atol = 512eps()
            @test scalar.gas.interior ≈
                fill(5.0, length(scalar.gas.interior)) atol = 512eps()
        end
        for phase in 1:2
            @test destination[phase].interior ≈
                velocity[phase].interior atol = 512eps()
        end
    end
end
