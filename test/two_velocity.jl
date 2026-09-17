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
    expected = min(
        stable_timestep(liquid, workspace; cfl = 0.3),
        stable_timestep(gas, workspace; cfl = 0.3),
    )
    @test stable_timestep(velocity, workspace; cfl = 0.3) == expected

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

    # The tuple API already has an exact one-velocity reduction.  Distinct
    # arrays are used so this also tests equality by value rather than aliasing.
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
        tuple_interface, (common, common_copy),
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

    unequal = (common, copy(common))
    unequal[2].interior[1] += 0.01
    @test_throws ArgumentError advect!(
        copy(direct_interface), unequal, dt,
        WYAdvectionWorkspace(direct_interface, unequal),
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
    workspace = AMRWorkspace(mesh, interface, geometry, velocity)
    @test adapt!(workspace, [:refine]; step = 1, force = true).changed
    reconstruct!(interface)
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
