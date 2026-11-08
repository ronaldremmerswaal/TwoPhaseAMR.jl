using Test, TwoPhaseAMR
include("../helpers.jl")
const TV = TwoPhaseAMR
_velocity_test_interface(work) = work.interface

@testset "Concrete WY method caches" begin
    mesh = BlockTreeMesh(block_size = (4, 4), periodic = (false, false))
    geometry = GeometryField(mesh, (x, y) -> x - 0.25; lipschitz = 1)
    velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
    for solid in (nothing, geometry)
        interface = InterfaceField(mesh; init = 0.375, geometry = solid)
        frozen = WYAdvectionWorkspace(
            interface, velocity; interface_transport = FrozenInterfaceFluxes(),
        )
        @test fieldtype(typeof(frozen), :method_workspace) === Base.RefValue{Nothing}
        @test frozen.interface.geometry === solid
        @test !hasproperty(frozen, :geometry)
        @test (@inferred stable_timestep(velocity, frozen)) == Inf
        before = copy(interface.fraction)
        advect!(interface, velocity, 0.001, frozen)
        for block in eachindex(mesh.blocks)
            @test interior(interface.fraction, block) ≈ interior(before, block) atol = 128eps()
        end
        @test TV._reconfigure_wy_workspace!(frozen, interface) === frozen
        other_geometry = isnothing(solid) ? geometry : nothing
        other = InterfaceField(mesh; init = 0.375, geometry = other_geometry)
        retargeted = TV._reconfigure_wy_workspace!(frozen, other)
        @test retargeted !== frozen
        @test retargeted.interface === other
        @test retargeted.interface.geometry === other_geometry
        @test (@inferred stable_timestep(velocity, retargeted)) == Inf

        momentum = MomentumAdvectionWorkspace(interface, velocity)
        @test fieldtype(typeof(momentum), :interface) === typeof(interface)
        @test (@inferred _velocity_test_interface(momentum)) === interface

        projected = WYAdvectionWorkspace(
            interface, velocity; interface_transport = ProjectedLiquidVelocity(),
        )
        @test isconcretetype(fieldtype(typeof(projected), :method_workspace))
        state = @inferred getindex(projected.method_workspace)
        @test state isa TV.ProjectedLiquidAdvectionWorkspace
        @test fieldtype(typeof(projected), :method_workspace) === typeof(projected.method_workspace)
        @test stable_timestep(velocity, projected) == Inf
        @test projected.method_workspace[] === state
        rebuilt = TV._reconfigure_wy_workspace!(projected, interface)
        @test rebuilt !== projected
        @test rebuilt.interface === interface
        @test typeof(rebuilt.method_workspace[]) === typeof(state)
    end

    # The projected operator is rebuilt when phase activity changes its
    # singularity, without changing the concrete cache type.
    interface = InterfaceField(mesh; init = 1)
    work = WYAdvectionWorkspace(
        interface, velocity; interface_transport = ProjectedLiquidVelocity(),
    )
    full = work.method_workspace[]
    @test full.operator.singular
    fill!(interior(interface.fraction, 1), 0)
    @test all(iszero, TV._project_liquid_velocity!(work, velocity).interior)
    empty = @inferred getindex(work.method_workspace)
    @test empty !== full
    @test typeof(empty) === typeof(full)
    @test !empty.operator.singular
    fill!(interior(interface.fraction, 1), 1)
    @test all(iszero, TV._project_liquid_velocity!(work, velocity).interior)
    restored = @inferred getindex(work.method_workspace)
    @test restored !== empty
    @test typeof(restored) === typeof(full)
    @test restored.operator.singular
end

@testset "Float32 cut-cell CFL avoids per-face boxing" begin
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, false),
        origin = (0.0f0, 0.0f0), extent = (1.0f0, 1.0f0))
    geometry = GeometryField(mesh, (x, y) -> x - 0.25f0; lipschitz = 1)
    interface = InterfaceField(mesh, (x, y) -> y - 0.43f0;
        geometry, lipschitz = 1)
    velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
    work = WYAdvectionWorkspace(interface, velocity;
        interface_transport = FrozenInterfaceFluxes())
    stable_timestep(velocity, work; cfl = 0.35)
    stable_timestep(velocity, work; cfl = 0.35)
    @test @allocated(stable_timestep(velocity, work; cfl = 0.35)) == 0
end

@testset "Phase apertures are computed once per reconstructed state" begin
    mesh = BlockTreeMesh(block_size = (8, 8))
    interface = InterfaceField(mesh)
    initialize!(interface, (1.0, 0.0), 0.47)
    velocity = (
        FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? 0.1 : 0.0),
        FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? 0.1 : 0.0),
    )
    work = WYAdvectionWorkspace(
        interface, velocity; interface_transport = FrozenInterfaceFluxes(),
    )
    @test interface.phase_aperture_updates[] == 0
    stable_timestep(velocity, work)
    @test interface.phase_aperture_updates[] == 1
    frozen_liquid = copy(work.liquid_aperture)
    frozen_gas = copy(work.gas_aperture)
    stable_timestep(velocity, work)
    @test interface.phase_aperture_updates[] == 1
    @test work.liquid_aperture == frozen_liquid
    @test work.gas_aperture == frozen_gas

    projection = CMDGFMProjectionSolver(interface)
    projection_work = CMDGFMProjectionWorkspace(projection)
    data = TV._projection_face_data(projection, projection_work)
    @test data.liquid_aperture === interface.phase_liquid_aperture
    @test data.gas_aperture === interface.phase_gas_aperture
    @test interface.phase_aperture_updates[] == 1

    # The sweep retains the old apertures even after it changes the PLIC.
    for b in eachindex(mesh.blocks)
        values = interior(interface.fraction, b)
        for index in eachindex(values)
            0 < values[index] < 1 || continue
            values[index] = max(0.0, values[index] - 0.01)
            break
        end
        break
    end
    reconstruct!(interface)
    @test !interface.phase_apertures_valid[]
    @test work.liquid_aperture == frozen_liquid
    @test work.gas_aperture == frozen_gas
    data = TV._projection_face_data(projection, projection_work)
    @test interface.phase_aperture_updates[] == 2
    @test data.liquid_aperture === interface.phase_liquid_aperture
    stable_timestep(velocity, work)
    @test interface.phase_aperture_updates[] == 2
    @test work.liquid_aperture == interface.phase_liquid_aperture
    @test work.gas_aperture == interface.phase_gas_aperture
end

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
    @test workspace.interface_transport isa FrozenInterfaceFluxes
    @test WYAdvectionWorkspace(
        interface, velocity; interface_transport = FrozenInterfaceFluxes(),
    ).interface_transport isa FrozenInterfaceFluxes
    @test_throws ArgumentError WYAdvectionWorkspace(
        interface, velocity; interface_transport = :projected,
    )
    projected_workspace = WYAdvectionWorkspace(
        interface, velocity; interface_transport = ProjectedLiquidVelocity(),
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
        interface_transport = FrozenInterfaceFluxes(),
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

@testset "Two-velocity policy dispatch validation" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), periodic = (true, false),
    )
    interface = InterfaceField(mesh, (x, y) -> y - 0.47; lipschitz = 1)
    velocity = (
        FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? 0.1 : 0.0),
        FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? -0.2 : 0.0),
    )
    enforce_no_permeation!(velocity)

    one_velocity_work = WYAdvectionWorkspace(interface)
    expected = min(
        stable_timestep(velocity[1], one_velocity_work; cfl = 0.3),
        stable_timestep(velocity[2], one_velocity_work; cfl = 0.3),
    )
    @test stable_timestep(velocity, one_velocity_work; cfl = 0.3) == expected
    @test_throws ArgumentError TV._project_liquid_velocity!(
        one_velocity_work, velocity,
    )
    unsupported = try
        advect!(interface, velocity, -1.0, one_velocity_work;
            momentum = velocity[1])
        nothing
    catch err
        err
    end
    @test unsupported isa ErrorException
    @test sprint(showerror, unsupported) ==
        "unsupported two-velocity interface-advection policy"

    for (method, expected_message) in (
            (FrozenInterfaceFluxes(),
                "ArgumentError: sweep_order must be (1,2) or (2,1)"),
            (ProjectedLiquidVelocity(),
                "ArgumentError: time step exceeds the projected two-velocity WY CFL limit"),
        )
        work = WYAdvectionWorkspace(interface, velocity; interface_transport = method)
        bad_momentum = try
            advect!(interface, velocity, -1.0, work; momentum = velocity[1])
            nothing
        catch err
            err
        end
        @test bad_momentum isa ArgumentError
        @test sprint(showerror, bad_momentum) ==
            "ArgumentError: two-phase velocities require a liquid/gas momentum destination tuple"

        limit = stable_timestep(velocity, work)
        @test isfinite(limit) && limit > 0
        bad_step = try
            advect!(
                interface, velocity, 2limit, work;
                check_timestep = false, sweep_order = (1, 1),
            )
            nothing
        catch err
            err
        end
        @test bad_step isa ArgumentError
        @test sprint(showerror, bad_step) == expected_message
    end
end

@testset "Prepared transport dispatch matches public advection" begin
    for method in (FrozenInterfaceFluxes(), ProjectedLiquidVelocity())
        mesh = BlockTreeMesh(
            block_size = (8, 8), periodic = (true, false),
        )
        fields = (
            InterfaceField(mesh, (x, y) -> y - 0.47; lipschitz = 1),
            InterfaceField(mesh, (x, y) -> y - 0.47; lipschitz = 1),
        )
        velocity = (
            FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? 0.1 : 0.0),
            FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? -0.2 : 0.0),
        )
        enforce_no_permeation!(velocity)
        works = map(fields) do field
            WYAdvectionWorkspace(field, velocity; interface_transport = method)
        end
        scalars = map(fields) do field
            PhaseScalar(
                field; liquid = (x, y) -> 1 + 0.1sinpi(2x),
                gas = (x, y) -> 2 + 0.2cospi(2x),
            )
        end
        destinations = map(fields) do field
            (FaceVelocity(field.mesh), FaceVelocity(field.mesh))
        end
        momentum_workspaces = map(fields) do field
            MomentumAdvectionWorkspace(field, velocity)
        end
        dt = stable_timestep(velocity, works[1]; cfl = 0.2)
        @test dt == stable_timestep(velocity, works[2]; cfl = 0.2)
        advect!(
            fields[1], velocity, dt, works[1];
            scalars = (scalars[1],), sweep_order = (2, 1),
            momentum = destinations[1], density = (1000.0, 1.0),
            momentum_workspace = momentum_workspaces[1],
        )
        TV._advect_transport_substep!(
            method, fields[2], velocity, dt, works[2];
            scalars = (scalars[2],), sweep_order = (2, 1),
            momentum = destinations[2], density = (1000.0, 1.0),
            momentum_workspace = momentum_workspaces[2],
        )
        for block in eachindex(mesh.blocks)
            @test isequal(interior(fields[1], block), interior(fields[2], block))
            @test isequal(
                interior(scalars[1].liquid, block),
                interior(scalars[2].liquid, block),
            )
            @test isequal(
                interior(scalars[1].gas, block),
                interior(scalars[2].gas, block),
            )
        end
        for phase in 1:2
            @test reinterpret(UInt64, destinations[1][phase].interior) ==
                reinterpret(UInt64, destinations[2][phase].interior)
            @test reinterpret(UInt64, destinations[1][phase].boundary) ==
                reinterpret(UInt64, destinations[2][phase].boundary)
        end
        @test reinterpret(UInt64, works[1].open_flux) ==
            reinterpret(UInt64, works[2].open_flux)
        @test reinterpret(UInt64, works[1].liquid_flux) ==
            reinterpret(UInt64, works[2].liquid_flux)
        @test reinterpret(UInt64, works[1].gas_flux) ==
            reinterpret(UInt64, works[2].gas_flux)
    end
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
        interface_transport = ProjectedLiquidVelocity(
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
        interface_transport = ProjectedLiquidVelocity(),
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
        interface_transport = ProjectedLiquidVelocity(),
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
        interface_transport = ProjectedLiquidVelocity(
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
        interface_transport = ProjectedLiquidVelocity(),
    )
    projected_empty = TV._project_liquid_velocity!(
        empty_work, empty_velocity,
    )
    @test all(iszero, projected_empty.interior)
    @test phase_flux_diagnostics(empty_work).projection_iterations == 0
end

@testset "Compatible separate phase-momentum transport" begin
    schemes = (LaxWendroff(), MonotonizedCentral())
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
            interface, velocity; interface_transport = method,
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
            indices = _phase_interior_indices(interface, phase)
            boundary = _phase_boundary_indices(interface, phase)
            @test destination[phase].interior[indices] ≈
                velocity[phase].interior[indices] atol = 512eps()
            @test destination[phase].boundary[boundary] ≈
                velocity[phase].boundary[boundary] atol = 512eps()
        end
        @test _phase_storage_contract(destination, interface)
    end


    @testset "remaining schemes dispatch on a small uniform mesh" begin
        for scheme in (
                Upwind(), Centred(), Fromm(), Minmod(), VanLeer(), Superbee(),
            )
            smoke_mesh = BlockTreeMesh(
                block_size = (4, 4), periodic = (true, false),
            )
            smoke_interface = InterfaceField(
                smoke_mesh, (x, y) -> y - 0.53; lipschitz = 1,
            )
            smoke_velocity = (
                FaceVelocity(
                    smoke_mesh,
                    (x, y, time, axis) -> axis == 1 ? 0.15 : 0.0,
                ),
                FaceVelocity(
                    smoke_mesh,
                    (x, y, time, axis) -> axis == 1 ? -0.08 : 0.0,
                ),
            )
            destination = (
                FaceVelocity(smoke_mesh), FaceVelocity(smoke_mesh),
            )
            work = WYAdvectionWorkspace(
                smoke_interface, smoke_velocity;
                interface_transport = FrozenInterfaceFluxes(),
            )
            momentum = MomentumAdvectionWorkspace(
                smoke_interface, smoke_velocity,
            )
            advect!(
                smoke_interface, smoke_velocity,
                stable_timestep(smoke_velocity, work; cfl = 0.2), work;
                momentum = destination, density = (1000.0, 1.0),
                momentum_workspace = momentum, scheme,
                amr_interface_scheme = default_amr_interface_scheme(scheme),
            )
            @test _phase_storage_contract(destination, smoke_interface)
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
        interface, velocity; interface_transport = FrozenInterfaceFluxes(),
    )
    momentum = MomentumAdvectionWorkspace(interface, velocity)
    advect!(
        interface, velocity,
        stable_timestep(velocity, work; cfl = 0.2), work;
        momentum = velocity, density = (2.0, 3.0),
        momentum_workspace = momentum,
    )
    for phase in 1:2
        indices = _phase_interior_indices(interface, phase)
        boundary = _phase_boundary_indices(interface, phase)
        @test velocity[phase].interior[indices] ≈
            initial[phase].interior[indices] atol = 256eps()
        @test velocity[phase].boundary[boundary] ≈
            initial[phase].boundary[boundary] atol = 256eps()
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
        cell_partitions_valid = true
        cell_activity_valid = true
        for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
            ref = CellRef(block, i, j)
            liquid, gas = TV._phase_cell_fractions(interface, ref)
            open = TV._get(geometry.open_fraction, ref)
            cell_partitions_valid &=
                liquid >= 0 && gas >= -128eps() && liquid + gas == open
            cell_activity_valid &= TV._phase_cell_active(
                interface, ref, Val(:liquid),
            ) == (liquid > 0) && TV._phase_cell_active(
                interface, ref, Val(:gas),
            ) == (gas > 0)
        end
        @test cell_partitions_valid
        @test cell_activity_valid

        face_partitions_valid = true
        for (index, _) in pairs(mesh.faces)
            liquid, gas = TV._phase_face_apertures(interface, index)
            face_partitions_valid &= liquid >= 0 && gas >= 0 &&
                liquid + gas == geometry.aperture[index]
        end
        boundary_partitions_valid = true
        for (index, _) in pairs(mesh.boundary_faces)
            liquid, gas = TV._phase_boundary_face_apertures(
                interface, index,
            )
            boundary_partitions_valid &= liquid >= 0 && gas >= 0 &&
                liquid + gas == geometry.boundary_aperture[index]
        end
        @test face_partitions_valid
        @test boundary_partitions_valid

        for Axis in (1, 2)
            fraction = TV._phase_face_average!(
                FaceField{Axis}(mesh), interface.conditional_fraction,
                geometry,
            )
            staggered_partitions_valid = true
            staggered_activity_valid = true
            for index in eachindex(fraction.interior)
                liquid, gas = TV._phase_staggered_fractions(
                    interface, fraction, index,
                )
                open = geometry.dual_volumes[Axis][index]
                staggered_partitions_valid &=
                    liquid + gas == (open > 0 ? 1.0 : 0.0)
                staggered_activity_valid &= TV._phase_staggered_active(
                    interface, fraction, index, Val(:liquid),
                ) == (liquid > 0) && TV._phase_staggered_active(
                    interface, fraction, index, Val(:gas),
                ) == (gas > 0)
            end
            boundary_staggered_partitions_valid = true
            for index in eachindex(fraction.boundary)
                liquid, gas = TV._phase_staggered_fractions(
                    interface, fraction, index; boundary = true,
                )
                open = geometry.boundary_dual_volumes[Axis][index]
                boundary_staggered_partitions_valid &=
                    liquid + gas == (open > 0 ? 1.0 : 0.0)
            end
            @test staggered_partitions_valid
            @test staggered_activity_valid
            @test boundary_staggered_partitions_valid
        end

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
            interface_transport = FrozenInterfaceFluxes(),
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
            interface_transport = FrozenInterfaceFluxes(),
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
        interface, velocity; interface_transport = FrozenInterfaceFluxes(),
    )
    workspace = AMRWorkspace(mesh, interface, geometry, velocity)
    @test adapt!(workspace, [:refine]; step = 1, force = true).changed
    reconstruct!(interface)
    @test TV._reconfigure_wy_workspace!(wy, interface) === wy
    @test wy.interface_transport isa FrozenInterfaceFluxes
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
    @test _phase_storage_contract(velocity, interface)
    for (phase, exact) in enumerate((liquid_function, gas_function))
        for index in _phase_interior_indices(interface, phase)
            face = mesh.faces[index]
            @test velocity[phase].interior[index] ≈ exact(
                face.center..., 0.0, face.axis,
            ) atol = 64eps()
        end
        for index in _phase_boundary_indices(interface, phase)
            face = mesh.boundary_faces[index]
            @test velocity[phase].boundary[index] ≈ exact(
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
        index = TV._find_field_buffer(adaptation, interface.fraction)
        registry = adaptation.field_buffers[index].prolongation.staggered
        @test all(isconcretetype, Base.uniontypes(eltype(registry)))
        @test all(q -> q isa eltype(registry), staggered)
        @test all(q -> fieldtype(typeof(q), :interface) === typeof(interface), staggered)
        work = WYAdvectionWorkspace(
            interface, velocity; interface_transport = method,
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
            valid = _phase_interior_indices(interface, phase)
            @test destination[phase].interior[valid] ≈
                velocity[phase].interior[valid] atol = 512eps()
        end
        @test _phase_storage_contract(destination, interface)

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
            valid = _phase_interior_indices(interface, phase)
            @test destination[phase].interior[valid] ≈
                velocity[phase].interior[valid] atol = 512eps()
        end
        @test _phase_storage_contract(destination, interface)
    end
end
