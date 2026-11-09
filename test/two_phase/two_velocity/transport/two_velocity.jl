using Test, TwoPhaseAMR
isdefined(@__MODULE__, :test_allocated_call) ||
    include(joinpath(@__DIR__, "..", "..", "..", "test_utils.jl"))
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
    mesh = BlockTreeMesh(
        block_size = (8, 8), periodic = (true, false),
        origin = (0.0f0, 0.0f0), extent = (1.0f0, 1.0f0)
    )
    geometry = GeometryField(mesh, (x, y) -> x - 0.25f0; lipschitz = 1)
    interface = InterfaceField(
        mesh, (x, y) -> y - 0.43f0;
        geometry, lipschitz = 1
    )
    velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
    work = WYAdvectionWorkspace(
        interface, velocity;
        interface_transport = FrozenInterfaceFluxes()
    )
    stable_timestep(velocity, work; cfl = 0.35)
    stable_timestep(velocity, work; cfl = 0.35)
    @test test_allocated_call(() -> stable_timestep(velocity, work; cfl = 0.35)) == 0
    if VERSION >= v"1.12"
        @test @allocated(stable_timestep(velocity, work; cfl = 0.35)) == 0
    end
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
        advect!(
            interface, velocity, -1.0, one_velocity_work;
            momentum = velocity[1]
        )
        nothing
    catch err
        err
    end
    @test unsupported isa ErrorException
    @test sprint(showerror, unsupported) ==
        "unsupported two-velocity interface-advection policy"

    for (method, expected_message) in (
            (
                FrozenInterfaceFluxes(),
                "ArgumentError: sweep_order must be (1,2) or (2,1)",
            ),
            (
                ProjectedLiquidVelocity(),
                "ArgumentError: time step exceeds the projected two-velocity WY CFL limit",
            ),
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

@testset "Moving two-velocity WY carrier after regridding (#84)" begin
    dt = 0.02
    tolerance = 1.0e-10
    function make_flow(
            projection; update_interval = 1, step_size = dt, speed_scale = 1.0,
            t_init = 1.5step_size, t_end = 2.5step_size,
        )
        mesh = BlockTreeMesh(
            block_size = (4, 4), max_level = 2,
            periodic = (true, false),
            adaptation = MeshAdaptationOptions(;
                update_interval, coarsen_delay = 0, minimum_level = 1,
            ),
            refine = (key, origin, extent) -> key.level == 0,
        )
        geometry = GeometryField(mesh, (x, y) -> y - 0.1; lipschitz = 1)
        interface = InterfaceField(
            mesh, (x, y) -> y - 0.34 - 0.03cospi(2x);
            geometry, lipschitz = 1.02,
        )
        indicator = InterfaceRefinementIndicator(
            interface; level = 2, buffer_cells = 0,
            t_init, t_end,
        )
        liquid_velocity = (x, y, time, axis) -> speed_scale * (
            axis == 1 ? 0.18 + 0.04cospi(2y) : 0.02sinpi(2x) * sinpi(y)
        )
        gas_velocity = (x, y, time, axis) -> speed_scale * (
            axis == 1 ? -0.08 + 0.03cospi(2y) : 0.02sinpi(2x) * sinpi(y)
        )
        flow = FlowSimulation(
            mesh,
            TwoPhase(
                interface; density = (2.0, 1.0),
                dynamic_viscosity = (0.0, 0.0),
            );
            velocity_model = TwoVelocityModel(
                interface_transport = FrozenInterfaceFluxes(),
                pressure_projection = projection,
            ),
            velocity = (liquid_velocity, gas_velocity), boundary = FreeSlip(),
            dt = step_size, adaptivity = Adaptivity(criteria = (indicator,)),
        )
        return (; flow, mesh, geometry, interface, indicator)
    end

    # Inspect the actual frozen WY open flux on both axes, not Cartesian
    # divergence or a density-weighted average. Zero apertures must not read
    # absent-phase NaN storage. Walls are impermeable, as required by WY.
    function carrier_metrics(field, velocity)
        mesh = field.mesh
        nx, ny = mesh.block_size
        work = WYAdvectionWorkspace(
            field, velocity; interface_transport = FrozenInterfaceFluxes(),
        )
        TV._freeze_phase_apertures!(field, work)
        diagnostic_dt = 1.0e-4
        for axis in (1, 2)
            TV._compute_directional_wy_fluxes!(
                field, velocity, diagnostic_dt, work, axis,
            )
        end
        cell_flux = zeros(ncells(mesh))
        cell_flux_scale = zeros(ncells(mesh))
        max_flux_error = max_speed = max_jump = 0.0
        mixed_faces = 0
        for (index, face) in pairs(mesh.faces)
            liquid = work.liquid_aperture[index]
            gas = work.gas_aperture[index]
            liquid_rate = iszero(liquid) ? 0.0 :
                liquid * velocity[1].interior[index]
            gas_rate = iszero(gas) ? 0.0 :
                gas * velocity[2].interior[index]
            expected = face.area * (liquid_rate + gas_rate)
            actual = work.open_flux[index] / diagnostic_dt
            max_flux_error = max(max_flux_error, abs(actual - expected))
            max_speed = max(max_speed, abs(liquid_rate + gas_rate))
            if liquid > 0 && gas > 0
                mixed_faces += 1
                max_jump = max(
                    max_jump,
                    abs(velocity[1].interior[index] - velocity[2].interior[index]),
                )
            end
            minus = TV._cell_index(face.minus, nx, ny)
            plus = TV._cell_index(face.plus, nx, ny)
            cell_flux[minus] += actual
            cell_flux[plus] -= actual
            flux_scale = face.area * (abs(liquid_rate) + abs(gas_rate))
            cell_flux_scale[minus] += flux_scale
            cell_flux_scale[plus] += flux_scale
        end
        max_boundary_flux = 0.0
        for (index, face) in pairs(mesh.boundary_faces)
            liquid, gas = TV._phase_boundary_face_apertures(field, index)
            liquid_rate = iszero(liquid) ? 0.0 :
                liquid * velocity[1].boundary[index]
            gas_rate = iszero(gas) ? 0.0 :
                gas * velocity[2].boundary[index]
            max_boundary_flux = max(
                max_boundary_flux, abs(face.area * (liquid_rate + gas_rate)),
            )
        end
        max_divergence = max_divergence_scale = 0.0
        for (block, leaf) in pairs(mesh.blocks), j in 1:ny, i in 1:nx
            ref = CellRef(block, i, j)
            open = TV._get(field.geometry.open_fraction, ref)
            open > 0 || continue
            index = TV._cell_index(ref, nx, ny)
            max_divergence = max(
                max_divergence, abs(cell_flux[index]) / (leaf.volume * open),
            )
            max_divergence_scale = max(
                max_divergence_scale, cell_flux_scale[index] / (leaf.volume * open),
            )
        end
        return (;
            max_divergence, max_divergence_scale, max_flux_error, max_boundary_flux,
            max_speed, max_jump, mixed_faces,
        )
    end

    # The solvers control the fractional volume defect dt*max|div|. Allow
    # round-off in the independent face sum, including cut-cell amplification.
    carrier_defect_limit(metrics, step_size, tolerance) =
        tolerance + 64eps() * step_size * metrics.max_divergence_scale

    function check_carrier(field, velocity, step_size, tolerance)
        metrics = carrier_metrics(field, velocity)
        @test step_size * metrics.max_divergence <=
            carrier_defect_limit(metrics, step_size, tolerance)
        @test metrics.max_flux_error <= 64eps()
        @test metrics.max_boundary_flux <= 64eps()
        @test metrics.max_speed > 1.0e-4
        @test metrics.mixed_faces > 0
        @test metrics.max_jump > 1.0e-4
        return metrics
    end

    function transfer_reference!(reference, time, step)
        labels = Vector{Symbol}(undef, nblocks(reference.mesh))
        refinement_labels!(labels, reference.indicator; time)
        transfer = TV._navier_stokes_amr_workspace(
            reference.flow.velocity, reference.flow.pressure,
            reference.flow.solver,
        )
        @test adapt!(transfer, labels; step, force = true).changed
        return nothing
    end

    function check_regrid_projection(flow, reference, step_size, tolerance; max_cycles)
        solver, workspace = reference.flow.solver, reference.flow.workspace
        TV._rebuild_topology_dependent_state!(solver, workspace; dt = step_size)
        TV._enforce_phase_velocity_boundary!(reference.flow.velocity, solver, reference.flow.time)
        TV._mark_inactive_phase_velocities!(reference.flow.velocity, reference.interface)
        state = TV._two_velocity_flow_workspace!(workspace.two_phase, solver.two_phase)
        work = state.projection_workspace
        assembled = if work isa TV.CMDGFMProjectionWorkspace
            TV._assemble_cmdgfm_system!(
                state.projection_solver, work, step_size, reference.flow.velocity,
            )
        else
            TV._assemble_mimetic_system!(
                state.projection_solver, work, step_size, reference.flow.velocity,
            )
        end
        rhs = copy(assembled.right_hand_side)
        initial_defects = work isa TV.CMDGFMProjectionWorkspace ?
            TV._cmdgfm_residual_defects(rhs, assembled, step_size) :
            TV._local_mimetic_residual_defects(rhs, assembled, step_size)
        actual_work = flow.workspace.two_phase.two_velocity[].projection_workspace
        @test length(actual_work.right_hand_side) == length(rhs)
        rhs_scale = maximum(abs, rhs; init = 0.0)
        @test rhs_scale > 0
        if length(actual_work.right_hand_side) == length(rhs)
            # Independent assembly of the same physical input and step gauge;
            # this also detects normalizing the velocities before projection.
            @test maximum(abs, actual_work.right_hand_side .- rhs; init = 0.0) <=
                4096eps() * rhs_scale
        end
        stats = TV._project_regridded_two_velocity!(
            reference.flow.velocity, solver, workspace, step_size, reference.flow.time;
            tolerance, max_iterations = max_cycles,
        )
        @test stats.converged
        @test stats.target_defect == tolerance
        return maximum(initial_defects)
    end

    for projection in (CMDGFMProjection(), MimeticProjection())
        @testset "$(typeof(projection))" begin
            case = make_flow(projection)
            (; flow, mesh, geometry, interface) = case
            # A matching run skips its second adaptation. Its pressure after
            # the physical step is then transferred independently, providing
            # the pressure that regrid correction must leave untouched.
            reference = make_flow(projection; update_interval = 3)
            reference_volume = integrate(reference.interface)
            @test project!(flow; dt, tolerance, max_cycles = 4000).converged
            @test project!(reference.flow; dt, tolerance, max_cycles = 4000).converged
            check_carrier(interface, flow.velocity, dt, tolerance)
            @test any(
                any(value -> 0 < value < 1, interior(geometry.open_fraction, block))
                    for block in eachindex(mesh.blocks)
            )
            initial_volume = integrate(interface)
            @test 0 < initial_volume < 1
            initial_fraction = [
                copy(interior(interface.fraction, block)) for block in eachindex(mesh.blocks)
            ]
            initial_blocks = nblocks(mesh)
            previous_volume = initial_volume
            baseline_step_drift = 0.0

            # The time window gives one unrefined moving step, then a local
            # refinement and a coarsening, followed by coarsened-mesh steps.
            for step in 1:5
                result = simulate!(
                    flow, step * dt; dt, tolerance, max_cycles = 4000,
                )
                @test result.steps == 1
                @test result.pressure.converged
                @test result.mesh_updates == (step in (2, 3) ? 1 : 0)
                current_volume = integrate(interface)
                if step <= 2
                    reference_result = simulate!(
                        reference.flow, step * dt; dt, tolerance, max_cycles = 4000,
                    )
                    @test reference_result.steps == 1
                    @test reference_result.pressure.converged
                    next_reference_volume = integrate(reference.interface)
                    baseline_step_drift = max(
                        baseline_step_drift,
                        abs(next_reference_volume - reference_volume) / initial_volume,
                    )
                    reference_volume = next_reference_volume
                    @test reference_result.mesh_updates == 0
                end
                # chi=1 occupies at most twice the liquid volume, so a
                # projected step's relative WY dilation is at most 2*tolerance.
                # Use twice that bound for transfer/snapping and compare with
                # the measured no-regrid per-step drift; summation adds round-off.
                drift_limit = max(8baseline_step_drift, 4tolerance) +
                    512eps() / initial_volume
                @test abs(current_volume - previous_volume) / initial_volume <= drift_limit
                @test abs(current_volume - initial_volume) / initial_volume <= step * drift_limit
                previous_volume = current_volume
                if step == 1
                    movement = maximum(
                        maximum(
                            abs, interior(interface.fraction, block) .- initial_fraction[block],
                        ) for block in eachindex(mesh.blocks)
                    )
                    @test movement > 1.0e-7
                    check_carrier(interface, flow.velocity, dt, tolerance)
                elseif step == 2
                    @test nblocks(mesh) > initial_blocks
                    @test any(block -> block.key.level == 1, mesh.blocks)
                    @test any(block -> block.key.level == 2, mesh.blocks)
                    metrics = check_carrier(interface, flow.velocity, dt, tolerance)
                    println(
                        "two_velocity_regrid projection=$(nameof(typeof(projection))) " *
                            "stage=refine max_mixture_divergence=$(metrics.max_divergence)",
                    )

                    transfer_reference!(reference, step * dt, step)
                    # A raw AMR transfer is a negative control: it must
                    # expose the divergence fixed by the simulation path.
                    uncorrected = carrier_metrics(
                        reference.interface, reference.flow.velocity,
                    )
                    @test uncorrected.max_divergence > 1.0e-8
                    @test dt * uncorrected.max_divergence >
                        carrier_defect_limit(uncorrected, dt, tolerance)
                    println(
                        "two_velocity_regrid projection=$(nameof(typeof(projection))) " *
                            "stage=unprojected max_mixture_divergence=$(uncorrected.max_divergence)",
                    )
                    keys = [block.key for block in mesh.blocks]
                    reference_keys = [block.key for block in reference.mesh.blocks]
                    @test keys == reference_keys
                    if keys == reference_keys
                        # Pressure solve residual scales as tolerance / dt^2.
                        # Allow that absolute solve error plus round-off.
                        pressure_scale = max(
                            1.0,
                            maximum(
                                maximum(abs, interior(reference.flow.pressure, block))
                                    for block in eachindex(reference.mesh.blocks)
                            ),
                        )
                        pressure_atol = 2tolerance / dt^2 * pressure_scale
                        for block in eachindex(mesh.blocks)
                            @test maximum(
                                abs,
                                interior(flow.pressure, block) .-
                                    interior(reference.flow.pressure, block),
                            ) <= pressure_atol
                        end
                    end
                    check_regrid_projection(
                        flow, reference, dt, tolerance; max_cycles = 4000,
                    )
                elseif step == 3
                    @test nblocks(mesh) == initial_blocks
                    @test all(block -> block.key.level == 1, mesh.blocks)
                    metrics = check_carrier(interface, flow.velocity, dt, tolerance)
                    println(
                        "two_velocity_regrid projection=$(nameof(typeof(projection))) " *
                            "stage=coarsen max_mixture_divergence=$(metrics.max_divergence)",
                    )
                else
                    check_carrier(interface, flow.velocity, dt, tolerance)
                end
            end
        end
    end

    @testset "Regrid on a shortened final step" begin
        # A shortened final step still uses the untruncated proposed step as
        # the regrid defect gauge, protecting a resumed full-size step.
        remainder = 1.0e-9
        case = make_flow(
            CMDGFMProjection(); t_init = 2dt + remainder / 2,
            t_end = 2dt + 2remainder,
        )
        reference = make_flow(
            CMDGFMProjection(); update_interval = 4,
            t_init = 2dt + remainder / 2, t_end = 2dt + 2remainder,
        )
        (; flow, interface) = case
        @test project!(flow; dt, tolerance, max_cycles = 4000).converged
        @test project!(reference.flow; dt, tolerance, max_cycles = 4000).converged
        initial_volume = integrate(interface)
        last_time = 2dt + remainder
        result = simulate!(flow, last_time; dt, tolerance, max_cycles = 4000)
        @test result.steps == 3
        @test result.mesh_updates == 1
        @test result.pressure.converged
        metrics = check_carrier(interface, flow.velocity, dt, tolerance)
        short_volume = integrate(interface)
        # The three preceding transport steps use carriers projected with
        # steps no shorter than their own; allow twice the 2*tolerance bound
        # per step plus volume-summation round-off.
        @test abs(short_volume - initial_volume) / initial_volume <=
            3 * (4tolerance + 512eps() / initial_volume)

        reference_result = simulate!(
            reference.flow, last_time; dt, tolerance, max_cycles = 4000,
        )
        @test reference_result.steps == 3
        @test reference_result.mesh_updates == 0
        @test reference_result.pressure.converged
        transfer_reference!(reference, last_time, 3)
        check_regrid_projection(
            flow, reference, dt, tolerance; max_cycles = 4000,
        )

        result = simulate!(flow, last_time + dt; dt, tolerance, max_cycles = 4000)
        @test result.steps == 1
        @test result.pressure.converged
        # The chi-volume bound is 2dt*max|div|; double it for
        # transfer/snapping and retain the ordinary solver-tolerance floor.
        @test abs(integrate(interface) - short_volume) / initial_volume <=
            max(4tolerance, 4dt * metrics.max_divergence) + 512eps() / initial_volume
    end

    @testset "Fast phase velocities at regrid" begin
        # Reciprocal speed and time-step scaling keeps transport comparable,
        # while exercising the ordinary, unnormalized projection at high speed.
        for projection in (CMDGFMProjection(), MimeticProjection()), speed_scale in (100.0, 1000.0)
            step_size = dt / speed_scale
            case = make_flow(projection; step_size, speed_scale)
            (; flow, interface) = case
            @test project!(flow; dt = step_size, tolerance, max_cycles = 500).converged
            initial_volume = integrate(interface)
            for (step, target) in enumerate((2step_size, 3step_size))
                result = simulate!(
                    flow, target; dt = step_size, tolerance, max_cycles = 500,
                )
                @test result.mesh_updates == 1
                @test result.steps == (step == 1 ? 2 : 1)
                @test result.pressure.converged
                metrics = carrier_metrics(interface, flow.velocity)
                @test metrics.max_divergence * step_size <=
                    carrier_defect_limit(metrics, step_size, tolerance)
                # Two then three projected steps, each with a 2*tolerance
                # relative WY bound, doubled for transfer/snapping.
                @test abs(integrate(interface) - initial_volume) / initial_volume <=
                    (step + 1) * (4tolerance + 512eps() / initial_volume)

                if step == 1 && speed_scale == 1000.0
                    reference = make_flow(
                        projection; update_interval = 3, step_size, speed_scale,
                    )
                    @test project!(
                        reference.flow; dt = step_size, tolerance, max_cycles = 500,
                    ).converged
                    reference_result = simulate!(
                        reference.flow, target; dt = step_size, tolerance, max_cycles = 500,
                    )
                    @test reference_result.steps == 2
                    @test reference_result.mesh_updates == 0
                    @test reference_result.pressure.converged
                    transfer_reference!(reference, target, 2)
                    check_regrid_projection(
                        flow, reference, step_size, tolerance; max_cycles = 500,
                    )
                end
            end
        end
    end

    @testset "Ordinary regrid tolerance is not tightened" begin
        # This nonzero low-speed case has a regrid defect below the requested
        # tolerance but above a round-off-tightened target. Mimetic MINRES
        # must accept the zero correction just as for a physical projection.
        projection = MimeticProjection()
        case = make_flow(projection; speed_scale = 1.0e-10)
        reference = make_flow(projection; update_interval = 3, speed_scale = 1.0e-10)
        @test project!(case.flow; dt, tolerance, max_cycles = 500).converged
        @test project!(reference.flow; dt, tolerance, max_cycles = 500).converged
        result = simulate!(case.flow, 2dt; dt, tolerance, max_cycles = 500)
        @test result.steps == 2
        @test result.mesh_updates == 1
        @test result.pressure.converged
        reference_result = simulate!(
            reference.flow, 2dt; dt, tolerance, max_cycles = 500,
        )
        @test reference_result.steps == 2
        @test reference_result.mesh_updates == 0
        @test reference_result.pressure.converged
        transfer_reference!(reference, 2dt, 2)
        initial_defect = check_regrid_projection(
            case.flow, reference, dt, tolerance; max_cycles = 500,
        )
        @test 128eps() < initial_defect <= tolerance
        work = case.flow.workspace.two_phase.two_velocity[].projection_workspace
        @test all(iszero, work.solution)
    end
end

@testset "Two-velocity short-step reference flux" begin
    dt = 0.005
    reference_dt = 0.02
    tolerance = 1.0e-9
    max_cycles = 3000

    function reference_flux_case(projection)
        mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, false))
        interface = InterfaceField(
            mesh, (x, y) -> y - 0.43 - 0.04cospi(2x); lipschitz = 1.04,
        )
        liquid = (x, y, t, axis) -> axis == 1 ? 0.14 : 0.0
        gas = (x, y, t, axis) -> axis == 1 ? -0.06 : 0.0
        flow = FlowSimulation(
            mesh, TwoPhase(
                interface; density = (2.0, 1.0),
                dynamic_viscosity = (0.0, 0.0),
            );
            velocity_model = TwoVelocityModel(
                interface_transport = FrozenInterfaceFluxes(),
                pressure_projection = projection,
            ),
            velocity = (liquid, gas),
            pressure = (x, y) -> 2.5 + 0.05sinpi(2x) * cospi(y),
            boundary = FreeSlip(),
        )
        return (; flow, mesh, interface)
    end

    # Negative integrated outward mixture flux, with absent phases excluded
    # before multiplication. This is independent of projection assembly.
    function reference_mixture_flux(interface, velocity)
        mesh = interface.mesh
        nx, ny = mesh.block_size
        liquid, gas = TV._current_phase_face_apertures!(interface)
        result = zeros(ncells(mesh))
        phase_flux(weight, value) = iszero(weight) ? 0.0 : weight * value
        for (index, face) in pairs(mesh.faces)
            flux = face.area * (
                phase_flux(liquid[index], velocity[1].interior[index]) +
                    phase_flux(gas[index], velocity[2].interior[index])
            )
            result[TV._cell_index(face.minus, nx, ny)] -= flux
            result[TV._cell_index(face.plus, nx, ny)] += flux
        end
        for (index, face) in pairs(mesh.boundary_faces)
            liquid_open, gas_open = TV._phase_boundary_face_apertures(
                interface, interface.geometry, index,
            )
            flux = face.side * face.area * (
                phase_flux(liquid_open, velocity[1].boundary[index]) +
                    phase_flux(gas_open, velocity[2].boundary[index])
            )
            result[TV._cell_index(face.cell, nx, ny)] -= flux
        end
        return result
    end

    for projection in (CMDGFMProjection(), MimeticProjection())
        @testset "$(typeof(projection))" begin
            (; flow, mesh, interface) = reference_flux_case(projection)
            previous_pressure = copy(flow.pressure)
            initial_flux = reference_mixture_flux(interface, flow.velocity)
            initial_liquid = copy(first(TV._current_phase_face_apertures!(interface)))
            @test maximum(abs, initial_flux) > 1.0e-7

            # Snapshot the same old-geometry phase extension used by momentum
            # transport; it remains finite if the moving interface activates a face.
            state = TV._two_velocity_flow_workspace!(
                flow.workspace.two_phase, flow.solver.two_phase,
            )
            extended = (FaceVelocity(mesh), FaceVelocity(mesh))
            TV._prepare_momentum_advection!(state.momentum, flow.velocity)
            TV._copy_momentum_velocity!(extended, state.momentum)

            stats = step!(
                flow.velocity, flow.pressure, flow.solver, dt, flow.workspace;
                tolerance, max_cycles, _reference_dt = reference_dt,
            )
            @test stats.converged
            @test _phase_storage_contract(flow.velocity, interface)
            current_liquid = first(TV._current_phase_face_apertures!(interface))
            @test maximum(abs, current_liquid .- initial_liquid) > 1.0e-7
            current_old_flux = reference_mixture_flux(interface, extended)
            @test maximum(abs, current_old_flux .- initial_flux) > 1.0e-9

            references = flow.solver.pressure_solver.operator.references
            final_flux = reference_mixture_flux(interface, flow.velocity)
            target_scale = stats isa CMDGFMProjectionStats ? 1 - dt / reference_dt : 0.0
            if stats isa CMDGFMProjectionStats
                stored_flux = [
                    TV._get(flow.workspace.projection.inherited_divergence, ref)
                        for ref in references
                ]
                @test stored_flux ≈ initial_flux atol = 512eps()
            end
            target_error = maximum(eachindex(final_flux)) do index
                volume = mesh.blocks[references[index].block].volume
                abs(final_flux[index] - target_scale * initial_flux[index]) / volume
            end
            @test dt * target_error <= tolerance + 1024eps()
            if stats isa CMDGFMProjectionStats
                @test maximum(abs, final_flux) >
                    0.5 * target_scale * maximum(abs, initial_flux)
            end

            work = state.projection_workspace
            if stats isa CMDGFMProjectionStats
                for block in eachindex(mesh.blocks)
                    @test interior(flow.pressure, block) ≈
                        interior(previous_pressure, block) .+
                        interior(work.pressure_increment, block) atol = 512eps()
                end
                # The old extension has a significant normal jump on the
                # current stencils, but the new jump must still target zero.
                old_jump = [
                    TV._cmdgfm_normal_velocity_row(extended, stencil)
                        for stencil in work.stencils
                ]
                final_jump = [
                    TV._cmdgfm_normal_velocity_row(flow.velocity, stencil)
                        for stencil in work.stencils
                ]
                @test !isempty(old_jump)
                @test maximum(abs, old_jump; init = 0.0) > 1.0e-6
                @test maximum(abs, final_jump; init = 0.0) <= tolerance + 1024eps()
                @test stats.normal_velocity_defect <= tolerance + 1024eps()
            else
                # Mimetic replaces, rather than accumulates, physical pressure.
                @test any(eachindex(mesh.blocks)) do block
                    maximum(
                        abs, interior(flow.pressure, block) .-
                            interior(previous_pressure, block)
                    ) > 1.0e-3
                end
            end
            if stats isa MimeticProjectionStats
                # The legacy mimetic diagnostic reports the raw pressure work.
                data = work.cache.pressure.face_data
                dual_volumes = work.cache.face_dual_volumes
                cross = 0.0
                for phase in 1:2
                    masses = phase == 1 ? data.liquid_mass : data.gas_mass
                    gradient = phase == 1 ? work.liquid_gradient : work.gas_gradient
                    for index in eachindex(mesh.faces)
                        value = flow.velocity[phase].interior[index]
                        isnan(value) && continue
                        cross += masses[index] * dual_volumes[index] *
                            value * dt * gradient.interior[index]
                    end
                end
                @test stats.energy_identity_error ≈ abs(2cross) atol = 4096eps()
                @test stats.energy_identity_error < 1.0e-6
            end

            # A truncated simulate! call follows the same projection path as
            # an equivalent direct step with its reference proposal.
            simulation_case = reference_flux_case(projection)
            result = simulate!(
                simulation_case.flow, dt;
                dt = reference_dt, tolerance, max_cycles,
            )
            @test result.steps == 1
            @test result.pressure.converged
            for block in eachindex(mesh.blocks)
                @test interior(simulation_case.interface.fraction, block) ≈
                    interior(interface.fraction, block) atol = 512eps()
                @test interior(simulation_case.flow.pressure, block) ≈
                    interior(flow.pressure, block) atol = 1.0e-9 rtol = 1.0e-9
            end

            # The default reference and an explicit reference equal to dt
            # must take the same ordinary projection path.
            default_case = reference_flux_case(projection)
            equal_case = reference_flux_case(projection)
            @test_throws ArgumentError step!(
                default_case.flow.velocity, default_case.flow.pressure,
                default_case.flow.solver, dt, default_case.flow.workspace;
                tolerance, max_cycles, _reference_dt = dt / 2,
            )
            default_stats = step!(
                default_case.flow.velocity, default_case.flow.pressure,
                default_case.flow.solver, dt, default_case.flow.workspace;
                tolerance, max_cycles,
            )
            equal_stats = step!(
                equal_case.flow.velocity, equal_case.flow.pressure,
                equal_case.flow.solver, dt, equal_case.flow.workspace;
                tolerance, max_cycles, _reference_dt = dt,
            )
            @test default_stats.converged && equal_stats.converged
            for phase in 1:2
                a = default_case.flow.velocity[phase].interior
                b = equal_case.flow.velocity[phase].interior
                @test isnan.(a) == isnan.(b)
                active = findall(isfinite, a)
                @test a[active] ≈ b[active] atol = 1.0e-10 rtol = 1.0e-10
            end
            for block in eachindex(default_case.mesh.blocks)
                @test interior(default_case.interface.fraction, block) ≈
                    interior(equal_case.interface.fraction, block) atol = 512eps()
                @test interior(default_case.flow.pressure, block) ≈
                    interior(equal_case.flow.pressure, block) atol = 1.0e-10 rtol = 1.0e-10
            end
        end
    end
end
