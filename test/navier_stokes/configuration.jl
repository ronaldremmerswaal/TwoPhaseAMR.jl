using Test, TwoPhaseAMR

@testset "typed Navier--Stokes configuration" begin
    mesh = BlockTreeMesh(block_size = (4, 4), periodic = (false, false))
    linear = LinearSolverOptions(
        preconditioner = nothing,
        krylov_options = (; atol = 1.0e-10),
    )
    multigrid = MultigridOptions(
        coarse_size = 8, pre_sweeps = 1, post_sweeps = 3,
    )
    numerics = NavierStokesOptions(
        advection = AdvectionOptions(scheme = Upwind()),
        pressure = linear,
        diffusion = LinearSolverOptions(preconditioner = nothing),
        multigrid = multigrid,
    )
    physics = SinglePhase(density = 2.0, dynamic_viscosity = 0.1)
    flow = FlowSimulation(
        mesh, physics; numerics, dt = 0.01, boundary = FreeSlip(),
    )

    @test flow.system isa NavierStokesSystem
    @test flow.system.physics === physics
    @test flow.system.options === numerics
    @test flow.system.physics.density == 2.0
    @test flow.system.physics.dynamic_viscosity == 0.1
    @test flow.system.options.advection.scheme isa Upwind
    @test flow.system.options.advection.amr_interface_scheme isa Upwind
    @test flow.system.options.multigrid.pre_sweeps == 1
    @test flow.system.options.multigrid.post_sweeps == 3
    @test isnothing(flow.system.pressure_solver.preconditioner)

    marker = CellField(mesh)
    criterion = InterpolationErrorIndicator(
        marker; E_refine = 1.0, E_coarsen = 0.5,
    )
    adaptive = FlowSimulation(
        mesh, physics;
        numerics = NavierStokesOptions(multigrid = multigrid),
        adaptivity = Adaptivity(criteria = criterion, fields = marker),
    )
    @test adaptive.system.adaptivity.criteria == (criterion,)
    @test adaptive.system.adaptivity.fields == (marker,)

    @test_throws ArgumentError SinglePhase(
        density = 0.0, dynamic_viscosity = 0.1,
    )
    @test_throws ArgumentError LinearSolverOptions(krylov_options = [])
    @test_throws ArgumentError LinearSolverOptions(krylov = :bogus)
    @test_throws ArgumentError LinearSolverOptions(preconditioner = :bogus)
    @test_throws ArgumentError FlowSimulation(
        mesh, physics;
        numerics = NavierStokesOptions(
            pressure = LinearSolverOptions(:bogus, (;), :auto, (;)),
        ),
    )
    @test_throws ArgumentError FlowSimulation(
        mesh, physics;
        numerics = NavierStokesOptions(
            diffusion = LinearSolverOptions(nothing, (;), :bogus, (;)),
        ),
    )
    @test_throws ArgumentError MimeticProjection(preconditioner = :bogus)
    @test_throws ArgumentError NavierStokesSystem(
        mesh; density = 1.0, dynamic_viscosity = 0.1,
        diffusion_preconditioner = :bogus,
    )
    @test_throws ArgumentError NavierStokesSystem(
        mesh; density = 1.0, dynamic_viscosity = 0.1,
        pressure_krylov = :bogus,
    )
    @test_throws ArgumentError MultigridOptions(max_levels = 0)
    @test_throws ArgumentError MultigridOptions(0, 2, 2, 32)
    @test_throws ArgumentError MultigridOptions(coarse_size = true)
    @test_throws ArgumentError Adaptivity(criteria = :interface)
    @test_throws ArgumentError Adaptivity(fields = :temperature)
    @test_throws ArgumentError Adaptivity((:interface,), ())
end

@testset "positional configuration validates physical and policy invariants" begin
    mesh = BlockTreeMesh(block_size = (4, 4), periodic = (false, false))
    interface = InterfaceField(mesh)
    other_interface = InterfaceField(mesh)
    tension = SurfaceTension(0.1, CGFM())
    density = (2.0, 1.0)
    viscosity = (0.1, 0.01)
    gravity = (0.0, -1.0)

    @test SinglePhase(2.0, 0.1).density == 2.0
    for invalid in (0.0, -1.0, Inf, NaN)
        @test_throws ArgumentError SinglePhase(invalid, 0.1)
        @test_throws ArgumentError SinglePhase(1.0, invalid)
    end
    @test_throws ArgumentError SinglePhase(:invalid, 0.1)
    @test_throws ArgumentError SurfaceTension(-1.0, CGFM())
    @test_throws ArgumentError SurfaceTension(Inf, CGFM())
    @test_throws ArgumentError SurfaceTension(1.0, :invalid)
    @test tension.method isa CGFM

    physical = TwoPhase(interface, density, viscosity, tension, gravity, ())
    @test physical.interface === interface
    @test physical.density == density
    @test_throws ArgumentError TwoPhase(interface, (0.0, 1.0), viscosity, tension, gravity, ())
    @test_throws ArgumentError TwoPhase(interface, density, (-0.1, 0.1), tension, gravity, ())
    @test_throws ArgumentError TwoPhase(interface, density, (0.0, 0.1), tension, gravity, ())
    @test_throws ArgumentError TwoPhase(interface, density, viscosity, -1.0, gravity, ())
    @test_throws ArgumentError TwoPhase(interface, density, viscosity, tension, (0.0, Inf), ())
    @test_throws ArgumentError TwoPhase(interface, density, viscosity, tension, gravity, (:bad,))
    @test_throws ArgumentError TwoPhase(
        interface, density, viscosity, tension, gravity, (PhaseScalar(other_interface),),
    )
    phase_scalar = PhaseScalar(interface)
    with_scalar = TwoPhase(interface, density, viscosity, tension, gravity, (phase_scalar,))
    model = TwoPhaseAMR.TwoPhaseModel(with_scalar)
    @test model.interface === interface
    @test model.scalars[1] === phase_scalar
    @test model.density == density
    @test model.dynamic_viscosity == viscosity
    @test_throws ArgumentError TwoPhaseAMR.TwoPhaseModel(
        with_scalar; interface_transport = :invalid,
    )
    @test_throws ArgumentError TwoPhaseAMR.TwoPhaseModel(
        with_scalar; pressure_projection = :invalid,
    )
    periodic_interface = InterfaceField(BlockTreeMesh(block_size = (4, 4)))
    periodic_physics = TwoPhase(
        periodic_interface, density, viscosity, tension, gravity, (),
    )
    @test_throws ArgumentError TwoPhaseAMR.TwoPhaseModel(periodic_physics)

    @test_throws ArgumentError AdvectionOptions(Centred(), Upwind())
    @test_throws ArgumentError AdvectionOptions(:bogus, Upwind())
    @test_throws ArgumentError NavierStokesOptions(
        :bogus, LinearSolverOptions(), LinearSolverOptions(), MultigridOptions(),
    )
    @test_throws ArgumentError TwoVelocityModel(
        TwoPhaseAMR._OneVelocityInterfaceAdvection(), CMDGFMProjection(),
    )
    @test_throws ArgumentError TwoVelocityModel(FrozenInterfaceFluxes(), CSF())
    @test_throws ArgumentError ProjectedLiquidVelocity(0, 1.0e-8, 10)
    @test_throws ArgumentError ProjectedLiquidVelocity(1, 0.0, 10)
    @test_throws ArgumentError ProjectedLiquidVelocity(1, 1.0e-8, 0)
end

@testset "typed two-phase choice matrix constructs" begin
    for surface_method in (CGFM(), CSF())
        mesh = BlockTreeMesh(block_size = (4, 4))
        interface = InterfaceField(mesh, (x, y) -> x - 0.37)
        physics = TwoPhase(
            interface; density = (2.0, 1.0),
            dynamic_viscosity = (0.0, 0.0),
            surface_tension = SurfaceTension(0.0; method = surface_method),
        )
        shared = FlowSimulation(
            mesh, physics; velocity_model = OneVelocityModel(),
        )
        @test shared.system.two_phase.formulation == surface_method

        for transport in (
                    FrozenInterfaceFluxes(), ProjectedLiquidVelocity(),
                ), projection in (
                    CMDGFMProjection(), MimeticProjection(),
                    MimeticProjection(variant = :joint_face),
                )
            flow = FlowSimulation(
                mesh, physics;
                velocity_model = TwoVelocityModel(
                    interface_transport = transport,
                    pressure_projection = projection,
                ),
            )
            @test flow.system.two_phase.formulation == surface_method
            @test flow.system.two_phase.interface_transport == transport
            @test flow.system.two_phase.pressure_projection == projection
        end
    end
end

@testset "explicit one- and two-velocity models" begin
    mesh = BlockTreeMesh(block_size = (4, 4), periodic = (false, false))
    interface = InterfaceField(mesh, (x, y) -> x - 0.5)
    physics = TwoPhase(
        interface;
        density = (1000.0, 1.0),
        dynamic_viscosity = (1.0e-3, 1.0e-5),
        surface_tension = SurfaceTension(0.072; method = CGFM()),
        gravity = (0.0, -9.81),
    )
    numerics = NavierStokesOptions(
        multigrid = MultigridOptions(coarse_size = 8),
    )
    shared = (x, y, time, axis) -> axis == 1 ? 0.25 : -0.5
    @test SurfaceTension().method == CGFM()
    @test CSF() isa AbstractSurfaceTensionMethod
    @test CGFM() isa AbstractSurfaceTensionMethod
    @test FrozenInterfaceFluxes() isa AbstractInterfaceTransportMethod
    @test ProjectedLiquidVelocity() isa AbstractInterfaceTransportMethod
    @test CMDGFMProjection() isa AbstractTwoVelocityProjection
    @test MimeticProjection() isa AbstractTwoVelocityProjection
    @test TwoVelocityModel().interface_transport isa FrozenInterfaceFluxes
    @test TwoVelocityModel().pressure_projection isa CMDGFMProjection

    numeric_tension = TwoPhase(
        interface;
        density = (1000.0, 1.0),
        dynamic_viscosity = (1.0e-3, 1.0e-5),
        surface_tension = 0.05,
    )
    @test numeric_tension.surface_tension.coefficient == 0.05
    @test numeric_tension.surface_tension.method == CGFM()

    one = FlowSimulation(
        mesh, physics; velocity_model = OneVelocityModel(),
        velocity = shared, numerics,
    )
    @test one.velocity isa FaceVelocity
    @test one.system.physics === physics
    @test one.system.options === numerics
    @test length(typeof(one.system).parameters) <= 8
    @test one.system.two_phase.formulation isa CGFM

    two = FlowSimulation(
        mesh, physics; velocity_model = TwoVelocityModel(
            interface_transport = ProjectedLiquidVelocity(),
            pressure_projection = MimeticProjection(),
        ), velocity = shared, numerics,
    )
    @test two.velocity isa TwoPhaseVelocity
    @test two.system.physics === physics
    @test two.system.options === numerics
    @test length(typeof(two.system).parameters) <= 8
    @test two.velocity[1] !== two.velocity[2]
    @test two.system.two_phase.interface_transport isa ProjectedLiquidVelocity
    @test two.system.two_phase.pressure_projection isa MimeticProjection
    # Equal phase initializers agree where both phases are present. Inactive
    # values are deliberately NaN, rather than a dormant shared extension.
    for index in eachindex(two.velocity[1].interior)
        liquid, gas = two.velocity[1].interior[index], two.velocity[2].interior[index]
        @test isnan(liquid) || isnan(gas) || liquid == gas
    end

    canonical_system = NavierStokesSystem(
        mesh; interface, density = physics.density,
        dynamic_viscosity = physics.dynamic_viscosity,
        surface_tension = physics.surface_tension.coefficient,
        surface_tension_formulation = physics.surface_tension.method,
        gravity = physics.gravity,
        interface_transport = ProjectedLiquidVelocity(),
    )
    @test canonical_system.two_phase.interface_transport isa
        ProjectedLiquidVelocity

    phase_initializer = (shared, shared)
    @test_throws ArgumentError FlowSimulation(
        mesh, physics; velocity_model = OneVelocityModel(),
        velocity = phase_initializer, numerics,
    )
    @test_throws ArgumentError TwoVelocityModel(interface_transport = CSF())
    @test_throws ArgumentError TwoVelocityModel(pressure_projection = CGFM())
    @test_throws ArgumentError SurfaceTension(1.0; method = MimeticProjection())
    @test_throws ArgumentError SurfaceTension(-1.0)
end
