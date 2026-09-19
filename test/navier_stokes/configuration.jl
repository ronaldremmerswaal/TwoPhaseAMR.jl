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

    @test flow.solver isa NavierStokesSystem
    @test NavierStokesSystem === NavierStokesSolver
    @test flow.solver.density == 2.0
    @test flow.solver.dynamic_viscosity == 0.1
    @test flow.solver.scheme isa Upwind
    @test flow.solver.amr_interface_scheme isa Upwind
    @test flow.solver.pre_sweeps == 1
    @test flow.solver.post_sweeps == 3
    @test flow.solver.pressure_preconditioner === nothing

    marker = CellField(mesh)
    criterion = InterpolationErrorIndicator(
        marker; E_refine = 1.0, E_coarsen = 0.5,
    )
    adaptive = FlowSimulation(
        mesh, physics;
        numerics = NavierStokesOptions(multigrid = multigrid),
        adaptivity = Adaptivity(criteria = criterion, fields = marker),
    )
    @test adaptive.solver.refinement_indicators == (criterion,)
    @test adaptive.solver.adaptive_fields == (marker,)

    @test_throws ArgumentError SinglePhase(
        density = 0.0, dynamic_viscosity = 0.1,
    )
    @test_throws ArgumentError LinearSolverOptions(krylov_options = [])
    @test_throws ArgumentError MultigridOptions(max_levels = 0)
    @test_throws ArgumentError Adaptivity(criteria = :interface)
    @test_throws ArgumentError Adaptivity(fields = :temperature)
    @test_throws ArgumentError Adaptivity((:interface,), ())
end

@testset "typed two-phase choice matrix constructs" begin
    for surface_method in (CGFM(), CSF(), GFM(conservative = false))
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
        @test shared.solver.two_phase.formulation == surface_method

        for transport in (
                    FrozenInterfaceFluxes(), ProjectedLiquidVelocity(),
                ), projection in (CMDGFMProjection(), MimeticProjection())
            flow = FlowSimulation(
                mesh, physics;
                velocity_model = TwoVelocityModel(
                    interface_transport = transport,
                    pressure_projection = projection,
                ),
            )
            @test flow.solver.two_phase.formulation == surface_method
            @test flow.solver.two_phase.interface_transport == transport
            @test flow.solver.two_phase.pressure_projection == projection
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
        surface_tension = SurfaceTension(0.072; method = GFM()),
        gravity = (0.0, -9.81),
    )
    numerics = NavierStokesOptions(
        multigrid = MultigridOptions(coarse_size = 8),
    )
    shared = (x, y, time, axis) -> axis == 1 ? 0.25 : -0.5
    @test SurfaceTension().method == CGFM()
    @test CGFM() == GFM(conservative = true)
    @test CSF() isa AbstractSurfaceTensionMethod
    @test GFM() isa AbstractSurfaceTensionMethod
    @test FrozenInterfaceFluxes() isa AbstractInterfaceTransportMethod
    @test ProjectedLiquidVelocity() isa AbstractInterfaceTransportMethod
    @test CMDGFMProjection() isa AbstractTwoVelocityProjection
    @test MimeticProjection() isa AbstractTwoVelocityProjection
    @test AbstractTwoVelocityPressureProjection ===
        AbstractTwoVelocityProjection
    @test MimeticGradient === MimeticProjection
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
    @test one.solver.two_phase.formulation isa GFM

    two = FlowSimulation(
        mesh, physics; velocity_model = TwoVelocityModel(
            interface_transport = ProjectedLiquidVelocity(),
            pressure_projection = MimeticProjection(),
        ), velocity = shared, numerics,
    )
    @test two.velocity isa TwoPhaseVelocity
    @test two.velocity[1] !== two.velocity[2]
    @test two.solver.two_phase.interface_transport isa ProjectedLiquidVelocity
    @test two.solver.two_phase.pressure_projection isa MimeticProjection
    @test two.velocity[1].interior == two.velocity[2].interior

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
    @test_throws ArgumentError NavierStokesSystem(
        mesh; interface, density = physics.density,
        dynamic_viscosity = physics.dynamic_viscosity,
        surface_tension = physics.surface_tension.coefficient,
        surface_tension_formulation = physics.surface_tension.method,
        gravity = physics.gravity,
        interface_transport = ProjectedLiquidVelocity(),
        interface_method = FrozenInterfaceFluxes(),
    )

    phase_initializer = (shared, shared)
    @test_throws ArgumentError FlowSimulation(
        mesh, physics; velocity_model = OneVelocityModel(),
        velocity = phase_initializer, numerics,
    )
    @test_throws ArgumentError TwoVelocityModel(interface_transport = CSF())
    @test_throws ArgumentError TwoVelocityModel(pressure_projection = GFM())
    @test_throws ArgumentError SurfaceTension(1.0; method = MimeticProjection())
    @test_throws ArgumentError SurfaceTension(-1.0)
end
