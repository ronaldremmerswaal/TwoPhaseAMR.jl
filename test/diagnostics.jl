using Test, TwoPhaseAMR

@testset "simulation-owned diagnostics" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    physics = SinglePhase(density = 2.0, dynamic_viscosity = 0.1)
    flow = FlowSimulation(
        mesh, physics;
        velocity = (2.0, 3.0), dt = 1.0e-3,
        diagnostics = (
            KineticEnergy(), LinearMomentum(), DivergenceNorm(),
            LeafBlockCount(), CellCount(),
            Diagnostic(:reported_step, state -> state.step) => EveryNSteps(2),
        ),
    )

    @test diagnostic_names(flow) == [
        :kinetic_energy, :linear_momentum, :divergence_norm,
        :leaf_blocks, :cells, :reported_step,
    ]
    @test only(diagnostic_history(flow, :kinetic_energy).values) ≈ 13.0
    @test only(diagnostic_history(flow, :linear_momentum).values) ≈ [4.0, 6.0]
    @test only(diagnostic_history(flow, :divergence_norm).values) ≈ 0.0
    @test only(diagnostic_history(flow, :leaf_blocks).values) == nblocks(mesh)
    @test only(diagnostic_history(flow, :cells).values) == ncells(mesh)
    @test simulation_state(flow).step == 0

    fields = VisualizationFieldRegistry()
    register_field!(fields, :pressure, flow.pressure)
    register_diagnostic!(fields, flow, :kinetic_energy)
    register_diagnostic!(
        fields, flow, :linear_momentum;
        name = :momentum_x, component = 1,
    )
    session = VisualizationSession(
        fields; viewer = NoVisualization(), diagnostic_every = 1,
    )
    result = simulate!(flow, 3.0e-3; visualization = session)
    @test result.total_steps == 3
    @test diagnostic_history(flow, :reported_step).steps == [0, 2, 3]
    @test diagnostic_history(flow, :kinetic_energy).steps == [0, 1, 2, 3]
    @test all(
        sample[:kinetic_energy] ≈ 13.0 && sample[:momentum_x] ≈ 4.0
            for sample in visualization_diagnostics(session)
    )

    report = IOBuffer()
    @test isnothing(print_simulation_summary(report, flow, result))
    summary = String(take!(report))
    @test occursin("TwoPhaseAMR simulation summary", summary)
    @test occursin("model: single phase", summary)
    @test occursin("steps this run: 3", summary)
    @test occursin("active cells: 16", summary)
    @test occursin("pressure projection:", summary)
    @test occursin("kinetic_energy [4]", summary)

    add_diagnostic!(flow, :time, state -> state.time)
    @test only(diagnostic_history(flow, :time).values) == flow.time
    @test_throws ArgumentError add_diagnostic!(flow, KineticEnergy())
end

@testset "one- and two-velocity physical diagnostics" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    interface = InterfaceField(mesh, (x, y) -> x - 0.5)
    physics = TwoPhase(
        interface; density = (4.0, 1.0),
        dynamic_viscosity = (0.1, 0.1),
    )
    shared = (x, y, time, axis) -> axis == 1 ? 2.0 : -1.0
    requested = (KineticEnergy(), LinearMomentum(), PhaseVolume())
    one = FlowSimulation(
        mesh, physics; velocity_model = OneVelocityModel(),
        velocity = shared, diagnostics = requested,
    )
    two = FlowSimulation(
        mesh, physics; velocity_model = TwoVelocityModel(),
        velocity = shared, diagnostics = requested,
    )

    @test only(diagnostic_history(one, :kinetic_energy).values) ≈
        only(diagnostic_history(two, :kinetic_energy).values)
    @test only(diagnostic_history(one, :linear_momentum).values) ≈
        only(diagnostic_history(two, :linear_momentum).values)
    volumes = only(diagnostic_history(two, :phase_volume).values)
    @test volumes[1] ≈ 0.5
    @test volumes[2] ≈ 0.5
end
