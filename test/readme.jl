@testset "README quick start" begin
    mesh = BlockTreeMesh(
        block_size = (16, 16), extent = (2pi, 2pi),
    )
    physics = SinglePhase(
        density = 1.0, dynamic_viscosity = 0.02,
    )
    initial_velocity = (x, y, time, axis) ->
        axis == 1 ? sin(x) * cos(y) : -cos(x) * sin(y)

    flow = FlowSimulation(
        mesh, physics;
        velocity = initial_velocity,
        boundary = FreeSlip(),
        dt = 1.0e-3,
        diagnostics = (KineticEnergy(), DivergenceNorm()),
    )
    result = simulate!(flow, 0.002)

    energy = diagnostic_history(flow, :kinetic_energy)
    @test result.pressure.converged
    @test length(energy) == result.steps + 1
end
