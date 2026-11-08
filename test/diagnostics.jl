using Test, TwoPhaseAMR
using StaticArrays: SVector
include("embedded_operator_reference_helpers.jl")
isdefined(@__MODULE__, :test_allocated_call) ||
    include(joinpath(@__DIR__, "test_utils.jl"))

function _warmed_diagnostic_allocations(diagnostic, state)
    diagnostic_value(diagnostic, state)
    return @allocated diagnostic_value(diagnostic, state)
end

@testset "equation-system-independent simulation state" begin
    state = SimulationState(nothing, nothing, (;), 0.0, 0)
    @test isnothing(state.interface)
    @test isnothing(state.geometry)
    @test isnothing(state.workspace)
    @test_throws ArgumentError Diagnostic(:name, :measure)
end

@testset "simulation-owned diagnostics" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    physics = SinglePhase(density = 2.0, dynamic_viscosity = 0.1)
    evaluations = Ref(0)
    flow = FlowSimulation(
        mesh, physics;
        velocity = (2.0, 3.0), dt = 1.0e-3,
        diagnostics = (
            KineticEnergy(), PotentialEnergy(), SurfaceEnergy(), TotalEnergy(),
            LinearMomentum(), DivergenceNorm(),
            MaximumVelocity(), CFLTimeStep(), LeafBlockCount(), CellCount(),
            Diagnostic(:reported_step, state -> state.step) => EveryNSteps(2),
            Diagnostic(:complex_value, state -> complex(state.time, state.step)),
            Diagnostic(
                :vector_value,
                state -> SVector(state.time, typeof(state.time)(state.step)),
            ),
            Diagnostic(
                :evaluation_count, state -> begin
                    evaluations[] += 1
                    state.time
                end
            ),
        ),
    )

    @test diagnostic_names(flow) == [
        :kinetic_energy, :potential_energy, :surface_energy, :total_energy,
        :linear_momentum, :divergence_norm,
        :maximum_velocity, :cfl_timestep, :leaf_blocks, :cells,
        :reported_step, :complex_value, :vector_value, :evaluation_count,
    ]
    @test only(diagnostic_history(flow, :kinetic_energy).values) ≈ 13.0
    @test only(diagnostic_history(flow, :potential_energy).values) ≈ 0.0
    @test only(diagnostic_history(flow, :surface_energy).values) ≈ 0.0
    @test only(diagnostic_history(flow, :total_energy).values) ≈ 13.0
    @test only(diagnostic_history(flow, :linear_momentum).values) ≈ [4.0, 6.0]
    @test only(diagnostic_history(flow, :divergence_norm).values) ≈ 0.0
    @test only(diagnostic_history(flow, :maximum_velocity).values) ≈ 3.0
    @test only(diagnostic_history(flow, :cfl_timestep).values) ≈
        stable_timestep(flow)
    @test only(diagnostic_history(flow, :leaf_blocks).values) == nblocks(mesh)
    @test only(diagnostic_history(flow, :cells).values) == ncells(mesh)
    @test eltype(diagnostic_history(flow, :complex_value).values) ==
        ComplexF64
    @test eltype(diagnostic_history(flow, :vector_value).values) ==
        SVector{2, Float64}
    @test simulation_state(flow).step == 0

    fields = VisualizationFieldRegistry()
    register_field!(fields, :pressure, flow.pressure)
    register_diagnostic!(fields, flow, :kinetic_energy)
    register_diagnostic!(fields, flow, :evaluation_count)
    register_diagnostic!(
        fields, flow, :linear_momentum;
        name = :momentum_x, component = 1,
    )
    session = VisualizationSession(
        fields; viewer = NoVisualization(), diagnostic_every = 1,
    )
    result = simulate!(flow, 3.0e-3; visualization = session)
    @test result.total_steps == 3
    @test evaluations[] == 4
    @test diagnostic_history(flow, :reported_step).steps == [0, 2, 3]
    @test diagnostic_history(flow, :kinetic_energy).steps == [0, 1, 2, 3]
    @test all(
        sample[:kinetic_energy] ≈ 13.0 && sample[:momentum_x] ≈ 4.0
            for sample in visualization_diagnostics(session)
    )

    report = IOBuffer()
    @test isnothing(print_simulation_summary(report, flow, result))

    add_diagnostic!(flow, :time, state -> state.time)
    @test only(diagnostic_history(flow, :time).values) == flow.time
    @test_throws ArgumentError add_diagnostic!(flow, KineticEnergy())
end

@testset "one- and two-velocity physical diagnostics" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), periodic = (false, false),
    )
    interface = InterfaceField(mesh, (x, y) -> x - 0.5)
    physics = TwoPhase(
        interface; density = (4.0, 1.0),
        dynamic_viscosity = (0.1, 0.1),
        surface_tension = 0.25, gravity = (-2.0, 0.0),
    )
    shared = (x, y, time, axis) -> axis == 1 ? 2.0 : -1.0
    requested = (
        KineticEnergy(), PotentialEnergy(), SurfaceEnergy(), TotalEnergy(),
        LinearMomentum(), PhaseVolume(), PhaseMass(), InterfaceMeasure(),
        MaximumVelocity(),
    )
    one = FlowSimulation(
        mesh, physics; velocity_model = OneVelocityModel(),
        velocity = shared, diagnostics = requested,
    )
    two = FlowSimulation(
        mesh, physics; velocity_model = TwoVelocityModel(),
        velocity = shared, diagnostics = requested,
    )
    for flow in (one, two)
        state = simulation_state(flow)
        for diagnostic in (
                KineticEnergy(), LinearMomentum(), PhaseVolume(),
                DivergenceNorm(), LeafBlockCount(),
            )
            @test _warmed_diagnostic_allocations(diagnostic, state) == 0
        end
    end

    @test only(diagnostic_history(one, :kinetic_energy).values) ≈
        only(diagnostic_history(two, :kinetic_energy).values)
    @test only(diagnostic_history(one, :linear_momentum).values) ≈
        only(diagnostic_history(two, :linear_momentum).values)
    @test only(diagnostic_history(one, :potential_energy).values) ≈ 1.75
    @test only(diagnostic_history(two, :potential_energy).values) ≈ 1.75
    @test only(diagnostic_history(one, :surface_energy).values) ≈ 0.25
    @test only(diagnostic_history(two, :surface_energy).values) ≈ 0.25
    @test only(diagnostic_history(one, :total_energy).values) ≈
        only(diagnostic_history(one, :kinetic_energy).values) + 2.0
    @test only(diagnostic_history(two, :total_energy).values) ≈
        only(diagnostic_history(two, :kinetic_energy).values) + 2.0
    volumes = only(diagnostic_history(two, :phase_volume).values)
    @test volumes[1] ≈ 0.5
    @test volumes[2] ≈ 0.5
    @test only(diagnostic_history(two, :phase_mass).values) ≈ [2.0, 0.5]
    @test only(diagnostic_history(two, :interface_measure).values) ≈ 1.0
    @test only(diagnostic_history(two, :maximum_velocity).values) ≈ 2.0

    distinct = FlowSimulation(
        mesh, physics; velocity_model = TwoVelocityModel(),
        velocity = (
            (x, y, time, axis) -> axis == 1 ? 2.0 : -1.0,
            (x, y, time, axis) -> axis == 1 ? -1.0 : 3.0,
        ),
        diagnostics = (KineticEnergy(), LinearMomentum(), MaximumVelocity()),
    )
    # The discrete momentum inner product owns interior staggered unknowns;
    # prescribed physical-boundary faces are not degrees of freedom. On this
    # four-cell nonperiodic grid their dual volumes sum to 3/4 per component.
    @test only(diagnostic_history(distinct, :kinetic_energy).values) ≈ 5.625
    @test only(diagnostic_history(distinct, :linear_momentum).values) ≈
        [2.625, -0.375]
    @test only(diagnostic_history(distinct, :maximum_velocity).values) ≈ 3.0

    mixed_interface = InterfaceField(mesh, (x, y) -> x - 0.3)
    mixed_flow = FlowSimulation(
        mesh,
        TwoPhase(
            mixed_interface; density = (4.0, 1.0),
            dynamic_viscosity = (0.1, 0.1),
            surface_tension = 0.25, gravity = (-2.0, 0.0),
        );
        diagnostics = (PotentialEnergy(), SurfaceEnergy(), TotalEnergy()),
    )
    # The gravitational diagnostic integrates the first moment of each PLIC
    # phase polygon, rather than placing a mixed cell's mass at its centre.
    @test only(diagnostic_history(mixed_flow, :potential_energy).values) ≈ 1.27
    @test only(diagnostic_history(mixed_flow, :surface_energy).values) ≈ 0.25
    @test only(diagnostic_history(mixed_flow, :total_energy).values) ≈ 1.52
end

@testset "embedded and adaptive diagnostic measures" begin
    mesh = BlockTreeMesh(block_size = (8, 8), max_level = 1)
    geometry = GeometryField(mesh, (x, y) -> x - 0.5; lipschitz = 1)
    @test geometry.face_dual_volumes ==
        _component_volumes_by_global_face(
            mesh, geometry.dual_volumes, eltype(mesh.origin);
            covered_value = 0.0,
        )
    flow = FlowSimulation(
        mesh, SinglePhase(density = 2.0, dynamic_viscosity = 0.1);
        geometry, velocity = (2.0, 3.0),
        diagnostics = (
            KineticEnergy(), LinearMomentum(), DivergenceNorm(),
        ),
    )
    @test only(diagnostic_history(flow, :kinetic_energy).values) ≈ 6.5
    @test only(diagnostic_history(flow, :linear_momentum).values) ≈ [2.0, 3.0]
    divergence = CellField(mesh)
    divergence!(divergence, flow.velocity, geometry)
    expected_divergence = sqrt(
        sum(eachindex(mesh.blocks)) do block
            sum(
                abs2.(interior(divergence, block)) .*
                    interior(geometry.open_fraction, block),
            ) * mesh.blocks[block].volume
        end
    )
    @test only(diagnostic_history(flow, :divergence_norm).values) ≈
        expected_divergence

    cut_interface = InterfaceField(
        mesh, (x, y) -> y - 0.5; geometry, lipschitz = 1,
    )
    cut_flow = FlowSimulation(
        mesh,
        TwoPhase(
            cut_interface; density = (4.0, 1.0),
            dynamic_viscosity = (0.1, 0.1),
        );
        geometry,
        diagnostics = (PhaseVolume(), PhaseMass(), InterfaceMeasure()),
    )
    @test only(diagnostic_history(cut_flow, :phase_volume).values) ≈
        [0.25, 0.25]
    @test only(diagnostic_history(cut_flow, :phase_mass).values) ≈ [1.0, 0.25]
    @test only(diagnostic_history(cut_flow, :interface_measure).values) ≈ 1.0
    stable_timestep(cut_flow)
    @test test_allocated_call(() -> stable_timestep(cut_flow)) == 0
    if VERSION >= v"1.12"
        @test @allocated(stable_timestep(cut_flow)) == 0
    end
    cut_two = FlowSimulation(
        mesh,
        TwoPhase(
            cut_interface; density = (4.0, 1.0),
            dynamic_viscosity = (0.1, 0.1),
        );
        geometry,
        velocity_model = TwoVelocityModel(
            pressure_projection = CMDGFMProjection(),
        ),
        diagnostics = (),
    )
    stable_timestep(cut_two)
    @test test_allocated_call(() -> stable_timestep(cut_two)) == 0
    if VERSION >= v"1.12"
        @test @allocated(stable_timestep(cut_two)) == 0
    end

    energy_mesh = BlockTreeMesh(
        block_size = (4, 4), periodic = (false, false),
    )
    energy_geometry = GeometryField(
        energy_mesh, (x, y) -> x - 0.5; lipschitz = 1,
    )
    energy_interface = InterfaceField(
        energy_mesh, (x, y) -> y - 0.5;
        geometry = energy_geometry, lipschitz = 1,
    )
    energy_flow = FlowSimulation(
        energy_mesh,
        TwoPhase(
            energy_interface; density = (4.0, 1.0),
            dynamic_viscosity = (0.1, 0.1),
            surface_tension = 0.25, gravity = (0.0, -2.0),
        );
        geometry = energy_geometry,
        diagnostics = (PotentialEnergy(), SurfaceEnergy(), TotalEnergy()),
    )
    @test only(diagnostic_history(energy_flow, :potential_energy).values) ≈
        0.875
    @test only(diagnostic_history(energy_flow, :surface_energy).values) ≈ 0.125
    @test only(diagnostic_history(energy_flow, :total_energy).values) ≈ 1.0

    coarse_mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 1,
        periodic = (true, true),
    )
    coarse_interface = InterfaceField(
        coarse_mesh, (x, y) -> x - 0.5; lipschitz = 1,
    )
    refined_mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 1,
        periodic = (true, true),
    )
    refined_interface = InterfaceField(
        refined_mesh, (x, y) -> x - 0.5; lipschitz = 1,
    )
    adaptation = AMRWorkspace(refined_mesh, refined_interface)
    @test adapt!(adaptation, [:refine]; step = 1, force = true).changed
    phase(mesh, interface) = FlowSimulation(
        mesh,
        TwoPhase(
            interface; density = (4.0, 1.0),
            dynamic_viscosity = (0.1, 0.1),
        );
        velocity = (2.0, -1.0),
        diagnostics = (
            KineticEnergy(), LinearMomentum(), PhaseVolume(), PhaseMass(),
            InterfaceMeasure(),
        ),
    )
    coarse = phase(coarse_mesh, coarse_interface)
    refined = phase(refined_mesh, refined_interface)
    for name in (
            :kinetic_energy, :linear_momentum, :phase_volume, :phase_mass,
            :interface_measure,
        )
        @test only(diagnostic_history(refined, name).values) ≈
            only(diagnostic_history(coarse, name).values) atol = 2.0e-13
    end
end

@testset "diagnostic time schedules resume without duplicate boundaries" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    flow = FlowSimulation(
        mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.1);
        dt = 1.0e-3,
        diagnostics = (
            Diagnostic(:scheduled_time, state -> state.time) =>
                EveryTimeInterval(1.5e-3),
        ),
    )
    simulate!(flow, 3.0e-3)
    simulate!(flow, 5.0e-3)
    history = diagnostic_history(flow, :scheduled_time)
    @test count(==(3.0e-3), history.times) == 1
    @test length(unique(collect(zip(history.times, history.steps)))) ==
        length(history)
end
