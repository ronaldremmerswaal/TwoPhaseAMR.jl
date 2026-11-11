using Test, TwoPhaseAMR
include("helpers.jl")

@testset "Absent intrinsic velocities are NaN, not dormant data" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 1, periodic = (false, false),
        adaptation = MeshAdaptationOptions(coarsen_delay = 0)
    )
    interface = InterfaceField(mesh, (x, y) -> x + 0.2y - 0.49)
    velocity = (
        FaceVelocity(mesh, (x, y, t, a) -> a == 1 ? 2.0 : -1.0),
        FaceVelocity(mesh, (x, y, t, a) -> a == 1 ? -3.0 : 0.5),
    )
    TwoPhaseAMR._mark_inactive_phase_velocities!(velocity, interface)
    @test _phase_storage_contract(velocity, interface)
    @test any(isnan, velocity[1].interior)
    @test any(isnan, velocity[2].interior)

    # This also exercises the ordinary tuple + interface registration API.
    workspace = AMRWorkspace(mesh, velocity, interface)
    @test length(workspace.phase_velocities) == 1
    @test adapt!(workspace, [:refine]; step = 1, force = true).changed
    @test _phase_storage_contract(velocity, interface)
    for phase in 1:2, index in _phase_interior_indices(interface, phase)
        expected = phase == 1 ? (mesh.faces[index].axis == 1 ? 2.0 : -1.0) :
            (mesh.faces[index].axis == 1 ? -3.0 : 0.5)
        @test velocity[phase].interior[index] ≈ expected atol = 128eps()
    end
    @test adapt!(workspace, fill(:coarsen, nblocks(mesh)); step = 2, force = true).changed
    @test _phase_storage_contract(velocity, interface)

    index = first(_phase_interior_indices(interface, 1))
    velocity[1].interior[index] = NaN
    @test_throws ArgumentError TwoPhaseAMR._mark_inactive_phase_velocities!(velocity, interface)
end

@testset "Phase curl extends valid data and ignores absent-phase values" begin
    mesh = BlockTreeMesh(
        block_size = (12, 12), max_level = 1, periodic = (false, false),
        refine = (key, origin, extent) -> key.level == 0
    )
    interface = InterfaceField(mesh, (x, y) -> y - 0.47 - 0.1x)
    affine = (x, y, t, axis) -> axis == 1 ? 0.2 + 0.3x - 0.7y : -0.4 + 0.8x + 0.1y
    velocity = (FaceVelocity(mesh, affine), FaceVelocity(mesh, affine))
    TwoPhaseAMR._mark_inactive_phase_velocities!(velocity, interface)
    before = map(field -> copy(field.interior), velocity)
    for (phase, name) in ((1, :liquid), (2, :gas))
        curl = visualization_values(velocity[phase], Vorticity(); interface, phase = name)
        refs = TwoPhaseAMR._cell_references(mesh)
        for (index, cell) in pairs(refs)
            if TwoPhaseAMR._phase_cell_is_active(interface, cell, name)
                @test curl[index] ≈ 1.5 atol = 2.0e-12
            else
                @test isnan(curl[index])
            end
        end
        @test isequal(velocity[phase].interior, before[phase])
        # Even a caller's stale finite extension cannot influence this curl.
        poisoned = copy(velocity[phase])
        for index in eachindex(poisoned.interior)
            isnan(poisoned.interior[index]) && (poisoned.interior[index] = 1.0e9index)
        end
        @test isequal(curl, visualization_values(poisoned, Vorticity(); interface, phase = name))
    end

    resting = (FaceVelocity(mesh), FaceVelocity(mesh))
    TwoPhaseAMR._mark_inactive_phase_velocities!(resting, interface)
    criterion = VorticityRefinementIndicator(resting[1]; refine_tolerance = 0.2)
    workspace = AMRWorkspace(mesh, resting, interface)
    # Automatic AMR association supplies the liquid mask to an existing
    # criterion; every zero-curl block, including pure gas, may coarsen.
    @test all(==(:coarsen), refinement_labels!(workspace, criterion))
    @test adapt!(workspace, criterion; step = 1, force = true).changed
    @test _phase_storage_contract(resting, interface)
end

@testset "NaN phase storage survives transport, diffusion and projection" begin
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, false))
    interface = InterfaceField(mesh, (x, y) -> y - 0.47)
    flow = FlowSimulation(
        mesh, TwoPhase(
            interface; density = (1000.0, 1.0),
            dynamic_viscosity = (0.02, 0.001)
        );
        velocity_model = TwoVelocityModel(),
        velocity = (
            (x, y, t, a) -> a == 1 ? 0.05 : 0.0,
            (x, y, t, a) -> a == 1 ? -0.02 : 0.0,
        ), boundary = FreeSlip()
    )
    @test _phase_storage_contract(flow.velocity, interface)
    @test isfinite(stable_timestep(flow))
    @test simulate!(flow, 0.001; dt = 0.001).pressure.converged
    @test _phase_storage_contract(flow.velocity, interface)
    state = simulation_state(flow)
    @test isfinite(diagnostic_value(KineticEnergy(), state))
    @test all(isfinite, diagnostic_value(LinearMomentum(), state))
    @test isfinite(diagnostic_value(DivergenceNorm(), state))
    @test isfinite(diagnostic_value(MaximumVelocity(), state))
end
