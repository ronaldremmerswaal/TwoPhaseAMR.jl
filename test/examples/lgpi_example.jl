using Test, TwoPhaseAMR, LinearAlgebra

@testset "Standalone LGPI initial condition" begin
    example = load_example(:lgpi)
    initial = call_example(example, :lgpi_initial_condition)
    @test initial.domain == (1.0, 0.6)
    @test initial.interface_levelset(0.775, 0.38) == 0.0
    @test initial.bathymetry_levelset(0.9, 0.0) ≈ 0.0 atol = 8eps()
end

@testset "Adaptive LGPI example" begin
    example = load_example(:lgpi)
    # Run past the first five-step adaptation check with the corrected
    # hydrostatic force, which permits larger adaptive timesteps.
    run = call_example(
        example, :run_example;
        max_level = 2, final_time = 0.2, refinement_ymin = -Inf,
    )
    @test run.flow.velocity isa TwoPhaseVelocity
    @test run.flow.system.two_phase.interface_transport isa FrozenInterfaceFluxes
    @test run.flow.system.two_phase.pressure_projection isa CMDGFMProjection
    @test run.flow.system.adaptivity.criteria[2].field === run.flow.velocity[1]
    @test run.result.time ≈ 0.2
    @test run.result.steps >= 5
    store = snapshot_store(run.recorder)
    frame = store[last(keyframes(store))]
    vorticity = frame[:velocity_vorticity]
    @test vorticity isa TwoPhaseAMR.TwoPhaseVisualizationValues
    @test any(isfinite, vorticity.liquid)
    @test any(isfinite, vorticity.gas)
    @test run.result.mesh_updates > 0
    @test run.result.pressure.converged
    mesh = run.flow.system.mesh
    @test any(block -> block.key.level == 2, mesh.blocks)
    @test mesh.adaptation.last_update_step > 2
    test_finite_example_fields(run.flow)
    @test all(
        0 <= value <= 1
            for block in eachindex(mesh.blocks)
            for value in interior(run.flow.interface, block)
    )
    test_phase_volume_conservation(run.flow)
end

@testset "LGPI solid-adjacent refinement restriction" begin
    example = load_example(:lgpi)
    initial = call_example(example, :lgpi_initial_condition)
    run = call_example(example, :run_example; max_level = 2, final_time = 0)
    touching = filter(
        block -> initial.bathymetry_levelset(block.origin...) <= 0,
        run.flow.system.mesh.blocks,
    )
    @test !isempty(touching)
    @test all(block -> block.key.level <= 1, touching)
    shared = call_example(
        example, :run_example;
        max_level = 0, final_time = 0.01, velocity_model = OneVelocityModel(),
    )
    @test shared.flow.velocity isa FaceVelocity
    @test shared.result.time ≈ 0.01
end

@testset "Variable diffusion hierarchy after coarsening" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 2, periodic = (false, false),
        refine = (key, origin, extent) -> key.level < 2,
    )
    interface = InterfaceField(
        mesh, (x, y) -> x + 0.2y - 0.47; lipschitz = 1.1,
    )
    adaptation = AMRWorkspace(mesh, interface)
    viscosity = (0.02, 0.001)
    boundary = FreeSlip()
    dt = 0.5
    cache = TwoPhaseAMR.VariableDiffusionConstructionCache(Float64)
    operator = TwoPhaseAMR._staggered_diffusion_operator(
        interface; viscosity, boundary, construction_cache = cache,
    )
    density = TwoPhaseAMR.mixture_density(interface, (2.0, 1.0))
    old = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        operator, density, dt; multigrid = MultigridOptions(coarse_size = 8),
    )
    old_coarse_mesh = old.preconditioner.operators[2].mesh
    old_mismatched = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        operator, density, dt; multigrid = MultigridOptions(coarse_size = 8),
    )
    mismatched_coarse_mesh = old_mismatched.preconditioner.operators[2].mesh

    @test adapt!(
        adaptation, fill(:coarsen, nblocks(mesh)); step = 20, force = true,
    ).changed
    # Both constructions now use the active level-one lattice, even though
    # the live mesh retains its level-two refinement ceiling.
    @test old_coarse_mesh.max_level == 1
    @test mesh.max_level == 2
    @test [block.key for block in old_coarse_mesh.blocks] ==
        [block.key for block in mesh.blocks]
    @test old_coarse_mesh.faces == mesh.faces
    @test TwoPhaseAMR._same_tree_topology(old_coarse_mesh, mesh)
    # A genuinely different face order must still prevent reuse of that level.
    @test TwoPhaseAMR._same_tree_topology(mismatched_coarse_mesh, mesh)
    mismatched_coarse_mesh.faces[1], mismatched_coarse_mesh.faces[2] =
        mismatched_coarse_mesh.faces[2], mismatched_coarse_mesh.faces[1]
    @test !TwoPhaseAMR._same_tree_topology(mismatched_coarse_mesh, mesh)
    operator = TwoPhaseAMR._staggered_diffusion_operator(
        interface; viscosity, boundary, construction_cache = cache,
        reuse = operator,
    )
    density = TwoPhaseAMR.mixture_density(interface, (2.0, 1.0))
    reused = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        operator, density, dt; multigrid = MultigridOptions(coarse_size = 8),
        reuse_preconditioner = old.preconditioner, reuse_solver = old,
    )
    @test reused.preconditioner.reused_levels > 0
    rejected = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        operator, density, dt; multigrid = MultigridOptions(coarse_size = 8),
        reuse_preconditioner = old_mismatched.preconditioner,
        reuse_solver = old_mismatched,
    )
    @test all(
        level_operator -> level_operator.mesh !== mismatched_coarse_mesh,
        rejected.preconditioner.operators,
    )
    fresh_operator = StaggeredDiffusionOperator(interface; viscosity, boundary)
    fresh = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        fresh_operator, density, dt; multigrid = MultigridOptions(coarse_size = 8),
    )
    rhs = [sin(i / 3) for i in eachindex(density)]
    reused_result, fresh_result = similar(rhs), similar(rhs)
    ldiv!(reused_result, reused.preconditioner, rhs)
    ldiv!(fresh_result, fresh.preconditioner, rhs)
    rejected_result = similar(rhs)
    ldiv!(rejected_result, rejected.preconditioner, rhs)
    # Regridding storage may be reused, but applying the resulting hierarchy
    # must agree with constructing it afresh on the new face schedule.
    @test reused_result ≈ fresh_result rtol = 1.0e-12 atol = 1.0e-12
    @test rejected_result ≈ fresh_result rtol = 1.0e-12 atol = 1.0e-12
end
