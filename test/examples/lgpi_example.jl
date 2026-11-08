using Test, TwoPhaseAMR, LinearAlgebra

@testset "Standalone LGPI initial condition" begin
    example = load_example(:lgpi)
    initial = call_example(example, :lgpi_initial_condition)
    @test initial.domain == (4.0, 2.4)
    @test initial.interface_levelset(3.1, 1.52) == 0.0
    @test initial.bathymetry_levelset(3.6, 0.0) ≈ 0.0 atol = 8eps()
end

@testset "Adaptive LGPI example" begin
    example = load_example(:lgpi)
    # With adaptive timesteps this reaches the first vorticity-driven regrid.
    run = call_example(
        example, :run_example;
        max_level = 2, final_time = 0.105,
        visualize = false,
    )
    @test run.result.time ≈ 0.105
    @test run.result.steps >= 5
    @test run.result.mesh_updates > 0
    @test run.result.pressure.converged
    mesh = run.flow.solver.mesh
    @test any(block -> block.key.level == 2, mesh.blocks)
    @test mesh.adaptation.last_update_step > 2
    test_finite_example_fields(run.flow)
    test_phase_volume_conservation(run.flow)
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
        operator, density, dt; coarse_size = 8,
    )

    @test adapt!(
        adaptation, fill(:coarsen, nblocks(mesh)); step = 20, force = true,
    ).changed
    operator = TwoPhaseAMR._staggered_diffusion_operator(
        interface; viscosity, boundary, construction_cache = cache,
        reuse = operator,
    )
    density = TwoPhaseAMR.mixture_density(interface, (2.0, 1.0))
    reused = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        operator, density, dt; coarse_size = 8,
        reuse_preconditioner = old.preconditioner, reuse_solver = old,
    )
    fresh_operator = StaggeredDiffusionOperator(interface; viscosity, boundary)
    fresh = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        fresh_operator, density, dt; coarse_size = 8,
    )
    rhs = [sin(i / 3) for i in eachindex(density)]
    reused_result, fresh_result = similar(rhs), similar(rhs)
    ldiv!(reused_result, reused.preconditioner, rhs)
    ldiv!(fresh_result, fresh.preconditioner, rhs)
    # Regridding storage may be reused, but applying the resulting hierarchy
    # must agree with constructing it afresh on the new face schedule.
    @test reused_result ≈ fresh_result rtol = 1.0e-12 atol = 1.0e-12
end
