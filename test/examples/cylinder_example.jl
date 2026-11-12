using Test, TwoPhaseAMR

@testset "Standalone cylinder example" begin
    example = load_example(:cylinder_flow)
    run = call_example(
        example, :run_example;
        reynolds = 100, max_level = 0, final_time = 1.0e-6,
        start_studio = example_start_probe,
    )
    @test run.result.time == 1.0e-6
    @test isopen(run.app)
    close(run.app)
    @test run.result.pressure.converged
    @test !isnothing(run.flow.system.geometry)
    @test :pressure_drag in diagnostic_names(run.flow)
    pressure_drag = diagnostic_history(run.flow, :pressure_drag)
    # The example suppresses its impulsive-start force spike with a NaN at
    # the initial sample; require at least one finite completed-step sample.
    @test length(pressure_drag) >= 2
    @test first(pressure_drag.times) == 0
    @test isnan(first(pressure_drag.values))
    @test all(>(0), Iterators.drop(pressure_drag.steps, 1))
    @test all(isfinite, Iterators.drop(pressure_drag.values, 1))
end

@testset "Adaptive cylinder example" begin
    example = load_example(:cylinder_flow)
    # The example disables dynamic adaptation at level 1; level 2 enables
    # its geometry and vorticity criteria.
    run = call_example(
        example, :run_example;
        reynolds = 100, max_level = 2, final_time = 8.0e-4,
        dt = 1.0e-4,
    )
    @test run.result.time ≈ 8.0e-4
    @test run.result.steps == 8
    @test run.result.pressure.converged
    flow = run.flow
    mesh = flow.system.mesh
    geometry = flow.system.geometry
    @test any(block -> block.key.level == 2, mesh.blocks)
    @test mesh.adaptation.last_update_step > 2
    test_finite_example_fields(flow)

    # The force diagnostics sum over cut cells collected per geometry
    # revision. This scan of every cell repeats their order and expressions,
    # so it must reproduce them bit for bit.
    function scanned_force(pressure, axis)
        total = 0.0
        nx, ny = mesh.block_size
        for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
            ref = CellRef(block, i, j)
            segment = TwoPhaseAMR._local_plic_segment(geometry.solid, ref)
            isnothing(segment) && continue
            spacing = mesh.blocks[block].spacing
            segment_length = hypot(
                (segment[2][1] - segment[1][1]) * spacing[1],
                (segment[2][2] - segment[1][2]) * spacing[2],
            )
            normal = (
                TwoPhaseAMR._get(geometry.solid.normal_x, ref) / spacing[1],
                TwoPhaseAMR._get(geometry.solid.normal_y, ref) / spacing[2],
            )
            scale = hypot(normal...)
            iszero(scale) && continue
            total -= TwoPhaseAMR._get(pressure, ref) * normal[axis] / scale *
                segment_length
        end
        return 2total
    end
    for _ in 1:2
        step!(flow, 1.0e-4)
        @test last(diagnostic_history(flow, :pressure_drag).values) ===
            scanned_force(flow.pressure, 1)
        @test last(diagnostic_history(flow, :pressure_lift).values) ===
            scanned_force(flow.pressure, 2)
    end

    # Drive the example's cache through the adaptation path the example uses:
    # a forced adapt prolongs the solid and refreshes the geometry, which must
    # bump the version and refill the collected cut cells. Label one far-field
    # block for refinement so the topology change is deterministic.
    geometry_version(geometry) = TwoPhaseAMR._interface_geometry_revision(geometry)
    force = Base.invokelatest(getproperty(example, :CutCellPressureForce), geometry)
    pressure = CellField(mesh)
    initialize!(pressure, (x, y) -> sinpi(0.5x) * cospi(0.25y))
    @test force((; time = 1.0, pressure), 1) === scanned_force(pressure, 1)
    @test force((; time = 1.0, pressure), 2) === scanned_force(pressure, 2)
    revision = geometry_version(geometry)
    blocks = nblocks(mesh)
    labels = fill(:keep, blocks)
    labels[findfirst(block -> block.key.level < mesh.max_level, mesh.blocks)] =
        :refine
    stats = adapt!(
        AMRWorkspace(mesh, geometry), labels;
        step = mesh.adaptation.completed_steps + 1, force = true,
    )
    @test stats.changed
    @test nblocks(mesh) != blocks
    @test geometry_version(geometry) != revision
    pressure = CellField(mesh)
    initialize!(pressure, (x, y) -> cospi(0.5x) * sinpi(0.25y))
    @test force((; time = 1.0, pressure), 1) === scanned_force(pressure, 1)
    @test force((; time = 1.0, pressure), 2) === scanned_force(pressure, 2)

    # Re-initializing the solid changes the stored planes without refreshing
    # the geometry; the version must follow it, or the sums stay stale.
    revision = geometry_version(geometry)
    initialize!(geometry.solid, (1.0, 1.0), 0.2)
    @test geometry_version(geometry) != revision
    @test force((; time = 2.0, pressure), 1) === scanned_force(pressure, 1)
    @test force((; time = 2.0, pressure), 2) === scanned_force(pressure, 2)
    # The cache also refreshes on demand.
    refresh = getproperty(example, :refresh!)
    Base.invokelatest(refresh, force)
    @test force((; time = 2.0, pressure), 1) === scanned_force(pressure, 1)
end
