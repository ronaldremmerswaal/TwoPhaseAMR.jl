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
    mesh = run.flow.system.mesh
    @test any(block -> block.key.level == 2, mesh.blocks)
    @test mesh.adaptation.last_update_step > 2
    test_finite_example_fields(run.flow)
    # The force diagnostics sum over cut cells collected once per geometry
    # revision. Every topology change must bump the revision, and after every
    # step, regrids included, both forces equal a scan of every cell bit for
    # bit.
    geometry = run.flow.system.geometry
    function scanned_force(axis)
        force = 0.0
        nx, ny = mesh.block_size
        for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
            ref = CellRef(block, i, j)
            segment = TwoPhaseAMR._local_plic_segment(geometry.solid, ref)
            isnothing(segment) && continue
            spacing = mesh.blocks[block].spacing
            length = hypot(
                (segment[2][1] - segment[1][1]) * spacing[1],
                (segment[2][2] - segment[1][2]) * spacing[2],
            )
            normal = (
                TwoPhaseAMR._get(geometry.solid.normal_x, ref) / spacing[1],
                TwoPhaseAMR._get(geometry.solid.normal_y, ref) / spacing[2],
            )
            scale = hypot(normal...)
            iszero(scale) && continue
            force -= TwoPhaseAMR._get(run.flow.pressure, ref) * normal[axis] / scale * length
        end
        return 2force
    end
    topology_changes = 0
    for _ in 1:12
        block_keys, revision = [block.key for block in mesh.blocks], geometry.revision
        step!(run.flow, 1.0e-4)
        if block_keys != [block.key for block in mesh.blocks]
            topology_changes += 1
            @test geometry.revision != revision
        end
        @test last(diagnostic_history(run.flow, :pressure_drag).values) === scanned_force(1)
        @test last(diagnostic_history(run.flow, :pressure_lift).values) === scanned_force(2)
    end
    @test topology_changes > 0
end
