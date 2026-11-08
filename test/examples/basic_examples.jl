using Test, TwoPhaseAMR

@testset "Standalone single-phase examples" begin
    for name in (:taylor_green, :lid_driven_cavity, :decaying_turbulence)
        example = load_example(name)
        run = call_example(
            example, :run_example; max_level = 0, final_time = 1.0e-6,
        )
        @test run.flow.velocity isa FaceVelocity
        @test run.result.time == 1.0e-6
        # A cavity at rest has an infinite stability bound, so simulate! caps
        # its first step at 1/128 of the interval; the others take one step.
        @test run.result.steps == (name == :lid_driven_cavity ? 128 : 1)
        @test run.result.pressure.converged
        if name == :lid_driven_cavity
            @test length(keyframes(snapshot_store(run.recorder))) >= 2
        else
            @test length(keyframes(snapshot_store(run.recorder))) == 2
        end
        @test :kinetic_energy in diagnostic_names(run.flow)
    end
end

@testset "Standalone VOF transport example" begin
    example = load_example(:vortex_reverse)
    for reconstruction_method in (:lhf, :lhf_ghf, :wy)
        run = call_example(
            example, :run_example; max_level = 0, final_time = 1.0e-3,
            reconstruction_method,
        )
        @test run.interface.reconstruction_method == reconstruction_method
        @test run.result.time ≈ 1.0e-3
        @test run.result.steps == 1
        @test !isempty(keyframes(snapshot_store(run.recorder)))
        @test run.result.volume_error <= 128eps()
    end
end

@testset "Adaptive decaying turbulence example" begin
    example = load_example(:decaying_turbulence)
    # Levels through 2 are protected by the example's minimum_level, so use
    # level 3 to exercise its pressure-driven coarsening criterion.
    run = call_example(
        example, :run_example;
        max_level = 3, final_time = 8.0e-4, dt = 1.0e-4,
    )
    @test run.result.time ≈ 8.0e-4
    @test run.result.steps == 8
    @test run.result.pressure.converged
    @test run.result.mesh_updates > 0
    test_finite_example_fields(run.flow)
end

@testset "Adaptive vortex-reversal example" begin
    example = load_example(:vortex_reverse)
    # The smaller grid needs this much transport to cross a refinement
    # boundary and exercise an actual mesh update.
    run = call_example(
        example, :run_example;
        max_level = 2, final_time = 0.6,
    )
    @test run.result.time ≈ 0.6
    @test run.result.steps >= 5
    @test run.result.mesh_updates > 0
    @test all(isfinite, run.velocity.interior)
    @test all(isfinite, run.velocity.boundary)
    @test all(
        isfinite(value)
            for block in eachindex(run.interface.mesh.blocks)
            for value in interior(run.interface, block)
    )
    @test run.result.volume_error <=
        8 * run.interface.fraction_floor * prod(run.interface.mesh.extent)
end

@testset "Manufactured Poisson example" begin
    example = load_example(:manufactured_poisson)
    for method in (:mimetic, :joint_face, :cmdgfm)
        run = call_example(
            example, :run_example; resolution = 8, method,
        )
        @test run.method == method
        @test run.density[1] / run.density[2] == 1000
        @test 0.8 <= run.analytical_jump_components.rms_ratio <= 1.25
        @test run.stats.converged
        @test field_names(run.registry)[1] == :pressure
        @test all(isfinite, values(run.errors.pressure))
        @test all(isfinite, values(run.errors.gradient))
        @test all(isfinite, values(run.errors.xi.x))
        @test all(isfinite, values(run.errors.xi.y))
        @test run.errors.pressure.l2 <= run.errors.pressure.linf
        @test run.errors.gradient.l2 <= run.errors.gradient.linf
        @test run.errors.xi.x.l2 <= run.errors.xi.x.linf
        @test run.errors.xi.y.l2 <= run.errors.xi.y.linf
    end
end
