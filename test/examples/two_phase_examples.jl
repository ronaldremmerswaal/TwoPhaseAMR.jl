using Test, TwoPhaseAMR

const TWO_PHASE_EXAMPLES = (
    :capillary_wave,
    :sessile_droplet,
    :sloped_sessile_droplet,
    :stokes3_wave,
    :lgpi,
    :interfacial_instability,
)

@testset "Two-phase examples support both velocity models" begin
    for name in TWO_PHASE_EXAMPLES
        example = load_example(name)
        models = name === :interfacial_instability ?
            (nothing, OneVelocityModel()) :
            (
                OneVelocityModel(), TwoVelocityModel(),
                TwoVelocityModel(pressure_projection = MimeticProjection()),
            )
        for model in models
            options = isnothing(model) ? (;) : (; velocity_model = model)
            if name !== :lgpi
                options = (; options..., start_studio = example_start_probe)
            end
            run = call_example(
                example, :run_example;
                max_level = 0, final_time = 1.0e-6, options...,
            )
            if name !== :lgpi
                @test isopen(run.app)
                close(run.app)
            end
            expected = model isa OneVelocityModel ? FaceVelocity :
                TwoPhaseVelocity
            @test run.flow.velocity isa expected
            @test run.result.time == 1.0e-6
            @test run.result.steps == 1
            @test run.result.pressure.converged
            amplitude_example = name in (
                :capillary_wave, :interfacial_instability,
            )
            expected_diagnostics = amplitude_example ?
                [:amplitude, :analytical_amplitude] :
                [
                    :kinetic_energy, :linear_momentum, :phase_volume,
                    :divergence_norm, :leaf_blocks,
                ]
            if amplitude_example
                amplitude = diagnostic_history(run.flow, :amplitude)
                analytical = diagnostic_history(
                    run.flow, :analytical_amplitude,
                )
                @test length(amplitude) == 2
                @test all(isfinite, amplitude.values)
                @test length(analytical) == 2
                @test all(isfinite, analytical.values)
                @test first(analytical.values) ≈ 0.01
                if name === :capillary_wave
                    reference = 0.01 * Base.invokelatest(
                        example.prosperetti_amplitude,
                        1.0e-6;
                        liquid_density = 1.0, gas_density = 1.0e-3,
                        kinematic_viscosity = sqrt(1 / 3000),
                        gravity = 0.0, surface_tension = 1.0,
                        wavenumber = 2pi,
                    )
                    @test last(analytical.values) ≈ reference
                end
            end
            @test diagnostic_names(run.flow) == expected_diagnostics
            @test all(
                name -> length(diagnostic_history(run.flow, name)) == 2,
                diagnostic_names(run.flow)
            )
        end
    end
end

@testset "Two-phase AMR example smoke tests" begin
    for name in (
            :capillary_wave, :sessile_droplet, :sloped_sessile_droplet,
            :stokes3_wave, :interfacial_instability,
        )
        @testset "$name" begin
            example = load_example(name)
            # In particular, fixed-step interfacial instability starts from
            # nonzero raw wall-normal phase velocities before wall enforcement.
            run = call_example(
                example, :run_example;
                max_level = 1, final_time = 8.0e-4, dt = 1.0e-4,
            )
            @test run.result.time ≈ 8.0e-4
            @test run.result.steps == 8
            @test run.result.pressure.converged
            mesh = run.flow.system.mesh
            @test any(block -> block.key.level == 1, mesh.blocks)
            # Square cells on every level.
            @test all(
                block -> isapprox(block.spacing[1], block.spacing[2]; rtol = 8eps()),
                mesh.blocks
            )
            # Initial forced adaptation ends by step 2. Check that the
            # example also evaluated its criteria during physical steps.
            @test mesh.adaptation.last_update_step > 2
            test_finite_example_fields(run.flow)
            if name === :interfacial_instability
                @test all(
                    value -> isnan(value) || iszero(value),
                    Iterators.flatten(v.boundary for v in run.flow.velocity),
                )
            end
            if :phase_volume in diagnostic_names(run.flow)
                test_phase_volume_conservation(run.flow)
            end
            if :amplitude in diagnostic_names(run.flow)
                @test all(
                    isfinite, diagnostic_history(run.flow, :amplitude).values,
                )
            end
        end
    end
end

@testset "Sloped droplet AMR smoke test with the mimetic projection" begin
    # The embedded sloped wall cuts cells at the contact line and next to
    # finer blocks; the mimetic metric closes them with solid edges.
    example = load_example(:sloped_sessile_droplet)
    run = call_example(
        example, :run_example;
        max_level = 1, final_time = 8.0e-4, dt = 1.0e-4,
        velocity_model = TwoVelocityModel(pressure_projection = MimeticProjection()),
    )
    @test run.flow.system.two_phase.pressure_projection isa MimeticProjection
    @test run.result.time ≈ 8.0e-4
    @test run.result.steps == 8
    @test run.result.projection.converged
    # NUMERICS CONCERN: the moving contact line on the embedded wall yields cut
    # cells without an admissible pressure point (least-squares blocks): 8 in
    # these 8 steps, none in the static solid fixtures of the unit tests.
    @test_broken run.result.projection.defective_cells == 0
    @test run.result.projection.defective_cells <= 8
    @test run.result.projection.lumped_values == 0
    @test any(block -> block.key.level == 1, run.flow.system.mesh.blocks)
    @test run.flow.system.mesh.adaptation.last_update_step > 2
    test_finite_example_fields(run.flow)
    test_phase_volume_conservation(run.flow)
end

@testset "Sloped droplet transport through sliver cells" begin
    example = load_example(:sloped_sessile_droplet)
    # This angle creates open fractions below 1e-3 and previously failed in
    # the full default trajectory when division by open volume amplified
    # liquid-volume round-off beyond the conditional-fraction tolerance.
    run = call_example(
        example, :run_example;
        max_level = 2, wall_angle = 0.314,
    )
    @test run.result.steps > 1
    @test run.result.pressure.converged
    geometry = run.flow.system.geometry
    @test any(eachindex(geometry.mesh.blocks)) do block
        any(open -> 0 < open < 1.0e-3, interior(geometry.open_fraction, block))
    end
    @test all(eachindex(geometry.mesh.blocks)) do block
        fraction = interior(run.flow.interface.fraction, block)
        open = interior(geometry.open_fraction, block)
        all((0 .<= fraction) .& (fraction .<= open))
    end
    test_finite_example_fields(run.flow)
    test_phase_volume_conservation(run.flow)
end
