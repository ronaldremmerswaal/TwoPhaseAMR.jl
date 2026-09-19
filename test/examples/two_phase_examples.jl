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
            (OneVelocityModel(), TwoVelocityModel())
        for model in models
            options = isnothing(model) ? (;) : (; velocity_model = model)
            run = call_example(
                example, :run_example;
                max_level = 0, final_time = 1.0e-6, options...,
            )
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
            end
            @test diagnostic_names(run.flow) == expected_diagnostics
            @test all(name -> length(diagnostic_history(run.flow, name)) == 2,
                diagnostic_names(run.flow))
        end
    end
end
