using Test, TwoPhaseAMR

include(joinpath(@__DIR__, "..", "examples", "vortex_reverse.jl"))
include(joinpath(@__DIR__, "..", "examples", "capillary_wave.jl"))
include(joinpath(@__DIR__, "..", "examples", "sessile_droplet.jl"))

@testset "Visualized two-phase examples" begin
    vortex = visualize_vortex_reverse(
        base_cells = 8, max_level = 0, interface_level = 0,
        adaptive = false, period = 0.01,
        checkpoints = 2, live_dt = 0.01,
        viewer = NoVisualization(), asynchronous = false
    )
    @test vortex.mesh.block_size == (8, 8)
    @test vortex.curvature_indicator isa CurvatureRefinementIndicator
    @test field_names(vortex.fields) == [
        :volume_fraction, :curvature, :liquid_scalar, :gas_scalar,
        :liquid_staggered_scalar, :gas_staggered_scalar,
        :velocity_x, :velocity_y,
        :velocity_magnitude, :velocity_vorticity,
    ]
    vortex_result = wait_for_vortex_reverse!(vortex)
    @test vortex_result.time == vortex.period
    @test !isnothing(vortex_result.reverse_error)
    @test !isnothing(vortex_result.scalar_reverse_error)
    @test vortex_result.volume_error <= 128eps()
    @test vortex_result.unit_error <= 128eps()
    @test simulation_state(vortex.session) == :completed

    wave = visualize_capillary_wave(
        base_cells = 4, max_level = 0, interface_level = 0,
        final_time = 1.0e-4, dt = 1.0e-4,
        checkpoints = 2, live_dt = 1.0e-4,
        viewer = NoVisualization(), asynchronous = false
    )
    @test field_names(wave.fields) == [
        :volume_fraction, :velocity_x, :velocity_y,
        :velocity_magnitude, :velocity_vorticity, :pressure,
    ]
    @test diagnostic_names(wave.fields) == [:normalized_amplitude]
    wave_result = wait_for_capillary_wave!(wave)
    @test wave_result.time == 1.0e-4
    @test wave_result.pressure.converged
    @test simulation_state(wave.session) == :completed

    droplet = visualize_sessile_droplet(
        base_cells = 4, max_level = 0, interface_level = 0,
        final_time = 1.0e-4, dt = 1.0e-4,
        checkpoints = 2, live_dt = 1.0e-4,
        viewer = NoVisualization(), asynchronous = false
    )
    @test field_names(droplet.fields) == [
        :volume_fraction, :velocity_x, :velocity_y,
        :velocity_magnitude, :velocity_vorticity, :pressure,
    ]
    @test diagnostic_names(droplet.fields) == [:volume_fraction_error]
    @test !isapprox(
        droplet.initial_height, droplet.equilibrium_height;
        rtol = 1.0e-3
    )
    droplet_result = wait_for_sessile_droplet!(droplet)
    @test droplet_result.time == 1.0e-4
    @test droplet_result.pressure.converged
    @test simulation_state(droplet.session) == :completed
end
