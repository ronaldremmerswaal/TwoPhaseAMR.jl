using Test, TwoPhaseAMR

include(joinpath(@__DIR__, "..", "examples", "vortex_reverse.jl"))
include(joinpath(@__DIR__, "..", "examples", "capillary_wave.jl"))
include(joinpath(@__DIR__, "..", "examples", "sessile_droplet.jl"))
include(joinpath(
    @__DIR__, "..", "examples", "sloped_sessile_droplet.jl",
))

@testset "Visualized two-phase examples" begin
    vortex_defaults = vortex_reverse_setup(adaptive = false)
    @test vortex_defaults.mesh.block_size == (8, 8)
    @test vortex_defaults.mesh.max_level == 5
    @test vortex_defaults.period == 8.0
    @test vortex_defaults.interface_indicator.level == 1
    @test vortex_defaults.curvature_indicator isa CurvatureRefinementIndicator

    transfer_case = vortex_reverse_setup(
        base_cells = 4, max_level = 1, interface_level = 0,
        adaptive = false,
    )
    old_volume = integrate(transfer_case.interface)
    @test adapt!(
        transfer_case.adaptation, [:refine]; step = 1, force = true,
    ).changed
    @test integrate(transfer_case.interface) ≈ old_volume atol = 512eps()
    @test all(isfinite, transfer_case.staggered_scalar.liquid.interior)

    vortex = visualize_vortex_reverse(
        base_cells = 8, max_level = 0, interface_level = 0,
        adaptive = false, period = 0.01,
        checkpoints = 2, live_dt = 0.01,
        viewer = NoVisualization(), asynchronous = false
    )
    @test vortex.mesh.block_size == (8, 8)
    @test vortex.curvature_indicator isa CurvatureRefinementIndicator
    @test diagnostic_names(vortex.fields) == [:block_count]
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
    @test all(
        sample.values[:block_count] == nblocks(vortex.mesh)
            for sample in visualization_diagnostics(vortex.session)
    )

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
    @test diagnostic_names(wave.fields) == [
        :amplitude_magnitude, :analytical_amplitude_magnitude,
    ]
    wave_result = wait_for_capillary_wave!(wave)
    @test wave_result.time == 1.0e-4
    @test wave_result.pressure.converged
    @test simulation_state(wave.session) == :completed
    @test capillary_linear_amplitude(
        0.0;
        liquid_density = wave.density[1],
        gas_density = wave.density[2],
        kinematic_viscosity = wave.kinematic_viscosity,
        surface_tension = wave.surface_tension,
        wavenumber = wave.wavenumber
    ) ≈ 1.0 atol = 128eps()

    droplet = visualize_sessile_droplet(
        base_cells = 12, max_level = 0, interface_level = 0,
        final_time = 1.0e-4, dt = 1.0e-4,
        checkpoints = 2, live_dt = 1.0e-4,
        viewer = NoVisualization(), asynchronous = false
    )
    @test field_names(droplet.fields) == [
        :volume_fraction, :velocity_x, :velocity_y,
        :velocity_magnitude, :velocity_vorticity, :pressure,
    ]
    @test diagnostic_names(droplet.fields) == [:kinetic_energy]
    @test droplet.indicators[2] isa CurvatureRefinementIndicator
    @test !isapprox(
        droplet.initial_height, droplet.equilibrium_height;
        rtol = 1.0e-3
    )
    droplet_result = wait_for_sessile_droplet!(droplet)
    @test droplet_result.time == 1.0e-4
    @test droplet_result.pressure.converged
    @test simulation_state(droplet.session) == :completed
    @test all(
        sample.values[:kinetic_energy] >= 0
            for sample in visualization_diagnostics(droplet.session)
    )

    sloped = visualize_sloped_sessile_droplet(
        base_cells = 4, max_level = 1,
        interface_level = 1, geometry_level = 1,
        cells_per_radius = 4.0, adaptation_interval = 1,
        final_time = 1.0e-5, checkpoints = 2,
        live_dt = 1.0e-5, viewer = NoVisualization(),
    )
    @test nblocks(sloped.mesh) > 1
    @test field_names(sloped.fields) == [
        :volume_fraction, :velocity_x, :velocity_y,
        :velocity_magnitude, :velocity_vorticity, :pressure,
    ]
    @test sloped.fields.interfaces == [sloped.interface]
    @test sloped.fields.geometries == [sloped.geometry]
    @test diagnostic_names(sloped.fields) == [:kinetic_energy]
    @test sloped.result.time == 1.0e-5
    @test sloped.result.pressure.converged
    @test _open_phase_volume_example(
        sloped.interface, sloped.geometry,
    ) ≈ sloped.initial_volume atol = 2048eps()
    @test all(
        sample.values[:kinetic_energy] >= 0
            for sample in visualization_diagnostics(sloped.session)
    )
end
