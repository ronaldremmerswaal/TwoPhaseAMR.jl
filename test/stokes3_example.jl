using Test, TwoPhaseAMR

include(joinpath(@__DIR__, "..", "examples", "stokes3_wave.jl"))

@testset "Stokes-3 example setup and AMR API" begin
    initial = stokes3_initial_condition()
    @test initial.levelset(0.0, initial.elevation(0.0)) ≈ 0.0 atol = eps()
    epsilon = 0.55
    k = 2pi
    chi = inv(tanh(k / 2))
    expected_crest = epsilon / k * (
        1 + epsilon * chi * (3chi^2 - 1) / 4 +
            epsilon^2 * (
            -3(chi^4 - 3chi^2 + 3) / 8 +
                3(8chi^6 + (chi^2 - 1)^2) / 64
        )
    )
    @test initial.elevation(0.0) ≈ expected_crest rtol = 32eps()
    @test initial.velocity(0.0, -0.5, 0.0, 2) == 0.0
    @test initial.velocity(0.0, 0.49, 0.0, 1) == 0.0

    case = stokes3_wave(
        base_cells = 4, max_level = 1,
        final_time = 0.0, cells_per_radius = 4,
        vorticity_refine_tolerance = 0.5
    )
    @test !isnothing(case.flow.solver.two_phase)
    @test case.indicators[1] isa InterfaceRefinementIndicator
    @test case.indicators[1].level == 1
    @test case.indicators[2] isa CurvatureRefinementIndicator
    @test case.indicators[3] isa VorticityRefinementIndicator
    @test maximum(block.key.level for block in case.flow.solver.mesh.blocks) == 1
    @test all(isfinite, case.flow.velocity.interior)
    @test case.initial_projection.converged
    divergence = CellField(case.flow.solver.mesh)
    divergence!(divergence, case.flow.velocity)
    @test maximum(
        maximum(abs, interior(divergence, b))
            for b in eachindex(divergence.blocks)
    ) <= 1.0e-9

    raw = stokes3_wave(
        base_cells = 4, max_level = 0, interface_level = 0,
        final_time = 0.0, project_initial_velocity = false
    )
    @test isnothing(raw.initial_projection)

    evolved = stokes3_wave(
        base_cells = 8, max_level = 0, interface_level = 0,
        final_time = 1.0e-4, dt = 1.0e-4,
        cells_per_radius = 2
    )
    @test evolved.flow.time == 1.0e-4
    @test evolved.flow.steps == 1
    @test evolved.result.pressure.converged

    live = visualize_stokes3_wave(
        base_cells = 4, max_level = 0, interface_level = 0,
        final_time = 1.0e-4, dt = 1.0e-4,
        cells_per_radius = 2, checkpoints = 2,
        live_dt = 1.0e-4, viewer = NoVisualization(),
        asynchronous = false
    )
    @test field_names(live.fields) == [
        :volume_fraction, :velocity_x, :velocity_y,
        :velocity_magnitude, :velocity_vorticity, :pressure,
    ]
    live_result = wait_for_stokes3!(live)
    @test live_result.time == 1.0e-4
    @test live.flow.time == 1.0e-4
    @test_throws ArgumentError visualize_stokes3_wave(
        final_time = 0, viewer = NoVisualization()
    )
end
