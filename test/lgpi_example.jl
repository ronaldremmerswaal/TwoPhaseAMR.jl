using Test, TwoPhaseAMR

include(joinpath(@__DIR__, "..", "examples", "lgpi.jl"))

@testset "Thesis LGPI example" begin
    initial = lgpi_initial_condition()
    @test initial.scale == 5.0
    @test initial.wavelength == 4.0
    @test initial.domain == (4.0, 2.4)
    @test initial.mean_height == 1.52
    @test initial.amplitude == 0.72
    @test initial.horizontal_shift == 3.1
    @test initial.scaled_steepness == 2.2
    @test initial.bathymetry_half_axes == (3.6, 0.5599999999999999)
    @test initial.interface_levelset(
        initial.horizontal_shift, initial.mean_height,
    ) == 0.0
    @test initial.bathymetry_levelset(
        initial.bathymetry_half_axes[1], 0.0,
    ) ≈ 0.0 atol = 8eps()
    @test initial.bathymetry_levelset(0.0, 0.0) == -1.0

    case = lgpi_setup(
        max_level = 1, interface_level = 1, geometry_level = 1,
        adaptation_interval = 1,
    )
    @test case.flow.solver.boundary isa FreeSlip
    @test case.flow.solver.two_phase !== nothing
    @test case.interface.geometry === case.geometry
    @test case.interface.contact_angles == ntuple(_ -> pi / 2, 4)
    @test case.interface.embedded_contact_angle == pi / 2
    @test case.interface_indicator isa InterfaceRefinementIndicator
    @test case.interface_indicator.level == 1
    @test case.vorticity_indicator isa VorticityRefinementIndicator
    @test case.vorticity_indicator.field === case.flow.velocity
    @test case.vorticity_indicator.minimum_level == 0
    @test case.vorticity_indicator.maximum_level == 1
    @test case.geometry_indicator isa GeometryRefinementIndicator
    @test case.geometry_indicator.level == 1
    @test maximum(block.key.level for block in case.mesh.blocks) == 1
    @test nblocks(case.mesh) > 1
    @test !isempty(plic_segments(case.interface))
    @test any(
        TwoPhaseAMR._get(case.interface.reconstruction, ref) == 5
            for ref in TwoPhaseAMR._cell_references(case.mesh)
    )
    @test !isempty(plic_solid_geometry(case.geometry).triangles)
    @test case.gravity_time ≈ sqrt(
        1001 / 999 * 2pi * 4 / 9.81,
    )
    @test case.interface_cells_per_wavelength == 20
    @test case.maximum_cells_per_wavelength == 20
    for ref in TwoPhaseAMR._cell_references(case.mesh)
        fraction = TwoPhaseAMR._get(case.interface.fraction, ref)
        open = TwoPhaseAMR._get(case.geometry.open_fraction, ref)
        @test 0 <= fraction <= open + 64eps()
        curvature = TwoPhaseAMR._get(case.interface.curvature, ref)
        isfinite(curvature) || continue
        h = maximum(case.mesh.blocks[ref.block].spacing)
        @test abs(curvature) <= inv(2h)
    end

    evolved = lgpi(
        final_time = 1.0e-5, dt = 1.0e-5,
        max_level = 0, interface_level = 0, geometry_level = 0,
    )
    @test evolved.result.time == 1.0e-5
    @test evolved.result.steps == 1
    @test evolved.result.pressure.converged
    @test all(isfinite, evolved.flow.velocity.interior)

    live = visualize_lgpi(
        final_time = 1.0e-5, dt = 1.0e-5,
        max_level = 0, interface_level = 0, geometry_level = 0,
        checkpoints = 2, live_dt = 1.0e-5,
        viewer = NoVisualization(), asynchronous = false,
    )
    @test field_names(live.fields) == [
        :volume_fraction, :velocity_x, :velocity_y,
        :velocity_magnitude, :velocity_vorticity, :pressure,
    ]
    @test live.fields.geometries == [live.geometry]
    result = wait_for_lgpi!(live)
    @test result.time == 1.0e-5
    @test live.flow.time == 1.0e-5

    @test_throws ArgumentError lgpi_initial_condition(scale = 0)
    @test_throws ArgumentError visualize_lgpi(
        final_time = 0.0, viewer = NoVisualization(),
    )
end
