using Test, TwoPhaseAMR

include(joinpath(@__DIR__, "..", "examples", "cylinder_flow.jl"))

@testset "Verstappen--Droge cylinder example" begin
    @test CYLINDER_REFERENCE.reynolds == 100.0
    @test CYLINDER_REFERENCE.strouhal == 0.165
    @test CYLINDER_REFERENCE.total_drag == 1.24

    case = cylinder_flow(
        final_time = 1.0e-4, dt = 1.0e-4,
        base_level = 0, max_level = 1, geometry_level = 1,
        adaptation_interval = 1, perturbation = 0.0,
        checkpoints = 2, live_dt = 1.0,
        viewer = NoVisualization(), display = false,
    )
    @test case.mesh.origin == (-4.0, -4.0)
    @test case.mesh.extent == (14.0, 8.0)
    @test nblocks(case.mesh) > 1
    @test case.result.time == 1.0e-4
    @test case.result.pressure.converged
    @test all(isfinite, case.flow.velocity.interior)
    @test all(isfinite, case.drag)
    @test all(isfinite, case.lift)
    @test case.fields.geometries == [case.geometry]
    @test !isempty(plic_solid_geometry(case.geometry).triangles)
    net_flux = sum(pairs(case.mesh.boundary_faces)) do (index, face)
        face.side * face.area * case.geometry.boundary_aperture[index] *
            case.flow.velocity.boundary[index]
    end
    @test abs(net_flux) <= 512eps()
end
