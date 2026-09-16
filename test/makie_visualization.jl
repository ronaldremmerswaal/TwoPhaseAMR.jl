using Test
using TwoPhaseAMR
using GLMakie

const Makie = GLMakie.Makie

function _test_plot_matches_geometry(plot, geometry)
    mesh = plot[1][]
    @test length(mesh.position) == length(geometry.points)
    @test length(mesh.faces) == length(geometry.triangles)
    @test length(plot.color[]) == length(geometry.points)
    return nothing
end

@testset "Diagnostic plots follow their data range" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    pressure = CellField(mesh)
    velocity = FaceVelocity(mesh)
    registry = VisualizationFieldRegistry()
    register_default_fields!(registry, :pressure, pressure)
    register_diagnostic!(
        registry, :small_value,
        (velocity, pressure, time, step, solver) ->
        1.0e-4 * (step + 1)
    )
    viewer = visualize(registry; display = false)
    session = VisualizationSession(
        registry; viewer, asynchronous = false
    )
    initialize_visualization!(
        session; time = 0.0, step = 0, final_time = 1.0,
        velocity, pressure, solver = nothing
    )
    update_visualization!(
        session; event = :step, time = 0.5, step = 1,
        velocity, pressure, solver = nothing
    )
    update_visualization!(
        session; event = :completed, time = 1.0, step = 2,
        velocity, pressure, solver = nothing
    )
    ylimits = Makie.limits(viewer.diagnostic_axis.finallimits[], 2)
    @test first(ylimits) < 1.0e-4
    @test last(ylimits) > 3.0e-4
    @test last(ylimits) < 1.0e-2
end

@testset "Makie plots replace dynamic AMR topology atomically" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 1,
        adaptation = MeshAdaptationOptions(coarsen_delay = 0)
    )
    pressure = CellField(mesh)
    initialize!(pressure, (x, y) -> x + 2y)
    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? y : -x
    )
    registry = VisualizationFieldRegistry()
    register_default_fields!(
        registry, :pressure, pressure, :velocity, velocity
    )
    viewer = visualize(
        registry; field = :pressure, show_blocks = true, show_cells = true,
        display = false
    )
    plot_count = length(viewer.axis.scene.plots)
    border_depth_shifts = sort(
        [
            plot.depth_shift[] for plot in viewer.axis.scene.plots
                if haskey(plot.attributes, :linewidth) &&
                plot.linewidth[] in (0.5, 1.5)
        ]
    )
    @test border_depth_shifts == [-2.0f-5, -1.0f-5]
    Makie.limits!(viewer.axis, 0.2, 0.8, 0.1, 0.9)
    view_limits = viewer.axis.finallimits[]

    workspace = AMRWorkspace(mesh, pressure, velocity)
    @test adapt!(workspace, [:refine]; step = 0).changed
    update_visualization!(viewer)
    _test_plot_matches_geometry(viewer.cell_plot, viewer.cell_geometry)
    @test viewer.axis.finallimits[] == view_limits
    @test length(viewer.axis.scene.plots) == plot_count

    select_visualization_field!(viewer, :velocity_x)
    _test_plot_matches_geometry(
        viewer.face_plot, viewer.face_geometries[1]
    )
    @test length(viewer.axis.scene.plots) == plot_count

    select_visualization_field!(viewer, :velocity_y)
    _test_plot_matches_geometry(
        viewer.face_plot, viewer.face_geometries[2]
    )
    @test length(viewer.axis.scene.plots) == plot_count

    @test adapt!(
        workspace, fill(:coarsen, nblocks(mesh)); step = 1, force = true
    ).changed
    update_visualization!(viewer)
    _test_plot_matches_geometry(
        viewer.face_plot, viewer.face_geometries[2]
    )
    select_visualization_field!(viewer, :pressure)
    _test_plot_matches_geometry(viewer.cell_plot, viewer.cell_geometry)
    @test viewer.axis.finallimits[] == view_limits
    @test length(viewer.axis.scene.plots) == plot_count
end
