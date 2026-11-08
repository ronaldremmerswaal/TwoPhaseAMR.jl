using Test
using TwoPhaseAMR
using CairoMakie

const Makie = CairoMakie.Makie

function _test_plot_matches_geometry(plot, geometry)
    mesh = plot[1][]
    @test length(mesh.position) == length(geometry.points)
    @test length(mesh.faces) == length(geometry.triangles)
    @test length(plot.color[]) == length(geometry.points)
    # Recreated fields must still be drawn before anti-aliased overlays,
    # otherwise block, cell and interface outlines acquire white halos.
    @test Makie.transformationmatrix(plot)[][3, 4] == -1
    return nothing
end

@testset "Makie defaults to majority-phase cells, with PLIC opt-in" begin
    mesh = BlockTreeMesh(block_size = (4, 4), max_level = 1)
    interface = InterfaceField(mesh, (x, y) -> x - 0.625; lipschitz = 1)
    scalar = PhaseScalar(interface; liquid = 7, gas = -11)
    face_scalar = PhaseFaceScalar{2}(interface; liquid = 13, gas = -17)
    velocity = (
        FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? 2 : 3),
        FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? -4 : -5),
    )
    TwoPhaseAMR._mark_inactive_phase_velocities!(velocity, interface)
    registry = VisualizationFieldRegistry()
    register_default_fields!(registry, :velocity, velocity, interface)
    register_field!(registry, :scalar, scalar)
    register_field!(registry, :face_scalar, face_scalar)
    viewer = visualize(registry; field = :velocity_x, display = false)
    @test viewer.interface_checkbox.checked[]
    @test isnothing(viewer.solid_checkbox)
    @test viewer.cell_render_geometry[] isa AMRCellGeometry
    @test viewer.cell_plot.visible[]
    @test !viewer.face_plot.visible[]
    cell_frame = TwoPhaseAMR._capture_frame(registry, 0.0, 0)
    update_visualization!(viewer, cell_frame)
    for name in field_names(registry)
        select_visualization_field!(viewer, name)
        @test viewer.cell_render_geometry[] === cell_frame.topology.cell_geometry
        @test isequal(viewer.cell_colors[], Float32.(
            TwoPhaseAMR._visualization_vertex_values(cell_frame[name]),
        ))
        _test_plot_matches_geometry(viewer.cell_plot, viewer.cell_render_geometry[])
    end
    # A viewer keyword selects the same policy for subsequent session captures.
    sharp_viewer = visualize(registry; field = :velocity_x,
        phase_rendering = :plic, display = false)
    @test registry.phase_rendering == :plic
    @test sharp_viewer.face_render_geometry[] isa TwoPhaseAMR.PLICVisualizationGeometry{FaceRef}
    sharp_frame = TwoPhaseAMR._capture_frame(registry, 0.0, 0)
    source = registry.entries[1].source
    cache = sharp_frame.primitives[source].geometry_cache
    @test isempty(cache.geometries)
    @test adapt!(AMRWorkspace(mesh, interface, scalar, face_scalar, velocity...),
        [:refine]; step = 0).changed
    # Neither the lazy geometry nor cell values in an old checkpoint may
    # access the adapted live mesh, or consult its new registry policy.
    update_visualization!(sharp_viewer, sharp_frame)
    @test length(cache.geometries) == 1
    sharp_plot = sharp_viewer.face_plot
    select_visualization_field!(sharp_viewer, :velocity_x)
    @test sharp_viewer.face_plot === sharp_plot
    @test length(cache.geometries) == 1
    _test_plot_matches_geometry(sharp_viewer.face_plot,
        sharp_viewer.face_render_geometry[])
    update_visualization!(viewer, cell_frame)
    for name in field_names(registry)
        select_visualization_field!(viewer, name)
        @test viewer.cell_plot.visible[]
        @test !viewer.face_plot.visible[]
        @test viewer.cell_render_geometry[] === cell_frame.topology.cell_geometry
        @test isequal(viewer.cell_colors[], Float32.(
            TwoPhaseAMR._visualization_vertex_values(cell_frame[name]),
        ))
    end
    @test_throws ArgumentError visualize(registry; phase_rendering = :unknown, display = false)
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

@testset "Makie renders PLIC phase and filled-solid overlays" begin
    mesh = BlockTreeMesh(
        block_size = (6, 6), max_level = 1,
        periodic = (false, false),
    )
    geometry = GeometryField(mesh, (x, y) -> y - 0.18 - 0.1x)
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.48) - 0.2;
        geometry, contact_angle = pi / 2, lipschitz = 1,
    )
    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ? 8.0 + x + y : 2.0 - x,
    )
    scalar = PhaseScalar(
        interface; liquid = (x, y) -> 1 + x - y,
        gas = (x, y) -> 2 - x + y,
    )
    staggered_scalar = PhaseFaceScalar{2}(
        interface; liquid = (x, y) -> 1 + x - y,
        gas = (x, y) -> 2 - x + y,
    )
    registry = VisualizationFieldRegistry(phase_rendering = :plic)
    register_default_fields!(
        registry, :volume_fraction, interface, :velocity, velocity,
    )
    phase_velocity = (
        velocity,
        FaceVelocity(
            mesh,
            (x, y, time, axis) -> axis == 1 ? -y : x,
        ),
    )
    register_default_fields!(
        registry, :phase_velocity, phase_velocity, interface,
    )
    register_field!(registry, :scalar, scalar)
    register_field!(registry, :staggered_scalar, staggered_scalar)
    viewer = visualize(registry; display = false)
    @test viewer.interface_checkbox.checked[]
    @test viewer.solid_checkbox.checked[]
    @test viewer.interface_plot.visible[]
    @test viewer.solid_plot.visible[]
    viewer.interface_checkbox.checked[] = false
    viewer.solid_checkbox.checked[] = false
    @test !viewer.interface_plot.visible[]
    @test !viewer.solid_plot.visible[]
    @test Tuple(viewer.cell_plot.colorrange[]) isa
        Tuple{Float32, Float32}
    @test length(viewer.interface_segments[]) ==
        2length(plic_segments(interface))
    solid = plic_solid_geometry(geometry)
    @test length(viewer.solid_mesh[].position) == length(solid.points)
    @test length(viewer.solid_mesh[].faces) == length(solid.triangles)
    select_visualization_field!(viewer, :velocity_magnitude)
    @test viewer.current_field[] == :velocity_magnitude
    @test Tuple(viewer.cell_plot.colorrange[]) isa
        Tuple{Float32, Float32}
    select_visualization_field!(viewer, :velocity_vorticity)
    @test viewer.current_field[] == :velocity_vorticity
    lower, upper = viewer.cell_plot.colorrange[]
    @test lower == -upper
    @test upper >= maximum(
        abs, Float32.(visualization_values(velocity, Vorticity()))
    )
    select_visualization_field!(viewer, :volume_fraction)
    @test viewer.current_field[] == :volume_fraction
    select_visualization_field!(viewer, :scalar)
    @test viewer.current_field[] == :scalar
    @test viewer.cell_render_geometry[] isa
        TwoPhaseAMR.PLICVisualizationGeometry{CellRef}
    @test all(isfinite, viewer.cell_colors[])
    scalar_source = registry.entries[registry.indices[:scalar]].source
    @test viewer.cell_colors[] ≈ Float32.(
        TwoPhaseAMR._phase_visualization_vertex_values(
            TwoPhaseAMR._source_values(scalar_source),
            viewer.cell_render_geometry[],
        ),
    )
    select_visualization_field!(viewer, :staggered_scalar)
    @test viewer.current_field[] == :staggered_scalar
    @test viewer.face_render_geometry[] isa
        TwoPhaseAMR.PLICVisualizationGeometry{FaceRef}
    @test all(isfinite, viewer.face_colors[])
    select_visualization_field!(viewer, :phase_velocity_vorticity)
    @test viewer.current_field[] == :phase_velocity_vorticity
    @test viewer.cell_render_geometry[] isa
        TwoPhaseAMR.PLICVisualizationGeometry{CellRef}
    @test all(isfinite, viewer.cell_colors[])
    lower, upper = viewer.cell_plot.colorrange[]
    @test lower == -upper
    select_visualization_field!(viewer, :volume_fraction)

    workspace = AMRWorkspace(mesh)
    register_adaptive_field!(workspace, interface)
    register_adaptive_field!(workspace, geometry)
    @test adapt!(workspace, [:refine]; step = 1, force = true).changed
    update_visualization!(viewer)
    @test length(viewer.interface_segments[]) ==
        2length(plic_segments(interface))
    solid = plic_solid_geometry(geometry)
    @test length(viewer.solid_mesh[].faces) == length(solid.triangles)
    @test !viewer.interface_checkbox.checked[]
    @test !viewer.solid_checkbox.checked[]
    @test !viewer.interface_plot.visible[]
    @test !viewer.solid_plot.visible[]
    viewer.interface_checkbox.checked[] = true
    viewer.solid_checkbox.checked[] = true
    @test viewer.interface_plot.visible[]
    @test viewer.solid_plot.visible[]
end

@testset "Makie view controls toggle boundaries and reset framing" begin
    mesh = BlockTreeMesh(block_size = (4, 4), max_level = 1)
    pressure = CellField(mesh)
    initialize!(pressure, (x, y) -> x + y)
    registry = VisualizationFieldRegistry()
    register_field!(registry, :pressure, pressure)
    viewer = visualize(registry; show_blocks = false, show_cells = true,
        display = false)
    borders = Dict(
        plot.depth_shift[] => plot for plot in viewer.axis.scene.plots
            if haskey(plot.attributes, :linewidth) &&
            plot.depth_shift[] in (-2.0f-5, -1.0f-5)
    )
    blocks, cells = borders[-2.0f-5], borders[-1.0f-5]
    @test !viewer.block_checkbox.checked[]
    @test viewer.cell_checkbox.checked[]
    @test !blocks.visible[]
    @test cells.visible[]
    @test isnothing(viewer.interface_checkbox)
    @test isnothing(viewer.solid_checkbox)

    default_limits = viewer.axis.finallimits[]
    Makie.limits!(viewer.axis, 0.2, 0.8, 0.1, 0.9)
    zoomed_limits = viewer.axis.finallimits[]
    colors = viewer.cell_colors[]
    viewer.block_checkbox.checked[] = true
    viewer.cell_checkbox.checked[] = false
    @test blocks.visible[]
    @test !cells.visible[]
    @test viewer.cell_colors[] === colors
    @test viewer.axis.finallimits[] == zoomed_limits

    @test adapt!(AMRWorkspace(mesh, pressure), [:refine]; step = 0).changed
    update_visualization!(viewer)
    @test viewer.block_checkbox.checked[]
    @test !viewer.cell_checkbox.checked[]
    @test blocks.visible[]
    @test !cells.visible[]
    @test viewer.axis.finallimits[] == zoomed_limits
    _test_plot_matches_geometry(viewer.cell_plot, viewer.cell_geometry)

    viewer.reset_view_button.clicks[] += 1
    @test viewer.axis.finallimits[] == default_limits
    @test blocks.visible[]
    @test !cells.visible[]
    viewer.block_checkbox.checked[] = false
    viewer.cell_checkbox.checked[] = true
    @test !blocks.visible[]
    @test cells.visible[]

    solid_registry = VisualizationFieldRegistry()
    register_field!(solid_registry, :pressure, pressure)
    register_geometry!(solid_registry, GeometryField(mesh, (x, y) -> y - 0.3))
    solid_viewer = visualize(solid_registry; display = false)
    @test isnothing(solid_viewer.interface_checkbox)
    @test solid_viewer.solid_checkbox.checked[]
    solid_viewer.solid_checkbox.checked[] = false
    @test !solid_viewer.solid_plot.visible[]
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
    @test Makie.transformationmatrix(viewer.cell_plot)[][3, 4] == -1
    @test Makie.transformationmatrix(viewer.face_plot)[][3, 4] == -1
    border_plots = Dict(
        plot.depth_shift[] => plot for plot in viewer.axis.scene.plots
            if haskey(plot.attributes, :linewidth) &&
            plot.depth_shift[] in (-2.0f-5, -1.0f-5)
    )
    @test length(border_plots) == 2
    cell_border = border_plots[-1.0f-5]
    block_border = border_plots[-2.0f-5]
    @test cell_border.linewidth[] == 0.25
    @test block_border.linewidth[] == 0.75
    @test Makie.to_color(cell_border.color[]) == Makie.RGBAf(0.75, 0.75, 0.75, 1)
    @test Makie.to_color(block_border.color[]) == Makie.RGBAf(0.5, 0.5, 0.5, 1)
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

@testset "Static figures, diagnostics and recordings" begin
    mesh = BlockTreeMesh(block_size = (6, 6), max_level = 1)
    velocity = FaceVelocity(mesh, (x, y, t, axis) ->
        axis == 1 ? sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y))
    flow = FlowSimulation(mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.01);
        velocity, diagnostics = (KineticEnergy(),))
    fields = VisualizationFieldRegistry()
    register_default_fields!(fields, :velocity, velocity)
    session = VisualizationSession(fields; viewer = NoVisualization(),
        checkpoints = 3, asynchronous = false)
    initialize_visualization!(session; time = 0.0, step = 0, final_time = 1.0,
        velocity, pressure = flow.pressure, solver = flow.solver)
    update_visualization!(session; event = :step, time = 0.5, step = 1,
        velocity, pressure = flow.pressure, solver = flow.solver)
    update_visualization!(session; event = :completed, time = 1.0, step = 2,
        velocity, pressure = flow.pressure, solver = flow.solver)
    checkpoints = visualization_checkpoints(session)
    @test length(checkpoints) == 3
    fig = snapshot_figure(fields; field = :velocity_vorticity)
    @test fig isa Makie.Figure
    @test snapshot_figure(first(checkpoints).frame; field = :velocity_x) isa Makie.Figure
    @test snapshot_figure(fields; frame = first(checkpoints),
        field = :velocity_vorticity, blocks = false, colorbar = false) isa Makie.Figure
    panel = Makie.Figure(size = (900, 300))
    for (i, checkpoint) in enumerate(checkpoints)
        @test snapshot_figure!(panel[1, i], checkpoint;
            field = :velocity_vorticity, colorbar = false) isa Makie.Axis
    end
    axis = Makie.Axis(Makie.Figure()[1, 1])
    @test plot_diagnostics!(axis, flow, [:kinetic_energy];
        analytical = t -> 1.0, labels = ["energy"]) === axis
    @test length(axis.scene.plots) == 2
    @test_throws ArgumentError plot_diagnostics!(axis, flow, [:kinetic_energy]; labels = [])

    capmesh = BlockTreeMesh(block_size = (8, 8), periodic = (false, false))
    solid = GeometryField(capmesh, (x, y) -> y - 0.1)
    interface = InterfaceField(capmesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.25;
        geometry = solid, lipschitz = 1)
    capflow = FlowSimulation(capmesh, TwoPhase(interface; density = (1.0, 1.0),
        dynamic_viscosity = (0.01, 0.01),
        surface_tension = SurfaceTension(0.01)); geometry = solid)
    capfields = VisualizationFieldRegistry()
    register_default_fields!(capfields, :volume_fraction, interface)
    register_geometry!(capfields, solid)
    capfig = snapshot_figure(capfields; field = :volume_fraction)
    @test capfig isa Makie.Figure
    capframe = TwoPhaseAMR._capture_frame(capfields, 0.0, 0)
    capaxis = snapshot_figure!(Makie.Figure()[1, 1], capframe;
        field = :volume_fraction, colorbar = false)
    @test length(capaxis.scene.plots) >= 4
    @test capflow isa FlowSimulation
    no_overlay_axis = snapshot_figure!(Makie.Figure()[1, 1], capframe;
        field = :volume_fraction, colorbar = false, blocks = false,
        interface = false, geometry = false)
    @test length(no_overlay_axis.scene.plots) == 1
    capsession = VisualizationSession(capfields; viewer = NoVisualization(),
        checkpoints = 2, asynchronous = false)
    initialize_visualization!(capsession; time = 0.0, step = 0, final_time = 1.0,
        velocity = capflow.velocity, pressure = capflow.pressure, solver = capflow.solver)
    update_visualization!(capsession; event = :completed, time = 1.0, step = 1,
        velocity = capflow.velocity, pressure = capflow.pressure, solver = capflow.solver)
    add_diagnostic!(capflow, :time, state -> state.time)
    capdiag = Makie.Axis(Makie.Figure()[1, 1])
    @test plot_diagnostics!(capdiag, capflow, :time) === capdiag
    @test length(capdiag.scene.plots) == 1
    sharp = VisualizationFieldRegistry(phase_rendering = :plic)
    register_field!(sharp, :phase, PhaseScalar(interface; liquid = 1, gas = 0))
    @test snapshot_figure(sharp; field = :phase) isa Makie.Figure
    sharpframe = TwoPhaseAMR._capture_frame(sharp, 0.0, 0)
    @test snapshot_figure(sharpframe; field = :phase) isa Makie.Figure

    mktempdir() do dir
        path = joinpath(dir, "checkpoints.gif")
        @test record_checkpoints(path, session; field = :velocity_vorticity,
            framerate = 2) == path
        @test filesize(path) > 0
        cappath = joinpath(dir, "capillary.gif")
        @test record_checkpoints(cappath, capsession; field = :volume_fraction,
            framerate = 2, interface = true) == cappath
        @test filesize(cappath) > 0
    end
end
