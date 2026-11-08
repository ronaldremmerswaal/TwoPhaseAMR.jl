using Test
using TwoPhaseAMR
using CairoMakie

const Makie = CairoMakie.Makie

_image_plot(axis) = only(filter(plot -> plot isa Makie.Image, axis.scene.plots))

@testset "Static figures draw rasterized frames with vector overlays" begin
    mesh = BlockTreeMesh(block_size = (6, 6), max_level = 1)
    velocity = FaceVelocity(mesh, (x, y, t, axis) ->
        axis == 1 ? sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y))
    flow = FlowSimulation(mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.01);
        velocity, diagnostics = (KineticEnergy(),), dt = 0.01)
    fields = VisualizationFieldRegistry()
    register_default_fields!(fields, :velocity, flow.velocity)
    recorder = Recorder(fields; diagnostics = flow, keyframes = EquispacedTimes(3), preview = nothing)
    simulate!(flow, 0.05; observer = recorder)
    store = snapshot_store(recorder)
    frames = [store[info] for info in keyframes(store)]
    @test length(frames) == 3

    @test snapshot_figure(fields; field = :velocity_vorticity) isa Makie.Figure
    @test snapshot_figure(first(frames); field = :velocity_x) isa Makie.Figure
    panel = Makie.Figure(size = (900, 300))
    for (i, frame) in enumerate(frames)
        axis = snapshot_figure!(panel[1, i], frame;
            field = :velocity_vorticity, colorbar = false, resolution = 64)
        @test axis isa Makie.Axis
        image = _image_plot(axis)
        @test size(image[3][]) == (64, 64)
        # The image is the CPU raster of the frame.
        @test isequal(
            image[3][], rasterize(frame, :velocity_vorticity, (0, 1, 0, 1), (64, 64)),
        )
        # Vorticity uses a symmetric colour range by default.
        lower, upper = image.colorrange[]
        @test lower ≈ -upper
    end
    axis = Makie.Axis(Makie.Figure()[1, 1])
    @test plot_diagnostics!(axis, flow, [:kinetic_energy];
        analytical = t -> 1.0, labels = ["energy"]) === axis
    @test length(axis.scene.plots) == 2
    @test_throws ArgumentError plot_diagnostics!(axis, flow, [:kinetic_energy]; labels = [])
    @test_throws ArgumentError snapshot_figure(first(frames); field = :missing)

    capmesh = BlockTreeMesh(block_size = (8, 8), periodic = (false, false))
    solid = GeometryField(capmesh, (x, y) -> y - 0.1)
    interface = InterfaceField(capmesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.25;
        geometry = solid, lipschitz = 1)
    capflow = FlowSimulation(capmesh, TwoPhase(interface; density = (1.0, 1.0),
        dynamic_viscosity = (0.01, 0.01),
        surface_tension = SurfaceTension(0.01)); geometry = solid, dt = 1.0e-4)
    capfields = VisualizationFieldRegistry()
    register_default_fields!(capfields, :volume_fraction, interface)
    register_geometry!(capfields, solid)
    register_field!(capfields, :phase, PhaseScalar(interface; liquid = 1, gas = 0))
    @test snapshot_figure(capfields; field = :volume_fraction) isa Makie.Figure
    capframe = TwoPhaseAMR._registry_frame(capfields)
    capaxis = snapshot_figure!(Makie.Figure()[1, 1], capframe;
        field = :volume_fraction, colorbar = false)
    # Field image, solid, block outlines, and interface.
    @test length(capaxis.scene.plots) == 4
    no_overlay_axis = snapshot_figure!(Makie.Figure()[1, 1], capframe;
        field = :volume_fraction, colorbar = false, blocks = false,
        interface = false, geometry = false)
    @test length(no_overlay_axis.scene.plots) == 1
    @test Tuple(_image_plot(no_overlay_axis).colorrange[]) == (0.0f0, 1.0f0)
    sharp_axis = snapshot_figure!(Makie.Figure()[1, 1], capframe;
        field = :phase, phase_rendering = :plic, resolution = 128)
    @test isequal(
        _image_plot(sharp_axis)[3][],
        rasterize(capframe, :phase, (0, 1, 0, 1), (128, 128); phase_rendering = :plic),
    )
    add_diagnostic!(capflow, :time, state -> state.time)
    capdiag = Makie.Axis(Makie.Figure()[1, 1])
    @test plot_diagnostics!(capdiag, capflow, :time) === capdiag
    @test length(capdiag.scene.plots) == 1
end

@testset "Keyframe recordings write GIF and video files" begin
    mesh = BlockTreeMesh(block_size = (6, 6))
    flow = FlowSimulation(mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.05);
        boundary = FreeSlip(), dt = 0.01,
        velocity = (x, y, t, axis) -> axis == 1 ? sinpi(2y) : 0.0)
    fields = VisualizationFieldRegistry()
    register_default_fields!(fields, flow)
    recorder = Recorder(fields; keyframes = EquispacedTimes(4), preview = nothing)
    simulate!(flow, 0.3; observer = recorder)
    store = snapshot_store(recorder)
    @test length(keyframes(store)) == 4
    mktempdir() do dir
        path = joinpath(dir, "keyframes.gif")
        @test record_checkpoints(path, recorder; field = :velocity_magnitude,
            framerate = 2, size = (300, 240)) == path
        @test filesize(path) > 0
        video = joinpath(dir, "keyframes.mp4")
        @test record_checkpoints(video, store; field = :velocity_vorticity,
            framerate = 2, size = (300, 240), blocks = false) == video
        @test filesize(video) > 0
    end
    @test_throws ArgumentError record_checkpoints(
        "unused.gif", Recorder(VisualizationFieldRegistry()); field = :pressure,
    )
end
