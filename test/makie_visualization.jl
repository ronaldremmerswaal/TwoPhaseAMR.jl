using Test
using TwoPhaseAMR
using CairoMakie

const Makie = CairoMakie.Makie
const MakieExt = Base.get_extension(TwoPhaseAMR, :TwoPhaseAMRMakieExt)

function recorded_run(; keyframes = 4)
    mesh = BlockTreeMesh(block_size = (8, 8), max_level = 1)
    velocity = FaceVelocity(
        mesh, (x, y, t, axis) ->
        axis == 1 ? sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y)
    )
    flow = FlowSimulation(
        mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.01);
        velocity, diagnostics = (KineticEnergy(), LinearMomentum(), CellCount()),
        dt = 0.01
    )
    add_diagnostic!(flow, :energy_bound, state -> 0.3)
    fields = VisualizationFieldRegistry()
    register_default_fields!(fields, :velocity, flow.velocity)
    register_field!(
        fields, :speed, flow.velocity, CentredMagnitude();
        colorrange = (0, 2)
    )
    recorder = Recorder(
        fields; diagnostics = flow,
        keyframes = EquispacedTimes(keyframes), preview = nothing
    )
    simulate!(flow, 0.05; observer = recorder)
    return (; flow, fields, recorder, store = snapshot_store(recorder))
end

const RUN = recorded_run()
const FRAMES = [RUN.store[info] for info in keyframes(RUN.store)]

# Lay out and draw the figure as saving it would.
settle!(figure) = (Makie.colorbuffer(figure); figure)

field_image(plot) = first(plot.plots)
image_rect(image) = (image[1][]..., image[2][]...)
limits_rect(axis) = let (x0, y0) = minimum(axis.finallimits[]), (x1, y1) = maximum(axis.finallimits[])
    (x0, x1, y0, y1)
end
panel_size(axis) = Tuple(Makie.widths(axis.scene.viewport[]))

@testset "Field axes fill their panel with an isotropic view" begin
    frame = last(FRAMES)
    box = (0.4, 0.6, 0.4, 0.6)
    figure = Figure(size = (900, 300))
    axis = fieldaxis(figure[1, 1]; camera = box, decorations = :none)
    plot = amrfield!(axis, frame, :velocity_magnitude)
    settle!(figure)
    # The axis box is its whole layout cell, whatever the view's aspect.
    reference_figure = Figure(size = (900, 300))
    reference = Axis(reference_figure[1, 1])
    hidedecorations!(reference)
    hidespines!(reference)
    settle!(reference_figure)
    @test axis.scene.viewport[] == reference.scene.viewport[]
    w, h = panel_size(axis)
    x0, x1, y0, y1 = limits_rect(axis)
    # The box is shown completely and centred; the wide panel shows more of
    # the domain instead of empty space, with one scale in x and y.
    @test (x0 + x1) / 2 ≈ 0.5 && (y0 + y1) / 2 ≈ 0.5
    @test y0 ≈ 0.4 && y1 ≈ 0.6
    @test x1 - x0 ≈ 0.2 * w / h
    @test (x1 - x0) / w ≈ (y1 - y0) / h
    @test all(
        isapprox.(
            limits_rect(axis),
            visible_rect(zoom_to_box(ViewCamera((0, 0), 1), box, (w, h)), (w, h)),
        )
    )
    # The field is rasterized over the visible part of the domain at screen
    # resolution (two samples per pixel by default).
    image = field_image(plot)
    @test all(isapprox.(image_rect(image), (x0, x1, y0, y1)))
    @test size(image[3][]) == round.(Int, 2 .* (w, h))
    @test isequal(
        image[3][], rasterize(frame, :velocity_magnitude, (x0, x1, y0, y1), size(image[3][])),
    )

    # Zooming out past the domain rasterizes only the domain.
    figure = Figure(size = (400, 400))
    axis = fieldaxis(figure[1, 1]; camera = (-1, 2, -1, 2), decorations = :none)
    plot = amrfield!(axis, frame, :velocity_magnitude)
    settle!(figure)
    @test all(isapprox.(image_rect(field_image(plot)), (0, 1, 0, 1)))

    # A camera keeps its centre and scale when the panel is resized.
    camera = ViewCamera((0.3, 0.6), 0.002)
    figure = Figure(size = (500, 300))
    axis = fieldaxis(figure[1, 1]; camera)
    amrfield!(axis, frame, :velocity_x)
    settle!(figure)
    @test all(isapprox.(limits_rect(axis), visible_rect(camera, panel_size(axis))))
    resize!(figure, 700, 350)
    settle!(figure)
    @test all(isapprox.(limits_rect(axis), visible_rect(camera, panel_size(axis))))

    # The default camera fits the domain: tight along the limiting direction.
    figure = Figure(size = (600, 300))
    axis = fieldaxis(figure[1, 1]; decorations = :full)
    amrfield!(axis, frame, :velocity_x)
    settle!(figure)
    x0, x1, y0, y1 = limits_rect(axis)
    @test y0 ≈ 0 && y1 ≈ 1 && x0 < 0 && x1 > 1 && (x0 + x1) / 2 ≈ 0.5
    w, h = panel_size(axis)
    @test (x1 - x0) / w ≈ (y1 - y0) / h

    @test_throws ArgumentError fieldaxis(Figure()[1, 1]; camera = :zoom)
    @test_throws ArgumentError fieldaxis(Figure()[1, 1]; camera = (1, 0, 0, 1))
    @test_throws ArgumentError fieldaxis(Figure()[1, 1]; decorations = :some)
end

@testset "AMR field plots follow frames, colours, and overlays" begin
    figure = Figure(size = (500, 400))
    axis = fieldaxis(figure[1, 1])
    current = Observable(first(FRAMES))
    plot = amrfield!(axis, current, :velocity_vorticity)
    bar = Colorbar(figure[1, 2], plot)
    settle!(figure)
    rect = limits_rect(axis)
    image = field_image(plot)
    @test isequal(
        image[3][],
        rasterize(first(FRAMES), :velocity_vorticity, image_rect(image), size(image[3][]))
    )
    # Vorticity uses the diverging registered colormap and a symmetric range.
    lower, upper = image.colorrange[]
    @test lower ≈ -upper
    @test Tuple(bar.limits[]) == (lower, upper)
    # Swapping the frame re-rasterizes it without touching the view.
    current[] = last(FRAMES)
    settle!(figure)
    @test limits_rect(axis) == rect
    @test isequal(
        image[3][],
        rasterize(last(FRAMES), :velocity_vorticity, image_rect(image), size(image[3][]))
    )

    figure = Figure()
    axis = fieldaxis(figure[1, 1])
    # A velocity component is symmetric because its registered colormap diverges.
    lower, upper = field_image(amrfield!(axis, last(FRAMES), "velocity_x")).colorrange[]
    @test lower ≈ -upper
    # The registered range wins over the data; a requested range wins over both.
    @test Tuple(field_image(amrfield!(axis, last(FRAMES), :speed)).colorrange[]) == (0, 2)
    @test Tuple(
        field_image(
            amrfield!(
                axis, last(FRAMES), :speed;
                colorrange = (0.1, 0.5)
            )
        ).colorrange[]
    ) == (0.1f0, 0.5f0)
    @test_throws ArgumentError amrfield!(axis, last(FRAMES), :missing)

    # A registry is captured once; phase fields render by majority or sharply.
    capmesh = BlockTreeMesh(block_size = (8, 8), periodic = (false, false))
    solid = GeometryField(capmesh, (x, y) -> y - 0.1)
    interface = InterfaceField(
        capmesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.25;
        geometry = solid, lipschitz = 1
    )
    capfields = VisualizationFieldRegistry()
    register_default_fields!(capfields, :volume_fraction, interface)
    register_geometry!(capfields, solid)
    register_field!(capfields, :phase, PhaseScalar(interface; liquid = 1, gas = 0))
    capframe = TwoPhaseAMR._registry_frame(capfields)
    figure = Figure(size = (400, 400))
    axis = fieldaxis(figure[1, 1]; decorations = :none)
    sharp = amrfield!(
        axis, capfields, :phase; phase_rendering = :plic,
        interface = false, geometry = false
    )
    settle!(figure)
    image = field_image(sharp)
    @test isequal(
        image[3][], rasterize(
            capframe, :phase, image_rect(image),
            size(image[3][]); phase_rendering = :plic
        )
    )
    # Only the field is visible once all overlays are off.
    @test count(child -> child.visible[], sharp.plots) == 1
    fraction = amrfield!(axis, capframe, :volume_fraction; blocks = true)
    @test count(child -> child.visible[], fraction.plots) == 4
    @test Tuple(field_image(fraction).colorrange[]) == (0, 1)

    # The non-mutating form creates an isotropic axis.
    figure, axis, plot = amrfield(last(FRAMES), :velocity_vorticity)
    @test axis.autolimitaspect[] == 1 && isnothing(axis.aspect[])
end

@testset "Diagnostics are grouped by physical dimension" begin
    figure = Figure(size = (500, 700))
    plot_axes = diagnosticsgrid!(figure[1, 1], RUN.store)
    titles = [axis.title[] for axis in plot_axes]
    # One diagnostic per axis: its label and dimension; a shared axis: the quantity.
    @test titles == [
        "Kinetic energy (M L T⁻²)", "Linear momentum (M T⁻¹)", "Cells",
        "Energy bound",
    ]
    labels(axis) = [plot.label[] for plot in axis.scene.plots if plot isa Lines]
    @test labels(plot_axes[2]) == ["Linear momentum (x)", "Linear momentum (y)"]
    # The time axes are linked; a moving cursor never changes the limits.
    cursor = Observable(0.0)
    figure = Figure(size = (500, 700))
    plot_axes = diagnosticsgrid!(
        figure[1, 1], RUN.recorder; cursor,
        references = Dict(:kinetic_energy => :energy_bound, :cells => t -> 64)
    )
    settle!(figure)
    @test length(plot_axes) == 3
    limits = [axis.finallimits[] for axis in plot_axes]
    cursor[] = 0.04
    settle!(figure)
    @test [axis.finallimits[] for axis in plot_axes] == limits
    @test labels(plot_axes[1]) == ["Kinetic energy", "Energy bound"]
    @test labels(plot_axes[3]) == ["Cells", "Cells (reference)"]
    dashed(axis) = [!isnothing(plot.linestyle[]) for plot in axis.scene.plots if plot isa Lines]
    @test dashed(plot_axes[1]) == [false, true]

    # Other sources and options.
    for source in (RUN.flow, RUN.flow.diagnostics)
        @test length(diagnosticsgrid!(Figure()[1, 1], source)) == 4
    end
    @test length(
        diagnosticsgrid!(
            Figure()[1, 1], RUN.store; group = :name,
            direction = :row
        )
    ) == 4
    relative = diagnosticsgrid!(
        Figure()[1, 1], RUN.store; names = [:kinetic_energy],
        transform = :relative
    )
    energy = only(plot for plot in relative[1].scene.plots if plot isa Lines)
    @test first(energy[1][])[2] == 0
    @test last(energy[1][])[2] < 0
    @test_throws ArgumentError diagnosticsgrid!(Figure()[1, 1], RUN.store; names = [:none])
    @test_throws ArgumentError diagnosticsgrid!(Figure()[1, 1], RUN.store; group = :kind)
    @test_throws ArgumentError diagnosticsgrid!(Figure()[1, 1], RUN.store; transform = :log)
end

# GIF logical screen size from the header.
gif_size(path) = open(path) do io
    bytes = read(io, 10)
    (Int(bytes[7]) | Int(bytes[8]) << 8, Int(bytes[9]) | Int(bytes[10]) << 8)
end

function video_frames(path, (w, h))
    data = read(`$(Makie.FFMPEG_jll.ffmpeg()) -v error -i $path -f rawvideo -pix_fmt rgb24 -`)
    bytes = 3w * h
    return [
        permutedims(reshape(data[((k - 1) * bytes + 1):(k * bytes)], 3, w, h), (3, 2, 1))
            for k in 1:(length(data) ÷ bytes)
    ]
end

# The limits show `rect` completely and centred, exceeding it by less than one
# pixel: integer panel sizes allow no closer isotropic fit.
function shows_exactly(axis, rect)
    x0, x1, y0, y1 = limits_rect(axis)
    w, h = panel_size(axis)
    pixel = max((x1 - x0) / w, (y1 - y0) / h)
    return x0 <= rect[1] + 1.0e-12 && x1 >= rect[2] - 1.0e-12 &&
        y0 <= rect[3] + 1.0e-12 && y1 >= rect[4] - 1.0e-12 &&
        (x1 - x0) - (rect[2] - rect[1]) < pixel && (y1 - y0) - (rect[4] - rect[3]) < pixel &&
        (x0 + x1) / 2 ≈ (rect[1] + rect[2]) / 2 && (y0 + y1) / 2 ≈ (rect[3] + rect[4]) / 2
end

@testset "Animations keep the view and colours fixed" begin
    animation = MakieExt._animation_scene(
        RUN.store, :velocity_vorticity;
        diagnostics = [:kinetic_energy, :cells]
    )
    axis = only(animation.field_axes)
    image = field_image(animation.plot)
    range = Tuple(image.colorrange[])
    largest = maximum(frame.stats[:velocity_vorticity].absmax for frame in animation.frames)
    @test range[2] ≈ largest && range[1] ≈ -largest
    @test length(animation.frames) == length(FRAMES)
    @test length(animation.diagnostic_axes) == 2
    settle!(animation.figure)
    field_limits = limits_rect(axis)
    diagnostic_limits = [diagnostic.finallimits[] for diagnostic in animation.diagnostic_axes]
    for frame in reverse(animation.frames)
        animation.frame[] = frame
        settle!(animation.figure)
        # Regression: every frame shows the domain, never default limits.
        @test limits_rect(axis) == field_limits
        @test shows_exactly(axis, (0, 1, 0, 1))
        @test Tuple(image.colorrange[]) == range
        @test isequal(
            image[3][], rasterize(
                frame, :velocity_vorticity, image_rect(image),
                size(image[3][])
            )
        )
        @test [diagnostic.finallimits[] for diagnostic in animation.diagnostic_axes] ==
            diagnostic_limits
    end
    # A requested colour range wins; the registered one wins over the data.
    @test Tuple(
        field_image(
            MakieExt._animation_scene(
                RUN.store, :velocity_vorticity;
                colorrange = (-1, 1)
            ).plot
        ).colorrange[]
    ) == (-1, 1)
    @test Tuple(field_image(MakieExt._animation_scene(RUN.store, :speed).plot).colorrange[]) ==
        (0, 2)
    zoomed = MakieExt._animation_scene(
        RUN.store, :speed; camera = (0.25, 0.75, 0.4, 0.6),
        size = (600, 300)
    )
    settle!(zoomed.figure)
    @test shows_exactly(only(zoomed.field_axes), (0.25, 0.75, 0.4, 0.6))

    mktempdir() do dir
        path = joinpath(dir, "vorticity.gif")
        @test export_animation(
            path, RUN.recorder; field = :speed, framerate = 4,
            decorations = :none, title = nothing, size = (300, 200),
            colormap = :viridis
        ) == path
        frames = video_frames(path, gif_size(path))
        @test length(frames) == length(FRAMES)
        # The field fills the same square region of every frame: the first
        # run of coloured pixel columns, left of the colour bar.
        regions = map(frames) do pixels
            colored = [any(pixels[i, j, :] .< 0xf0) for i in axes(pixels, 1), j in axes(pixels, 2)]
            columns = vec(any(colored; dims = 1))
            left = findfirst(columns)
            right = findnext(!, columns, left) - 1
            rows = findall(vec(any(colored[:, left:right]; dims = 2)))
            ((first(rows), last(rows)), (left, right))
        end
        @test all(==(first(regions)), regions)
        (top, bottom), (left, right) = first(regions)
        @test abs((bottom - top) - (right - left)) <= 2
        @test bottom - top > 0.6 * gif_size(path)[2]
        # A time window restricts the stored keyframes.
        window = (FRAMES[2].time, last(FRAMES).time)
        @test length(MakieExt._animation_scene(RUN.store, :speed; window).frames) ==
            length(FRAMES) - 1
        video = joinpath(dir, "speed.mp4")
        @test export_animation(
            video, RUN.store; field = :speed, window,
            diagnostics = :all, framerate = 4
        ) == video
        @test filesize(video) > 0
    end
    @test_throws ArgumentError export_animation("unused.gif", RUN.store; field = :missing)
    @test_throws ArgumentError export_animation(
        "unused.gif", RUN.store; field = :speed,
        window = (1.0, 2.0)
    )
    @test_throws ArgumentError export_animation(
        "unused.gif", RUN.store; field = :speed,
        source = :both
    )
    @test_throws ArgumentError export_animation(
        "unused.gif", RUN.store; field = :speed,
        every = 2
    )
    @test_throws ArgumentError export_animation(
        "unused.gif",
        Recorder(VisualizationFieldRegistry()); field = :speed
    )
end

# A run whose store keeps only the newest preview, so that live frames are
# thinned from the store while they are recorded.
function thinned_run(; preview = EveryStep())
    mesh = BlockTreeMesh(block_size = (8, 8), max_level = 1)
    flow = FlowSimulation(
        mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.01);
        velocity = (x, y, t, axis) -> axis == 1 ? sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y),
        diagnostics = (KineticEnergy(),), dt = 0.005
    )
    fields = VisualizationFieldRegistry()
    register_field!(fields, :speed, flow.velocity, CentredMagnitude())
    recorder = Recorder(
        fields; diagnostics = flow, keyframes = EquispacedTimes(3),
        preview, max_capture_overhead = 1.0, retention = Retention(preview = :none)
    )
    return (; flow, recorder, store = snapshot_store(recorder))
end

committed(store) = let usage = store_usage(store)
    usage.frames + usage.thinned_previews
end

@testset "Live recordings encode the frame stream without slowing the run" begin
    plain = thinned_run()
    elapsed_plain = @elapsed plain_result = simulate!(plain.flow, 0.1; observer = plain.recorder)
    mktempdir() do dir
        live = thinned_run()
        path = joinpath(dir, "live.gif")
        recording = export_animation(
            path, live.recorder; field = :speed, source = :live,
            diagnostics = [:kinetic_energy], size = (400, 250), capacity = 1000
        )
        @test isopen(recording) && !isfile(path)
        # Reading the store from another task, as a viewer scrubbing its
        # history would, never alters the recording. The reader shares the
        # solver's thread and runs at every step boundary.
        reading = Threads.Atomic{Bool}(true)
        reads = Threads.Atomic{Int}(0)
        reader = @async while reading[]
            if !isempty(frames(live.store))
                frame_at(live.store, 0.05)
                latest_frame(live.store)
                diagnostic_history(live.store, :kinetic_energy)
                Threads.atomic_add!(reads, 1)
            end
            yield()
        end
        elapsed_live = @elapsed result = simulate!(live.flow, 0.1; observer = live.recorder)
        reading[] = false
        wait(reader)
        close(recording)
        @test !isopen(recording)
        # Same run, results, and roughly the same cost as without recording.
        @test result.steps == plain_result.steps
        @test last(diagnostic_history(live.flow, :kinetic_energy).values) ==
            last(diagnostic_history(plain.flow, :kinetic_energy).values)
        @test elapsed_live < 4 * elapsed_plain + 5
        # Every committed frame was encoded, including the previews that the
        # store thinned; one video frame per encoded frame.
        @test store_usage(live.store).thinned_previews > 0
        @test recording.dropped == 0 && recording.skipped == 0
        @test recording.encoded == committed(live.store)
        @test length(video_frames(path, gif_size(path))) == recording.encoded
        @test reads[] > 0

        # A small queue drops frames instead of waiting, and `every` thins
        # the stream; all frames are accounted for.
        dropping = thinned_run()
        recording = export_animation(
            joinpath(dir, "drop.mp4"), dropping.recorder;
            field = :speed, source = :live, capacity = 1, every = 2
        )
        simulate!(dropping.flow, 0.1; observer = dropping.recorder)
        close(recording)
        @test recording.encoded + recording.dropped + recording.skipped ==
            committed(dropping.store)
        @test recording.encoded >= 1 && recording.skipped >= recording.encoded - 1
    end
    @test_throws ArgumentError export_animation(
        "unused.gif", RUN.store; field = :speed,
        source = :live
    )
    @test_throws ArgumentError export_animation(
        "unused.gif", RUN.recorder; field = :speed,
        source = :live, window = (0, 1)
    )
    @test_throws ArgumentError export_animation(
        "unused.png", RUN.recorder; field = :speed,
        source = :live
    )
    @test_throws ArgumentError export_animation(
        "unused.gif", RUN.recorder; field = :speed,
        source = :live, every = 0
    )
end

png_size(path) = open(path) do io
    bytes = read(io, 24)
    (Int(ntoh(reinterpret(UInt32, bytes[17:20])[1])), Int(ntoh(reinterpret(UInt32, bytes[21:24])[1])))
end

@testset "Stills share a view and a colour range" begin
    mktempdir() do dir
        times = [frame.time for frame in FRAMES]
        path = joinpath(dir, "panels.png")
        @test export_figure(
            path, RUN.store; field = :velocity_vorticity, at = times,
            blocks = true, diagnostics = [:kinetic_energy, :cells]
        ) == path
        width, height = png_size(path)
        @test width > height > 0
        single = joinpath(dir, "single.pdf")
        @test export_figure(single, last(FRAMES); field = :velocity_x) == single
        @test filesize(single) > 0
        registry = joinpath(dir, "registry.png")
        @test export_figure(
            registry, RUN.fields; field = :speed, camera = (0, 0.5, 0, 0.5),
            size = (400, 300)
        ) == registry
        @test png_size(registry) == (800, 600)
    end
    @test_throws ArgumentError export_figure(
        "unused.png", last(FRAMES); field = :speed,
        at = 0.0
    )
    @test_throws ArgumentError export_figure("unused.png", RUN.store; field = :missing)
    @test_throws ArgumentError export_figure(
        "unused.png", RUN.store; field = :speed,
        camera = :zoom
    )
end
