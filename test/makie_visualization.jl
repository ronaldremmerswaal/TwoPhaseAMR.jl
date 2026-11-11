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
    @test length(plot_axes) == 4
    diagnostic_lines(axis) = [plot for plot in axis.scene.plots if plot isa Lines]
    for (axis, name) in zip(
            plot_axes, (:kinetic_energy, :linear_momentum, :cells, :energy_bound),
        )
        history = diagnostic_history(RUN.store, name)
        @test length(diagnostic_lines(axis)) == size(history.values, 2)
        for (column, line) in pairs(diagnostic_lines(axis))
            @test getindex.(line[1][], 1) == Float32.(history.times)
            @test getindex.(line[1][], 2) == Float32.(history.values[:, column])
        end
    end
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
    @test length(diagnostic_lines(plot_axes[1])) == 2
    @test getindex.(diagnostic_lines(plot_axes[1])[2][1][], 2) ==
        Float32.(diagnostic_history(RUN.store, :energy_bound).values[:, 1])
    @test length(diagnostic_lines(plot_axes[3])) == 2
    reference_cells = getindex.(diagnostic_lines(plot_axes[3])[2][1][], 2)
    @test length(reference_cells) == 256 && all(==(64.0f0), reference_cells)
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

# Detached store with controlled signed values, extreme hidden values, and
# the real captured field frames used by the animation renderer.
function diagnostic_plot_store()
    store = SnapshotStore(Retention())
    for frame in FRAMES
        push!(store.records, frame.info)
        store.frames[frame.id] = frame
    end
    times = collect(0.0:0.01:0.05)
    dimension = PhysicalDimension(0, 1, 0)
    pair_info = DiagnosticInfo(; label = "Pair", dimension, components = ["small", "large"])
    peer_info = DiagnosticInfo(; label = "Peer", dimension)
    reference_info = DiagnosticInfo(; label = "Reference", dimension, components = ["small", "large"])
    defect_info = DiagnosticInfo(; label = "Defect", kind = :defect, scale = :log)
    empty_info = DiagnosticInfo(; label = "Empty", components = ["a", "b"])
    function add_mirror(name, info, columns)
        sample_times = isempty(columns) ? Float64[] : copy(times)
        mirror = TwoPhaseAMR.MirroredDiagnostic(
            name, info, nothing, length(sample_times),
            TwoPhaseAMR._component_names(info, length(columns)), sample_times,
            collect(0:(length(sample_times) - 1)), columns,
        )
        push!(store.diagnostics, mirror)
        return store.diagnostic_index[name] = length(store.diagnostics)
    end
    add_mirror(:pair, pair_info, [[-2.0, 0, 1, 2, 3, 4], fill(1.0e6, 6)])
    add_mirror(:peer, peer_info, [collect(5.0:10.0)])
    add_mirror(:reference_pair, reference_info, [collect(1.0:6.0), fill(1.0e8, 6)])
    add_mirror(:defect, defect_info, [[-2.0, 0, 1, 2, 3, 4]])
    add_mirror(:empty, empty_info, Vector{Float64}[])
    return store
end

@testset "Per-group diagnostic scale and component visibility" begin
    store = diagnostic_plot_store()
    pair_option = (;
        names = [:peer, :pair], scale = :log,
        hidden = [(; name = :pair, component = 2)],
    )
    names = [:pair, :peer, :defect]
    figure = Figure(size = (700, 500))
    axes = diagnosticsgrid!(
        figure[1, 1], store; names, diagnostic_options = [pair_option],
        references = Dict(:pair => :reference_pair),
    )
    settle!(figure)
    @test length(axes) == 2
    @test axes[1].yscale[] === log10
    @test axes[2].yscale[] === log10 # Unconfigured group keeps metadata.
    labelled(axis) = [plot for plot in axis.scene.plots if plot isa Lines]
    @test length(labelled(axes[1])) == 3
    @test length(last(Makie.get_labeled_plots(axes[1]; merge = false, unique = false))) == 3
    @test getindex.(labelled(axes[1])[2][1][], 2) == Float32.(1:6)
    @test getindex.(labelled(axes[1])[3][1][], 2) == Float32.(5:10)
    @test limits_rect(axes[1])[3] > 0
    @test limits_rect(axes[1])[4] < 100 # Neither hidden million-valued line sets the range.
    pair_values = getindex.(first(labelled(axes[1]))[1][], 2)
    @test isnan(pair_values[1]) && isnan(pair_values[2])
    @test pair_values[3:end] == Float32[1, 2, 3, 4]
    defect_values = getindex.(only(labelled(axes[2]))[1][], 2)
    @test isnan(defect_values[1]) && isnan(defect_values[2])
    @test defect_values[3:end] == Float32[1, 2, 3, 4]

    # Function and named references both follow their corresponding component.
    figure = Figure()
    function_axes = diagnosticsgrid!(
        figure[1, 1], store; names = [:pair, :peer],
        diagnostic_options = [pair_option], references = Dict(:pair => t -> (2.0, 1.0e9)),
    )
    settle!(figure)
    @test length(labelled(only(function_axes))) == 3
    reference_values = getindex.(labelled(only(function_axes))[2][1][], 2)
    @test length(reference_values) == 256 && all(==(2.0f0), reference_values)
    @test limits_rect(only(function_axes))[4] < 100

    linear_option = (; names = [:defect], scale = :linear, hidden = NamedTuple[])
    figure = Figure()
    linear_axes = diagnosticsgrid!(
        figure[1, 1], store; names, diagnostic_options = [pair_option, linear_option],
    )
    settle!(figure)
    @test linear_axes[2].yscale[] === identity
    @test getindex.(only(labelled(linear_axes[2]))[1][], 2)[1:2] == Float32[-2, 0]

    # A log axis with no visible series has valid limits and no legend entry.
    hidden_all = (;
        names = [:pair, :peer], scale = :log,
        hidden = [
            (; name = :pair, component = 1), (; name = :pair, component = 2),
            (; name = :peer, component = 1),
        ],
    )
    figure = Figure()
    empty_axes = diagnosticsgrid!(
        figure[1, 1], store; names = [:pair, :peer],
        diagnostic_options = [hidden_all], cursor = 0.02
    )
    settle!(figure)
    @test isempty(labelled(only(empty_axes)))
    @test isempty(last(Makie.get_labeled_plots(only(empty_axes); merge = false, unique = false)))
    @test 0 < limits_rect(only(empty_axes))[3] < limits_rect(only(empty_axes))[4]

    # Explicit options retain an unsampled group, including its metadata
    # component count, even for a static export.
    blank = (; names = [:empty], scale = :log, hidden = [(; name = :empty, component = 2)])
    figure = Figure()
    blank_axis = only(diagnosticsgrid!(figure[1, 1], store; names = [:empty], diagnostic_options = [blank]))
    settle!(figure)
    @test length(labelled(blank_axis)) == 1
    @test isempty(only(labelled(blank_axis))[1][])

    bad = (
        [(; names = [:pair], scale = :log, hidden = NamedTuple[])], # Not the whole selected group.
        [(; names = [:pair, :pair], scale = :log, hidden = NamedTuple[])],
        [(; names = [:pair, :peer], scale = :symlog, hidden = NamedTuple[])],
        [(; names = [:pair, :peer], scale = :log, hidden = [(; name = :pair, component = 0)])],
        [(; names = [:pair, :peer], scale = :log, hidden = [(; name = :pair, component = 3)])],
        [(; names = [:pair, :peer], scale = :log, hidden = [(; name = :pair, component = true)])],
        [(; names = [:pair, :peer], scale = :log, hidden = [(; name = :defect, component = 1)])],
        [
            (;
                names = [:pair, :peer], scale = :log, hidden = [
                    (; name = :pair, component = 1), (; name = :pair, component = 1),
                ],
            ),
        ],
        [pair_option, pair_option],
        [(; names = [:pair, :peer], scale = :log, hidden = NamedTuple[], extra = 1)],
    )
    for invalid in bad
        @test_throws ArgumentError diagnosticsgrid!(
            Figure()[1, 1], store; names = [:pair, :peer], diagnostic_options = invalid,
        )
    end
    @test_throws ArgumentError MakieExt._panel_options(; diagnostic_options = [pair_option])

    # The stored animation scene and the live scene share the same options.
    animation = MakieExt._animation_scene(
        store, :speed; diagnostics = names, diagnostic_options = [pair_option], size = (600, 400),
    )
    @test animation.diagnostic_axes[1].yscale[] === log10
    @test length(labelled(animation.diagnostic_axes[1])) == 2
    for frame in animation.frames
        animation.frame[] = frame
        settle!(animation.figure)
        @test animation.diagnostic_axes[1].yscale[] === log10
        @test limits_rect(animation.diagnostic_axes[1])[4] < 100
    end
    until = Observable(0.01)
    live_options = MakieExt._panel_options(;
        diagnostics = [:pair, :peer], diagnostic_options = [pair_option], size = (600, 400),
    )
    live = MakieExt._export_scene(
        store, [Observable(first(FRAMES))], :speed, (0, 2), :viridis, live_options;
        cursor = until, until,
    )
    settle!(live.figure)
    live_axis = only(live.diagnostic_axes)
    @test length(first(labelled(live_axis))[1][]) == 2
    Base.@lock store.lock begin
        for (name, values) in ((:pair, (20.0, 1.0e9)), (:peer, (12.0,)))
            mirror = store.diagnostics[store.diagnostic_index[name]]
            push!(mirror.times, 0.06)
            push!(mirror.steps, 6)
            for (column, value) in zip(mirror.columns, values)
                push!(column, value)
            end
        end
    end
    until[] = 0.06
    settle!(live.figure)
    @test last(first(labelled(live_axis))[1][])[2] == 20
    @test length(labelled(live_axis)) == 2
    @test last(labelled(live_axis)[2][1][])[2] == 12
    @test live_axis.yscale[] === log10 && limits_rect(live_axis)[4] < 100
    until[] = 0.01
    settle!(live.figure)
    @test length(first(labelled(live_axis))[1][]) == 2
    @test live_axis.yscale[] === log10 && limits_rect(live_axis)[4] < 100

    # Verify actual CairoMakie/FFmpeg output changes, not just scene metadata.
    mktempdir() do directory
        default = joinpath(directory, "default.gif")
        configured = joinpath(directory, "configured.gif")
        export_animation(default, store; field = :speed, diagnostics = names, size = (600, 400))
        export_animation(
            configured, store; field = :speed, diagnostics = names,
            diagnostic_options = [pair_option], size = (600, 400)
        )
        default_frames = video_frames(default, gif_size(default))
        configured_frames = video_frames(configured, gif_size(configured))
        @test length(default_frames) == length(configured_frames) == 41
        @test first(default_frames) != first(configured_frames)
    end
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
    @test length(animation.frames) == 41
    @test length(animation.diagnostic_axes) == 2
    settle!(animation.figure)
    field_limits = limits_rect(axis)
    diagnostic_limits = [diagnostic.finallimits[] for diagnostic in animation.diagnostic_axes]
    titles = String[]
    for frame in reverse(animation.frames)
        animation.frame[] = frame
        settle!(animation.figure)
        # Regression: every frame shows the domain, never default limits.
        @test limits_rect(axis) == field_limits
        @test shows_exactly(axis, (0, 1, 0, 1))
        @test Tuple(image.colorrange[]) == range
        push!(titles, axis.title[])
        @test isequal(
            image[3][], rasterize(
                frame, :velocity_vorticity, image_rect(image),
                size(image[3][])
            )
        )
        @test [diagnostic.finallimits[] for diagnostic in animation.diagnostic_axes] ==
            diagnostic_limits
    end
    @test length(unique(titles)) == length(FRAMES) # sparse sampling repeats frames
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
    # A world-rectangle camera remains isotropic when diagnostics share a
    # differently shaped output figure.
    zoomed_diagnostics = MakieExt._animation_scene(
        RUN.store, :speed; camera = (0.1, 0.9, 0.2, 0.7),
        diagnostics = [:kinetic_energy], size = (900, 400),
    )
    settle!(zoomed_diagnostics.figure)
    output_axis = only(zoomed_diagnostics.field_axes)
    x0, x1, y0, y1 = limits_rect(output_axis)
    width, height = panel_size(output_axis)
    @test isapprox((x0 + x1) / 2, 0.5; atol = 1.0e-6)
    @test isapprox((y0 + y1) / 2, 0.45; atol = 1.0e-6)
    @test isapprox((x1 - x0) / width, (y1 - y0) / height; rtol = 1.0e-5)
    @test x0 <= 0.1 + 1.0e-6 && x1 >= 0.9 - 1.0e-6
    @test y0 <= 0.2 + 1.0e-6 && y1 >= 0.7 - 1.0e-6
    # A wide current view must not consume the entire fixed-size output and
    # collapse the diagnostics on its right. Both panels stay usable and the
    # requested rectangle remains visible with isotropic field pixels.
    wide = MakieExt._animation_scene(
        RUN.store, :speed; camera = (-2.5, 3.5, 0, 1),
        diagnostics = [:kinetic_energy], arrangement = :right, size = (640, 360),
    )
    settle!(wide.figure)
    wide_field = only(wide.field_axes)
    wide_diagnostic = only(wide.diagnostic_axes)
    field_width, field_height = panel_size(wide_field)
    diagnostic_width, diagnostic_height = panel_size(wide_diagnostic)
    @test field_width >= 100 && field_height >= 100
    @test diagnostic_width >= 100 && diagnostic_height >= 100
    x0, x1, y0, y1 = limits_rect(wide_field)
    @test x0 <= -2.5 + 1.0e-6 && x1 >= 3.5 - 1.0e-6
    @test y0 <= 1.0e-6 && y1 >= 1 - 1.0e-6
    @test isapprox((x1 - x0) / field_width, (y1 - y0) / field_height; rtol = 1.0e-5)
    for arrangement in (:auto, :right, :below)
        arranged = MakieExt._animation_scene(
            RUN.store, :speed; diagnostics = [:kinetic_energy, :cells],
            arrangement, size = (900, 600),
        )
        settle!(arranged.figure)
        field_box = only(arranged.field_axes).layoutobservables.computedbbox[]
        diagnostic_boxes = [axis.layoutobservables.computedbbox[] for axis in arranged.diagnostic_axes]
        @test length(diagnostic_boxes) == 2
        if arrangement === :below
            @test all(box -> maximum(box)[2] <= minimum(field_box)[2], diagnostic_boxes)
            @test minimum(diagnostic_boxes[1])[1] < minimum(diagnostic_boxes[2])[1]
        else
            @test all(box -> minimum(box)[1] >= maximum(field_box)[1], diagnostic_boxes)
            @test minimum(diagnostic_boxes[1])[2] > minimum(diagnostic_boxes[2])[2]
        end
    end
    # The historical automatic layout still places diagnostics below
    # multiple field panels, even though single-field animations default right.
    multi_options = MakieExt._panel_options(;
        diagnostics = [:kinetic_energy, :cells], size = (900, 600),
    )
    multi = MakieExt._export_scene(
        RUN.store, [Observable(FRAMES[1]), Observable(FRAMES[2])],
        :speed, (0, 2), :viridis, multi_options; cursor = nothing,
    )
    settle!(multi.figure)
    field_bottom = minimum(minimum(axis.layoutobservables.computedbbox[])[2] for axis in multi.field_axes)
    @test all(axis -> maximum(axis.layoutobservables.computedbbox[])[2] <= field_bottom, multi.diagnostic_axes)

    mktempdir() do dir
        path = joinpath(dir, "vorticity.gif")
        @test export_animation(
            path, RUN.recorder; field = :speed, fps = 8, duration = 1.0,
            decorations = :none, title = nothing, size = (300, 200),
            colormap = :viridis
        ) == path
        frames = video_frames(path, gif_size(path))
        @test length(frames) == 9 # round(8 × 1) + 1, both ends included
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
        # The default 20 fps over 2 s gives 41 output frames; the four
        # keyframes repeat instead of mapping one-to-one.
        titled = joinpath(dir, "titled.gif")
        export_animation(titled, RUN.store; field = :speed, size = (300, 200))
        decoded = video_frames(titled, gif_size(titled))
        @test length(decoded) == 41
        @test length(unique(decoded)) <= length(FRAMES)
        @test !allunique(decoded)
        # A time window keeps only candidates inside it and spans it linearly.
        window = (FRAMES[2].time, last(FRAMES).time)
        scene = MakieExt._animation_scene(RUN.store, :speed; window)
        @test length(scene.frames) == 41
        @test all(index -> 1 <= index <= length(FRAMES) - 1, scene.schedule.indices)
        @test first(scene.schedule.indices) == 1 && last(scene.schedule.indices) == length(FRAMES) - 1
        @test first(scene.frames).id == FRAMES[2].id && last(scene.frames).id == last(FRAMES).id
        video = joinpath(dir, "speed.mp4")
        video_progress = Tuple{Int, Int}[]
        @test export_animation(
            video, RUN.store; field = :speed, window, fps = 4, duration = 1.0,
            diagnostics = :all,
            progress = (index, total) -> push!(video_progress, (index, total)),
        ) == video
        @test filesize(video) > 0
        @test video_progress == [(i, 5) for i in 1:5]
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
        "unused.gif", RUN.store; field = :speed, duration = 2.0, nframes = 5,
    )
    @test_throws ArgumentError export_animation(
        "unused.gif", RUN.store; field = :speed, sim_per_second = 1.0,
    )
    @test_throws ArgumentError export_animation(
        "unused.gif", RUN.store; field = :speed, frames = :everything,
    )
    @test_throws ArgumentError export_animation(
        "unused.gif", RUN.store; field = :speed, arrangement = :diagonal,
    )
    @test_throws ArgumentError export_animation(
        "unused.gif",
        Recorder(VisualizationFieldRegistry()); field = :speed
    )
end

@testset "Stored animation progress and cancellation" begin
    @test length(FRAMES) > 1
    mktempdir() do dir
        path = joinpath(dir, "progress.gif")
        events = Tuple{Int, Int}[]
        @test export_animation(
            path, RUN.store; field = :speed, size = (300, 200), fps = 4, nframes = length(FRAMES),
            progress = (index, total) -> push!(events, (index, total)),
            cancel = () -> false,
        ) == path
        @test events == [(index, length(FRAMES)) for index in eachindex(FRAMES)]
        @test length(video_frames(path, gif_size(path))) == length(FRAMES)
        original = read(path)

        # Cancel after a real submitted frame. The previous destination stays
        # intact and the private FFmpeg output is not published.
        empty!(events)
        requested = Ref(false)
        @test isnothing(
            export_animation(
                path, RUN.store; field = :speed, size = (300, 200), fps = 4, nframes = length(FRAMES),
                progress = (index, total) -> begin
                    push!(events, (index, total))
                    requested[] = true
                end,
                cancel = () -> requested[],
            )
        )
        @test events == [(1, length(FRAMES))]
        @test read(path) == original
        @test readdir(dir) == ["progress.gif"]
        new_path = joinpath(dir, "cancelled.gif")
        requested[] = false
        @test isnothing(
            export_animation(
                new_path, RUN.store; field = :speed, size = (300, 200), fps = 4, nframes = length(FRAMES),
                progress = (index, total) -> (requested[] = true),
                cancel = () -> requested[],
            )
        )
        @test !isfile(new_path) && readdir(dir) == ["progress.gif"]

        # Cancellation immediately after finalization still discards the
        # staged result instead of replacing the old destination.
        checks = Ref(0)
        @test isnothing(
            export_animation(
                path, RUN.store; field = :speed, size = (300, 200), fps = 4, nframes = length(FRAMES),
                cancel = () -> (checks[] += 1; checks[] > length(FRAMES) + 2),
            )
        )
        @test checks[] == length(FRAMES) + 3
        @test read(path) == original
        @test readdir(dir) == ["progress.gif"]

        failed = joinpath(dir, "failed.gif")
        @test_throws ErrorException export_animation(
            failed, RUN.store; field = :speed, size = (300, 200),
            progress = (index, total) -> error("progress callback failed"),
        )
        @test !isfile(failed) && read(path) == original
        @test_throws ArgumentError export_animation(
            failed, RUN.store; field = :speed, cancel = () -> "not a boolean",
        )
        @test_throws ArgumentError export_animation(
            failed, RUN.store; field = :speed, progress = 1,
        )
        @test_throws ArgumentError export_animation(
            failed, RUN.recorder; field = :speed, source = :live, cancel = () -> false,
        )
        @test readdir(dir) == ["progress.gif"]
    end
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
    add_diagnostic!(
        flow, :signed_pair, state -> (state.time - 0.03, 1000.0 + state.time);
        label = "Signed pair", components = ["signal", "large"],
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

@testset "Live recordings sample the frame stream linearly in simulation time" begin
    plain = thinned_run()
    elapsed_plain = @elapsed plain_result = simulate!(plain.flow, 0.1; observer = plain.recorder)
    mktempdir() do dir
        live = thinned_run()
        path = joinpath(dir, "live.gif")
        recording = export_animation(
            path, live.recorder; field = :speed, source = :live,
            diagnostics = [:signed_pair], arrangement = :below,
            diagnostic_options = [
                (; names = [:signed_pair], scale = :log, hidden = [(; name = :signed_pair, component = 2)]),
            ],
            size = (400, 300), capacity = 1000, fps = 20, sim_per_second = 0.1,
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
                newest_frame(live.store)
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
        # The stream kept frames that the store thinned, and output times are
        # a uniform simulation-time grid with Δt = 0.1 / 20 = 0.005.
        @test store_usage(live.store).thinned_previews > 0
        @test recording.dropped == 0
        @test recording.encoded == length(recording.ids) == length(recording.times)
        @test issorted(recording.ids)
        @test issorted(recording.times)
        @test first(recording.times) ≈ 0.0 && last(recording.times) ≈ 0.1
        @test abs(recording.encoded - 21) <= 1
        configured_frames = video_frames(path, gif_size(path))
        @test !isempty(configured_frames)
        @test reads[] > 0

        baseline = thinned_run()
        baseline_path = joinpath(dir, "live-default.gif")
        baseline_recording = export_animation(
            baseline_path, baseline.recorder; field = :speed, source = :live,
            diagnostics = [:signed_pair], arrangement = :below, size = (400, 300), capacity = 1000,
        )
        simulate!(baseline.flow, 0.1; observer = baseline.recorder)
        close(baseline_recording)
        baseline_frames = video_frames(baseline_path, gif_size(baseline_path))
        @test baseline_recording.encoded > 0 && !isempty(baseline_frames)
        @test last(baseline_frames) != last(configured_frames)

        # A dense stream skips stored frames instead of encoding each one.
        dense = thinned_run()
        dense_path = joinpath(dir, "live-dense.gif")
        dense_recording = export_animation(
            dense_path, dense.recorder; field = :speed, source = :live,
            capacity = 1000, fps = 10, sim_per_second = 0.2, # Δt = 0.02
        )
        simulate!(dense.flow, 0.1; observer = dense.recorder)
        close(dense_recording)
        @test dense_recording.encoded < committed(dense.store)
        @test issorted(dense_recording.ids)
        @test !isempty(video_frames(dense_path, gif_size(dense_path)))

        # A sparse stream repeats the last frame for output times in between.
        sparse = thinned_run(preview = EveryNSteps(5))
        sparse_path = joinpath(dir, "live-sparse.gif")
        sparse_recording = export_animation(
            sparse_path, sparse.recorder; field = :speed, source = :live,
            capacity = 1000, fps = 20, sim_per_second = 0.02, # Δt = 0.001
        )
        simulate!(sparse.flow, 0.1; observer = sparse.recorder)
        close(sparse_recording)
        @test sparse_recording.encoded > length(unique(sparse_recording.ids))
        @test issorted(sparse_recording.ids) && issorted(sparse_recording.times)

        # A small queue drops frames instead of waiting.
        dropping = thinned_run()
        dropping_recording = export_animation(
            joinpath(dir, "drop.mp4"), dropping.recorder;
            field = :speed, source = :live, capacity = 1,
        )
        simulate!(dropping.flow, 0.1; observer = dropping.recorder)
        close(dropping_recording)
        @test dropping_recording.encoded >= 1
        # Every output frame is accounted for once, and the accounting agrees
        # with the video that was written.
        @test dropping_recording.encoded ==
            length(dropping_recording.ids) == length(dropping_recording.times)
        @test issorted(dropping_recording.times)
        @test last(dropping_recording.times) <= last(step_log(dropping.store).time) + 1e-12
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
        "unused.gif", RUN.recorder; field = :speed, source = :live, duration = 2.0,
    )
    @test_throws ArgumentError export_animation(
        "unused.gif", RUN.recorder; field = :speed, source = :live, nframes = 5,
    )
    @test_throws ArgumentError export_animation(
        "unused.gif", RUN.recorder; field = :speed, source = :live, frames = :all,
    )
    @test_throws ArgumentError export_animation(
        "unused.gif", RUN.recorder; field = :speed, source = :live, progress = (i, n) -> nothing,
    )
    @test_throws ArgumentError export_animation(
        "unused.png", RUN.recorder; field = :speed,
        source = :live
    )
    fresh = thinned_run()
    @test_throws ArgumentError export_animation(
        "unused.gif", fresh.recorder; field = :speed, source = :live,
        diagnostics = [:signed_pair], diagnostic_options = [
            (; names = [:signed_pair], scale = :log, hidden = [(; name = :signed_pair, component = 3)]),
        ],
    )
    @test isempty(fresh.store.subscriptions)
end

@testset "Live recording stops at its stop time" begin
    mesh = BlockTreeMesh(block_size = (8, 8), max_level = 1)
    flow = FlowSimulation(
        mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.01);
        velocity = (x, y, t, axis) -> axis == 1 ? sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y),
        dt = 0.002,
    )
    fields = VisualizationFieldRegistry()
    register_field!(fields, :speed, flow.velocity, CentredMagnitude())
    recorder = Recorder(
        fields; preview = EveryStep(), max_capture_overhead = 1.0,
        retention = Retention(preview = :all),
    )
    store = snapshot_store(recorder)
    # Keep every step slow enough that the run is still advancing when the
    # recording stops.
    step_lock = ReentrantLock()
    last_midpoint = Ref(-Inf)
    function slow_force(x, y, midpoint, axis)
        Base.@lock step_lock begin
            if midpoint > last_midpoint[]
                last_midpoint[] = midpoint
                sleep(0.02)
            end
        end
        return axis == 1 ? 0.1 * cospi(2y) : 0.0
    end
    mktempdir() do dir
        path = joinpath(dir, "stop-time.gif")
        simulation = Threads.@spawn simulate!(flow, 0.2; observer = recorder, force = slow_force)
        recording = export_animation(
            path, recorder; field = :speed, source = :live,
            capacity = 1000, fps = 20, sim_per_second = 0.2,
        )
        timedwait(() -> length(frames(store)) >= 2, 120.0) === :ok ||
            error("the run produced too few frames before the stop")
        @test recording.stop_time === nothing
        # Stop the recording while the run keeps stepping: the frozen stop time
        # must bound the output, even though the encoder finalizes later.
        close(recording)
        stopped_at = recording.stop_time
        @test stopped_at !== nothing
        @test fetch(simulation).steps > 0
        continued = last(step_log(store).time)
        @test continued > stopped_at
        @test recording.encoded == length(recording.ids) == length(recording.times)
        @test issorted(recording.times)
        @test maximum(recording.times) <= stopped_at + 1e-12
        @test isfile(path) && filesize(path) > 0
        @test !isempty(video_frames(path, gif_size(path)))
    end
    @test isempty(store.subscriptions)
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

# Pixels that are clearly red, as red overlays over a grey field are.
red_pixels(pixels) = count(
    Int(pixels[i, j, 1]) - max(Int(pixels[i, j, 2]), Int(pixels[i, j, 3])) > 100
        for i in axes(pixels, 1), j in axes(pixels, 2)
)
luminance(color) = (
    Makie.Colors.red(color) + Makie.Colors.green(color) +
        Makie.Colors.blue(color)
) / 3
# The scale bar's backdrop, bar, and label: the axis scene's pixel-space plots.
function scalebar_plots(axis)
    overlays = filter(plot -> plot.space[] === :pixel, axis.scene.plots)
    return (
        backdrop = only(filter(plot -> plot isa Makie.Poly, overlays)),
        bar = only(filter(plot -> plot isa Makie.LineSegments, overlays)),
        label = only(filter(plot -> plot isa Makie.Text, overlays)),
    )
end

@testset "Overlay colours and the scale bar follow the theme" begin
    # A circular interface on 4 × 4 blocks, over a field drawn in greys.
    mesh = BlockTreeMesh(block_size = (8, 8), base_mesh = (32, 32))
    interface = InterfaceField(mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.25)
    fields = VisualizationFieldRegistry()
    register_default_fields!(fields, :volume_fraction, interface)
    mktempdir() do dir
        # The pixels of a still of the grey field with the given overlays.
        function still(name; kwargs...)
            path = export_figure(
                joinpath(dir, name), fields; field = :volume_fraction,
                colormap = :grays, size = (300, 300), kwargs...
            )
            return only(video_frames(path, png_size(path)))
        end
        # Interface and block outlines take the requested colours.
        @test red_pixels(still("default.png"; blocks = true)) == 0
        @test red_pixels(still("interface.png"; interfacecolor = :red)) > 100
        @test red_pixels(
            still(
                "blocks.png"; interface = false, blocks = true,
                blockcolor = :red
            )
        ) > 100
    end
    # Animations pass them to the field plot as well.
    animation = MakieExt._animation_scene(
        RUN.store, :speed; blocks = true,
        interfacecolor = :white, blockcolor = (:red, 0.5)
    )
    @test Makie.to_color(animation.plot.interfacecolor[]) == Makie.to_color(:white)
    @test Makie.to_color(animation.plot.blockcolor[]) == Makie.to_color((:red, 0.5))

    # The scale bar takes the axis text colour on a contrasting backdrop, in
    # light and dark themes.
    for (theme, ink) in ((Theme(), :black), (theme_dark(), :gray85))
        axis = with_theme(() -> fieldaxis(Figure()[1, 1]), theme; textcolor = ink)
        scalebar = scalebar_plots(axis)
        @test Makie.to_color(axis.titlecolor[]) == Makie.to_color(ink)
        @test Makie.to_color(scalebar.bar.color[]) == Makie.to_color(ink)
        @test Makie.to_color(scalebar.label.color[]) == Makie.to_color(ink)
        backdrop = Makie.to_color(scalebar.backdrop.color[])
        @test abs(luminance(backdrop) - luminance(Makie.to_color(ink))) > 0.5
    end
end
