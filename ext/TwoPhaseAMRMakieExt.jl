module TwoPhaseAMRMakieExt

using TwoPhaseAMR
import Makie

mutable struct MakieAMRVisualization{F, A, CP, FP, OP, CB, ME, CM, LB, PB, M} <:
    AbstractVisualization
    figure::F
    axis::A
    cell_plot::CP
    face_plot::FP
    outline_plot::OP
    interface_plot::Any
    solid_plot::Any
    colorbar::CB
    cell_colors::Makie.Observable
    face_mesh::Makie.Observable
    face_colors::Makie.Observable
    selection_points::Makie.Observable
    selected_cell_observable::Makie.Observable
    selected_face_observable::Makie.Observable
    current_field::Makie.Observable
    displayed_frame::Makie.Observable
    status::Makie.Observable
    registry::VisualizationFieldRegistry
    menu::ME
    checkpoint_menu::CM
    live_button::LB
    pause_button::PB
    session_ref::Base.RefValue{Any}
    updating_checkpoint_controls::Base.RefValue{Bool}
    session_status::Makie.Observable
    cell_mesh::Makie.Observable
    cell_segments::Makie.Observable
    block_segments::Makie.Observable
    interface_segments::Makie.Observable
    solid_mesh::Makie.Observable
    cell_geometry::AMRCellGeometry
    face_geometries::Dict{Int, AMRFaceGeometry}
    topology::VisualizationTopology
    cell_render_geometry::Base.RefValue{Any}
    face_render_geometry::Base.RefValue{Any}
    cell_plot_topology::Base.RefValue{Any}
    face_plot_topology::Base.RefValue{Any}
    face_plot_axis::Base.RefValue{Int}
    diagnostic_axis::Any
    diagnostic_data::Dict{Symbol, Any}
    diagnostic_count::Base.RefValue{Int}
    checkpoint_controls_count::Base.RefValue{Int}
    checkpoint_plot_state::Base.RefValue{Any}
    checkpoint_times::Any
    checkpoint_points::Any
    checkpoint_plot::Any
    mesh::M
end

_state_label(state::Symbol) = replace(string(state), '_' => ' ')

function _diagnostic_label(entry)
    isempty(entry.units) && return entry.label
    return "$(entry.label) [$(entry.units)]"
end

function _checkpoint_marker_height(visualization::MakieAMRVisualization)
    finite_values = Float32[]
    for observable in values(visualization.diagnostic_data)
        for point in observable[]
            isfinite(point[2]) && push!(finite_values, point[2])
        end
    end
    isempty(finite_values) && return 0.0f0
    lower, upper = extrema(finite_values)
    padding = lower == upper ? max(abs(upper) * 0.05f0, 0.5f0) :
        (upper - lower) * 0.05f0
    return upper + padding
end

function _refresh_checkpoint_plot!(
        visualization::MakieAMRVisualization, checkpoints
    )
    isnothing(visualization.diagnostic_axis) && return visualization
    height = _checkpoint_marker_height(visualization)
    state = (length(checkpoints), height)
    isequal(visualization.checkpoint_plot_state[], state) && return visualization
    visualization.checkpoint_plot_state[] = state
    times = Float32[checkpoint.frame.time for checkpoint in checkpoints]
    visualization.checkpoint_times[] = times
    visualization.checkpoint_points[] =
        Makie.Point2f[Makie.Point2f(time, height) for time in times]
    return visualization
end

function _refresh_diagnostic_time_limits!(axis, samples)
    isempty(samples) && return axis
    first_time = Float64(first(samples).time)
    latest_time = Float64(last(samples).time)
    span = latest_time - first_time
    padding = span > 0 ? 0.02 * span :
        max(0.02 * max(abs(latest_time), 1.0), 1.0e-3)
    Makie.xlims!(axis, first_time, latest_time + padding)
    # `xlims!` deliberately preserves the current y limits. The diagnostic
    # plots start empty, so preserving them would retain Makie's [0, 10]
    # fallback forever. Re-enable y autolimits after every data update.
    Makie.ylims!(axis, nothing, nothing)
    return axis
end

function _refresh_diagnostic_plot!(
        visualization::MakieAMRVisualization,
        session::VisualizationSession
    )
    isnothing(visualization.diagnostic_axis) && return visualization
    samples = visualization_diagnostics(session)
    count = visualization.diagnostic_count[]
    if count > length(samples)
        for observable in values(visualization.diagnostic_data)
            observable[] = Makie.Point2f[]
        end
        count = 0
    end
    if count < length(samples)
        for entry in visualization.registry.diagnostics
            observable = visualization.diagnostic_data[entry.name]
            points = observable[]
            for sample in @view samples[(count + 1):end]
                push!(points, Makie.Point2f(sample.time, sample[entry.name]))
            end
            observable[] = points
        end
        visualization.diagnostic_count[] = length(samples)
    end
    count == length(samples) ||
        _refresh_diagnostic_time_limits!(visualization.diagnostic_axis, samples)
    return _refresh_checkpoint_plot!(
        visualization, visualization_checkpoints(session)
    )
end

function _select_checkpoint_marker!(
        visualization::MakieAMRVisualization,
        index::Integer
    )
    session = _bound_session(visualization)
    isnothing(session) && return false
    checkpoints = visualization_checkpoints(session)
    1 <= index <= length(checkpoints) || return false
    select_checkpoint!(session, index)
    return true
end

function _bound_session(visualization::MakieAMRVisualization)
    reference = visualization.session_ref[]
    reference isa WeakRef || return nothing
    return reference.value
end

function _checkpoint_label(checkpoint, index)
    requested = round(checkpoint.requested_time; sigdigits = 6)
    stored = round(checkpoint.frame.time; sigdigits = 6)
    if requested == stored
        return "#$index · t = $stored"
    end
    return "#$index · requested $requested · stored $stored"
end

function _refresh_checkpoint_controls!(
        visualization::MakieAMRVisualization
    )
    session = _bound_session(visualization)
    isnothing(session) && return visualization
    checkpoints = visualization_checkpoints(session)
    information = TwoPhaseAMR._visualization_session_state(session)
    visualization.updating_checkpoint_controls[] = true
    try
        if visualization.checkpoint_controls_count[] != length(checkpoints)
            options = isempty(checkpoints) ? [("No checkpoints available", 0)] :
                [(_checkpoint_label(checkpoint, index), index)
                    for (index, checkpoint) in pairs(checkpoints)]
            visualization.checkpoint_menu.options[] = options
            visualization.checkpoint_controls_count[] = length(checkpoints)
        end
        selected = information.mode == :checkpoint ?
            something(information.selected_checkpoint, 0) : 0
        _set_if_changed!(visualization.checkpoint_menu.i_selected,
            1 <= selected <= length(checkpoints) ? selected : 0
        )
    finally
        visualization.updating_checkpoint_controls[] = false
    end

    frame = information.latest_frame
    time_text = isnothing(frame) ? "" :
        " · t = $(round(frame.time; sigdigits = 6))"
    if information.mode == :checkpoint &&
            !isnothing(information.selected_checkpoint)
        index = information.selected_checkpoint
        _set_if_changed!(visualization.session_status,
            "Checkpoint $index/$(length(checkpoints)) · " *
            _state_label(information.state)
        )
    else
        _set_if_changed!(visualization.session_status,
            "Live · $(_state_label(information.state))$time_text"
        )
    end
    _set_if_changed!(visualization.pause_button.label,
        information.state in (:pause_requested, :paused) ? "Resume" :
        information.state == :completed ? "Completed" :
        information.state == :failed ? "Failed" : "Pause"
    )
    _refresh_checkpoint_plot!(visualization, checkpoints)
    return visualization
end

function _makie_points(points)
    return Makie.Point2f[Makie.Point2f(point...) for point in points]
end

function _makie_mesh(geometry::AMRCellGeometry)
    points = _makie_points(geometry.points)
    faces = Makie.GLTriangleFace[
        Makie.GLTriangleFace(triangle...) for triangle in geometry.triangles
    ]
    return Makie.GeometryBasics.Mesh(points, faces)
end

function _makie_mesh(geometry::AMRFaceGeometry)
    points = _makie_points(geometry.points)
    faces = Makie.GLTriangleFace[
        Makie.GLTriangleFace(triangle...) for triangle in geometry.triangles
    ]
    return Makie.GeometryBasics.Mesh(points, faces)
end

function _makie_mesh(geometry::TwoPhaseAMR.PLICVisualizationGeometry)
    points = _makie_points(geometry.points)
    faces = Makie.GLTriangleFace[
        Makie.GLTriangleFace(triangle...) for triangle in geometry.triangles
    ]
    return Makie.GeometryBasics.Mesh(points, faces)
end

function _makie_mesh(geometry::PLICSolidGeometry)
    points = _makie_points(geometry.points)
    faces = Makie.GLTriangleFace[
        Makie.GLTriangleFace(triangle...) for triangle in geometry.triangles
    ]
    return Makie.GeometryBasics.Mesh(points, faces)
end

function _interface_overlay_points(registry, frame = nothing)
    points = Makie.Point2f[]
    for interface in registry.interfaces
        segments = isnothing(frame) ? plic_segments(interface) :
            frame.primitives[interface]
        for segment in segments
            push!(points, Makie.Point2f(segment[1]...))
            push!(points, Makie.Point2f(segment[2]...))
        end
    end
    return points
end

function _combined_solid_geometry(registry, frame = nothing)
    points = NTuple{2, Float32}[]
    triangles = NTuple{3, Int}[]
    for geometry in registry.geometries
        solid = isnothing(frame) ? plic_solid_geometry(geometry) :
            frame.primitives[geometry]
        offset = length(points)
        append!(points, NTuple{2, Float32}.(solid.points))
        append!(
            triangles, [
                (face[1] + offset, face[2] + offset, face[3] + offset)
                    for face in solid.triangles
            ]
        )
    end
    return PLICSolidGeometry(points, triangles)
end

function _refresh_plic_overlays!(visualization, frame = nothing)
    visualization.interface_segments[] =
        _interface_overlay_points(visualization.registry, frame)
    visualization.solid_mesh[] = _makie_mesh(
        _combined_solid_geometry(visualization.registry, frame)
    )
    return visualization
end

function _all_cell_segments(geometry::AMRCellGeometry)
    segments = Vector{eltype(geometry.points)}(
        undef, 8length(geometry.cells)
    )
    for (index, cell) in pairs(geometry.cells)
        outline = TwoPhaseAMR._cell_outline(geometry, cell)
        copyto!(segments, 8index - 7, outline, 1, 8)
    end
    return segments
end

function _field_title(entry)
    isempty(entry.units) && return entry.label
    return "$(entry.label) [$(entry.units)]"
end

function _automatic_colorrange(values; symmetric = false)
    if symmetric
        largest_magnitude = 0.0f0
        found_finite = false
        for value in values
            isfinite(value) || continue
            found_finite = true
            largest_magnitude = max(
                largest_magnitude, abs(Float32(value))
            )
        end
        found_finite || return (-1.0f0, 1.0f0)
        iszero(largest_magnitude) && return (-0.5f0, 0.5f0)
        return (-largest_magnitude, largest_magnitude)
    end

    lower = Inf32
    upper = -Inf32
    for value in values
        isfinite(value) || continue
        lower = min(lower, Float32(value))
        upper = max(upper, Float32(value))
    end
    isfinite(lower) || return (0.0f0, 1.0f0)
    if lower == upper
        padding = iszero(lower) ? 0.5f0 : max(0.01f0 * abs(lower), eps(lower))
        return (lower - padding, upper + padding)
    end
    return (lower, upper)
end

function _makie_colorrange(entry, values)
    if isnothing(entry.colorrange) || entry.colorrange === Makie.automatic
        symmetric = entry.source isa Union{
            TwoPhaseAMR.DerivedFaceVisualizationSource,
            TwoPhaseAMR.PhaseVelocityDerivedVisualizationSource,
        } &&
            entry.source.derived isa Vorticity
        return _automatic_colorrange(values; symmetric)
    end
    # Makie fixes the concrete type of the plot/colorbar Observable from its
    # initial value.  Interface fields default to the integer tuple `(0, 1)`;
    # leaving that tuple unconverted makes a later automatically scaled scalar
    # try to store Float32 limits in `Observable{Tuple{Int,Int}}`.
    return Float32.(entry.colorrange)
end

function _makie_colorrange(
        entry, values::TwoPhaseAMR.TwoPhaseVisualizationValues,
    )
    return _makie_colorrange(
        entry, TwoPhaseAMR._phase_visualization_colorrange_values(values),
    )
end

function _cell_description(cell)
    isnothing(cell) && return "No cell or face selected"
    return "Block $(cell.block), cell ($(cell.i), $(cell.j))"
end

function _face_description(topology, reference::FaceRef)
    face = reference.boundary ? topology.boundary_faces[reference.index] :
        topology.interior_faces[reference.index]
    kind = reference.boundary ? "Boundary" : "Interior"
    direction = face.axis == 1 ? "x-normal" : "y-normal"
    x, y = face.center
    return "$kind $direction face at ($(round(x; sigdigits = 6)), " *
        "$(round(y; sigdigits = 6)))"
end

function _replace_topology!(
        visualization::MakieAMRVisualization,
        topology::VisualizationTopology
    )
    visualization.topology === topology && return visualization
    visualization.topology = topology
    visualization.cell_geometry = topology.cell_geometry
    visualization.face_geometries = Dict{Int, AMRFaceGeometry}(
        1 => topology.face_geometries[1],
        2 => topology.face_geometries[2]
    )
    visualization.cell_segments[] = _makie_points(
        _all_cell_segments(topology.cell_geometry)
    )
    visualization.block_segments[] = _makie_points(
        topology.cell_geometry.block_segments
    )
    return _clear_selection!(visualization)
end

function _clear_selection!(visualization::MakieAMRVisualization)
    visualization.selected_cell_observable[] = nothing
    visualization.selected_face_observable[] = nothing
    visualization.selection_points[] = Makie.Point2f[]
    visualization.status[] = "No cell or face selected"
    return visualization
end

function _set_if_changed!(observable, value)
    isequal(observable[], value) || (observable[] = value)
    return observable
end

function _set_plot_style!(visualization::MakieAMRVisualization, entry, values)
    colorrange = _makie_colorrange(entry, values)
    _set_if_changed!(visualization.cell_plot.colormap, entry.colormap)
    _set_if_changed!(visualization.cell_plot.colorrange, colorrange)
    _set_if_changed!(visualization.face_plot.colormap, entry.colormap)
    _set_if_changed!(visualization.face_plot.colorrange, colorrange)
    _set_if_changed!(visualization.colorbar.colormap, entry.colormap)
    _set_if_changed!(visualization.colorbar.colorrange, colorrange)
    _set_if_changed!(visualization.axis.title, _field_title(entry))
    return visualization
end

function _replace_cell_plot!(
        visualization::MakieAMRVisualization,
        geometry, colors
    )
    old_plot = visualization.cell_plot
    old_limits = visualization.axis.finallimits[]
    mesh = Makie.Observable(_makie_mesh(geometry))
    color = Makie.Observable(colors)
    new_plot = Makie.mesh!(
        visualization.axis, mesh; color,
        colormap = old_plot.colormap[], colorrange = old_plot.colorrange[],
        nan_color = :transparent,
        shading = Makie.NoShading, inspectable = true,
        visible = old_plot.visible[]
    )
    delete!(visualization.axis, old_plot)
    Makie.limits!(visualization.axis, old_limits)
    visualization.cell_plot = new_plot
    visualization.cell_mesh = mesh
    visualization.cell_colors = color
    visualization.cell_render_geometry[] = geometry
    visualization.cell_plot_topology[] = visualization.topology
    return visualization
end

function _replace_face_plot!(
        visualization::MakieAMRVisualization,
        geometry, colors, axis::Int
    )
    old_plot = visualization.face_plot
    old_limits = visualization.axis.finallimits[]
    mesh = Makie.Observable(_makie_mesh(geometry))
    color = Makie.Observable(colors)
    new_plot = Makie.mesh!(
        visualization.axis, mesh; color,
        colormap = old_plot.colormap[], colorrange = old_plot.colorrange[],
        nan_color = :transparent,
        shading = Makie.NoShading, inspectable = true,
        visible = old_plot.visible[]
    )
    delete!(visualization.axis, old_plot)
    Makie.limits!(visualization.axis, old_limits)
    visualization.face_plot = new_plot
    visualization.face_mesh = mesh
    visualization.face_colors = color
    visualization.face_render_geometry[] = geometry
    visualization.face_plot_topology[] = visualization.topology
    visualization.face_plot_axis[] = axis
    return visualization
end

function _display_field!(
        visualization::MakieAMRVisualization, name::Symbol;
        clear_selection = true
    )
    entry = TwoPhaseAMR._field_entry(visualization.registry, name)
    frame = visualization.displayed_frame[]
    mode = isnothing(frame) ? visualization.registry.phase_rendering : frame.phase_rendering
    placement = TwoPhaseAMR._source_placement(entry.source, mode)
    values = isnothing(frame) ?
        TwoPhaseAMR._render_source_values(visualization.registry, entry.source, mode) : frame[name]

    if placement isa Cell
        discontinuous = mode == :plic && TwoPhaseAMR._is_discontinuous_source(entry.source)
        geometry = if discontinuous
            isnothing(frame) ? TwoPhaseAMR._phase_visualization_geometry(
                    entry.source, visualization.topology,
                ) : TwoPhaseAMR._frame_phase_geometry(frame, entry.source)
        else
            visualization.cell_geometry
        end
        colors = if discontinuous
            Float32.(TwoPhaseAMR._phase_visualization_vertex_values(values, geometry))
        else
            length(values) == length(visualization.cell_geometry.cells) ||
                throw(DimensionMismatch("cell field size differs from viewer geometry"))
            Float32.(TwoPhaseAMR._visualization_vertex_values(values))
        end
        _set_plot_style!(visualization, entry, colors)
        if visualization.cell_plot_topology[] !== visualization.topology ||
                visualization.cell_render_geometry[] !== geometry
            # A topology change resizes positions, indices, and colors at
            # once. Recreate the plot so Makie never observes a mixture of
            # old and new GPU buffers while its Observables update
            # synchronously.
            _replace_cell_plot!(
                visualization, geometry, colors
            )
        else
            visualization.cell_colors[] = colors
        end
        _set_if_changed!(visualization.cell_plot.visible, true)
        _set_if_changed!(visualization.face_plot.visible, false)
    else
        axis = placement isa Face{1} ? 1 : 2
        base_geometry = visualization.face_geometries[axis]
        discontinuous = mode == :plic && TwoPhaseAMR._is_discontinuous_source(entry.source)
        geometry = if discontinuous
            isnothing(frame) ? TwoPhaseAMR._phase_visualization_geometry(
                    entry.source, visualization.topology,
                ) : TwoPhaseAMR._frame_phase_geometry(frame, entry.source)
        else
            base_geometry
        end
        colors = if discontinuous
            Float32.(TwoPhaseAMR._phase_visualization_vertex_values(values, geometry))
        else
            length(values) == maximum(base_geometry.value_indices; init = 0) ||
                throw(DimensionMismatch("face field size differs from viewer geometry"))
            Float32.(
                TwoPhaseAMR._visualization_face_vertex_values(values, base_geometry)
            )
        end
        _set_plot_style!(visualization, entry, colors)
        if visualization.face_plot_topology[] !== visualization.topology ||
                visualization.face_render_geometry[] !== geometry ||
                visualization.face_plot_axis[] != axis
            _replace_face_plot!(visualization, geometry, colors, axis)
        else
            visualization.face_colors[] = colors
        end
        _set_if_changed!(visualization.cell_plot.visible, false)
        _set_if_changed!(visualization.face_plot.visible, true)
    end

    visualization.current_field[] = name
    clear_selection && _clear_selection!(visualization)
    return visualization
end

function TwoPhaseAMR.visualize(
        registry::VisualizationFieldRegistry; field = nothing,
        phase_rendering = registry.phase_rendering,
        show_blocks = true, show_cells = false,
        figure = (; size = (900, 700)), axis = (;), display = true
    )
    isempty(registry.entries) &&
        throw(ArgumentError("cannot visualize an empty field registry"))
    initial_name = isnothing(field) ? first(field_names(registry)) : Symbol(field)
    initial_entry = TwoPhaseAMR._field_entry(registry, initial_name)
    registry.phase_rendering = TwoPhaseAMR._check_phase_rendering(phase_rendering)
    mesh = registry.mesh::BlockTreeMesh
    cell_geometry = visualization_geometry(mesh)
    face_geometries = Dict{Int, AMRFaceGeometry}(
        1 => visualization_geometry(mesh, Face{1}()),
        2 => visualization_geometry(mesh, Face{2}())
    )
    topology = VisualizationTopology(
        cell_geometry,
        (face_geometries[1], face_geometries[2]),
        copy(mesh.faces), copy(mesh.boundary_faces)
    )

    initial_placement = TwoPhaseAMR._source_placement(initial_entry.source, phase_rendering)
    initial_values = TwoPhaseAMR._render_source_values(registry, initial_entry.source, phase_rendering)
    initial_discontinuous =
        phase_rendering == :plic && TwoPhaseAMR._is_discontinuous_source(initial_entry.source)
    initial_phase_geometry = initial_discontinuous ?
        TwoPhaseAMR._phase_visualization_geometry(
            initial_entry.source, topology,
        ) : nothing
    initial_cell_geometry = initial_placement isa Cell && initial_discontinuous ?
        initial_phase_geometry : cell_geometry
    cell_values = if initial_placement isa Cell
        initial_discontinuous ? Float32.(
                TwoPhaseAMR._phase_visualization_vertex_values(
                    initial_values, initial_cell_geometry,
                )
            ) : Float32.(TwoPhaseAMR._visualization_vertex_values(initial_values))
    else
        zeros(Float32, length(cell_geometry.points))
    end
    initial_axis = initial_placement isa Face{2} ? 2 : 1
    initial_face_geometry = !(initial_placement isa Cell) && initial_discontinuous ?
        initial_phase_geometry : face_geometries[initial_axis]
    face_values = if initial_placement isa Cell
        zeros(Float32, length(initial_face_geometry.points))
    elseif initial_discontinuous
        Float32.(
            TwoPhaseAMR._phase_visualization_vertex_values(
                initial_values, initial_face_geometry,
            )
        )
    else
        Float32.(
            TwoPhaseAMR._visualization_face_vertex_values(
                initial_values, initial_face_geometry
            )
        )
    end
    active_values = initial_placement isa Cell ? cell_values : face_values
    initial_colorrange = _makie_colorrange(initial_entry, active_values)

    cell_colors = Makie.Observable(cell_values)
    cell_mesh = Makie.Observable(_makie_mesh(initial_cell_geometry))
    face_mesh = Makie.Observable(_makie_mesh(initial_face_geometry))
    face_colors = Makie.Observable(face_values)
    selection_points = Makie.Observable(Makie.Point2f[])
    selected_cell_observable =
        Makie.Observable{Union{Nothing, CellRef}}(nothing)
    selected_face_observable =
        Makie.Observable{Union{Nothing, FaceRef}}(nothing)
    current_field = Makie.Observable(initial_name)
    displayed_frame =
        Makie.Observable{Union{Nothing, VisualizationFrame}}(nothing)
    status = Makie.Observable("No cell or face selected")

    fig = Makie.Figure(; figure...)
    axis_attributes = merge(
        (; aspect = Makie.DataAspect(), title = _field_title(initial_entry)), axis
    )
    ax = Makie.Axis(fig[1, 1]; axis_attributes...)
    cell_plot = Makie.mesh!(
        ax, cell_mesh; color = cell_colors,
        colormap = initial_entry.colormap,
        colorrange = initial_colorrange,
        nan_color = :transparent,
        shading = Makie.NoShading, inspectable = true,
        visible = initial_placement isa Cell
    )
    face_plot = Makie.mesh!(
        ax, face_mesh; color = face_colors,
        colormap = initial_entry.colormap,
        colorrange = initial_colorrange, nan_color = :transparent,
        shading = Makie.NoShading,
        inspectable = true, visible = !(initial_placement isa Cell)
    )

    cell_segments = Makie.Observable(
        _makie_points(_all_cell_segments(cell_geometry))
    )
    block_segments = Makie.Observable(
        _makie_points(cell_geometry.block_segments)
    )
    interface_segments = Makie.Observable(
        _interface_overlay_points(registry)
    )
    solid_mesh = Makie.Observable(
        _makie_mesh(_combined_solid_geometry(registry))
    )
    if show_cells
        Makie.linesegments!(
            ax, cell_segments;
            color = (:black, 0.22), linewidth = 0.5, inspectable = false,
            depth_shift = -1.0f-5
        )
    end
    if show_blocks
        Makie.linesegments!(
            ax, block_segments;
            color = :black, linewidth = 1.5, inspectable = false,
            depth_shift = -2.0f-5
        )
    end
    solid_plot = Makie.mesh!(
        ax, solid_mesh; color = :black, shading = Makie.NoShading,
        inspectable = false, depth_shift = -3.0f-4
    )
    interface_plot = Makie.linesegments!(
        ax, interface_segments; color = :black, linewidth = 2,
        inspectable = false, depth_shift = -4.0f-4
    )
    outline_plot = Makie.linesegments!(
        ax, selection_points; color = :yellow, linewidth = 4,
        inspectable = false, depth_shift = -1.0f-4
    )

    diagnostic_axis = nothing
    diagnostic_data = Dict{Symbol, Any}()
    diagnostic_count = Ref(0)
    checkpoint_times = nothing
    checkpoint_points = nothing
    checkpoint_plot = nothing
    controls_row = 2
    if !isempty(registry.diagnostics)
        diagnostic_axis = Makie.Axis(
            fig[2, 1:2]; xlabel = "Simulation time", ylabel = "Value",
            title = "Diagnostics"
        )
        for entry in registry.diagnostics
            data = Makie.Observable(Makie.Point2f[])
            diagnostic_data[entry.name] = data
            attributes = isnothing(entry.color) ?
                (; label = _diagnostic_label(entry)) :
                (; label = _diagnostic_label(entry), color = entry.color)
            Makie.lines!(diagnostic_axis, data; attributes...)
        end
        checkpoint_times = Makie.Observable(Float32[])
        checkpoint_lines_visible =
            Makie.@lift(!isempty($checkpoint_times))
        Makie.vlines!(
            diagnostic_axis, checkpoint_times;
            color = (:black, 0.35), linestyle = :dash, inspectable = false,
            visible = checkpoint_lines_visible
        )
        checkpoint_points = Makie.Observable(Makie.Point2f[])
        checkpoint_plot = Makie.scatter!(
            diagnostic_axis, checkpoint_points; color = :orange,
            strokecolor = :black, strokewidth = 1, markersize = 13,
            inspectable = true
        )
        Makie.axislegend(diagnostic_axis; position = :rt)
        controls_row = 3
    end

    options = [(entry.label, entry.name) for entry in registry.entries]
    controls = Makie.GridLayout(fig[controls_row, 1:2])
    Makie.Label(controls[1, 1], "Field")
    menu = Makie.Menu(
        controls[1, 2]; options,
        default = registry.indices[initial_name], width = 220
    )
    Makie.Label(controls[1, 3], status; tellwidth = false)
    Makie.Label(controls[2, 1], "Checkpoint")
    checkpoint_menu = Makie.Menu(
        controls[2, 2]; options = [("No checkpoints available", 0)],
        default = nothing, prompt = "Select checkpoint", width = 220
    )
    live_button = Makie.Button(controls[2, 3]; label = "Live")
    pause_button = Makie.Button(controls[2, 4]; label = "Pause")
    session_status = Makie.Observable("Standalone")
    Makie.Label(controls[2, 5], session_status; tellwidth = false)
    colorbar = Makie.Colorbar(
        fig[1, 2]; colormap = initial_entry.colormap,
        colorrange = initial_colorrange
    )
    Makie.tightlimits!(ax)

    viewer = MakieAMRVisualization(
        fig, ax, cell_plot, face_plot, outline_plot,
        interface_plot, solid_plot, colorbar,
        cell_colors, face_mesh, face_colors, selection_points,
        selected_cell_observable, selected_face_observable,
        current_field, displayed_frame, status, registry, menu,
        checkpoint_menu, live_button, pause_button,
        Ref{Any}(nothing), Ref(false),
        session_status, cell_mesh, cell_segments, block_segments,
        interface_segments, solid_mesh,
        cell_geometry, face_geometries, topology,
        Ref{Any}(initial_cell_geometry), Ref{Any}(initial_face_geometry),
        Ref{Any}(topology), Ref{Any}(topology), Ref(initial_axis),
        diagnostic_axis, diagnostic_data, diagnostic_count, Ref(-1), Ref{Any}(nothing),
        checkpoint_times, checkpoint_points, checkpoint_plot, mesh
    )

    Makie.on(menu.selection) do selection
        isnothing(selection) || _display_field!(viewer, Symbol(selection))
    end

    Makie.on(checkpoint_menu.selection) do selection
        viewer.updating_checkpoint_controls[] && return
        session = _bound_session(viewer)
        if !isnothing(session) && selection isa Integer && selection > 0
            select_checkpoint!(session, selection)
        end
        return
    end

    Makie.on(live_button.clicks) do _
        session = _bound_session(viewer)
        isnothing(session) || live_mode!(session)
        return
    end

    Makie.on(pause_button.clicks) do _
        session = _bound_session(viewer)
        isnothing(session) && return
        state = simulation_state(session)
        if state == :running
            request_pause!(session)
        elseif state in (:pause_requested, :paused)
            resume_simulation!(session)
        end
        return
    end

    if !isnothing(diagnostic_axis)
        Makie.on(
            Makie.events(diagnostic_axis.scene).mousebutton,
            priority = 2
        ) do event
            if event.button == Makie.Mouse.left &&
                    event.action == Makie.Mouse.press
                plot, index = Makie.pick(diagnostic_axis.scene)
                if plot === checkpoint_plot &&
                        _select_checkpoint_marker!(viewer, index)
                    return Makie.Consume(true)
                end
            end
            return Makie.Consume(false)
        end
    end

    Makie.on(Makie.events(ax.scene).mousebutton, priority = 2) do event
        if event.button == Makie.Mouse.left && event.action == Makie.Mouse.press
            plot, vertex_index = Makie.pick(ax.scene)
            entry = TwoPhaseAMR._field_entry(
                viewer.registry, viewer.current_field[]
            )
            frame = viewer.displayed_frame[]
            mode = isnothing(frame) ? viewer.registry.phase_rendering : frame.phase_rendering
            placement = TwoPhaseAMR._source_placement(entry.source, mode)
            if placement isa Cell && plot === viewer.cell_plot
                render_geometry = viewer.cell_render_geometry[]
                cell = cell_for_vertex(render_geometry, vertex_index)
                viewer.selected_cell_observable[] = cell
                viewer.selected_face_observable[] = nothing
                viewer.selection_points[] = isnothing(cell) ? Makie.Point2f[] :
                    _makie_points(
                        TwoPhaseAMR._cell_outline(
                            viewer.cell_geometry, cell
                        )
                    )
                viewer.status[] = _cell_description(cell)
                return Makie.Consume(true)
            elseif !(placement isa Cell) && plot === viewer.face_plot
                direction = placement isa Face{1} ? 1 : 2
                geometry = viewer.face_geometries[direction]
                render_geometry = viewer.face_render_geometry[]
                face = face_for_vertex(render_geometry, vertex_index)
                viewer.selected_cell_observable[] = nothing
                viewer.selected_face_observable[] = face
                if isnothing(face)
                    viewer.selection_points[] = Makie.Point2f[]
                    viewer.status[] = "No cell or face selected"
                else
                    viewer.selection_points[] = _makie_points(
                        TwoPhaseAMR._face_outline(
                            geometry, face
                        )
                    )
                    viewer.status[] = _face_description(viewer.topology, face)
                end
                return Makie.Consume(true)
            end
        end
        return Makie.Consume(false)
    end

    display && Base.display(fig)
    return viewer
end

function TwoPhaseAMR.visualize(
        field::CellField; colormap = :viridis, colorrange = Makie.automatic,
        show_blocks = true, show_cells = false, title = "Cell field",
        figure = (; size = (900, 700)), axis = (;), display = true
    )
    registry = VisualizationFieldRegistry()
    stored_colorrange = colorrange === Makie.automatic ? nothing : colorrange
    register_field!(
        registry, :field, field; label = title, colormap,
        colorrange = stored_colorrange
    )
    return visualize(
        registry; field = :field, show_blocks, show_cells,
        figure, axis, display
    )
end

function TwoPhaseAMR.select_visualization_field!(
        visualization::MakieAMRVisualization, name::Symbol
    )
    index = get(visualization.registry.indices, name, 0)
    iszero(index) && throw(ArgumentError("unknown visualization field :$name"))
    if visualization.menu.i_selected[] == index
        _display_field!(visualization, name)
    else
        visualization.menu.i_selected[] = index
    end
    return visualization
end

function TwoPhaseAMR.update_visualization!(
        visualization::MakieAMRVisualization; kwargs...
    )
    _replace_topology!(
        visualization,
        TwoPhaseAMR._visualization_topology(visualization.mesh)
    )
    visualization.displayed_frame[] = nothing
    _refresh_plic_overlays!(visualization)
    return _display_field!(
        visualization, visualization.current_field[]; clear_selection = false
    )
end

function TwoPhaseAMR.update_visualization!(
        visualization::MakieAMRVisualization, frame::VisualizationFrame
    )
    for name in field_names(visualization.registry)
        haskey(frame, name) ||
            throw(ArgumentError("visualization frame has no field :$name"))
    end
    _replace_topology!(visualization, frame.topology)
    visualization.displayed_frame[] = frame
    _refresh_plic_overlays!(visualization, frame)
    _display_field!(
        visualization, visualization.current_field[]; clear_selection = false
    )
    return _refresh_checkpoint_controls!(visualization)
end

function TwoPhaseAMR.update_visualization!(
        visualization::MakieAMRVisualization, field::CellField
    )
    field.mesh === visualization.mesh ||
        throw(
        ArgumentError(
            "the field mesh differs from the viewer mesh"
        )
    )
    _replace_topology!(
        visualization, TwoPhaseAMR._visualization_topology(field.mesh)
    )
    visualization.displayed_frame[] = nothing
    cell_values = Float32.(visualization_values(field))
    entry = TwoPhaseAMR._field_entry(
        visualization.registry, visualization.current_field[]
    )
    _set_plot_style!(visualization, entry, cell_values)
    colors = Float32.(
        TwoPhaseAMR._visualization_vertex_values(cell_values)
    )
    if visualization.cell_plot_topology[] !== visualization.topology ||
            visualization.cell_render_geometry[] !== visualization.cell_geometry
        _replace_cell_plot!(
            visualization, visualization.cell_geometry, colors
        )
    else
        visualization.cell_colors[] = colors
    end
    visualization.cell_plot.visible[] = true
    visualization.face_plot.visible[] = false
    return visualization
end

TwoPhaseAMR.selected_cell(visualization::MakieAMRVisualization) =
    visualization.selected_cell_observable[]

TwoPhaseAMR.selected_face(visualization::MakieAMRVisualization) =
    visualization.selected_face_observable[]

TwoPhaseAMR.visualization_figure(visualization::MakieAMRVisualization) =
    visualization.figure

function TwoPhaseAMR._bind_visualization_session!(
        visualization::MakieAMRVisualization,
        session::VisualizationSession
    )
    visualization.session_ref[] = WeakRef(session)
    visualization.diagnostic_count[] = 0
    visualization.checkpoint_controls_count[] = -1
    visualization.checkpoint_plot_state[] = nothing
    for observable in values(visualization.diagnostic_data)
        observable[] = Makie.Point2f[]
    end
    return visualization
end

TwoPhaseAMR._update_visualization_state!(
    visualization::MakieAMRVisualization,
    ::VisualizationSession
) = _refresh_checkpoint_controls!(visualization)

TwoPhaseAMR._update_visualization_diagnostics!(
    visualization::MakieAMRVisualization,
    session::VisualizationSession
) =
    _refresh_diagnostic_plot!(visualization, session)

end
