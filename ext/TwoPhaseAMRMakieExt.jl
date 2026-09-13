module TwoPhaseAMRMakieExt

using TwoPhaseAMR
import Makie

struct MakieAMRVisualization{F,A,CP,FP,OP,CB,ME,CM,LB,M} <:
       AbstractVisualization
    figure::F
    axis::A
    cell_plot::CP
    face_plot::FP
    outline_plot::OP
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
    session_ref::Base.RefValue{Any}
    updating_checkpoint_controls::Base.RefValue{Bool}
    session_status::Makie.Observable
    cell_geometry::AMRCellGeometry
    face_geometries::Dict{Int,AMRFaceGeometry}
    mesh::M
end

function _bound_session(visualization::MakieAMRVisualization)
    reference = visualization.session_ref[]
    reference isa WeakRef || return nothing
    return reference.value
end

function _checkpoint_label(checkpoint, index)
    requested = round(checkpoint.requested_time; sigdigits=6)
    stored = round(checkpoint.frame.time; sigdigits=6)
    if requested == stored
        return "#$index · t = $stored"
    end
    return "#$index · requested $requested · stored $stored"
end

function _refresh_checkpoint_controls!(
        visualization::MakieAMRVisualization)
    session = _bound_session(visualization)
    isnothing(session) && return visualization
    checkpoints = visualization_checkpoints(session)
    information = TwoPhaseAMR._visualization_session_state(session)
    options = isempty(checkpoints) ?
        [("No checkpoints available", 0)] :
        [(_checkpoint_label(checkpoint, index), index)
         for (index, checkpoint) in pairs(checkpoints)]

    visualization.updating_checkpoint_controls[] = true
    try
        visualization.checkpoint_menu.options[] = options
        selected = information.mode == :checkpoint ?
            something(information.selected_checkpoint, 0) : 0
        visualization.checkpoint_menu.i_selected[] =
            1 <= selected <= length(checkpoints) ? selected : 0
    finally
        visualization.updating_checkpoint_controls[] = false
    end

    frame = information.latest_frame
    time_text = isnothing(frame) ? "" :
        " · t = $(round(frame.time; sigdigits=6))"
    if information.mode == :checkpoint &&
            !isnothing(information.selected_checkpoint)
        index = information.selected_checkpoint
        visualization.session_status[] =
            "Checkpoint $index/$(length(checkpoints)) · $(information.state)"
    else
        visualization.session_status[] =
            "Live · $(information.state)$time_text"
    end
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

function _all_cell_segments(geometry::AMRCellGeometry)
    segments = Vector{eltype(geometry.points)}(
        undef, 8length(geometry.cells))
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

function _automatic_colorrange(values)
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
        return _automatic_colorrange(values)
    end
    return entry.colorrange
end

function _cell_description(cell)
    isnothing(cell) && return "No cell or face selected"
    return "Block $(cell.block), cell ($(cell.i), $(cell.j))"
end

function _face_description(mesh, reference::FaceRef)
    face = reference.boundary ? mesh.boundary_faces[reference.index] :
                               mesh.faces[reference.index]
    kind = reference.boundary ? "Boundary" : "Interior"
    direction = face.axis == 1 ? "x-normal" : "y-normal"
    x, y = face.center
    return "$kind $direction face at ($(round(x; sigdigits=6)), " *
           "$(round(y; sigdigits=6)))"
end

function _clear_selection!(visualization::MakieAMRVisualization)
    visualization.selected_cell_observable[] = nothing
    visualization.selected_face_observable[] = nothing
    visualization.selection_points[] = Makie.Point2f[]
    visualization.status[] = "No cell or face selected"
    return visualization
end

function _set_plot_style!(visualization::MakieAMRVisualization, entry, values)
    colorrange = _makie_colorrange(entry, values)
    visualization.cell_plot.colormap[] = entry.colormap
    visualization.cell_plot.colorrange[] = colorrange
    visualization.face_plot.colormap[] = entry.colormap
    visualization.face_plot.colorrange[] = colorrange
    visualization.colorbar.colormap[] = entry.colormap
    visualization.colorbar.colorrange[] = colorrange
    visualization.axis.title[] = _field_title(entry)
    return visualization
end

function _display_field!(visualization::MakieAMRVisualization, name::Symbol;
                         clear_selection=true)
    entry = TwoPhaseAMR._field_entry(visualization.registry, name)
    placement = TwoPhaseAMR._source_placement(entry.source)
    frame = visualization.displayed_frame[]
    values = Float32.(isnothing(frame) ?
        TwoPhaseAMR._source_values(entry.source) : frame[name])
    _set_plot_style!(visualization, entry, values)

    if placement isa Cell
        length(values) == length(visualization.cell_geometry.cells) ||
            throw(DimensionMismatch("cell field size differs from viewer geometry"))
        visualization.cell_colors[] = Float32.(
            TwoPhaseAMR._visualization_vertex_values(values))
        visualization.cell_plot.visible[] = true
        visualization.face_plot.visible[] = false
    else
        axis = placement isa Face{1} ? 1 : 2
        geometry = visualization.face_geometries[axis]
        length(values) == maximum(geometry.value_indices; init=0) ||
            throw(DimensionMismatch("face field size differs from viewer geometry"))
        visualization.face_mesh[] = _makie_mesh(geometry)
        visualization.face_colors[] = Float32.(
            TwoPhaseAMR._visualization_face_vertex_values(values, geometry))
        visualization.cell_plot.visible[] = false
        visualization.face_plot.visible[] = true
    end

    visualization.current_field[] = name
    clear_selection && _clear_selection!(visualization)
    return visualization
end

function TwoPhaseAMR.visualize(
        registry::VisualizationFieldRegistry; field=nothing,
        show_blocks=true, show_cells=false,
        figure=(; size=(900, 700)), axis=(;), display=true)
    isempty(registry.entries) &&
        throw(ArgumentError("cannot visualize an empty field registry"))
    initial_name = isnothing(field) ? first(field_names(registry)) : Symbol(field)
    initial_entry = TwoPhaseAMR._field_entry(registry, initial_name)
    mesh = registry.mesh::BlockTreeMesh
    cell_geometry = visualization_geometry(mesh)
    face_geometries = Dict{Int,AMRFaceGeometry}(
        1 => visualization_geometry(mesh, Face{1}()),
        2 => visualization_geometry(mesh, Face{2}()))

    initial_placement = TwoPhaseAMR._source_placement(initial_entry.source)
    initial_values = Float32.(TwoPhaseAMR._source_values(initial_entry.source))
    initial_colorrange = _makie_colorrange(initial_entry, initial_values)
    cell_values = initial_placement isa Cell ?
        Float32.(TwoPhaseAMR._visualization_vertex_values(initial_values)) :
        zeros(Float32, length(cell_geometry.points))
    initial_axis = initial_placement isa Face{2} ? 2 : 1
    initial_face_geometry = face_geometries[initial_axis]
    face_values = initial_placement isa Cell ?
        zeros(Float32, length(initial_face_geometry.points)) :
        Float32.(TwoPhaseAMR._visualization_face_vertex_values(
            initial_values, initial_face_geometry))

    cell_colors = Makie.Observable(cell_values)
    face_mesh = Makie.Observable(_makie_mesh(initial_face_geometry))
    face_colors = Makie.Observable(face_values)
    selection_points = Makie.Observable(Makie.Point2f[])
    selected_cell_observable =
        Makie.Observable{Union{Nothing,CellRef}}(nothing)
    selected_face_observable =
        Makie.Observable{Union{Nothing,FaceRef}}(nothing)
    current_field = Makie.Observable(initial_name)
    displayed_frame =
        Makie.Observable{Union{Nothing,VisualizationFrame}}(nothing)
    status = Makie.Observable("No cell or face selected")

    fig = Makie.Figure(; figure...)
    axis_attributes = merge(
        (; aspect=Makie.DataAspect(), title=_field_title(initial_entry)), axis)
    ax = Makie.Axis(fig[1, 1]; axis_attributes...)
    makie_mesh = _makie_mesh(cell_geometry)
    cell_plot = Makie.mesh!(
        ax, makie_mesh; color=cell_colors,
        colormap=initial_entry.colormap,
        colorrange=initial_colorrange,
        shading=Makie.NoShading, inspectable=true,
        visible=initial_placement isa Cell)
    face_plot = Makie.mesh!(
        ax, face_mesh; color=face_colors,
        colormap=initial_entry.colormap,
        colorrange=initial_colorrange, shading=Makie.NoShading,
        inspectable=true, visible=!(initial_placement isa Cell))

    if show_cells
        Makie.linesegments!(
            ax, _makie_points(_all_cell_segments(cell_geometry));
            color=(:black, 0.22), linewidth=0.5, inspectable=false)
    end
    if show_blocks
        Makie.linesegments!(
            ax, _makie_points(cell_geometry.block_segments);
            color=:black, linewidth=1.5, inspectable=false)
    end
    outline_plot = Makie.linesegments!(
        ax, selection_points; color=:yellow, linewidth=4,
        inspectable=false, depth_shift=-1.0f-4)

    options = [(entry.label, entry.name) for entry in registry.entries]
    controls = Makie.GridLayout(fig[2, 1])
    Makie.Label(controls[1, 1], "Field")
    menu = Makie.Menu(
        controls[1, 2]; options,
        default=registry.indices[initial_name], width=220)
    Makie.Label(controls[1, 3], status; tellwidth=false)
    Makie.Label(controls[2, 1], "Checkpoint")
    checkpoint_menu = Makie.Menu(
        controls[2, 2]; options=[("No checkpoints available", 0)],
        default=nothing, prompt="Select checkpoint", width=220)
    live_button = Makie.Button(controls[2, 3]; label="Live")
    session_status = Makie.Observable("Standalone")
    Makie.Label(controls[2, 4], session_status; tellwidth=false)
    colorbar = Makie.Colorbar(
        fig[1, 2]; colormap=initial_entry.colormap,
        colorrange=initial_colorrange)
    Makie.tightlimits!(ax)

    viewer = MakieAMRVisualization(
        fig, ax, cell_plot, face_plot, outline_plot, colorbar,
        cell_colors, face_mesh, face_colors, selection_points,
        selected_cell_observable, selected_face_observable,
        current_field, displayed_frame, status, registry, menu,
        checkpoint_menu, live_button, Ref{Any}(nothing), Ref(false),
        session_status, cell_geometry, face_geometries, mesh)

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

    Makie.on(Makie.events(ax.scene).mousebutton, priority=2) do event
        if event.button == Makie.Mouse.left && event.action == Makie.Mouse.press
            plot, vertex_index = Makie.pick(ax.scene)
            entry = TwoPhaseAMR._field_entry(
                viewer.registry, viewer.current_field[])
            placement = TwoPhaseAMR._source_placement(entry.source)
            if placement isa Cell && plot === viewer.cell_plot
                cell = cell_for_vertex(viewer.cell_geometry, vertex_index)
                viewer.selected_cell_observable[] = cell
                viewer.selected_face_observable[] = nothing
                viewer.selection_points[] = isnothing(cell) ? Makie.Point2f[] :
                    _makie_points(TwoPhaseAMR._cell_outline(
                        viewer.cell_geometry, cell))
                viewer.status[] = _cell_description(cell)
                return Makie.Consume(true)
            elseif !(placement isa Cell) && plot === viewer.face_plot
                direction = placement isa Face{1} ? 1 : 2
                geometry = viewer.face_geometries[direction]
                face = face_for_vertex(geometry, vertex_index)
                viewer.selected_cell_observable[] = nothing
                viewer.selected_face_observable[] = face
                if isnothing(face)
                    viewer.selection_points[] = Makie.Point2f[]
                    viewer.status[] = "No cell or face selected"
                else
                    viewer.selection_points[] = _makie_points(
                        TwoPhaseAMR._face_outline(
                            geometry, vertex_index))
                    viewer.status[] = _face_description(viewer.mesh, face)
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
        field::CellField; colormap=:viridis, colorrange=Makie.automatic,
        show_blocks=true, show_cells=false, title="Cell field",
        figure=(; size=(900, 700)), axis=(;), display=true)
    registry = VisualizationFieldRegistry()
    stored_colorrange = colorrange === Makie.automatic ? nothing : colorrange
    register_field!(registry, :field, field; label=title, colormap,
                    colorrange=stored_colorrange)
    return visualize(registry; field=:field, show_blocks, show_cells,
                     figure, axis, display)
end

function TwoPhaseAMR.select_visualization_field!(
        visualization::MakieAMRVisualization, name::Symbol)
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
        visualization::MakieAMRVisualization; kwargs...)
    visualization.displayed_frame[] = nothing
    return _display_field!(
        visualization, visualization.current_field[]; clear_selection=false)
end

function TwoPhaseAMR.update_visualization!(
        visualization::MakieAMRVisualization, frame::VisualizationFrame)
    for name in field_names(visualization.registry)
        haskey(frame, name) ||
            throw(ArgumentError("visualization frame has no field :$name"))
    end
    visualization.displayed_frame[] = frame
    _display_field!(
        visualization, visualization.current_field[]; clear_selection=false)
    return _refresh_checkpoint_controls!(visualization)
end

function TwoPhaseAMR.update_visualization!(
        visualization::MakieAMRVisualization, field::CellField)
    field.mesh === visualization.mesh ||
        throw(ArgumentError(
            "the field mesh differs from the viewer mesh; topology replacement is not implemented"))
    visualization.displayed_frame[] = nothing
    cell_values = Float32.(visualization_values(field))
    entry = TwoPhaseAMR._field_entry(
        visualization.registry, visualization.current_field[])
    _set_plot_style!(visualization, entry, cell_values)
    visualization.cell_colors[] = Float32.(
        TwoPhaseAMR._visualization_vertex_values(cell_values))
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
        session::VisualizationSession)
    visualization.session_ref[] = WeakRef(session)
    return visualization
end

TwoPhaseAMR._update_visualization_state!(
    visualization::MakieAMRVisualization,
    ::VisualizationSession) = _refresh_checkpoint_controls!(visualization)

end
