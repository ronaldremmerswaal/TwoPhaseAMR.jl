module TwoPhaseAMRMakieExt

using TwoPhaseAMR
import Makie

struct MakieAMRVisualization{F,A,P,O,S,M} <: AbstractVisualization
    figure::F
    axis::A
    field_plot::P
    outline_plot::O
    selected::S
    geometry::AMRCellGeometry
    mesh::M
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

function _selection_label(cell)
    isnothing(cell) && return "No cell selected"
    return "Block $(cell.block), cell ($(cell.i), $(cell.j))"
end

function TwoPhaseAMR.visualize(
        field::CellField; colormap=:viridis, colorrange=Makie.automatic,
        show_blocks=true, show_cells=false, title="Cell field",
        figure=(; size=(900, 700)), axis=(;), display=true)
    geometry = visualization_geometry(field.mesh)
    makie_mesh = _makie_mesh(geometry)
    colors = Makie.Observable(
        Float32.(TwoPhaseAMR._visualization_vertex_values(field)))
    selected = Makie.Observable{Union{Nothing,CellRef}}(nothing)
    selection_points = Makie.Observable(Makie.Point2f[])

    fig = Makie.Figure(; figure...)
    axis_attributes = merge((; aspect=Makie.DataAspect(), title), axis)
    ax = Makie.Axis(fig[1, 1]; axis_attributes...)
    field_plot = Makie.mesh!(
        ax, makie_mesh; color=colors, colormap, colorrange,
        shading=Makie.NoShading, inspectable=true)
    if show_cells
        Makie.linesegments!(
            ax, _makie_points(_all_cell_segments(geometry));
            color=(:black, 0.22), linewidth=0.5, inspectable=false)
    end
    if show_blocks
        Makie.linesegments!(
            ax, _makie_points(geometry.block_segments);
            color=:black, linewidth=1.5, inspectable=false)
    end
    outline_plot = Makie.linesegments!(
        ax, selection_points; color=:yellow, linewidth=4,
        inspectable=false, depth_shift=-1.0f-4)
    Makie.Label(fig[2, 1], Makie.lift(_selection_label, selected))
    Makie.Colorbar(fig[1, 2], field_plot)
    Makie.tightlimits!(ax)

    Makie.on(Makie.events(ax.scene).mousebutton, priority=2) do event
        if event.button == Makie.Mouse.left && event.action == Makie.Mouse.press
            plot, vertex_index = Makie.pick(ax.scene)
            if plot === field_plot
                cell = cell_for_vertex(geometry, vertex_index)
                selected[] = cell
                selection_points[] = isnothing(cell) ? Makie.Point2f[] :
                    _makie_points(TwoPhaseAMR._cell_outline(geometry, cell))
                return Makie.Consume(true)
            end
        end
        return Makie.Consume(false)
    end

    viewer = MakieAMRVisualization(
        fig, ax, field_plot, outline_plot, selected, geometry, field.mesh)
    display && Base.display(fig)
    return viewer
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

function TwoPhaseAMR.update_visualization!(
        visualization::MakieAMRVisualization, field::CellField)
    field.mesh === visualization.mesh ||
        throw(ArgumentError(
            "the field mesh differs from the viewer mesh; topology replacement is not implemented"))
    values = Float32.(TwoPhaseAMR._visualization_vertex_values(field))
    Makie.update!(visualization.field_plot; color=values)
    return visualization
end

TwoPhaseAMR.selected_cell(visualization::MakieAMRVisualization) =
    visualization.selected[]

TwoPhaseAMR.visualization_figure(visualization::MakieAMRVisualization) =
    visualization.figure

end
