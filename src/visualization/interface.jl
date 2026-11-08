"""
    visualize(source; kwargs...)

Create an interactive AMR field viewer from a `CellField` or
`VisualizationFieldRegistry`. This method is supplied by the optional Makie
extension; load a Makie backend such as GLMakie, WGLMakie, or CairoMakie before calling it.
Discontinuous quantities default to majority-phase cell values. Pass
`phase_rendering=:plic` for sharp geometry-based rendering of both phases.
"""
function visualize end

"""
    snapshot_figure(source; field, frame=nothing, blocks=true, interface=true,
                    geometry=true, colorbar=true, colormap=:viridis,
                    size=(600, 500), title=nothing)

Create a widget-free Makie figure from a field registry, retained frame, or
checkpoint. With a registry, `frame` may select a retained frame or checkpoint.
Load a Makie backend before calling this function.
"""
snapshot_figure(args...; kwargs...) = throw(
    ArgumentError(
        "snapshot_figure requires a loaded Makie backend, such as CairoMakie",
    )
)

"""
    snapshot_figure!(gridposition, source; field, kwargs...)

Draw a static snapshot in a Makie grid position and return its axis. Accepts
the same options as [`snapshot_figure`](@ref), except `size` is unused.
Load a Makie backend before calling this function.
"""
snapshot_figure!(args...; kwargs...) = throw(
    ArgumentError(
        "snapshot_figure! requires a loaded Makie backend, such as CairoMakie",
    )
)

"""
    plot_diagnostics!(axis, flow, names; analytical=nothing, labels=names)

Plot named diagnostic histories on a Makie axis. If supplied, `analytical`
is a function of time, drawn dashed for each history. `labels` supplies the
corresponding series labels. Returns the axis.
"""
plot_diagnostics!(args...; kwargs...) = throw(
    ArgumentError(
        "plot_diagnostics! requires a loaded Makie backend, such as CairoMakie",
    )
)

"""
    record_checkpoints(path, session; field, framerate=10, kwargs...)

Write the retained checkpoints of a visualization session as a GIF or video.
Additional keywords are passed to [`snapshot_figure!`](@ref). Returns `path`.
Load a Makie backend before calling this function.
"""
record_checkpoints(args...; kwargs...) = throw(
    ArgumentError(
        "record_checkpoints requires a loaded Makie backend, such as CairoMakie",
    )
)

"""Initialize a visualization observer for one simulation run."""
function initialize_visualization! end

"""Replace the values displayed by an existing visualization."""
function update_visualization! end

"""Return the cell selected in a visualization, or `nothing`."""
selected_cell(::NoVisualization) = nothing

"""Return the face selected in a visualization, or `nothing`."""
selected_face(::NoVisualization) = nothing

"""Switch an interactive viewer to another registered scalar field."""
function select_visualization_field! end

"""Return the backend-specific figure owned by a visualization."""
function visualization_figure end

# Optional backend hooks used by a session to bind GUI controls and refresh
# lifecycle/checkpoint state without exposing solver-owned arrays.
_bind_visualization_session!(::AbstractVisualization, session) = nothing
_update_visualization_state!(::AbstractVisualization, session) = nothing
_update_visualization_diagnostics!(::AbstractVisualization, session) = nothing

# Give asynchronous observers a cooperative scheduling point outside their locks.
_yield_visualization!(::AbstractVisualization) = nothing

update_visualization!(visualization::NoVisualization, args...; kwargs...) =
    visualization

function initialize_visualization!(
        visualization::AbstractVisualization;
        step_procedure = nothing, kwargs...
    )
    return update_visualization!(visualization; event = :initial, kwargs...)
end
