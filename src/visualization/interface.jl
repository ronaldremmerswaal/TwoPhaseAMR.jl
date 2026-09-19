"""
    visualize(source; kwargs...)

Create an interactive AMR field viewer from a `CellField` or
`VisualizationFieldRegistry`. This method is supplied by the optional Makie
extension; load a Makie backend such as GLMakie, WGLMakie, or CairoMakie before calling it.
"""
function visualize end

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

update_visualization!(visualization::NoVisualization, args...; kwargs...) =
    visualization

function initialize_visualization!(
        visualization::AbstractVisualization;
        step_procedure = nothing, kwargs...
    )
    return update_visualization!(visualization; event = :initial, kwargs...)
end
