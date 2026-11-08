"""
    snapshot_figure(source; field, phase_rendering=:cell, blocks=true,
                    interface=true, geometry=true, colorbar=true,
                    colormap=:viridis, colorrange=nothing, resolution=1024,
                    size=(600, 500), title=nothing)

Create a widget-free Makie figure of one field. `source` is a stored
[`Frame`](@ref), or a [`VisualizationFieldRegistry`](@ref) whose current state
is captured first. The field is drawn as an image from [`rasterize`](@ref)
with `resolution` pixels along the longer domain side; block outlines, PLIC
interfaces, and embedded solids are vector overlays. Load a Makie backend
before calling this function.
"""
snapshot_figure(args...; kwargs...) = throw(ArgumentError(
    "snapshot_figure requires a loaded Makie backend, such as CairoMakie",
))

"""
    snapshot_figure!(gridposition, source; field, kwargs...)

Draw a static snapshot in a Makie grid position and return its axis. Accepts
the same options as [`snapshot_figure`](@ref), except `size` is unused.
Load a Makie backend before calling this function.
"""
snapshot_figure!(args...; kwargs...) = throw(ArgumentError(
    "snapshot_figure! requires a loaded Makie backend, such as CairoMakie",
))

"""
    plot_diagnostics!(axis, flow, names; analytical=nothing, labels=names)

Plot named diagnostic histories on a Makie axis. If supplied, `analytical`
is a function of time, drawn dashed for each history. `labels` supplies the
corresponding series labels. Returns the axis.
"""
plot_diagnostics!(args...; kwargs...) = throw(ArgumentError(
    "plot_diagnostics! requires a loaded Makie backend, such as CairoMakie",
))

"""
    record_checkpoints(path, source; field, framerate=10, size=(600, 500),
                       kwargs...)

Write the keyframes stored by a [`Recorder`](@ref) or in a
[`SnapshotStore`](@ref) as a GIF or video. The figure is built once; every
frame uses the same limits and, unless `colorrange` is given, the colour range
of the field over all keyframes. Other keywords are passed to
[`snapshot_figure!`](@ref). Returns `path`. Load a Makie backend before calling
this function.
"""
record_checkpoints(args...; kwargs...) = throw(ArgumentError(
    "record_checkpoints requires a loaded Makie backend, such as CairoMakie",
))
