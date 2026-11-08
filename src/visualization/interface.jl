_requires_makie(name) = throw(
    ArgumentError(
        "$name requires a loaded Makie backend, such as CairoMakie",
    )
)

"""
    fieldaxis(gridposition; camera=:fit, decorations=:minimal,
              scalebar=decorations === :minimal, kwargs...)

Create a Makie `Axis` for AMR fields in a figure grid position. Its limits
fill the layout cell with one isotropic scale (`aspect = nothing`,
`autolimitaspect = 1`): cells stay square, and a panel whose shape differs
from the requested view shows more of the domain instead of empty space.

`camera` selects the view:

- `:fit`: the domain of the plotted fields;
- `(xmin, xmax, ymin, ymax)`: this rectangle, shown completely and centred;
- a domain (a [`Frame`](@ref), [`TopologySnapshot`](@ref), or
  `BlockTreeMesh`): its rectangle;
- a [`ViewCamera`](@ref): centre and world units per pixel, applied to the
  panel's current pixel size.

`decorations=:minimal` hides ticks and labels and draws a scale bar in the
lower-left corner, `:full` shows coordinate ticks and the labels `x` and
`y`, and `:none` also hides the frame. Other keywords are passed to `Axis`.
Draw fields with [`amrfield!`](@ref). Load a Makie backend first.
"""
fieldaxis(args...; kwargs...) = _requires_makie("fieldaxis")

"""
    amrfield!(axis, source, field; phase_rendering=:cell, colormap=automatic,
              colorrange=automatic, interface=true, geometry=true,
              blocks=false, oversample=2, kwargs...)
    amrfield(source, field; kwargs...)

Makie recipe that draws field `field` of an AMR frame. `source` is a stored
[`Frame`](@ref), an `Observable` of frames (animations swap frames without
rebuilding the figure), or a [`VisualizationFieldRegistry`](@ref), whose
current state is captured once.

The field is an image of [`rasterize`](@ref) over the visible part of the
domain, with `oversample` samples per screen pixel along each axis. It is
rasterized again when the frame, the axis limits, or the panel size change,
so cell edges stay sharp at every zoom. `phase_rendering` is `:cell` (each
cell shows its majority phase) or `:plic` (mixed cells are split at their
PLIC plane). PLIC interfaces (`interface`), embedded solids (`geometry`), and
leaf-block outlines (`blocks`) are vector overlays; `interfacecolor`,
`interfacewidth`, `geometrycolor`, `blockcolor`, and `blockwidth` style them.

`colormap=automatic` uses the registered colormap. `colorrange=automatic`
uses the registered range, else the range of the shown frame, symmetric about
zero for vorticity and diverging colormaps; pass a fixed `(lo, hi)` to keep
colours constant across frames. The plot's data limits are the domain, so
`fieldaxis(...; camera=:fit)` fits the domain, and `Colorbar(gridposition,
plot)` draws the colour bar. `amrfield` creates its own figure and isotropic
axis. Load a Makie backend first.
"""
function amrfield! end

"""
    amrfield(source, field; kwargs...)

Create a figure with one isotropic axis showing an AMR field; see
[`amrfield!`](@ref).
"""
function amrfield end

"""
    diagnosticsgrid!(gridposition, source; names=:all, group=:dimension,
                     transform=:raw, cursor=nothing, until=nothing,
                     references=Dict(), direction=:column, legend=:rt,
                     xlabel="t")

Plot simulation diagnostics in a figure grid position, one axis per group,
and return the axes. `source` is a [`SnapshotStore`](@ref) or
[`Recorder`](@ref) (its mirrored diagnostics), a `FlowSimulation`, or a
`DiagnosticCollection`; `names` selects diagnostics (default: all).

- `group=:dimension` plots diagnostics with the same kind, physical dimension,
  and scale on one axis (see [`diagnostic_groups`](@ref)); `group=:name` gives
  every diagnostic its own axis. An axis with one diagnostic is titled by its
  label and dimension, a shared axis by the quantity, with a legend. Vector
  diagnostics are split into labelled components, and `:log` groups use a
  logarithmic axis.
- `transform=:relative` plots the change relative to the first sample,
  `(v - v₀)/|v₀|`.
- `cursor` is a time or an `Observable` of times, drawn as a vertical line on
  every axis; animations use it to show the current frame.
- `until` (a time or an `Observable` of times) shows only the samples up to
  that time. With an `Observable`, the source is read again whenever it
  changes and the axes follow the data, so the diagnostics of a running
  simulation grow.
- `references` maps a diagnostic name to a function of time or to the name of
  another diagnostic. The reference is drawn dashed on the same axis, and a
  referenced diagnostic gets no axis of its own.
- `direction=:column` stacks the axes and links their time axes; `:row`
  places them side by side. `legend` is an `axislegend` position or
  `nothing`.

Load a Makie backend first.
"""
diagnosticsgrid!(args...; kwargs...) = _requires_makie("diagnosticsgrid!")

"""
    export_animation(path, recording; field, source=:stored, window=nothing,
                     diagnostics=nothing, framerate=12, size=automatic,
                     px_per_unit=1, camera=:fit, colorrange=nothing,
                     colormap=nothing, phase_rendering=:cell, interface=true,
                     geometry=true, blocks=false, decorations=:minimal,
                     title=automatic, group=:dimension, references=Dict(),
                     transform=:raw, every=1, capacity=32)

Animate field `field` of a recording as a GIF or video (`.gif`, `.mp4`,
`.mkv`, or `.webm`). An animation shows either stored or live frames, never
a mix:

- `source=:stored` (default) writes the keyframes of a
  [`SnapshotStore`](@ref) or [`Recorder`](@ref) after the run, optionally
  only those with times in `window = (t0, t1)`, and returns `path`. The
  colour range is `colorrange` if given, else the registered range, else the
  range of the field over the exported frames, symmetric about zero for
  vorticity and diverging colormaps.
- `source=:live` records the frames of a [`Recorder`](@ref) as the run
  produces them (previews and keyframes, including previews that retention
  later thins) and returns a handle at once; start it before
  [`simulate!`](@ref) and `close` it afterwards to write `path`. Every
  `every`-th frame is encoded on a thread other than the solver's, through a
  queue of `capacity` frames: the solver never waits, and frames that arrive
  while the queue is full are dropped and counted. The handle reports
  `encoded`, `dropped`, and `skipped` frames. Without a registered or given
  range, the colour range is that of the frames recorded so far. Viewing or
  reading the store from other tasks does not affect the recording.

The figure is built once and the frames are swapped through an `Observable`.
The axis limits follow `camera` (see [`fieldaxis`](@ref); `:fit` is the
domain) and are the same on every frame. `title` is `automatic` (`t = …`),
`nothing`, a string, or a function of the frame. `diagnostics` (names, or
`:all`) adds [`diagnosticsgrid!`](@ref) panels, grouped by `group`, with a
time cursor synced to the frame; live diagnostics grow with the run. Load a
Makie backend first.
"""
export_animation(args...; kwargs...) = _requires_makie("export_animation")

"""
    export_figure(path, source; field, at=nothing, diagnostics=nothing,
                  size=automatic, camera=:fit, colorrange=nothing,
                  colormap=nothing, phase_rendering=:cell, interface=true,
                  geometry=true, blocks=false, decorations=:minimal,
                  title=automatic, group=:dimension, references=Dict(),
                  transform=:raw)

Save a still image of field `field` (PNG, SVG, or PDF, from the extension of
`path`) and return `path`. `source` is a [`SnapshotStore`](@ref) or
[`Recorder`](@ref), a [`Frame`](@ref), or a
[`VisualizationFieldRegistry`](@ref) (its current state). For stores, `at`
is a time or a vector of times; each shows the stored frame nearest to it in
its own panel, and `nothing` shows the latest frame. The panels share one
camera, colour range, and colour bar, chosen as in
[`export_animation`](@ref). `diagnostics` adds [`diagnosticsgrid!`](@ref)
panels with cursors at the shown times. Load a Makie backend first.
"""
export_figure(args...; kwargs...) = _requires_makie("export_figure")
