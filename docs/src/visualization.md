# Recording and visualization

The numerical core records simulations without any graphics dependency:

1. a [`VisualizationFieldRegistry`](@ref) names the fields to record;
2. a [`Recorder`](@ref), passed to [`simulate!`](@ref) as its `observer`,
   copies owned frames of those fields into a bounded [`SnapshotStore`](@ref);
3. [`rasterize`](@ref) samples a stored [`Frame`](@ref) on a pixel grid, and a
   [`ViewCamera`](@ref) describes isotropic views.

Static figures, GIFs, and videos are drawn by an optional Makie extension
(load CairoMakie). The solver never waits for anything that reads the store.

## Registering fields

```julia
using TwoPhaseAMR

mesh = BlockTreeMesh(block_size=(16, 16))
velocity = FaceVelocity(mesh, (x, y, t, axis) ->
    axis == 1 ? sinpi(2x)*cospi(2y) : -cospi(2x)*sinpi(2y))
fields = VisualizationFieldRegistry()
register_default_fields!(fields, :velocity, velocity)
```

`register_default_fields!` accepts alternating names and sources. Cell fields
and individual directional face fields produce one entry. A `FaceVelocity` or
an `(x_faces, y_faces)` tuple produces four entries; for a source named
`:velocity`, these are `:velocity_x`, `:velocity_y`, `:velocity_magnitude`, and
`:velocity_vorticity`. Components and vorticity use the diverging `:balance`
colormap, while scalar and magnitude fields use `:viridis`. Labels are derived
from the names; units remain empty because the solver may be dimensional or
nondimensional. `register_default_fields!(fields, flow)` registers the standard
fields of a [`FlowSimulation`](@ref), including its interface. Use
`register_field!` when names, labels, units, colormaps, or color ranges need
explicit control.

The registry accepts `CellField`, `FaceField{1}`, `FaceField{2}`, and a
directional component of a `FaceVelocity` as scalar views. A two-component
face field can be supplied either as a `FaceVelocity` or as an
`(x_faces, y_faces)` tuple. Both representations support the cell-centred
`CentredMagnitude()` and `Vorticity()` views:

```julia
x_faces = FaceField{1}(mesh, (x, y) -> 0.5 - y)
y_faces = FaceField{2}(mesh, (x, y) -> x - 0.5)

register_field!(fields, :x_component, x_faces; label="x component")
register_field!(fields, :magnitude, (x_faces, y_faces), CentredMagnitude())
register_field!(fields, :curl, (x_faces, y_faces), Vorticity())
```

Discontinuous phase quantities are registered as one field that holds both
intrinsic values: `PhaseScalar`, `PhaseFaceScalar`, and intrinsic two-velocity
fields. For a two-velocity field, pass the liquid/gas tuple together with its
interface:

```julia
temperature = PhaseScalar(interface; liquid=liquid_T, gas=gas_T)
register_field!(fields, :temperature, temperature)
register_default_fields!(fields, :velocity, (liquid_velocity, gas_velocity), interface)
```

Magnitude and vorticity of a phase velocity are computed **within each phase**,
not from an already switched velocity. Registering a phase field also
registers its interface, whose PLIC curve is drawn as an overlay; an embedded
`GeometryField` is registered with its interface or with
`register_geometry!`.

## Recording a run

```julia
flow = FlowSimulation(mesh, SinglePhase(density=1.0, dynamic_viscosity=1e-3);
    velocity=(x, y, t, axis) -> axis == 1 ? sinpi(2y) : 0.0,
    diagnostics=(KineticEnergy(),))
fields = VisualizationFieldRegistry()
register_default_fields!(fields, flow)
recorder = Recorder(fields; diagnostics=flow, keyframes=EquispacedTimes(21))
result = simulate!(flow, 1.0; observer=recorder)

store = snapshot_store(recorder)
keyframes(store)              # FrameInfo of the never-dropped frames
frame = store[last(keyframes(store))]
frame[:velocity_vorticity]    # owned values, block-major cell order
frame_at(store, 0.5)          # stored frame nearest to t = 0.5
```

A frame owns copies of every registered field, in native precision for
keyframes, and stays valid after later mesh adaptation. Cell values are
block-major, then `j`, then `i` ([`visualization_values`](@ref) order); face
values list the interior faces of one direction, then its boundary faces.
Phase fields return their `liquid` and `gas` vectors. `frame.topology` is a
[`TopologySnapshot`](@ref): the leaf-block table and staggered face tiles,
shared by all frames until the leaf set changes (detected through the mesh
topology generation).

**Keyframes** are lossless and scheduled by `keyframes`:
[`EquispacedTimes`](@ref) (a count including the initial and final time),
[`EveryTimeInterval`](@ref) (also including both end times), explicit
[`AtTimes`](@ref), or a step schedule ([`EveryNSteps`](@ref),
[`EveryStep`](@ref)). By default the completed state nearest to each
requested time is stored (the earlier one on ties); the requested times are
listed in `FrameInfo.requested_times`. With `exact_keyframes=true`, the
recorder advances an independent copy of the state from the previous step
boundary to each requested time, without changing the simulation's steps or
state. Exact keyframes require a fixed mesh and are rejected when the solver
adapts its mesh. The initial and final states are always stored.

**Previews** show progress between keyframes. They are throttled by
`preview` ([`EveryWallInterval`](@ref), [`EveryNSteps`](@ref),
[`EveryStep`](@ref), [`EveryTimeInterval`](@ref), or `nothing` for headless
scripts) and further so that their capture costs at most
`max_capture_overhead` (default 5 %) of the wall time between previews. They
are stored in `preview_precision` (default `Float32`).

**Retention** bounds memory. [`Retention`](@ref)`(; memory=:auto,
preview=:logarithmic, recent=64)` keeps the newest `recent` previews at full
density and thins older ones so that density halves each time their age
doubles. `memory=:auto` is min(2 GiB, 20 % of physical memory). Over budget,
previews are thinned evenly; keyframes are never dropped, and if they alone
exceed the budget they are kept with one warning. The recorder logs a
per-frame memory estimate when it first records. [`store_usage`](@ref)
reports the frame counts, bytes, and dropped or skipped previews.

**Diagnostics.** With `diagnostics=flow` (or a `DiagnosticCollection`), the
simulation-owned samples are mirrored at every step boundary, with vector
values split into components; `diagnostic_history(store, name)` returns
[`DiagnosticColumns`](@ref). Each completed step is logged by
[`step_log`](@ref) (time, step, time step, solver wall time, projection
iterations and residuals, topology changes, and capture time).

**Control.** All controls are thread-safe and may be called while
`simulate!` runs on another task:

```julia
task = Threads.@spawn simulate!(flow, 10.0; observer=recorder)
pause!(recorder); wait_paused(recorder)   # paused state stored as a :pause frame
resume!(recorder)
request_keyframe!(recorder)               # store the next step as :manual
pause_when!(recorder, :kinetic_energy, e -> e < 0.1)   # breakpoint
request_stop!(recorder)                   # simulate! returns with stopped=true
```

[`run_state`](@ref) reports the lifecycle and [`run_events`](@ref) lists
pauses, breakpoints, stops, and failures. Errors after the initial state, for
example in a breakpoint predicate, are logged and recorded as `:failure`
events; the simulation continues. Configuration errors, such as a registry on
another mesh, throw before the first step. [`subscribe`](@ref)`(store;
capacity)` returns a bounded queue of the newly stored frames for consumers on
other tasks: a consumer receives every frame it keeps up with, even previews
that retention later thins, and a full queue drops (and counts) its oldest
frame instead of waiting.

**Custom observers** subtype [`AbstractObserver`](@ref) and implement
[`observe!`](@ref), optionally [`stop_requested`](@ref). A custom time loop can
drive a recorder directly:

```julia
observe!(recorder, :initial, SimulationState(velocity, nothing, nothing, 0.0, 0),
    StepInfo(; dt=0.0, final_time=1.0))
# after every step:
observe!(recorder, :step, SimulationState(velocity, nothing, nothing, time, step),
    StepInfo(; dt, final_time=1.0))
observe!(recorder, :completed, state, StepInfo(; dt=0.0, final_time=1.0))
```

## Studio

Studio is an optional subpackage with its own Julia environment and a committed
browser bundle. From the repository root, instantiate it once:

```sh
julia --project=lib/TwoPhaseAMRStudio -e 'using Pkg; Pkg.instantiate()'
julia --project=lib/TwoPhaseAMRStudio -t 1,1
```

The second, interactive thread lets HTTP requests be serviced while compute
threads are busy. Create a flow, register its fields, and create a `Recorder`
as above; then start Studio before running the simulation:

```julia
using TwoPhaseAMRStudio

app = studio(recorder; open=:none)
url(app)                              # open this URL in a browser
result = simulate!(flow, 1.0; observer=recorder)
close(app)
```

The dockable shell shows the run and a timeline. Follow mode tracks the newest
frame; stepping, scrubbing and playback enter a per-browser history mode
without pausing recording. Pause, resume, stop and request-keyframe controls
apply to the shared solver and take effect at step boundaries. Recording
controls select exactly one source, `stored` or `live`: live encoding uses a
bounded queue and reports dropped frames, independently of the timeline.
With no interactive thread and one compute thread, the live queue remains
bounded and encoding is deferred until the solver is idle; frames lost while
the queue is full are included in the dropped count.
The `● REC` status is shared between connected browsers. This phase does not
yet render fields or plot diagnostics in Studio; use the static/export API
below for these until those panels are added.

Client extensions can register dockable panel types through
`window.twoPhaseAMRStudio.registerPanel`; the supplied client exposes
`getField` for binary fields and `getData` for topology, phase planes,
diagnostics, and the step log. Each request is matched to its response by ID.

Studio follows the browser's light/dark preference by default. The status-bar
theme selector and light/dark toggle override it per browser and persist the
choice. Client panels should use the same CSS variables instead of fixed
colours; they change together when the theme changes:

| Purpose | CSS variables |
|---|---|
| Shell and dialogs | `--surface`, `--surface-raised`, `--text`, `--muted`, `--border`, `--focus`, `--shadow`, `--dialog-backdrop`, `--hover`, `--drop-overlay`, `--record`, `--error` |
| Timeline | `--timeline-keyframe`, `--timeline-regrid`, `--timeline-playhead` |
| Field panels and colourbars | `--field-background`, `--field-foreground`, `--colorbar-text` |
| Field overlays | `--overlay-interface`, `--overlay-blocks`, `--overlay-solid` |
| Diagnostics | `--diagnostic-background`, `--diagnostic-text`, `--diagnostic-axis`, `--diagnostic-grid`, `--diagnostic-series-1`, `--diagnostic-series-2`, `--diagnostic-series-3` |

The server listens on `127.0.0.1` and chooses an available port by default.
Its URL contains a fresh random 128-bit token required for HTTP assets and
WebSocket connections: treat the entire URL as a secret. It grants access to
run data, solver controls, and recording output, so do not publish it in logs
or issue reports. Binding another address is an explicit `host` choice. For remote
use, keep the default and forward the port over SSH
(`ssh -N -L local_port:127.0.0.1:server_port host`), then open
`http://127.0.0.1:local_port/?token=...` with the same token. Closing Studio
does not stop the simulation.

Automatic browser launch uses an owner-only temporary redirect file rather
than placing the tokenized URL in process arguments; the file is removed when
the Studio server closes.

### Studio protocol v1

Studio uses a single authenticated WebSocket at `/ws?token=…`. JSON messages
have a `type`. The server sends `hello` (protocol version, run and frame
metadata), `frame`, `frames`, `state`, and `error`. Clients send
`control` with `pause`, `resume`, `stop`, or `keyframe`; `get` with `frames`,
`field`, `topology`, `planes`, `diagnostics`, or `steplog`; and `recording`
with `start` or `stop`. A field request specifies
the stored frame ID, field name, request ID, encoding, and, for two-phase
values, `liquid` or `gas`. It returns a whole-field binary message; it does
not imply a change to the viewer's playhead.

The initial `hello` has `protocol: 1`, a `runs` array, a `frames` array,
and a `recording` object. Run metadata includes an ID, name, domain, block
size, initial/final times, fields, and diagnostics. Diagnostic metadata
includes label, units, component names, kind, physical dimension and scale,
so a future client can group series by `(kind, dimension, scale)`. Frame
metadata contains `id`, `time`, `step`, `kind`, and `topology`. Subsequent
`state` messages carry `run_state`, current time, step, `dt`, store bytes,
and the shared recording status. A `frame` announces a committed frame;
request `frames` to reconcile promotions and previews removed by retention.

For example, `{"type":"control","action":"pause"}` requests a cooperative
pause and `{"type":"get","what":"field","frame":4,"field":"pressure",
"encoding":"f32","request":7}` requests one complete field. The request ID
in its binary reply is `7`. A two-phase field additionally specifies
`"phase":"liquid"` or `"phase":"gas"`. Clients must check the version and
message length before interpreting a reply.

Recording is a server-owned session, for example:

```jsonl
{"type":"recording","action":"start","source":"live","path":"run.mp4","field":"pressure","diagnostics":[]}
{"type":"recording","action":"stop"}
```

Use `"source":"stored"` instead to freeze the keyframe selection at stop and
export it asynchronously. The status reports `active`, `source`, `field`,
`path`, `status`, `encoded`, `dropped`, and any error. Navigating the
timeline sends neither of these commands.

The optional `mode` (`cell` or `plic`) is a validated display hint; it does
not transform the numeric payload. A phase field is requested separately for
each intrinsic phase. `get` requests for `planes` provide majority masks and
PLIC data, and `topology` provides block geometry, face-tile mappings and
solids, so the client can perform either display.

Binary messages have a 32-byte little-endian header followed by packed values:

| Byte offset | Type | Meaning |
|---:|---|---|
| 0 | 4 bytes | ASCII `TPAS` magic |
| 4 | uint16 | protocol version (`1`) |
| 6 | uint16 | kind (`1` = field) |
| 8 | uint32 | client request ID |
| 12 | uint8 | data type (`1` = float32, `2` = uint16) |
| 13 | uint8 | flags |
| 14 | uint16 | reserved |
| 16 | uint32 | element count |
| 20 | float32 | scale |
| 24 | float32 | offset |
| 28 | uint32 | reserved |

Float32 payloads are direct, including nonfinite values. A uint16 payload
decodes as `offset + scale * value`; `0xffff` represents missing data and
decodes to NaN. All nonfinite inputs use this sentinel in the quantized
encoding. The complete field is transferred in the ordering of its `Frame`;
the renderer can build on this without changing the recording.

Frontend contributors can run `npm ci`, `npm test -- --run`, and
`npm run check:bundle` in `lib/TwoPhaseAMRStudio/frontend`. `npm run build`
rebuilds the committed assets; the bundle-freshness check is also run by CI.
`npm run test:smoke` exercises a real Studio server in headless Chromium
after Playwright's Chromium browser is installed.

## Phase display

Whether a phase field is shown by majority cells or sharply at the PLIC plane
is a drawing choice, not a recording one. With `phase_rendering=:cell` (the
default), each cell shows the value of its majority phase: liquid when its
liquid volume fraction is **at least 50 %**, gas otherwise, with no blending.
For embedded geometry, the fraction is relative to the **open fluid volume** and
fully solid cells have no value. Face phase fields keep their staggered dual
cells, and each part of a dual cell takes the majority phase of the cell it
lies in. With `phase_rendering=:plic`, mixed cells are split at their PLIC
plane and the solid part of cut cells is empty; staggered fields use the plane
of the cell containing the point.

## Rasterizing frames and cameras

[`rasterize`](@ref)`(frame, field, (xmin, xmax, ymin, ymax), (w, h);
phase_rendering)` returns a `w × h` `Matrix{Float32}` of values at pixel
centres (`NaN` outside the domain or without a value). Cell fields are
piecewise constant; a face value fills its staggered dual cell, the half cells
on both sides of its face, and each of the two fine faces of a coarse--fine
interface fills its half of the coarse cell. Periodic seams need no special
treatment. [`locate`](@ref) returns the cell containing a point.

A [`ViewCamera`](@ref) is a centre and one isotropic scale (world units per pixel),
so a view always fills its panel and cells stay square. [`fit_camera`](@ref)
shows a whole domain, [`zoom`](@ref) keeps the point under the cursor fixed,
[`zoom_to_box`](@ref) shows a box completely, and [`visible_rect`](@ref)
returns the rectangle to rasterize.

## Static figures and animations

Load CairoMakie for figures, GIFs, and videos, including in headless scripts.
Two functions cover the common cases:

```julia
using TwoPhaseAMR, CairoMakie
store = snapshot_store(recorder)
export_figure("vorticity.png", store; field=:velocity_vorticity,
    at=[0.0, 0.5, 1.0], blocks=true)
export_animation("vorticity.gif", store; field=:velocity_vorticity,
    diagnostics=[:kinetic_energy], framerate=12)
```

[`export_figure`](@ref) draws one panel per requested time (the stored frame
nearest to it; by default the latest frame), and also accepts a `Frame` or a
registry, whose current state is captured. [`export_animation`](@ref) builds
the figure once and swaps the frames through an `Observable`. Panels and
frames share one view, colour range, and colour bar:

- the view is `camera=:fit` (the domain) or a rectangle
  `(xmin, xmax, ymin, ymax)`, for example a zoom window; field panels take the
  aspect of the view, so the view fills them exactly and cells stay square;
- the colour range is `colorrange` if given, else the registered range, else
  the range of the field over all shown frames, symmetric about zero for
  vorticity and diverging colormaps such as `:balance`.

With `diagnostics` (names or `:all`), the diagnostics are drawn next to the
field with a time cursor at the shown frame. `title` is `automatic`
(`t = …`), `nothing`, a string, or a function of the frame.

`interface`, `geometry`, and `blocks` select the overlays, and
`interfacecolor` (default `:black`) and `blockcolor` (default
`(:gray40, 0.8)`) colour the interface and block-outline lines. Exports
follow the active Makie theme, and the scale bar takes the axis text colour
on a contrasting backdrop. A dark variant, for example of a README animation
that follows the reader's colour scheme, combines a dark theme with light
text, a diverging colormap whose neutral is dark, and light lines:

```julia
with_theme(theme_dark(); textcolor=:gray85) do
    export_animation("vorticity_dark.gif", store; field=:velocity_vorticity,
        colormap=:berlin, blocks=true, interfacecolor=:white,
        blockcolor=(:white, 0.2))
end
```

### Stored or live animations

An animation shows either stored or live frames, never a mix. The default
`source=:stored` animates the keyframes in the store after the run,
optionally only those in `window=(t0, t1)`. `source=:live` records the frames
of a recorder as the run produces them, previews and keyframes alike,
including the previews that retention later thins from the store:

```julia
recorder = Recorder(fields; diagnostics=flow, preview=EveryNSteps(5))
recording = export_animation("live.mp4", recorder; field=:velocity_vorticity,
    source=:live, diagnostics=[:kinetic_energy])
simulate!(flow, 1.0; observer=recorder)
close(recording)    # encodes the queued frames and writes live.mp4
recording.encoded, recording.dropped, recording.skipped
```

A live recording never waits for anything and never slows the solver: it
receives frames through a [`subscribe`](@ref) queue of `capacity` frames
(default 32) and encodes every `every`-th frame (default 1) on a thread other
than the solver's. If the encoder falls behind, frames arriving at a full
queue are dropped and counted, and `close` reports the counts. The
recording is independent of any viewer: reading the store, viewing old
frames, or scrubbing history from other tasks does not change it. Without a
second thread (Julia started with `-t 1` and no interactive thread), frames
are queued during the run and encoded after it. Without a registered or given
colour range, a live recording uses the range of the frames recorded so far,
and its diagnostics grow with the run.

### Building blocks

For custom layouts, compose the building blocks:

```julia
fig = Figure(size=(1000, 420))
ax = fieldaxis(fig[1, 1]; camera=(0.0, 1.0, 0.1, 0.4))
frame = Observable(store[first(keyframes(store))])
plot = amrfield!(ax, frame, :volume_fraction; phase_rendering=:plic, blocks=true)
Colorbar(fig[1, 2], plot)
diagnosticsgrid!(fig[1, 3], store; names=[:amplitude],
    references=Dict(:amplitude => :analytical_amplitude),
    cursor=lift(f -> f.time, frame))
save("panel.png", fig)
```

[`fieldaxis`](@ref) creates an axis whose limits always fill its panel with
one isotropic scale: a box zoom on a wide panel shows more of the domain
instead of empty space. `camera` is `:fit`, a rectangle, a domain, or a
[`ViewCamera`](@ref) (centre and world units per pixel). Field axes hide ticks
and draw a scale bar by default (`decorations=:minimal`); `:full` shows
coordinates. [`amrfield!`](@ref) draws a field of a frame, an `Observable` of
frames, or a registry: it rasterizes the visible part of the domain at the
panel's resolution whenever the frame, the limits, or the panel size change,
and draws PLIC interfaces, embedded solids, and optionally block outlines as
vector overlays. Rasterized fields keep vector output small and scale to large
meshes.

[`diagnosticsgrid!`](@ref) draws the diagnostics of a store, recorder,
`FlowSimulation`, or `DiagnosticCollection` in one axis per group of
[`diagnostic_groups`](@ref): diagnostics with the same kind, physical
dimension, and scale share an axis, vector diagnostics are split into
components, and defects use a logarithmic axis. `references` adds dashed
reference curves from functions of time or other diagnostics,
`transform=:relative` plots changes relative to the first sample, and
`until` (a time or an `Observable` of times) shows only the samples up to
that time, reading a live store again whenever it changes.
