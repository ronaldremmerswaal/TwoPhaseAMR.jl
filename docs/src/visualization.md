# Recording and visualization

The numerical core records simulations without any graphics dependency:

1. a [`VisualizationFieldRegistry`](@ref) names the fields to record;
2. a [`Recorder`](@ref), passed to [`simulate!`](@ref) as its `observer`,
   copies owned frames of those fields into a bounded [`SnapshotStore`](@ref);
3. [`rasterize`](@ref) samples a stored [`Frame`](@ref) on a pixel grid, and a
   [`Camera`](@ref) describes isotropic views.

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
another mesh, throw before the first step. [`subscribe`](@ref) returns a
latest-wins notification of newly stored frames for consumers on other tasks.

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

A [`Camera`](@ref) is a centre and one isotropic scale (world units per pixel),
so a view always fills its panel and cells stay square. [`fit_camera`](@ref)
shows a whole domain, [`zoom`](@ref) keeps the point under the cursor fixed,
[`zoom_to_box`](@ref) shows a box completely, and [`visible_rect`](@ref)
returns the rectangle to rasterize.

## Static figures

Load CairoMakie for widget-free figures, including in headless scripts:

```julia
using TwoPhaseAMR, CairoMakie
fig = snapshot_figure(fields; field=:velocity_vorticity, blocks=true)
save("vorticity.png", fig)
```

A registry source captures its current state; pass a stored `Frame` to plot a
recorded state. The field is drawn as an image of [`rasterize`](@ref) with
`resolution` pixels along the longer domain side; block outlines, PLIC
interfaces, and embedded solids are vector overlays. `phase_rendering`
selects the phase display. Use `snapshot_figure!(fig[1, i], frame; field=...)`
for panels, `plot_diagnostics!` for simulation histories, and
`record_checkpoints(path, recorder; field=...)` to write the stored keyframes
as a GIF or video with fixed limits and a colour range taken over all
keyframes.
