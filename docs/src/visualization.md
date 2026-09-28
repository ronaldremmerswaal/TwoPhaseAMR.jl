# Interactive visualization

Visualization is implemented as an optional Makie extension. Loading GLMakie
or WGLMakie loads Makie and activates the extension; using the numerical solver
alone does not load a graphics stack.

```julia
using TwoPhaseAMR, GLMakie

mesh = BlockTreeMesh(
    block_size=(12, 12), periodic=(false, false), max_level=2,
    refine=(key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
)
field = CellField(mesh)
initialize!(field, (x, y) -> sinpi(2x) * cospi(2y))
velocity = FaceVelocity(
    mesh, (x, y, time, axis) -> axis == 1 ?
        sinpi(x) * cospi(y) : -cospi(x) * sinpi(y))

fields = VisualizationFieldRegistry()
register_default_fields!(fields, :pressure, field, :velocity, velocity)
viewer = visualize(fields; field=:pressure)
```

Use `WGLMakie` in place of `GLMakie` for browser rendering. Backends should be
installed into the application environment rather than added as dependencies
of `TwoPhaseAMR`.

`register_default_fields!` accepts alternating names and sources. Cell fields
and individual directional face fields produce one entry. A `FaceVelocity` or
an `(x_faces, y_faces)` tuple produces four entries; for a source named
`:velocity`, these are `:velocity_x`, `:velocity_y`, `:velocity_magnitude`, and
`:velocity_vorticity`. Components and vorticity use the diverging `:balance`
colormap, while scalar and magnitude fields use `:viridis`. Labels are derived
from the names; units remain empty because the solver may be dimensional or
nondimensional. Use `register_field!` when names, labels, units, colormaps, or
color ranges need explicit control.

The registry accepts `CellField`, `FaceField{1}`, `FaceField{2}`, and a
directional component of a `FaceVelocity` as scalar views. A two-component
face field can be supplied either as a `FaceVelocity` or as an
`(x_faces, y_faces)` tuple. Both representations support demand-driven
`CentredMagnitude()` and `Vorticity()` views (`CenteredMagnitude()` is an
alias):

```julia
x_faces = FaceField{1}(mesh, (x, y) -> 0.5 - y)
y_faces = FaceField{2}(mesh, (x, y) -> x - 0.5)

register_field!(fields, :x_component, x_faces; label="x component")
register_field!(fields, :y_component, y_faces; label="y component")
register_field!(fields, :magnitude, (x_faces, y_faces), CentredMagnitude())
register_field!(fields, :curl, (x_faces, y_faces), Vorticity())
```

The explicit interface can also expose only a subset of the derived views or
assign application-specific names and styles.

Discontinuous phase scalars are registered as one field. By default, a cell
shows the liquid value when its liquid volume fraction is **at least 50%**, and
the gas value otherwise. There is no blending between phases:

```julia
temperature = PhaseScalar(interface; liquid=liquid_T, gas=gas_T)
register_field!(fields, :temperature, temperature)
```

The same rule applies to `PhaseFaceScalar` and intrinsic two-velocity fields.
Staggered components are centred within each phase first, averaging the two
cell-side values with area weighting across coarse--fine faces. A missing side
uses the other valid side; inactive phase values are never donors. For embedded
geometry, the 50% threshold uses the liquid fraction of **open fluid volume**.
Fully solid cells have no value; filled-solid overlays cover the solid regions.

The sharp geometry-based display remains available as an option:

```julia
fields = VisualizationFieldRegistry(phase_rendering=:plic)
# Register the same fields as usual, then:
viewer = visualize(fields)
# Alternatively: visualize(fields; phase_rendering=:plic)
```

In `:plic` mode both intrinsic values are shown simultaneously on the liquid
and gas PLIC polygons. Staggered fields use dual cells intersected with adjacent
cell reconstructions. `phase_rendering=:cell` is the default and applies to
all discontinuous registered fields; ordinary shared face fields keep their
staggered display in either mode. The Stokes3 example accepts the same
`phase_rendering` keyword.

For a two-velocity field, pass the liquid/gas tuple together with its
interface. The default registration adds four combined component and derived
views:

```julia
phase_velocity = (liquid_velocity, gas_velocity)
register_default_fields!(fields, :velocity, phase_velocity, interface)
```

The resulting names are `:velocity_x`, `:velocity_y`,
`:velocity_magnitude`, and `:velocity_vorticity`. Every view follows the chosen
phase-rendering policy. Magnitude and vorticity are computed **within each
phase before selection**, not from an already switched velocity field. Phase
vorticity uses the same automatic symmetric color range as single-velocity
vorticity. An individual combined view can be registered with, for example,
`register_field!(fields, :curl, phase_velocity, interface, Vorticity())`.

Diagnostics should normally be attached to a [`FlowSimulation`](@ref), then
subscribed to by name. They continue to be evaluated in headless runs and the
simulation retains their authoritative histories:

```julia
add_diagnostic!(flow, KineticEnergy())
register_default_fields!(fields, flow)
register_diagnostic!(
    fields, flow, :kinetic_energy;
    label="Kinetic energy", units="J")
```

For a vector diagnostic such as `LinearMomentum()`, pass `component=1` or
`component=2` and optionally a scalar display `name`. The older overload that
accepts a five-argument visualization reduction remains available, but it is
viewer-local. See [Diagnostics](diagnostics.md) for schedules and custom
simulation reductions.

Plotted cells and face dual-control-volume tiles contain owned values only;
halo entries are not rendered. Continuous fields use four private vertices per
cell, including majority-phase views. Optional `:plic` views triangulate the
liquid and gas polygons with a phase value attached to every private vertex.
Directional shared face values fill staggered dual cells between adjacent cell
centres; in `:plic` mode discontinuous face fields intersect those tiles with
each adjacent cell reconstruction.
Coarse--fine interfaces and periodic seams remain covered, and picked
primitives retain their `CellRef` or `FaceRef`. Leaf-block outlines are a
separate plot layer.

`update_visualization!(viewer)` resamples the selected registry entry.
Continuous entries reuse their mesh geometry while the leaf topology is
unchanged. PLIC entries build clipped geometry only when displayed, sharing
one geometry per interface and placement in each frame. Captures retain small,
typed reconstruction snapshots instead of eagerly triangulating every hidden
field. These snapshots own their data, including embedded open polygons, so
delayed field selection remains valid after AMR. Centring/curl workspaces are
reused, and captured scalar arrays are not duplicated again for the field cache.
`select_visualization_field!(viewer, name)` is the programmatic equivalent of
the field menu. Every registered source must share the exact `BlockTreeMesh`
object used to construct the viewer. After a dynamic regrid, a live or
checkpoint frame carries its matching topology into the viewer. The Makie mesh
primitive is then replaced atomically so its vertices, triangle indices, and
colors always describe one topology; the current zoom and pan limits are
preserved. Cell and block borders use explicit foreground depth offsets, so
replacing the filled mesh cannot cover the existing outline layers.

The current API includes:

- `register_field!`, `register_default_fields!`, `register_diagnostic!`,
  `field_names`, and `diagnostic_names`;
- `VisualizationFieldRegistry(; phase_rendering=:cell)`;
- `visualize(registry; field=nothing, phase_rendering=registry.phase_rendering,
  display=true, ...)`;
- `update_visualization!(viewer)` and `select_visualization_field!`;
- the compatible single-cell-field `visualize(field)` and update method;
- `selected_cell(viewer)` and `selected_face(viewer)`; and
- `visualization_figure(viewer)` for further Makie customization.

## Live sessions, checkpoints, and diagnostics

A `VisualizationSession` owns copied live frames and retained checkpoints. It
is the object passed to `simulate!`; the solver initializes it with a lazy
independent-step procedure, then reports every successfully completed step.
Copied frame arrays never alias the mutable fields owned by the solver or its
workspaces. Each frame also references an immutable topology snapshot shared
while the leaf set is unchanged. Historical frames therefore retain the
correct cell/face geometry and cached derived scalars after a dynamic regrid.

Choose exactly one checkpoint specification:

```julia
by_count = VisualizationSession(fields; viewer, checkpoints=10)
by_interval = VisualizationSession(fields; viewer, checkpoint_dt=0.1)
by_time = VisualizationSession(
    fields; viewer, checkpoint_times=[0.0, 0.2, 0.7, 1.0])
```

An integer `checkpoints` value means an equispaced count. A real-valued
`checkpoints` value is also accepted as shorthand for `checkpoint_dt`, and a
vector as shorthand for `checkpoint_times`. Count and interval schedules
include the initial and final simulation states; explicit times are absolute,
sorted, deduplicated, and must lie inside the run interval.

By default, when a step crosses a requested time the session compares the
completed states on either side and retains the closer one (choosing the
earlier state on an exact tie). Set `exact_checkpoints=true` to compute exact
off-grid states on copied solver data:

```julia
session = VisualizationSession(
    fields; viewer, checkpoints=10, exact_checkpoints=true)
result = simulate!(
    velocity, pressure, solver, final_time, workspace;
    visualization=session)
```

When an update crosses a checkpoint strictly inside a nominal step, the
session copies the preceding velocity and pressure and invokes the supplied
step procedure. That procedure lazily creates and owns an independent solver
workspace, then calls the same `step!` kernel. The session retains the side-step
result and discards the copied numerical state. `simulate!` does not inspect
checkpoint schedules or alter its nominal step. Exact checkpoints therefore
cost additional solves and make the synchronous observer update more
expensive, but do not change the simulation's step sequence or discrete
trajectory.

Nearest-solution checkpoints support dynamic refinement. Exact off-grid
checkpoints currently require fixed topology and are rejected when a solver
has refinement criteria, because their independent side step does not yet
clone historical tree and operator state.

Live field publication is independent of checkpoint capture. It defaults to
every completed step, but can be sampled by step count or simulation time:

```julia
by_step = VisualizationSession(fields; viewer, live_every=5)
by_time = VisualizationSession(fields; viewer, live_dt=0.02)
```

Specify at most one of `live_every` and `live_dt`. Initial and final states are
always published. Scheduled checkpoints remain lossless even when a live frame
is not due.

Diagnostic plotting is independent of full field publication and defaults to
every completed step when subscribed diagnostics are registered:

```julia
session = VisualizationSession(
    fields; viewer, live_every=10, diagnostic_every=1)
# or: VisualizationSession(fields; viewer, live_dt=0.1, diagnostic_dt=0.01)
```

Initial and final plot samples are retained even when asynchronous GUI
notifications coalesce. `visualization_diagnostics(session)` returns the
scalar plot samples; `diagnostic_history(flow, name)` returns the independent
simulation history.

For a GUI, opt into a capacity-one latest-wins mailbox so rendering does not
block the producer:

```julia
session = VisualizationSession(
    fields; viewer, checkpoints=10, live_every=5,
    diagnostic_every=1, asynchronous=true)
simulation = Threads.@spawn simulate!(
    velocity, pressure, solver, final_time, workspace;
    visualization=session)
result = fetch(simulation)
wait_visualization!(session)
```

Construct the viewer and session on the GUI/event-loop task before launching
the simulation task. If several sampled frames arrive while the viewer is
busy, only the newest transient live frame is rendered. Query
`dropped_visualization_frames(session)` for the replacement count. Checkpoints
do not pass through this mailbox. After the simulation completes,
`wait_visualization!(session)` waits until the final frame has been applied and
propagates viewer-task errors.

Pause and resume are cooperative at completed-step boundaries:

```julia
request_pause!(session)
@assert simulation_state(session) == :pause_requested
wait_until_paused(session)
@assert simulation_state(session) == :paused
resume_simulation!(session)
```

The pause request never interrupts a numerical kernel, so its latency is at
most one step. `resume_simulation!` can also cancel a request before the next
boundary. The Makie **Pause** button provides the same operations and changes
to **Resume** for both pending and reached pauses. The observer blocks its own
completed-step update while paused; `simulate!` remains unaware of control
state. A multi-threaded Julia process is required for the GUI and numerical
solve to make progress concurrently.

When diagnostics are subscribed, the viewer adds one time-series axis below
the field view. All scalar values are shown in that plot with a legend. Retained
checkpoints are overlaid at their stored-state times as vertical lines and
orange markers. Clicking a marker selects its historical field frame; the
**Live** button returns to the newest frame without clearing the diagnostic
history.

The diagnostic time axis expands from the initial time through the latest
sample as new history arrives; it does not retain Makie's empty-axis default
range.

## Adding a viewer to an example

The standalone scripts deliberately keep `FlowSimulation`, field registration,
and `simulate!` visible. Included examples default to `visualize=false`, while
direct terminal execution passes `visualize=true` and opens GLMakie. For
example:

```sh
julia --project=. --threads=auto examples/decaying_turbulence.jl
```

That case constructs a deterministic band-limited random streamfunction and
uses interpolation error in pressure to coarsen a mesh that starts uniformly
at the finest level. Each live example subscribes to simulation-owned
diagnostics, so the same histories remain available in headless runs.

`visualization_checkpoints(session)` returns records with `requested_time` and
the retained `frame.time` and `frame.step`. Retrieve a registered scalar with
`frame[:field_name]`. Checkpoints retain owned scalar values, including
phase-derived magnitude and vorticity computed at capture time. Optional PLIC
geometry is built and cached when first requested. Call
`select_checkpoint!(session, i)` to show a historical frame. Live ingestion
continues in that mode; `live_mode!(session)` immediately shows the newest
complete frame. A bound Makie viewer exposes the same operations through its
checkpoint menu and **Live** button. Its session label distinguishes live and
historical modes and reports the current lifecycle state. Selecting a
checkpoint changes only the copied frame displayed by the viewer; it never
rewinds or mutates the running solver.
