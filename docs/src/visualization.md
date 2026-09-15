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

Scalar diagnostics are registered on the same registry. A reduction receives
`(velocity, pressure, time, step, solver)` and returns one real value. It may
also close over any additional application field:

```julia
register_diagnostic!(
    fields, :maximum_velocity,
    (velocity, pressure, time, step, solver) ->
        maximum(abs, velocity.interior);
    label="Maximum velocity", units="m/s")
register_diagnostic!(
    fields, :total_pressure,
    (velocity, pressure, time, step, solver) -> integrate(pressure);
    label="Total pressure")
```

Plotted cells and face dual-control-volume tiles contain owned values only;
halo entries are not rendered. Each cell has four private vertices, preserving
piecewise-constant colors and an unambiguous mapping from a picked primitive to
`CellRef`. Directional face values fill the staggered dual cells between
adjacent cell centres, including split tiles at coarse--fine interfaces and
periodic seams. This covers the domain without gaps while retaining a mapping
from every tile to its `FaceRef`. Leaf-block outlines are a separate plot
layer.

`update_visualization!(viewer)` resamples the selected registry entry while
reusing mesh geometry. `select_visualization_field!(viewer, name)` is the
programmatic equivalent of the field menu. Every registered source must share
the exact `BlockTreeMesh` object used to construct the viewer. Replacing the
topology is deferred until solver regridding exists.

The current API includes:

- `register_field!`, `register_default_fields!`, `register_diagnostic!`,
  `field_names`, and `diagnostic_names`;
- `visualize(registry; field=nothing, display=true, ...)`;
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
has refinement indicators, because their independent side step does not yet
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

Diagnostic sampling is independent of full field publication and defaults to
every completed step when diagnostics are registered:

```julia
session = VisualizationSession(
    fields; viewer, live_every=10, diagnostic_every=1)
# or: VisualizationSession(fields; viewer, live_dt=0.1, diagnostic_dt=0.01)
```

Initial and final diagnostic samples are mandatory. The complete scalar
history remains retained even when asynchronous GUI notifications coalesce;
retrieve it with `visualization_diagnostics(session)` and index a sample as
`sample[:maximum_velocity]`.

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

When diagnostics are registered, the viewer adds one time-series axis below
the field view. All reductions are shown in that plot with a legend. Retained
checkpoints are overlaid at their stored-state times as vertical lines and
orange markers. Clicking a marker selects its historical field frame; the
**Live** button returns to the newest frame without clearing the diagnostic
history.

The diagnostic time axis expands from the initial time through the latest
sample as new history arrives; it does not retain Makie's empty-axis default
range.

## Complete GLMakie example

Run the interactive decaying-turbulence showcase with multiple Julia threads:

```sh
julia --project=. --threads=auto examples/visualize_decaying_turbulence.jl
```

The example constructs a deterministic band-limited random streamfunction,
samples it on a periodic staggered AMR mesh, and projects it into the discrete
divergence-free space before starting the solver. The default mesh begins at a
uniform level 4 (128-by-128 equivalent). Interpolation error in the live
cell-centred pressure field coarsens smooth regions while level 2 remains the
protected 32-by-32 floor. `update_interval` limits regrid frequency and
`coarsen_delay` prevents newly refined blocks from immediately coarsening;
level 5 is available as an explicit large-run option.

The viewer starts on vorticity and registers velocity components, centred
magnitude, pressure, normalized energy and enstrophy, peak speed, and maximum
divergence. It also plots the fraction of cells used relative to the initial
uniform finest grid and logs every observed topology change. Its diagnostic
cache rebuilds its face and divergence workspaces after each regrid. The
simulation runs on a worker while Makie updates stay on the GUI task. The
example is two-dimensional freely decaying isotropic turbulence; its dynamics
should not be interpreted as the three-dimensional HIT cascade.

`visualization_checkpoints(session)` returns records with `requested_time` and
the retained `frame.time` and `frame.step`. Retrieve a registered scalar with
`frame[:field_name]`. Checkpoints retain copied primitive fields only; velocity
magnitude and vorticity are recomputed when requested. Call
`select_checkpoint!(session, i)` to show a historical frame. Live ingestion
continues in that mode; `live_mode!(session)` immediately shows the newest
complete frame. A bound Makie viewer exposes the same operations through its
checkpoint menu and **Live** button. Its session label distinguishes live and
historical modes and reports the current lifecycle state. Selecting a
checkpoint changes only the copied frame displayed by the viewer; it never
rewinds or mutates the running solver.

The checkpoint, live-session, diagnostics, and simulation-control design is
specified in milestone 7 of `DEVELOPMENT.md`.
