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
register_field!(fields, :pressure, field; label="Pressure", units="Pa")
register_field!(fields, :velocity_x, velocity, Face{1}();
                label="Velocity x", units="m/s", colormap=:balance)
register_field!(fields, :velocity_y, velocity, Face{2}();
                label="Velocity y", units="m/s", colormap=:balance)
register_field!(fields, :speed, velocity, CentredMagnitude();
                label="Speed", units="m/s")
register_field!(fields, :vorticity, velocity, Vorticity();
                label="Vorticity", units="1/s", colormap=:balance)
viewer = visualize(fields; field=:pressure)
```

Use `WGLMakie` in place of `GLMakie` for browser rendering. Backends should be
installed into the application environment rather than added as dependencies
of `TwoPhaseAMR`.

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

Names, labels, units, colormaps, and which scalar views to expose remain an
application decision; the registry does not automatically create a
velocity-specific bundle.

Plotted cells and face segments contain owned values only; halo entries are not
rendered. Each cell has four private vertices, preserving piecewise-constant
colors and an unambiguous mapping from a picked primitive to `CellRef`.
Directional face geometry follows the immutable face schedules, including
coarse--fine segments, and maps selections to `FaceRef`. Leaf-block outlines
are a separate plot layer.

`update_visualization!(viewer)` resamples the selected registry entry while
reusing mesh geometry. `select_visualization_field!(viewer, name)` is the
programmatic equivalent of the field menu. Every registered source must share
the exact `BlockTreeMesh` object used to construct the viewer. Replacing the
topology is deferred until solver regridding exists.

The current API includes:

- `register_field!` and `field_names`;
- `visualize(registry; field=nothing, display=true, ...)`;
- `update_visualization!(viewer)` and `select_visualization_field!`;
- the compatible single-cell-field `visualize(field)` and update method;
- `selected_cell(viewer)` and `selected_face(viewer)`; and
- `visualization_figure(viewer)` for further Makie customization.

## Live sessions and checkpoints

A `VisualizationSession` owns copied live frames and retained checkpoints. It
is the object passed to `simulate!`; the solver initializes it with a lazy
independent-step procedure, then reports every successfully completed step.
Copied frame arrays never alias the mutable fields owned by the solver or its
workspaces.

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

Live field publication is independent of checkpoint capture. It defaults to
every completed step, but can be sampled by step count or simulation time:

```julia
by_step = VisualizationSession(fields; viewer, live_every=5)
by_time = VisualizationSession(fields; viewer, live_dt=0.02)
```

Specify at most one of `live_every` and `live_dt`. Initial and final states are
always published. Scheduled checkpoints remain lossless even when a live frame
is not due.

For a GUI, opt into a capacity-one latest-wins mailbox so rendering does not
block the producer:

```julia
session = VisualizationSession(
    fields; viewer, checkpoints=10, live_every=2, asynchronous=true)
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
propagates viewer-task errors. A multi-threaded Julia process is required for
the GUI and numerical solve to make progress concurrently; integrated launch
and pause/resume controls remain a later milestone.

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
