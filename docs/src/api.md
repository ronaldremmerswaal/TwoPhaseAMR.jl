# Public API

```@docs
TwoPhaseAMR
```

## High-level composition

| Layer | Public configuration | Owns |
| --- | --- | --- |
| Mesh | [`BlockTreeMesh`](@ref), [`MeshAdaptationOptions`](@ref) | topology, geometric schedules, AMR timing |
| Physics | [`SinglePhase`](@ref), [`TwoPhase`](@ref), [`SurfaceTension`](@ref) | material properties and body forces |
| Velocity model | [`OneVelocityModel`](@ref), [`TwoVelocityModel`](@ref) | shared or intrinsic phase velocity choices |
| Numerics | [`NavierStokesOptions`](@ref), [`AdvectionOptions`](@ref) | discretization choices |
| Linear algebra | [`LinearSolverOptions`](@ref), [`MultigridOptions`](@ref) | Krylov, preconditioner, multigrid policy |
| Adaptivity | [`Adaptivity`](@ref) | refinement criteria and additional fields |
| State | [`FlowSimulation`](@ref) | fields, equation system, workspace, force, time, histories |

`flow.system` is the configured [`NavierStokesSystem`](@ref); it retains the
physical configuration as `flow.system.physics` and numerical choices as
`flow.system.options`.
The `solvers/` source area is reserved for linear solvers.

### Compact standard API

Most applications need only this surface:

| Task | Standard entry points |
| --- | --- |
| mesh | [`BlockTreeMesh`](@ref), [`MeshAdaptationOptions`](@ref) |
| physics | [`SinglePhase`](@ref), [`TwoPhase`](@ref), [`SurfaceTension`](@ref) |
| velocity model | [`OneVelocityModel`](@ref), [`TwoVelocityModel`](@ref) |
| numerical policy | [`NavierStokesOptions`](@ref), [`LinearSolverOptions`](@ref), [`MultigridOptions`](@ref), [`Adaptivity`](@ref) |
| run and inspect | [`FlowSimulation`](@ref), [`step!`](@ref), [`simulate!`](@ref), [`diagnostic_history`](@ref), [`print_simulation_summary`](@ref) |
| optional recording | [`VisualizationFieldRegistry`](@ref), [`Recorder`](@ref), [`snapshot_store`](@ref) |

The remaining exports are supported advanced building blocks for custom
operators, low-level stepping, validation, and visualization backends. Their
presence does not imply that an introductory application must assemble those
layers manually.

### Advancing a flow

`FlowSimulation(...; force=nothing)` stores the optional body-acceleration
callback as `flow.force`. `step!(flow, dt; force=flow.force,
observer=NoObserver(), tolerance=sqrt(eps(T)), max_cycles=nothing)` advances
one full step and returns its pressure-projection statistics. It updates
time and counters, performs any due mesh adaptation and associated rebuild,
records step diagnostics, and emits a `:step` observer event.

`simulate!(flow, final_time; ...)` selects and repeats full steps to an
absolute end time, emitting `:initial`, `:completed`, and `:failed` lifecycle
events as appropriate. Consecutive calls continue from the retained state and,
when applicable, the unshortened automatic timestep proposal. A standalone
`step!` emits no run lifecycle events; observers such as `Recorder` that rely
on them should be used with `simulate!`. By contrast,
the field-level `step!(velocity, pressure, system, dt, workspace; ...)`
performs the physical solve without flow-owned time, counters, adaptation,
diagnostics, or observer events.

## Two-phase option matrix

| Choice | Configure on | Values | Default |
| --- | --- | --- | --- |
| velocity formulation | `FlowSimulation` | `OneVelocityModel()`, `TwoVelocityModel(...)` | one velocity |
| surface-tension force | `SurfaceTension` | `CGFM()`, `CSF()` | CGFM |
| two-velocity interface carrier | `TwoVelocityModel` | `FrozenInterfaceFluxes()`, `ProjectedLiquidVelocity()` | frozen |
| two-velocity projection | `TwoVelocityModel` | `CMDGFMProjection()`, `MimeticProjection()` | CMDGFM |

Interface transport and two-velocity projection are meaningful only for
`TwoVelocityModel`. Surface-tension force selection applies to either velocity
model. A liquid/gas velocity tuple is rejected by `OneVelocityModel` instead
of silently changing the formulation.

A nonnegative number is shorthand for the default conservative ghost-fluid
choice: `surface_tension=0.05` is equivalent to
`surface_tension=SurfaceTension(0.05; method=CGFM())`.

`SinglePhase`, `TwoPhase`, `SurfaceTension`, `AdvectionOptions`, and
`NavierStokesOptions` validate values at construction, including when fields
are supplied positionally. Phase densities must be positive, and phase
viscosities must either both be positive or both be zero.
`InterfaceField` validates `reconstruction_method` as `:lhf`, `:lhf_ghf`, or
`:wy`; [`rasterize`](@ref) and the static figure functions validate
`phase_rendering` as `:cell` or `:plic`. Pressure projection's `compatibility`
is a solve-time keyword and accepts `:project` or `:error`.


## Diagnostics

Built-in diagnostics are [`KineticEnergy`](@ref),
[`PotentialEnergy`](@ref), [`SurfaceEnergy`](@ref), [`TotalEnergy`](@ref),
[`LinearMomentum`](@ref), [`PhaseVolume`](@ref),
[`PhaseMass`](@ref), [`InterfaceMeasure`](@ref),
[`MaximumVelocity`](@ref), [`CFLTimeStep`](@ref),
[`DivergenceNorm`](@ref), [`LeafBlockCount`](@ref), and [`CellCount`](@ref).
Add custom reductions with [`Diagnostic`](@ref) or
[`add_diagnostic!`](@ref). Sampling schedules are [`EveryStep`](@ref),
[`EveryNSteps`](@ref), and [`EveryTimeInterval`](@ref). Use
[`print_simulation_summary`](@ref) for an end-of-run model, mesh, solver, AMR,
and diagnostics report.

See [Diagnostics](diagnostics.md) for definitions and recording of
simulation-owned histories.

## Recording, Studio, and static output

[`register_default_fields!`](@ref) populates a
[`VisualizationFieldRegistry`](@ref) from a flow. Pass a [`Recorder`](@ref)
as `observer` to [`simulate!`](@ref); [`snapshot_store`](@ref) exposes its
owned frames, scheduled keyframes, mirrored diagnostics, and step log.
[`AbstractObserver`](@ref), [`StepInfo`](@ref), and [`observe!`](@ref) are the
extension points for custom observers. See
[Recording and visualization](visualization.md#recording) for schedules,
retention, controls, and examples.

The separately installed [Studio](visualization.md#studio) subpackage opens a
browser workspace with `studio(recorder)`. With a Makie backend loaded,
[`export_figure`](@ref) and [`export_animation`](@ref) produce static output;
[`fieldaxis`](@ref), [`amrfield`](@ref), [`amrfield!`](@ref), and
[`diagnosticsgrid!`](@ref) compose custom layouts. [`rasterize`](@ref),
[`locate`](@ref), [`locate_face`](@ref), and [`ViewCamera`](@ref) provide
graphics-independent field sampling and view geometry.

### Studio subpackage API

These names are exported by `TwoPhaseAMRStudio`, which is installed separately
from the numerical core. See [Studio setup and usage](visualization.md#studio).

| Entry point | Purpose |
| --- | --- |
| `studio(recorder; name="Run 1", layout=:auto, host="127.0.0.1", port=0, open=:browser, theme=:system)` | Start a server for a recorder. `open=:none` skips browser launch; `close(app)` closes the server without stopping the solver. |
| `url(app)` | Return the server URL, including its secret access token. |
| `encode_field(values, request; encoding=:f32)` | Encode a real-valued array as a protocol-v1 binary field message using `:f32` or quantized `:u16`; `request` is a UInt32-range identifier. |
| `decode_field(bytes)` | Validate and decode a protocol-v1 field message, returning `request`, `encoding`, Float32 `values`, `scale`, and `offset`. |

The protocol's `:u16` encoding reserves `0xffff` for nonfinite values, which
decode to `NaN`. See [Studio protocol v1](visualization.md#studio-protocol-v1)
for the message format and security guidance.

### Staggered face queries

`locate_face(frame, (x, y), axis)` returns a face reference in the recorded
topology's global interior or boundary schedule; `axis` is 1 or 2. The
reference is not a direct index into the per-axis values stored in a frame.

```@docs
FaceRef
locate_face
```

## Lower-level layers

The high-level API does not introduce duplicate field representations.
Advanced users can work directly with:

- fields, placements, halo exchange, and conservative transfer;
- mesh-adaptation workspaces and refinement criteria;
- cell and staggered advection/diffusion operators;
- interface reconstruction and compatible phase transport;
- scalar and staggered elliptic operators;
- Krylov and multigrid workspaces;
- field registries, recorders, snapshot stores, and frame rasterization.

## Types and constants

```@autodocs
Modules = [TwoPhaseAMR]
Private = false
Filter = t -> t !== TwoPhaseAMR.FaceRef
Order = [:type, :constant]
```
