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
| State | [`FlowSimulation`](@ref) | fields, equation system, workspace, time, histories |

`NavierStokesSystem` names the configured PDE system.
The `solvers/` source area is reserved for linear solvers.

### Compact standard API

Most applications need only this surface:

| Task | Standard entry points |
| --- | --- |
| mesh | [`BlockTreeMesh`](@ref), [`MeshAdaptationOptions`](@ref) |
| physics | [`SinglePhase`](@ref), [`TwoPhase`](@ref), [`SurfaceTension`](@ref) |
| velocity model | [`OneVelocityModel`](@ref), [`TwoVelocityModel`](@ref) |
| numerical policy | [`NavierStokesOptions`](@ref), [`LinearSolverOptions`](@ref), [`MultigridOptions`](@ref), [`Adaptivity`](@ref) |
| run and inspect | [`FlowSimulation`](@ref), [`simulate!`](@ref), [`diagnostic_history`](@ref), [`print_simulation_summary`](@ref) |
| optional recording | [`VisualizationFieldRegistry`](@ref), [`Recorder`](@ref), [`snapshot_store`](@ref) |

The remaining exports are supported advanced building blocks for custom
operators, low-level stepping, validation, and visualization backends. Their
presence does not imply that an introductory application must assemble those
layers manually.

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

See [Diagnostics](diagnostics.md) for definitions and visualization
subscription.

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
Order = [:type, :constant]
```
