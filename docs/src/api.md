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

`NavierStokesSystem` is the preferred name for the configured PDE system.
`NavierStokesSolver` and `IncompressibleNavierStokesSolver` remain available
for compatibility. The `solvers/` source area is reserved for linear solvers.

### Compact standard API

Most applications need only this surface:

| Task | Standard entry points |
| --- | --- |
| mesh | [`BlockTreeMesh`](@ref), [`MeshAdaptationOptions`](@ref) |
| physics | [`SinglePhase`](@ref), [`TwoPhase`](@ref), [`SurfaceTension`](@ref) |
| velocity model | [`OneVelocityModel`](@ref), [`TwoVelocityModel`](@ref) |
| numerical policy | [`NavierStokesOptions`](@ref), [`LinearSolverOptions`](@ref), [`MultigridOptions`](@ref), [`Adaptivity`](@ref) |
| run and inspect | [`FlowSimulation`](@ref), [`simulate!`](@ref), [`diagnostic_history`](@ref), [`print_simulation_summary`](@ref) |
| optional live view | [`VisualizationFieldRegistry`](@ref), [`VisualizationSession`](@ref), [`visualize`](@ref) |

The remaining exports are supported advanced building blocks for custom
operators, low-level stepping, validation, and visualization backends. Their
presence does not imply that an introductory application must assemble those
layers manually. Compatibility names such as `NavierStokesSolver`,
`NavierStokesSimulation`, `MimeticGradient`, and the non-tree multigrid names
are intentionally outside the standard surface.

## Two-phase option matrix

| Choice | Configure on | Values | Default |
| --- | --- | --- | --- |
| velocity formulation | `FlowSimulation` | `OneVelocityModel()`, `TwoVelocityModel(...)` | one velocity |
| surface-tension force | `SurfaceTension` | `CGFM()`, `CSF()`, `GFM(conservative=false)` | CGFM |
| two-velocity interface carrier | `TwoVelocityModel` | `FrozenInterfaceFluxes()`, `ProjectedLiquidVelocity()` | frozen |
| two-velocity projection | `TwoVelocityModel` | `CMDGFMProjection()`, `MimeticProjection()` | CMDGFM |

Interface transport and two-velocity projection are meaningful only for
`TwoVelocityModel`. Surface-tension force selection applies to either velocity
model. A liquid/gas velocity tuple is rejected by `OneVelocityModel` instead
of silently changing the formulation.

A nonnegative number is shorthand for the default conservative ghost-fluid
choice: `surface_tension=0.05` is equivalent to
`surface_tension=SurfaceTension(0.05; method=CGFM())`.

## Migration from the legacy constructors

The compatibility entry points remain supported, but new code should compose
the physical model and numerical choices explicitly:

| Legacy spelling | Preferred spelling |
| --- | --- |
| `NavierStokesSolver(...)` or `IncompressibleNavierStokesSolver(...)` | `NavierStokesSystem(...)` for low-level equation-system construction |
| `NavierStokesSimulation(...)` | `FlowSimulation(...)` |
| `FlowSimulation(mesh; density=..., dynamic_viscosity=...)` | `FlowSimulation(mesh, SinglePhase(...); ...)` or `FlowSimulation(mesh, TwoPhase(...); ...)` |
| `interface_method=FrozenInterfaceFluxes()` | `interface_transport=FrozenInterfaceFluxes()` |
| `surface_tension_formulation=CGFM()` with a numeric coefficient | `surface_tension=SurfaceTension(coefficient; method=CGFM())`, or the numeric `surface_tension=coefficient` shorthand |
| `two_velocity_projection=CMDGFMProjection()` | `velocity_model=TwoVelocityModel(pressure_projection=CMDGFMProjection())` |
| `MimeticGradient()` | `MimeticProjection()` |
| `GeometricMultigrid*` compatibility names | corresponding `TreeGeometricMultigrid*` names |

Do not pass both `interface_transport` and `interface_method`; construction
rejects the ambiguous combination. A tuple of liquid/gas initial velocities
does not select a model implicitly: pass `velocity_model=TwoVelocityModel()`.
Invalid surface-force, transport, and projection objects are rejected during
configuration rather than during a time step.

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
- visualization registries and checkpoint sessions.

## Types and constants

```@autodocs
Modules = [TwoPhaseAMR]
Private = false
Order = [:type, :constant]
```
