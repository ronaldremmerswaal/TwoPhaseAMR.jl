# TwoPhaseAMR.jl

[![CI](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/ronaldremmerswaal/TwoPhaseAMR.jl/graph/badge.svg)](https://codecov.io/gh/ronaldremmerswaal/TwoPhaseAMR.jl)
[![Documentation](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/actions/workflows/Documentation.yml/badge.svg)](https://ronaldremmerswaal.github.io/TwoPhaseAMR.jl/dev/)
[![Julia 1.11+](https://img.shields.io/badge/Julia-1.11%2B-9558B2?logo=julia)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

TwoPhaseAMR.jl is a Julia package for two-dimensional, incompressible
single- and two-phase flow on block-structured quadtree meshes. It combines
geometric VOF, conservative AMR transfer, embedded solid geometry, one- and
two-velocity formulations, and tree-geometric multigrid behind one simulation
API.

The package is under active development. Julia 1.11 or newer is supported.

## Install

Clone the repository, activate it, and instantiate its dependencies:

```sh
git clone https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl.git
cd TwoPhaseAMR.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Run the test suite with:

```sh
julia --project=. --threads=auto -e 'using Pkg; Pkg.test()'
```

## Quick start

Physics, numerical methods, and evolving state are separate objects. This
small Taylor--Green problem also records kinetic energy and divergence without
creating a visualization:

```julia
using TwoPhaseAMR

mesh = BlockTreeMesh(
    block_size = (16, 16), extent = (2pi, 2pi),
)
physics = SinglePhase(
    density = 1.0, dynamic_viscosity = 0.02,
)
initial_velocity = (x, y, time, axis) ->
    axis == 1 ? sin(x) * cos(y) : -cos(x) * sin(y)

flow = FlowSimulation(
    mesh, physics;
    velocity = initial_velocity,
    boundary = FreeSlip(),
    dt = 1.0e-3,
    diagnostics = (KineticEnergy(), DivergenceNorm()),
)
result = simulate!(flow, 0.002)

energy = diagnostic_history(flow, :kinetic_energy)
@assert result.pressure.converged
@assert length(energy) == result.steps + 1
```

`FlowSimulation` owns the velocity, pressure, equation system, reusable
workspace, time, cumulative step count, and diagnostic histories. Calling
`simulate!` again continues from the stored state.

## Choose a flow model

The high-level API makes the physical and numerical choices explicit:

| Concern | Main types | Default |
| --- | --- | --- |
| Physics | `SinglePhase`, `TwoPhase`, `SurfaceTension` | — |
| Phase velocity | `OneVelocityModel`, `TwoVelocityModel` | one velocity |
| Interface transport | `FrozenInterfaceFluxes`, `ProjectedLiquidVelocity` | frozen |
| Two-velocity pressure | `CMDGFMProjection`, `MimeticProjection` | CMDGFM |
| Surface tension | `CGFM`, `CSF` | CGFM |
| Momentum advection | `AdvectionOptions` | Lax--Wendroff |
| Linear solution | `LinearSolverOptions`, `MultigridOptions` | automatic Krylov and tree GMG |
| Mesh adaptation | `Adaptivity` | disabled until criteria are supplied |

A two-phase setup differs only in its physics and explicit velocity model:

```julia
mesh = BlockTreeMesh(
    block_size = (16, 16), periodic = (false, false),
)
interface = InterfaceField(
    mesh, (x, y) -> hypot(x - 0.5, y - 0.6) - 0.15,
)
physics = TwoPhase(
    interface;
    density = (1000.0, 1.0),
    dynamic_viscosity = (1.0e-3, 1.8e-5),
    surface_tension = 0.072, # numeric shorthand uses CGFM()
    gravity = (0.0, -9.81),
)

flow = FlowSimulation(
    mesh, physics;
    velocity_model = TwoVelocityModel(
        interface_transport = FrozenInterfaceFluxes(),
        pressure_projection = CMDGFMProjection(),
    ),
    boundary = FreeSlip(),
    diagnostics = (KineticEnergy(), LinearMomentum(), PhaseVolume()),
)
```

Replace `TwoVelocityModel(...)` with `OneVelocityModel()` to use a shared
velocity. The same `TwoPhase` object, mesh, boundary conditions, AMR criteria,
and run code work for both formulations.

## Diagnostics and visualization

Diagnostics belong to the simulation and work in headless runs:

```julia
add_diagnostic!(
    flow, :maximum_pressure,
    state -> maximum(
        maximum(abs, interior(state.pressure, block))
        for block in eachindex(state.solver.mesh.blocks)
    );
    schedule = EveryNSteps(5),
)
simulate!(flow, 0.1)
history = diagnostic_history(flow, :maximum_pressure)
```

A live visualization can subscribe to the same stored diagnostic. It does not
own the reduction or maintain a second authoritative history:

```julia
fields = VisualizationFieldRegistry()
register_default_fields!(fields, flow)
register_diagnostic!(fields, flow, :kinetic_energy; label = "Kinetic energy")

session = VisualizationSession(fields; viewer = NoVisualization())
simulate!(flow, 0.2; visualization = session)
```

Load GLMakie and replace `NoVisualization()` with `visualize(fields)` for an
interactive viewer. Diagnostics can use `EveryStep()`, `EveryNSteps(n)`, or
`EveryTimeInterval(dt)`. Custom manual stepping loops call
`record_diagnostics!(flow)` after a completed step.

Discontinuous quantities default to cell-wise values: liquid at a liquid
fraction of at least 50%, gas otherwise. Use
`visualize(fields; phase_rendering = :plic)` to show both phases on sharply
clipped interface geometry instead.

## Examples

The [examples gallery](examples/README.md) collects standalone, illustrated
programs, including:

- [Taylor–Green](examples/01_taylor_green/) — single-phase verification.
- [Capillary wave](examples/06_capillary_wave/) — comparison with
  Prosperetti's analytical solution.
- [Interfacial instability](examples/07_interfacial_instability/) — comparison
  of one- and two-velocity models.

Each example defines `run_example`, runs headlessly when included, and opens a
GLMakie viewer when executed directly. The gallery documents the available
parameters, diagnostics, results, and costs.

## Documentation

- [Hosted documentation](https://ronaldremmerswaal.github.io/TwoPhaseAMR.jl/dev/)
- [Getting started](docs/src/getting-started.md)
- [Choosing a flow model](docs/src/choosing-a-model.md)
- [How-to guides](docs/src/how-to.md)
- [Example catalogue](examples/README.md)
- [High-level API and option matrix](docs/src/api.md)
- [Diagnostics](docs/src/diagnostics.md)
- [Visualization](docs/src/visualization.md)
- [Adaptive refinement](docs/src/adaptive-refinement.md)
- [Navier--Stokes discretization](docs/src/navier-stokes.md)
- [Two-velocity formulation](docs/src/two-velocity-advection.md)
- [Developer architecture](docs/src/architecture.md)
- [Testing and validation](docs/src/testing.md)

The numerical-method pages document the discrete operators and conservation
properties in detail. The public API remains available alongside lower-level
mesh, field, operator, and linear-solver interfaces for method development.

## Benchmarks

The [benchmark suite](benchmark/README.md) covers setup, stepping, adaptivity,
interface transport, linear solution, and two-velocity projection using the
current typed API. It has its own environment so timing dependencies do not
leak into the package or test environments. CI performs a one-sample smoke run;
machine-local timing results are intentionally not committed.

## Scope

The current implementation targets 2-D shared-memory calculations. Three-
dimensional meshes, distributed-memory execution, general pressure Dirichlet
data, and local time subcycling are outside the present scope.

Contributions should preserve numerical invariants, add focused unit tests,
and update the relevant user and developer documentation together with code.
