# TwoPhaseAMR.jl

[![CI](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/ronaldremmerswaal/TwoPhaseAMR.jl/graph/badge.svg)](https://codecov.io/gh/ronaldremmerswaal/TwoPhaseAMR.jl)
[![Documentation](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/actions/workflows/Documentation.yml/badge.svg)](https://ronaldremmerswaal.github.io/TwoPhaseAMR.jl/dev/)
[![Julia 1.10+](https://img.shields.io/badge/Julia-1.10%2B-9558B2?logo=julia)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

TwoPhaseAMR.jl is a Julia package for two-dimensional, incompressible
single- and two-phase flow on block-structured quadtree meshes. It combines
geometric VOF, conservative AMR transfer, embedded solid geometry, one- and
two-velocity formulations, and tree-geometric multigrid behind one simulation
API.

The package is under active development. Julia 1.10 or newer is supported.

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
| Surface tension | `CGFM`, `CSF`, `GFM(conservative=false)` | CGFM |
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

## Examples

The examples are standalone Julia files. Each defines `run_example`, takes
physical parameters plus `max_level`, `final_time`, and `visualize=false`,
forwards remaining keywords to `FlowSimulation`, and lets the solver select a
stable time step. Direct terminal execution passes `visualize=true`.

| Example | What it demonstrates |
| --- | --- |
| [`taylor_green.jl`](examples/taylor_green.jl) | concise single-phase verification |
| [`lid_driven_cavity.jl`](examples/lid_driven_cavity.jl) | moving-wall boundary conditions |
| [`decaying_turbulence.jl`](examples/decaying_turbulence.jl) | pressure-driven dynamic AMR |
| [`capillary_wave.jl`](examples/capillary_wave.jl) | numerical and linear capillary-wave amplitudes |
| [`sessile_droplet.jl`](examples/sessile_droplet.jl) | contact angle, gravity, and dynamic AMR |
| [`sloped_sessile_droplet.jl`](examples/sloped_sessile_droplet.jl) | embedded PLIC geometry |
| [`stokes3_wave.jl`](examples/stokes3_wave.jl) | nonlinear wave with three AMR criteria |
| [`lgpi.jl`](examples/lgpi.jl) | large gas-pocket impact benchmark |
| [`interfacial_instability.jl`](examples/interfacial_instability.jl) | numerical/linear KH/RT amplitudes; two velocity by default |
| [`cylinder_flow.jl`](examples/cylinder_flow.jl) | cut-cell flow and force diagnostics |
| [`vortex_reverse.jl`](examples/vortex_reverse.jl) | prescribed-velocity geometric VOF transport |

Every Navier--Stokes two-phase example accepts either velocity model through a
forwarded keyword. The instability intentionally defaults to two velocity.
For example:

```julia
include("examples/capillary_wave.jl")
run = run_example(
    max_level=1,
    final_time=0.01,
    velocity_model=TwoVelocityModel(),
)
print_simulation_summary(run.flow, run.result)
```

See the [example catalogue](examples/README.md) for commands, expected costs,
and model defaults. `vortex_reverse.jl` prescribes its carrier velocity and
therefore has no Navier--Stokes velocity-model selection.

## Documentation

- [Hosted documentation](https://ronaldremmerswaal.github.io/TwoPhaseAMR.jl/dev/)
- [Getting started](docs/src/getting-started.md)
- [High-level API and option matrix](docs/src/api.md)
- [Diagnostics](docs/src/diagnostics.md)
- [Visualization](docs/src/visualization.md)
- [Adaptive refinement](docs/src/adaptive-refinement.md)
- [Navier--Stokes discretization](docs/src/navier-stokes.md)
- [Two-velocity formulation](docs/src/two-velocity-advection.md)
- [Developer architecture](docs/src/architecture.md)

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
