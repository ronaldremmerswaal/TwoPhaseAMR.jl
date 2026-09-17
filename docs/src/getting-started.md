# Getting started

TwoPhaseAMR provides a 2-D, serial incompressible-flow solver on uniform and
block-adaptive Cartesian meshes. Single- and two-phase calculations use the
same three high-level objects: a mesh, a simulation, and its result.

```@example getting-started
using TwoPhaseAMR

mesh = BlockTreeMesh(block_size=(8, 8))
flow = FlowSimulation(
    mesh;
    viscosity=0.02,
    velocity=(x, y, time, axis) ->
        axis == 1 ? sinpi(2x) * cospi(2y) :
                    -cospi(2x) * sinpi(2y),
    boundary=FreeSlip(),
    dt=0.01,
)

result = simulate!(
    flow, 0.01; rtol=1e-8, max_cycles=40)
(result.time, result.steps, result.pressure.converged)
```

`flow.velocity` and `flow.pressure` are the evolved fields. The object also
owns `flow.solver`, `flow.workspace`, the current `flow.time`, and cumulative
step and mesh-update counters. Calling `simulate!` again continues rather than
restarting:

```@example getting-started
continued = simulate!(
    flow, 0.02; rtol=1e-8, max_cycles=40)
(flow.time, continued.steps, continued.total_steps)
```

## Initial conditions

The default velocity and pressure are zero. Constant velocity can be a tuple,
and pressure can be a number:

```julia
flow = FlowSimulation(
    mesh; viscosity=1e-3, velocity=(1.0, 0.0), pressure=0.0)
```

Use callables for spatially varying initial data. A velocity callable receives
`(x, y, time, axis)` and a pressure callable receives `(x, y)`. Existing
`FaceVelocity` and `CellField` objects are accepted without copying, provided
they use the same mesh and scalar type.

## Two-phase initial conditions

Supplying an [`InterfaceField`](@ref) selects the one-velocity two-phase model.
Construct physical interfaces from a level set `phi(x,y)`, with liquid where
`phi <= 0`; the constructor integrates each cut-cell fraction and reconstructs
its PLIC geometry. A cell-centre Boolean indicator is not an accurate volume
fraction initializer.

```julia
mesh = BlockTreeMesh(
    block_size=(32, 32), periodic=(false, false))
phi = (x, y) -> hypot(x - 0.5, y - 0.6) - 0.15
interface = InterfaceField(mesh, phi)

flow = FlowSimulation(
    mesh;
    interface,
    density=(1000.0, 1.0),
    dynamic_viscosity=(1e-3, 1.8e-5),
    surface_tension=0.072,
    surface_tension_formulation=GFM(),
    gravity=(0.0, -9.81),
    dt=1e-4,
)
```

The entries in `density` and `dynamic_viscosity` are ordered `(liquid, gas)`.
The separate single-phase `viscosity` keyword denotes kinematic viscosity.
Volume fraction, density, phase scalar values, and velocity are stored as
intrinsic quantities. Conservative phase products are temporary values inside
transport and mesh-transfer operations.
The active coupled stress operator is available as
`flow.diffusion_operator`; its cell-centred geometric mixture can be inspected
with `dynamic_viscosity(flow.interface; viscosity=(mu_liquid, mu_gas))`.

## Boundaries and time steps

Meshes are periodic by default. Set `periodic=(false, false)` for walls;
no-permeation is then automatic. `NoSlip()` is the default viscous wall model,
while `FreeSlip()` and moving-wall `NoSlip(f)` are available. Fixed embedded
geometry supports stationary `NoSlip()` and `FreeSlip()` as well; free slip
omits the physical cut-wall shear stresses while retaining the constraints on
fully covered background-grid velocity unknowns.

If `dt` is omitted, `simulate!` chooses a convective CFL step. Implicit
diffusion deliberately adds no viscous CFL restriction. A fixed `dt` is useful
for reproducible studies and maximizes reuse of the shifted diffusion
hierarchy. Pass it to `FlowSimulation` when it is known at setup; it then
becomes the default for every call to `simulate!` and avoids constructing a
temporary zero-shift hierarchy. A `dt` passed to `simulate!` overrides that
default for the call.

## Where to go next

- [Single-phase flow tutorial](single-phase-tutorial.md) develops and checks a
  complete Taylor--Green calculation.
- [Two-phase implementation record](two-phase-development.md) documents the
  geometric transport, material operators, forces, AMR, and validation.
- [Adaptive refinement](adaptive-refinement.md) adds indicators and field
  transfer.
- [Visualization](visualization.md) attaches the optional interactive GUI.
- [Advanced workflows](advanced-workflows.md) exposes operators, workspaces,
  Krylov methods, and preconditioners directly.
