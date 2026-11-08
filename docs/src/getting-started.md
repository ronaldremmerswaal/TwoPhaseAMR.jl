# Getting started

This tutorial builds a flow, advances it, and inspects a diagnostic history.
The same sequence applies to single-phase and both two-phase velocity models.

## A single-phase simulation

```@example getting-started
using TwoPhaseAMR

mesh = BlockTreeMesh(
    block_size = (8, 8), extent = (1.0, 1.0),
)
physics = SinglePhase(
    density = 1.0, dynamic_viscosity = 0.02,
)
flow = FlowSimulation(
    mesh, physics;
    velocity = (x, y, time, axis) ->
        axis == 1 ? sinpi(2x) * cospi(2y) :
                    -cospi(2x) * sinpi(2y),
    boundary = FreeSlip(),
    dt = 0.01,
    diagnostics = (KineticEnergy(), DivergenceNorm()),
)

result = simulate!(flow, 0.01; tolerance = 1e-8, max_cycles = 40)
(result.time, result.steps, result.pressure.converged)
```

`flow.velocity`, `flow.pressure`, and `flow.solver` are the same field and
equation-system objects used by the lower-level API. `flow.workspace` is
retained between calls. A second call continues the calculation:

```@example getting-started
continued = simulate!(flow, 0.02; tolerance = 1e-8, max_cycles = 40)
(flow.time, continued.steps, continued.total_steps)
```

Diagnostic histories are independent of visualization:

```@example getting-started
energy = diagnostic_history(flow, :kinetic_energy)
(energy.times, energy.steps, energy.values)
```

Print the complete end-of-run summary with:

```@example getting-started
print_simulation_summary(flow, continued)
```

## Mesh resolution and storage blocks

`base_mesh` is the physical level-zero resolution. `block_size` controls
storage and scheduling. The base resolution must be divisible by the block
size componentwise:

```@example getting-started
rectangular_mesh = BlockTreeMesh(
    extent = (1.0, 3.0),
    base_mesh = (32, 96),
    block_size = (8, 8),
    periodic = (true, false),
)
(rectangular_mesh.base_mesh, rectangular_mesh.block_size,
 nblocks(rectangular_mesh))
```

When `base_mesh` is omitted, it equals `block_size`. Refinement levels are
measured relative to this base mesh.

## A two-phase simulation

An [`InterfaceField`](@ref) stores liquid volume divided by Cartesian cell
volume. Construct it from a level set; liquid is the region where `phi <= 0`.
The constructor integrates cut-cell volume rather than sampling the centre.

```@example getting-started
mesh = BlockTreeMesh(
    block_size = (8, 8), periodic = (false, false), max_level = 2,
)
interface = InterfaceField(
    mesh, (x, y) -> hypot(x - 0.5, y - 0.6) - 0.15,
)
physics = TwoPhase(
    interface;
    density = (1000.0, 1.0),
    dynamic_viscosity = (1e-3, 1.8e-5),
    surface_tension = 0.072, # shorthand for SurfaceTension(...; method=CGFM())
    gravity = (0.0, -9.81),
)
flow = FlowSimulation(
    mesh, physics;
    velocity_model = OneVelocityModel(),
    boundary = FreeSlip(),
    diagnostics = (KineticEnergy(), LinearMomentum(), PhaseVolume()),
)
simulate!(flow, 1e-4; dt = 1e-4)
```

Select separate intrinsic phase velocities explicitly:

```@example getting-started
two_velocity_flow = FlowSimulation(
    mesh, physics;
    velocity_model = TwoVelocityModel(
        interface_transport = FrozenInterfaceFluxes(),
        pressure_projection = CMDGFMProjection(),
    ),
    boundary = FreeSlip(),
)
simulate!(two_velocity_flow, 1e-4; dt = 1e-4)
```

A shared velocity initializer is copied into both phases. A tuple of liquid
and gas callables or `FaceVelocity` fields initializes them independently.

## Numerical configuration

Physics is not mixed into the linear-solver keyword list:

```@example getting-started
numerics = NavierStokesOptions(
    advection = AdvectionOptions(scheme = MonotonizedCentral()),
    pressure = LinearSolverOptions(
        preconditioner = :tree_geometric,
        krylov_options = (; rtol = 1e-8),
    ),
    diffusion = LinearSolverOptions(preconditioner = :tree_geometric),
    multigrid = MultigridOptions(
        coarse_size = 32, pre_sweeps = 2, post_sweeps = 2,
    ),
)
```

AMR state is similarly explicit:

```@example getting-started
criterion = InterfaceRefinementIndicator(interface; level = 2)
adaptivity = Adaptivity(criteria = (criterion,))
flow = FlowSimulation(
    mesh, physics; numerics, adaptivity,
    velocity_model = OneVelocityModel(),
)
simulate!(flow, 1e-4; dt = 1e-4)
```

See [Public API](api.md) for the option matrix,
[Adaptive refinement](adaptive-refinement.md) for regridding, and
[Visualization](visualization.md) for live output.
