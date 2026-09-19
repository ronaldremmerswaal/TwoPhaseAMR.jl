# Diagnostics

Diagnostics are simulation services, not visualization callbacks.
[`FlowSimulation`](@ref) evaluates them at completed states and stores typed
histories whether or not a viewer exists. A visualization may subscribe to a
simulation-owned diagnostic without becoming its owner.

## Built-in quantities

Let ``V_f`` be the open staggered dual volume, ``c_f`` the staggered liquid
fraction, and ``\rho_l,\rho_g`` the phase densities.

| Diagnostic | Stored value |
| --- | --- |
| `KineticEnergy()` | scalar ``\frac12\sum_f \rho_f u_f^2 V_f``; for two velocities, ``\frac12\sum_f[\rho_lc_fu_{l,f}^2+\rho_g(1-c_f)u_{g,f}^2]V_f`` |
| `LinearMomentum()` | two-component static vector, integrated on faces of each coordinate direction |
| `PhaseVolume()` | `[liquid, gas]`; embedded-solid volume is excluded |
| `DivergenceNorm()` | volume-weighted cell L2 norm of the configured velocity constraint |
| `LeafBlockCount()` | current number of leaf blocks |
| `CellCount()` | current number of active cells |

On embedded geometry, energy and momentum use cut dual volumes. The
two-velocity divergence diagnostic first forms the phase-mass-weighted mixture
velocity used by the coupled formulation.

## Headless collection

Pass diagnostics while constructing the simulation:

```julia
flow = FlowSimulation(
    mesh, physics;
    diagnostics = (
        KineticEnergy(),
        LinearMomentum() => EveryNSteps(5),
        PhaseVolume() => EveryTimeInterval(0.1),
    ),
)
simulate!(flow, 1.0)

energy = diagnostic_history(flow, :kinetic_energy)
energy.times
energy.steps
energy.values
```

The initial state is always stored when a diagnostic is registered. A final
state is stored if its time/step pair was not already sampled. Histories are
detached copies, so reading them cannot mutate a running simulation.

For a compact end-of-run report covering the model, mesh, pressure solve,
AMR activity, and all diagnostic histories, pass the simulation and result to:

```julia
result = simulate!(flow, 1.0)
print_simulation_summary(flow, result)
```

An `IO` may be supplied first when the report should be captured in a file or
buffer.

## Custom diagnostics

A custom reduction receives one [`SimulationState`](@ref):

```julia
add_diagnostic!(
    flow, :pressure_integral,
    state -> integrate(state.pressure);
    schedule = EveryNSteps(10),
)
```

The state exposes `velocity`, `pressure`, `solver`, `interface`, `geometry`,
`time`, and `step`. The first evaluation fixes the history's value type. A
later value of a different type is rejected rather than widening the history
silently.

Manual stepping loops call:

```julia
stats = step!(
    flow.velocity, flow.pressure, flow.solver, dt, flow.workspace;
    time = flow.time,
)
flow.time += dt
flow.steps += 1
record_diagnostics!(flow)
```

## Live visualization

Register the simulation and diagnostic name on a visualization registry:

```julia
fields = VisualizationFieldRegistry()
register_default_fields!(fields, flow)
register_diagnostic!(
    fields, flow, :kinetic_energy;
    label = "Kinetic energy", units = "J",
)

viewer = visualize(fields)
session = VisualizationSession(fields; viewer)
simulate!(flow, 1.0; visualization = session)
```

The adapter reads the manager-owned value. If the visualization cadence asks
for a state outside the diagnostic's normal schedule, the manager records
that state once; it is not independently recomputed by the viewer.

Vector diagnostics need a component for scalar plotting:

```julia
register_diagnostic!(
    fields, flow, :linear_momentum;
    name = :momentum_x, component = 1,
)
```

The legacy `register_diagnostic!(registry, name, five_argument_reduction)`
method remains available for viewer-local reductions. New application code
should prefer simulation-owned diagnostics whenever the quantity is useful
without graphics.
