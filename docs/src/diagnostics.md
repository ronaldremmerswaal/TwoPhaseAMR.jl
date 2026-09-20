# Diagnostics

Diagnostics are simulation services, not visualization callbacks.
[`FlowSimulation`](@ref) evaluates them at completed states and stores typed
histories whether or not a viewer exists. A visualization may subscribe to a
simulation-owned diagnostic without becoming its owner.

## Built-in quantities

Let ``V_f`` be the open staggered dual volume, ``c_f`` the staggered liquid
fraction, and ``\rho_l,\rho_g`` the phase densities.

The face sums below run over the interior staggered unknowns. Prescribed
physical-boundary faces are not degrees of freedom in the discrete momentum
inner product and are therefore excluded. Periodic faces are interior
unknowns.

| Diagnostic | Stored value |
| --- | --- |
| `KineticEnergy()` | scalar ``\frac12\sum_f \rho_f u_f^2 V_f``; for two velocities, ``\frac12\sum_f[\rho_lc_fu_{l,f}^2+\rho_g(1-c_f)u_{g,f}^2]V_f`` |
| `PotentialEnergy()` | gravitational potential energy ``-\int_\Omega \rho\,\mathbf{g}\mathbin{\cdot}\mathbf{x}\,\mathrm{d}V`` |
| `SurfaceEnergy()` | surface energy ``\sigma |\Gamma|`` |
| `TotalEnergy()` | sum of kinetic, gravitational-potential, and surface energies |
| `LinearMomentum()` | two-component static vector, integrated on faces of each coordinate direction |
| `PhaseVolume()` | `[liquid, gas]`; embedded-solid volume is excluded |
| `PhaseMass()` | `[liquid mass, gas mass]`, using the configured phase densities |
| `InterfaceMeasure()` | physical PLIC interface length (the interface measure in two dimensions) |
| `MaximumVelocity()` | maximum absolute staggered velocity component, across both fields for a two-velocity model |
| `CFLTimeStep(cfl=0.45)` | current convective timestep bound from the configured model and workspace |
| `DivergenceNorm()` | volume-weighted cell L2 norm of the configured velocity constraint |
| `LeafBlockCount()` | current number of leaf blocks |
| `CellCount()` | current number of active cells |

On embedded geometry, energy and momentum use cut dual volumes and the
divergence norm uses open cell volumes. The two-velocity divergence diagnostic
first forms the phase-mass-weighted mixture velocity used by the coupled
formulation. `InterfaceMeasure()` sums physical PLIC segment lengths and also
counts pure-phase jumps that lie exactly on an open mesh face.

`PotentialEnergy()` integrates the first moments of the reconstructed liquid
and gas polygons, including cut cells, and takes the coordinate origin as the
zero of gravitational potential. This arbitrary zero does not affect energy
changes. `SurfaceEnergy()` uses the same interface measure as
`InterfaceMeasure()`. Both quantities are zero for a single-phase simulation,
whose current physics model has neither gravity nor a phase interface; in that
case `TotalEnergy()` equals `KineticEnergy()`.

## Headless collection

Pass diagnostics while constructing the simulation:

```julia
flow = FlowSimulation(
    mesh, physics;
    diagnostics = (
        KineticEnergy(), TotalEnergy(),
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

The state exposes `velocity`, `pressure`, `solver`, `workspace`, `interface`,
`geometry`, `time`, and `step`. The first evaluation fixes the history's value
type. A later value of a different type is rejected rather than widening the
history silently.

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
