# Diagnostics

Diagnostics are simulation services, not visualization callbacks.
[`FlowSimulation`](@ref) evaluates them at completed states and stores typed
histories whether or not a viewer exists. A recorder can mirror those
simulation-owned samples without becoming their owner.

## Built-in quantities

Let ``V_f`` be the open staggered dual volume, ``c_f`` the staggered liquid
fraction, and ``\rho_l,\rho_g`` the phase densities.

The kinetic-energy and momentum face sums below run over the interior
staggered unknowns. Prescribed physical-boundary faces are not degrees of
freedom in the discrete momentum inner product and are therefore excluded.
Periodic faces are interior unknowns.

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
| `DivergenceNorm()` | volume-weighted cell L2 norm of the finite-volume incompressibility constraint |
| `LeafBlockCount()` | current number of leaf blocks |
| `CellCount()` | current number of active cells |

On embedded geometry, energy and momentum use cut dual volumes and the
divergence norm uses open cell volumes. For two velocities, `DivergenceNorm()`
measures the same volume flux as the pressure projection, with liquid and gas
face apertures ``a_l,a_g`` and intrinsic velocities ``u_l,u_g``:

```math
\|\nabla_h\cdot\mathbf{u}\|_{L^2}
= \left[\sum_c V_c^{\mathrm{open}}
  \left(\frac{1}{V_c^{\mathrm{open}}}
  \sum_{f\in\partial c} s_{cf}|f|
  (a_{l,f}u_{l,f}+a_{g,f}u_{g,f})\right)^2\right]^{1/2}.
```

The sum includes physical-boundary faces as well as interior faces. Phase
apertures are the projection's current physical face openings, including any
embedded-solid aperture; they are not density weights. Closed faces and absent
phases contribute zero. For a single velocity the diagnostic instead uses its
standard finite-volume divergence, with open apertures on embedded geometry.

`InterfaceMeasure()` sums physical PLIC segment lengths and also counts
pure-phase jumps that lie exactly on an open mesh face.

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

The report uses the latest pressure solver statistics. For two-velocity
projections, `CMDGFMProjectionStats` reports the dimensionless
`mixture_volume_defect = dt * max(abs(div(u_mix)))` and
`normal_velocity_defect = max(dt * abs(normal_jump) / h)`. The latter is not
a divergence: it measures the normal-jump row of the projected state. A
face row uses the smallest edge of either adjacent cell, which is the fine
cell's smallest edge at a coarse–fine face.

The CMDGFM jump-row algebraic residual has acceleration units, so its
dimensionless convergence defect is `max(dt^2 * abs(r_jump) / h)`. The
pressure-row volume residual is
`dt^2 * max(abs(r_pressure) / volume)`. These defects and the projection
tolerance are dimensionless; the raw algebraic residual norms below are
not.

The iterative systems are nondimensionalized as well. With `L` the largest
domain extent and `rho` the larger density, pressure and gradient-jump
unknowns use scales `rho * (L / dt)^2` and `L / dt^2`. This prevents the
Krylov residual norm, warm-start choice, and built-in preconditioner drop
thresholds from changing simply because the length, time, or mass units change.
The stopping tests are unit-free; floating-point roundoff can still change
the iteration count of an unpreconditioned Krylov solve near convergence.

For one-velocity and CMDGFM projections, a step smaller than the reference
step deliberately retains part of the incoming mixture-divergence defect.
The reported CMDGFM defects remain those of the actual projected state;
convergence is assessed against the retained mixture target, while the normal-jump
constraint still targets zero.

The internal functions `TwoPhaseAMR.converged(stats)`,
`TwoPhaseAMR.iterations(stats)`, and `TwoPhaseAMR.final_residual(stats)` give
the common convergence status, iteration count, and final residual for linear
and CMDGFM statistics. They are not exported. The residual uses the
linear solver's infinity norm and the two-velocity solvers' algebraic system
residual 2-norm, respectively, so its magnitude is not directly comparable
between solver types; the reported physical defects have their own meanings.

The CMDGFM projection caches topology- and solid-geometry-dependent pressure
data across solves. Moving-interface coefficients and active jump faces are
updated each solve, while mesh adaptation or a change of solid-geometry
instance or revision invalidates the affected cached structures. The embedded
cell mask is also refreshed if a reused workspace sees a different interface
fraction floor.

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
    flow.velocity, flow.pressure, flow.system, dt, flow.workspace;
    time = flow.time,
)
flow.time += dt
flow.steps += 1
record_diagnostics!(flow)
```

## Metadata and recording

Every diagnostic carries a [`DiagnosticInfo`](@ref): a label, optional display
units, its [`PhysicalDimension`](@ref) (two-dimensional quantities per unit
depth, for example `M L T⁻²` for the energies), component names, a kind
(`:physical`, `:count`, `:solver`, or `:defect`), and a preferred scale.
[`diagnostic_info`](@ref) returns it; custom diagnostics take the same
metadata as keywords:

```julia
add_diagnostic!(flow, :amplitude, state -> amplitude(state);
    label = "Amplitude", dimension = PhysicalDimension(0, 1, 0))
```

[`diagnostic_groups`](@ref) groups diagnostics that share kind, dimension, and
scale. When the dimension is unknown, diagnostics share a group only if they
have the same nonempty units; otherwise each forms its own group. Known
dimensions determine grouping rather than the unit string, so use consistent
units for quantities with the same dimension.

A [`Recorder`](@ref) given `diagnostics = flow` mirrors the simulation-owned
samples at every step boundary; it never evaluates diagnostics itself. For a
flow configured with `LinearMomentum()`:

```julia
fields = VisualizationFieldRegistry()
register_default_fields!(fields, flow)
recorder = Recorder(fields; diagnostics = flow)
simulate!(flow, 2.0; observer = recorder)  # continue the flow from the earlier example
momentum = diagnostic_history(snapshot_store(recorder), :linear_momentum)
momentum.values[:, 1]   # x component; `momentum.components == ["x", "y"]`
```

The returned [`DiagnosticColumns`](@ref) is a detached copy with one column
per component. The simulation's own
`diagnostic_history(flow, :linear_momentum)` remains available without a
recorder; its sampling schedule is set by the diagnostic, not by keyframes
or previews. Studio and [`diagnosticsgrid!`](@ref) group the mirrored series
using the same metadata. See [Recording and visualization](visualization.md).
