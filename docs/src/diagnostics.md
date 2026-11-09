# Diagnostics

Diagnostics are simulation services, not visualization callbacks.
[`FlowSimulation`](@ref) evaluates them at completed states and stores typed
histories whether or not a viewer exists. A recorder can mirror those
simulation-owned samples without becoming their owner.

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

The report uses the latest pressure solver statistics. For two-velocity
projections, `CMDGFMProjectionStats` and `MimeticProjectionStats` report the
achieved `mixture_volume_defect` and `normal_velocity_defect`; the mimetic
statistics also report the tangential defect, the independent constraint-row
count, and the momentum and energy identity errors. A normal velocity defect
is not a divergence or a volume defect, and is reported only once.

The internal functions `TwoPhaseAMR.converged(stats)`,
`TwoPhaseAMR.iterations(stats)`, and `TwoPhaseAMR.final_residual(stats)` give
the common convergence status, iteration count, and final residual for linear,
CMDGFM, and mimetic statistics. They are not exported. The residual uses the
linear solver's infinity norm and the two-velocity solvers' algebraic system
residual 2-norm, respectively, so its magnitude is not directly comparable
between solver types; the reported physical defects have their own meanings.

The mimetic block-Cholesky preconditioner applies sparse factors in place on
Julia 1.12 and later. Julia 1.11's CHOLMOD interface lacks an in-place solve,
so the same factorization uses an allocating solve on that supported version.
The mimetic projection caches topology- and solid-geometry-dependent pressure
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
    flow.velocity, flow.pressure, flow.solver, dt, flow.workspace;
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
