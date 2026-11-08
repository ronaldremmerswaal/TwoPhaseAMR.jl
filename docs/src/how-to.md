# How-to guides

These recipes assume that a mesh and physical model have already been
constructed. The [Getting started](getting-started.md) tutorial develops a
complete simulation first.

## Choose numerical methods

Keep physical properties in `SinglePhase` or `TwoPhase` and group numerical
policy in [`NavierStokesOptions`](@ref):

```julia
numerics = NavierStokesOptions(
    advection=AdvectionOptions(scheme=MonotonizedCentral()),
    pressure=LinearSolverOptions(preconditioner=:tree_geometric),
    diffusion=LinearSolverOptions(preconditioner=:tree_geometric),
    multigrid=MultigridOptions(
        coarse_size=32, pre_sweeps=2, post_sweeps=2,
    ),
)
flow = FlowSimulation(mesh, physics; numerics)
```

The pressure and diffusion linear solvers are independent. The defaults pick
the Krylov method from the assembled operator and use tree-geometric
preconditioning.

## Add refinement criteria

Refinement criteria belong to the mesh/adaptivity layer; they are not physical
phase indicators:

```julia
criterion = InterfaceRefinementIndicator(interface; level=2)
adaptivity = Adaptivity(criteria=(criterion,), fields=(tracer,))
flow = FlowSimulation(mesh, physics; adaptivity)
```

Set update cadence and coarsening hysteresis with
[`MeshAdaptationOptions`](@ref) when constructing the mesh.

## Add a custom diagnostic

A reduction receives one [`SimulationState`](@ref). Its first result fixes the
history value type:

```julia
add_diagnostic!(
    flow, :mean_pressure,
    state -> integrate(state.pressure) / prod(state.solver.mesh.extent);
    schedule=EveryNSteps(5),
)
result = simulate!(flow, final_time)
history = diagnostic_history(flow, :mean_pressure)
```

Built-ins such as [`KineticEnergy`](@ref), [`TotalEnergy`](@ref),
[`PhaseMass`](@ref), and
[`CFLTimeStep`](@ref) can instead be passed in `diagnostics=(...)` at
construction.

## Combine diagnostics with live visualization

The simulation remains the owner of evaluation and history. The viewer only
subscribes to a named diagnostic:

```julia
fields = VisualizationFieldRegistry()
register_default_fields!(fields, flow)
register_diagnostic!(fields, flow, :kinetic_energy)
viewer = visualize(fields)
session = VisualizationSession(fields; viewer)
simulate!(flow, final_time; visualization=session)
```

The same `diagnostic_history(flow, :kinetic_energy)` is available when no
viewer is installed or when the viewer is closed.

## Run and modify an example

Every example is a standalone file exposing `run_example`. Start headlessly
from the REPL:

```julia
include("examples/capillary_wave.jl")
run_example(max_level=1, final_time=0.01)
```

Direct terminal execution enables visualization. Physics parameters and mesh
resolution are named arguments; additional keywords are forwarded to
`FlowSimulation`, including `velocity_model` and `numerics`. See the
[example catalogue](examples.md) for model choices and expected cost.
