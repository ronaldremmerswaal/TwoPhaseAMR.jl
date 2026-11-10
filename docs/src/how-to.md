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

## Record a run

The simulation remains the owner of evaluation and history. A recorder copies
field frames and mirrors the diagnostics without re-evaluating them:

```julia
fields = VisualizationFieldRegistry()
register_default_fields!(fields, flow)
recorder = Recorder(fields; diagnostics=flow, keyframes=EquispacedTimes(21))
simulate!(flow, final_time; observer=recorder)
store = snapshot_store(recorder)
```

The same `diagnostic_history(flow, :kinetic_energy)` is available without a
recorder. To inspect a live run, use the Studio environment described in
[Recording and visualization](visualization.md#studio). For a fresh flow,
create its fields and recorder as above, then start Studio **before** advancing:

```julia
using TwoPhaseAMRStudio
app = studio(recorder; open=:none)
url(app)  # open this secret URL in a browser
simulate!(flow, final_time; observer=recorder)
# After inspecting the recorded frames:
close(app)
```

For headless static output, load CairoMakie and call
`export_figure("vorticity.png", store; field=:velocity_vorticity)` or
`export_animation("vorticity.gif", store; field=:velocity_vorticity)`.
See [Static and export](visualization.md#static-and-export) for setup and
options.

## Run and modify an example

Every example is a standalone file exposing `run_example`. Start headlessly
from the REPL:

```julia
include("examples/06_capillary_wave/capillary_wave.jl")
run_example(max_level=1, final_time=0.01)
```

The example's environment also includes Studio. To view the capillary-wave
example live from a terminal, instantiate that environment and pass
`--studio`:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples -t 2,1 examples/06_capillary_wave/capillary_wave.jl --studio
```

In the REPL, `using TwoPhaseAMRStudio` followed by
`run_example(max_level=1, final_time=0.01, start_studio=studio)` starts the
server before the solve; close `run.app` when finished if the result was
assigned to `run`. See the [example catalogue](examples.md) for which
examples support live Studio.

Physics parameters and mesh resolution are named arguments; additional
keywords are forwarded to `FlowSimulation`, including `velocity_model` and
`numerics`. See the [example catalogue](examples.md) for model choices and
expected cost.
