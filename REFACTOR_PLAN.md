# TwoPhaseAMR refactor plan

## Purpose

This refactor will reorganize the package around its numerical domains, make
the one-velocity and two-velocity choices explicit, simplify diagnostics and
examples, and improve test and documentation coverage without changing the
validated discretizations.

The work should be delivered incrementally. A file move must not be combined
with a numerical-method change in the same step: first preserve behavior, then
improve interfaces and internals behind tests.

## Current-state findings

The plan is based on the current repository rather than only the desired
folder names:

- `src/` contains 31 Julia files and about 25,400 lines in one flat module.
  Several files exceed 1,000 lines; `multigrid.jl` exceeds 2,500 lines.
- `src/TwoPhaseAMR.jl` directly controls all include order and exports about
  185 names. Domain dependencies are therefore implicit.
- `IncompressibleNavierStokesSolver` currently accepts material properties,
  surface-tension choices, two-velocity choices, Krylov configuration, AMR
  fields, geometry, and transport schemes through one large keyword list.
- The phase model is inferred from the presence of `interface`, while the
  velocity model is inferred later from whether the initial velocity is a
  `FaceVelocity` or a tuple. This makes invalid option combinations possible
  and makes examples harder to parameterize consistently.
- The related method selectors use unrelated keyword names:
  `surface_tension_formulation`, `interface_method`, and
  `two_velocity_projection`.
- Diagnostics belong to `VisualizationFieldRegistry`, accept a five-argument
  callback, and can only return a real scalar. This couples simulation data
  collection to visualization and forces examples to implement their own
  kinetic-energy, momentum, amplitude, and caching logic.
- The 12 example files contain about 3,100 lines. The physical setup,
  analytical reference solution, custom time loop, diagnostics, GUI setup,
  and command-line entry point are often combined in one file.
- Tests have good numerical breadth, but most unit tests are flat in `test/`,
  example files are included as test support, allocation tests are mixed into
  the default suite, and only the longest studies use `test/integration/`.
- `TODO_LATER.md` already has unrelated user changes. It is not part of this
  refactor and must remain untouched.

## Goals

1. Make each source file have one clear numerical or orchestration concern.
2. Make dependencies visible while retaining one top-level `TwoPhaseAMR`
   module and one user-facing namespace.
3. Give single-phase, one-velocity two-phase, and two-velocity two-phase flow
   explicit model objects and dispatch paths.
4. Represent algorithm choices with typed option objects and use consistent
   terminology.
5. Make common diagnostics one-line choices that work for every flow model.
6. Make every two-phase example selectable between one- and two-velocity
   models; the instability example must default to two velocity.
7. Mirror the source domains in the test layout and separate unit,
   integration, and performance responsibilities. Example smoke tests are
   unit tests; convergence, manufactured-solution, and literature-reference
   studies are integration tests.
8. Preserve numerical results, conservation properties, Float32 support,
   dynamic AMR behavior, optional extensions, and low-level expert APIs.
9. Use a bounded hierarchy: top-level domain folders may contain cohesive
   subsystem folders such as `mesh/adaptivity/` or `solvers/multigrid/`, but
   avoid directory layers that do not express an abstraction boundary.

## Non-goals

- Do not change the mesh representation, discretization, or published default
  numerical methods merely as a consequence of moving files.
- Do not introduce one Julia submodule per folder. That would make method
  extension and the existing low-level API substantially more complex without
  providing useful user-facing isolation.
- Do not rewrite the visualization backend, implement new physics, or solve
  the broader documentation/report items in `TODO_LATER.md` as part of this
  refactor.
- Do not remove low-level constructors. The high-level path should become
  smaller, while research-level operator access remains available.

## Terminology decision: physical indicators versus AMR criteria

`indicator` has two meanings in the current discussion. This plan uses the
following vocabulary consistently:

- `src/indicator/` owns physical characteristic/indicator functions and their
  geometry: the pointwise zero-or-one phase/solid indicators, their
  finite-volume cell and face fractions, PLIC reconstruction, height
  functions, level-set initialization, and VOF transport.
- AMR objects such as `InterfaceRefinementIndicator` belong in
  `src/mesh/adaptivity/`. New documentation and APIs should call them
  **refinement criteria** where compatibility permits. Existing exported type
  names can be retained or deprecated gradually.

This separation prevents phase geometry from appearing to be an AMR feature.

## Target source layout

Use at most two meaningful levels below a top-level domain. A directory must
contain multiple cohesive files; prospective future growth is not enough to
justify a singleton directory. A single file stays at the nearest meaningful
parent until an actual split creates a subsystem. Apply the same rule to
`test/`. `src/TwoPhaseAMR.jl` remains the single, commented include manifest so
that staged dependencies are visible in one place.

```text
src/
  TwoPhaseAMR.jl
  threading.jl
  mesh/
    variables.jl
    transfer.jl
    topology/
      placements.jl
      mesh.jl
    adaptivity/
      configuration.jl
      indicators.jl
      phase_criteria.jl
      geometry_criteria.jl
      workspace.jl
      field_transfer.jl
      adapt.jl
  operators/
    gradient.jl
    staggered_diffusion.jl
    elliptic.jl
    advection/
      cell_advection.jl
      staggered_schemes.jl
      staggered_schedules.jl
      staggered_fluxes.jl
      staggered_update.jl
  solvers/
    linear_options.jl
    multigrid/
      options.jl
      hierarchy.jl
      geometric_preconditioner.jl
      cycles.jl
      elliptic_solver.jl
  navier_stokes/
    options.jl
    system.jl
    workspace.jl
    stages.jl
    projection.jl
    implicit_diffusion.jl
    independent_step.jl
    simulate.jl
    flow_simulation.jl
    flow_diagnostics.jl
  indicator/
    embedded_geometry.jl
    embedded_operators.jl
    plic_geometry.jl
    phase/
      interface_field.jl
      levelset_initialization.jl
      polygon_initialization.jl
      height_functions.jl
      scalars.jl
      vof_workspace.jl
      vof_fluxes.jl
      scalar_transfer.jl
  single_phase/
    physics.jl
    stepping.jl
  two_phase/
    physics.jl
    model.jl
    one_velocity.jl
    forces.jl
    stepping.jl
    phase_face_fields.jl
    momentum_workspace.jl
    phase_face_transport.jl
    phase_face_transfer.jl
    variable_diffusion_geometry.jl
    variable_diffusion_operator.jl
    variable_diffusion_preconditioner.jl
    variable_diffusion_solver.jl
    two_velocity/
      model.jl
      transport_methods.jl
      transport_workspace.jl
      interface_transport.jl
      transport_geometry.jl
      velocity_fields.jl
      momentum_workspace.jl
      flow_workspace.jl
      stepping.jl
      projection/
        methods.jl
        mimetic_projection.jl
        cmdgfm_projection.jl
  simulation/
    state.jl
    diagnostics/
      interface.jl
      schedules.jl
      history.jl
  visualization/
    types.jl
    registry.jl
    phase_geometry.jl
    sampling.jl
    interface.jl
    checkpoints.jl
    session_types.jl
    frame_capture.jl
    updates.jl
    controls.jl
```

This is the immediate layout. Further splits from the priority table below
must create at least two coherent files at once; until then, the larger file
stays at its parent rather than occupying a one-file placeholder directory.

The exact split of a large kernel may change during implementation, but the
ownership and subsystem boundaries should not. In particular:

- `mesh/topology/` owns the tree and its connectivity;
  `mesh/adaptivity/` owns topology changes and all refinement criteria.
- `indicator/` owns physical phase/solid indicator geometry.
- `operators/` owns discrete linear and nonlinear operators, including the
  assembled/matrix-free elliptic operators, but does not choose a solution
  strategy.
- `solvers/` means **linear solvers only**. Common options stay in
  `solvers/linear_options.jl` until Krylov integration and common contracts
  form a genuine multi-file subsystem; `solvers/multigrid/` owns the existing
  multi-file hierarchy and configuration.
- `navier_stokes/` owns the incompressible equation system, pressure
  projection, split time step, high-level flow simulation, and flow-specific
  diagnostics.
- `single_phase/` and `two_phase/` provide model-specific methods to the
  shared orchestration; intrinsic two-velocity methods are a cohesive
  `two_phase/two_velocity/` specialization.
- `simulation/` owns equation-system-independent observation state, sampling
  schedules, diagnostic interfaces, and histories; `visualization/` only
  renders or subscribes to those facilities. Flow-specific time integration
  remains in `navier_stokes/`.

### Dependency direction

The intended dependency graph is:

```text
core
  -> mesh/topology
       -> mesh/variables
            -> mesh/transfer
            -> operators -> solvers
            -> indicator
                 -> mesh/adaptivity (physical refinement criteria load late)

operators + solvers + indicator
  -> single_phase / two_phase -> two_phase/two_velocity
  -> navier_stokes

simulation diagnostics infrastructure
  -> navier_stokes
  -> visualization

navier_stokes -> visualization
```

There may be Julia method extensions in a later layer, but data ownership must
follow this direction. For example, a mesh refinement criterion may dispatch
on an `InterfaceField`, but its type and AMR policy still live in
`mesh/adaptivity/`.
The root include manifest should label such late extensions explicitly instead
of relying on an accidental file order.

### Mapping from current files

The initial move should be mechanical:

| Current file(s) | Target ownership |
| --- | --- |
| topological placements from `placements.jl` | `mesh/topology/placements.jl` |
| `AbstractMeshField` from `placements.jl`, plus `fields.jl` | `mesh/variables.jl` |
| topology from `mesh.jl` | `mesh/topology/` |
| `transfer.jl` | `mesh/transfer.jl` |
| `adaptivity.jl` and refinement types currently in `interface.jl` and `geometry.jl` | `mesh/adaptivity/` |
| `advection.jl`, `staggered_advection.jl` | `operators/advection/` |
| `gradient.jl` | `operators/gradient.jl` |
| `staggered_diffusion.jl` | `operators/staggered_diffusion.jl` |
| operator construction/application from `elliptic.jl` | `operators/elliptic.jl` |
| common linear-solver configuration | `solvers/linear_options.jl`; create `solvers/linear/` only with a real multi-file split |
| multigrid hierarchy, smoothers, transfers, and preconditioners from `multigrid.jl` | `solvers/multigrid/` |
| pressure projection from `multigrid.jl`, plus `implicit_diffusion.jl` | `navier_stokes/`; add a child only when projection/time integration is split into multiple files |
| `interface.jl`, `height_functions.jl`, and `levelset_initialization.jl` | `indicator/phase/` |
| `geometry.jl` | split embedded-field, operator-extension, and PLIC geometry files under `indicator/` |
| common `vof_advection.jl` code | split scalar, workspace, flux, and transfer files under `indicator/phase/` |
| `phase_staggered_advection.jl`, `two_phase_forces.jl`, `variable_staggered_diffusion.jl`, `two_phase_model.jl` | direct files under `two_phase/` until a child subsystem has multiple files |
| two-velocity methods from `vof_advection.jl` and `phase_staggered_advection.jl`, plus `two_velocity.jl` | `two_phase/two_velocity/` |
| `mimetic_projection.jl`, `cmdgfm_projection.jl` | `two_phase/two_velocity/projection/` |
| single-phase physics configuration and stepping | `single_phase/` |
| shared stage orchestration from `navier_stokes.jl` | `navier_stokes/` |
| model-specific branches of `two_phase_step.jl` | `two_phase/stepping.jl` and `two_phase/two_velocity/stepping.jl` |
| `flow_simulation.jl` | `navier_stokes/` |
| `visualization.jl`, `visualization_session.jl` | `visualization/` |

`ext/` remains flat. Its imports and qualified method extensions must be
updated after the moves, but extension behavior must not change.

### File-splitting priorities

Moving a large file intact into a better-named folder is not sufficient when
it still mixes construction, data types, kernels, orchestration, and reporting.
Split files along stable responsibilities, while keeping tightly coupled hot
kernels together when separation would obscure their invariants.

| Current file | Planned split |
| --- | --- |
| `multigrid.jl` (~2,566 lines) | hierarchy/coarsening, transfer operators, smoothers, scalar method, staggered method, preconditioner, and Navier--Stokes projection extraction |
| `variable_staggered_diffusion.jl` (~1,988 lines) | material coefficients, operator data, assembly/update, linear-solver setup, and cache/reuse logic |
| `staggered_advection.jl` (~1,780 lines) | schemes, graph schedules, flux construction, component update, and conservation helpers |
| `visualization.jl` (~1,450 lines) | field sources, registry, derived fields, geometry conversion, and backend-neutral display API |
| `visualization_session.jl` (~1,347 lines) | frames, checkpoints, sampling, diagnostic subscription, concurrency/mailbox, and session control |
| `navier_stokes.jl` (~1,264 lines) | system construction, configuration validation, workspace/regrid rebuild, staged stepping, and single-phase specialization |
| `adaptivity.jl` (~1,257 lines) | options/criteria, workspace/field registration, arbitration, regrid execution, and statistics |
| `geometry.jl` (~1,187 lines) | geometry field, cut-cell metrics, reconstruction, boundary enforcement, and late operator extensions |
| `vof_advection.jl` (~1,186 lines) | public types, workspace, directional fluxes, update, diagnostics, and AMR transfer |
| `cmdgfm_projection.jl` (~1,124 lines) | public types, jump stencils, assembly, block preconditioner, solve, and gradient reconstruction |
| `two_velocity.jl` (~951 lines) | velocity utilities, Frozen transport, Projected transport, momentum coupling, boundaries, and AMR hooks |
| `elliptic.jl` (~947 lines) | scalar/staggered operator construction and application versus linear-solver workspaces/solve policy |
| `mesh.jl` (~844 lines) | keys/blocks, connectivity/ownership, metrics/lookups, and public mesh construction |
| `interface.jl` (~801 lines) | phase field, PLIC reconstruction, height-function integration, and refinement criteria extraction |
| `mimetic_projection.jl` (~788 lines) | public types, assembly, solve, statistics, and phase-gradient reconstruction |

The line counts are diagnostic, not hard limits. A 700-line allocation-free
kernel with one invariant may be more maintainable than five mutually
dependent 140-line files. Conversely, a 300-line file that mixes public
configuration and an unrelated numerical kernel should still be split.

## Equation-system and solver naming

Use **equation systems** (or **PDE systems** in explanatory prose) for
top-level physical evolution models such as incompressible Navier--Stokes and
a possible future Maxwell implementation. Do not call this collection
"solvers": that word is reserved for algorithms that solve linear algebraic
systems.

The bounded hierarchy does not need an `equations/` parent. Each equation
system receives a sibling folder:

```text
src/navier_stokes/   # current incompressible-flow system
src/maxwell/         # possible future electromagnetic system
```

This scales without producing `src/equations/navier_stokes/...` nesting. The
manual may group both folders under an "Equation systems" heading even though
the filesystem does not add that intermediate directory.

Apply the same distinction to public names:

- `LinearSolver`, `EllipticSolver`, `solve!`, and solver statistics refer to
  linear systems;
- `NavierStokesSystem` describes the configured semi-discrete PDE;
- `FlowSimulation` owns its evolving fields, workspace, time, and histories;
- `step!` and `simulate!` advance an equation system.

`IncompressibleNavierStokesSolver`/`NavierStokesSolver` should remain as
compatibility aliases or translating constructors for the agreed deprecation
period, rather than establishing "solver" as the new architecture's name for
a PDE system. If a future Maxwell implementation is added, its parallel names
would be `MaxwellSystem` and, if needed, `MaxwellSimulation`.

## Public API redesign

### 1. Select physics and velocity model explicitly

The model must no longer be inferred from the shape of `velocity`. The target
high-level API is conceptually:

```julia
physics = TwoPhase(
    interface;
    density=(1000.0, 1.0),
    dynamic_viscosity=(1e-3, 1.8e-5),
    surface_tension=SurfaceTension(0.072; method=GFM()),
    gravity=(0.0, -9.81),
)

flow = FlowSimulation(
    mesh, physics;
    velocity_model=OneVelocityModel(),
    velocity=initial_velocity,
    boundary=FreeSlip(),
    dt=1e-4,
)
```

The corresponding two-velocity selection is:

```julia
flow = FlowSimulation(
    mesh, physics;
    velocity_model=TwoVelocityModel(
        interface_transport=FrozenInterfaceFluxes(),
        pressure_projection=CMDGFMProjection(),
    ),
    velocity=(liquid_velocity, gas_velocity),
    dt=1e-4,
)
```

For single-phase flow:

```julia
physics = SinglePhase(density=1.0, dynamic_viscosity=0.01)
flow = FlowSimulation(mesh, physics; velocity, boundary=NoSlip(), dt=0.002)
```

Names should be finalized in a short API decision record before coding, but
the separation itself is required:

- a physics object owns materials and physical forces;
- a velocity-model object owns one- versus two-velocity algorithms;
- numerical options own discretization and linear-solver choices;
- a simulation owns evolving fields, time, workspaces, AMR, and diagnostics.

This resolves the existing ambiguity between a constant velocity tuple and a
liquid/gas tuple. In two-velocity mode, a single initializer should be accepted
as shorthand and copied to both phases; a pair remains available for a genuine
velocity jump.

### 2. Use consistent typed choices

Introduce narrow abstract interfaces:

- `AbstractSurfaceTensionMethod`: `CSF()`, `GFM()`;
- `AbstractInterfaceTransportMethod`:
  `FrozenInterfaceFluxes()`, `ProjectedLiquidVelocity()`;
- `AbstractTwoVelocityProjection`:
  `CMDGFMProjection()`, `MimeticProjection()`;
- existing `AbstractStaggeredAdvectionScheme` implementations.

Every selector is passed as a typed value under a descriptive noun:

```julia
SurfaceTension(sigma; method=GFM())
TwoVelocityModel(
    interface_transport=FrozenInterfaceFluxes(),
    pressure_projection=CMDGFMProjection(),
)
AdvectionOptions(
    scheme=LW(),
    amr_interface_scheme=default_amr_interface_scheme(LW()),
)
```

Use `method` only inside a local option object; do not mix top-level keywords
named `formulation`, `interface_method`, and `two_velocity_projection`.

The public terminology should also be settled: `MimeticProjection()` is the
algorithm selector, while `mimetic_gradient!` is the reconstruction operation.
If `MimeticGradient` appears in external usage or thesis terminology, provide
one documented compatibility alias rather than two competing selector names.

### 3. Group generic Navier--Stokes options

Replace the long solver keyword surface with immutable configuration values:

```julia
numerics = NavierStokesOptions(
    advection=AdvectionOptions(scheme=LW()),
    pressure=LinearSolverOptions(
        krylov=nothing,
        preconditioner=:tree_geometric,
        tolerance=nothing,
    ),
    diffusion=LinearSolverOptions(
        krylov=nothing,
        preconditioner=:tree_geometric,
    ),
    multigrid=MultigridOptions(
        coarse_size=64,
        pre_sweeps=2,
        post_sweeps=2,
        max_levels=32,
    ),
)
```

`Adaptivity` should similarly group runtime refinement criteria and transferred
fields, while cadence remains in `MeshAdaptationOptions`:

```julia
adaptivity = Adaptivity(
    criteria=(interface_criterion, vorticity_criterion),
    fields=(passive_scalar,),
)
```

Each configuration constructor validates only its own domain. Model-specific
options must not be accepted and silently ignored by another model. Defaults
are defined in one place and can be printed clearly with `show`.

Avoid a deeply nested parametric solver type that increases compile time.
Configuration types should be concrete and type-stable, with function
barriers around workspace construction and stepping.

### 4. Compatibility policy

Keep the existing keyword constructor for one release cycle as a translating
wrapper:

```julia
FlowSimulation(mesh; interface, density, dynamic_viscosity,
               surface_tension_formulation, interface_method,
               two_velocity_projection, ...)
```

It should:

1. construct the new physics/model/options objects;
2. emit one actionable deprecation per obsolete keyword family;
3. preserve current defaults and numerical behavior;
4. reject contradictory old/new configuration in the same call.

The current low-level `NavierStokesSolver` name remains available through the
compatibility period. The replacement system, operator, and workspace APIs
remain public, but their docstrings should point standard users to
`FlowSimulation`.

## Diagnostics redesign

Diagnostics should be independent of visualization and operate on a stable
one-argument state view:

```julia
flow = FlowSimulation(
    mesh, physics;
    diagnostics=(
        KineticEnergy(),
        LinearMomentum(),
        PhaseVolume(),
    ),
)

add_diagnostic!(
    flow,
    Diagnostic(:wave_amplitude, state -> wave_amplitude(state.interface));
    schedule=EveryStep(),
)

simulate!(flow, final_time)
energy = diagnostic_history(flow, :kinetic_energy)
```

The same definitions are available to a live visualization without moving
their ownership back into the visualization layer:

```julia
viewer = visualize(
    flow;
    fields=(:volume_fraction, :velocity_magnitude),
    diagnostics=(:kinetic_energy, :momentum_x),
)

simulate!(flow, final_time; visualization=viewer)
```

The intended data flow is:

```text
completed simulation state
  -> diagnostics engine evaluates each due diagnostic once
  -> simulation-owned typed history
       -> headless user access
       -> file/checkpoint output
       -> optional live-visualization subscriber
```

Closing or omitting the viewer must not disable collection. Attaching a viewer
should expose the history already collected and then receive new samples. The
viewer may downsample for rendering, but it must not independently re-evaluate
the diagnostic or change its scientific sampling schedule.

Required design points:

- `SimulationState` provides named access to velocity, pressure, interface,
  geometry, solver, time, and step. Custom callbacks no longer receive five
  positional arguments.
- A diagnostic owns its name, label, units, evaluator, and result type.
- Histories support real, complex, and fixed-size vector results. Momentum
  should not be forced into a scalar-only dictionary.
- Sampling policies (`EveryStep`, `EveryNSteps`, and `EveryTimeInterval`) are
  shared by headless runs and visualization.
- Repeated calls to `simulate!` append to the simulation-owned history without
  duplicating the boundary sample.
- `VisualizationSession` subscribes to selected diagnostic series, can render
  them as live plots next to fields, and does not own or re-evaluate their
  definitions.
- The diagnostic core has no Makie dependency. Notifications carry immutable
  samples or copied values across asynchronous visualization boundaries, so a
  slow GUI cannot race with or stall numerical field mutation.
- A thin compatibility wrapper maps `register_diagnostic!` to the new system
  during migration.

Built-in diagnostics should include at least:

- kinetic energy;
- linear momentum, with named x/y components for plotting;
- phase volume and phase mass;
- divergence norms;
- interface measure/length;
- leaf-block and cell counts;
- maximum velocity/CFL information.

The discrete definition of energy and momentum must be documented and tested
for single phase, one-velocity two phase, two-velocity two phase, AMR, and
embedded geometry. Two-velocity energy and momentum must use each phase's
intrinsic velocity and phase mass; they must not silently use the mixture
velocity approximation.

## Example redesign

### Target layout

Keep examples flat while each executable remains one self-contained file.
Do not create `transport/` or `mesh/` for a single example:

```text
examples/
  README.md
  taylor_green.jl
  lid_driven_cavity.jl
  cylinder_flow.jl
  decaying_turbulence.jl
  capillary_wave.jl
  sessile_droplet.jl
  sloped_sessile_droplet.jl
  stokes3_wave.jl
  lgpi.jl
  interfacial_instability.jl
  vortex_reverse.jl
```

Examples do not share runner plumbing: mesh, physics, diagnostics,
`FlowSimulation`, and `simulate!` remain visible in every file.

### Example contract

Every executable example should have the same small shape:

1. define physical parameters and initial conditions;
2. construct mesh, physics, velocity model, and simulation;
3. attach built-in diagnostics by name/type;
4. call `simulate!` once;
5. print the standard detailed simulation summary when executed directly.

Aim for roughly 50--120 lines per introductory script. Move lengthy analytical
dispersion relations, convergence loops, and published-reference comparisons
to the corresponding `test/integration/` subsystem or to documentation pages.
Do not make integration tests import example files as libraries.

Each Navier--Stokes example exposes one entry point whose named arguments are
the nondimensional physical parameters, `max_level`, and `final_time`:

```julia
run_example(; physical_parameters..., max_level, final_time=nothing,
            visualize=false, kwargs...)
```

Direct terminal execution passes `visualize=true`; included examples remain
headless by default. The remaining keywords are forwarded to `FlowSimulation`, including
`velocity_model`. `simulate!` selects the time step automatically. Every
two-phase example supports `TwoVelocityModel()` without editing the file.
Shared continuous initial velocity is duplicated automatically in two-velocity
mode. Where a case has a physical velocity jump, the example supplies
phase-specific initial data; its one-velocity mode documents the chosen
mixture/mean projection and that it represents a different closure.

Defaults:

| Example | Default | Also required |
| --- | --- | --- |
| capillary wave | one velocity | two velocity |
| sessile droplet | one velocity | two velocity |
| sloped sessile droplet | one velocity | two velocity |
| Stokes-3 wave | one velocity | two velocity |
| LGPI | one velocity | two velocity |
| vortex reversal/phase transport | prescribed carrier; not a Navier--Stokes model | not applicable |
| KH/RT instability | **two velocity** | one velocity |

The instability filename and public entry points should no longer embed
`two_velocity` in their names once both models are supported. The default
configuration remains explicitly `TwoVelocityModel`, with
`FrozenInterfaceFluxes()` and `CMDGFMProjection()` unless validation motivates
a separately reviewed default change.

`examples/README.md` provides a two-column command table with expected run
cost and model default. Full derivations and representative validation results
belong in the documentation, not in the executable script.

## Test reorganization

The tests should mirror meaningful source subsystems. A test directory need
not be created until it contains more than one cohesive test file, so the test
tree may be slightly shallower than `src/`:

```text
test/
  runtests.jl
  diagnostics.jl
  solvers.jl
  threading.jl
  mesh/
    topology.jl
    adaptivity.jl
    variables.jl
  operators/
    advection/
    gradient.jl
    staggered_diffusion.jl
    poisson.jl
  indicator/
    phase/
    geometry.jl
  two_phase/
    transport.jl
    forces.jl
    variable_staggered_diffusion.jl
    two_velocity/
      diffusion.jl
      transport/
      projection/
  navier_stokes/
  visualization/
    fields.jl
    session.jl
  examples/
  integration/
  performance/
```

For example, AMR criterion and regridding tests currently share
`test/mesh/adaptivity.jl`; that becomes a child directory only when the suite
is split into multiple cohesive files. Cross-domain behavior stays in
`test/integration/` rather than being forced into a source-mirroring folder.

### Suite responsibilities

- Domain folders contain fast deterministic unit and component tests.
- `test/solvers/` tests only linear-solver behavior; equation-level projection,
  stepping, and flow configuration belong in `test/navier_stokes/`.
- `test/integration/` contains end-to-end flows, AMR/regrid tests,
  manufactured solutions, convergence studies, literature-reference checks,
  and their tolerances. One `runtests.jl` entry point runs the complete suite.
- `test/examples/` runs every example at a tiny resolution and final time,
  without opening a GUI.
- `test/performance/` contains allocation and setup/compile-time regression
  checks. Exploratory timing and scaling remain in `benchmark/`.

`test/runtests.jl` should include named suites through one small helper and
must not depend on fragile filename order. CI can run fast tests on all
supported Julia versions and sharded integration jobs. The complete integration
runner remains the single local entry point for running every integration test.

### Required new tests

1. **Configuration contract**
   - valid construction for every model;
   - informative errors for incompatible method/model combinations;
   - old keywords translate to the same new configuration;
   - defaults are tested once rather than repeated across examples.
2. **Method-selection matrix**
   - one-velocity with CSF and GFM;
   - two-velocity with Frozen and Projected interface transport;
   - two-velocity with CMDGFM and Mimetic projection;
   - representative combinations take at least one successful step;
   - invalid combinations fail at construction, not deep in `step!`.
3. **Diagnostics**
   - analytic uniform-flow energy and momentum;
   - one- and two-velocity equivalence when both phase velocities match;
   - correct phase-weighted result when they differ;
   - embedded-solid volume weighting;
   - AMR invariance for constant states;
   - custom real, complex, and vector diagnostics;
   - schedules and resumed simulation;
   - identical histories with no viewer, a synchronous viewer, and an
     asynchronous viewer;
   - live subscription receives existing and future samples without duplicate
     evaluation;
   - detaching/closing a viewer does not stop diagnostic collection.
4. **Examples**
   - every two-phase example smoke-tested in both velocity modes;
   - instability default asserted to be two velocity;
   - no example test requires Makie or a display;
   - each example's public function returns a consistent result object.
5. **Structural/API checks**
   - package loads with each optional extension absent and present;
   - Documenter `checkdocs=:exports` remains clean;
   - exported aliases and deprecations are exercised;
   - Float32 construction and regridding remain covered.
6. **Numerical regression**
   - conservation of total phase volume and momentum;
   - projection divergence and jump constraints;
   - hydrostatic/capillary balance;
   - dynamic-regrid transfer invariants;
   - existing convergence orders and reference values retained.

Tests should prefer public APIs. Tests of a genuinely internal numerical
invariant may use qualified internal names, but those files should state why.
Common mesh/field factories belong in `test_utils.jl`; case-specific helpers
stay beside their tests.

## README and documentation workstream

The documentation rewrite proceeds with the code refactor, not after it. A
change that moves an abstraction, renames a public concept, or introduces a
new API is incomplete until its navigation, reference, examples, and migration
guidance are updated in the same change.

### README

Rewrite `README.md` as a concise landing page rather than a second manual:

1. one-paragraph purpose and scope;
2. installation and supported Julia version;
3. a 20--30 line high-level quick start;
4. a compact capability/model table;
5. one-velocity versus two-velocity links and defaults;
6. one diagnostics example that works headlessly and in a live viewer;
7. a small result/visualization gallery;
8. links to tutorials, examples, API reference, numerical methods, and
   contribution/developer material.

Long operator explanations, implementation history, and complete example
catalogues move out of the README. The README example must be executed in CI
so that the first user experience cannot silently become stale.

### Manual information architecture

Rebuild the manual around user tasks and concepts, while retaining the useful
existing numerical-method material:

```text
docs/src/
  index.md
  getting_started/
    installation.md
    quick_start.md
    choosing_a_model.md
  tutorials/
    mesh_and_fields.md
    single_phase.md
    two_phase_one_velocity.md
    two_phase_two_velocity.md
    embedded_geometry.md
    adaptivity.md
    diagnostics.md
    visualization.md
  how_to/
    choose_numerical_methods.md
    configure_linear_solvers.md
    add_refinement_criteria.md
    add_custom_diagnostics.md
    combine_diagnostics_and_live_visualization.md
    run_and_extend_examples.md
  explanation/
    indicator_geometry.md
    advection.md
    diffusion.md
    pressure_projection.md
    two_velocity.md
    adaptivity.md
    parallelism.md
  reference/
    high_level_api.md
    equation_systems.md
    mesh.md
    variables.md
    operators.md
    linear_solvers.md
    diagnostics.md
    visualization.md
  developer/
    architecture.md
    source_ownership.md
    testing.md
    adding_an_equation_system.md
  migration/
    refactor_api.md
```

The exact filenames may be consolidated where pages are short, but the four
documentation modes remain distinct:

- tutorials teach a complete workflow;
- how-to guides solve one concrete task;
- explanation pages describe numerical and architectural reasoning;
- reference pages state the API precisely.

### Documentation requirements

- Use one glossary consistently: physical indicator, refinement criterion,
  operator, linear solver, equation system, velocity model, simulation, and
  diagnostic.
- Add a high-level API page organized around physics, velocity model,
  numerics, equation system, simulation, diagnostics, and results.
- Document the complete CSF/GFM, Frozen/Projected, and CMDGFM/Mimetic option
  matrix, including defaults and invalid combinations, in one canonical table.
- Give every public type and public function a docstring; include a minimal
  runnable example when it clarifies usage.
- Document discrete energy, momentum, phase-volume, and divergence diagnostic
  definitions for all velocity models and embedded geometry.
- Show both headless diagnostic collection and subscription by live
  visualization, making clear that visualization does not own evaluation.
- Add a developer architecture page containing the dependency graph, folder
  ownership, extension contracts, and root include order.
- Keep derivations and numerical-method detail in explanation pages and link
  to them from tutorials/reference rather than duplicating them.
- Give each example a short companion page or catalogue entry stating its
  physics, default velocity model, alternate model invocation, expected cost,
  diagnostics, and representative result.
- Provide an old-to-new constructor and keyword migration table, including the
  `NavierStokesSolver` to `NavierStokesSystem` terminology transition.
- Check internal links, doctests, exported docstrings, and all executable code
  blocks in CI.

### Documentation deliverable in every phase

| Refactor phase | Documentation delivered with it |
| --- | --- |
| Baseline/design | content audit, glossary, target navigation, API decision record |
| File moves | developer ownership map, updated source links, no broken references |
| File splits | updated architecture/invariant notes and internal docstrings |
| Typed configuration | rewritten high-level reference, option matrix, migration guide, affected tutorials |
| Diagnostics | headless tutorial, live-visualization how-to, mathematical definitions, API reference |
| Examples | concise scripts, companion pages/catalogue, gallery, executable README quick start |
| Stabilization | docstring/export audit, link check, doctests, removal of obsolete terminology |

## Implementation sequence

### Phase 0: baseline and decisions

- Record the currently passing/failing unit, integration, performance,
  documentation, and formatting jobs. The current TODO says CI has failures,
  so do not treat a
  red baseline as a refactor regression without classifying it.
- Capture representative numerical outputs and conservation errors for one
  case per model.
- Record package load time, first simulation construction time, first-step
  time, steady-state step time, and allocations.
- Write a short API decision record finalizing public type names and the
  deprecation window.
- Audit every README/manual/example page against the current API, define the
  glossary, and approve the target documentation navigation.

Gate: baseline artifacts, accepted API sketch, documentation content audit,
and target navigation exist; no source behavior has changed.

### Phase 1: mechanical source and test moves

- Create the domain/subsystem folders and rewrite the root include manifest in
  labeled dependency sections.
- Move files without renaming public objects or altering algorithms.
- Move tests to their target domain folders and update includes/CI paths.
- Update extensions, developer ownership documentation, source links, and
  every path-sensitive manual reference in the same commits as the moves.

Gate: results match the baseline, package load succeeds with optional
extensions, documentation has no stale paths or links, docs build, and
formatting passes.

### Phase 2: split oversized files behind existing APIs

- Split `multigrid.jl`, `visualization*.jl`, `navier_stokes.jl`,
  `variable_staggered_diffusion.jl`, and phase-transport files by concern.
- Extract model-specific `step!` and workspace methods.
- Move refinement criteria out of physical indicator files into
  `mesh/adaptivity/`.
- Add internal interface comments for conservation, ownership, and rebuild
  invariants; avoid comments that merely restate code.
- Update developer architecture and numerical-method implementation links as
  each responsibility moves to its final file.

Gate: no public API change and no numerical/performance regression outside
predeclared tolerances; architecture documentation describes the actual split.

### Phase 3: typed model and solver configuration

- Introduce physics, velocity-model, advection, linear-solver, multigrid, and
  adaptivity option types.
- Build workspaces through model dispatch instead of `nothing` checks and
  runtime velocity-shape branches.
- Add the old-keyword compatibility translator and deprecation tests.
- Make shared initial velocity expansion explicit for two-velocity mode.
- Rewrite the high-level reference, model-selection guide, option matrix,
  migration page, and affected tutorials alongside the new constructors.

Gate: the full selection matrix constructs correctly and representative cases
complete a step in all valid combinations; old and new APIs are both
documented and all new code examples execute.

### Phase 4: first-class diagnostics

- Implement `SimulationState`, diagnostic definitions, schedules, and typed
  histories in `simulation/`.
- Add built-in energy, momentum, phase, divergence, and mesh diagnostics.
- Adapt visualization to consume diagnostic histories.
- Replace custom diagnostic boilerplate in examples and retain a compatibility
  route for `register_diagnostic!`.
- Publish the diagnostics tutorial/reference, exact discrete definitions, and
  the headless/live-visualization integration guide.

Gate: diagnostics tests pass for all models, AMR, geometry, resumed runs, and
non-scalar values; documented headless and live workflows produce the same
history in doctests.

### Phase 5: examples, README, and documentation integration

- Rewrite examples to the common contract and target size.
- Parameterize every two-phase example by `velocity_model`.
- Make instability two velocity by default and add its documented
  one-velocity closure.
- Move validation-only mathematics and loops out of examples.
- Add companion example pages/catalogue entries and representative results.
- Finish the README landing page and executable quick start using the API
  documentation already rewritten in Phases 3 and 4.
- Cross-link tutorials, examples, explanation pages, and reference pages; add
  remaining doctests.

Gate: all example smoke tests pass in both applicable velocity modes, scripts
remain directly executable, and documentation builds without missing exports.

### Phase 6: cleanup and stabilization

- Remove internal compatibility shims that were only needed between refactor
  phases; retain public deprecations for the agreed period.
- Audit exports and distinguish the compact standard API from documented
  advanced APIs. Do not remove advanced exports without a compatibility plan.
- Run the full integration and performance suites and compare with Phase 0.
- Update benchmark scripts whose paths or constructors changed.
- Run the final terminology, docstring, link, doctest, and executable-snippet
  audit and remove obsolete documentation rather than leaving redirects
  indefinitely.

Gate: all acceptance criteria below are met.

## Review and merge strategy

Prefer small reviewable changes in this order:

1. baseline tests, documentation audit, and API decision record;
2. source moves with ownership/source-link documentation;
3. test moves with testing-guide updates;
4. large-file splits with architecture/invariant documentation;
5. explicit model/configuration types with reference/tutorial/migration edits;
6. diagnostics with headless/live tutorials and mathematical definitions;
7. examples, companion pages, gallery, and README;
8. compatibility, performance, and documentation-quality audits.

Documentation edits belong in every change above. Avoid a final monolithic
"docs PR" whose examples describe an API that evolved separately.

Never combine a mechanical move with changed floating-point algebra. Use
`git diff --color-moved` during reviews, and compare numerical artifacts after
each model-specific extraction.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Include-order bugs after moving files | One labeled root include manifest, documented dependency order, package-load tests |
| Dispatch ambiguity between a constant tuple and phase velocities | Explicit velocity model; validate/expand initializers at construction |
| Numerical drift while splitting kernels | Mechanical move first; reference and invariant tests before edits |
| Invalid options accepted by an unrelated model | Model-local typed configuration and constructor-time validation |
| Compile-time growth from nested option types | Shallow immutable configs, function barriers, setup-time benchmark |
| Diagnostics change scientific meaning between models | Document and unit-test discrete phase/geometry weighting |
| Two-velocity examples are prohibitively slow in CI | Tiny smoke parameters in `test/examples/`; full studies remain validation jobs |
| Example helpers hide important setup | Share only CLI/viewer plumbing; keep physics visible in each script |
| Optional Makie/preconditioner extensions break | Dedicated load and smoke tests with weak dependencies absent/present |
| Existing users are broken abruptly | Translating constructors, actionable deprecations, migration table |
| Documentation describes an intermediate or obsolete API | Documentation deliverable and executable examples in every phase; no deferred docs-only cleanup |

## Acceptance criteria

The refactor is complete when:

- `src/TwoPhaseAMR.jl` contains imports, exports, and domain includes rather
  than implementation details.
- Source and test hierarchies use subsystem folders only where they express a
  cohesive abstraction boundary; no directory exists merely to wrap one small
  file.
- Physical indicator geometry lives in `src/indicator/`; AMR refinement
  criteria live in `src/mesh/adaptivity/`.
- `src/solvers/` contains only linear-system solution machinery;
  Navier--Stokes lives in its own top-level `src/navier_stokes/` equation-system
  folder, following the convention intended for a future `src/maxwell/`.
- Single-phase, one-velocity two-phase, and two-velocity two-phase choices are
  explicit in the high-level API.
- CSF/GFM, Frozen/Projected, and CMDGFM/Mimetic choices use typed,
  consistently named configuration fields and reject invalid combinations
  early.
- Kinetic energy and momentum can be enabled without writing a visualization
  callback, and their histories are available in headless simulations.
- Those same diagnostic histories can be selected as live visualization plots
  without duplicate evaluation or a Makie dependency in the simulation core.
- Every two-phase example runs in one- and two-velocity modes; instability
  defaults to two velocity.
- Example scripts are concise, directly executable, headless-testable, and do
  not serve as hidden test libraries.
- Unit, integration, validation, example, and performance tests have distinct
  ownership and CI treatment.
- Existing conservation, convergence, Float32, AMR, geometry, and optional
  extension coverage passes within documented tolerances.
- The rewritten README is concise, its quick start executes in CI, and it
  links rather than duplicating the manual.
- The manual follows the approved tutorial/how-to/explanation/reference
  structure, documents the final architecture and option matrix, and includes
  a complete legacy API migration guide.
- Documentation builds with `checkdocs=:exports`; links, doctests, and
  executable snippets pass; every public API has an appropriate reference
  entry and the high-level workflows execute.
- Package load, construction, first-step, steady-state step, and allocation
  measurements show no unexplained regression from the recorded baseline.
