# Developer architecture

The source tree is organized by ownership, not by call depth. Equation systems
such as Navier--Stokes are top-level domains. `solvers/` means linear-system
algorithms only. Phase reconstruction, solid geometry, and VOF transport live
under `interface/`, `embedded/`, and `vof/`; AMR refinement criteria live
under `mesh/adaptivity/`.

## Dependency direction

```text
mesh topology
    -> mesh variables and transfer
    -> generic operators
    -> interface and embedded geometry, then VOF transport
    -> single-/two-phase constitutive models and linear solvers
    -> intrinsic two-velocity specialization and projections
    -> Navier--Stokes equation system
    -> simulation state and diagnostics
    -> optional observers (recording) and visualization utilities
```

`src/TwoPhaseAMR.jl` is the commented include manifest. The diagram describes
ownership rather than a strict total include order: type dependencies determine
the actual order, and mesh criteria follow the physical fields they extend.
Numerical kernels should not depend on examples, documentation, or a graphics
backend.

## Source ownership

```text
src/
  mesh/
    topology/          block tree, ownership, placements
    adaptivity/        criteria, arbitration, regridding
    variables.jl       centred and staggered field storage
    transfer.jl        conservative field transfer
  operators/
    advection/         cell and staggered transport
    gradient.jl
    staggered_diffusion.jl
    elliptic.jl
  interface/           phase field, reconstruction and initialization
  embedded/            solid geometry, utilities and cut-cell operators
  vof/                 transport policies, workspaces, fluxes and scalars
  solvers/
    linear_options.jl
    multigrid/         multilevel linear solution
  navier_stokes/       equation-system construction, projection, advancement
  single_phase/        material configuration and FaceVelocity stepping
  two_phase/           shared two-phase physics and FaceVelocity extensions
    diffusion/         variable coupled-stress diffusion
    projection/        intrinsic projection methods and implementations
    two_velocity/      intrinsic phase transport and stepping
  simulation/
    diagnostics/       reductions, schedules, typed histories
  observation/         observer interface, Recorder, snapshot store
  visualization/       field registry, sampling, rasterizer, camera, metadata
```

The optional `lib/TwoPhaseAMRStudio` subpackage provides an HTTP/WebSocket
adapter and a browser shell over `Recorder` and `SnapshotStore`. Browser
playhead state is per client; the recorder and solver controls remain on the
server. Studio dependencies are not dependencies of the numerical core.

A directory must earn its nesting: it should contain multiple cohesive files
with a shared interface or independent invariants. Do not introduce a folder
for one file. Place a singleton at the nearest meaningful parent and create a
subdirectory only when the subsystem is actually split.

Large files should be split at stable boundaries such as public types versus
assembly kernels versus orchestration. Do not split an allocation-sensitive
kernel merely to meet a line-count target.

## Split interfaces and invariants

The Phase 2 files communicate through a small set of internal protocols:

- `navier_stokes/stages.jl` defines shared step utilities. Concrete
  `step!` methods and stage extensions live with `single_phase/`,
  `two_phase/`, or `two_phase/two_velocity/` according to their primary
  dispatch type.
- `vof/fluxes.jl` owns the conservative WY sweep. Momentum
  workspaces extend its validation, directional update, and finalization
  hooks from the appropriate velocity-model folder; the physical indicator
  layer does not inspect a concrete two-velocity workspace type.
- `mesh/adaptivity/` owns all refinement decisions and all topology-changing
  transfer. Physical phase and solid fields only provide reconstruction and
  prolongation hooks. A topology rebuild must preserve field object identity
  and retarget every cached schedule before the next numerical step.
- `solvers/multigrid/` returns linear-solver statistics and knows nothing
  about a physical projection model. `navier_stokes/projection.jl` owns the
  shared-velocity pressure projection, while intrinsic projections extend the
  residual-reporting interface from `two_phase/projection/`.
- Recorded frames own copies or immutable snapshots. An observer must not
  retain solver-owned mutable arrays across a step, and diagnostic histories
  remain simulation-owned; a recorder only mirrors their samples.

The conservation-sensitive transport kernels remain cohesive: one directional
WY sweep updates liquid volume, gas complement, phase scalars, and momentum
from the same frozen face partition. Splitting those loops further would make
that invariant harder to review, so the file boundary is around workspace,
flux/update, and AMR-transfer responsibilities rather than individual loops.

## Vocabulary

- **physical indicator**: a phase or solid characteristic function and its
  discrete fractions;
- **refinement criterion**: an AMR object that requests refine/keep/coarsen;
- **operator**: a discrete spatial mapping;
- **linear solver**: Krylov, multigrid, or a preconditioner;
- **equation system**: configured Navier--Stokes discretization;
- **velocity model**: shared or intrinsic phase velocities;
- **simulation**: evolving fields, workspace, time, and histories;
- **diagnostic**: a scheduled reduction of a completed simulation state.

Compatibility type names containing `Indicator` or `Solver` may remain while
the documentation uses this vocabulary consistently.

## Tests

The test tree mirrors source domains only where a domain has several files.
Singleton tests stay at the nearest useful parent. Unit tests cover public
configuration, state ownership, diagnostics, and isolated numerical
invariants. Larger convergence and literature-reference cases live under
`test/integration/`.

Every structural or API change should include:

1. focused unit tests for validation and dispatch;
2. regression coverage for numerical behavior at risk;
3. updated examples using the public path;
4. updated README/manual/reference text;
5. a package-load check and the broadest practical test run.

The optional Makie, AMG, and incomplete-Cholesky integrations remain package
extensions in `ext/`; core source code must load without them.
