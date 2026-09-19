# Developer architecture

The source tree is organized by ownership, not by call depth. Equation systems
such as Navier--Stokes are top-level domains. `solvers/` means linear-system
algorithms only. Physical phase indicators live under `indicator/`; AMR
refinement criteria live under `mesh/adaptivity/`.

## Dependency direction

```text
mesh topology
    -> mesh variables and transfer
    -> generic operators
    -> physical indicator geometry and transport
    -> single-/two-phase constitutive models
    -> intrinsic two-velocity specialization
    -> linear solvers
    -> Navier--Stokes equation system
    -> simulation state and diagnostics
    -> optional visualization adapters
```

`src/TwoPhaseAMR.jl` is the commented include manifest. Includes must follow
this dependency direction; numerical kernels should not depend on examples,
documentation, or a graphics backend.

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
  indicator/
    phase/             physical 0/1 phase indicator and reconstruction
                       VOF workspaces, fluxes, scalars, and AMR transfer
    embedded_*.jl      fixed solid metrics and late cut-cell operators
    plic_geometry.jl   backend-neutral PLIC geometry extraction
  solvers/
    linear_options.jl
    multigrid/         multilevel linear solution
  navier_stokes/       equation-system construction, projection, advancement
  single_phase/        material configuration and FaceVelocity stepping
  two_phase/           shared two-phase physics and FaceVelocity extensions
    two_velocity/      intrinsic phase transport, stepping, and projections
  simulation/
    diagnostics/       reductions, schedules, typed histories
  visualization/       backend-neutral fields and sessions
```

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
- `indicator/phase/vof_fluxes.jl` owns the conservative WY sweep. Momentum
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
  residual-reporting interface from `two_phase/two_velocity/projection/`.
- Visualization frames own copies or immutable snapshots. A viewer must not
  retain solver-owned mutable arrays across an update, and diagnostic
  histories remain simulation-owned even when a live viewer subscribes.

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
`test/integration/` and `test/validation/`.

Every structural or API change should include:

1. focused unit tests for validation and dispatch;
2. regression coverage for numerical behavior at risk;
3. updated examples using the public path;
4. updated README/manual/reference text;
5. a package-load check and the broadest practical test run.

The optional Makie, AMG, and incomplete-Cholesky integrations remain package
extensions in `ext/`; core source code must load without them.
