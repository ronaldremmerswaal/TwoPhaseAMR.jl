# Development guide for a block-tree two-phase flow solver

This document defines the scope and development order for a Julia package that
solves the incompressible Navier--Stokes equations on block-structured adaptive
tree meshes. The final solver must support two immiscible fluids, geometric
volume-of-fluid (VOF) interface advection, and quantities stored at different
mesh locations.

A mesh consists of fixed-size Cartesian blocks arranged as the leaves of a
refinement tree. Cell-centred quantities, such as pressure and volume fraction,
are stored at cell centres. Face-staggered quantities, such as normal velocity
and mass or volume flux, are stored on the corresponding block faces. Dense
block-local arrays and halos provide the data used by numerical kernels; the
tree provides refinement, neighbour, and block-connectivity information.

Implement this as a separate, tree-native package. Do not retrofit the tree and
staggered layouts into `MimeticFV.jl`: mesh hierarchy, field placement,
coarse--fine coupling, and memory layout are part of the solver's core data
model and determine both readability and performance. Use existing packages as
references instead:

- use `MimeticFV.jl` as the correctness reference for single-phase,
  cell-centred advection on uniform meshes;
- use `IncompressibleNavierStokes.jl` as the numerical and performance
  reference for a uniform, single-block pressure projection; and
- couple the two-phase advection stage to `GeometricVOF.jl` if its `Meshes.jl`
  interface can be used without abandoning the block-local data layout.

Develop the solver in working vertical slices: centred single-phase advection,
staggered transport, a single-phase pressure projection, wall diffusion and a
fixed-tree single-phase solver, two-phase advection, and finally the complete
two-phase Navier--Stokes solver. The milestones below make this sequence
precise.

## Design rules

- Use an existing tree package. Do not implement a new tree. Evaluate
  [P4est.jl](https://github.com/trixi-framework/P4est.jl),
  [T8code.jl](https://github.com/DLR-AMR/T8code.jl), and
  [RegionTrees.jl](https://github.com/JuliaGeometry/RegionTrees.jl), then record
  the choice and its licensing, balancing, neighbour, ghost, and parallelism
  implications in a short architecture decision record. A tree leaf represents
  a fixed-size Cartesian block.
- Begin in 2-D on the CPU, with refinement ratio two, 2:1 balance, and one
  global time step. Add 3-D, accelerators, distributed execution, and level
  subcycling only after the numerical interfaces are stable.
- Store field data in dense block-local arrays with halos. Tree traversal may
  build execution and exchange schedules after regridding, but must not occur
  inside stencil kernels.
- Represent placement explicitly, e.g. `Cell` and `Face{Axis}`. Keep block
  interiors, same-level exchange, coarse--fine transfer, and physical boundary
  treatment as separate operations.
- Every milestone needs correctness, conservation, allocation, and timing tests.
  Benchmark setup/regridding separately from time-stepping kernels.

## Research before implementation

Perform and document a focused literature review of staggered, block- or
tree-adaptive incompressible solvers. At minimum, determine:

- ownership of face degrees of freedom at coarse--fine interfaces;
- construction of staggered transport fluxes from centred fluxes, including
  the algebraic compatibility of the two divergence operators;
- compatible divergence and pressure-gradient operators;
- conservative restriction, prolongation, synchronization, and refluxing;
- stability and null spaces of the composite pressure operator;
- divergence-preserving interpolation and possible spurious interface modes;
- treatment of variable density, viscosity, and surface tension.

Useful starting points are:

1. Remmerswaal,
   [*Numerical Modelling of Variability in Liquid Impacts*](../phd_project/thesis/thesis.pdf),
   PhD thesis, University of Groningen, 2023. The source is locally available
   in `/home/ronald/Documents/phd_project/thesis`. In particular, Chapter 6,
   Sections 6.4 and 6.4.1 (pp. 181--184), constructs the staggered mass or
   volume flux by applying a flux interpolant to the centred flux. This makes
   the staggered mass equation the algebraic image of the centred equation,
   keeps centred and staggered mass synchronized, and avoids additional
   subgrid flux computations. Treat this as the primary design reference for
   centred-to-staggered flux construction.
2. Olshanskii, Terekhov, and Vassilevski,
   [*An octree-based solver for the incompressible Navier--Stokes equations with enhanced stability and low dissipation*](https://doi.org/10.1016/j.compfluid.2013.04.027).
   This is especially important because it identifies spurious velocity modes
   associated with staggered coarse--fine interfaces.
3. Gerya et al.,
   [*An adaptive staggered grid finite difference method for modeling geodynamic Stokes flows with strongly variable viscosity*](https://doi.org/10.1002/ggge.20078).
4. Almgren et al.,
   [*A Conservative Adaptive Projection Method for the Variable Density Incompressible Navier--Stokes Equations*](https://doi.org/10.1006/jcph.1998.5890).
5. Popinet,
   [*Gerris: a tree-based adaptive solver for the incompressible Euler equations in complex geometries*](https://doi.org/10.1016/S0021-9991(03)00298-5).

Summarize the selected discrete layout and its invariants before implementing
staggered fields. Preserve the review as package documentation and extend it as
new issues appear.

## Incremental milestones

### 1. Centred, single-phase foundation

Implement cell-centred fields and prescribed face fluxes on:

1. one block;
2. several same-level blocks;
3. a static two-level tree with a coarse--fine interface.

Support initialization, integration, boundary/halo filling, conservative scalar
advection, restriction, and prolongation. Match `MimeticFV.jl` on equivalent
uniform grids. A constant state must remain constant and total scalar mass must
be conserved across block and coarse--fine boundaries.

### 2. Staggered fields and operators

Add directional face fields, explicit ownership rules, and staggered halo and
coarse--fine transfer. Implement compatible divergence and pressure-gradient
operators. Following Remmerswaal's thesis, first derive staggered transport
fluxes algebraically from centred fluxes; do not independently reconstruct
them geometrically unless tests demonstrate that the algebraic construction
cannot meet the tree-interface requirements. Test that centred and staggered
mass remain synchronized, in addition to manufactured fields, discrete adjoint
identities, preservation of constants, exact linear-momentum conservation, and
the absence of coarse--fine checkerboard or oscillatory velocity modes.

#### 2.1 Directionally split LW at AMR interfaces

Status: implemented and benchmarked; nonlinear TVD limiters are deferred.

- Keep factored AMR connections separate by coordinate direction even when
  both directions connect the same pair of staggered unknowns.
- Apply LW through two full, published directional intermediate states.
  Alternate x/y and y/x order on successive steps, and include the
  directional dual-divergence term in the variable-coefficient LW predictor.
- Test the amplification matrix of the complete split step, rather than
  constructing an update from a simultaneous old-state flux sum.
- Localize the LW/interface-upwind truncation error. The isolated upwind bias
  is `O(1)` after dual-volume division over an `O(h)` interface strip, which
  explains the observed global `O(h^(1/2))` norm.
- Reconsider enhanced Monotonized Central, other nonuniform TVD limiters, or
  full-graph FCT only when bounded transport of under-resolved fields becomes
  a requirement. A correct AMR limiter needs a geometric upstream patch; a
  local two-state interface switch is insufficient.

The reproducible convergence, interface-localization, and spectral experiment
is `benchmark/staggered_advection_interface.jl`.

### 3. Single-phase pressure--Poisson solve

Implement the pressure operator and a projection that reduces divergence to the
requested solver tolerance on single- and multilevel meshes. Establish the
operator's null-space convention and verify convergence independently of block
partitioning.

For a single block, pressure--Poisson performance must be comparable to
`IncompressibleNavierStokes.jl`. Use a reproducible, like-for-like benchmark:
the same grid, precision, boundary conditions, residual tolerance, initial
guess, hardware, and thread/backend configuration. Record warm median solve
time, memory, allocations, iterations, final residual, and projected divergence.
Define the numerical performance threshold before optimizing; the single-block
path must not pay for tree traversal or coarse--fine machinery.

#### 3.1 Linear-solver composition

Status: implemented; parameter tuning remains.

- Keep geometric multigrid independent of Krylov iteration. A hierarchy is a
  reusable three-argument `ldiv!` preconditioner and owns no convergence state.
- Compose elliptic operators with external Krylov.jl workspaces. Retain CG as
  the pressure default and BiCGSTAB as the staggered-diffusion default.
- Provide weak-dependency extensions for AlgebraicMultigrid.jl,
  IncompleteLU.jl, and LimitedLDLFactorizations.jl.
- Test every applicable operator/preconditioner pairing and compare setup,
  warm solve time, iterations, residual, and retained memory on one common
  graded mesh. Tune the generic AMG/ILU parameters only after this untuned
  baseline is recorded.

### 4. Wall boundaries, staggered diffusion, and single-phase flow

On every non-periodic domain boundary, impose no permeation automatically and
use the corresponding homogeneous Neumann condition for the pressure
correction. Add a staggered viscous operator with `NoSlip()` as the default
wall closure and `FreeSlip()` as an explicit alternative. At coarse--fine
interfaces, prioritize a stable conservative diffusion core and use a widened
consistent stencil where the compact stencil loses accuracy.

Advance the incompressible equations with one split momentum step:

1. one second-order Lax--Wendroff convection update;
2. one Crank--Nicolson/implicit-midpoint diffusion solve; and
3. exactly one compatible pressure projection.

The shifted staggered diffusion problem defaults to the package's geometric
multigrid preconditioner, with direct factorization confined to the coarsest
grid; alternate public-package preconditioners use the same solver interface.
Keep the elliptic operator contract suitable for later variable viscosity.
Validate the solver with a periodic Taylor--Green vortex and a non-periodic
lid-driven cavity on uniform and static mixed-level trees.

#### 4.1 Implicit-diffusion solver optimization

Status: hybrid stable-core hierarchy and public-preconditioner comparison
implemented; fine-grid matrix-free application remains optional follow-up.

The [implicit-diffusion solver review](docs/src/implicit-diffusion-solver-review.md)
found excessive grid and Galerkin operator complexity in the original face
hierarchy. The implementation now preserves implicit midpoint without a
diffusion CFL and BiCGSTAB as the default nonsymmetric Krylov method, smooths
the exact corrected matrix on the fine level, and uses compact symmetric
`M + alpha*K` coarse levels. Face aggregation follows logical dual-cell
topology rather than coordinate bins. Transfers are row-truncated schedules;
their adjoint restriction is applied without storing `P'`, and setup-only
stable stiffnesses are not retained as duplicate level matrices. Repeated
`(A+A')/2` sparse cleanup is deliberately absent because Galerkin symmetry is
already inherited algebraically.

Named AMG, ILU, and incomplete-Cholesky diffusion preconditioners are now
built from the stable core while the Krylov operator remains the exact
corrected matrix. CG, BiCGSTAB, GMRES, and FGMRES workspaces from Krylov.jl are
supported, with method settings passed through `krylov_options` and the
pressure/diffusion defaults unchanged.

The optimization gate records grid/operator/transfer complexity, setup
allocation, retained bytes, V-cycle time, true residual history, and
setup-amortized solve time over mesh-size and stiffness sweeps. A new default
must preserve mixed-level convergence and use no more than 2.5 times the
pressure-preconditioner storage on the common reference mesh. The implemented
hierarchy passes that storage gate and scales better on the largest benchmark,
but plain BiCGSTAB remains faster on the small reference problem. Keep both
paths in benchmarks before introducing an automatic small-problem policy.

#### 4.2 Dynamic block adaptation

Status: first single-phase implementation complete; problem-specific indicators
and level subcycling remain future work.

- Keep one `BlockTreeMesh` type and mutate its RegionTrees topology only at
  explicit completed-step boundaries. Rebuild flat leaf, face, halo, elliptic,
  and advection schedules only after an actual leaf-set change.
- Combine any number of centred block indicators with
  `:refine > :keep > :coarsen` precedence. Require complete sibling families
  for coarsening and restore 2:1 grading with refinement-only closure.
- Store mesh-level update cadence, a protected minimum level, and the last
  refinement step of new children. Use both threshold hysteresis and a
  coarsening cooldown to suppress refine/coarsen chatter.
- Restrict cell averages conservatively and prolong them with conservative
  limited linear reconstruction. Restrict generic face fields by area and use
  limited tangential/normal reconstruction for prolongation.
- Apply a local incidence correction to newly created internal velocity faces
  so every fine child inherits one quarter of its old parent cell's integrated
  divergence. Verify mass, periodic staggered momentum, affine exactness, and
  divergence preservation on uniform and graded topology changes.
- Let a Navier--Stokes solver own optional refinement indicators and additional
  adaptive fields. Transfer velocity and pressure automatically, perform no
  extra physical step or projection, and reconstruct only topology-dependent
  solver/workspace state.
- Pool cell-block payloads and reuse vector/scratch capacity. Benchmark the
  allocation-free transfer kernels separately from the necessarily allocating
  tree/connectivity/operator rebuild.

The implementation and literature basis are documented in
[Dynamic adaptive refinement](docs/src/adaptive-refinement.md).

### 5. Two-phase advection

Add geometric volume-fraction advection and consistent phase-volume, mass, and
staggered momentum fluxes by coupling to `GeometricVOF.jl`.

First build a small compatibility prototype that exposes block cells and faces
as `Meshes.jl` geometries without materializing a global unstructured mesh or
allocating geometry in the hot loop. Verify that the required clipping,
intersection, reconstruction, and donating-region operations are supported in
the target dimensions. If the `Meshes.jl` API cannot provide this adapter, or
doing so would compromise the block layout, stop and ask the package owner how
to proceed before implementing an alternative geometry layer.

Validate interface translation and reversal, boundedness of the volume
fraction, phase-volume conservation, and uniform velocity across a density
jump before adding further two-phase physics.

### 6. Two-phase Navier--Stokes

Only after the previous milestones pass, add the variable-coefficient pressure
projection, discontinuous viscosity, balanced surface tension, gravity, and
two-phase-aware refinement criteria/transfer. Add hydrostatic equilibrium,
capillary-wave, and rising-
bubble validation cases one feature at a time.

### 7. Interactive visualization

Build visualization as an optional observer of the time-integration loop, not
as part of a numerical operator or workspace. The public interface may expose
one `VisualizationSession` object, but that object must compose independent
simulation control, live-frame, checkpoint, scalar-history, and rendering
components. The GUI must never read solver-owned arrays while a time step is
mutating them.

Use Makie as the rendering abstraction. Support GLMakie for a local desktop
window and WGLMakie, with Bonito where application-level browser controls are
needed, for a browser interface. Keep these packages out of the mandatory
solver dependencies by loading the renderer through a Julia package extension.
The numerical and snapshot layers must remain usable and testable without an
active graphics backend.

Development proceeds through the following intermediate milestones.

#### 7.1 Static mixed-level visualization

Status: initial implementation complete; backend performance benchmarks remain.

- Convert the owned cells of a `BlockTreeMesh` to backend-independent plotting
  geometry without including halos or traversing `RegionTrees`.
- Render a scalar `CellField` with piecewise-constant cell colors and separate
  block-boundary and optional cell-boundary layers.
- Maintain explicit primitive-to-`CellRef` and cell-to-block mappings so a
  picked graphical primitive identifies its numerical owner.
- Permit replacement of field values without rebuilding unchanged mesh
  geometry.
- Exercise the same viewer with GLMakie and WGLMakie and benchmark geometry
  construction and value-update cost on representative mixed-level meshes.

#### 7.2 Field registry and derived scalars

Status: 2-D implementation complete, including dynamic-topology replacement
and retained derived scalar values.

- Register primitive cell and directional face fields by stable names, units,
  placement, and color defaults.
- Provide one-call default registration that expands each two-component face
  field into its directional components, centred magnitude, and vorticity.
- Render face fields as filled staggered dual-control-volume tiles, preserving
  the package's coarse--fine ownership convention and covering the domain
  without gaps.
- Add demand-driven scalar derivations, beginning with cell-centred magnitude
  and vorticity for generic two-component face fields. Store the primitive
  inputs needed to recompute a
  derived field at a retained state rather than storing every derivation.
- Let the user switch fields without changing solver state.

#### 7.3 Live session and sampling policies

Status: solver lifecycle hooks, immutable copied frames, step- or time-based
live sampling, and opt-in asynchronous latest-wins publication are implemented.
Control polling and separate scalar-diagnostic sampling remain.

- Add `VisualizationSession`, lifecycle hooks, and an optional
  `visualization` keyword to `simulate!`. Invoke the hook before the first
  step, after every successful step, and on completion or failure.
- Initialize the observer with a lazy independent-step procedure that owns a
  separate solver workspace. Keep checkpoint schedules and off-grid time
  decisions entirely inside the observer; `simulate!` only publishes lifecycle
  updates and follows its unmodified numerical step sequence.
- Copy data selected for asynchronous use at the completed-step boundary. No
  GUI task may retain a borrowed view of mutable solver storage.
- Separate control polling, scalar sampling, live-field publication, and
  checkpoint sampling policies. Calling the observer every step must not imply
  rendering or serializing a full mesh every step.
- Publish transient live data through a bounded, latest-wins mailbox so a slow
  viewer cannot apply backpressure to the simulation.
- Keep initial and final live publications mandatory. Expose the number of
  replaced transient frames and a completion wait for tests and orderly
  application shutdown. Never send retained checkpoints through the lossy
  live mailbox.

#### 7.4 Checkpoints and historical mode

Status: in-memory schedules, nearest/exact capture, historical/live modes,
dynamic geometry snapshots, and Makie checkpoint navigation are implemented.
Exact off-grid side steps remain restricted to fixed topology.

- Implement a fixed-capacity in-memory checkpoint store with an evenly spaced
  simulation-time policy when the final time is known, including initial and
  final states where possible.
- Accept a checkpoint count, checkpoint time interval, or explicit absolute
  times. Store the nearest completed state by default. Exact mode advances a
  copy of the preceding state by invoking the independent-step procedure
  supplied at observer initialization. The session owns this decision and
  retains the result without changing the actual simulation's step sequence.
- Store copied primitive field data and cached registered scalar views. Share
  one immutable mesh-geometry object across consecutive frames until the
  active block keys change, so historical frames retain their old topology.
- Keep live ingestion active while a historical checkpoint is displayed.
  Switching back to live mode must immediately display the most recent
  complete live frame.
- Call these records visualization snapshots unless they contain all state
  required for a numerical restart.

#### 7.5 Simulation control and diagnostics

Status: complete for the fixed-topology 2-D solver, including cooperative
pause/resume, independently sampled scalar histories, and linked Makie
checkpoint markers.

- Run the simulation in a task separate from GUI event handling. GUI callbacks
  send commands through synchronized control state and never mutate numerical
  fields directly.
- Implement pause and resume at completed-step boundaries. Report separately
  that a pause was requested and that the simulation has reached the paused
  state; pause latency is therefore bounded by one time step.
- Support registered scalar reductions at a higher frequency than field
  frames. Display selected reductions in one linked time-series axis, overlay
  checkpoint markers, and select a checkpoint by picking a marker.
- Surface running, pause-requested, paused, completed, and failed states in the
  interface, and ensure errors wake any task waiting on the control state.

#### 7.6 Browser application and future scaling

Status: browser work deferred; current interactive development targets the
GLMakie desktop application.

- Provide a single-user browser application using WGLMakie and Bonito, binding
  to a loopback address by default. Document WebSocket/proxy configuration for
  deliberate remote use.
- Keep the Makie scene usable with GLMakie and use native browser widgets where
  they reduce Julia/browser round trips.
- Dynamic-topology replacement is implemented for live and nearest-checkpoint
  frames. Add 3-D,
  distributed in-situ visualization, disk-backed snapshots, restart files, and
  multi-user control only as separate, measured milestones. Reconsider
  ParaView Catalyst when distributed execution becomes an active requirement.

The first release is complete at milestone 7.5 for the current 2-D solver,
including dynamically adapted nearest-state frames. It need not support
rotation, because rotation has no useful
meaning for the present planar fields, and it must not claim restartable
checkpoints or exact dynamic-AMR side-step checkpoints.

## Completion discipline

A milestone is complete only when its tests pass on every mesh configuration
introduced so far and its benchmark results are recorded. Prefer a narrow
working vertical slice over speculative general interfaces; extract shared code
from existing packages only after a second implementation demonstrates that the
abstraction is genuinely common.
