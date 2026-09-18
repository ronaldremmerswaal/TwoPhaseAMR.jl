# Two-velocity model implementation plan

This document plans the two-velocity model described in the thesis. Each main subsection is a separate milestone with its own implementation tasks, tests, and acceptance gate. Passive interface, scalar, and momentum transport were implemented first; Milestone 5 now specifies and validates the pressure projection. Two-velocity diffusion remains open.

The implementation must preserve the existing one-velocity and single-phase paths. It must work on uniform meshes, adaptive block-tree meshes, coarse-fine interfaces, embedded cut-cell geometry, and combinations of AMR with cut-cell geometry.

## Scope and governing contracts

The first implementation stage covers:

1. two-phase velocity data and activity semantics;
2. two candidate interface-advection methods;
3. compatible centred scalar advection;
4. compatible staggered momentum advection; and
5. passive-advection validation and comparison of the two interface methods;
6. the three-block mimetic pressure projection; and
7. inviscid integration in the existing Navier--Stokes solver.

The following contracts apply to every milestone.

### Store intrinsic quantities

Persistent fields store intrinsic quantities:

- volume fraction, not phase volume;
- density or another intrinsic scalar, not phase mass;
- velocity, not momentum.

Phase volumes, phase masses, and momenta are temporary conservative carriers used inside an advection step. They may be cached in a workspace, but they are not the user-facing solution variables.

### One geometric source of truth

The interface PLIC reconstruction is the source of phase geometry. In cells intersected by fixed solid geometry:

- liquid and gas volumes add to the open cell volume;
- liquid and gas face apertures add to the open face aperture;
- neither phase volume may exceed the geometry-open volume; and
- neither phase face aperture may exceed the geometry-open face aperture.

Existing geometry-aware PLIC reconstruction, swept-volume integration, solid face apertures, dual volumes, AMR neighbour queries, and geometric restriction/prolongation must be reused. Parallel copies of these algorithms must not be introduced for the two-velocity model.

### Shared fluxes define compatibility

For a given phase and directional sweep, interface advection, centred scalar advection, and staggered momentum advection must use the same phase-volume flux. Reconstructing a similar flux independently in each kernel is insufficient because roundoff and limiting decisions can then differ.

The required identities are:

- advecting the liquid one-function reproduces the updated liquid volume fraction;
- advecting the gas one-function reproduces the complementary open fraction;
- a constant intrinsic phase scalar remains constant wherever that phase is active; and
- equal-valued liquid and gas velocities recover the continuous-velocity limit at the convergence order stated in the thesis while still exercising the two-velocity discretization.

### Activity and the global fraction floor

A phase velocity is active on a face exactly when its staggered phase fraction is positive after the existing global wisp-removal rule has been applied. The default fraction floor remains `sqrt(eps(T))`; callers may lower it only to a positive value, such as `eps(T)`, for targeted robustness experiments.

No independent activity tolerance or stored Boolean activity mask is introduced. Activity is derived from the current geometrical phase fraction so that it remains consistent after advection and mesh adaptation.

Division by a phase volume or staggered phase volume is performed only in active control volumes. In inactive control volumes, a documented fill/extension rule sets the intrinsic field; a small denominator is never silently substituted into the conservative quotient.

### Mesh support required at each gate

Unless a test is physically inapplicable, every milestone must be exercised on:

- a uniform mesh;
- a fixed mixed-level AMR mesh;
- a dynamically adapting AMR mesh;
- a uniform mesh with fixed cut-cell geometry; and
- a mixed-level AMR mesh with fixed cut-cell geometry.

Coarse-fine fluxes must be computed on the finest physical subfaces and refluxed so that the sum of fine fluxes is the coarse flux. Tests must include an interface and non-constant phase fields crossing coarse-fine boundaries rather than merely residing beside them.

## Existing foundation to reuse

Before adding code, record the relevant implementation entry points and invariants in a short developer note beside the new workspaces. The expected reusable foundation includes:

- `CellField`, `FaceField`, and `FaceVelocity` storage;
- `InterfaceField`, its level-set constructor, conditional and physical fractions, PLIC data, and global fraction-floor handling;
- `PhaseScalar` and `PhaseFaceScalar` for phase-specific intrinsic fields;
- `WYAdvectionWorkspace` and its liquid/gas volume-flux arrays;
- the directionally split geometric WY sweep and geometry-aware swept-fraction calculation;
- the current compatible centred scalar update;
- `MomentumAdvectionWorkspace` and the staggered WY/CTU machinery;
- solid face apertures, phase face-aperture evaluation from adjacent PLIC traces, and cut-cell dual volumes;
- block-local traversal, coarse-fine adjacency, refluxing, and tree-based restriction/prolongation; and
- the existing matrix-free operators and tree-based geometric multigrid components that can support an auxiliary velocity projection.

This inventory is a gate: new geometric searches or storage are added only when none of these tools can express the required operation.

## Milestone 1: data structures and common API

### Public representation

`TwoPhaseVelocity` is a concept represented by an ordinary two-element tuple:

```julia
velocity = (liquid_velocity, gas_velocity)
```

Both entries are existing `FaceVelocity` objects on the same mesh and use the fixed ordering `(liquid, gas)`. No wrapper with duplicated storage is required. Methods may dispatch on `Tuple{<:FaceVelocity,<:FaceVelocity}`. A lightweight type alias may be added only if it improves signatures or documentation without changing this representation.

The same rule applies to centred phase data when a pair is needed:

```julia
scalar = (liquid_scalar, gas_scalar)
```

The existing `PhaseScalar` and `PhaseFaceScalar` abstractions should remain usable when they already encode this meaning cleanly. The implementation should generalize them rather than create a competing family of containers.

### Construction and validation

Add internal validation that checks once at construction or workspace creation that:

- both tuple entries use the same mesh object and numeric type;
- their boundary-condition layouts are compatible;
- their face index spaces agree with the interface and geometry fields; and
- a supplied geometry belongs to that mesh.

Do not repeat structural checks inside performance-critical face loops.

### Activity queries

Provide allocation-free internal helpers for:

- liquid and gas cell fractions relative to the open cell volume;
- liquid and gas staggered fractions relative to the open dual volume;
- liquid and gas physical face apertures clipped by solid geometry; and
- phase activity on cells and faces.

These helpers must call the existing reconstruction/aperture tools. They should make the identities between liquid, gas, and solid fractions explicit and testable.

### Common advection API

Use `advect!` for both one-velocity and two-velocity paths. The intended high-level shape is:

```julia
velocity = (liquid_velocity, gas_velocity)
workspace = WYAdvectionWorkspace(
    interface,
    velocity;
    interface_method = ProjectedLiquidVelocity(),
)

advect!(
    interface,
    velocity,
    dt,
    workspace;
    scalars = phase_scalars,
    staggered_scalars = phase_face_scalars,
    momentum = destination_velocity,
    density = (rho_liquid, rho_gas),
    momentum_workspace,
    sweep_order = (1, 2),
    scheme = LW(),
)
```

The exact keyword grouping can be refined during implementation, but the following API rules are fixed:

- `advect!` remains the common verb;
- a single call owns one complete split step and its shared phase fluxes;
- the interface and passive scalars can be advanced without requesting momentum;
- momentum self-advection may use the same velocity tuple as source and destination because the workspace snapshots the old state; and
- `stable_timestep` accepts the velocity tuple and applies the method-specific two-velocity CFL definition.

The single-phase and one-velocity methods remain valid specializations rather than being routed through unnecessary two-velocity allocations.

### AMR transfer semantics

Restriction and prolongation act on each tuple entry using the existing intrinsic-velocity transfer policy. They do not transfer momentum and divide it back into velocity. After transfer:

1. reconstruct the interface geometrically;
2. recompute staggered phase fractions from that reconstruction and the fixed geometry;
3. determine activity from those fractions; and
4. fill newly active velocities with the method-specific extension rule before they are used.

Flux registers, temporary conservative carriers, and activity derived from the old topology are invalidated or rebuilt through the normal topology-generation mechanism.

### Tests

- tuple construction and rejected mesh/type mismatches;
- activity in pure, interfacial, solid, cut, and coarse-fine control volumes;
- phase-volume and phase-aperture partition identities;
- restriction/prolongation of constant and linear intrinsic fields;
- appearance and disappearance of active phase faces after adaptation; and
- zero allocation for the lowest-level activity/aperture queries after compilation.

### Gate

Milestone 1 is complete when the tuple works with the common `advect!` and `stable_timestep` entry points, all geometry identities hold to roundoff, and AMR transfer introduces neither invalid active values nor stored phase-weighted state.

## Milestone 2: interface and centred scalar advection

Two interface methods will be implemented behind the same policy argument and compared under identical mesh evolution, time steps, sweep ordering, reconstruction, and diagnostics.

### 2.1 Shared derivation for directionally split WY

Before coding either method, write down the discrete equations for one directional sweep. This research task must resolve:

- which velocity supplies the indicator-flux term and the WY dilation term;
- how liquid and gas conservative carriers are updated during intermediate sweeps;
- whether the PLIC geometry is frozen for a full time step or reconstructed between directional sweeps;
- how the two phase fluxes partition the geometry-open swept volume;
- the exact two-velocity CFL condition, including the WY-specific sum of directional Courant numbers;
- how fluxes on coarse-fine subfaces are accumulated and refluxed; and
- the reduction to the present one-velocity WY equations.

The derivation must cite the relevant thesis equations and compare them with the split geometric VOF literature already used by the package. An executable one-dimensional reference implementation should verify the algebra before the optimized block-local kernel is written.

### 2.2 Shared phase-flux storage

Extend `WYAdvectionWorkspace` rather than create an unrelated workspace. It must hold, for every directional sweep:

- geometry-open swept volume;
- liquid swept volume;
- gas swept volume;
- WY dilation terms required by the chosen method;
- per-phase flux registers at coarse-fine boundaries; and
- optional diagnostics for non-partitioning fluxes and discarded gas carrier.

The flux arrays are produced once and consumed by interface, centred-scalar, and staggered-momentum updates. Workspace capacity may grow with topology size and is reused between steps.

### 2.3 Method A: projected liquid-velocity extension

This method follows the conservative-liquid construction in the thesis.

#### Research and formulation

Specify the constant-extrapolation band for missing face velocities, including how far it must reach for PLIC reconstruction, a full WY sweep, CTU momentum stencils, coarse-fine neighbours, and cut-cell neighbours. The auxiliary Poisson problem itself follows `pt1_mass.tex`: pressure is unknown only in cells with positive liquid fraction, pressure is zero in liquid-empty cells, and only faces with positive staggered liquid fraction receive a pressure correction. Derive boundary conditions for:

- the external domain boundary;
- the liquid-gas interface;
- solid geometry;
- coarse-fine interfaces; and
- disconnected or vanishing phase regions.

The auxiliary projection must produce the discrete divergence property required by the same flux-divergence operator used by WY. A continuum divergence-free extension is not sufficient if the discrete face fluxes fail that identity.

#### Implementation tasks

1. Build the cell and staggered-face liquid-activity masks with block-local tree traversal.
2. Extrapolate missing liquid face velocities by constant extrapolation into the required gas band.
3. Solve the thesis Poisson problem in positive-liquid cells, with homogeneous Dirichlet pressure in liquid-empty cells, using the same discrete divergence and gradient pair as WY and fixed cut-cell geometry.
4. Compute liquid interface fluxes with the projected field and existing geometry-aware swept-volume routine.
5. Compute the auxiliary gas carrier with the gas velocity and its own phase fluxes.
6. At the start of each directional substep, initialize the auxiliary gas carrier from the current physical complement. Recover intrinsic gas scalars from the conservative auxiliary result, then discard that carrier before the perpendicular sweep. This split-specific reset is necessary because the PLIC geometry represents the physical complement rather than a possibly greater-than-one auxiliary carrier.
7. Accumulate the gas-volume and gas-momentum discrepancy caused by every directional reset rather than hiding it.
8. Extend/fill newly active intrinsic scalars and velocities before division or interpolation.

The solid-open aperture is applied before the liquid-gas partition. The projected extension cannot create flux through blocked parts of a face.

#### Method-specific gates

- projected-band divergence meets the same nondimensional tolerance used by its flux operator;
- liquid volume is conserved to solver tolerance on fixed meshes and through AMR refluxing;
- the reconstructed liquid fraction remains within `[0, geometry_fraction]` without post-update clipping beyond the global wisp-removal rule;
- gas carrier non-conservation is reported and converges under refinement in smooth passive tests; and
- distinct, equal-valued phase fields pass the classical vortex-reversal refinement study without being dispatched to the one-velocity kernel.

### 2.4 Method B: frozen-interface phase cut-cell fluxes

This method treats the liquid-gas PLIC as a frozen internal cut-cell geometry while computing each directional phase flux.

#### Geometry tasks

1. Reuse the existing phase face-aperture evaluation; generalize it only where the flux integral needs subface interval endpoints in addition to a measure.
2. Intersect phase subfaces with the existing solid-open aperture before extrusion.
3. Reconcile the traces reconstructed from both adjacent cells with one documented symmetric rule.
4. Split coarse faces into finest-level subfaces so one geometrical phase partition is used on both sides of every AMR connection.
5. Treat wall/interface triple points with the same LHF reconstruction and geometry clipping used elsewhere in the package.

#### Flux tasks

1. Extrude each phase subface with its corresponding normal phase velocity.
2. Integrate the PLIC fraction over the resulting swept region.
3. Construct liquid and gas fluxes and measure their failure, if any, to partition the geometry-open flux.
4. Insert the per-phase fluxes into the split WY update with the derived dilation terms.
5. Use the same flux arrays for all compatible scalar and momentum updates.
6. Preserve boundedness by the method's discrete construction; do not conceal an inconsistent flux with ad hoc clipping.

#### Method-specific gates

- exact phase-flux partition for stationary and common-normal-velocity interfaces;
- exact preservation of a planar interface under tangential velocity jumps;
- bounded fractions without repair other than global wisp removal;
- conservation errors and convergence rates reported separately for both phases; and
- a documented result for the thesis convergence concern under directionally split WY.

### 2.5 Compatible centred scalar advection

Centred phase scalars use the phase-volume fluxes from the selected interface method. For each phase and sweep:

1. form the temporary phase amount from intrinsic scalar times geometrical phase volume;
2. advect it with the stored phase-volume flux and the existing phase-interface upwind interpolant;
3. update the corresponding phase volume with exactly the same flux and WY dilation term; and
4. recover the intrinsic scalar only in active cells.

The constant-one identity is tested at every directional sweep, not only after a full time step. A non-constant field must not be overwritten by arbitrary values from an inactive phase when that phase becomes active; it uses the extension or upwind state defined by the selected interface method.

### 2.6 Passive-advection validation matrix

All errors are measured through refinement studies. A single-resolution comparison with an ad hoc error threshold is not an accuracy test.

#### Exact and invariant tests

- stationary planar interface with unequal arbitrary phase velocities that have zero normal component;
- planar interface translated by equal normal velocities;
- slab with a common normal velocity and discontinuous tangential velocities;
- constant liquid and gas centred scalars, including values different from one;
- liquid and gas one-functions;
- circle rotation and translation across block and coarse-fine boundaries; and
- the same cases clipped by a stationary planar or circular solid geometry.

The tangentially discontinuous circular-disc rotation is a dedicated
integration gate for both interface policies. The projected-liquid policy
rebuilds its active projection and projects at every advection step. The
frozen-subface policy receives a velocity pair satisfying the current
aperture-weighted mixture-divergence constraint; the passive test therefore
reprojects that mixture after every aperture update. This test will gain
centred and staggered passive fields during Milestone 3.

#### Reversal tests

- classical vortex reversal for the interface;
- non-constant smooth centred scalar in each phase;
- fields with non-zero interface-normal gradients to exercise phase-interface upwinding;
- forward and reversed velocity histories with exact comparison at the final time; and
- interface crossing fixed mixed-level and dynamically adapted regions.

Use the thesis norms and expected orders. The target is approximately second-order convergence in integral norms for smooth transported fields and the documented lower pointwise order associated with PLIC/interface geometry. Report observed orders rather than forcing them with loose pass constants.

#### Comparison gate

Run both methods with identical initial data, time steps, sweep ordering, topology sequence, and reconstruction. Compare:

- liquid and gas volume conservation;
- boundedness and wisp removal;
- interface shape error and observed order;
- centred scalar error and observed order;
- projection iterations and cost for Method A;
- phase-flux partition defect for Method B;
- discarded gas carrier or momentum;
- allocations and time per cell update; and
- sensitivity to density ratio as a diagnostic even though the test is passive.

Milestone 2 closes only after this comparison supports a documented default method. Both methods remain selectable until that decision is reviewed.

## Milestone 3: compatible staggered momentum advection

This milestone begins only after one interface method passes the passive centred-scalar gate. It retains both phase velocities rather than forming a mass-weighted mixture velocity.

### Discrete formulation

For each staggered component and phase:

1. interpolate the already computed centred phase-volume fluxes to the corresponding dual control-volume faces using the present algebraic compatible interpolation;
2. preserve the one-function identity on the dual mesh;
3. form the temporary phase momentum from intrinsic phase velocity, phase density, and staggered phase volume;
4. advect it with the interpolated phase-volume flux and phase-interface upwinding;
5. use the existing bulk advection scheme away from the interface and the compatible CTU treatment where required near the interface;
6. update the staggered phase volume from the same interpolated flux; and
7. divide back to intrinsic velocity only where the updated staggered phase volume is positive.

Liquid and gas updates remain separate. The kernel must not sum their momenta and divide by a mixture density. Density factors should cancel algebraically for constant-density passive transport where possible, while remaining explicit in conservation diagnostics.

### Implementation tasks

1. Generalize `MomentumAdvectionWorkspace` to hold two phase carriers and two output velocity fields while reusing the existing dual-flux interpolation schedules.
2. Make every supported flux interpolant accept the tuple representation and phase apertures.
3. Snapshot both old velocity fields so in-place self-advection is well-defined.
4. Define the value of a newly active phase velocity from the selected interface-extension/upwind rule.
5. Apply solid apertures and cut dual volumes before liquid-gas subdivision.
6. Reflux each phase momentum across coarse-fine boundaries using the finest physical subfaces.
7. Preserve and expose the liquid/gas momentum discrepancy introduced by Method A's complementary gas reset.
8. Keep block-first traversal and thread-local accumulation used by the optimized one-velocity kernels.

### Passive momentum tests

- constant phase velocities on stationary planar and curved interfaces;
- common normal velocity with discontinuous tangential velocity;
- unequal smooth velocity fields confined to their phases;
- one-function transport on every staggered component;
- all supported flux interpolants at an interface, a coarse-fine boundary, and a cut face;
- vortex reversal with non-constant centred and staggered phase fields;
- refinement studies on uniform and fixed mixed-level meshes;
- dynamic AMR with a prescribed topology history shared by both candidate interface methods; and
- embedded geometry with an interface and velocity discontinuity crossing cut cells and AMR boundaries.

For the vortex reversal, report separate errors for liquid velocity, gas velocity, liquid momentum, gas momentum, and interface geometry. Establish the convergence order stated in the thesis; do not accept a test solely because the final error lies below a hand-chosen constant.

### Performance tests

Add serial and threaded benchmarks for:

- phase dual-flux interpolation;
- two separate staggered carrier updates;
- activity and safe-division recovery;
- AMR refluxing; and
- the complete coupled interface/scalar/momentum `advect!` call.

Compare against the existing one-velocity WY path at identical mesh topology and number of active faces. Report initialization separately from steady per-step time, allocations, memory traffic where available, and parallel speedup.

### Gate

Milestone 3 is complete when all interpolants preserve the constant-one identity, both intrinsic velocity fields converge at the expected order, AMR and cut-cell flux balances close to roundoff or solver tolerance, and distinct equal-valued phase carriers recover the thesis continuous-velocity convergence limit through the two-velocity implementation itself.

The rotating-disc results do not settle the best location for a discontinuous phase velocity. The present auxiliary projections use a continuous pressure gradient, whereas projection into a divergence-free space is not unique once the admissible velocity and pressure-gradient spaces permit phase jumps. The failure of phase-face-centroid sampling to improve convergence under this continuous-gradient projection must therefore be treated as a result for the current projection, rather than a final rejection of that velocity representation.

## Milestone 4: integrated passive-transport decision

Before adding pressure projection or diffusion:

1. run the complete validation matrix for both interface methods;
2. compare accuracy, conservation, boundedness, solver robustness, and cost;
3. select and document the default interface method;
4. retain the alternative behind the same policy interface if it provides a useful accuracy or robustness trade-off;
5. freeze the public tuple and `advect!` API; and
6. record all conservation defects that the later pressure model must account for.

The integration gate requires clean package tests in serial and threaded modes plus a reproducible benchmark report. It also requires a small example showing both phase velocity fields and the reconstructed PLIC interface on an adapting mesh with fixed cut-cell geometry.

## Milestone 5: pressure projection

Implement the aperture-based mimetic pressure projection from
`pt2_poisson.tex`, using the symmetric three-block reformulation in
`model_from_scratch.tex`.  The retained unknowns are the pressure `p`, the
phase-gradient jump `xi` on mixed staggered faces, and the cell multiplier
`lambda` for the aperture-normal constraint.  The integrated system is

```text
[ -A_p   C    0 ] [p     ]   [ dt^-1 div <a u***>       ]
[  C'   K_xi  B'] [xi    ] = [ 0                         ]
[  0     B    0 ] [lambda]   [-dt^-1 div ([a] [u***])   ]
```

where `A_p` reuses the existing variable-density pressure operator, `C`
couples pressure to the aperture/mass mismatch, `K_xi` is the positive
relative phase mass on retained faces, and `B` is the existing conservative
cell/face incidence with the aperture jump.  The same AMR-compatible
divergence, gradient Riesz map, geometry apertures, staggered fractions, and
cut-cell volumes used by the one-velocity pressure projection must be reused.

### 5.1 Algebra and active sets

- retain `xi` only where both staggered phase fractions are active under the
  global interface `fraction_floor`;
- extend the velocity and gradient jump by zero outside that set;
- remove one pressure gauge per connected open-fluid component;
- remove all null modes of `[a] grad(lambda)` before solving;
- never divide by `[a]` or an individual phase mass; and
- reconstruct both phase gradients with the conservative mass-average
  identity and the solved jump.

### 5.2 Initial solver

Use MINRES because the assembled KKT operator is symmetric indefinite.  Start
with the positive block diagonal preconditioner consisting of `A_p`, `K_xi`,
and the approximate multiplier Schur complement `B K_xi^-1 B'`.  An exact
sparse Cholesky factor of this positive block is the dependency-free default;
the existing optional limited-memory incomplete Cholesky integration remains
available as `preconditioner=:ichol`.  Check the true unpreconditioned residual
after MINRES and use residual-correction iterations when its recurrence-based
stopping estimate is optimistic.

**TODO (solver optimisation):** cache the three fixed sparsity patterns and
MINRES vectors across time steps; refresh only phase-dependent coefficients
and active-set rows; replace the exact block Cholesky with a scalable tree
multigrid approximation for `A_p` and the multiplier Schur complement; measure
iteration counts and allocations over AMR topology changes; and investigate a
matrix-free block action once the assembled reference is fully validated.

### 5.3 Algebraic validation gate

On periodic uniform, fixed mixed-level AMR, cut-cell, and combined AMR/cut-cell
meshes, verify:

- exact symmetry of the integrated three-block matrix;
- both final divergence constraints to the requested physical residual;
- the conservative phase-momentum identity, with global pressure momentum at
  roundoff on periodic grids;
- orthogonality of the pressure correction to the admissible final velocity;
- the corresponding Pythagorean kinetic-energy relation;
- constant-pressure and multiplier gauges and the rank of every retained
  block; and
- robustness for density ratios from one through at least `1e3`, including
  faces approaching the fraction floor.

### 5.4 Transport and instability validation gate

The two-velocity state is advanced through the existing
`IncompressibleNavierStokesSolver` and `step!`/`simulate!` API.  Dispatch on a
liquid/gas `FaceVelocity` tuple selects two separate compatible momentum
updates and the mimetic pressure projection; a single `FaceVelocity` retains
the established one-velocity path.  Surface tension and gravity modify the
predictor and therefore enter the two three-block right-hand sides.  They are
not applied as a post-projection velocity correction.  Two-velocity viscous
diffusion remains explicitly unsupported until Milestone 6.

1. Repeat the tangentially discontinuous rotating-disc refinement study.  At
   every step reinitialise the prescribed analytical phase velocities, apply
   this physical mimetic projection, and then call the unchanged common
   `advect!` API.  Report pre/post-projection phase-velocity `Linf` errors,
   interface `L1`/`Linf` errors, and staggered passive-scalar errors beside the
   earlier continuous-gradient projection results.
2. Implement Kelvin--Helmholtz waves from `pt2_validation.tex` for unit and
   `1e3` liquid-to-gas density ratio, including stable and unstable Weber
   numbers.  Fit the numerical complex frequency from interface amplitude and
   phase and compare it with the analytical dispersion relation under mesh
   refinement.
3. Implement Rayleigh--Taylor waves for several Atwood/Bond combinations from
   the thesis.  Fit oscillation frequency in the stable regime and exponential
   growth rate in the unstable regime, and establish convergence to the
   analytical dispersion relation.
4. Run each dispersion case on a uniform mesh and a prescribed fixed
   mixed-level mesh before enabling dynamic AMR.  Dynamic AMR must reproduce
   the same rate with interface-presence refinement and a topology-independent
   diagnostic.

The gate is passed only by refinement rates and dispersion errors.  A bounded
single-resolution run is not sufficient.

**Current rotating-disc result.**  The three-block solve closes both constraints
to the requested tolerance, but this alone does not establish an accurate
projection.  With reinitialised analytical face velocities, the
projected-liquid interface path gives a decreasing interface `L1` error with
an aggregate rate of about 1.6 on grids with 8, 16, and 32 cells per domain
width.  The frozen-aperture path decreases much more slowly, and its
phase-velocity and interface `Linf` errors are not monotone.  The largest
corrections occur as a staggered phase mass approaches zero, which is the
loss-of-coercivity limit identified in `model_from_scratch.tex`.  The test
records all errors and only gates rates that are presently asymptotic.  The
full-period frozen study now uses this three-block gradient directly; its
16--64 interface `L1` errors are nonmonotone even though both divergence
constraints close to tolerance.  The former common-gradient surrogate has
been removed, and the expected frozen convergence checks remain explicit
broken gates.  The frozen-aperture pressure/advection combination therefore
remains an open accuracy issue.

The general instability example evaluates the thesis dispersion relation with
shear, gravity, and surface tension present simultaneously.  Its pure KH
amplitude histories converge for unit and `1e3` liquid-to-gas density ratio
and for stable and unstable Weber numbers.  On the current 8--32 grids, the
observed KH rate is approximately first order because the split transport and
coarse-grid pressure error dominate.  The pure RT histories converge at about
second order for all tested density ratios and stable/unstable Bond numbers,
and a combined KH/RT refinement case exercises all three terms.  These tests
compare the complete complex amplitude history with the analytical dispersion
solution; no fitted single-resolution frequency is used as a substitute for
refinement.  The visualized example plots normalized numerical and theoretical
amplitude magnitudes together.

The auxiliary narrow-band projection used by the projected-extension advection method belongs to Milestone 2 and does not decide the eventual physical two-velocity pressure projection.

**TODO:** Revisit the nonuniform frozen-aperture rotating-disc correction with
MDGFM and with an explicit study of the retained mixed-face mass threshold.
Measure whether either alternative preserves the prescribed tangential jump
without losing the exact mixture and normal constraints.

## Milestone 6: diffusion

Open milestone. Its interfacial stress conditions, coupling between phase velocities, cut-cell viscous discretization, AMR treatment, solver structure, and validation cases will be planned separately after passive transport has passed Milestone 4.

## Proposed implementation sequence

Each item should be a reviewable commit with its gate passing before the next begins:

1. tuple API, activity/aperture helpers, and geometry identity tests;
2. written split-WY derivation plus executable one-dimensional reference equations;
3. shared phase-flux workspace and diagnostics;
4. projected liquid-velocity extension and centred passive scalar tests;
5. frozen-interface phase cut-cell method and the same tests;
6. controlled accuracy, conservation, and performance comparison;
7. separate compatible staggered phase-momentum updates;
8. AMR, cut-cell, threaded, and all-interpolant validation;
9. integrated passive vortex-reversal convergence study and benchmark report; and
10. default-method decision and public API review.

Milestone 5 builds on this sequence through the existing `step!` and
`simulate!` API. Two-velocity diffusion remains excluded; the auxiliary
projection used by Method A is part of interface advection and is independent
of the physical three-block pressure projection.
