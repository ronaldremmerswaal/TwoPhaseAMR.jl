# Dynamic adaptive refinement

`BlockTreeMesh` remains the package's only mesh type. Its global `base_mesh` is
partitioned into permanent roots according to `block_size`, with one
RegionTrees quadtree per root. These trees can be refined and coarsened in
place, but their level-zero roots cannot be removed. The leaf, face, and halo
schedules are rebuilt only after the active leaf set changes. Numerical
operators still traverse flat schedules rather than the trees.

## Indicator and arbitration

The first indicator is a centred interpolation-defect estimate for a
cell-average field. For every owned cell it compares the stored value with the
average of its four axial neighbours and normalizes the defect by the largest
local magnitude (and a small positive floor). The largest cell defect in a
block determines its request:

- error greater than `E_refine`: `:refine`;
- error less than `E_coarsen`: `:coarsen`;
- otherwise: `:keep`.

Thus `E_coarsen < E_refine` supplies value hysteresis. When several indicators
are present, their requests are combined in the strict order
`:refine > :keep > :coarsen`. A sibling family is coarsened only if all four
children request coarsening. Refinement-only closure after arbitration restores
the 2:1 condition, including corner neighbours, so grading can override a
coarsening request.

Every built-in indicator is active on the inclusive physical-time interval
`t_init <= time <= t_end`; the defaults `t_init=-Inf` and `t_end=Inf` keep it
active throughout a run. `simulate!` passes the current physical time to AMR.
When calling the lower-level API directly, pass the same time as
`adapt!(amr, criteria; step, time)`. An inactive indicator makes no refinement
or keep request, so the other active indicators retain control and the mesh may
coarsen normally.

Error-based criteria (`InterpolationErrorIndicator`,
`CurvatureRefinementIndicator`, and `VorticityRefinementIndicator`) have
independent `minimum_level` and `maximum_level` bounds. The fixed-band criteria
`InterfaceRefinementIndicator` and `GeometryRefinementIndicator` instead have
one exact target `level`: blocks near the phase interface or embedded solid
boundary move toward that level, while blocks outside the band request
coarsening. Both detect boundaries aligned with cell faces through halo jumps.

`MeshAdaptationOptions` adds temporal hysteresis. `update_interval` limits
adaptation attempts in a solver run, `coarsen_delay` prevents a newly created
child from being coarsened for the given number of completed simulation steps,
and `minimum_level` protects a base level.

```julia
mesh = BlockTreeMesh(
    block_size = (16, 16), max_level = 4,
    adaptation = MeshAdaptationOptions(
        update_interval = 5, coarsen_delay = 20, minimum_level = 1
    )
)

marker = CellField(mesh)
temperature = CellField(mesh)
velocity = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
pressure = CellField(mesh)

criterion = InterpolationErrorIndicator(
    marker; E_refine = 0.08, E_coarsen = 0.025,
    minimum_level = 1, maximum_level = 4,
    t_init = 0.0, t_end = 1.0
)
solver = NavierStokesSolver(
    mesh; density = 1.0, dynamic_viscosity = 1.0e-3,
    refinement_indicators = (criterion,),
    adaptive_fields = (temperature,)
)
workspace = NavierStokesWorkspace(solver)

simulate!(velocity, pressure, solver, 1.0, workspace)
```

Velocity, pressure, every indicator field, and every entry in
`adaptive_fields` are transferred. The time integrator takes no extra physical
step and performs no extra pressure correction: after transfer it only rebuilds
the topology-dependent advection, diffusion, pressure, and Krylov workspace
state. `simulate!` reports the number of actual topology changes as
`result.mesh_updates`.

The lower-level interface accepts either indicators or one explicit label per
active block:

```julia
amr = AMRWorkspace(mesh, pressure, velocity, temperature)
labels = refinement_labels!(amr, criterion; time = 0.4)
statistics = adapt!(amr, labels; step = 100)
```

## Transfer operators and invariants

Cell data are interpreted as finite-volume averages. Restriction is the volume
average of the four fine cells. Prolongation is a conservative piecewise-linear
reconstruction: monotonized-central slopes produce the four subcell averages,
whose mean is exactly the parent average. It is affine exact and second order
in smooth monotone regions, while limiting deliberately reduces it to first
order at extrema and non-smooth interfaces.

Generic face fields use area averaging for restriction. Prolongation first
reconstructs the tangential variation with the same limited linear slope and
then interpolates between old normal face lines. For a `FaceVelocity`, newly
created internal faces receive an additional local incidence correction: each
fine cell over an old coarse cell is assigned one quarter of the old integrated
divergence. This preserves a discretely divergence-free velocity without a
second projection. On periodic component-normal directions, a uniform
divergence-free correction also preserves the discrete staggered momentum to
roundoff. At physical component-normal boundaries the boundary half-control
volumes are not owned momentum unknowns, so no global momentum correction is
applied; this retains affine exactness and does not inject boundary-cell
divergence.

The implementation reuses the mesh schedule vectors, face-field vectors, label
and geometry scratch, and pools discarded cell-block matrices. A repeated
refine/coarsen/refine cycle therefore reuses cell payload storage. A real
topology change cannot be fully allocation-free: RegionTrees nodes, temporary
connectivity dictionaries, face/halo schedule entries, and the new elliptic
hierarchies must be constructed. These costs are intentionally measured
separately from the allocation-free `restrict!` and `prolong!` kernels in
`benchmark/adaptivity.jl`.

Face transfer groups the old face segments by exact normal lattice coordinate.
Each line is sorted once. Interval queries use binary lookup followed by a walk
over only the overlapping segments; interpolation to a newly exposed face
walks outward from the insertion point until it finds the two local bracketing
lines. It therefore does not scan all old faces or all normal coordinates for
each new face. Cell ownership and halo donors use quadtree leaf descent.

Visualization frames carry an immutable, shared snapshot of their topology, so
nearest-solution checkpoints remain displayable after later regrids. Exact
off-grid checkpoints are currently rejected with dynamic adaptation because an
independent side step would require a historical clone of the tree and all
topology-dependent solver state.

## End-to-end validation

`test/amr_manufactured_flow.jl` advances a localized analytical vortex on the
periodic unit square. A periodic exponential bump defines a streamfunction,
so `u = (partial_y psi, -partial_x psi)` is exactly divergence-free; an exact
body force supplies its time derivative, nonlinear convection, and viscous
terms with zero pressure. Unlike Taylor--Green, its significant curvature is
confined to one part of the domain.

Increasing only the maximum AMR level from 1 through 3 gives observed velocity
L2 rates of 1.92 and 1.94. The level-3 tree has 2,368 active cells and the same
error as a uniform 64-by-64 mesh with 4,096 cells. Against a nearly
equal-cell-count 48-by-48 mesh (2,304 cells), it reduces the error by 43%.
The test also checks 2:1 balance, pressure convergence, and divergence at
roundoff. A second case starts from the uniform root and verifies that three
in-simulation regrids can cross from uniform to genuinely mixed topology.

The mesh for the convergence cases is adapted before the exact initial
velocity is sampled. Otherwise a maximum-level study with one fixed coarse
base mesh would mostly measure information lost in the initial coarse samples,
not the accuracy of the final adapted spatial discretization. The solver still
owns the indicator during integration and would react to later changes.

The equations, parameters, accuracy-per-cell comparisons, and end-to-end
timing/allocation results are recorded in
the [recorded AMR usefulness benchmark](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/main/benchmark/amr-usefulness-results.md).
The present implementation wins on accuracy per active cell for this case,
but its mixed-level setup cost does not yet beat the small uniform references
in wall time or allocated memory.

## Literature basis

The design follows the finite-volume principle that regridding must preserve
cell-integrated conserved quantities. Berger and Colella's classical AMR method
also separates error tagging, grid management, and conservative fine/coarse
synchronization. AMReX exposes the same distinction through conservative cell
interpolators, face-linear interpolation, divergence-preserving face
interpolation, and flux registers. Its documentation explicitly states that
fine faces over a coarse cell must reproduce the coarse divergence.

Van der Plas develops compact local refinement for the finite-volume ComFLOW
free-surface solver, including refinement-ratio-two interfaces and conservative
interface corrections. Basilisk's adaptive-wavelet documentation likewise
treats scalar values as cell averages, uses average restriction and linear or
bilinear prolongation, and permits field-specific transfer operators. Tóth and
Roe give explicit divergence-preserving restriction/prolongation formulae for
face-centred fields; the local incidence correction here uses the same
constraint-first idea, specialized to a 2-D staggered velocity and ratio two.
Finally, p4est's public API makes the topology sequence explicit: refine,
coarsen complete sibling families, then balance by additional refinement to
enforce at most 2:1 neighbour sizes.

References:

- P. van der Plas, [*Local grid refinement for free-surface flow simulations*](https://research.rug.nl/en/publications/local-grid-refinement-for-free-surface-flow-simulations/), PhD thesis, University of Groningen, 2017.
- M. J. Berger and P. Colella, [*Local Adaptive Mesh Refinement for Shock Hydrodynamics*](https://www.sciencedirect.com/science/article/pii/0021999189900351), JCP 82 (1989), 64--84.
- [AMReX `AmrCore`, interpolation, and flux-register documentation](https://amrex-codes.github.io/amrex/docs_html/AmrCore.html).
- [Basilisk adaptive-wavelet algorithm notes](https://basilisk.dalembert.upmc.fr/sandbox/Antoonvh/the_adaptive_wavelet_algorithm) and [wavelet example](https://basilisk.fr/src/examples/wavelet.c).
- G. Tóth and P. L. Roe, [*Divergence- and Curl-Preserving Prolongation and Restriction Formulas*](https://public.websites.umich.edu/~gtoth/Papers/Toth2002_amrct.pdf), JCP 180 (2002), 736--750.
- [p4est topology and 2:1 balancing API](https://github.com/cburstedde/p4est).
