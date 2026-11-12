# Dynamic adaptive refinement

`BlockTreeMesh` remains the package's only mesh type. Its global `base_mesh` is
partitioned into permanent roots according to `block_size`, with one
RegionTrees quadtree per root. These trees can be refined and coarsened in
place, but their level-zero roots cannot be removed. The leaf, face, and halo
schedules are rebuilt only after the active leaf set changes. Numerical
operators still traverse flat schedules rather than the trees.

## Refinement criteria and arbitration

The first criterion is a centred interpolation-defect estimate for a
cell-average field. For every owned cell it compares the stored value with the
average of its four axial neighbours and normalizes the defect by the largest
local magnitude (and a small positive floor). The largest cell defect in a
block determines its request:

- error greater than `E_refine`: `:refine`;
- error less than `E_coarsen`: `:coarsen`;
- otherwise: `:keep`.

Thus `E_coarsen < E_refine` supplies value hysteresis. When several criteria
are present, their requests are combined in the strict order
`:refine > :keep > :coarsen`. A sibling family is coarsened only if all four
children request coarsening. Refinement-only closure after arbitration restores
the 2:1 condition, including corner neighbours, so grading can override a
coarsening request.

Every built-in criterion is active on the inclusive physical-time interval
`t_init <= time <= t_end`; the defaults `t_init=-Inf` and `t_end=Inf` keep it
active throughout a run. `step!(flow, dt)` and `simulate!` pass the current
physical time to AMR. When calling the lower-level API directly, pass the same
time as `adapt!(amr, criteria; step, time)`. An inactive criterion makes no
refinement or keep request, so the other active criteria retain control and the
mesh may coarsen normally.

Error-based criteria (`InterpolationErrorIndicator`,
`CurvatureRefinementIndicator`, and `VorticityRefinementIndicator`) have
independent `minimum_level` and `maximum_level` bounds. The fixed-band criteria
`InterfaceRefinementIndicator` and `GeometryRefinementIndicator` instead have
one exact target `level`: blocks near the phase interface or embedded solid
boundary move toward that level, while blocks outside the band request
coarsening. With a nonzero buffer, both detect boundaries aligned with faces
between blocks through halo jumps. Their `buffer_cells` keyword (default 1,
between 0 and the mesh halo width) specifies the number of halo-cell rings
examined at each block's level. Before allowing a sibling family at or below
the target level to coarsen, the criteria also conservatively examine the
prospective parent's band. Eligible siblings are kept if their unaveraged
leaf-cell fractions indicate that the parent could be near the boundary;
siblings outside the indicator's region still request coarsening. This avoids
the static-band refine/coarsen cycle, although the conservative test may also
retain some families whose averaged parent would not be near.

`CurvatureRefinementIndicator` evaluates its error only in cells whose
open-fluid conditional liquid fraction is strictly between 0 and 1, after
fraction snapping. A liquid-filled cut cell is not a liquid–gas interface
cell and does not trigger refinement merely because its curvature is
undefined. Unresolved curvature or a filament in a genuinely mixed cell
still requests refinement up to the configured maximum level.

A block that holds a mixed cell in its own cells or in its one-cell halo
borders the interface, and every liquid-filled cut cell of that block is then
the place where the interface meets the solid: its conditional fraction is one
and its curvature undefined while the contact line is unresolved (#148).
Counting those cells keeps the contact line refined without refining the
wetted solid boundary away from the interface (#118). The test is per block
because the refinement decision is: a cell-level test would read a finer
neighbour's restriction average at a level step, which is mixed even where the
fine interface is resolved, so the coarse block would refine while its own
children coarsened again.

The curvature criterion protects the coarsening siblings of a family that
holds such a wetted cut cell when the family plus a two-cell ring spans both
phases, the same conservative parent test the fixed-band criteria use: the
merged parent would restrict to a block that is again near the interface and
would immediately refine, so the family is kept. Covered cells are excluded
from that test, because their conditional fraction is zero by construction,
not gas; without the exclusion every window crossing the solid would span both
phases and the family would never coarsen (#118). The halo of a coarse block
whose neighbours are finer is restricted from the liquid and open fractions
with the same weights for the same reason: averaging per-cell ratios would
count a covered neighbour as gas and make the block appear to border the
interface. Together these rules let the contact line refine without the
refine/coarsen cycle that made it unresolved (#148).

The interface-band criterion has different semantics: covered solid cells
have conditional liquid fraction zero, so liquid touching a solid can make a
block appear near an interface even far from a liquid–gas contact line.
Consequently the band can still refine wetted solid boundaries.

Every built-in indicator also accepts `xmin`, `xmax`, `ymin`, and `ymax`,
defaulting to an unbounded region. A block participates when its closed
rectangle intersects that region, including when it only touches an edge.
Outside it the indicator requests `:coarsen` and does not enforce its level
floor. Other indicators, the mesh minimum level, and 2:1 balancing can still
cause refinement there. For example, to exclude blocks entirely below a
solid-adjacent height:

```julia
criterion = InterfaceRefinementIndicator(interface; level = 3, ymin = 0.5)
```

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
system = NavierStokesSystem(
    mesh; density = 1.0, dynamic_viscosity = 1.0e-3,
    refinement_indicators = (criterion,),
    adaptive_fields = (temperature,)
)
workspace = NavierStokesWorkspace(system)

simulate!(velocity, pressure, system, 1.0, workspace)
```

Velocity, pressure, every refinement-criterion field, and every entry in
`adaptive_fields` are transferred. The time integrator takes no extra physical
step. After transfer it rebuilds the topology-dependent advection, diffusion,
pressure, and Krylov workspace state. Two-velocity flows also correct the
transferred velocity constraint as described below, without changing the
transferred physical pressure. `simulate!` reports the number of actual
topology changes as `result.mesh_updates`; full steps also update
`flow.mesh_updates`.

The lower-level interface accepts either criteria or one explicit label per
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
Cartesian divergence. On an uncut mesh this preserves a discretely
divergence-free velocity without a second projection. On periodic
component-normal directions, a uniform divergence-free correction also
preserves the discrete staggered momentum to roundoff. At physical
component-normal boundaries the boundary half-control
volumes are not owned momentum unknowns, so no global momentum correction is
applied; this retains affine exactness and does not inject boundary-cell
divergence.

For two-velocity flow, independent Cartesian phase-velocity transfers do not
preserve the divergence of the aperture-weighted mixture when phase apertures
change. Reconstructed apertures can also change a refined parent's net boundary
flux, which cannot be fixed by adjusting only its internal fine faces. After a
topology change, `step!(flow, dt)` and `simulate!` rebuild the operators and
apply the configured two-velocity projection to the transferred velocities,
using a scratch pressure so that the transferred physical pressure is
retained. This adds one projection per regrid, using the same tolerance and
time-step-scaled defect as the regular physical-step projection. The defect uses the
untruncated proposed step, so a short final step does not weaken the
constraint when the simulation resumes.
Standalone `adapt!` only transfers fields; passive-transport callers must
enforce their carrier constraint themselves.

The implementation reuses the mesh schedule vectors, face-field vectors, label
and geometry scratch, and pools discarded cell-block matrices. A repeated
refine/coarsen/refine cycle therefore reuses cell payload storage. A real
topology change cannot be fully allocation-free: RegionTrees nodes, temporary
connectivity dictionaries, face/halo schedule entries, and the new elliptic
hierarchies must be constructed. These costs are intentionally measured
separately from the allocation-free `restrict!` and `prolong!` kernels in
the `adaptivity` group of the current benchmark suite.

Face transfer groups the old face segments by exact normal lattice coordinate.
Each line is sorted once. Interval queries use binary lookup followed by a walk
over only the overlapping segments; interpolation to a newly exposed face
walks outward from the insertion point until it finds the two local bracketing
lines. It therefore does not scan all old faces or all normal coordinates for
each new face. Cell ownership and halo donors use quadtree leaf descent.

Recorded frames carry an immutable, shared snapshot of their topology, so
nearest-state keyframes remain displayable after later regrids. Exact off-grid
keyframes are currently rejected with dynamic adaptation because an
independent side step would require a historical clone of the tree and all
topology-dependent solver state.

## Accuracy expectations

Adaptive refinement preserves cell-integrated scalars, phase volume, and the
discrete divergence constraint of transferred face velocity. Smooth localized
flows retain close to second-order global velocity convergence when the
criteria resolve their active region. Compare at least two maximum levels: a
criterion that never activates can produce a plausible-looking but
under-resolved result.

The block-tree projection setting follows the line of adaptive incompressible
flow solvers introduced by [Popinet (2003)](https://doi.org/10.1016/S0021-9991(03)00298-5).
Phase-interface refinement and conservative free-surface transfer are also
described by [van der Plas (2017)](https://research.rug.nl/en/publications/local-grid-refinement-for-free-surface-flow-simulations/)
and [Remmerswaal (2023)](https://doi.org/10.33612/diss.562133319).

## References

- M. J. Berger and P. Colella, [*Local adaptive mesh refinement for shock
  hydrodynamics*](https://doi.org/10.1016/0021-9991(89)90035-1), Journal of
  Computational Physics 82 (1989), 64--84.
- S. Popinet, [*Gerris: A tree-based adaptive solver for the incompressible
  Euler equations in complex
  geometries*](https://doi.org/10.1016/S0021-9991(03)00298-5), Journal of
  Computational Physics 190 (2003), 572--600.
- P. H. A. J. M. van der Plas, [*Local grid refinement for free-surface flow
  simulations*](https://research.rug.nl/en/publications/local-grid-refinement-for-free-surface-flow-simulations/),
  PhD thesis, University of Groningen (2017).
- R. Remmerswaal, [*Numerical modelling of variability in liquid
  impacts*](https://doi.org/10.33612/diss.562133319), PhD thesis, University
  of Groningen (2023).
- W. Zhang, A. Myers, K. Gott, A. Almgren, and J. Bell, [*AMReX:
  Block-structured adaptive mesh refinement for multiphysics
  applications*](https://doi.org/10.1177/10943420211022811), International
  Journal of High Performance Computing Applications 35 (2021), 508--526.
- G. Tóth and P. L. Roe, [*Divergence- and curl-preserving prolongation and
  restriction formulas*](https://doi.org/10.1006/jcph.2002.7120),
  Journal of Computational Physics 180 (2002), 736--750.
- C. Burstedde, L. C. Wilcox, and O. Ghattas, [*p4est: Scalable algorithms for
  parallel adaptive mesh refinement on forests of octrees*](https://doi.org/10.1137/100791634),
  SIAM Journal on Scientific Computing 33 (2011), 1103--1133.
- A. van Hooft, [*The Grid Adaptation Algorithm Based on a Wavelet-Estimated
  Discretization Error*](https://basilisk.dalembert.upmc.fr/sandbox/Antoonvh/the_adaptive_wavelet_algorithm),
  Basilisk documentation.
