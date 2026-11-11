# Literature coverage TODOs

This page tracks implemented numerical ingredients that do not yet have a
direct literature reference in the user documentation. It is intentionally a
contributor-facing audit: the numerical-method pages contain the references
that already support the documented model, while this page identifies the
remaining gaps without selecting candidate papers.

A broad background reference near a method does not close an item below. The
item is complete only when the exact implemented choice is linked to a primary
source, or when the documentation clearly labels it as a package-specific
construction and explains its derivation and validation.

## Mesh adaptation and transfer

- **TODO:** Reference or derive the exact centred-interpolation defect used by
  `InterpolationErrorIndicator`, including its normalization, block maximum,
  and refine/keep/coarsen hysteresis.
- **TODO:** Reference or derive the exact geometry, phase-band, curvature, and
  vorticity refinement labels and their multi-criterion arbitration rules.
- **TODO:** Reference or derive the package's mixed-level face transfer,
  divergence-restoring correction, and coarsening-delay policy. Existing AMR
  references establish the conservation goals, but not this complete local
  construction.
- **TODO:** Reference or derive phase-aware PLIC prolongation/restriction and
  its coupling to intrinsic `PhaseScalar` and phase-face-field transfer.

## Spatial operators

- **TODO:** Add a numerical-method description and direct reference for the
  standalone first-order cell-centred upwind advection operator.
- **TODO:** Reference or derive the nonconforming centred-to-staggered flux
  factorization, product-weight transverse routing, and exact
  `D_s I_F = I_C D` construction. The current references cover its uniform-grid
  basis and related AMR ideas, not this graph extension.
- **TODO:** Reference or derive the variable-velocity directional
  Lax--Wendroff predictor and the alternating two-sweep update used on the AMR
  staggered graph.
- **TODO:** Reference or derive the exact vertex-patch inverse Riesz map used
  by `GradientOperator`, including its hanging-face stabilization.
- **TODO:** Reference or derive the symmetric non-diagonal coefficient
  insertion `sqrt(kappa) * K * sqrt(kappa)` used by the variable-coefficient
  elliptic operator.
- **TODO:** Reference or derive the mixed-level staggered viscous closures,
  including their geometric AMR contacts and the exact no-slip/free-slip wall
  treatment.

## Interface and embedded geometry

- **TODO:** Reference the polygon-clipping algorithm used by
  `PolygonIndicator` to compute exact cell fractions for piecewise-linear
  interfaces.
- **TODO:** Reference or derive the embedded-solid PLIC clipping, cut-cell
  divergence/gradient closures, and their mixed-level extension. The present
  embedded-boundary references do not describe this complete combination.
- **TODO:** Reference or derive the precise fallback and clipping policy among
  local height functions, geometric height functions, Weymouth--Yue curvature,
  and unavailable (`NaN`) curvature.
- **TODO:** Reference or derive the `fraction_floor` wisp removal and phase
  activation rules used consistently by scalar recovery, velocity recovery,
  projection, and mesh transfer.

## Two-velocity model

- **TODO:** Add a direct reference for the complete CMDGFM pressure projection,
  including its bounded multidimensional jump interpolation and two-block
  preconditioner.

## Time integration and linear solution

- **TODO:** Reference or derive the full Navier--Stokes stage ordering,
  including force splitting around implicit midpoint diffusion and the single
  final pressure projection.
- **TODO:** Reference the capillary, gravity-wave, and combined automatic
  timestep restrictions used by `stable_timestep`.
- **TODO:** Reference or derive the dimensionless pressure/diffusion residual
  targets and their floating-point roundoff floors.
- **TODO:** Reference or derive the coefficient-dependent smoothed transfer
  used by tree geometric multigrid and the exact rediscretized staggered and
  coupled-stress hierarchies.
- **TODO:** Add method references for the supported Ruge--Stuben AMG and
  incomplete-Cholesky choices. Smoothed aggregation, ILUT, and the Krylov
  methods currently used by the package are already referenced.

## Keeping this audit current

When a new numerical option is added, update its method page and bibliography
in the same change. If no direct reference is available, add a TODO here and
state explicitly whether the implementation is a package-specific extension.
