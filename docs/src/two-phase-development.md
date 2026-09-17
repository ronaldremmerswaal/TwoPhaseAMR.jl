# Two-phase extension: numerical design and implementation record

This branch extends the existing Navier--Stokes solver with an optional
interface. `step!` has one staged control flow for both configurations:
transport, optional external forcing, implicit diffusion, phase forces, and
one pressure projection. Dispatch selects the single- or two-phase numerical
kernel inside each stage. This keeps the original high-level API while adding
compatible transport, variable material operators, and balanced forces.

## Existing structure and API

- `mesh.jl`: RegionTrees-backed 2-D blocks, owned composite face segments,
  periodic/physical boundaries, immutable execution schedules between regrids.
- `fields.jl`, `placements.jl`, `transfer.jl`: haloed cell averages and compact
  face unknowns; conservative restriction and limited cell prolongation.
- `advection.jl`: centred upwind transport and primary divergence.
- `staggered_advection.jl`: dual-volume flux schedules. Their divergence equals
  the volume-weighted interpolation of the primary divergence, including AMR.
- `gradient.jl`, `elliptic.jl`, `multigrid.jl`: compatible gradient/divergence,
  variable face coefficients, Krylov solvers and geometric multigrid. Variable
  coefficient projection already exists. Its mixed-level inverse face metric
  is not diagonal: coefficient application must retain the symmetric
  `sqrt(coefficient) * inverse_metric * sqrt(coefficient)` construction.
- `staggered_diffusion.jl`, `implicit_diffusion.jl`: constant-viscosity,
  componentwise staggered diffusion and implicit midpoint stepping.
- `navier_stokes.jl`, `flow_simulation.jl`: low-level solver/workspace/`step!`
  plus the high-level `FlowSimulation` owner; configuration cloning and topology
  rebuilds are centralized.
- `adaptivity.jl`: field registration, refinement indicators and in-place
  conservative topology changes.

## Primary numerical specification

The user's thesis is available locally in
`/home/ronald/Documents/phd_project/thesis`. Relevant sources are
`basic_interface.tex`, `int_reconstruction.tex`, `int_advection_split.tex`,
`pt1_mass.tex`, `pt1_momentum.tex`, `pt2_diffusion.tex`, `st_curvature.tex`,
`st_csf.tex`, `st_gfm.tex`, and `pt2_gravity.tex`.

The model is **one velocity and two incompressible phases**. Persistent fields
store intrinsic physical values: density rather than cell mass, scalar value
rather than phase-weighted amount, and velocity rather than momentum.
Conservative phase products exist only as temporary algebra inside transport
and transfer operations.

### Interface reconstruction and curvature

`InterfaceField` owns liquid fractions, dimensionless PLIC plane coefficients,
curvature and a reconstruction-method flag. The liquid is on the negative side
of the plane; positive curvature denotes a convex liquid droplet.

1. LHF uses seven-cell columns, pure opposite endpoints and monotone mixed
   fractions, as specified in `basic_interface.tex`. Three consecutive valid
   heights give centred differences for the normal and curvature.
2. Multiple valid directions use the thesis's quality-weighted normal blending.
   **Approved correction (user, this session):** replace the inconsistent
   printed quality polynomial with bounded cubic smoothstep across the stated
   angular overlap `π/4 ± π/20`. The printed expression can give negative
   qualities even though the text requires values in `[0,1]`.
3. If LHF is unavailable, GHF fits a quadratic in local normal/tangent
   coordinates using separated height points and the thesis's squared normal
   projection weights. If necessary, it expands the neighbourhood through
   three cell widths and adds neighbouring reconstructed PLIC centroids with
   unit weights. Rank-deficient fits are rejected.
4. The final normal fallback is WY reconstruction: three-cell height sums and
   an orientation/filling-selected one-sided difference. It is not a
   volume-fraction-gradient normal. See Weymouth & Yue (2010), Section 2.1,
   [original article](https://doi.org/10.1016/j.jcp.2009.12.018).

The method flag is 0 (pure), 1 (LHF), 2 (GHF), 3 (WY), or 4 (contact-angle
reconstruction). Unavailable curvature is `NaN`, never silently zero. Geometry
tests inspect these flags.
Height halos are sampled on the local level by conservative restriction and
PLIC subcell integration. Three synchronous passes refresh coarse–fine planes;
this finite iteration is a current accuracy limitation requiring convergence
checks at refinement boundaries. The approach follows the local-level sampling
principle in [Popinet (2009)](https://doi.org/10.1016/j.jcp.2009.04.042).

Wall-adjacent reconstruction accepts one contact angle or a
`(left, right, bottom, top)` tuple. Near-right angles use height Neumann data.
Acute and obtuse angles use the thesis quadratic construction with an unknown
contact location. The reconstruction flag is 4 in these cells. Uniform,
fixed-mixed, and dynamically adapted sessile-droplet tests check the resulting
equilibrium shape through mesh refinement.

### Fixed arbitrary geometry

`GeometryField(mesh, phi)` represents `phi <= 0` as solid with the same
cell-average fraction and PLIC plane used by `InterfaceField`. Its complement
defines open cell volumes. Intersections of the shared PLIC traces with cell
faces define open apertures, and line-by-line bisection of each open cell gives
the staggered control volumes in [Verstappen and Dröge
(2005)](https://doi.org/10.1016/j.crme.2004.09.021).

The cut divergence multiplies each Cartesian flux by its open aperture. The
pressure gradient uses its negative adjoint with the cut staggered volumes, so
the pressure matrix remains symmetric. Primary mass fluxes are averaged with
weights one half onto staggered control faces; centred transported velocity
then gives a skew-symmetric convection operator. Diffusive connections share
one conductance in both adjacent equations, including stationary embedded-wall
connections, which gives a symmetric positive semidefinite stiffness.

Covered background unknowns remain in the arrays with zero physical volume
and a unit algebraic constraint. Tree multigrid reconstructs this constraint
on every coarse level from the restricted open-face coefficient, preventing
solid interiors from creating extra null modes. Geometry PLIC data use exact
geometric restriction and prolongation during AMR. The stored phase fraction
is the physical liquid volume divided by the complete Cartesian-cell volume,
so it always lies between zero and `open_fraction`. Phase transfer clips its
PLIC polygon against the open polygon and conserves that stored physical
liquid volume. Reconstruction, height functions, and constitutive laws use a
derived conditional fraction within the open volume.

At an embedded triple point, ordinary LHF columns stop at the solid rather
than treating it as gas. GHF adds the thesis's high-weight linear
contact-angle equation at the intersection of the phase and solid PLIC lines.
When a phase jump crosses an open face whose two conditional cell fractions
are pure, balanced surface tension extends the same finite GHF curvature from
the local physical stencil. Closed solid faces carry neither surface tension
nor gravity force.

### WY transport

The initial `c > 1/2` indicator stays frozen across all directional sweeps.
Each sweep reconstructs the updated interface. Owned face segments carry one
signed time-integrated liquid volume; gas volume is total minus liquid. In a
cut cell, the swept depth is obtained by inverting the solid-clipped PLIC area
so its open volume exactly matches the aperture flux. This keeps both phase
fluxes bounded by their available donor volumes beside a sloped wall.

The WY CFL is total incoming open volume divided by open cell volume, at most
`1/2`.
Outgoing volume and donor extrusion lengths are also checked; the extrusion
check protects coarse-cell subfaces. The additional constraint can be more
restrictive than the thesis's uniform-grid bound.

Centred phase scalars reuse these fluxes. They use the selected bulk
interpolant and switch to phase upwinding whenever an old or predicted carrier
fraction is mixed. The dilation concentrations are frozen for the step. Staggered phase scalars
and one-velocity momentum use the existing algebraic dual-flux interpolant.
Every staggered bulk interpolant (`Upwind`, `Centred`, `LW`, `Fromm`, `Minmod`,
`MC`, `VanLeer`, and `Superbee`) is supported. The modified interpolant switches
to term-wise CTU upwinding when either adjacent predicted phase fraction is
below `ctu_threshold=0.5`, following `pt1_momentum.tex`. Their dilation source
uses `interpolate(chi * directional_divergence)`, not a product of separately
interpolated factors. Division during intrinsic-value recovery uses a finite
`interface.fraction_floor`, which defaults to `sqrt(eps(T))`. The same value is
the global VOF cleanup threshold: a conditional phase fraction at or below the
floor is removed, while a fraction within the floor of one is set to the
cell's full open volume. This deliberately permits a tiny loss of either phase
to prevent persistent wisps and makes every intrinsic-value division safe.

Current boundary support is impermeable or periodic interface transport.
Staggered phase transport includes physical boundary half-volumes. Topology
dependent workspaces are rebuilt after mesh adaptation. Allocation scaling of
transport, variable diffusion, variable-coefficient projection, and balanced
force kernels is checked on uniform and mixed-level meshes.

### Diffusion and pressure coefficients

The diffusion operator implements `div(2*mu*D(u))` as the negative adjoint
product `-E' W E` on staggered control volumes. Normal stresses are collocated
with cell centres and shear stresses with cell vertices. At every stress
location it first interpolates the liquid fraction and then evaluates the
thesis coefficient

```math
\bar\mu=(\mu^\ell)^c(\mu^g)^{1-c}.
```

The implementation performs the equivalent operation of interpolating
`log(mu)` because it is affine in `c`. `dynamic_viscosity` exposes the
cell-centred values. The operator retains its geometric strain and fraction-
interpolation stencils across VOF updates; only stress weights and the coupled
stiffness are refreshed. Positive phase viscosities give a symmetric positive
semidefinite stiffness, and density appears only in the implicit staggered
mass matrix. Both phase viscosities may be zero to select inviscid flow.
The widened transition stencil on fixed mixed-level grids presently converges
at first order in the manufactured variable-viscosity test; this is recorded
as an accuracy limit rather than hidden by an absolute tolerance.

The pressure projection uses `1/<rho>` at staggered locations, with
`<rho>=rho_l*c_f+rho_g*(1-c_f)`. On mixed-level grids the inverse face metric
is non-diagonal. Applying the coefficient symmetrically as
`sqrt(1/rho) K sqrt(1/rho)` preserves the gradient/divergence adjoint relation.
CSF, conservative GFM, and gravity use this same weighted gradient operation.
On a fixed topology, coefficient refresh updates the fine pressure operator in
place on uniform grids and rebuilds only the coefficient-dependent sparse
matrix on mixed grids. The default tree-geometric preconditioner refreshes its
rediscretized coarse operators from the current coefficient. After an AMR
change it also retains any matching coarse-tree suffix. This is required for
robust FGMRES convergence as a large density jump moves; true residuals always
use the current fine operator.

### AMR criteria

The phase-interface and embedded-geometry fixed-band criteria use
`sqrt(eps(T))` for their presence tests. The curvature criterion enforces the
thesis hysteresis condition
`2*epsilon_coarsen < epsilon_refine`. `VorticityRefinementIndicator` implements
the block maximum of `h*abs(vorticity)` with the same hysteresis and explicit
minimum/maximum levels. The interpolation-error and curvature criteria also
provide explicit minimum/maximum levels. The two fixed-band criteria use one
exact target `level`, since separate bounds do not describe their purpose. All
five criteria accept the inclusive activation window `t_init` and `t_end`,
defaulting to `-Inf` and `Inf`. The Stokes-3 example combines the interface,
curvature, and vorticity criteria.

## Audit coverage

| Requirement | Implementation evidence |
|---|---|
| Level-set fractions | Adaptive root-solved height integration; affine, curved, scaled, enclosed-component, uniform and AMR tests |
| LHF/GHF/WY geometry | Method flags, planar exactness, circle curvature convergence, second-order symmetric-difference convergence at an inclined contact line, and coarse/fine geometric sampling |
| WY interface transport | Incoming-volume CFL, exact phase-volume flux partition, reversal convergence on uniform/fixed-mixed/dynamic AMR grids, and a conservative wall-tangent cut-cell translation/reversal study for both initial sweep directions |
| Centred phase scalars | Intrinsic storage, shared WY fluxes, `sqrt(eps)` recovery, exact one-functions, all public interpolants |
| Staggered scalars/momentum | Algebraically interpolated WY fluxes, CTU near interface, exact constants and momentum conservation, all public interpolants |
| Variable diffusion | Coupled stress operator, stress-location geometric viscosity tests, energy dissipation and manufactured convergence |
| Variable pressure | Positive variable coefficient, discrete duality/SPD/nullspace tests, variable-density projection tests |
| Balanced forces | CSF and GFM constant-curvature cancellation; horizontal and rotated hydrostatic cancellation at density ratio 1000 |
| AMR transfer | Geometric PLIC refinement, conservative restriction, intrinsic scalar conservation/convergence |
| Fixed geometry | PLIC solid convergence, exact filled-solid area, cut projection duality, convective energy conservation, diffusion symmetry, tree-MG covered-cell constraints, and AMR physical-volume conservation |
| Embedded interfaces | GHF contact-angle curvature and reconstruction on a sloped wall, no-slip/free-slip coupled diffusion, combined phase/solid projection, and an AMR sloped sessile-drop step |
| Performance | Allocation-scaling tests and `benchmark/two_phase_extension.jl` comparisons for transport, reconstruction, momentum, diffusion, and Poisson kernels |
| End-to-end validation | Reverse vortex, capillary/gravity-wave, floating-droplet, sessile-droplet, and Re=100 cylinder refinement studies; Stokes-3, LGPI, and sloped sessile examples with AMR |

The validation programs under `test/integration` are deliberately separate
from fast unit tests because each performs full mesh/time refinement studies.
CI runs the reverse-vortex, capillary/gravity-wave, floating-droplet, and
sessile-droplet suites as four independent jobs. Each job uses four Julia
threads and one BLAS thread. The ordinary package-test matrix retains a
single-thread run and adds a four-thread run; its scheduler regression asserts
that more than one worker actually executes a representative numerical loop.

### Transfer choices

Geometric refinement integrates the parent's reconstructed PLIC into children.
Restriction is the exact volume-weighted sum; a separate geometric operation
cannot improve that sum. Derived normals/curvature are reconstructed afterward.
Phase scalar refinement uses a limited linear intrinsic reconstruction followed
by a local phase-volume correction. It preserves constants and each phase
integral without storing a phase-weighted field. Coarsening computes a temporary
phase-volume-weighted average. Staggered intrinsic scalars use limited linear
face transfer without a global integral correction.

The one-velocity field retains the existing divergence-preserving face
transfer, prioritizing an accurate, projection-compatible velocity field during
regridding. Two-phase momentum transport itself is conservative because it uses
the interpolated centered WY phase fluxes; momentum is formed only in local
expressions and is not stored as a field.

## Level-set initialization: focused literature review

`InterfaceField(mesh, phi; ...)` initializes the negative phase `phi(x,y) <= 0`.
`phi` need not be a signed distance. Initialization is distinct from PLIC
reconstruction of the resulting fractions, and from AMR transfer.

| Approach | Accuracy/cost tradeoff | Decision |
|---|---|---|
| Cell-centre indicator or dense subcell counting | Cheap but discontinuous sampling error; expensive when fractions must be accurate enough for second derivatives | Use only as benchmark baselines |
| Piecewise-linear clipping/marching simplices | Exact for linear geometry; curvature introduces geometric error requiring subdivisions | Good robust low-order reference, insufficient as the sole curvature-test initializer |
| Implicit height integration | Root finding plus Gaussian quadrature resolves curved geometry directly | Selected for Cartesian cells |
| Divergence theorem and curved boundary quadrature | Reduces integration dimension; attractive on arbitrary polygons/polyhedra | Relevant existing local implementation and possible future generalization |
| Osculating-paraboloid integration | High-order geometric approximation, particularly useful for arbitrary polyhedra | More machinery than required for this 2-D Cartesian initializer |

[Saye (2015)](https://math.lbl.gov/~saye/96629.pdf) constructs high-order
quadrature by reducing implicit geometry to height graphs, using one-dimensional
root finding and Gaussian integration, recursively subdividing where needed.
His rigorous algorithm uses bounds to establish admissibility of a height
direction. The implementation here adopts dimension reduction and adaptive
quadrature, but **does not implement the complete interval-arithmetic algorithm**
and must not inherit its certified claims. [Algoim](https://algoim.github.io/)
is an implementation reference if certified/general polynomial geometry is
needed later.

[The initialisation of volume fractions for unstructured grids using implicit
surface definitions (2019)](https://doi.org/10.1016/j.compfluid.2018.10.021)
uses an implicit surface and numerical quadrature on arbitrary grids, and
emphasizes initialization accuracy for height-function curvature. The local
`GeometricVOF.jl/src/initialization.jl` similarly finds edge intersections and
adds root-solved height corrections to a polygonal area. Its edge-only pure-cell
shortcut can miss a closed component entirely within a cell; that shortcut is
not sufficient for the requested general level-set API.

[Kromer & Bothe (2021 preprint)](https://arxiv.org/abs/2111.01073) use local
osculating paraboloids and recursive divergence-theorem reduction for general
polyhedra, reporting third- to fourth-order spatial convergence. That approach
is attractive for future non-Cartesian cells, but not necessary here.

### Selected algorithm and explicit limits

- Test the centre, corners and edge midpoints. An optional known global
  Lipschitz constant supplies a rigorous pure-box exclusion test;
  `lipschitz=1` is appropriate for a true signed-distance function.
- Choose a locally monotone height direction and solve bracketed one-dimensional
  roots. Split the transverse integral at cell-edge crossings to remove clipping
  kinks before applying Gaussian quadrature.
- Compare an eight-point rule with two half-interval rules, subdividing until
  the requested fraction tolerance is reached. Subdivide boxes when no sampled
  height direction is valid.
- `min_depth` forces extra feature searching. With no supplied bound, pure-cell
  and monotonicity decisions assume smooth geometry resolved by the samples.
  No finite-sampling black-box method guarantees detection of arbitrarily small
  disconnected components or oscillations. A Lipschitz bound certifies the
  exclusion test, not the sampled monotonicity test or Gaussian error estimate.
- Invalid function values and exhausted refinement throw. Existing fractions
  are replaced only after every cell has initialized successfully.
- The transactional quadrature first removes endpoint roundoff at
  `64eps(T)`. The subsequent reconstruction applies the interface-wide
  `fraction_floor` to initialization, transport, and AMR-transferred fields.
  For embedded geometry, the threshold applies to the conditional phase
  fraction within the open part of each cell.
- Solid interfaces use the same default floor. Fixed-geometry stress tests
  that intentionally retain extremely small cut cells use
  `fraction_floor=eps(T)`. Zero is never accepted, so intrinsic scalar
  recovery remains protected from division by zero.
- Initialize each leaf using its actual physical rectangle. Comparing uniform,
  mixed-level and dynamically refined grids checks partition independence.

The high-order initializer is intentionally a setup operation. It does not
resample the original level set during time evolution or mesh adaptation.

### Constructor example

```julia
mesh = BlockTreeMesh(block_size=(32,32))
interface = InterfaceField(mesh, (x,y) -> hypot(x-0.5,y-0.5)-0.2;
                           lipschitz=1, atol=1e-10,
                           fraction_floor=sqrt(eps(Float64)))
volume = integrate(interface)
tracer = PhaseScalar(interface; liquid=1000, gas=1)
```

For a non-distance level set such as
`((x-0.5)/0.27)^2 + ((y-0.5)/0.17)^2 - 1`, omit `lipschitz` or provide a
valid bound for that particular function. Positive scaling does not change the
phase, but does scale the appropriate Lipschitz bound.

### Initialization measurements

Local CPU, Julia 1.12.7, warmed run, signed-distance circle of radius 0.21.
The high-order timings include reconstruction; counting timings measure only
sampling/integration. Reproduce with `benchmark/levelset_initialization.jl`.
These are single-run setup measurements, not claims of asymptotic timing.

| Cells | High-order time | Level-set evaluations | Circle volume error | 16×16 counting time | Counting volume error |
|---|---:|---:|---:|---:|---:|
| 16² | 0.95 ms | 31,247 | 8.3e-17 | 0.53 ms | 4.0e-5 |
| 32² | 1.79 ms | 65,978 | 5.6e-17 | 2.09 ms | 5.9e-6 |
| 64² | 3.77 ms | 134,018 | 5.6e-17 | 8.31 ms | 6.5e-6 |

The analytic circle happens to integrate near roundoff here; the requested
`atol` and quadrature error estimate remain the general accuracy controls.
Tests also cover affine cuts, ellipses, sinusoidal height graphs, complementary
phases, scaled level sets, enclosed subcell droplets and transactional failures.
