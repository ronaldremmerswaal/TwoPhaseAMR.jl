# Two-phase extension: numerical design and implementation record

This branch extends the existing Navier--Stokes solver with an optional
interface. Without an interface it follows the original single-phase path.
With an interface it uses compatible two-phase transport, variable material
operators, balanced forces, and the same pressure-projection API.

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
   projection weights. If necessary, neighbouring reconstructed centroids
   provide unit-weight points. Rank-deficient fits are rejected.
4. The final normal fallback is WY reconstruction: three-cell height sums and
   an orientation/filling-selected one-sided difference. It is not a
   volume-fraction-gradient normal. See Weymouth & Yue (2010), Section 2.1,
   [original article](https://doi.org/10.1016/j.jcp.2009.12.018).

The method flag is 0 (pure), 1 (LHF), 2 (GHF), or 3 (WY). Unavailable curvature
is `NaN`, never silently zero. Geometry tests must inspect these flags.
Height halos are sampled on the local level by conservative restriction and
PLIC subcell integration. Three synchronous passes refresh coarse–fine planes;
this finite iteration is a current accuracy limitation requiring convergence
checks at refinement boundaries. The approach follows the local-level sampling
principle in [Popinet (2009)](https://doi.org/10.1016/j.jcp.2009.04.042).

Wall-adjacent reconstruction accepts one contact angle or a
`(left, right, bottom, top)` tuple. Near-right angles use height Neumann data.
Acute and obtuse angles use the thesis quadratic construction with an unknown
contact location. The reconstruction flag is 4 in these cells. Sessile-drop
equilibrium remains an end-to-end validation requirement beyond the local
geometry tests.

### WY transport

The initial `c > 1/2` indicator stays frozen across all directional sweeps.
Each sweep reconstructs the updated interface. Owned face segments carry one
signed time-integrated liquid volume; gas volume is total minus liquid.

The WY CFL is total incoming volume divided by cell volume, at most `1/2`.
Outgoing volume and donor extrusion lengths are also checked; the extrusion
check protects coarse-cell subfaces. The additional constraint can be more
restrictive than the thesis's uniform-grid bound.

Centred phase scalars reuse these fluxes and upwind intrinsic phase values.
The dilation concentrations are frozen for the step. Staggered phase scalars
and one-velocity momentum use the existing algebraic dual-flux interpolant.
Every staggered bulk interpolant (`Upwind`, `Centred`, `LW`, `Fromm`, `Minmod`,
`MC`, `VanLeer`, and `Superbee`) is supported. The modified interpolant switches
to term-wise CTU upwinding when either adjacent predicted phase fraction is
below `ctu_threshold=0.5`, following `pt1_momentum.tex`. Their dilation source
uses `interpolate(chi * directional_divergence)`, not a product of separately
interpolated factors. Division during intrinsic-value recovery uses a finite
`fraction_floor=sqrt(eps(T))`; this numerical floor does not modify the stored
geometric volume fraction.

Current boundary support is impermeable or periodic interface transport.
Staggered phase transport includes physical boundary half-volumes. Topology
dependent workspaces are rebuilt after mesh adaptation. Allocation scaling of
transport, variable diffusion, variable-coefficient projection, and balanced
force kernels is checked on uniform and mixed-level meshes.

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
- Initialize each leaf using its actual physical rectangle. Comparing uniform,
  mixed-level and dynamically refined grids checks partition independence.

The high-order initializer is intentionally a setup operation. It does not
resample the original level set during time evolution or mesh adaptation.

### Constructor example

```julia
mesh = BlockTreeMesh(block_size=(32,32))
interface = InterfaceField(mesh, (x,y) -> hypot(x-0.5,y-0.5)-0.2;
                           lipschitz=1, atol=1e-10)
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
