# Discrete-layout review and required invariants

This review defines the invariants for staggered transport, projection, and
viscosity. The implemented slices cover periodic and impermeable-wall
staggered advection, composite pressure projection, staggered diffusion, and
fixed-tree single-phase flow on uniform and mixed-level trees. The remaining
constraints are not claims that the two-phase milestones are complete.

## Selected layout

Pressure, scalar mass, and volume fraction are cell averages. Normal velocity,
volume flux, mass flux, and normal momentum use directional face locations.
At a coarse--fine boundary, the physical face is partitioned at the fine
tangential resolution. Each fine segment owns one normal flux degree of
freedom; the coarse face flux is its area-weighted sum and is not an
independent unknown. This makes one geometric face segment contribute exactly
once to each adjacent divergence.

`InteriorFace` implements this single-owner rule. `FaceField{Axis}` and its
dual schedule retain it for staggered transport. A coarse cell's
share of a split segment is its area fraction on that side, so all shares sum
to one and the directional dual volumes partition the domain.

The non-negotiable algebraic invariants are:

1. `D_f I_F = I_C D_c`: staggered divergence of interpolated centred flux is
   the interpolation of centred divergence.
2. `G = -M_f⁻¹ Dᵀ M_c`: pressure gradient and divergence are negative
   adjoints in their volume-weighted inner products.
3. Restriction preserves integrals; prolongation preserves constants and its
   face-flux variant preserves the coarse divergence.
4. A coarse flux equals the sum of its fine face-segment fluxes before either
   level updates a conserved quantity. Refluxing applies only the mismatch
   between time-integrated coarse and fine flux registers.
5. Regridding and synchronization commute with the centred-to-staggered mass
   interpolation, to roundoff.

## Centred-to-staggered transport

Remmerswaal, Chapter 6, Section 6.4, compares three constructions. Computing
new donating regions on staggered control-volume faces evolves a geometrically
defined staggered mass that can drift from the interpolation of centred mass;
resetting it loses momentum conservation. The selected construction instead
sets the staggered mass/volume flux to an area-weighted interpolation of the
already computed centred flux. Its divergence identity makes the staggered
mass equation the algebraic image of the centred one, yielding exact linear
momentum conservation without another subgrid VOF calculation.

Near a phase interface, division of staggered momentum by a nearly empty phase
mass can be unbounded. The same thesis derives a CTU interpolant that is bounded
when donating regions have neither overlap nor transit errors, then recommends
a modified interpolant that uses the less dissipative Lax--Wendroff form away
from endangered control volumes. This switch must be driven by the *updated*
staggered phase mass; it cannot be an unrelated limiter.

Primary source: R. Remmerswaal, *Numerical Modelling of Variability in Liquid
Impacts* (2023), Chapter 6, Sections 6.4--6.4.1, locally available at
`/home/ronald/Documents/phd_project/thesis/thesis.pdf` and in
`pt1_momentum.tex`.

The implementation applies the same argument to nonconforming topology. If
``P_\alpha`` contains the coarse/fine side fractions and ``B`` is the primal
incidence, the integrated dual divergence is factored from
``R=P_\alpha^T B/2``. Normal primal fluxes use face-centred local stars;
transverse fluxes use a conservative product-weight transport plan between
matching cell sides. This is a somewhat wider stencil than a compact virtual
velocity closure, but it gives the exact operator identity
``D_s I_F=I_C D`` and exact momentum conservation. For centred face states it
also gives ``2 diag(C)=I_C D``. The interface state is independently
selectable. The public default uses LW on regular and non-regular AMR
connections because it retains roughly second-order mixed-level convergence;
interface upwind remains an explicit robustness option. The all-centred
operator is used when the diagonal/skew identity is required across the
complete mesh.

Sources:

- [Remmerswaal and Veldman, mass and momentum transport](https://arxiv.org/abs/2209.14934)
- [Verstappen and Dröge, 2005](https://doi.org/10.1016/j.crme.2004.09.021)

## Van der Plas interface options

Chapter 4 of van der Plas's thesis is directly relevant to this design. It
first distinguishes a *continuity interface*, coinciding with primal
mass-volume faces, from the component-dependent *staggered interfaces* of the
momentum volumes. It then decomposes a Cartesian refinement transition into
abrupt stretching normal to the interface and cell splitting tangential to
it. This is a useful classification because the former belongs in metric
coefficients, whereas the latter needs an interpolation or a modified
stencil.

The thesis compares non-overlapping composite grids, partly overlapping
grids, and full multilevel grids, and selects a non-overlapping composite grid
to keep the effective operator and its errors inspectable. Its convection
study reaches four conclusions used here:

1. mass conservation is mandatory, while energy conservation may be relaxed
   at refinement interfaces when robustness or stencil quality requires it;
2. conservative momentum transport requires one interface flux, with a coarse
   momentum flux equal to the sum of its fine fluxes;
3. direct treatment of abrupt normal stretching is preferable to hiding that
   metric jump in a virtual uniform stencil; and
4. simple averaging of the transport velocity and linear tangential
   interpolation are at least first-order locally, while extra dissipation can
   be necessary to suppress reflection of under-resolved waves.

Van der Plas's implemented convection closure did not itself enforce the
momentum-flux summation property; the thesis identifies it as a straightforward
future extension. The present factorization performs that extension and adds
Remmerswaal's exact divergence compatibility. It does not copy pressure
scheme F1 into convection: F1 is one of the thesis's divergence/gradient
closures, not a momentum-flux construction.

Primary source: [van der Plas, *Local grid refinement for free-surface flow
simulations*, 2017](https://research.rug.nl/en/publications/local-grid-refinement-for-free-surface-flow-simulations/),
especially Sections 4.1, 4.2, and 4.5.

## Limiting and energy-stable interface closures

An ordinary uniform-grid slope limiter should not be transplanted unchanged
to a two-to-one refinement transition. Berger, Aftosmis, and Murman show that
obvious limiter generalizations on irregular grids can be either nonlinear
inconsistent or fall outside the TVD region. They derive mesh-ratio-aware,
linearity-preserving limiters and also explain why purely face-based limiting
is ineffective on general multidimensional irregular meshes. Zeng develops a
generalized Sweby region for nonuniform rectilinear grids and gives enhanced
minmod, superbee, MC, van Leer, and van Albada limiters. In particular, the
enhanced MC limiter is TVD and linearity preserving for the local mesh ratios;
it still assumes a reconstruct-evolve-project stencil with defined upstream
and downstream cells.

The AMR staggered flux schedule here is an algebraic conservative graph. Near
a split transverse side, one connection can combine several primal fluxes and
there is no unique second-upwind degree of freedom. Zalesak's multidimensional
flux-corrected transport is consequently a more natural limiting framework:
write LW as an antidiffusive correction to a monotone upwind update, sum each
node's admissible positive and negative increments, and apply one symmetric
correction factor to each edge. Conservation then follows from the same
equal-and-opposite graph flux used by the unlimited scheme. A local
maximum-principle proof, however, requires the complete low-order update to be
monotone. Using bulk LW and limiting only refinement-edge corrections does not
meet that hypothesis; the numerical prototype confirms that it fails to stop
long-time amplification. Full-graph FCT is bounded but is a different bulk
scheme rather than an AMR-interface selector.

Kramer, Pantano, and Pullin address the refinement interface itself rather
than limiting a uniform-grid formula. They construct wider finite-difference
closures for patch-refined grids by imposing summation by parts, obtaining
nondissipative energy stability for scalar advection. Their two-dimensional
analysis treats edge, concave-corner, and convex-corner geometries and reports
that local order must be reduced at corners. Although their node-based
operators cannot be copied directly into the present finite-volume flux graph,
their constrained-coefficient construction is the strongest literature match
for a future low-dissipation AMR closure.

Sources:

- [Berger, Aftosmis, and Murman, *Analysis of Slope Limiters on Irregular Grids*](https://www.nas.nasa.gov/assets/nas/pdf/papers/nas-05-007.pdf)
- [Zeng, *A general approach to enhance slope limiters on non-uniform rectilinear grids*](https://arxiv.org/abs/1301.0967)
- [Zalesak, *Fully multidimensional flux-corrected transport algorithms for fluids*](https://doi.org/10.1016/0021-9991(79)90051-2)
- [Kramer, Pantano, and Pullin, 2009](https://doi.org/10.1016/j.jcp.2009.04.010)

## Divergence, gradient, and interface modes

The pressure gradient is defined from the divergence adjoint, including
coarse--fine metric weights, rather than discretized independently. The
implemented construction keeps
`D = V⁻¹ B A` as the primary cellwise Gauss balance and uses a sparse SPD
inverse face Riesz map `K` to define `G = -K A Bᵀ`. Thus
`-V D G = B A K A Bᵀ` is symmetric positive semidefinite. `K` is diagonal on a
conforming Cartesian mesh and uses affine-exact vertex-patch couplings only
where isotropic refinement makes the cell-centre connection oblique. The
current mixed-level construction is deliberately gated to isotropic cells;
stretched interfaces require a tensor-valued patch metric. See
[Compatible pressure gradient](pressure-gradient.md) for the construction and
homogeneous-Neumann physical-wall closure.

This differs deliberately from the fully adjoint F1 scheme in van der Plas's
2017 thesis. F1 was the only fully adjoint interface scheme that passed all of
that thesis's test cases, and its larger stencil suppressed interface modes
better than compact fine-velocity alternatives. However, F1 obtains its
divergence by applying the adjoint of a linear pressure interpolation. The
resulting tangential flux redistribution is not the desired cell-local Gauss
operator and is only zeroth-order consistent near refinement corners. The
implemented Riesz map retains wide, symmetric coupling without placing that
correction in the primal divergence.

With periodic or all-Neumann pressure boundaries, constants form the pressure
operator null space. The composite solve must either enforce volume-weighted
zero-mean pressure or pin one explicit degree of freedom, and compatibility
requires a zero-mean right-hand side. The projected divergence must be measured
with the same composite `D` used to assemble `L = D β G`.

Olshanskii, Terekhov, and Vassilevski show that a straightforward staggered
octree extension can admit oscillatory velocity modes at fine-to-coarse
interfaces. Their local low-pass filter reduces those modes, but filtering is
not accepted here as the first remedy because it can alter momentum and
divergence. Tests must first probe the singular vectors/null space of the
coarse--fine interpolation and composite operator, with alternating tangential
face data. Any stabilization must state which conservation and adjoint
identities it preserves.

Gerya, May, and Duretz enforce stress-conservative constraints at hanging
velocity nodes, obtaining a compact stencil without spurious pressure
oscillations even across discontinuous viscosity. This supports treating
hanging-face constraints as part of the operator definition, not as halo
post-processing.

Sources:

- [van der Plas, 2017](https://research.rug.nl/en/publications/local-grid-refinement-for-free-surface-flow-simulations/)
- [Bonelle and Ern, 2014](https://doi.org/10.1051/m2an/2013104)
- [Olshanskii et al., 2013](https://doi.org/10.1016/j.compfluid.2013.04.027)
- [Gerya et al., 2013](https://doi.org/10.1002/ggge.20078)

## Staggered viscosity at refinement interfaces

Van der Plas separates a refinement transition into abrupt stretching normal
to the interface and splitting tangential to it. For normal stretching, the
conservative unequal-spacing second difference is the natural stable core;
its local truncation error is first order at an abrupt two-to-one jump. For
tangential splitting, linear guard interpolation is robust but
over-diffusive and inconsistent, whereas the quadratic correction restores
consistency at the cost of a wider stencil and possible local anti-diffusion.
The lesson used here is to keep a symmetric positive-semidefinite conservative
core and add consistency only in a narrow interface band, then test the
corrected spectrum rather than assume energy stability.

The implemented component Laplacian follows that split. Unique dual-grid
connections form a quadratic energy and hence a symmetric stable stiffness.
Within two graph rings of a hanging-face connection, a three-ring weighted
least-squares stencil reproduces all quadratic polynomials. The corrected
operator is mildly nonsymmetric, but its tested eigenvalues remain in the
closed left half-plane and its manufactured mixed-level error converges at
better than order 1.5 on the tested sequence. This is intentionally a wider
stencil. It is not described as exactly energy dissipative after correction.

At a physical wall, normal velocity is constrained to zero. `NoSlip()` adds
the tangential half-cell wall flux (and permits a time-dependent moving-wall
value); `FreeSlip()` omits it to impose zero tangential shear. The later
two-phase extension must replace scalar viscosity multiplication with
stress-conservative face/edge coefficients. Gerya, May, and Duretz provide the
relevant hanging-node precedent for strongly discontinuous viscosity.

The implicit midpoint matrix retains the corrected operator. Its two
component-wise geometric/Galerkin hierarchies use transfers smoothed by the
symmetric stable core but do not symmetrize the corrected Galerkin matrices.
This permits the package multigrid algorithm to solve the actual viscous
equation; direct LU is restricted to the coarsest grids.

Sources:

- [van der Plas, 2017](https://research.rug.nl/en/publications/local-grid-refinement-for-free-surface-flow-simulations/)
- [Gerya, May, and Duretz, 2013](https://doi.org/10.1002/ggge.20078)

## AMR synchronization and projection

Almgren et al. use conservative scalar advection, recursively advance nested
levels, and synchronize after levels reach the same time. Even though this
package begins with one global step, its interface contracts should already
separate restriction (state replacement), flux-register reflux (conservation),
and a composite synchronization projection (divergence correction). Combining
them into a generic halo fill obscures which invariant is being restored.

Popinet's tree solver uses a multilevel Poisson solve and distinguishes an
exact face-centred MAC projection from an approximate cell-centred projection.
That distinction reinforces the selected face-normal primary velocity and the
requirement that block partitioning not change the composite operator.

The implemented pressure solver follows the composite-grid multigrid lineage
of FAC and the block-tree realization demonstrated by Afivo. Complete dyadic
cell siblings are merged geometrically, the transfer is smoothed with the
diffusion stiffness, and coarse matrices use Galerkin products. This is
important for the present wider interface metric: rediscretizing a compact
five-point Laplacian on coarse levels would discard the operator that was
chosen to preserve divergence/gradient duality. AMReX's separation between a
multilevel solver and a canonical `a*alpha - b*div(beta*grad)` operator is the
model for keeping this hierarchy reusable for shifted implicit diffusion and
future variable coefficients. See [Geometric multigrid and pressure
projection](poisson.md).

Sources:

- [Almgren et al., 1998](https://doi.org/10.1006/jcph.1998.5890)
- [Popinet, 2003](https://doi.org/10.1016/S0021-9991(03)00298-5)
- [McCormick and Thomas, 1986](https://doi.org/10.1090/S0025-5718-1986-0829618-X)
- [Teunissen and Ebert, 2018](https://doi.org/10.1016/j.cpc.2018.06.018)
- [Zhang et al., 2021](https://doi.org/10.1177/10943420211022811)

## Variable coefficients and two-phase forces

Density must be derived from the bounded volume fraction with one stated time
centering. The pressure projection uses a face coefficient such as `1/ρ`;
its coarse--fine restriction must preserve normal pressure flux and symmetry.
Viscous stresses need conservative face/edge coefficients suited to
discontinuous viscosity; naive arithmetic averaging is not assumed valid.

Surface tension must enter through a balanced-force pair: curvature/pressure
jump and pressure gradient use compatible locations and the identical
interface geometry. Hydrostatic and static-drop equilibria are required before
capillary-wave or rising-bubble cases. Gravity must use the same density and
face interpolation as the projection to avoid equilibrium drift.

## GeometricVOF compatibility gate

The local `GeometricVOF.jl` API currently exposes `Meshes.jl` polygons and
segments, fixed-capacity `StaticNgon` workspaces, in-place clipping and
intersection, LVIRA-family reconstruction, moments, and 2-D donating regions.
That is sufficient to prototype a block-cell adapter without a global
unstructured mesh and without hot-loop geometry allocation. Before milestone
4, the prototype must prove:

- block cell and face geometries can be views or stack/static values;
- clipping, reconstruction, intersection, and donating-region calls are
  allocation-free after workspace creation;
- a coarse--fine face uses the same owned face segments as centred transport;
  and
- 3-D is not inferred from the working 2-D donating-region API.

If any item fails, development stops at the gate defined in `DEVELOPMENT.md`.

Source: [GeometricVOF.jl](https://github.com/ronaldremmerswaal/GeometricVOF.jl).
