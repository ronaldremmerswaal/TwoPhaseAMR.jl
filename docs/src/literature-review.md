# Discrete-layout review and required invariants

This is the pre-implementation review for staggered transport and projection.
It records constraints, not claims that milestones 2--5 are complete.

## Selected layout

Pressure, scalar mass, and volume fraction are cell averages. Normal velocity,
volume flux, mass flux, and normal momentum use directional face locations.
At a coarse--fine boundary, the physical face is partitioned at the fine
tangential resolution. Each fine segment owns one normal flux degree of
freedom; the coarse face flux is its area-weighted sum and is not an
independent unknown. This makes one geometric face segment contribute exactly
once to each adjacent divergence.

For the current centred milestone, `InteriorFace` already implements this
single-owner rule. The staggered milestone must retain it while introducing
dual, face-centred control volumes.

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

## Divergence, gradient, and interface modes

The pressure gradient must be defined from the divergence adjoint, including
coarse--fine metric weights, rather than discretized independently. With
periodic or all-Neumann pressure boundaries, constants form the pressure
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

- [Olshanskii et al., 2013](https://doi.org/10.1016/j.compfluid.2013.04.027)
- [Gerya et al., 2013](https://doi.org/10.1002/ggge.20078)

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

Sources:

- [Almgren et al., 1998](https://doi.org/10.1006/jcph.1998.5890)
- [Popinet, 2003](https://doi.org/10.1016/S0021-9991(03)00298-5)

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
