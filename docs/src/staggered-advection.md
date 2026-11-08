# Staggered advection

## Discrete construction

`FaceField{Axis}` contains one scalar per owned face segment normal to `Axis`.
A staggered control volume surrounds each such face. Its boundary fluxes are
not sampled or reconstructed independently: they are factored algebraically
from the already-owned primal face-normal fluxes. For a uniform Cartesian
grid this factorization reduces to the ordinary area-weighted average, for
example,

```math
F^u_{i,j+1/2} = \frac{\Delta x}{2}
                 \left(v_{i-1/2,j+1/2}+v_{i+1/2,j+1/2}\right).
```

This follows the mass/momentum-compatible construction described by
[Remmerswaal (2023, Chapter 6)](https://doi.org/10.33612/diss.562133319).
Every dual face is stored once and contributes equal-and-opposite
integrated fluxes to its two neighbouring staggered volumes, so each component
of linear momentum is conserved to roundoff on a periodic mesh.

## Nonconforming AMR construction

At a coarse--fine interface the coarse geometrical side is partitioned into
fine owned face segments; there is no additional coarse velocity unknown. For
a segment ``f`` adjacent to cell ``c`` and normal to component direction
``a``, define

```math
\alpha_{cf} = \frac{|f|}{|\partial c|_{\mathrm{side},a}}.
```

Thus ``\alpha=1`` on an unsplit side and the segment fractions on every cell
side sum to one. The dual volume and volume-weighted cell-to-face interpolant
are

```math
W_f = \frac12\sum_{c\sim f}\alpha_{cf}V_c,
\qquad
(I_C d)_f = W_f^{-1}\frac12
             \sum_{c\sim f}\alpha_{cf}V_c d_c.
```

Let ``B`` be the oriented cell--face incidence matrix and ``m=A u`` the
integrated primal flux. Workspace construction makes a local conservative
factorization of

```math
R = \frac12 P_\alpha^T B,
```

where ``(P_\alpha)_{cf}=\alpha_{cf}``. Normal fluxes are routed by local stars
centred on the transported face. Transverse fluxes use a product-weight
transport plan between the matching sides of their two cells. This may create
a wider stencil at a hanging face, but its row and column marginals are exact.
The resulting dual incidence and flux interpolant satisfy

```math
D_s I_F m = W^{-1}Rm = I_C D u
```

to roundoff. The construction reduces exactly to the existing two-flux MAC
schedule on any uniform level. It is built only when the mesh changes, so the
hot kernel traverses compact connection and coefficient arrays without tree
queries or allocations.

At physical walls, the boundary half-volume connections also include a
transverse primal flux when a fine interior face meets a coarse wall face.
Boundary and interior terms use their own solid-open apertures in embedded
geometry. These wall connections contribute to both the convective flux and
the staggered CFL bound. With cut and hanging cells together, the compatible
identity uses the side-fraction-weighted integrated cell fluxes; this need
not equal interpolation by the geometric cut-strip volume.

For an oriented dual-face flux `F`, neighbouring states `q⁻` and `q⁺`, normal
speed `a`, spacing `h`, and time step `Δt`, the upwind and centred face states
are

```math
q_\mathrm{up} = \begin{cases}q^- & F \ge 0,\\q^+ & F < 0,\end{cases}
\qquad
q_\mathrm{cen} = \frac{q^-+q^+}{2},
```

For one coordinate direction, the Lax--Wendroff face state is

```math
q_\mathrm{LW} = \frac{q^-+q^+}{2}
                 - \frac{\Delta t\,a}{2h}(q^+-q^-),
```

where `Δt` is the current directional substep.

The AMR factorization can contribute x- and y-oriented primal fluxes to the
same pair of staggered unknowns. Schedule construction now groups a connection
by `(minus, plus, coordinate direction)`, rather than only `(minus, plus)`.
This leaves the total factored flux and ``D_s I_F=I_C D`` unchanged, but makes
an actual coordinate sweep possible. On a hanging-face connection the local
Courant number is ``|\Delta t F_g|/\min(W_-,W_+)``.

The complete LW update has two full sweeps. Their order alternates between
successive time steps:

```math
q^{(1)} = L_x(\Delta t)q^n,\quad
q^{n+1} = L_y(\Delta t)q^{(1)},
\qquad\text{then}\qquad
q^{(1)} = L_y(\Delta t)q^n,\quad
q^{n+1} = L_x(\Delta t)q^{(1)}.
```

Every sweep consumes the field produced by the preceding sweep. Within a
velocity sweep, both component tendencies are evaluated from one input vector
and are published together; this is a coordinate split, not a split between
velocity components. Alternating the order cancels the leading Lie-splitting
error over a pair of equal steps and retains the second-order Taylor--Green
regression without a third Strang substep.

For variable transporting velocity, classical constant-coefficient LW misses
the ``q\,\partial_a u_a`` part of the directional predictor. Each face state
therefore also contains

```math
-\frac{\Delta t}{4}
 \left(q^- (D_a u_a)^- + q^+ (D_a u_a)^+\right).
```

The x and y directional divergences cancel for a discretely divergence-free
velocity only after both sweeps; including them separately is necessary for
second-order split transport. The Navier--Stokes step freezes the transporting
mass flux at the old velocity over the two sweeps, matching the previous
explicit treatment.

For positive `dt`, `convection!` returns the effective rate of the workspace's
next two-sweep update without advancing its order flag, so `advected - dt *
destination` equals the next result of `advect!`. At `dt == 0`, it evaluates
the limiting spatial operator directly. The interface scheme remains an
explicit API choice:

```julia
convection!(rate, q, velocity, dt, workspace)
advect!(q, velocity, dt, workspace;
        scheme=LaxWendroff(), amr_interface_scheme=LaxWendroff())
```

The bulk `scheme` defaults to `LaxWendroff()`. When `amr_interface_scheme` is omitted,
`default_amr_interface_scheme(scheme)` applies the following policy:

- `Upwind()` uses Upwind everywhere;
- `Centred()` and `LaxWendroff()` use `MonotonizedCentral()` at non-regular
  AMR connections; and
- Fromm and the limited three-point schemes use themselves on those connections.

Bounded phase transport also applies the interface scheme to connections
incident to a physical boundary unknown.

Either keyword accepts the corresponding lower-case symbols. Interface-only
Upwind is rejected; use `scheme=Upwind()` when first-order upwinding is wanted
throughout. For example, the decaying-turbulence showcase's effective default
is

```julia
scheme=LaxWendroff(), amr_interface_scheme=MonotonizedCentral()
```

so that non-regular AMR connections use the nonlinear reconstruction.
An explicit compatible interface keyword has no numerical effect on a uniform
fully periodic mesh.

## Three-point interface fluxes

For an oriented connection, let ``q_{UU}``, ``q_U``, and ``q_D`` denote the
second-upstream, upwind, and downwind values. Workspace construction follows
the connection graph into the upstream half-space and selects the candidate
closest to the geometrically reflected edge. The smoothness ratio includes
the projected spacing ratio,

```math
r = \frac{|x_D-x_U|}{|x_U-x_{UU}|}
    \frac{q_U-q_{UU}}{q_D-q_U},
```

so a linear profile has ``r=1`` across a two-to-one change in spacing. The
high-resolution state blends the conservative upwind and LW connection
fluxes,

```math
F_\phi = F_\mathrm{up} + \phi(r)
         \left(F_\mathrm{LW}-F_\mathrm{up}\right).
```

The implemented standard three-point functions are

```math
\begin{aligned}
\phi_\mathrm{Fromm}(r) &= \tfrac12(1+r),\\
\phi_\mathrm{minmod}(r) &= \max(0,\min(1,r)),\\
\phi_\mathrm{MC}(r) &= \max(0,\min(2r,\tfrac12(1+r),2)),\\
\phi_\mathrm{vanLeer}(r) &= \frac{r+|r|}{1+|r|},\\
\phi_\mathrm{superbee}(r) &=
    \max(0,\min(2r,1),\min(r,2)).
\end{aligned}
```

These are the classical flux-limiter forms catalogued by
[Sweby (1984)](https://doi.org/10.1137/0721062); the spacing correction follows
the nonuniform-grid consistency analysis of
[Zeng (2016)](https://doi.org/10.1137/140970185).

Fromm is the unrestricted average of the Lax--Wendroff and Beam--Warming
corrections; it is a larger-stencil control, not a
TVD limiter. Minmod, MC, van Leer, and superbee set the correction to zero at
a local extremum. A small subset of factored hanging-face edges joins degrees
of freedom at the same transport coordinate and therefore has no meaningful
upstream half-space. Those edges explicitly fall back to the upwind state
rather than constructing a fictitious slope. Reconstruction metadata is
precomputed in the workspace; the flux loop adds one indexed read and does not
allocate.

These classical limiter formulas do not establish a global TVD theorem for
the multidimensional branched AMR graph. They do, however, address the
specific interface mechanism without sacrificing conservation: the selected
state still multiplies one connection flux which is added to one dual volume
and subtracted from the other.

The upwind state is monotone under the returned multidimensional CFL bound.
Using Upwind only on refinement-interface connections is not recommended: the
isolated first-order closure creates a larger consistency error than a globally
upwind discretization. Use one of the limited three-point schemes at refinement
interfaces, and select a limited bulk `scheme` as well when monotonicity is more
important than the lower dissipation of Lax--Wendroff.

The public scheme values are `Upwind()`, `Centred()`, `LaxWendroff()`,
`Fromm()`, `Minmod()`, `MonotonizedCentral()`, `VanLeer()`, and `Superbee()`.
For Lax--Wendroff and the three-point
schemes, `convection!` evaluates the effective split-step rate and `advect!`
applies the split conservative update.
A `StaggeredVelocityAdvectionWorkspace` updates both velocity components from
the same state within each coordinate sweep.

## Skew-symmetry invariant

Let `D_c` be the cell divergence and `I_F` the algebraic flux interpolation.
The dual divergence satisfies

```math
D_f I_F = I_C D_c.
```

When both `scheme` and `amr_interface_scheme` are centred, the integrated
convection matrix satisfies the stronger identity

```math
W C(u) + C(u)^T W = \operatorname{diag}(W I_C D_c u),
\qquad
2\operatorname{diag}(C(u))=I_C D_c u.
```

The primary divergence is therefore recovered directly from the
diagonal of convection, including at hanging faces. Summation by parts on a
periodic mesh gives

```math
\langle p,C(u)q\rangle_V = -\langle C(u)p,q\rangle_V
\quad\text{when}\quad D_c u=0.
```

This identity gives exact semi-discrete momentum conservation and weighted
skew-symmetry for a divergence-free carrier. It is a spatial property; a
forward Euler step with the centred operator does not by itself preserve
kinetic energy in time.

## Current scope

The dual schedule supports uniform and 2:1-balanced mixed-level meshes with
periodic or physical boundaries. Wall conditions (`NoSlip()` and `FreeSlip()`)
are impermeable; `VelocityBoundaryConditions` can instead prescribe inflow or
extrapolating outflow on individual sides. Truncated dual volumes are assembled
explicitly. Viscous tangential wall data is handled separately. Refinement
subcycling and
local time stepping remain future synchronization concerns; dynamic regridding
currently uses one global time step and rebuilds the advection workspace after
each accepted mesh change.

## References

- R. Remmerswaal, [*Numerical modelling of variability in liquid
  impacts*](https://doi.org/10.33612/diss.562133319), Chapter 6, PhD thesis,
  University of Groningen (2023).
- R. A. Remmerswaal and A. E. P. Veldman, [*Towards a sharp, structure
  preserving two-velocity model for two-phase flow: transport of mass and
  momentum*](https://arxiv.org/abs/2209.14934).
- P. K. Sweby, [*High resolution schemes using flux limiters for hyperbolic
  conservation laws*](https://doi.org/10.1137/0721062), SIAM Journal on
  Numerical Analysis 21 (1984), 995--1011.
- X. Zeng, [*A general approach to enhance slope limiters in MUSCL schemes on
  nonuniform rectilinear grids*](https://doi.org/10.1137/140970185), SIAM
  Journal on Scientific Computing 38 (2016), A789--A813.
- M. J. Berger, M. J. Aftosmis, and S. M. Murman, [*Analysis of slope
  limiters on irregular grids*](https://doi.org/10.2514/6.2005-490), AIAA
  Paper 2005-0490.
- S. T. Zalesak, [*Fully multidimensional flux-corrected transport algorithms
  for fluids*](https://doi.org/10.1016/0021-9991(79)90051-2), Journal of
  Computational Physics 31 (1979), 335--362.
- R. M. J. Kramer, C. Pantano, and D. I. Pullin, [*Nondissipative and
  energy-stable high-order finite-difference interface schemes for 2-D patch
  refined grids*](https://doi.org/10.1016/j.jcp.2009.04.010), Journal of
  Computational Physics 228 (2009), 5280--5297.
