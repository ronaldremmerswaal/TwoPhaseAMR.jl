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

This is the centred-to-staggered flux construction selected in the literature
review. Every dual face is stored once and contributes equal-and-opposite
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

where `Δt` is the current directional substep. The old implementation added a
tangential donor correction and accumulated both coordinate directions from
the same old field. It was consequently unsplit.

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
        scheme=LW(), amr_interface_scheme=LW())
```

The bulk `scheme` defaults to `LW()`, and `amr_interface_scheme` inherits its
value when omitted. The no-keyword call therefore uses LW throughout. Either
can be `Upwind()`, `Centred()`, or `LW()` (and accepts the corresponding
symbols). Pass `amr_interface_scheme=Upwind()` explicitly for the more
dissipative hybrid closure. The interface keyword has no numerical effect on
a uniform mesh.

The upwind state is monotone under the returned multidimensional CFL bound.
At CFL 0.8 on the fixed level-1/level-2 test tree, the one-step spectral radii
after directional splitting are:

| Component placement / sweep order | Uniform LW | AMR LW/interface upwind | AMR LW throughout |
|---:|---:|---:|---:|
| x / x-y | 0.999999999999998 | 1.000000000000001 | 1.000000000000002 |
| x / y-x | 1.000000000000004 | 0.999999999999997 | 0.999999999999996 |
| y / x-y | 1.000000000000003 | 1.000000000000004 | 0.999999999999996 |
| y / y-x | 1.000000000000001 | 1.000000000000003 | 1.000000000000004 |

The neutral constant mode makes one the expected radius. The former AMR LW
values 1.0035--1.0036 were therefore caused by the unsplit mixed-axis
connection, not by an unavoidable refinement-interface instability.

The robustness switch is not accuracy-neutral. For the smooth manufactured
operator on successively finer copies of the same mixed-level tree:

| Method | Errors for block sizes 4, 8, 16 | Orders |
|---|---:|---:|
| LW / interface upwind | 0.3140, 0.2176, 0.1541 | 0.529, 0.498 |
| LW throughout | 0.1172, 0.03614, 0.01246 | 1.697, 1.536 |

The hybrid loss is the expected consistency defect of changing isolated
connections, rather than an error in the upwind sign. For smooth data, the
upwind face state differs from LW by ``O(h)``. A dual face has measure
``O(h)``, so this creates an ``O(h^2)`` integrated-flux error. Dividing by the
adjacent ``O(h^2)`` dual volume leaves an ``O(1)`` pointwise truncation error.
Interface-adjacent volumes occupy an ``O(h)`` fraction of the domain, hence

```math
\|e\|_{L^2} = O(1)\sqrt{O(h)}=O(h^{1/2}).
```

The benchmark confirms each factor: the hybrid interface RMS error stays near
0.45 while its volume fraction halves from 0.236 to 0.118 between the last two
meshes. Globally upwind is first-order because corresponding biases occur on
opposite faces and their leading terms cancel in the divergence; changing only
the interface face removes that cancellation.

## Interface-limiter decision

The requested larger-stencil limiter was conditional on directional splitting
not improving the LW behavior. Splitting removes the measured amplification,
so Monotonized Central and its TVD relatives have not been added to the public
API in this change. They solve a different problem--monotonicity near
under-resolved gradients--and would add nonlinear cost to the smooth-flow
default.

There is also no correct two-state MC shortcut at a hanging interface. MC
needs a second upstream value. At a two-to-one spacing jump, ordinary
uniform-grid MC is not linearity preserving; Zeng's enhanced limiter includes
the neighboring mesh ratios. A factored transverse AMR connection may branch
to several upstream degrees of freedom, so a proper implementation needs a
geometric patch reconstruction and a nonuniform limiter. If bounded transport
becomes a requirement, the defensible next steps are enhanced MC over such a
patch or full-graph flux-corrected transport. Limiting only interface
corrections cannot provide the maximum-principle premise while bulk LW remains
the high-order update.

### Historical unsplit experiments

The following blend and FCT experiments predate the directional-splitting fix
and explain why they were not promoted as workarounds for the old
amplification.

A linear antidiffusive-flux blend was tested first:

```math
F_\theta = F_{\mathrm{up}} +
           \theta(F_{\mathrm{LW}}-F_{\mathrm{up}}),
\qquad 0\leq\theta\leq1.
```

For the same mixed-level spectral problem, the largest stable constant blend
factors (spectral radius no larger than ``1+10^{-10}``) were:

| CFL | Maximum ``\theta``, x/y |
|---:|---:|
| 0.2 | 0.902 / 0.903 |
| 0.4 | 0.686 / 0.698 |
| 0.6 | 0.404 / 0.446 |
| 0.8 | 0.041 / 0.151 |

Thus a fixed blend retains almost none of the x-component LW correction at
CFL 0.8. It also leaves an ``O(h)`` interface-flux error whenever
``\theta<1``, so it does not repair the observed loss of mixed-level order.

Monotonized Central is a MUSCL slope limiter and needs a second upwind value.
At a two-to-one mesh-size jump, the ordinary uniform-grid MC formula is not
the appropriate formula: the enhanced MC construction of Zeng includes the
neighbouring mesh ratios to retain both TVD stability and linearity. The
current dual factorization is more general than a one-dimensional nonuniform
line. A hanging-face connection may be the product of several primal fluxes
and has no unique second upwind neighbour. Implementing enhanced MC would
therefore require a geometric gradient reconstruction over a wider patch; it
cannot be added correctly as a local two-state function.

A multidimensional flux-corrected-transport (FCT) limiter is a closer match to
the present algebra. The high-order flux is LW, the low-order flux is upwind,
and the pairwise antidiffusive difference can be limited once and applied with
opposite signs to the two adjacent dual volumes. This preserves momentum and
uses the existing connection graph rather than inventing a second-upwind
edge. Two variants were prototyped for a smooth wave at CFL 0.8:

| Method | ``L^2`` error, 1 crossing | Range, 1 crossing | ``L^2`` error, 5 crossings | Range, 5 crossings |
|---|---:|---:|---:|---:|
| bulk LW / interface upwind | 0.0728 | 0.8625--1.1937 | 0.1362 | 0.8780--1.1360 |
| LW everywhere | 0.0774 | 0.8261--1.2133 | 0.2614 | 0.6304--1.4014 |
| FCT of interface corrections only | 0.0757 | 0.8289--1.2096 | 0.2478 | 0.6666--1.3777 |
| Full-graph FCT | 0.0684 | 0.8420--1.1620 | 0.1560 | 0.9037--1.0996 |

The initial range was 0.8--1.2. Interface-only FCT does not supply a
maximum-principle guarantee because bulk LW is not the monotone low-order
update assumed by FCT, and it did not arrest the long-time growth. Full-graph
FCT remained bounded and accepted about 90% of the antidiffusive correction on
average after five crossings, but it changes the bulk algorithm and is not an
honest implementation of `amr_interface_scheme`. It is therefore a promising
future *bulk* scheme, not promoted to the public interface by this experiment.

For smooth momentum transport, another AMR-specific alternative in
the literature is not a limiter: Kramer, Pantano, and Pullin construct wider
patch-interface closures by enforcing summation by parts. Their two-dimensional
closures are nondissipative and energy stable, including edge and corner
geometries. Adapting that constrained-stencil approach while retaining
``D_s I_F=I_C D`` remains relevant if a stronger energy-stability guarantee is
needed than the present spectral regression provides.

The public scheme values are `Upwind()`, `Centred()`, and `LaxWendroff()`;
`Centered()` and `LW()` are aliases. For LW, `convection!` evaluates the
effective split-step rate and `advect!` applies the split conservative update.
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

The requested primary divergence is therefore recovered directly from the
diagonal of convection, including at hanging faces. Summation by parts on a
periodic mesh gives

```math
\langle p,C(u)q\rangle_V = -\langle C(u)p,q\rangle_V
\quad\text{when}\quad D_c u=0.
```

The tests assemble complete convection matrices on both single-level and
mixed-level trees. They recover ``I_C D_c u`` from twice the diagonal, check
weighted skew-symmetry for a divergence-free velocity, verify exact momentum
conservation, and show that upwind transport damps an alternating interface
mode without creating new extrema. This is a semi-discrete property; a forward
Euler step with the centred operator does not by itself preserve kinetic
energy in time.

## Current scope

The dual schedule supports uniform and 2:1-balanced mixed-level meshes with
periodic or physical boundaries. A non-periodic side is impermeable: its
normal velocity and advective momentum flux are zero, while its truncated
dual volume is assembled explicitly. Viscous tangential wall data is handled
separately by `NoSlip()` or `FreeSlip()`. Refinement subcycling and
time-dependent regridding remain future synchronization concerns; the current
milestone uses one global time step and a static tree.
