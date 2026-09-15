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
can be `Upwind()`, `Centred()`, `LW()`, `Fromm()`, `Minmod()`, `MC()`,
`VanLeer()`, or `Superbee()` and accepts the corresponding lower-case symbols.
For example, the decaying-turbulence showcase uses

```julia
scheme=LW(), amr_interface_scheme=MC()
```

so that only non-regular AMR connections use the nonlinear reconstruction.
The interface keyword has no numerical effect on a uniform mesh.

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
corrections; it is included as the requested larger-stencil control, not as a
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
| LW / interface Fromm | 0.1361, 0.04070, 0.01369 | 1.742, 1.572 |
| LW / interface minmod | 0.1774, 0.04694, 0.01489 | 1.918, 1.657 |
| LW / interface MC | 0.1367, 0.04058, 0.01433 | 1.752, 1.502 |
| LW / interface van Leer | 0.1512, 0.04071, 0.01405 | 1.893, 1.535 |
| LW / interface superbee | 0.1272, 0.03872, 0.01430 | 1.716, 1.436 |
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

## Isolated refinement-interface regression

The small regression is deliberately time independent. It puts a unit impulse
on the face unknown with the largest coarse--fine graph incidence, leaves all
other values zero, prescribes constant transport velocity, and evaluates
``C(q)`` at `dt == 0`. Because ``dq/dt=-C(q)``, a positive rate at a zero node
would create a negative value for every positive forward step, however small.

| Interface closure | Maximum offending rate, x/y placement |
|---|---:|
| LW | 2.4 / 1.2 |
| Fromm | 1.2 / 0.6 |
| minmod | 0 / 0 |
| MC | 0 / 0 |
| van Leer | 0 / 0 |
| superbee | 0 / 0 |

This isolates the dispersive refinement-interface mechanism from diffusion,
pressure projection, dynamic adaptation, transfer operators, and the periodic
seam. The same test also checks that every limited rate integrates to zero.
It is a semi-discrete result, not a global TVD claim: with bulk LW retained, a
complete two-direction impulse step at CFL 0.4 still has a small
``5.76\times10^{-4}`` undershoot generated after the first intermediate has
been published. Use a limited bulk `scheme` as well when global one-dimensional
TVD behavior is required; the uniform step regression verifies that mode.
It is part of the normal test suite. `benchmark/staggered_advection_interface.jl`
reproduces this table and the manufactured convergence table above.

For smooth momentum transport, another AMR-specific alternative in
the literature is not a limiter: Kramer, Pantano, and Pullin construct wider
patch-interface closures by enforcing summation by parts. Their two-dimensional
closures are nondissipative and energy stable, including edge and corner
geometries. Adapting that constrained-stencil approach while retaining
``D_s I_F=I_C D`` remains relevant if a stronger energy-stability guarantee is
needed than the present spectral regression provides.

The public scheme values are `Upwind()`, `Centred()`, `LaxWendroff()`,
`Fromm()`, `Minmod()`, `MonotonizedCentral()`, `VanLeer()`, and `Superbee()`;
`Centered()`, `LW()`, and `MC()` are aliases. For LW and the three-point
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
local time stepping remain future synchronization concerns; dynamic regridding
currently uses one global time step and rebuilds the advection workspace after
each accepted mesh change.
