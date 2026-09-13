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

On a uniform level, the Lax--Wendroff state follows the multidimensional
construction in Remmerswaal's thesis. It first linearly interpolates the
upwind state in the tangential direction to the line through the
donating-region centroid, then interpolates that value and the downwind state
in the normal direction. When the tangential velocity is zero, this reduces to

```math
q_\mathrm{LW} = \frac{q^-+q^+}{2}
                 - \frac{\Delta t\,a}{2h}(q^+-q^-),
```

the classical one-dimensional Lax--Wendroff state. The local staggered CFL
bound controls both interpolation fractions.

There is no unique transverse donor on a factored hanging-face connection.
The mixed-level schedule therefore uses the corresponding one-dimensional LW
state with ``|\Delta t F_g|/\min(W_-,W_+)`` as its local Courant number. The
interface scheme is an explicit API choice:

```julia
convection!(rate, q, velocity, dt, workspace)
advect!(q, velocity, dt, workspace;
        scheme=LW(), amr_interface_scheme=LW())
```

The bulk `scheme` defaults to `LW()`, while `amr_interface_scheme` defaults to
`Upwind()` for robustness. Either can be `Upwind()`, `Centred()`, or `LW()`
(and accepts the corresponding symbols). The interface keyword has no
numerical effect on a uniform mesh.

The upwind state is monotone under the returned multidimensional CFL bound.
Centred and LW remain available for smooth resolved fields, but stability is
not inferred from their uniform-grid behavior. On the fixed level-1/level-2
test tree, the spectral radii of the one-step update using bulk LW are:

| CFL | Interface upwind, x/y | Interface LW, x/y | Interface centred, x/y |
|---:|---:|---:|---:|
| 0.2 | 1.000000 / 1.000000 | 1.000140 / 1.000153 | 1.000198 / 1.000213 |
| 0.4 | 1.000000 / 1.000000 | 1.000761 / 1.000806 | 1.001029 / 1.001054 |
| 0.6 | 1.000000 / 1.000000 | 1.001887 / 1.001967 | 1.002547 / 1.002563 |
| 0.8 | 1.000000 / 1.000000 | 1.003519 / 1.003643 | 1.004773 / 1.004771 |

Thus the present interface LW closure is mildly but genuinely unstable for
this divergence-free constant-velocity problem, even at CFL 0.2. It remains
available for experimentation, but is not the default. The hybrid bulk-LW /
interface-upwind update has spectral radius one in this test; its weighted
operator norm is slightly above one, so transient non-normal amplification is
still possible and is not concealed by the spectral result.

The robustness switch is not accuracy-neutral. In a smooth manufactured
spatial-operator test on successively finer copies of the same mixed-level
tree, bulk-LW/interface-upwind converges at about order 0.53 and 0.50 in
``L^2``. Applying LW everywhere gives orders 1.72 and 1.57, but has the
amplification reported above; applying upwind everywhere gives the coherent
first-order rates 0.93 and 0.92. Consequently, all-upwind is the current
robust accuracy baseline. The hybrid and all-LW combinations are exposed for
experimentation, not claimed as a finished second-order AMR closure.

The public scheme values are `Upwind()`, `Centred()`, and `LaxWendroff()`;
`Centered()` and `LW()` are aliases. `convection!` evaluates the operator,
while `advect!` applies its forward-Euler conservative update. A
`StaggeredVelocityAdvectionWorkspace` advances both velocity components from
the same old-time advecting field.

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
periodic boundaries in both directions. Physical boundaries remain out of
scope: they require explicit boundary dual volumes and boundary momentum
fluxes rather than an implicit halo convention. Refinement subcycling and
time-dependent regridding likewise remain future synchronization concerns;
the current milestone uses one global time step and a static tree.
