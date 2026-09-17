# Two-velocity interface advection

This note fixes the discrete contract for the two candidate interface methods
before their optimized kernels are implemented. It uses the notation and
model from `pt1_mass.tex`, `pt1_cutcell_advection.tex`, and
`int_advection_split.tex` in the thesis source. In particular, the physical
interface is advected by the liquid velocity in the projected method; the gas
update in that method is an auxiliary conservative carrier.

## Directional WY balance

Let ``V_c^o`` be the geometry-open volume of cell ``c``. Every entry in the
mesh face schedule is a physical leaf-level face segment, including the fine
segments of a coarse--fine connection. A signed, time-integrated face volume
is positive from `face.minus` to `face.plus`. Define

```math
({\cal D}^k F)_c = \frac{1}{V_c^o}
  \sum_{f\in\partial_k c}\sigma_{cf}F_f,
```

where ``\sigma_{cf}=+1`` for `face.minus` and ``-1`` for `face.plus`. For one
directional sweep, the Weymouth--Yue (WY) update is

```math
C_{\ell,c}^{[k+1]} = C_{\ell,c}^{[k]}
 - {\cal D}^k L_c + \chi_{\ell,c}^n {\cal D}^k Q_c. \tag{1}
```

Here ``C_\ell`` is the liquid fraction relative to the open volume, ``L_f``
is the liquid swept volume, ``Q_f`` is the complete carrier volume, and
``\chi_{\ell,c}^n = [C_{\ell,c}^n>1/2]`` is frozen for the complete time step.
Equation (1) is equation
`eqn:int:advection:wy:single_step` of `int_advection_split.tex`, written using
time-integrated fluxes and the package's face orientation.

The PLIC plane is reconstructed after every directional sweep. The indicator,
velocity, and time-step-start data used by dilation remain frozen. Alternating
the sweep order between time steps is independent of these balance identities.

If liquid and gas fluxes partition the carrier,

```math
L_f+G_f=Q_f, \tag{2}
```

the matching gas equation is

```math
C_{g,c}^{[k+1]} = C_{g,c}^{[k]}
 - {\cal D}^k G_c + (1-\chi_{\ell,c}^n){\cal D}^k Q_c. \tag{3}
```

Equations (1)--(3) preserve ``C_\ell+C_g=1`` after every sweep. They reduce
exactly to the existing one-velocity implementation when
``Q_f=\Delta t\,A_f a_f^o u_f`` and the same donating region is split by the
PLIC plane.

## Compatible centred scalars

Persistent scalar fields store intrinsic values. During a sweep only, define
the phase amount ``M_c^\pi=C_c^\pi s_c^\pi``. Its update is

```math
M_c^{\pi,[k+1]} = M_c^{\pi,[k]}
 - {\cal D}^k(F^\pi s_f^\pi)_c
 + \chi_c^{\pi,n}s_c^{\pi,n}{\cal D}^k Q_c. \tag{4}
```

The phase-interface interpolant in ``s_f^\pi`` is upwind. Higher-order bulk
interpolation is allowed only when all cells in its stencil contain that
phase. Applying (4) to ``s^\pi=1`` gives the phase-volume equation bit for bit;
this is the constant-one compatibility requirement. Recovery divides by the
new phase fraction only in an active cell. A newly active cell receives the
upwind/extension value selected while forming the flux; no artificial small
denominator is inserted. The global interface `fraction_floor` remains the
single rule that removes wisps and determines activity.

## Method A: projected liquid velocity

The thesis equations following
`eqn:pt1:interface:twofluid:cutcell:div_approx` define
``\widehat u_\ell`` by a Poisson projection. The pressure unknown exists in
cells with positive liquid fraction and is fixed to zero in liquid-empty
cells. A face receives the pressure correction exactly when its staggered
liquid fraction is positive. Missing face velocities outside that set are
filled by constant extrapolation far enough to construct every donating
region used by the split sweep. Its discrete divergence must vanish in every
cell with positive liquid volume. The physical liquid carrier and flux are

```math
\widehat Q_f=\Delta t\,A_f a_f^o\widehat u_{\ell,f},\qquad
L_f=|D_f(\widehat u_\ell)\cap\Omega_\ell^{[k]}|_{\rm signed}. \tag{5}
```

Equation (1) uses ``\widehat Q`` for both the WY dilation and the PLIC swept
volume. No gas velocity appears in the physical interface update. Since
``\chi_\ell^n`` is zero in an empty liquid cell and
``\nabla_h\!\cdot\widehat u_\ell=0`` in every nonempty liquid cell, the summed
dilation vanishes and liquid volume is conservative.

Gas transport uses a separate auxiliary carrier, as required by
`eqn:pt1:mass:twofluid:mass_cons_gas`:

```math
C_g^*{}^{[k+1]}=C_g^*{}^{[k]}-{\cal D}^kG(u_g), \tag{6}
```

with no WY dilation. This is a conservative geometric carrier and may exceed
one because ``u_g`` is not separately divergence free at the interface. The
physical gas fraction is always ``1-C_\ell``. Intrinsic gas scalars use the
same auxiliary carrier in numerator and denominator, so their constant-one
identity is retained. The difference between ``C_g^*`` and ``1-C_\ell`` is an
explicit diagnostic; it must never be silently folded into a stored
phase-weighted field.

The thesis equation is dimensionally unsplit, whereas this package uses
directional splitting. Consequently, every one-dimensional gas substep starts
from the current physical complement and discards its auxiliary carrier after
recovering the intrinsic scalar. Carrying ``C_g^*`` into the perpendicular
sweep would combine it with a PLIC reconstructed from ``1-C_\ell`` rather than
from ``C_g^*``. That mismatch can make the supposedly lower-bounded carrier
negative. The reported reset discrepancy is therefore accumulated over both
directional substeps. This is the split analogue of the thesis reset and is a
measurable additional approximation, not stored state.

The constant-extrapolation band is recomputed far enough into the gas to cover
the largest PLIC, scalar, staggered-CTU, coarse--fine, and cut-cell stencil used
in that step. It is distinct from the pressure domain: extending the
Dirichlet-pressure solve into gas cells would change the thesis operator. At a
fixed solid, the pressure correction uses the cut-cell gradient which is
adjoint to the open-volume divergence.

## Method B: frozen phase subfaces

The alternative method follows equations
`eqn:pt1:advection:cc:partial_volumeflux`--
`eqn:pt1:advection:cc:volume_enforce`. At ``t^n``, split every geometry-open
face into frozen liquid and gas subfaces with apertures
``a_f^{\ell,n}`` and ``a_f^{g,n}``. For each carrier phase ``\pi'``, extrude
that subface with its own velocity. For either transported phase ``\pi``,

```math
F_f^\pi=\sum_{\pi'\in\{\ell,g\}}
 |D_{f^{\pi'}}(u_{\pi'})\cap\Omega_\pi^{[k]}|_{\rm signed}, \tag{7}
```

and

```math
Q_f=\Delta t\,A_f
 (a_f^{\ell,n}u_{\ell,f}+a_f^{g,n}u_{g,f}). \tag{8}
```

Each donating region is geometrically divided into liquid and gas, so (2)
holds to roundoff even when the two carrier velocities have opposite signs.
Equations (1), (3), and (4) therefore use the mixture carrier (8) in their WY
dilation. The PLIC geometry is reconstructed between directional sweeps, while
the face subdivision ``a_f^{\pi,n}`` remains frozen for the complete time
step. Recomputing that subdivision after the first sweep would no longer be
the thesis method.

The approximation replaces one phase velocity by the other away from the
moving interface. The thesis derives a local time-integrated error
``O(\Delta t^2h^{d-1})`` in
`eqn:pt1:advection:cc:replace_bdy_intd`, and reports failed mesh convergence at
large density ratio. This is a property to measure, not a defect to hide with
fraction clipping. The implementation therefore records the face partition
defect and any fraction-bound failure separately.

Fixed solid geometry is applied before the liquid/gas subdivision:
``a_f^\ell+a_f^g=a_f^o``. Both partial donating regions are clipped to the
solid-open cell. At an AMR connection, equation (7) is evaluated on each owned
fine face segment. The same integrated value is shared by its two incident
cells, so summing the segments is the reflux operation; a second coarse-face
flux must not be constructed.

### Projection timing

Both methods use geometry from the current time level, but their projection
requirements differ. The projected-liquid method recomputes its active cells,
staggered liquid fractions, operator coefficients, extension, and auxiliary
projection at the start of every `advect!` call. Reusing a projected field
after the interface has changed would solve the wrong discrete problem.

The frozen-subface method does not add an auxiliary projection. It assumes
that the supplied physical phase velocities already satisfy

```math
\nabla_h\!\cdot(a_f^\ell u_f^\ell+a_f^g u_f^g)=0
```

for apertures reconstructed at the current time level. A stationary analytic
velocity sampled on the grid generally fails this discrete constraint when
the apertures change. A passive prescribed-velocity test must therefore
reimpose the mixture constraint after each aperture update. In a complete
two-velocity flow solve this is the responsibility of the physical pressure
projection. `advect!` always refreshes the frozen apertures itself; it does not
assume that a preceding CFL query did so.

`test/integration/two_velocity_rotation.jl` exercises this distinction with a
circular liquid disc in solid-body rotation and quiescent gas. The velocity
jump is tangential, so the exact interface is stationary. Both policies use
the same time steps and split order. The test establishes convergence of the
stationary-interface error and verifies liquid-volume conservation; its
test-side common correction for the frozen method preserves the phase
velocity jump and enforces only the aperture-weighted mixture constraint.

## Two-velocity WY CFL

The boundedness proof in `int_advection_split.tex` bounds incoming swept
volumes, rather than a maximum velocity component. For frozen subfaces the
safe local condition is

```math
\frac{\Delta t}{V_c^o}\sum_{f\in\partial c}
 \sum_{\pi\in\{\ell,g\}}
 \max(-\sigma_{cf}A_fa_f^\pi u_f^\pi,0)\le C_{WY},
 \qquad C_{WY}\le\tfrac12. \tag{9}
```

The implementation also checks the corresponding outgoing sum when the input
velocity has not yet met the mixture divergence constraint. This prevents a
cell from being drained while keeping the intended condition unchanged for a
discretely divergence-free field. Every active partial extrusion additionally
satisfies
``\Delta t|u_f^\pi|/h_{\rm donor}\le 1/2``; this protects a fine face segment
whose adjacent donor is coarser. Counterflowing phase subfaces are counted
separately in (9), since cancellation in their net mixture flux does not make
either donating region disappear.

For Method A, the physical-interface restriction is (9) with the projected
liquid carrier. The auxiliary gas carrier has its own incoming, outgoing, and
extrusion bounds. The common time step is the minimum of both. Before the
projection exists, taking the minimum of the two existing full-face WY bounds
is conservative but is intentionally only a construction-time fallback.

## Executable algebraic reference

`test/two_velocity_wy_reference.jl` implements equations (1)--(4) on a
periodic one-dimensional ring without calling the production advection
kernel. It verifies phase complementarity, the phase-one-function identity,
constant intrinsic scalars, global conservation for a divergence-free
carrier, and exact reduction to the one-velocity equations. Later optimized
kernels will be compared directly with this reference on one-cell-thick
meshes.

The reference is algebraic: in one spatial dimension an interface cannot cut
the interior of a point face. Its supplied liquid/gas flux arrays represent
the geometrically integrated results of (5) or (7), which is exactly the
interface between geometry and conservative transport that it is intended to
test.
