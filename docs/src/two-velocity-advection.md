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

## Compatible staggered scalars and phase momentum

The same centred phase-volume fluxes are algebraically interpolated onto each
staggered dual grid. For phase ``\pi`` and one velocity component, the local
conservative update is

```math
(C_f^\pi u_f^\pi)^{[k+1]}
 = (C_f^\pi u_f^\pi)^{[k]}
   - {\cal D}_{f}^{k}(F^\pi u^\pi)
   + {\cal I}_f(\chi^\pi {\cal D}^k Q)u_f^{\pi,n}.
```

The implementation stores only ``u_f^\pi``. It forms the products in (5)
inside the update and divides by the staggered phase fraction only when that
phase is active according to the interface `fraction_floor`. At an interface,
the CTU state is selected separately for every partial interpolated flux; the
configured higher-order interpolant is used only when both adjacent dual
volumes contain at least `ctu_threshold` of the phase. This is the modified
interpolant in `pt1_momentum.tex` with the thesis default
``C_{\min}=1/2``.

For constant per-phase densities, multiplying (5) by ``\rho^\pi`` and dividing
by the new phase mass cancels the density algebraically. The API still accepts
the density pair because it defines the conserved momentum and is needed when
the two phase results are recombined or diagnosed. Liquid and gas velocities
remain the transported state; advection does not replace them with a
mass-weighted mixture. The diffusion step forms its equivalent one-velocity
field only in reusable scratch storage.

```julia
momentum_workspace = MomentumAdvectionWorkspace(interface, velocity)
passive_x = PhaseFaceScalar{1}(
    interface; liquid=liquid_tracer, gas=gas_tracer)

advect!(
    interface, velocity, dt, workspace;
    staggered_scalars=(passive_x,),
    momentum=velocity,
    density=(rho_liquid, rho_gas),
    momentum_workspace,
    scheme=Fromm(),
)
```

The workspace snapshots both old intrinsic velocity fields, so the source and
destination tuple may be the same object. Under the projected-liquid policy,
the physical liquid uses the projected carrier while momentum values come
from the original liquid velocity. The gas velocity uses the same auxiliary
gas carrier and per-sweep reset as centred gas scalars. Under the frozen-face
policy, both staggered phase updates consume the complementary flux pair.
Applying the update to a one-function reproduces the interpolated staggered
phase-volume balance exactly. `phase_flux_diagnostics(workspace)` exposes the
signed and absolute gas-momentum reset for both components as
`gas_momentum_reset_discrepancy_x/y` and
`absolute_gas_momentum_reset_discrepancy_x/y`.
The tuple entry point always uses this separate phase-momentum update. Equal
phase velocities are therefore a validation limit of the two-velocity
discretization rather than a runtime dispatch to the one-velocity operator.
Callers that want the one-velocity model pass a `FaceVelocity` directly.

## Passive staggered validation

`test/integration/two_velocity_vortex.jl` repeats the classical vortex-reverse
study with two distinct `FaceVelocity` objects containing the same continuous
advecting field. Consequently, the test exercises the two-velocity policy and
cannot enter the one-velocity implementation. It advects the discontinuous
staggered fields from `pt1_validation.tex`, reports liquid and gas velocity and
momentum errors separately, and covers uniform, fixed mixed-level, and
dynamically adapting meshes.

For the 16--32--64 uniform series, the aggregate observed orders of the
projected method are 1.56 in the liquid and 1.62 in the gas; its finest-pair
orders are 2.01 and 1.30. The frozen-interface method gives aggregate orders
1.55 and 2.26, with finest-pair orders 2.01 and 2.34. On the fixed mixed-level
and dynamically adapting series, both phase fields converge between order
1.81 and 2.34. These results establish the thesis's close-to-second-order
phase-weighted ``L^1`` behavior through mesh refinement. The slower uniform
projected-gas pair accompanies the method's auxiliary gas-carrier reset; its
reported conservation defect converges at approximately first order rather
than being hidden by the physical complementary fraction.

The tangentially discontinuous rotating-disc study separately checks the case
for which no single continuous carrier exists. At every step it reinitializes
the two analytical face-centred velocity fields and projects that fresh state,
so projection errors do not accumulate in the prescribed carrier velocity.
The solid-body liquid velocity is already divergence-free for the projected
extension operator and receives no correction. The frozen-interface method's
aperture-weighted mixture does require a correction, which is reported
separately in normal and tangential components.

Both policies preserve staggered one-functions exactly and show first-order
convergence of smooth phase-specific staggered fields at the persistently
moving CTU interface. The modified phase flux interpolant switches to CTU
whenever either predicted phase fraction is below `ctu_threshold`, as
prescribed in `pt1_momentum.tex`. The vortex-reversal test has a moving CTU
band and cancellation from reversal, and therefore recovers the thesis's
higher phase-weighted integral rates.

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
the same time steps and split order. The projected-extension policy retains
its narrow-band liquid projection. The frozen-aperture policy uses the
physical three-block mimetic gradient before every advection step, enforcing
both mixture incompressibility and the aperture-based normal-jump constraint.
The latter exposes a current accuracy problem: although both constraints close
to tolerance and liquid volume is conserved, the full-period frozen interface
error is nonmonotone under refinement. The expected convergence checks remain
explicit broken gates until the near-empty staggered-face treatment is fixed.

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

## Physical pressure projection

The complete inviscid model passes the same velocity tuple to
`IncompressibleNavierStokesSolver`. After compatible interface and momentum
advection, it solves the symmetric `(p, xi, lambda)` mimetic system described
in [Incompressible Navier--Stokes solver](navier-stokes.md#two-velocity-projection).
This projection enforces mixture incompressibility and the aperture-based
normal-jump constraint while retaining tangential slip. The rotating-disc
integration study reinitializes the analytical discontinuous velocity before
each projection so its measured error belongs to the pressure model rather
than accumulated velocity transport.
