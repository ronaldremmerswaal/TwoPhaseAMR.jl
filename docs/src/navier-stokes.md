# Incompressible Navier--Stokes solver

One solver advances face velocity and cell pressure on a static or dynamically
adapted uniform/mixed-level tree. Supplying an interface selects two-phase
coefficients and compatible geometric transport. A single `FaceVelocity`
selects the one-velocity model; a `(liquid_velocity, gas_velocity)` tuple
selects the two-velocity model. The outer time-step sequence stays the same.

## Common step

Each step executes transport, midpoint body forcing, implicit diffusion,
optional phase forces, and one pressure projection. Dispatch changes the
kernel inside a stage rather than selecting a second time integrator.

For single-phase flow, the transport and user-force state is

```math
u^c = u^n - \Delta t\,C_{LW}(u^n) + \Delta t\,f^{n+1/2}.
```

The bulk scheme defaults to `LaxWendroff()` and the scheme-aware AMR-interface
default selects `MonotonizedCentral()` on non-regular coarse--fine connections. Global Upwind retains
Upwind at the interface, and another limited bulk scheme is used unchanged.
Interface-only Upwind is rejected because its unmatched first-order closure
has poor mixed-level accuracy. `amr_interface_scheme` can still be overridden
with any compatible non-Upwind scheme.

For two-phase flow, directionally split Weymouth--Yue interface advection first
computes the liquid and gas volumes crossing every centred face. The momentum
stage interpolates those same volume fluxes to staggered control volumes and
advects the two intrinsic phase velocities with CTU upwinding near the
interface. Only intrinsic velocity is stored. Compatible phase scalars follow
the same construction, so a scalar equal to one in either phase is recovered
exactly. Intrinsic recovery divides only above `sqrt(eps(T))` phase fraction.

The generic user force `f` is sampled at the time midpoint and added to the
transported state before diffusion in both models. Construct a flow with
`force=(x, y, time, axis) -> ...`; the callback is stored as `flow.force`
and defaults to `nothing`. Both `step!(flow, dt)` and `simulate!(flow, ...)`
use this stored force unless it is overridden for the call. Single-phase
viscosity is then centred by Crank--Nicolson/implicit midpoint:

```math
\left(I-\frac{\Delta t\nu}{2}L\right)u^{**}
= u^c + \frac{\Delta t\nu}{2}Lu^n
  + \Delta t\nu b^{n+1/2}_{wall}.
```

The affine wall term contains both half-step contributions. The shifted system
is solved independently on the x- and y-dual grids by
`StaggeredEllipticSolver`. It selects external Krylov.jl CG for a symmetric
operator and BiCGSTAB otherwise, with the
package's geometric V-cycle preconditioner. The V-cycle objects contain no
Krylov logic. The unchanged operator, preconditioner, and workspaces are reused
for a fixed `dt`, and LU appears only in the two geometric bottom solves.

For two phases, the viscous term is the coupled stress divergence
`div(2*mu*D(u))`. At cell-centred normal stresses and vertex-centred shear
stresses, the dynamic viscosity is the geometric mixture

```math
\bar\mu=(\mu^\ell)^c(\mu^g)^{1-c},
```

where `c` is interpolated to the stress location before evaluating the
formula. Density appears in the staggered inertia matrix, not in this dynamic
viscosity. The midpoint system is

```math
\left(M_\rho+\frac{\Delta t}{2}K_\mu\right)u^{**}
=M_\rho u^*-\frac{\Delta t}{2}K_\mu u^*,
```

where `K_mu` is the positive-semidefinite integrated stress stiffness.
Here `u*` is the state after compatible momentum transport and any generic
body force applied before diffusion, matching the split stress
`mu*symgrad(u* + u**)`.
This staggered variable-density/variable-viscosity construction and the
two-velocity diffusion model are documented by
[Remmerswaal (2023)](https://doi.org/10.33612/diss.562133319); related
finite-volume Stokes solver structure is described by
[Cai et al. (2014)](https://doi.org/10.4208/cicp.070114.170614a).

Balanced CSF or CGFM surface tension and gravity are added after diffusion
using a weighted-gradient discretization compatible with the pressure
operator. Denote the state after these forces by ``u^H`` and let ``G`` include
inverse density and the appropriate geometric coefficients.
The one-velocity and CMDGFM projections form a pressure-predicted state and
solve for the pressure correction ``\phi``:

```math
u^* = u^H - \Delta t\,G p^n,\qquad
-DG\phi = -\frac{Du^*-D^n u^n}{\Delta t}
          -\frac{D^n u^n}{\Delta t_{\mathrm{ref}}},
\qquad
u^{n+1}=u^*-\Delta t\,G\phi,\qquad
p^{n+1}=p^n+\phi.
```

For the one-velocity two-phase model, `G` includes the face coefficient `1/<rho>`, where
`<rho>=rho_l*c_f+rho_g*(1-c_f)`. The same finite-volume `D`
and its dual-derived `G` are used in the Poisson matrix, old-pressure
predictor, and velocity correction.

For these incremental projections, ``D^n u^n`` is the inherited divergence
measured on the old state and old apertures. In `simulate!`, the reference
step ``\Delta t_{\mathrm{ref}}`` is the maximum of the unshortened proposal,
the current step, and the previous actual step when available. After a regrid,
the new transfer defect instead uses the proposal associated with that
transfer (and with the two-velocity regrid projection). A direct
`step!(flow, dt)` uses the maximum of `dt` and any retained defect reference;
the field-level `step!` defaults its reference to the supplied step. An
ordinary step with equal actual and reference steps recovers the
usual full projection. Otherwise the solve removes the fraction
``\Delta t/\Delta t_{\mathrm{ref}}`` of the inherited defect rather than
dividing all of it by the short step. Up to solver residual and boundary
enforcement, the remaining defect is
``(1-\Delta t/\Delta t_{\mathrm{ref}})D^n u^n``. This uses the same solve,
not an additional projection.

An input pressure supplied to `FlowSimulation` is the initial physical
pressure; if omitted, it is zero. A fresh hydrostatic calculation therefore
starts with ``p^0=0``; its first projection builds the pressure.
A nonzero initial pressure enters the first incremental predictor; a mimetic
step instead replaces it with the absolute pressure solution.
The mimetic method's existing hydrostatic limitation is not removed.
Resumed calculations use the retained physical pressure, including after
transfer to an adapted mesh.
The standalone `project!(flow)` can prepare the initial velocity constraint;
it uses scratch pressure for that geometric impulse and leaves `flow.pressure`
unchanged. The lower-level one-velocity and CMDGFM `project!` methods instead
add their solved increments to the supplied pressure and expect an input
velocity already predicted with that pressure. Repeatedly projecting an
unpredicted velocity with a reused pressure field therefore accumulates
impulses; pass fresh zero pressure for an absolute velocity projection.
The lower-level mimetic `project!` overwrites its pressure with the absolute
pressure solve and expects a predictor without an old-pressure subtraction.

### Two-velocity projection

[`TwoVelocityModel`](@ref) selects either CMDGFM or the legacy symmetric
three-block mimetic formulation through its `pressure_projection` field.
The default is `CMDGFMProjection()`. The default interface carrier is
`FrozenInterfaceFluxes()`; `ProjectedLiquidVelocity()` is a legacy carrier.

The mimetic formulation solves for the absolute physical pressure
``p^{n+1}``, the phase-gradient jump correction `xi` on mixed faces, and a
cell multiplier `lambda`:

```math
\begin{bmatrix}-A_p&C&0\\C^T&K_\xi&B^T\\0&B&0\end{bmatrix}
\begin{bmatrix}p^{n+1}\\\xi\\\lambda\end{bmatrix}
=
\begin{bmatrix}\Delta t^{-1}D\langle au^H\rangle\\0\\
-\Delta t^{-1}D([a][u^H])\end{bmatrix}.
```

`A_p` uses the existing AMR Riesz gradient and cut-cell pressure operator.
`B` uses the same conservative face/cell incidence and the aperture jump.
Only faces on which both staggered phase fractions exceed the interface's
global `fraction_floor` retain `xi`; the method does not divide by an
individual phase mass or by the aperture jump. MINRES uses a symmetric
positive block preconditioner. Surface tension and gravity are added to the
predictor before this solve and therefore enter its right hand side.

The mimetic predictor does not subtract the old pressure: a common pressure
gradient alone is not the complete phase-gradient correction of the KKT
system. This legacy projection is unchanged from the absolute-pressure
method and does not use the reference-step divergence policy above. Its
mixture-divergence, normal-jump, and inactive-face jump rows divide their
velocity defects by the actual step. Very short steps can therefore produce
pressure spikes through these rows; this aspect of issue #90 remains unresolved.

`CMDGFMProjection()` instead solves the nonsymmetric two-block system

```math
\begin{bmatrix}L_{pp}&L_{p\xi}\\L_{\xi p}&L_{\xi\xi}\end{bmatrix}
\begin{bmatrix}\phi\\\xi\end{bmatrix}=\begin{bmatrix}b_p\\b_\xi\end{bmatrix}.
```

It uses the same conservative `D`, AMR Riesz gradient, pressure block,
staggered phase fractions, PLIC apertures, and cut-cell geometry. The local
jump interpolant uses transverse incident faces and a bounded coefficient.
Its jump rows enforce normal-velocity continuity also on shortened steps;
only the mixture-divergence target retains an inherited defect. Writing
`R=|n_t/n_f|`, `gamma_lg=gamma^l_(h^g)`, and
`gamma_gl=gamma^g_(h^l)`, the implemented choice is

```math
\beta_f=\begin{cases}
1,&R\leq1,\\
\dfrac{1-\gamma_{lg}-\gamma_{gl}}
      {R-\gamma_{lg}-\gamma_{gl}},&R>1.
\end{cases}
```

Both inference faces are neighboring same-axis faces. This leaves the jump
block diagonal equal to one and makes every row weakly diagonally dominant.
BiCGSTAB uses a lower triangular block preconditioner. The pressure action is
one application of the existing rediscretized tree-geometric GMG hierarchy,
mapped between the full compatible pressure space and CMDGFM's gauge-reduced
unknowns. The jump action is a Crout ILU of the active jump block. A jump is
active only when both staggered phase fractions exceed `fraction_floor` and
its local interpolation stencil is resolved.

On a fixed pressure space the GMG hierarchy is retained and, by default,
its coefficients are refreshed when they change. The jump ILU is reused only
when the active-face ordering and matrix entries are unchanged. After a mesh
update, topology-dependent pressure data is rebuilt while reusable hierarchy,
Krylov, and geometric-template storage is retained where compatible.
Set `refresh_pressure_coefficients=false` to retain a lagged hierarchy or
`reuse_jump_factorization=false` to force jump-factor rebuilds.
`CMDGFMProjection(preconditioner=:ilu)` remains available as a whole-system
ILU comparison.

Momentum and pressure-work diagnostics use the weak face inner product induced
by the same Riesz map. This equals the familiar diagonal dual-volume kinetic
inner product on a uniform Cartesian mesh; at a coarse--fine interface it
includes the vertex-patch couplings used by the AMR gradient.

## Two-velocity diffusion

Viscosity uses an equivalent-one-velocity construction. With the
liquid fraction `c_f` in an open staggered dual volume, the transported
intrinsic velocities first form

```math
\bar u^* = \frac{\rho^\ell c_f u^{\ell,*}
 + \rho^g(1-c_f)u^{g,*}}
 {\rho^\ell c_f + \rho^g(1-c_f)}.
```

The existing one-velocity two-phase midpoint solver advances this field using
the same variable-density inertia and geometric dynamic-viscosity average.
Both phases then receive the common increment

```math
u^{\pi,**}=u^{\pi,*}+(\bar u^{**}-\bar u^*).
```

The phase-weighted momentum therefore matches the one-velocity solve, the
intrinsic jump `u_g-u_liquid` is preserved, and equal phase velocities recover
the one-velocity result exactly. On embedded geometry, `c_f` is weighted over
the open part of the staggered dual volume. Only intrinsic velocities are
stored; phase-weighted values are local temporaries.

The integration suite also advances the finite-viscosity `La = 3000`
capillary wave over a quarter period on 8, 16, and 32 cells per wavelength.
It compares the complete amplitude history with Prosperetti's linear solution
and with the one-velocity solver on the same meshes and time steps:

| Density ratio | Finest theory error | Finest theory rate | Finest one-velocity difference | Finest reference rate |
|:--|--:|--:|--:|--:|
| 1 | `4.70e-4` | `2.38` | `3.03e-4` | `3.67` |
| `1e3` | `5.20e-3` | `2.73` | `6.27e-4` | `4.84` |

These are normalized time-integrated amplitude errors. The time step scales
as `h^2` so the refinement rates measure spatial convergence rather than the
first-order force splitting.

## Dimensionless solver tolerance

`step!` and `simulate!` expose one dimensionless `tolerance`, shared by the
pressure and diffusion solves. They do not expose algebraic `rtol` or
dimensional `atol` values.

For pressure, the implementation follows Basilisk. If `r_p` is the strong
Poisson residual, the requested stopping condition is

```math
\Delta t^2\lVert r_p\rVert_\infty \leq \mathtt{tolerance}.
```

This quantity measures the fractional volume defect associated with the
linear-solve residual. On a shortened step the intended, deliberately
retained inherited divergence above is additional; boundary enforcement can
also change the final divergence.

For diffusion, the integrated momentum residual is divided by the local
staggered inertia. This gives a velocity defect. Its infinity norm is divided
by the corresponding strong right-hand-side velocity scale, so the same
dimensionless `tolerance` applies to both single- and variable-density solves.
This normalization depends on the equation data, not on the initial guess.
Both criteria use a floating-point roundoff floor from the starting solve
state. A current-iterate floor can increase the stop level only after the
residual decreases and while that floor is less than 1% of the initial residual.

`LinearSolverStats` reports the initial and final L2 and infinity norms, the
effective infinity-norm target, the residual scale, iterations, the
componentwise removed strong pressure-RHS means and their volume-weighted
aggregate over unanchored components, Krylov method, and preconditioner.
`removed_rhs_component_anchors` identifies the smallest cell index of each
unanchored component in the same order as `removed_rhs_component_means`.
For one-velocity projections, `compatible` is `true` exactly when
`dt^2 * maximum(abs, removed_rhs_component_means)` does not exceed the
requested projection tolerance (the maximum is zero when there are no
components). This compares the mean in the same dimensionless units as the
projection residual, using the actual step and the RHS formed with its
reference step. Unlike `converged`, it does not use the roundoff-raised linear
solve target: a projected linear solve can converge even if `compatible` is
false. The flag and component values also appear in the simulation summary.
An incompatible projection emits a warning once per projection workspace,
including across mesh rebuilds and subsequent calls using that workspace.
A failed high-level solve includes the diagnostics, requested tolerance,
achieved dimensionless residual, iteration cap, timestep, and simulation time.

## Boundary conditions

The default wall conditions on non-periodic sides impose no permeation,
`u dot n = 0`. The pressure increment has homogeneous Neumann data there,
so projection cannot change the prescribed boundary-normal velocity. For
each connected component without an open pressure boundary, the one-velocity
increment has volume-weighted zero mean. On a connected unanchored domain the
accumulated physical pressure retains its incoming mean; when the domain has
several components, the unanchored component means are set to zero. A
component with unbalanced prescribed boundary flux retains a divergence
contribution even when the linear solve converges: inspect `compatible` and
the componentwise removed RHS means, or use `compatibility=:error` to reject
such data. The two-velocity systems anchor each connected pressure component.
The accumulated CMDGFM
pressure retains its incoming anchor gauge; the mimetic absolute solve uses
its anchored gauge.

The viscous closure is independent:

- `NoSlip()` is the default and uses zero tangential wall velocity;
- `NoSlip(wall_velocity)` supports a moving wall such as a cavity lid; and
- `FreeSlip()` imposes zero tangential shear.

For open boundaries, `VelocityBoundaryConditions(; left, right, bottom, top)`
selects a condition per side (unspecified sides default to `NoSlip()`).
`Inflow(f)` prescribes velocity from `f(x, y, time, axis)`.
When applied to initialize a boundary, `ZeroGradientVelocity()` extrapolates
the normal velocity from the interior. `Outflow()` uses area- and
aperture-weighted extrapolation, then applies a uniform outward correction
over outflow faces to make the total boundary flux zero. Without either kind
of open side, prescribed boundary velocities must already have zero net flux.
WY interface transport requires impermeable boundaries.

The two-velocity pressure projection retains its supplied phase boundary
fluxes instead of applying an open-pressure correction. Its boundary
preparation requires each phase's net prescribed flux to balance, unless an
`Outflow()` supplies the balancing correction; an incompatible flux with
only `ZeroGradientVelocity()` open sides raises `ArgumentError`. Standalone
two-velocity projections consume already-prepared boundary values, so callers
must also ensure the componentwise mixture flux is compatible.

For one velocity, initial open-face velocities are input data, not
automatically extrapolated. In particular, a `FaceVelocity` filled only on
interior faces still has zero boundary values. If an extrapolated initial
outlet is wanted, call
`enforce_velocity_boundary!(velocity, boundary; geometry=geometry)` once before
the initial projection or simulation (omit `geometry` without a solid).
Do not repeat this after projection: it would overwrite the corrected flux.

The one-velocity pressure increment uses homogeneous Dirichlet data on
`Outflow()` and `ZeroGradientVelocity()` normal faces and homogeneous Neumann
data on fixed-velocity faces. An open face therefore participates in the
projection; its corrected normal velocity is retained rather than
re-extrapolated afterward. The old physical pressure is applied with the same
weighted gradient, including open faces, before solving for the increment.
For a connected fluid region, positive open pressure conductance removes the
constant pressure nullspace, so the solver does not subtract the
right-hand-side mean or impose a mean-zero pressure. Pure Neumann and periodic
problems retain both conventions. During a step, the open predictor is the
previous projected boundary velocity plus the area- and aperture-weighted
extrapolation of the interior velocity change. This retains the projected
flux in the zero-step limit instead of introducing a boundary impulse.

The physical pressure used by the incremental predictor is also evaluated
with zero pressure at open faces. Initial pressure data that are inconsistent
with that boundary value, including a nonzero hydrostatic outlet pressure,
can cause an initial spurious outlet impulse. The present gravity and
surface-tension force discretizations do not supply balancing open-face
forces; a vertical open boundary under gravity can therefore drain rather
than preserve a hydrostatic state. Use boundary data and a configuration
consistent with this pressure condition.

For a standalone one-velocity projection, the pressure operator must use the
same boundary configuration, for example
`EllipticSolver(PoissonOperator(mesh; velocity_boundary=boundary))`, and
`project!(...; velocity_boundary=boundary)` must receive that boundary too.
The projection retains the supplied open normal velocity as its predictor;
call `enforce_velocity_boundary!` first when extrapolation is desired.

These increment boundary conditions do not by themselves establish high-order
physical pressure boundary values. Especially at no-slip walls, a separate
boundary-accuracy study is needed before pressure is used as a validation
observable. Brown, Cortez, and Minion discuss this distinction for projection
methods.

## API

```julia
mesh = BlockTreeMesh(block_size=(32, 32), periodic=(false, false))
lid = NoSlip((x, y, time, axis) ->
    axis == 1 && y > 0.5 ? 1.0 : 0.0)

system = NavierStokesSystem(
    mesh; density=1.0, dynamic_viscosity=0.01, boundary=lid,
    scheme=LaxWendroff()) # MonotonizedCentral is selected at AMR interfaces
workspace = NavierStokesWorkspace(system)
velocity = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
pressure = CellField(mesh)

result = simulate!(
    velocity, pressure, system, 1.0, workspace;
    dt=0.002, tolerance=1e-8, max_cycles=80)
```

The corresponding two-phase setup supplies a physics object:

```julia
interface = InterfaceField(
    mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.15)
physics = TwoPhase(
    interface;
    density=(1000.0, 1.0),
    dynamic_viscosity=(1e-3, 1.8e-5),
    surface_tension=0.072,
    gravity=(0.0, -9.81),
)
flow = FlowSimulation(
    mesh, physics; velocity_model=OneVelocityModel(),
    boundary=FreeSlip(), dt=1e-4)
```

The two-velocity form supplies two intrinsic velocity initializers and uses the
same material-property API:

```julia
flow = FlowSimulation(
    mesh, physics;
    velocity_model=TwoVelocityModel(
        interface_transport=FrozenInterfaceFluxes(),
        pressure_projection=CMDGFMProjection(),
    ),
    velocity=(liquid_velocity, gas_velocity),
    boundary=FreeSlip(),
)
```

`examples/07_interfacial_instability/interfacial_instability.jl` uses this API for a combined
Kelvin--Helmholtz/Rayleigh--Taylor wave with surface tension. The example is
parameterized by the liquid-to-gas density ratio, Weber number, and Bond
number:

```julia
include("examples/07_interfacial_instability/interfacial_instability.jl")
run = run_example(
    density_ratio=1000.0,
    weber=0.25,
    bond=0.5,
    max_level=1,
    final_time=0.1,
)
```

It uses the capillary frequency as its time scale. With wavenumber `k`, the
derived coefficients satisfy
`sigma*k^3/(rho_liquid + rho_gas) = 1`. Consequently the dispersion relation
for a heavy liquid above the gas has offset
`sqrt(1 - weber - bond)`. The dimensional density tuple, phase velocity jump,
surface tension, and gravity passed to `FlowSimulation` are derived internally
from those nondimensional inputs.

Pressure, single-phase staggered diffusion, and coupled two-phase stress
diffusion default to preconditioned Krylov.jl CG because their operators are
symmetric positive (semi-)definite. Automatic selection still falls back to
BiCGSTAB for a nonsymmetric operator. Pressure and diffusion use rediscretized
tree-geometric hierarchies for both single-phase and two-phase flow. The
single-phase diffusion hierarchy keeps separate x/y component operators; the
two-phase hierarchy retains the coupled
stress operator. Both can be configured independently. Below the root level,
each hierarchy agglomerates 2×2 base roots until it reaches `coarse_size` or
`max_levels`, either root count cannot be halved, or a periodic axis would have
fewer than four cells. The last sparse operator is the bottom solve even if it
still exceeds `coarse_size`:

```julia
import AlgebraicMultigrid, IncompleteLU, Krylov

system = NavierStokesSystem(
    mesh; density=1.0, dynamic_viscosity=0.01, boundary=lid,
    pressure_krylov=Krylov.CgWorkspace,
    pressure_krylov_options=(;),
    pressure_preconditioner=:amg,
    diffusion_krylov=Krylov.BicgstabWorkspace,
    diffusion_krylov_options=(;),
    diffusion_preconditioner=:ilu,
    diffusion_preconditioner_options=(; drop_tolerance=1e-3))
```

The two `*_krylov_options` named tuples forward method-specific settings; for
example, a `GmresWorkspace` factory can use
`diffusion_krylov_options=(; restart=true, reorthogonalization=true)`.
The dimensionless convergence tolerance and iteration ceiling come from
`step!` or `simulate!`.

Available specifications and their symmetry restrictions are listed in
[Geometric multigrid and pressure projection](poisson.md#preconditioner-choices).

`step!(flow, dt; force=flow.force, observer=NoObserver(),
tolerance=sqrt(eps(T)), max_cycles=nothing)` performs one complete step and
returns the pressure-projection statistics. It advances `flow.time` and the
step, mesh-update, and unresolved-curvature counters, applies due mesh
adaptation, rebuilds topology-dependent state when needed, records step
diagnostics, and sends the observer a `:step` event. It throws if the pressure
projection does not converge. Here `T` is the flow's scalar type, and
`max_cycles=nothing` selects 100 cycles for one phase or 500 for two phases.

For lower-level control, `step!(velocity, pressure, system, dt, workspace;
time=..., force=...)` performs the physical solve but does not manage flow
time or counters, post-step adaptation, diagnostics, or observer events. Its
returned statistics report projection convergence to the caller.

`simulate!(flow, final_time; ...)` repeatedly performs full steps to the
absolute final time. It uses `flow.force` unless overridden. If `dt` is
omitted, the solver combines the staggered convective
CFL bound with the two-phase capillary and gravity-wave restrictions when
those forces are present. The implicit-midpoint diffusion stage imposes no
diffusion-based timestep restriction; pass `dt` explicitly when a particular
temporal accuracy resolution is required. Crank--Nicolson is second-order but
does not strongly damp unresolved stiff modes, so reduce the time step when a
viscous transient oscillates even though the implicit solve converges.

When a proposed step would leave less than `1e-6` of that proposal before
the target time, `simulate!` stretches it to land on the target instead.
The unmodified automatic proposal is retained for a subsequent call when the
fields, system, time, workspace, and topology still match. A full
`step!(flow, dt)` resets the automatic proposal while retaining its defect
reference; a field-level `step!` or `project!(flow)` invalidates both.
Stability checks allow a relative excess of `1e-6` plus roundoff, including for explicitly
requested steps and advection calls, to accommodate this endpoint adjustment.

Pass an observer such as a [`Recorder`](@ref) with `observer=recorder` to
store copied frames at scheduled keyframes:

```julia
fields = VisualizationFieldRegistry()
register_default_fields!(fields, flow)
recorder = Recorder(fields; diagnostics=flow, keyframes=EquispacedTimes(21))
result = simulate!(flow, final_time; observer=recorder)
store = snapshot_store(recorder)
```

The observer is optional: diagnostics are owned and evaluated by the
simulation whether or not a recorder is present. Each full `step!` emits
`:step` after adaptation and diagnostic recording. `simulate!` additionally
emits `:initial`, `:completed`, and, on failure, `:failed` through
[`observe!`](@ref); a requested stop is honored at a step boundary
(`stopped = true`). At the initial event, it provides an independent-step
procedure used for exact keyframes without changing the main time-step
sequence. Exact keyframes require a fixed mesh. See
[Recording and visualization](visualization.md) for scheduling, Studio, and
static export.

Dynamic refinement is enabled by passing `refinement_indicators` and optional
additional `adaptive_fields` to the solver constructor. Velocity and pressure
are always transferred. Adaptation happens after a completed physical step,
subject to `mesh.adaptation.options.update_interval`, without an additional
physical time step. Two-velocity flows project the transferred velocity
without replacing the transferred pressure. Topology-dependent operators and
workspaces are rebuilt only when the leaf set changes. See
[Dynamic adaptive refinement](adaptive-refinement.md) for the API, transfer
invariants, cooldown, benchmarks, and exact-keyframe limitation.

## Accuracy notes

The complete time step is verified with periodic Taylor--Green flow on uniform
and mixed-level meshes. The lid-driven-cavity example provides a familiar
viscous reference with moving and stationary no-slip walls. As with other
projection methods, velocity and divergence are the primary observables;
accurate physical pressure at a no-slip boundary requires a separate boundary
treatment and refinement study.

## References

- R. Remmerswaal, [*Numerical modelling of variability in liquid
  impacts*](https://doi.org/10.33612/diss.562133319), PhD thesis, University
  of Groningen (2023).
- D. L. Brown, R. Cortez, and M. L. Minion, [*Accurate projection methods for
  the incompressible Navier--Stokes equations*](https://doi.org/10.1006/jcph.2001.6715),
  Journal of Computational Physics 168 (2001), 464--499.
- A. S. Almgren, J. B. Bell, P. Colella, L. H. Howell, and M. L. Welcome,
  [*A conservative adaptive projection method for the variable density
  incompressible Navier--Stokes equations*](https://doi.org/10.1006/jcph.1998.5890),
  Journal of Computational Physics 142 (1998), 1--46.
- L. Batteux, T. Gallouët, R. Herbin, J.-C. Latché, and P. Poullet,
  [*Convergence of the implicit MAC-discretized Navier--Stokes equations with
  variable density and viscosity on non-uniform
  grids*](https://arxiv.org/abs/2107.10117).
- U. Ghia, K. N. Ghia, and C. T. Shin, [*High-Re solutions for incompressible
  flow using the Navier--Stokes equations and a multigrid method*](https://doi.org/10.1016/0021-9991(82)90058-4),
  Journal of Computational Physics 48 (1982), 387--411.
- J. U. Brackbill, D. B. Kothe, and C. Zemach, [*A continuum method for
  modeling surface tension*](https://doi.org/10.1016/0021-9991(92)90240-Y),
  Journal of Computational Physics 100 (1992), 335--354.
- S. Popinet, [*An accurate adaptive solver for surface-tension-driven
  interfacial flows*](https://doi.org/10.1016/j.jcp.2009.04.042), Journal of
  Computational Physics 228 (2009), 5838--5866.
- M. Cai, A. Nonaka, J. B. Bell, B. E. Griffith, and A. Donev, [*Efficient
  variable-coefficient finite-volume Stokes
  solvers*](https://doi.org/10.4208/cicp.070114.170614a), Communications in
  Computational Physics 16 (2014), 1263--1297.
- T. C. Clevenger and T. Heister, [*Comparison between algebraic and
  matrix-free geometric multigrid for a Stokes problem on adaptive meshes with
  variable viscosity*](https://doi.org/10.1002/nla.2375), Numerical Linear
  Algebra with Applications 28 (2021), e2375.
- P. McCorquodale, P. Colella, and H. Johansen, [*A Cartesian grid embedded
  boundary method for the heat equation on irregular
  domains*](https://doi.org/10.1006/jcph.2001.6900), Journal of Computational
  Physics 173 (2001), 620--635.
