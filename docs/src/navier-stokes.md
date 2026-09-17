# Incompressible Navier--Stokes solver

One solver advances face velocity and cell pressure on a static or dynamically
adapted uniform/mixed-level tree. Supplying an interface selects one-velocity
two-phase coefficients and compatible geometric transport; the outer time-step
sequence stays the same.

## Common step

Each step executes transport, midpoint body forcing, implicit diffusion,
optional phase forces, and one pressure projection. Dispatch changes the
kernel inside a stage rather than selecting a second time integrator.

For single-phase flow, the transport and user-force state is

```math
u^c = u^n - \Delta t\,C_{LW}(u^n) + \Delta t\,f^{n+1/2}.
```

The bulk scheme defaults to `LW()` and the scheme-aware AMR-interface default
selects `MC()` on non-regular coarse--fine connections. Global Upwind retains
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
transported state before diffusion in both models. Single-phase viscosity is
then centred by Crank--Nicolson/implicit midpoint:

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
stresses, the dynamic viscosity is the thesis geometric mixture

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
body force applied before diffusion, matching the split state used in the
thesis stress `mu*symgrad(u* + u**)`.

Balanced CSF or GFM surface tension and gravity are added after diffusion.
They use the same compatible weighted-gradient operator as the following
pressure correction. Finally, both models perform one projection:

```math
-DGp^{n+1} = -\frac{1}{\Delta t}Du^{**},
\qquad
u^{n+1}=u^{**}-\Delta t\,Gp^{n+1}.
```

For two phases, `G` includes the face coefficient `1/<rho>`, where
`<rho>=rho_l*c_f+rho_g*(1-c_f)`. There is no projection between convection and diffusion and no second
projection after the correction above. The same primary finite-volume `D`
and its dual-derived `G` are used in the Poisson matrix and velocity update.

## Dimensionless solver tolerance

`step!` and `simulate!` expose one dimensionless `tolerance`, shared by the
pressure and diffusion solves. They do not expose algebraic `rtol` or
dimensional `atol` values.

For pressure, the implementation follows Basilisk. If `r_p` is the strong
Poisson residual, the requested stopping condition is

```math
\Delta t^2\lVert r_p\rVert_\infty \leq \mathtt{tolerance}.
```

This quantity is the maximum fractional cell-volume defect that the remaining
velocity divergence would produce during one timestep. It remains meaningful
when the provisional velocity is already nearly divergence free and does not
depend on the initial pressure residual.

For diffusion, the integrated momentum residual is divided by the local
staggered inertia. This gives a velocity defect. Its infinity norm is divided
by the corresponding strong right-hand-side velocity scale, so the same
dimensionless `tolerance` applies to both single- and variable-density solves.
This normalization depends on the equation data, not on the initial guess.
Both criteria are raised to a computed floating-point roundoff floor when the
requested tolerance cannot be represented reliably.

`LinearSolverStats` reports the initial and final L2 and infinity norms, the
effective infinity-norm target, the residual scale, iterations, removed
pressure-RHS mean, Krylov method, and preconditioner. A failed high-level solve
includes all of these fields, the requested tolerance, achieved dimensionless
residual, iteration cap, timestep, and simulation time in its exception.

## Boundary conditions

Every non-periodic side automatically imposes no permeation, `u dot n = 0`.
The associated pressure *correction* has homogeneous Neumann data, so the
projection cannot change the prescribed wall-normal velocity. The all-Neumann
pressure nullspace is fixed by a volume-weighted zero mean.

The viscous closure is independent:

- `NoSlip()` is the default and uses zero tangential wall velocity;
- `NoSlip(wall_velocity)` supports a moving wall such as a cavity lid; and
- `FreeSlip()` (alias `Slip()`) imposes zero tangential shear.

This is an incremental pressure-correction formulation. Accurate physical
pressure boundary values, especially at no-slip walls, require a separate
boundary-accuracy study before pressure itself is used as a validation
observable. Brown, Cortez, and Minion explain this distinction for
second-order projection methods.

## API

```julia
mesh = BlockTreeMesh(block_size=(32, 32), periodic=(false, false))
lid = NoSlip((x, y, time, axis) ->
    axis == 1 && y > 0.5 ? 1.0 : 0.0)

solver = NavierStokesSolver(
    mesh; density=1.0, dynamic_viscosity=0.01, boundary=lid,
    scheme=LW()) # MC is selected automatically at AMR interfaces
workspace = NavierStokesWorkspace(solver)
velocity = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
pressure = CellField(mesh)

result = simulate!(
    velocity, pressure, solver, 1.0, workspace;
    dt=0.002, tolerance=1e-8, max_cycles=80)
```

The corresponding two-phase setup changes only constructor keywords:

```julia
interface = InterfaceField(
    mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.15)
flow = FlowSimulation(
    mesh;
    interface,
    density=(1000.0, 1.0),
    dynamic_viscosity=(1e-3, 1.8e-5),
    surface_tension=0.072,
    surface_tension_formulation=GFM(),
    gravity=(0.0, -9.81),
    boundary=FreeSlip(),
    dt=1e-4,
)
```

Pressure and coupled two-phase stress diffusion default to preconditioned
Krylov.jl CG because their operators are symmetric positive (semi-)definite.
Single-phase staggered diffusion selects CG for a symmetric operator and
BiCGSTAB when the AMR consistency correction makes it nonsymmetric. Pressure
and diffusion use rediscretized tree-geometric hierarchies for both
single-phase and two-phase flow. The single-phase diffusion hierarchy keeps
separate x/y component operators; the two-phase hierarchy retains the coupled
stress operator. Both can be configured independently. The root-block sparse
operator is the bottom solve even when its unknown count exceeds
`coarse_size`:

```julia
import AlgebraicMultigrid, IncompleteLU, Krylov

solver = NavierStokesSolver(
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

`step!` advances one supplied `dt`; `simulate!` repeats the same operation to
a final time. If `dt` is omitted, the solver uses only the staggered convective
CFL bound. The implicit-midpoint diffusion stage imposes no diffusion-based
timestep restriction; pass `dt` explicitly when a particular temporal
accuracy resolution is required. The distinction between A-stability,
stiff-mode damping, and linear-solver efficiency is developed in the
[implicit-diffusion solver review](implicit-diffusion-solver-review.md).

Pass a `VisualizationSession` with `visualization=session` to receive copied
live frames and retain scheduled checkpoints. Nearest and non-intrusive exact
checkpoint policies are described in [Interactive visualization](visualization.md).
`simulate!` initializes the observer with an independent-step procedure but
does not inspect its checkpoint schedule or modify the main time-step sequence.

Dynamic refinement is enabled by passing `refinement_indicators` and optional
additional `adaptive_fields` to the solver constructor. Velocity and pressure
are always transferred. Adaptation happens after a completed physical step,
subject to `mesh.adaptation.options.update_interval`; it performs no additional
time step or pressure correction. Topology-dependent operators and workspaces
are rebuilt only when the leaf set changes. See
[Dynamic adaptive refinement](adaptive-refinement.md) for the API, transfer
invariants, cooldown, benchmarks, and exact-checkpoint limitation.

## Validation

The periodic Taylor--Green vortex checks the complete convection, midpoint
diffusion, and one-projection sequence. With equal-CFL refinement, the current
uniform test requires better than order 1.8 velocity convergence and the
mixed-level test checks both velocity error and projected divergence.

The lid-driven cavity is advanced at `Re=100` on a 32×32 cell mesh to
`t=15`, followed by one additional time unit to verify that the maximum
velocity change is below `1e-4`. Linear interpolation to all 15 non-boundary
locations in Tables I and II of Ghia, Ghia, and Shin gives the current errors:

| Profile | RMS error | Maximum error |
|---|---:|---:|
| `u(0.5,y)` | 1.67e-3 | 2.96e-3 |
| `v(x,0.5)` | 3.79e-3 | 8.12e-3 |

The same regression checks the moving no-slip lid, stationary walls,
all-Neumann pressure correction, and divergence after projection. Ghia's
reference values came from the paper's 129×129 steady solution; the test
tolerances include the expected error from the coarser 32×32 mesh and linear
centerline sampling.

References:

- [Brown, Cortez, and Minion, accurate projection methods](https://www.ljll.fr/~frey/papers/Navier-Stokes/Brown%20D.L.%2C%20Accurate%20projection%20methods%20for%20the%20incompressible%20Navier-Stokes%20equations.pdf)
- [van der Plas, local grid refinement for free-surface flow](https://research.rug.nl/en/publications/local-grid-refinement-for-free-surface-flow-simulations/)
- [Ghia, Ghia, and Shin, high-resolution cavity flow](https://doi.org/10.1016/0021-9991(82)90058-4)
