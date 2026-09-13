# Fixed-tree incompressible Navier--Stokes solver

The single-phase solver advances face velocity and cell pressure on a static
uniform or mixed-level tree. With convection operator `C_LW`, unit-viscosity
staggered Laplacian `L`, body force `f`, compatible divergence `D`, and
pressure gradient `G`, one time step has exactly three stages.

First, one Lax--Wendroff update supplies second-order space/time convection on
regular staggered regions:

```math
u^c = u^n - \Delta t\,C_{LW}(u^n) + \Delta t\,f^{n+1/2}.
```

The bulk and AMR-interface schemes both default to `LW()`. As for the public
advection API, `amr_interface_scheme` may be selected separately.

Second, viscosity is centred by Crank--Nicolson/implicit midpoint:

```math
\left(I-\frac{\Delta t\nu}{2}L\right)u^{**}
= u^c + \frac{\Delta t\nu}{2}Lu^n
  + \Delta t\nu b^{n+1/2}_{wall}.
```

The affine wall term contains both half-step contributions. The shifted
system is solved on the x- and y-dual grids by
`StaggeredGeometricMultigridSolver`; its default automatically chooses between
standalone V-cycles and V-cycle-preconditioned BiCGStab using the shifted
system's measured stiffness crossover. The unchanged hierarchy is reused for
a fixed `dt`, and LU appears only in the two bottom solves. Pass
`diffusion_method=:multigrid` or `:bicgstab` to override the automatic choice.

Finally, one pressure projection is performed:

```math
-DGp^{n+1} = -\frac{1}{\Delta t}Du^{**},
\qquad
u^{n+1}=u^{**}-\Delta t\,Gp^{n+1}.
```

There is no projection between convection and diffusion and no second
projection after the correction above. The same primary finite-volume `D`
and its dual-derived `G` are used in the Poisson matrix and velocity update.

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
    mesh; viscosity=0.01, boundary=lid,
    scheme=LW(), amr_interface_scheme=LW(),
    diffusion_method=:auto, pressure_method=:auto)
workspace = NavierStokesWorkspace(solver)
velocity = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
pressure = CellField(mesh)

result = simulate!(
    velocity, pressure, solver, 1.0, workspace;
    dt=0.002, rtol=1e-10, max_cycles=80)
```

`step!` advances one supplied `dt`; `simulate!` repeats the same operation to
a final time. If `dt` is omitted, the solver combines the LW CFL bound with a
viscous accuracy cap. Implicit midpoint is A-stable for the linear diffusion
stage, so that viscous bound is not a stability restriction.

Pass a `VisualizationSession` with `visualization=session` to receive copied
live frames and retain scheduled checkpoints. Nearest and non-intrusive exact
checkpoint policies are described in [Interactive visualization](visualization.md).
`simulate!` initializes the observer with an independent-step procedure but
does not inspect its checkpoint schedule or modify the main time-step sequence.

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
