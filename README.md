# TwoPhaseAMR.jl

`TwoPhaseAMR.jl` is a tree-native, block-structured finite-volume foundation
for incompressible two-phase flow solvers in Julia. It implements the first
four development milestones on uniform and mixed-level 2-D quadtree meshes.

Cell kernels work on dense, haloed block arrays; owned face kernels work on
compact contiguous arrays and immutable dual-face schedules. Tree traversal
happens only while constructing or regridding a mesh. Coarse--fine centred
faces have one owner, so the flux applied to the coarse cell is exactly the sum
of the equal-and-opposite fluxes applied to adjacent fine cells.

## Implemented

- one-block, uniform multi-block, and balanced two-level meshes;
- explicit `Cell` and `Face{Axis}` placement types;
- automatic full 2:1 balancing around `RegionTrees.jl` leaves;
- same-level, periodic, coarse--fine, and physical halo filling;
- conservative restriction and piecewise-constant prolongation;
- prescribed face-normal velocities and first-order upwind scalar advection;
- owned directional `FaceField{Axis}` degrees of freedom;
- algebraically interpolated dual-control-volume fluxes for upwind, centred,
  and Lax--Wendroff staggered advection on uniform and mixed-level meshes;
- an independent `amr_interface_scheme` selection that defaults to `scheme`;
- simultaneous conservative transport of both staggered velocity components;
- exact recovery of the interpolated primary divergence from constant-state
  convection and twice the all-centred convection-matrix diagonal;
- a centred convection operator that is volume-weighted skew-symmetric for
  discretely divergence-free velocity on periodic mixed-level trees;
- a divergence-derived pressure gradient whose sparse SPD vertex-patch Riesz
  map is affine-exact on isotropic mixed-level meshes;
- a tree-geometric/Galerkin multigrid solver, reusable shifted elliptic
  operator, and pressure projection with variable face coefficients;
- V-cycle-preconditioned CG as the default symmetric elliptic solve, with the
  standalone V-cycle path retained for comparison and asymmetric smoothing;
- automatic no-permeation on every non-periodic boundary and the corresponding
  homogeneous-Neumann pressure-correction closure;
- no-slip (default), moving-wall, and free-slip staggered diffusion closures;
- widened, convergent AMR-interface diffusion stencils and component-wise
  staggered geometric multigrid, with LU used only on the bottom grids;
- a fixed-tree incompressible Navier--Stokes solver with one LW convection
  update, one implicit-midpoint diffusion solve, and one projection per step;
- Taylor--Green-vortex and lid-driven-cavity validation cases;
- allocation-free time-step, halo-fill, and CFL kernels after setup; and
- correctness, conservation, reference, allocation, and benchmark coverage.

General pressure Dirichlet data, geometric VOF, and two-phase physics are not
yet implemented. Their required discrete invariants are recorded in the
[literature review](docs/src/literature-review.md) and
[architecture decision](docs/src/adr/0001-tree-backend.md).

## Quick start

```julia
using TwoPhaseAMR

mesh = BlockTreeMesh(
    block_size=(16, 16),
    max_level=2,
    refine=(key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
)

q = CellField(mesh)
initialize!(q, (x, y) -> 1 + 0.2sinpi(2x) * cospi(2y))

u = FaceVelocity(mesh, (x, y, time, axis) -> axis == 1 ? 0.3 : -0.2)
workspace = AdvectionWorkspace(q)
dt = stable_timestep(u, workspace; cfl=0.5)

mass_before = integrate(q)
advect!(q, u, dt, workspace)
@assert integrate(q) ≈ mass_before
```

Directional staggered transport uses the same owned face-normal velocity:

```julia
staggered_mesh = BlockTreeMesh(block_size=(16, 16))
staggered_velocity = FaceVelocity(
    staggered_mesh, (x, y, time, axis) -> axis == 1 ? 0.3 : -0.2)
momentum_x = FaceField{1}(
    staggered_mesh, (x, y) -> sinpi(2x) * cospi(2y))
staggered_workspace = StaggeredAdvectionWorkspace(momentum_x)
dt = stable_timestep(staggered_velocity, staggered_workspace; cfl=0.5)

advect!(momentum_x, staggered_velocity, dt, staggered_workspace)
```

`Upwind()`, `Centred()` (also `Centered()`), and `LaxWendroff()` (also `LW()`)
are supported for both `scheme` and `amr_interface_scheme`. Staggered
convection and advection default to `LW()` everywhere:
`amr_interface_scheme` inherits `scheme` when omitted. Explicitly pass
`amr_interface_scheme=Upwind()` for the more dissipative hybrid closure. The
global-LW default gives roughly second-order mixed-level convergence rather
than the hybrid closure's observed half-order convergence, but has a
documented one-step spectral radius slightly above one. Staggered advection
accepts uniform and 2:1-balanced mixed-level meshes. Non-periodic sides have
zero normal advective flux. Cell-centred scalar advection remains the separate
first-order upwind method described above.

The single-phase solver uses the requested three-part step:

```julia
mesh = BlockTreeMesh(block_size=(32, 32), periodic=(false, false))
lid = NoSlip((x, y, time, axis) -> axis == 1 && y > 0.5 ? 1.0 : 0.0)
solver = NavierStokesSolver(mesh; viscosity=0.01, boundary=lid)
workspace = NavierStokesWorkspace(solver)
velocity = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
pressure = CellField(mesh)

simulate!(velocity, pressure, solver, 1.0, workspace; dt=0.002)
```

Convection defaults to global `LW()`. Diffusion is Crank--Nicolson/implicit
midpoint and is solved by staggered geometric multigrid. Each `step!` then
performs exactly one compatible pressure projection.

Run the tests with `julia --project -e 'using Pkg; Pkg.test()'`. Benchmark
environment instructions are in [benchmark/README.md](benchmark/README.md).
The block-size type-parameter experiment prompted by `RESEARCH.md` is recorded
in [benchmark/block-size-results.md](benchmark/block-size-results.md), and the
standalone/PCG comparison is recorded in
[benchmark/pcg-multigrid-results.md](benchmark/pcg-multigrid-results.md).

## Development status

| Milestone | Status |
|---|---|
| 1. Centred single-phase foundation | Implemented for first-order upwind transport |
| 2. Staggered fields and compatible operators | Implemented for static, globally stepped uniform and mixed-level meshes |
| 3. Pressure projection | Implemented for periodic and impermeable-wall uniform/isotropic mixed-level meshes |
| 4. Wall diffusion and single-phase Navier--Stokes | Implemented on fixed uniform and mixed-level trees |
| 5. Geometric VOF | Compatibility gate documented, not implemented |
| 6. Two-phase Navier--Stokes | Not implemented |

See [DEVELOPMENT.md](DEVELOPMENT.md) for the complete roadmap.
