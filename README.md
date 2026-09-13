# TwoPhaseAMR.jl

`TwoPhaseAMR.jl` is a tree-native, block-structured finite-volume foundation
for incompressible two-phase flow solvers in Julia. It implements the first
development milestone and the periodic staggered-field/operator milestone on
uniform and mixed-level 2-D quadtree meshes.

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
- an independent `amr_interface_scheme` selection, defaulting to upwind;
- simultaneous conservative transport of both staggered velocity components;
- exact recovery of the interpolated primary divergence from constant-state
  convection and twice the all-centred convection-matrix diagonal;
- a centred convection operator that is volume-weighted skew-symmetric for
  discretely divergence-free velocity on periodic mixed-level trees;
- a divergence-derived pressure gradient whose sparse SPD vertex-patch Riesz
  map is affine-exact on isotropic mixed-level periodic meshes;
- allocation-free time-step, halo-fill, and CFL kernels after setup; and
- correctness, conservation, reference, allocation, and benchmark coverage.

Physical staggered/pressure boundaries, pressure projection, geometric VOF,
and two-phase physics are intentionally not advertised as implemented. Their
required discrete invariants are recorded in the
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
convection and advection default to bulk `LW()` and interface `Upwind()`. The
interface default is more dissipative because the current AMR LW closure has a
one-step spectral radius slightly above one in the documented mixed-level
experiment. Staggered advection accepts uniform and 2:1-balanced mixed-level
meshes with periodic boundaries in both directions. Cell-centred scalar
advection remains the separate first-order upwind method described above.

Run the tests with `julia --project -e 'using Pkg; Pkg.test()'`. Benchmark
environment instructions are in [benchmark/README.md](benchmark/README.md).
The block-size type-parameter experiment prompted by `RESEARCH.md` is recorded
in [benchmark/block-size-results.md](benchmark/block-size-results.md).

## Development status

| Milestone | Status |
|---|---|
| 1. Centred single-phase foundation | Implemented for first-order upwind transport |
| 2. Staggered fields and compatible operators | Implemented for static, globally stepped periodic uniform and mixed-level meshes; physical boundary dual volumes pending |
| 3. Pressure projection | Not implemented |
| 4. Geometric VOF | Compatibility gate documented, not implemented |
| 5. Two-phase Navier--Stokes | Not implemented |

See [DEVELOPMENT.md](DEVELOPMENT.md) for the complete roadmap.
