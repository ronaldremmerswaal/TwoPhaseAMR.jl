# TwoPhaseAMR.jl

`TwoPhaseAMR.jl` is a tree-native, block-structured finite-volume foundation
for incompressible two-phase flow solvers in Julia. It currently implements the
first development milestone: conservative, cell-centred scalar transport on
2-D quadtree meshes.

The hot path works on dense, haloed block arrays and immutable exchange/face
schedules. Tree traversal happens only while constructing or regridding a
mesh. Coarse--fine faces have one owner, so the flux applied to the coarse cell
is exactly the sum of the equal-and-opposite fluxes applied to adjacent fine
cells.

## Implemented

- one-block, uniform multi-block, and balanced two-level meshes;
- explicit `Cell` and `Face{Axis}` placement types;
- automatic full 2:1 balancing around `RegionTrees.jl` leaves;
- same-level, periodic, coarse--fine, and physical halo filling;
- conservative restriction and piecewise-constant prolongation;
- prescribed face-normal velocities and first-order upwind scalar advection;
- allocation-free time-step, halo-fill, and CFL kernels after setup; and
- correctness, conservation, reference, allocation, and benchmark coverage.

Staggered transport, pressure projection, geometric VOF, and two-phase physics
are intentionally not advertised as implemented. Their required discrete
invariants are recorded in the [literature review](docs/src/literature-review.md)
and [architecture decision](docs/src/adr/0001-tree-backend.md).

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

Run the tests with `julia --project -e 'using Pkg; Pkg.test()'` and the
benchmarks with `julia --project benchmark/benchmarks.jl`.

## Development status

| Milestone | Status |
|---|---|
| 1. Centred single-phase foundation | Implemented for first-order upwind transport |
| 2. Staggered fields and compatible operators | Design invariants documented |
| 3. Pressure projection | Not implemented |
| 4. Geometric VOF | Compatibility gate documented, not implemented |
| 5. Two-phase Navier--Stokes | Not implemented |

See [DEVELOPMENT.md](DEVELOPMENT.md) for the complete roadmap.
