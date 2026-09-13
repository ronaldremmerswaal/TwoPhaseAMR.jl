# Milestone 1 benchmark results

Recorded 2026-09-13 with Julia 1.12.7, one Julia thread, on an AMD Ryzen 9
5950X. Values are warm medians from `BenchmarkTools`.

| Case | Kernel | Median time | Memory | Allocations |
|---|---:|---:|---:|---:|
| 1 block, 1,024 cells | mesh setup | 213.3 μs | 846,544 B | 1,761 |
| 1 block, 1,024 cells | advection step | 14.731 μs | 0 B | 0 |
| 1 block, 1,024 cells | halo fill | 0.471 μs | 0 B | 0 |
| 16 blocks, 4,096 cells | mesh setup | 1.447 ms | 5,357,208 B | 11,449 |
| 16 blocks, 4,096 cells | advection step | 61.012 μs | 0 B | 0 |
| 16 blocks, 4,096 cells | halo fill | 3.978 μs | 0 B | 0 |

Setup includes quadtree creation, balance checking, and construction of all
face and halo schedules. It is intentionally reported separately from the hot
time-stepping path. These values are a baseline rather than a cross-package
performance claim.

The milestone-1 performance contract enforced by tests is zero allocations for
`advect!`, `fill_halo!`, and `stable_timestep` after mesh/workspace setup.
Pressure-solve comparison against `IncompressibleNavierStokes.jl` belongs to
milestone 3 and has no fabricated result here.
