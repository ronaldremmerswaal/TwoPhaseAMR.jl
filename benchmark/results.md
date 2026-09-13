# Milestone 1 and staggered-advection benchmark results

Recorded 2026-09-13 with Julia 1.12.7, one Julia thread, on an AMD Ryzen 9
5950X. Values are warm medians from `BenchmarkTools`.

| Case | Kernel | Median time | Memory | Allocations |
|---|---:|---:|---:|---:|
| 1 block, 1,024 cells | mesh setup | 213.681 μs | 846,544 B | 1,761 |
| 1 block, 1,024 cells | staggered schedule setup | 200.865 μs | 476,024 B | 35 |
| 1 block, 1,024 cells | cell advection step | 14.721 μs | 0 B | 0 |
| 1 block, 1,024 cells | one-component staggered LW step | 10.360 μs | 0 B | 0 |
| 1 block, 1,024 cells | two-component staggered LW step | 25.531 μs | 0 B | 0 |
| 1 block, 1,024 cells | halo fill | 0.467 μs | 0 B | 0 |
| 16 blocks, 4,096 cells | mesh setup | 1.410 ms | 5,357,208 B | 11,449 |
| 16 blocks, 4,096 cells | staggered schedule setup | 192.219 μs | 1,901,432 B | 35 |
| 16 blocks, 4,096 cells | cell advection step | 60.531 μs | 0 B | 0 |
| 16 blocks, 4,096 cells | one-component staggered LW step | 41.151 μs | 0 B | 0 |
| 16 blocks, 4,096 cells | two-component staggered LW step | 107.232 μs | 0 B | 0 |
| 16 blocks, 4,096 cells | halo fill | 3.935 μs | 0 B | 0 |

Mesh setup includes quadtree creation, balance checking, and construction of
all centred face and halo schedules. Staggered setup builds one component's
dual-control-volume and multidimensional LW-neighbour schedules from that
immutable mesh. Both are intentionally reported separately from the hot
time-stepping path. These values are a baseline rather than a cross-package
performance claim.

The performance contract enforced by tests is zero allocations for cell and
staggered `advect!`, staggered `convection!`, `divergence!`, `fill_halo!`, and
both CFL kernels after mesh/workspace setup. Staggered entries above use the
multidimensional Lax--Wendroff interpolant; the one-component case advances
one `FaceField{1}`, and the two-component case advances a `FaceVelocity` from
the same old-time advecting field.

Pressure-solve comparison against `IncompressibleNavierStokes.jl` belongs to
milestone 3 and has no fabricated result here.
