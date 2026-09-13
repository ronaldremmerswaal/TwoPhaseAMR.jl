# Milestone 1 and staggered-operator benchmark results

Recorded 2026-09-13 with Julia 1.12.7, one Julia thread, on an AMD Ryzen 9
5950X. Values are warm medians from `BenchmarkTools`.

| Case | Kernel | Median time | Memory | Allocations |
|---|---:|---:|---:|---:|
| 1 block, 1,024 cells | mesh setup | 166.149 μs | 846,544 B | 1,761 |
| 1 block, 1,024 cells | staggered schedule setup | 116.733 μs | 492,848 B | 40 |
| 1 block, 1,024 cells | gradient setup | 1.980 ms | 5,843,696 B | 47,216 |
| 1 block, 1,024 cells | cell advection step | 14.711 μs | 0 B | 0 |
| 1 block, 1,024 cells | one-component staggered LW step | 10.290 μs | 0 B | 0 |
| 1 block, 1,024 cells | two-component staggered LW step | 25.340 μs | 0 B | 0 |
| 1 block, 1,024 cells | pressure gradient | 3.874 μs | 0 B | 0 |
| 1 block, 1,024 cells | halo fill | 0.461 μs | 0 B | 0 |
| 16 blocks, 4,096 cells | mesh setup | 997.511 μs | 5,357,208 B | 11,449 |
| 16 blocks, 4,096 cells | staggered schedule setup | 654.432 μs | 1,967,408 B | 40 |
| 16 blocks, 4,096 cells | gradient setup | 7.109 ms | 23,582,808 B | 188,546 |
| 16 blocks, 4,096 cells | cell advection step | 60.322 μs | 0 B | 0 |
| 16 blocks, 4,096 cells | one-component staggered LW step | 40.881 μs | 0 B | 0 |
| 16 blocks, 4,096 cells | two-component staggered LW step | 107.143 μs | 0 B | 0 |
| 16 blocks, 4,096 cells | pressure gradient | 18.491 μs | 0 B | 0 |
| 16 blocks, 4,096 cells | halo fill | 3.979 μs | 0 B | 0 |
| Mixed levels, 1,792 cells | staggered schedule setup | 1.113 ms | 2,686,432 B | 3,699 |
| Mixed levels, 1,792 cells | one-component staggered upwind step | 19.680 μs | 0 B | 0 |
| Mixed levels, 1,792 cells | two-component staggered upwind step | 48.241 μs | 0 B | 0 |
| Mixed levels, 1,792 cells | gradient setup | 3.660 ms | 10,536,480 B | 83,907 |
| Mixed levels, 1,792 cells | pressure gradient | 14.070 μs | 0 B | 0 |

Mesh setup includes quadtree creation, balance checking, and construction of
all centred face and halo schedules. Staggered setup builds one component's
dual-control-volume schedule from that immutable mesh. On mixed levels this
includes the side-fraction incidence and local conservative factorization of
the primal flux graph. Both are intentionally reported separately from the
hot time-stepping path. These values are a baseline rather than a cross-package
performance claim.

The performance contract enforced by tests is zero allocations for cell and
staggered `advect!`, staggered `convection!`, `divergence!`, `gradient!`,
`fill_halo!`, and both CFL kernels after mesh/workspace setup. Staggered entries
above use the multidimensional Lax--Wendroff interpolant; the one-component
case advances one `FaceField{1}`, and the two-component case advances a
`FaceVelocity` from the same old-time advecting field. Gradient setup assembles
the vertex-patch map after mesh construction; the uniform hot path stores only
its diagonal, while the mixed-level hot path applies a sparse map. The
mixed-level staggered entries use the robust upwind bulk/interface choice.

Pressure-solve comparison against `IncompressibleNavierStokes.jl` belongs to
milestone 3 and has no fabricated result here.
