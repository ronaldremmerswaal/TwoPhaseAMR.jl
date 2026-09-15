# Adaptive-refinement benchmark

Recorded 2026-09-15 with Julia 1.12.7 on an AMD Ryzen 9 5950X, using one Julia
thread. The command was:

```sh
julia --project=. -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path="."); Pkg.add("BenchmarkTools"); include("benchmark/adaptivity.jl")'
```

The full-update case starts from 16 level-2 blocks of 16-by-16 cells. It carries
two cell fields, a two-component `FaceVelocity`, and two directional
`FaceField`s. Half of the checkerboard blocks request refinement to level 3;
the coarsening case starts from the resulting tree and requests coarsening for
the complete level-3 sibling families. Reported values are BenchmarkTools
medians; topology-changing samples use one evaluation and 20 samples.

| operation | median time | allocated memory | allocations |
|---|---:|---:|---:|
| cell `prolong!` (16-by-16 interior, four children) | 4.323 μs | 0 bytes | 0 |
| cell `restrict!` (four children, 16-by-16 parent) | 2.344 μs | 0 bytes | 0 |
| full refine with five registered field objects | 96.865 ms | 20.46 MiB | 30,103 |
| full coarsen with five registered field objects | 4.709 ms | 9.00 MiB | 13,006 |

The kernel result confirms that the direct cell transfer operators are
allocation-free. Full regridding is intentionally not: refining RegionTrees
nodes, reconstructing neighbour/face/halo schedules, and forming transfer
lookup structures dominate allocation. The much larger refinement time also
includes construction and transfer to newly created face and cell degrees of
freedom; discarded cell matrices are pooled for reuse by subsequent topology
changes.
