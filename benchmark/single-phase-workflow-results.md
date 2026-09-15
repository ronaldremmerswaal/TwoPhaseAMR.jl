# Single-phase workflow benchmark

Recorded on 2026-09-15 with Julia 1.12.7, one thread, and an AMD Ryzen 9
5950X. The static mixed mesh contains 1,792 cells. Values are medians; setup
and first-step cases use one evaluation per sample.

| Operation | Time | Allocated | Allocations |
|---|---:|---:|---:|
| `FlowSimulation` setup, known `dt` | 35.475 ms | 105.48 MiB | 291,984 |
| Setup + first step, unknown `dt` at setup | 48.951 ms | 134.05 MiB | 334,930 |
| Setup + first step, known `dt` at setup | 37.058 ms | 105.57 MiB | 292,020 |
| Warm low-level `step!` | 1.668 ms | 2.42 KiB | 17 |
| Warm one-step high-level `simulate!` | 1.675 ms | 2.42 KiB | 17 |

Passing the fixed step when the workspace is built removes construction of a
zero-shift diffusion hierarchy that the first step would replace. In this
case it reduces setup-plus-first-step time by 24%, allocated bytes by 21%, and
allocation count by 13%. It also seeds replacement workspaces after a dynamic
regrid. The high-level continuation driver adds no measurable allocation and
less than 1% time over the underlying warm step in this measurement.

The accompanying flat sampling profile identifies shifted staggered
geometric-multigrid construction as the dominant mixed-level setup cost.
During 200 fixed-topology steps, sparse matrix-vector products, Krylov CG and
BiCGSTAB, and geometric V-cycles dominate; the high-level API is not a hot
section.

Reproduce with the command in [`README.md`](README.md). Results are
machine-dependent and serve as a regression baseline, not a cross-machine
performance guarantee.
