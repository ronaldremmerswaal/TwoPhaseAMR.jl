# Shared-memory core-kernel scaling

## Run configuration

- Date: 2026-09-16
- Julia: 1.12.7
- CPU: AMD Ryzen 9 5950X, 16 physical cores / 32 hardware threads
- Julia compute threads: 1, 2, 4, and 8
- BLAS threads: 1
- Samples: three timed samples after one warm call; reported time is the median
- Uniform case: 256 blocks of 8×8 cells, 16,384 cells
- Mixed case: 436 blocks of 8×8 cells, 27,904 cells
- Pressure density ratio: 1000; viscosity pair: `(0.01, 1e-4)`

Setup is outside steady-kernel timing. Every result includes allocated bytes,
allocation count, and a numerical checksum. The driver compared each checksum
with the one-thread result and all comparisons passed. Raw data are in
[`threaded-kernel-results/scaling.csv`](threaded-kernel-results/scaling.csv).

Three samples are sufficient to identify the large effects discussed below,
but sub-millisecond results remain sensitive to CPU frequency and operating
system noise. Use at least ten samples and thread pinning for publication-grade
numbers.

## Numerical kernels

Speedup is relative to the one-thread process.

| Kernel | Uniform 1T | Uniform 4T | Uniform 8T | Mixed 1T | Mixed 4T | Mixed 8T |
|---|---:|---:|---:|---:|---:|---:|
| Cell initialization | 0.246 ms | 2.86× | 3.69× | 0.427 ms | 2.52× | 4.19× |
| Level-set interface initialization | 11.770 ms | 2.00× | 3.08× | 24.351 ms | 2.70× | 4.74× |
| Interface reconstruction | 5.816 ms | 2.25× | 3.54× | 12.208 ms | 2.56× | 3.87× |
| Height sampling | 0.350 ms | 1.96× | 4.03× | 0.920 ms | 1.53× | 4.32× |
| Cell advection | 0.219 ms | 1.89× | 2.01× | 0.397 ms | 2.17× | 3.55× |
| Staggered velocity advection | 1.993 ms | 2.02× | 2.68× | 3.914 ms | 1.90× | 2.25× |
| WY interface advection | 20.007 ms | 2.46× | 3.56× | 40.170 ms | 2.62× | 4.30× |
| WY phase-scalar advection | 27.092 ms | 2.35× | 3.81× | 51.146 ms | 2.40× | 4.17× |
| WY momentum advection | 27.081 ms | 2.37× | 3.43× | 52.884 ms | 2.52× | 3.53× |
| Pressure gradient | 0.111 ms | 1.06× | 1.34× | 0.346 ms | 1.77× | 1.52× |
| Poisson operator application | 0.261 ms | 1.54× | 2.59× | 0.526 ms | 2.34× | 2.17× |
| Poisson solve | 9.745 ms | 1.85× | 1.93× | 24.770 ms | 2.25× | 1.91× |
| Diffusion-coefficient refresh | 0.968 ms | 1.80× | 2.16× | 2.006 ms | 2.32× | 2.03× |
| Diffusion operator application | 0.246 ms | 1.58× | 1.68× | 0.459 ms | 2.30× | 1.84× |
| Diffusion solve | 8.414 ms | 1.53× | 1.39× | 18.998 ms | 1.29× | 1.52× |
| Gravity | 2.395 ms | 2.06× | 3.30× | 4.785 ms | 2.13× | 3.68× |

Interface geometry and transport scale best. The face-flux kernels use a
race-free two-stage algorithm: faces own flux evaluation and cells or dual
volumes own the gather. This removes atomics and makes the result independent
of scheduling. The mixed-level interface kernels scale at least as well as the
uniform versions. The extra work near refinement boundaries is distributed by
the fine-grained batches, so these data do not justify a separate
interface-weighted load balancer.

The elliptic solves improve less than their fine-grid operator applications.
Coarse multigrid levels are intentionally serial below the grain cutoff, and
Krylov norms and compatibility reductions retain serial work. The pressure
solve is best at eight threads on the uniform case and four on the mixed case;
the differences after four threads are small. Diffusion also plateaus: four
threads is best for the uniform case, while eight narrowly wins on the mixed
case. Each assembled multigrid level stores a CSC row-form mirror so its
parallel action computes `A*x` for nonsymmetric boundary operators rather than
assuming that `A == A'`.

CSF surface tension takes only 0.25--0.53 ms and shows no useful speedup. Its
curvature-validation loop remains serial so it can issue a precise error for
an unresolved face; threading this sub-millisecond operation would not improve
the measured total. Gravity has enough independent face/cell work to scale.

## Construction, adaptation, and transfer

| Kernel | 1T | 2T | 4T | 8T | 1T allocated |
|---|---:|---:|---:|---:|---:|
| Uniform mesh construction | 4.276 ms | 1.34× | 1.53× | 1.34× | 14.56 MiB |
| Mixed mesh construction | 9.540 ms | 1.03× | 1.23× | 1.21× | 39.51 MiB |
| Refine mesh and transfer fields | 25.649 ms | 1.03× | 1.33× | 1.42× | 40.14 MiB |
| Coarsen mesh and transfer fields | 13.672 ms | 1.09× | 1.56× | 1.69× | 11.35 MiB |
| Prolong one 8×8 block | 0.003 ms | 0.95× | 1.00× | 0.96× | 0 B |
| Restrict one 8×8 block | 0.001 ms | 1.03× | 1.01× | 0.86× | 0 B |
| Prolong one 128×128 block | 0.698 ms | 1.69× | 3.31× | 5.92× | 0 B |
| Restrict one 128×128 block | 0.184 ms | 1.46× | 2.09× | 3.06× | 0 B |

Mesh construction parallelizes the independent tree searches used to discover
halo donors. Face-schedule assembly, tree mutation, 2:1 balancing, and final
vector compaction remain serial, which caps speedup near 1.5×. Ordinary Julia
dynamic tasks work better for the pointer-rich tree search than Polyester:
the initial Polyester version copied/lowered heterogeneous captured state,
increased allocation by tens of MiB, and slowed construction. The final task
version holds allocation essentially constant between one and eight threads.

An individual 8×8 transfer contains too little work to thread profitably. AMR
therefore schedules independent blocks, while direct transfer within a large
block is tiled. Refinement gains less than coarsening because it also rebuilds
serial topology schedules and performs geometric interface reconstruction.

## Remaining setup bottleneck

| Constructor | Uniform time / allocation | Mixed time / allocation | 8T speedup |
|---|---:|---:|---:|
| Pressure-gradient geometry | 14.889 ms / 48.75 MiB | 26.456 ms / 82.67 MiB | 0.98× / 0.84× |
| Variable Poisson operator | 38.332 ms / 116.42 MiB | 72.543 ms / 196.01 MiB | 0.96× / 0.77× |
| Variable diffusion operator | 115.428 ms / 126.87 MiB | 230.814 ms / 245.59 MiB | 1.01× / 0.94× |

These constructors still dominate setup and do not scale. They discover
geometric patches, assemble sparse coordinate lists, sort/group contributions,
and create CSC matrices. The numerical refresh and matrix-free/factorized
actions are threaded, but the symbolic constructor is still a serial sparse
assembly pipeline. Threading the later loops cannot hide this cost; rebuilding
less state after AMR and retaining patch schedules remains the higher-value
optimization.

Steady operator applications are effectively allocation-free on one thread.
Polyester adds small fixed task metadata on threaded calls (typically below
2 KiB per scalar operator and about 10 KiB for the two-component staggered
advection call). WY calls allocate 0.6--2.8 MiB because their public operation
currently creates per-step scalar/source fields; those allocations exist on
one thread as well and rise only modestly with thread count. Constructor
allocation is unchanged by thread count in the final implementation.

## Decision before Stokes-3

1. Use block/cell tiling without a cost-weighted interface load balancer. The
   mixed interface results show no load-balance penalty through eight threads.
2. Prefer four threads as the conservative default for these 16k--28k-cell
   elliptic solves. Interface transport benefits from eight threads at both
   tested sizes.
3. Keep 8×8 block transfers serial internally and parallelize across blocks.
4. Treat symbolic pressure/diffusion construction as the main remaining AMR
   bottleneck. The next optimization should reuse construction schedules and
   update only changed patches, rather than adding threads around global sparse
   assembly.
5. Proceed to the matched Stokes-3 `t=1` scaling run only after the full test
   suite passes; use the same mesh evolution at every thread count.
