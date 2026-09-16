# Shared-memory core-kernel scaling

## Run configuration

- Date: 2026-09-16
- Julia: 1.12.7
- CPU: AMD Ryzen 9 5950X, 16 physical cores / 32 hardware threads
- Julia compute threads: 1, 2, 4, and 8 in separate processes
- BLAS threads: 1
- Samples: seven timed samples after one warm call
- Uniform case: 256 blocks of 8x8 cells, 16,384 cells
- Mixed case: 436 blocks of 8x8 cells, 27,904 cells
- Pressure density ratio: 1,000; viscosity pair: `(0.01, 1e-4)`

Setup is outside steady-kernel timing. Every result includes allocated bytes,
allocation count, and a numerical checksum. The driver compared each checksum
with the one-thread result and all comparisons passed. An independent
eight-thread process used 11 samples; its checksums also passed. Raw results
are in
[`threaded-kernel-results-block-local`](threaded-kernel-results-block-local),
including the [repeat comparison](threaded-kernel-results-block-local/eight-thread-repeat-comparison.csv).

Sub-millisecond measurements remain sensitive to CPU scheduling and frequency.
The repeat is therefore used to distinguish a stable conclusion from a noisy
single median. End-to-end conclusions use the separately repeated Stokes-3
run.

## Numerical kernels

Speedup is relative to the one-thread process from this run.

| Kernel | Uniform 1T | Uniform 4T | Uniform 8T | Mixed 1T | Mixed 4T | Mixed 8T |
|---|---:|---:|---:|---:|---:|---:|
| Cell initialization | 0.167 ms | 1.82x | 3.34x | 0.287 ms | 1.85x | 3.57x |
| Level-set interface initialization | 10.814 ms | 2.14x | 2.84x | 21.987 ms | 2.44x | 3.89x |
| Interface reconstruction | 4.898 ms | 2.31x | 2.94x | 10.287 ms | 2.59x | 3.96x |
| Height sampling | 0.349 ms | 2.13x | 2.61x | 0.899 ms | 2.46x | 2.08x |
| Cell advection | 0.191 ms | 2.11x | 2.35x | 0.351 ms | 2.40x | 2.51x |
| Staggered velocity advection | 2.096 ms | 1.90x | 1.89x | 4.000 ms | 2.06x | 2.25x |
| WY interface advection | 17.033 ms | 1.88x | 3.23x | 35.756 ms | 2.62x | 3.83x |
| WY phase-scalar advection | 20.388 ms | 2.20x | 3.34x | 43.042 ms | 2.77x | 4.02x |
| WY momentum advection | 23.581 ms | 2.28x | 3.04x | 46.324 ms | 2.37x | 3.33x |
| Pressure gradient | 0.118 ms | 2.29x | 2.28x | 0.374 ms | 2.11x | 3.38x |
| Poisson operator application | 0.233 ms | 1.89x | 2.46x | 0.456 ms | 2.11x | 2.26x |
| Poisson solve | 10.080 ms | 1.98x | 2.19x | 24.742 ms | 1.72x | 2.28x |
| Diffusion-coefficient refresh | 0.851 ms | 1.86x | 1.90x | 1.915 ms | 1.81x | 2.25x |
| Diffusion operator application | 0.256 ms | 1.59x | 2.21x | 0.472 ms | 1.54x | 2.14x |
| Diffusion solve | 8.580 ms | 1.28x | 1.37x | 19.165 ms | 1.55x | 1.58x |
| CSF surface tension | 0.279 ms | 0.98x | 1.17x | 0.509 ms | 0.86x | 0.94x |
| Gravity | 3.107 ms | 2.21x | 4.82x | 4.766 ms | 2.24x | 4.45x |

At the stated 4x threshold, gravity is the only numerical kernel that passes
in both cases. Mixed-mesh WY phase-scalar advection also passes. The repeat
moves several borderline results but not this conservative classification:
mixed WY interface advection ranges from 3.83x to 4.23x, while mixed interface
reconstruction ranges from 3.85x to 3.96x.

The block-local change improves serial work enough that its speedup ratio can
fall even when threaded elapsed time improves. For example, uniform WY
phase-scalar advection drops from 27.092 to 20.388 ms on one thread and from
7.102 to 6.099 ms on eight threads. Its reported speedup consequently changes
from 3.81x to 3.34x despite both executions becoming faster. The serial report
and two independent runs are in
[`data-locality-results.md`](data-locality-results.md).

Eight-thread repeat variation is below 10% for the material interface and
transport kernels except the tiny mixed cell initialization. Small kernels and
iterative solves are noisier: height sampling, coefficient refresh, and the
solves vary by 17--33%. Their qualitative result is still clear, but a single
sub-millisecond median should not be used for fine tuning.

## Construction, adaptation, and transfer

| Kernel | 1T | 2T | 4T | 8T | 1T allocated |
|---|---:|---:|---:|---:|---:|
| Uniform mesh construction | 3.397 ms | 1.00x | 1.23x | 1.61x | 14.56 MiB |
| Mixed mesh construction | 9.704 ms | 1.12x | 1.24x | 1.27x | 39.51 MiB |
| Refine mesh and transfer fields | 25.094 ms | 1.19x | 1.25x | 1.37x | 40.14 MiB |
| Coarsen mesh and transfer fields | 12.936 ms | 1.34x | 1.53x | 1.58x | 11.37 MiB |
| Prolong one 8x8 block | 0.001 ms | 1.08x | 0.82x | 0.83x | 0 B |
| Restrict one 8x8 block | 0.001 ms | 1.07x | 0.99x | 0.99x | 0 B |
| Prolong one 128x128 block | 0.317 ms | 1.46x | 2.07x | 3.85x | 0 B |
| Restrict one 128x128 block | 0.152 ms | 1.42x | 3.21x | 3.53x | 0 B |

Column-major transfer loops make serial prolongation 2.31--2.40x faster for a
128x128 block. This lowers its eight-thread speedup ratio from 5.92x to 3.85x,
while its absolute eight-thread time improves from 0.118 to 0.082 ms. An
individual 8x8 transfer remains too small to thread internally; production AMR
parallelizes across blocks.

The expensive symbolic constructors remain effectively serial:

| Constructor | Uniform time / allocation / 8T | Mixed time / allocation / 8T |
|---|---:|---:|
| Pressure-gradient geometry | 14.694 ms / 48.75 MiB / 0.97x | 26.646 ms / 82.67 MiB / 1.03x |
| Variable Poisson operator | 34.687 ms / 116.42 MiB / 0.78x | 73.682 ms / 196.01 MiB / 0.94x |
| Variable diffusion operator | 126.668 ms / 125.66 MiB / 1.06x | 230.531 ms / 242.89 MiB / 0.93x |

These paths discover geometry, build and sort sparse coordinate data, and
construct CSC matrices. Their allocations and global symbolic steps dominate;
changing dense cell traversal cannot make them parallel.

## Current conclusion

Block-local traversal is a clear serial improvement and preserves or improves
absolute eight-thread time for the isolated WY transport kernels. Compact face
and dual arrays should remain in flat contiguous order. The remaining scaling
limit is broader than cell locality: sparse topology construction, Krylov
vector work and reductions, coarse multigrid levels, repeated short passes,
and task overhead for small kernels all remain.

The matched [Stokes-3 scaling study](stokes3-thread-scaling-results.md) is the
decisive integration measurement. It confirms exact mesh and solver work and
shows that topology rebuild still dominates the eight-thread shortfall.
