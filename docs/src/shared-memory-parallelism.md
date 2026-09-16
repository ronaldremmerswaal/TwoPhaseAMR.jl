# Shared-memory parallelism

## Execution model

Numerical kernels use the ordinary Julia `:default` thread pool through
`Polyester.@batch`. Polyester keeps a small persistent worker pool and uses a
static loop schedule, avoiding the task-allocation cost that was measurable
with a fresh `Threads.@threads` region inside every Krylov matrix-vector
product. Pointer-rich tree searches use Julia's dynamic task scheduler instead;
their larger work items amortize task launch, and avoid Polyester's dense-array
lowering for heterogeneous captured topology.

The implementation does not index scratch storage by `Threads.threadid()`.
Julia tasks may migrate between threads after a yield, so thread-indexed state
is not generally a safe ownership model. Storage is instead owned by a mesh
entity:

- one iteration owns one cell, face, sparse row, or multigrid unknown;
- conservative face operators first compute one value per face, then gather
  incident values with a fixed cell/dual-volume schedule;
- coefficient refreshes group all contributions to a sparse entry before the
  parallel loop starts;
- reductions that do not dominate runtime remain serial and deterministic.

This follows Julia's task-migration guidance and the low-overhead persistent
task model documented by Polyester:

- [Julia multi-threading manual](https://docs.julialang.org/en/v1/manual/multi-threading/)
- [Julia PSA on task migration and thread-local state](https://julialang.org/blog/2023/07/PSA-dont-use-threadid/)
- [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl)

## Blocks and tiles

The public mesh remains a tree of fixed-size blocks. Work is exposed as a
flat range of cells or faces whenever possible, so a uniform mesh with one
large block is still split between threads. Operations that naturally act on
whole blocks, such as height-cache copies and adaptive cell-field transfer,
schedule independent blocks. This is the same two-level idea used by patch AMR
systems: schedule boxes when there are enough boxes and tile within a box when
there are not. See [AMReX's box and tiling
model](https://amrex-codes.github.io/amrex/docs_html/Basics.html#mfiter-with-tiling).

Kernels remain serial below 1,024 flat iterations. Above that cutoff the batch
size is chosen from the iteration count and thread count, targeting four
batches per thread. Block-level operations use a four-block cutoff because a
block transfer or height-cache operation is substantially more expensive than
one scalar array iteration.

## AMR and load balance

Within one shared-memory process, the current schedule balances iteration
counts rather than assigning persistent tree branches to threads. Interface
work is more expensive than pure-phase work, so a few large static block
partitions would be vulnerable to imbalance. Finer batches allow interface
and bulk work to be distributed across the worker pool without a separate
runtime weight model.

This choice is consistent with AMReX's dynamic tiling option and with the
broader tree-AMR practice of weighted partitions. `p4est`, for example,
supports user-defined leaf weights, while Afivo demonstrates shared-memory
parallel geometric multigrid on quadtree/octree meshes:

- [p4est scalable forest-of-octrees algorithms](https://doi.org/10.1137/100791634)
- [p4est weighted partition API](https://p4est.github.io/api/p4est-latest/p4est_8h.html)
- [Afivo shared-memory AMR and geometric multigrid](https://doi.org/10.1016/j.cpc.2018.06.018)

A persistent cost-weighted block scheduler should only be added if the
kernel-level measurements and the subsequent matched Stokes-3 scaling run show
material imbalance. It adds bookkeeping during every adaptation and cannot
improve serial topology construction.

## Reproducible kernel scaling

Run the core suite before an end-to-end Stokes-3 study:

```bash
TWOPHASE_THREAD_COUNTS=1,2,4,8 \
  julia --project=. benchmark/thread_scaling.jl
```

Each thread count runs in a new Julia process with one BLAS thread. The worker
warms each operation, keeps setup outside steady-kernel timings, records median
wall time, allocated bytes, allocation count, and a numerical checksum, and
rejects a threaded result whose checksum differs from the one-thread result.
Uniform and mixed-level cases cover initialization, reconstruction, height
functions, centred and staggered advection, compatible phase transport,
pressure gradient, Poisson and diffusion construction/application/solve,
surface tension, gravity, halo exchange, mesh construction/adaptation, and
restriction/prolongation.

Use `TWOPHASE_KERNEL_SAMPLES` to change the default five timed samples. Use
`TWOPHASE_KERNEL_CASE=uniform` or `mixed` to isolate one topology. Results are
written to `benchmark/threaded-kernel-results/scaling.csv`.
