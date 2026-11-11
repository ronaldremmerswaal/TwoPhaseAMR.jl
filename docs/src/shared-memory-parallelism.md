# Shared-memory parallelism

## Execution model

Numerical kernels use the ordinary Julia `:default` thread pool through
`Polyester.@batch`. Polyester keeps a small persistent worker pool and uses a
static loop schedule, avoiding the task-allocation cost that was measurable
with a fresh `Threads.@threads` region inside every Krylov matrix-vector
product. All threaded loops, including the pointer-rich tree searches of mesh
construction, go through one launch function, `_parallel_for(f, n; work)`, and
its loop form `@_parallel for i in range … end`; the block-column traversal of
centred fields is a schedule on top of `_parallel_for`. A loop runs serially,
in index order, on one thread or when `n * work` (`work` estimates one item's
cost in cell updates) is below the grain `_PARALLEL_GRAIN[]`.

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

The public mesh remains a tree of fixed-size blocks. Kernels over block-backed
centred fields schedule leaf blocks and traverse each dense block matrix in
column-major order. When a mesh has too few leaf blocks, the scheduler splits
their columns into tiles, so a uniform mesh with one large block still uses
all available threads. This also hoists the block matrix, halo width, and
geometry outside the innermost cell loop.

Compact `FaceField`, `FaceVelocity`, and dual-volume arrays retain their flat,
contiguous traversal. Operations that naturally act on whole blocks, such as
height-cache copies and adaptive cell-field transfer, schedule independent
blocks. This is the same two-level idea used by patch AMR systems: schedule
boxes when there are enough boxes and tile within a box when there are not.
See [AMReX's box and tiling model](https://amrex-codes.github.io/amrex/docs_html/Basics.html#mfiter-with-tiling).

Kernels remain serial below a work of 1,024 cell updates. Above it, Polyester
splits the index range into at most one contiguous batch per thread, each with
at least a quarter of the grain. Block-level operations, such as block
transfer, height-cache copies or refinement indicators, count one block as a
quarter of the grain, so they thread from four blocks of any size. The loop
form `@_parallel` expands its body into a plain serial loop and into the
`Polyester.@batch` loop: Polyester boxes a closure argument for its workers
(32 B per launch), whereas a `@batch` body captures its arrays as pointers and
does not allocate. The grain is a `Ref` so tests can set it to one and run the
threaded path on small meshes.

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

## Reproducible kernel scaling

Instantiate the dedicated benchmark environment once, then run the same suite
in separate Julia processes for each thread count:

```bash
julia --project=benchmark -e \
  'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'

JULIA_NUM_THREADS=1 julia --project=benchmark benchmark/benchmarks.jl
JULIA_NUM_THREADS=4 julia --project=benchmark benchmark/benchmarks.jl
```

The runner fixes BLAS to one thread, reports the Julia thread count and CPU,
and keeps fixture construction outside timed steady-state samples. Use
`TWOPHASE_BENCHMARK_SAMPLES`, `TWOPHASE_BENCHMARK_SECONDS`, and
`TWOPHASE_BENCHMARK_OUTPUT` to control sampling and save machine-local raw
results. The suite covers both uniform and mixed meshes; compare matching
groups rather than combining setup, adaptation, and stepping into one timing.
