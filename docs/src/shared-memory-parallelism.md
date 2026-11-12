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

## Errors and user callbacks

A threaded loop body never throws. Polyester rethrows an exception raised in
the calling thread's index range as itself, but one raised in a worker's range
as a `TaskFailedException`, while the other ranges keep running; the error a
user sees would depend on the thread count. Two patterns keep the error
thread-count independent:

- A check on data the loop already holds is fused into the loop. The body
  records the failure in an owned, preallocated output — one flag per
  iteration or per block column, never `push!` — and the caller throws after
  the launch, naming the offending cell, block or face. The WY and conditional
  fraction bounds, the variable-diffusion diagonal and the AMR face transfer
  use this pattern, so none of them costs a serial pass over the mesh per step.
- A check on fixed topology runs once, when that topology is built. The
  elliptic matrix locates its diagonal slots at assembly and every coefficient
  refresh indexes them, so no threaded body searches the sparse pattern. A
  multigrid level likewise keeps the source slot of every entry of its stored
  row form (the transpose its smoother multiplies with); a refresh compares the
  level matrix's pattern with the recorded one and copies the values through
  the slots, both threaded (the comparison in fixed chunks with one mismatch
  flag each), so only a changed pattern costs a serial counting sort.

Only cheap input checks still run serially before a loop. A rejected input
leaves the state unchanged where the old serial code did, except where the
check shares its pass with the update it guards: the conditional-fraction
refresh has already clamped the cells it visited before the recorded failure.

User callbacks (`initialize!` functions, `FaceVelocity`/`set_velocities!`
initializers, level sets, `PhaseScalar` initializers, `Dirichlet` halo and
inflow values, wall velocities, body forces and diagnostics) are evaluated
serially on the calling task, in a fixed order. They need not be thread-safe,
their exceptions propagate unchanged, and a callback with side effects (a
random perturbation, a counter) gives the same result for every thread count.
A `Dirichlet` halo is filled before the threaded halo pass only to keep its
callback off the workers: every halo donor is an interior cell, so the ordering
cannot change the result. That pre-pass iterates a list of the physical halo
entries collected once per topology, not the whole halo schedule.

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

## Determinism across thread counts

Results are bitwise identical for every thread count: a thread count is a
performance choice, not a numerical one. Two rules give this. Every index owns
its outputs, so a loop body computes the same value whichever task runs it.
Every reduction has a schedule that depends only on the number of items: sums
run over fixed chunks of 1,024 items that are folded in chunk order, and maxima
fold canonicalised `NaN` terms in chunk order. A thread count therefore never
changes a sign, a rounding, or an iteration count, and a flow can be restarted
or compared across machines that differ only in `-t`.

The regression oracle records one state, valid for every thread count. The
`threading_equivalence` test group checks the claim directly. It runs five short
cases in fresh Julia processes with one, two and four threads, with the grain
`_PARALLEL_GRAIN[]` at one so that small meshes take the threaded branch of
every loop:

| Case | Workload |
| --- | --- |
| `capillary` | two-phase CMDGFM capillary wave with interface AMR |
| `droplet` | two-phase CMDGFM on an embedded sloped wall (example 09), AMR |
| `stokes3_two` | the two-velocity Stokes-3 wave (example 10); must regrid |
| `cylinder` | the one-phase cylinder (example 04) with geometry and vorticity AMR |
| `taylor_green` | one-phase Taylor--Green vortices, 16² root block refined to level 3 |

Each process prints a SHA-256 over every state field (velocity components per
phase, pressure, liquid fraction) and over every recorded diagnostic history
(its name, the time and step of each sample and the sample values, which are
vectors for the momentum and volume diagnostics), the step and mesh-update
counts and the leaf block count, and the group requires all of them to be equal
across thread counts. The Stokes-3 case must adapt its mesh during the run, so the transfer,
rebuild and re-solve paths are compared as well. A warm step of the Taylor--Green and the capillary
case must also stay within a recorded allocation budget at one and at two
threads, so a threaded launch cannot start allocating per cell or per launch.
Run it locally with

```sh
julia --project=. -e 'using Pkg; Pkg.test(test_args=["threading_equivalence"])'
```

and set `THREADING_EQUIVALENCE_THREADS=1,2,4,8` to add a thread count that CI
does not run. The `threading-equivalence` CI job runs the group on Julia 1.12 on
every pull request.

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

The thread-scaling probe times one fixture per process and prints the per-step
wall times, a state checksum, and with a third argument a profile split of the
main thread into serial samples and samples inside a Polyester batch:

```sh
julia --project=. --threads=4 benchmark/threading/scaling.jl taylor_green 20 0
```

The manual `scaling` tier of the CI workflow runs the ten target cases
(Taylor--Green, the cylinder, the capillary wave and the one- and two-velocity
Stokes-3 wave, each at its default resolution and one level finer) at one, two
and four threads, and uploads the logs as an artifact:

```sh
gh workflow run CI.yml --ref <branch> -f tier=scaling
```

That tier is a measurement, not a gate: it is not part of `ci-ok`.
