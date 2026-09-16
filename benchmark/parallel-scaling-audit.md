# Shared-memory parallelisation audit and optimization plan

## Scope

This audit curates the measurements recorded on 2026-09-16 in
[`threaded-kernel-results.md`](threaded-kernel-results.md) and
[`stokes3-thread-scaling-results.md`](stokes3-thread-scaling-results.md). The
raw inputs are
[`threaded-kernel-results/scaling.csv`](threaded-kernel-results/scaling.csv)
and
[`stokes3-thread-results/stage-scaling.csv`](stokes3-thread-results/stage-scaling.csv).
No timings in this document are estimates or new runs.

The comparison is unusually well controlled. Each thread count ran in a fresh
Julia process with one BLAS thread. The Stokes-3 runs used exactly the same 448
time steps, 71 topology changes, time-step sizes, AMR decisions, ordered block
keys, pressure and diffusion iterations, and multigrid reuse history. Core
kernel checksums also agree across thread counts.

This report calls a kernel *well scaling* only when its speedup at eight Julia
threads is at least 4x, corresponding to 50% parallel efficiency. Missing that
threshold can have three very different meanings:

1. a large serial or synchronization-bound bottleneck;
2. a useful parallel kernel limited by memory traffic or algorithmic barriers;
3. a kernel too small for eight-way execution to pay for scheduling.

Absolute time and end-to-end contribution therefore determine priority.

## Current status

Only height-function sampling exceeds 4x in both isolated mesh cases. Four
other kernels exceed 4x only on the larger mixed mesh: cell initialization,
level-set initialization, WY interface advection, and WY phase-scalar
advection. Direct prolongation of one artificial 128x128 block reaches 5.92x,
but production uses 8x8 blocks and schedules across blocks. No complete
Stokes-3 stage reaches 4x.

The complete Stokes-3 physical stages improve from 36.829 s to 27.204 s, a
1.35x speedup. Fixed-topology steps improve by 1.66x and topology-changing
steps by 1.10x. Four threads give almost all the measured benefit: moving from
four to eight reduces total stage time by only 1.8%.

### Isolated numerical kernels below 4x

The table contains every numerical kernel for which either the uniform or
mixed-level measurement is below 4x. Times in parentheses are the eight-thread
medians. A case above the threshold is retained to show why the kernel remains
in the table.

| Kernel | Uniform 8T | Mixed 8T | Interpretation |
|---|---:|---:|---|
| Cell initialization | 3.69x (0.067 ms) | 4.19x (0.102 ms) | Borderline; launch cost dominates the uniform case |
| Level-set initialization | 3.08x (3.823 ms) | 4.74x (5.139 ms) | Enough work on the larger case; block costs are nonuniform |
| Interface reconstruction | 3.54x (1.645 ms) | 3.87x (3.157 ms) | Useful scaling, close to the target |
| Cell advection | 2.01x (0.109 ms) | 3.55x (0.112 ms) | Very small, bandwidth and barriers dominate |
| Cell halo exchange | 2.37x (0.016 ms) | 2.40x (0.029 ms) | Too small to optimize in isolation |
| Staggered velocity advection | 2.68x (0.743 ms) | 2.25x (1.741 ms) | Multiple face and dual-volume passes |
| WY interface advection | 3.56x (5.626 ms) | 4.30x (9.350 ms) | Useful scaling; uniform case has less work |
| WY phase-scalar advection | 3.81x (7.102 ms) | 4.17x (12.277 ms) | Near the target |
| WY momentum advection | 3.43x (7.900 ms) | 3.53x (14.979 ms) | Useful scaling but still a material transport cost |
| Pressure gradient | 1.34x (0.083 ms) | 1.52x (0.227 ms) | Several tiny face passes and synchronizations |
| Poisson operator application | 2.59x (0.101 ms) | 2.17x (0.242 ms) | Factorized action has several dependent passes |
| Poisson solve | 1.93x (5.059 ms) | 1.91x (12.948 ms) | Serial BLAS-1 work, reductions, and coarse MG levels |
| Diffusion-coefficient refresh | 2.16x (0.447 ms) | 2.03x (0.989 ms) | Sequential traversal of dependent MG levels |
| Diffusion operator application | 1.68x (0.147 ms) | 1.84x (0.249 ms) | Two sparse strain passes separated by stress scaling |
| Diffusion solve | 1.39x (6.053 ms) | 1.52x (12.476 ms) | Same Krylov/MG limits, amplified by the coupled operator |
| CSF surface tension | 0.94x (0.262 ms) | 1.06x (0.504 ms) | Tiny application plus a serial validation pass |
| Gravity | 3.30x (0.726 ms) | 3.68x (1.302 ms) | Useful scaling; three dependent potential/face passes |

Height-function sampling, at 4.03x and 4.32x, is the only numerical kernel
that passes in both cases. The larger mixed-level transport cases generally
scale at least as well as the uniform cases. There is still no evidence that a
persistent interface-weighted load balancer would address the measured limit.

### Construction, mesh adaptation, and grid transfer below 4x

| Kernel | 8T speedup | 8T time | One-thread allocation | Main serial work |
|---|---:|---:|---:|---|
| Uniform mesh construction | 1.34x | 3.180 ms | 14.56 MiB | tree mutation, balancing, face schedules, compaction |
| Mixed mesh construction | 1.21x | 7.888 ms | 39.51 MiB | same, with more dictionary and sorting work |
| Refine mesh and transfer fields | 1.42x | 18.024 ms | 40.14 MiB | schedule rebuild, sequential fields, interface transfer |
| Coarsen mesh and transfer fields | 1.69x | 8.103 ms | 11.35 MiB | schedule rebuild and sequential fields |
| Prolong one 8x8 block | 0.96x | 0.003 ms | 0 B | deliberately too small to thread internally |
| Restrict one 8x8 block | 0.86x | 0.001 ms | 0 B | deliberately too small to thread internally |
| Restrict one 128x128 block | 3.06x | 0.060 ms | 0 B | short reduction-style inner loop |
| Pressure-gradient geometry, uniform | 0.98x | 15.137 ms | 48.75 MiB | patch dictionaries, sorting, COO-to-CSC assembly |
| Pressure-gradient geometry, mixed | 0.84x | 31.385 ms | 82.67 MiB | same |
| Poisson operator construction, uniform | 0.96x | 40.117 ms | 116.42 MiB | sparse products, pattern completion, refresh map |
| Poisson operator construction, mixed | 0.77x | 94.563 ms | 196.01 MiB | same |
| Diffusion operator construction, uniform | 1.01x | 113.859 ms | 126.87 MiB | stress schedules and sparse construction |
| Diffusion operator construction, mixed | 0.94x | 246.289 ms | 245.59 MiB | same |

The direct 128x128 prolongation benchmark is excluded because it reaches
5.92x. Its result confirms that the arithmetic is parallel when one operation
is large enough. It does not describe the production 8x8-block granularity.

The expensive constructors are effectively serial and their allocations do
not fall with more threads. Current code builds shared dictionaries and COO
vectors, sorts contributions, and calls global sparse constructors. Julia's
standard collections cannot be mutated concurrently without synchronization,
so placing a parallel loop directly around those `push!` and `Dict` operations
would be unsafe and lock-heavy. The appropriate design is a count/prefix/fill
pipeline with disjoint output slices, followed by retained sparse-pattern
storage.

### End-to-end stages below 4x

Every measured Stokes-3 stage is below the threshold. The table is ordered by
the *excess over a 4x target*, defined as `T8 - T1/4`. This measures how much
of the observed eight-thread time would disappear if that stage alone reached
4x while doing identical numerical work.

| Stage | 1T time | 8T time | 8T speedup | Excess over 4x |
|---|---:|---:|---:|---:|
| Topology-dependent rebuild | 11.202 s | 11.697 s | 0.96x | 8.896 s |
| Diffusion solve | 5.255 s | 3.470 s | 1.51x | 2.156 s |
| Pressure projection | 3.853 s | 2.039 s | 1.89x | 1.076 s |
| AMR field transfer | 1.759 s | 1.481 s | 1.19x | 1.041 s |
| Diffusion solver refresh | 1.295 s | 1.135 s | 1.14x | 0.812 s |
| Gravity | 1.798 s | 1.240 s | 1.45x | 0.790 s |
| Pressure operator update | 0.927 s | 1.007 s | 0.92x | 0.776 s |
| Advection | 6.577 s | 2.393 s | 2.75x | 0.749 s |
| Surface tension | 0.960 s | 0.943 s | 1.02x | 0.703 s |
| AMR indicators | 2.257 s | 1.055 s | 2.14x | 0.491 s |
| Pressure preconditioner refresh | 0.536 s | 0.489 s | 1.10x | 0.355 s |
| Viscosity coefficient update | 0.300 s | 0.166 s | 1.81x | 0.091 s |
| Time-step bound | 0.070 s | 0.071 s | 0.98x | 0.054 s |
| Mixture density | 0.036 s | 0.012 s | 2.90x | 0.003 s |
| State copy | 0.004 s | 0.004 s | 0.99x | 0.003 s |
| User body force | 0.001 s | 0.001 s | 1.02x | 0.001 s |

The total gap between the actual 27.204 s and a hypothetical 4x result is
17.996 s. Topology rebuild accounts for 49.4% of it. Topology rebuild,
diffusion solve, pressure projection, and AMR transfer together account for
73.2%. The tiny state-copy, body-force, time-step, halo, and single-block
transfer kernels should remain serial at this problem size.

## What the code currently does

### Scheduling model

Dense numerical loops use a persistent `Polyester.@batch` worker pool through
[`src/threading.jl`](../src/threading.jl). Pointer-rich tree searches use
Julia's dynamic scheduler. The adaptive batch size targets four batches per
thread. Scalar loops require 1,024 iterations before threading; expensive
block loops use a four-block threshold. Outputs are owned by cells, faces, or
rows, and scatter operators are split into a parallel face evaluation and a
parallel gather. This is data-race free and avoids task-local scratch indexed
by `threadid()`.

This remains the right base policy. Polyester documents its persistent static
worker pool, `minbatch`, allocation-free reductions, and the need to benchmark
thread pinning. Julia documents both task migration and the requirement to
lock concurrent mutations of `Array` and `Dict` objects:

- [Julia multi-threading manual](https://docs.julialang.org/en/v1/manual/multi-threading/)
- [Julia task-migration guidance](https://julialang.org/blog/2023/07/PSA-dont-use-threadid/)
- [Polyester.jl scheduling and reductions](https://github.com/JuliaSIMD/Polyester.jl)

### Topology construction

Several global symbolic paths remain:

- [`_build_face_schedules`](../src/mesh.jl) fills dictionaries of face sides
  and sorts each line;
- [`_balance!`](../src/mesh.jl) repeatedly checks pairs of leaves;
- [`_assemble_inverse_metric`](../src/gradient.jl) gathers vertex patches,
  appends COO triplets, sorts patch keys, and calls `sparse`;
- [`EllipticOperator`](../src/elliptic.jl) builds incidence matrices, performs
  sparse products, completes the pattern, and sorts a coefficient refresh
  map;
- the variable-stress constructor in
  [`src/variable_staggered_diffusion.jl`](../src/variable_staggered_diffusion.jl)
  retains local shear patches but still assembles global strain and
  interpolation matrices;
- each unmatched tree-GMG prefix constructs new pruned meshes, transfers, and
  coarse operators in [`src/multigrid.jl`](../src/multigrid.jl).

The local shear-patch cache and matching coarse-suffix reuse have already
removed much repeated work. They do not remove global numbering, sparse
pattern assembly, or hierarchy bookkeeping after a changed fine topology.

### Solves and multigrid

Fine operator actions are threaded, but the V-cycle descends and ascends one
level at a time. Small coarse levels intentionally run serially below the
grain threshold. Residual norms, compatibility projections, and several
other reductions are serial.

The installed Krylov.jl version dispatches ordinary `Vector{Float64}` dot,
norm, and axpy operations to BLAS. The benchmark fixes BLAS to one thread, so
those vector operations are serial even though the package-owned operator and
preconditioner actions are threaded. This explains why an operator
application can scale better than the solve containing it. Krylov.jl's own
paper identifies operator-vector products, norms, dot products, and axpy as
the parallel building blocks of its methods:

- [Krylov.jl JOSS paper](https://doi.org/10.21105/joss.05187)

The multigrid transfer is already level-by-level in the algorithmic sense:
each V-cycle restricts to the next coarser level and later prolongs back before
continuing. Its current transfers are generic sparse matrix-vector products.
The hierarchy also refactorizes the bottom matrix after coefficient refresh.

### Adaptive field transfer

AMR field transfer is parallel across new cell blocks and across new faces,
but the surrounding plan is rebuilt and applied repeatedly:

- old/new block dictionaries are reconstructed for every cell field;
- registered fields are staged and transferred one after another;
- each face field repeats local face numbering and donor searches against the
  same line geometry;
- flux accumulation, refined divergence correction, and global momentum
  correction remain serial;
- mesh schedules are globally rebuilt before field transfer.

The current regrid changes a block by at most one tree level. Every destination
reads an immutable old-field snapshot, so transfers at different levels have
no data dependency today. Grouping by level is still useful because it removes
per-block dictionary probes and the prolong/restrict branch from the hot loop,
improves spatial locality, and exposes groups with similar cost.

If recursive transfers across more than one level are added later, ordering
becomes mandatory: restriction runs from finest to coarsest and prolongation
runs from coarsest to finest, with a barrier between levels. AMReX follows this
model for average-down operations and parallelizes boxes or tiles within one
level. Its CPU implementation opens the parallel region outside the box loop,
which avoids a new team for every small tile:

- [AMReX MFIter and logical tiling](https://amrex-codes.github.io/amrex/docs_html/Basics.html#mfiter-and-tiling)
- [AMReX parallel average-down implementation](https://amrex-codes.github.io/amrex/docs_html/doxygen/AMReX__MultiFabUtil__I_8H_source.html)

deal.II likewise defines adaptive multigrid levels explicitly and supplies a
matrix-free two-level transfer that stores per-cell coarse/fine index lists:

- [deal.II multilevel support](https://dealii.org/current/doxygen/deal.II/group__mg.html)
- [deal.II matrix-free two-level transfer](https://dealii.org/developer/doxygen/deal.II/classMGTwoLevelTransfer.html)

## Recommended optimizations

### 1. Materialize one reusable, level-bucketed AMR transfer plan

This is the first bounded implementation experiment. Build one topology-only
plan immediately after the new leaves are known:

```text
unchanged blocks:  new block -> old block
prolong[level]:    new child -> old parent, child quadrant
restrict[level]:   new parent -> four old children
interior faces:    destination -> compact donor indices and weights
boundary faces:    destination -> compact donor indices and weights
refined parents:   disjoint divergence-correction patches
```

Reuse the plan for every registered field. Bucket block work by destination
level and transfer kind. For the current one-level contract, concatenate
populated buckets when this gives more parallel work; a barrier between levels
is unnecessary. For a future recursive contract, execute restriction buckets
in descending level order and prolongation buckets in ascending order.

Within a populated bucket, parallelize across blocks and keep each 8x8 block
serial internally. Process compatible fields inside the same block task where
possible, after all field-specific preparation has completed. This amortizes
task dispatch and reuses hot source geometry. Preserve separate typed groups
for geometric interface prolongation, intrinsic scalar prolongation, and plain
cell fields.

For faces, precompute the donor segment indices and geometric weights once.
Field application then becomes a compact gather rather than repeated line
lookups. Store separate schedule flags for piecewise-constant and limited
linear transfer. Parallelize divergence correction by refined parent after
proving that each parent owns distinct internal faces. Replace serial
face-to-cell flux scatter with the existing package pattern: compute one flux
per face, then gather through a cell-face incidence schedule.

Expected effect: this targets the 1.19x AMR-transfer stage and also removes
allocation and serial planning work from topology-changing steps. The key
measurement is plan construction versus application per field and per level;
the combined `adapt!` time is too coarse to judge it.

### 2. Replace global symbolic assembly with parallel count/prefix/fill

Do not parallelize shared `Dict` insertion or COO `push!`. For each face line,
vertex patch, strain row, and cell incidence:

1. enumerate stable geometric entities, grouped by tree level;
2. compute each entity's contribution count in parallel;
3. form offsets with an exclusive prefix sum;
4. fill disjoint slices of typed arrays in parallel;
5. sort only small local slices when geometry does not provide canonical order;
6. create CSC column pointers directly or use preallocated `sparse!` buffers;
7. retain the pattern and direct numeric-slot maps until affected topology
   changes.

Julia's `SparseArrays.sparse!` expert driver accepts caller-provided work
buffers, so it is a useful intermediate step if direct canonical CSC emission
is premature:

- [Julia `SparseArrays.sparse!` and `spzeros!`](https://docs.julialang.org/en/v1/stdlib/SparseArrays/)

The stronger target is to avoid global sparse conversion after a local regrid.
Assign stable geometric keys to patches and cache their local rows. Rebuild
only patches touching the changed block frontier, then deterministically pack
retained and new rows into the current numbering. Parallelize the local patch
computations; perform the small final prefix sum and pointer publication
serially.

Replace the pairwise `_balance!` scan with a work queue of newly split blocks
and tree-neighbour queries. A split can only invalidate the 2:1 constraint in
its local neighbour ring. This changes the work from repeated global leaf-pair
testing to work proportional to the changed frontier.

Expected effect: this addresses the 0.96x topology rebuild and the three
0.77--1.01x constructors. It has much higher value than adding threads to
already cached coefficient updates.

### 3. Thread and fuse Krylov vector work, then tune MG by level

Add a package-owned PCG experiment for the default symmetric pressure and
diffusion paths, retaining Krylov.jl for the other public solver choices. Use
Polyester's allocation-free reduction support or fixed task-owned partials for
dot products and norms. Fuse vector updates when dependencies allow, and keep
a deterministic fixed-chunk reduction mode for regression testing.

This experiment must separately time, for every Krylov iteration:

- fine operator action;
- preconditioner/V-cycle;
- dot products and norms;
- axpy and residual updates;
- convergence callback.

Add per-level MG counters for operator, smoothing, restriction, prolongation,
mean projection, and bottom solve. PETSc exposes per-level MG logging for the
same reason: aggregate solve time conceals the level at which parallelism ends.

- [PETSc PCMG options and per-level logging](https://petsc.org/release/manualpages/PC/PCMG/)

Keep the multiplicative V-cycle initially. Additive or kaskade cycles expose
cross-level concurrency, but they change the preconditioner and may increase
iterations; PETSc lists them as distinct MG algorithms. They are a later
solver experiment, not a scheduling substitution.

Replace generic sparse inter-level multiplication with the same compact
two-level gather schedules proposed for AMR transfer. The V-cycle already
provides the required level ordering. Avoid launching threads on coarse levels
whose measured serial time is below launch overhead; choose the cutoff from
per-level timings rather than a global item count alone.

Expected effect: this directly targets diffusion solve at 1.51x and pressure
projection at 1.89x. It will also show whether the current factorized fine
operators or serial vector algebra dominates each solve.

### 4. Fuse dependent face passes

The pressure gradient, factorized Poisson action, variable diffusion action,
surface tension, and gravity each launch several short dependent loops. Reduce
barriers and memory traffic without changing the discretization:

- fuse pressure-jump formation with diagonal metric application on uniform
  grids;
- for mixed grids, evaluate a precomputed row stencil directly so source
  pressure differences need not be materialized when this is faster;
- fuse stress scaling into one of the two strain traversals where the compact
  schedule permits it;
- accumulate surface-tension and gravity jumps into one pressure-compatible
  jump vector and apply the shared weighted inverse metric once;
- make curvature validation a parallel status calculation followed by a
  deterministic search for the first failing face.

The combined-force optimization is algebraically valid because both forces
use the same linear weighted-gradient operator. Roundoff and well-balanced
equilibrium tests must still be rerun because summation order changes.

Expected effect: individual applications are small, but surface tension,
gravity, and pressure updates recur 448 times and contribute 2.73 s at eight
threads in Stokes-3.

### 5. Treat transport as a bandwidth and fusion problem

WY interface and scalar transport already approach or exceed 4x on the larger
case. Momentum advection remains at 3.43--3.53x, and the complete advection
stage reaches 2.75x. The matched mixed-mesh results do not indicate an
interface load imbalance.

The follow-up serial
[`block-locality experiment`](data-locality-results.md) resolves one scheduler
question before parallel work. Explicit block/local-index loops are 4.2--4.5x
faster for the small centred-cell subkernels, even though the existing
`CellRef` sequence is already block-major. Hoisting each block matrix removes
the remaining indirection. Applying the same owner-block permutation to flat
face arrays is 8--16% slower, and a fully block-grouped momentum component has
no repeatable gain. The measured hybrid keeps face and dual arrays contiguous
and changes only block-backed cell loops; it reduces complete component time
by 7--16%. This traversal split should precede further transport threading.

Profile bytes moved and barriers per directional sweep before changing the
scheduler. Reuse face fluxes across all compatible scalar and momentum fields,
fuse gathers sharing the same ownership schedule, and keep structure-of-arrays
storage for unit-stride loops. Directional splitting imposes a real barrier
between sweeps, so it should not be removed merely to improve a threading
metric.

## Benchmark changes required for each experiment

1. Pin Julia threads to physical cores and use at least ten timed samples for
   sub-millisecond kernels. Polyester itself recommends pinning when comparing
   scaling.
2. Retain the current checksum and exact Stokes-3 mesh-evolution checks.
3. Add problem-size sweeps. The present 16k--28k-cell cases establish current
   latency but cannot distinguish a fundamental limit from insufficient work.
4. Split topology rebuild into mesh schedules, pressure geometry/operator,
   pressure hierarchy, diffusion geometry/operator, diffusion hierarchy, and
   workspace resizing.
5. Split AMR transfer into plan construction and application, then report each
   field kind and each populated level bucket.
6. Report MG time and unknown count per level, along with Krylov vector,
   reduction, operator, and preconditioner time.
7. Record effective memory bandwidth and CPU utilization for operator and
   transfer kernels. A flat speedup with saturated bandwidth requires fusion
   or fewer bytes, rather than more tasks.
8. Continue reporting allocation separately. The current eight-thread
   Stokes-3 run allocates essentially the same memory as the one-thread run, so
   parallel scratch growth is not the problem; topology construction remains
   the allocation target.

## Implementation order and acceptance criteria

1. **AMR transfer plan and level buckets.** Require identical transferred
   fields and refined divergence, lower plan-plus-apply time, and no allocation
   growth. Compare flat buckets with strict level barriers on the current
   one-level workload.
2. **Topology count/prefix/fill.** First convert face and incidence schedules,
   then gradient patches, then stress schedules. Require identical sparse
   patterns and coefficients. A topology-changing step should improve at four
   and eight threads, not only in isolation.
3. **Per-level MG instrumentation and threaded PCG vector kernels.** Require
   unchanged convergence decisions and comparable iteration counts. Measure
   deterministic and fastest reduction modes separately.
4. **Fused force and operator paths.** Require the existing balanced-force,
   hydrostatic, and solver equivalence tests, plus a matched Stokes-3 trace.
5. **Transport fusion.** Pursue only after hardware counters show bandwidth or
   repeated schedule traversal dominates.

The first performance target should be topology-changing steps, where the
largest absolute loss occurs. The second should be solver throughput. Small
standalone kernels should only be changed when fusion removes an end-to-end
pass or synchronization point.
