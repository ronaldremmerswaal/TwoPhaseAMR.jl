# Shared-memory parallelisation audit and optimization plan

## Scope

This audit was remeasured after the production-wide block-local cell traversal
change committed as `245781f`. The current inputs are
[`threaded-kernel-results-block-local/scaling.csv`](threaded-kernel-results-block-local/scaling.csv)
and
[`stokes3-thread-results-block-local/stage-scaling.csv`](stokes3-thread-results-block-local/stage-scaling.csv).
The preceding measurements remain in `threaded-kernel-results` and
`stokes3-thread-results` for direct before/after comparison.

Each thread count ran in a fresh Julia process with one BLAS thread. The
Stokes-3 runs used exactly the same 448 steps, 71 topology changes, time-step
sizes, AMR decisions, ordered block keys, pressure and diffusion iterations,
and multigrid reuse history. Core-kernel checksums agree across all thread
counts. Independent eight-thread repeats preserve the checksums and exact
Stokes-3 work trace.

A kernel is called *well scaling* here only when its eight-thread speedup is at
least 4x, or 50% parallel efficiency. Absolute time and before/after time are
reported as well, because a serial optimization can lower the speedup ratio
while making every execution faster.

## Current status after block-local traversal

Gravity is the only isolated numerical kernel above 4x on both meshes.
Mixed-mesh WY phase-scalar advection also reaches 4.02x. Several interface
kernels are close, but their repeats straddle or remain below the threshold.
Symbolic operator construction, mesh adaptation, the elliptic solves, and all
complete Stokes-3 stages remain below 4x.

The complete Stokes-3 physical stages improve from 36.276 s to 27.157 s, a
1.34x speedup. An independent eight-thread run takes 27.144 s. Fixed-topology
steps improve by 1.59x and topology-changing steps by 1.13x. Four threads give
most of the benefit; moving from four to eight reduces measured stage time by
2.9%.

Compared with the preceding implementation, one-thread stage time falls by
1.5% and eight-thread time by 0.2%. The block-local work makes isolated serial
cell kernels 4.2--4.6x faster than flattened `CellRef` traversal and improves
one-thread WY transport by 7--25%. The overall parallel ceiling barely moves
because topology rebuild and solver synchronization were unaffected.

### Isolated numerical kernels below 4x

This table contains every numerical kernel for which either mesh is below 4x.
The first controlled seven-sample run is shown; the 11-sample eight-thread
repeat is used in the interpretation below.

| Kernel | Uniform 8T | Mixed 8T | Interpretation |
|---|---:|---:|---|
| Cell initialization | 3.34x (0.050 ms) | 3.57x (0.080 ms) | Tiny after the serial improvement; repeat noise is large |
| Level-set initialization | 2.84x (3.802 ms) | 3.89x (5.653 ms) | Useful scaling; adaptive quadrature has nonuniform work |
| Interface reconstruction | 2.94x (1.664 ms) | 3.96x (2.600 ms) | Mixed case is at the threshold; uniform case has less work |
| Height-function sampling | 2.61x (0.133 ms) | 2.08x (0.433 ms) | Sub-millisecond and variable; repeat gives 2.58x/3.12x |
| Cell advection | 2.35x (0.081 ms) | 2.51x (0.140 ms) | Bandwidth and launch overhead dominate |
| Cell halo exchange | 2.17x (0.017 ms) | 2.10x (0.033 ms) | Too small to optimize alone |
| Staggered velocity advection | 1.89x (1.106 ms) | 2.25x (1.776 ms) | Multiple face and dual-volume passes |
| WY interface advection | 3.23x (5.275 ms) | 3.83x (9.336 ms) | Repeat reaches 3.49x/4.23x |
| WY phase-scalar advection | 3.34x (6.099 ms) | 4.02x (10.702 ms) | Eight-thread absolute time improves in both cases |
| WY momentum advection | 3.04x (7.753 ms) | 3.33x (13.899 ms) | Material transport cost; repeat is 3.17x/3.43x |
| Pressure gradient | 2.28x (0.052 ms) | 3.38x (0.111 ms) | Short face passes and synchronization |
| Poisson operator application | 2.46x (0.095 ms) | 2.26x (0.202 ms) | Several dependent factorized passes |
| Poisson solve | 2.19x (4.610 ms) | 2.28x (10.846 ms) | Krylov vector work, reductions, and coarse MG levels |
| Diffusion-coefficient refresh | 1.90x (0.448 ms) | 2.25x (0.851 ms) | Dependent traversal of MG levels |
| Diffusion operator application | 2.21x (0.116 ms) | 2.14x (0.221 ms) | Two strain passes separated by stress scaling |
| Diffusion solve | 1.37x (6.242 ms) | 1.58x (12.109 ms) | Krylov/MG limits dominate |
| CSF surface tension | 1.17x (0.239 ms) | 0.94x (0.543 ms) | Short application plus validation and barriers |

Gravity reaches 4.82x and 4.45x. The repeat moves those values to 5.78x and
5.36x. No persistent interface imbalance is visible: mixed interface transport
usually scales at least as well as the uniform case.

### Construction, adaptation, and transfer below 4x

| Kernel | 8T speedup | 8T time | One-thread allocation | Main serial work |
|---|---:|---:|---:|---|
| Uniform mesh construction | 1.61x | 2.112 ms | 14.56 MiB | tree mutation, balancing, schedules, compaction |
| Mixed mesh construction | 1.27x | 7.618 ms | 39.51 MiB | dictionaries, sorting, and tree mutation |
| Refine mesh and transfer fields | 1.37x | 18.364 ms | 40.14 MiB | global schedule rebuild and sequential field stages |
| Coarsen mesh and transfer fields | 1.58x | 8.183 ms | 11.37 MiB | schedule rebuild and sequential field stages |
| Prolong one 8x8 block | 0.83x | 0.002 ms | 0 B | too small to thread internally |
| Restrict one 8x8 block | 0.99x | 0.001 ms | 0 B | too small to thread internally |
| Prolong one 128x128 block | 3.85x | 0.082 ms | 0 B | serial baseline is now 2.3x faster |
| Restrict one 128x128 block | 3.53x | 0.043 ms | 0 B | short reduction-style stencil |
| Pressure-gradient geometry, uniform | 0.97x | 15.185 ms | 48.75 MiB | patch dictionaries, sorting, sparse assembly |
| Pressure-gradient geometry, mixed | 1.03x | 25.883 ms | 82.67 MiB | same |
| Poisson operator construction, uniform | 0.78x | 44.610 ms | 116.42 MiB | sparse products, pattern completion, refresh map |
| Poisson operator construction, mixed | 0.94x | 78.485 ms | 196.01 MiB | same |
| Diffusion operator construction, uniform | 1.06x | 118.947 ms | 125.66 MiB | stress schedules and sparse construction |
| Diffusion operator construction, mixed | 0.93x | 247.470 ms | 242.89 MiB | same |

The transfer speedup ratios fall partly because column-major traversal makes
serial prolongation 2.31--2.40x faster. Its absolute eight-thread time also
improves. The expensive constructors remain global symbolic pipelines using
shared dictionaries, sorting, sparse products, and CSC construction. They are
still the principal setup target.

### End-to-end stages below 4x

Every Stokes-3 stage remains below the threshold. The table is ordered by
`T8 - T1/4`, the excess over a hypothetical 4x stage.

| Stage | 1T time | 8T time | 8T speedup | Excess over 4x |
|---|---:|---:|---:|---:|
| Topology-dependent rebuild | 11.186 s | 11.593 s | 0.96x | 8.797 s |
| Diffusion solve | 5.541 s | 3.353 s | 1.65x | 1.968 s |
| Advection | 5.908 s | 2.734 s | 2.16x | 1.257 s |
| Pressure projection | 3.823 s | 1.942 s | 1.97x | 0.986 s |
| AMR field transfer | 2.220 s | 1.436 s | 1.55x | 0.881 s |
| Gravity | 1.444 s | 1.237 s | 1.17x | 0.876 s |
| Diffusion solver refresh | 1.306 s | 1.097 s | 1.19x | 0.770 s |
| Pressure operator update | 0.906 s | 0.919 s | 0.99x | 0.692 s |
| AMR indicators | 2.207 s | 1.200 s | 1.84x | 0.649 s |
| Surface tension | 0.815 s | 0.846 s | 0.96x | 0.642 s |
| Pressure preconditioner refresh | 0.534 s | 0.483 s | 1.11x | 0.349 s |
| Viscosity coefficient update | 0.287 s | 0.212 s | 1.35x | 0.140 s |
| Time-step bound | 0.060 s | 0.086 s | 0.69x | 0.072 s |
| Mixture density | 0.037 s | 0.013 s | 2.75x | 0.004 s |
| State copy | 0.003 s | 0.004 s | 0.95x | 0.003 s |
| User body force | 0.001 s | 0.001 s | 1.05x | 0.001 s |

The total gap above a 4x result is 18.087 s. Topology rebuild accounts for
48.6%. Topology rebuild, diffusion solve, advection, and pressure projection
account for 71.9%. The ordering changes because serial advection improved while
its aggregate eight-thread time did not; the repeated eight-thread stage is
2.836 s, confirming that this is not a single outlier.

## What the code currently does

### Scheduling model

Dense numerical loops use a persistent `Polyester.@batch` worker pool through
[`src/threading.jl`](../src/threading.jl). Block-backed centred fields schedule
leaf blocks and traverse dense block columns locally. If a topology has too
few leaves, the scheduler tiles columns inside each block. Compact face, dual,
and Krylov arrays retain contiguous flat partitions. Pointer-rich tree searches
use Julia's dynamic scheduler. The adaptive batch size targets four batches
per thread. Scalar loops require 1,024 iterations before threading; expensive
block loops use a four-block threshold. Outputs are owned by blocks, columns,
faces, or rows, and scatter operators are split into parallel face evaluation
and gather stages. This is data-race free and avoids task-local scratch indexed
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

Expected effect: this targets the 1.55x AMR-transfer stage and also removes
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

Expected effect: this directly targets diffusion solve at 1.65x and pressure
projection at 1.97x. It will also show whether the current factorized fine
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
gravity, and pressure-operator updates recur 448 times and contribute 3.00 s
at eight threads in Stokes-3.

### 5. Treat transport as a bandwidth and fusion problem

WY interface and scalar transport approach or exceed 4x on the larger case.
Momentum advection reaches 3.04--3.43x across the two meshes and repeats. The
complete advection stage reaches only 2.16x. The matched mixed-mesh results do
not indicate an interface load imbalance.

The
[`block-locality experiment`](data-locality-results.md) has now been applied
throughout production. Explicit block/local-index loops are 4.2--4.6x faster
for the small centred-cell subkernels, even though the previous `CellRef`
sequence was already block-major. Applying the same owner-block permutation to
flat face arrays is 8--16% slower, and a fully block-grouped momentum component
has no repeatable gain. Production therefore keeps face and dual arrays
contiguous and changes only block-backed cell loops. This improves the serial
complete momentum component and isolated WY kernels.

The matched Stokes-3 result exposes the next issue: serial advection improves
by 10.2%, while its eight-thread aggregate rises from 2.393 to 2.734--2.836 s.
The isolated threaded kernels do not reproduce that regression, so the current
data do not distinguish parallel granularity, barriers, and whole-run context.
Instrument each directional sweep and count its scheduled regions before
fusing block-backed cell work sharing the same immutable sweep inputs. This
tests repeated barriers and task dispatch without undoing the serial locality
gain.

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
