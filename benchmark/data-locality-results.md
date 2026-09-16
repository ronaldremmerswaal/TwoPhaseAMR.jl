# Block-major locality experiment

## Question and scope

The mesh stores centred fields as one dense haloed matrix per leaf block, but
several transport kernels traverse flattened cell or face schedules. Before
changing their parallel scheduling, this experiment asks whether a serial
block-first traversal is faster for compatible staggered two-phase advection.

The experiment is implemented in
[`staggered_two_phase_locality.jl`](staggered_two_phase_locality.jl). It changes
only traversal order and loop structure. It does not change interpolation,
the dual-volume graph, phase fluxes, or floating-point operation order within
one output. Every compared array is required to be bitwise equal before the
script writes its results.

## Method

The benchmark uses 8x8 leaf blocks and two meshes:

| Case | Blocks | Cells | Interior faces |
|---|---:|---:|---:|
| Uniform level 5 | 1,024 | 65,536 | 130,816 |
| Mixed levels 3--6 | 880 | 56,320 | 113,040 |

An interface is initialized from a smooth wave level set. One ordinary WY
step supplies geometric liquid/gas volume fluxes, cell dilation, and fraction
increments. Those setup operations and every traversal-plan construction are
outside the timed regions. The measured momentum kernel is the transverse
advection of the x-face velocity during the y sweep, at density ratio 1,000.

Each operation has 15 timed samples after compilation; the table reports the
median. Composite momentum inputs are reset outside every timed sample. All
kernels are allocation-free. The run used Julia 1.12.7, one Julia thread, one
BLAS thread, and an AMD Ryzen 9 5950X. An independent Julia process repeated
the complete experiment.

Four comparisons separate distinct storage layouts:

1. **Centred cell update:** the production-style flat `Vector{CellRef}` loop
   versus `for block`, then local `j, i`, with block matrices hoisted outside
   the inner loop.
2. **Centred-to-face interpolation:** the production scan of the mesh face
   vectors, an axis-specific global index schedule, and an owner-block grouped
   schedule. The latter two use the same precomputed face plan.
3. **Dual phase transport:** contiguous dual-output order versus the same
   outputs grouped by the leaf block that owns the physical staggered face.
4. **Complete momentum component:** the current production function, a fully
   grouped block-major variant, and a hybrid that changes only the two loops
   over block-backed `CellField` data. The hybrid retains production traversal
   for flat face and dual-schedule vectors.

The primary raw result is
[`staggered-two-phase.csv`](data-locality-results/staggered-two-phase.csv); the
independent run is
[`staggered-two-phase-repeat.csv`](data-locality-results/staggered-two-phase-repeat.csv).

## Results

### Primary run

| Kernel | Case | Global traversal | Block traversal | Global / block |
|---|---|---:|---:|---:|
| Predicted cell fraction | Uniform | 0.199 ms | 0.045 ms | **4.45x** |
| Predicted cell fraction | Mixed | 0.162 ms | 0.036 ms | **4.44x** |
| Phase face average | Uniform | 0.418 ms | 0.483 ms | **0.86x** |
| Phase face average | Mixed | 0.365 ms | 0.397 ms | **0.92x** |
| Dual phase transport | Uniform | 3.450 ms | 3.301 ms | **1.05x** |
| Dual phase transport | Mixed | 3.173 ms | 2.868 ms | **1.11x** |
| Complete component, all block-grouped | Uniform | 6.242 ms | 6.617 ms | **0.94x** |
| Complete component, all block-grouped | Mixed | 5.396 ms | 5.547 ms | **0.97x** |
| Complete component, block cells/flat faces | Uniform | 6.242 ms | 5.791 ms | **1.08x** |
| Complete component, block cells/flat faces | Mixed | 5.396 ms | 4.674 ms | **1.15x** |

The axis-specific indexed global face average is also slower than the direct
production face scan: 0.472 versus 0.418 ms on the uniform mesh and 0.398
versus 0.365 ms on the mixed mesh. Removing the axis test does not compensate
for indirect access through the index map.

### Reproducibility

The independent process preserves all conclusions:

| Comparison | Uniform speedup range | Mixed speedup range |
|---|---:|---:|
| Nested blocks for centred cells | 4.24--4.45x | 4.34--4.44x |
| Block-grouped face interpolation | 0.86--0.88x | 0.86--0.92x |
| Block-grouped dual outputs | 1.05--1.05x | 1.10--1.11x |
| Fully block-grouped component | 0.94--1.00x | 0.97--1.03x |
| Block cells with flat faces | 1.08--1.12x | 1.15--1.18x |

The fully block-grouped component ranges from a 6% regression to a 3% gain.
That is not a useful or robust optimization. The hybrid improves both cases
in both processes: elapsed time falls by 7--10% on the uniform mesh and
13--16% on the mixed mesh.

## Interpretation

The current flattened cell references are created in block-major order in
[`_cell_references`](../src/elliptic.jl), so the 4.2--4.5x cell-loop gain does
not come from a different sequence of cells. It comes from representing that
sequence directly in the loop nest: each block matrix is loaded once, the
halo and local bounds are loop invariants, and every inner access uses simple
matrix indices instead of a `CellRef` followed by a lookup in the vector of
block matrices.

Face data have a different layout. `FaceField` values and dual-volume
connectivity are compact flat vectors. The production
[`_phase_face_average!`](../src/phase_staggered_advection.jl) reads the mesh
face vector sequentially and writes the destination vector sequentially.
Grouping faces by a cell-block owner improves locality for one centred input,
but adds a permutation read and scatters the face-vector writes. The measured
8--16% regression shows that the latter costs dominate.

The dual transport graph receives a modest 5--11% benefit from geometric
grouping because neighbouring output faces reuse nearby states. That gain is
too small to offset block-grouped face interpolation and its extra schedule
lookups in the full update. It would also require another topology-dependent
permutation, whose construction and retained memory are intentionally excluded
from these best-case timings.

## Production implementation and serial verification

The block-backed part of the measured hybrid was then applied throughout the
numerical code. Cell initialization and integration, level-set initialization,
interface reconstruction and height passes, centred advection, WY interface
and scalar transport, compatible momentum preparation, viscosity refresh, and
the pressure right-hand side now use one common block-column scheduler. Direct
restriction and prolongation use the same column-major tiling for a single
dense block. The WY workspace no longer retains a duplicate flattened
`CellRef` vector.

The scheduler visits every leaf block once in serial execution. In threaded
execution it distributes blocks first and subdivides block columns when there
are too few leaves, including a uniform mesh represented by one large block.
Compact face fields, dual schedules, and Krylov vectors retain contiguous flat
traversal, as the experiment above recommends.

The complete one-thread core suite was measured twice after the change, with
seven timed samples per kernel. Speedup is relative to the recorded pre-change
one-thread result in
[`threaded-kernel-results/threads-1.csv`](threaded-kernel-results/threads-1.csv).
The ranges below contain both independent post-change processes.

| Kernel | Uniform speedup | Mixed speedup |
|---|---:|---:|
| Cell initialization | 1.47x | 1.49x |
| Level-set initialization | 1.04--1.07x | 1.09x |
| Interface reconstruction | 1.09--1.10x | 1.14--1.19x |
| Centred cell advection | 1.14--1.16x | 1.14--1.16x |
| WY interface advection | 1.10--1.11x | 1.07x |
| WY phase-scalar advection | 1.22--1.27x | 1.16--1.18x |
| WY momentum advection | 1.14--1.22x | 1.12--1.17x |
| Diffusion-coefficient refresh | 1.06--1.08x | 1.05x |

Direct prolongation improves by 2.31--2.40x for a 128x128 block and
2.42--2.80x for an 8x8 block. Restriction improves by 1.21--1.22x for the
large block; the sub-microsecond 8x8 restriction is unchanged within timing
noise. Every checksum agrees with the pre-change result. Operations that were
allocation-free remain allocation-free, and the public WY operations retain
their previous allocation totals. Raw post-change results are in
[`core-kernels-after.csv`](data-locality-results/core-kernels-after.csv) and
[`core-kernels-after-repeat.csv`](data-locality-results/core-kernels-after-repeat.csv).

Repeating the isolated momentum benchmark against the resulting production
code confirms the storage-specific choice. Its block-backed cell subkernel is
4.45x faster on the uniform mesh and 4.56x faster on the mixed mesh. Grouping
the flat face interpolation remains slower, by 12% and 14%, respectively.
The complete production component is 4.5% faster than the indexed flattened
cell reference variant on uniform blocks and 3.2% faster on the mixed mesh.
All compared outputs are bitwise equal and every timed kernel is
allocation-free. These measurements are in
[`staggered-two-phase-after.csv`](data-locality-results/staggered-two-phase-after.csv).

## Decision

The measurements support a storage-specific traversal policy:

- traverse block-backed `CellField` kernels as block, local `j`, local `i`,
  and hoist the block arrays;
- keep compact `FaceField`, `FaceVelocity`, and dual-graph kernels in their
  contiguous global order;
- do not add an owner-block face permutation solely for staggered two-phase
  advection.

This hybrid is also the right starting point for threading. A parallel cell
iterator should distribute leaf blocks and run a dense local loop inside each
block, while retaining a fallback that subdivides a block when the topology
has too few leaves. Face-vector kernels should continue to partition their
contiguous output ranges.
