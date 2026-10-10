# Stage 3 all-order scalar-selector study: 10 October 2026

A copied-source selector for all 256 packed four-face orders passed the compact
numerical suite and every accepted full-field comparison, but it did not earn
promotion to the general advection operator. On the 1024² one-block fixture it
cost about 22% more at one worker and all three four-worker pairs were slower.
The many-block eight-worker median was faster, but its three-pair interval
included parity; the one-block eight-worker samples varied substantially.
Production remains the checked narrow-ID implementation at advection SHA-256
`047457e224baa1c5b5dfa26376eff71acd61de02b225ca7e13360df6c3184541`.

The research candidate has advection SHA-256
`a07db90aff3c0e3efa19b1848e202bef7e444c9ee14d4c299ead5e5ec34ad127`.
It replaces only the ordered-sum helper: each packed 2-bit direction chooses a
scalar flux through balanced `ifelse` selection, then the helper starts at
`zero(T)` and performs the same four sequential additions as the control.
There is no canonical-order guard, uniform-grid branch, block-size branch, or
worker-count dispatch. The helper applies inside every AMR leaf. The copied
control and candidate each passed 14,604 assertions at one worker. These
include an independent exact-result check for every UInt8 packed code with
Float32 and Float64 fluxes, promoted accumulation, signed zeros, and
cancellation, followed by the compact mixed-AMR, seam, boundary, cut-cell,
CFL, and stale-workspace tests. This validates numerical behavior; it does not
make the timed uniform fixtures representative of mixed-level cost.

## Complete `advect!` measurements

Both periodic 1024² fixtures use the same prescribed constant face carrier
`(0.7, -0.3)`, initial scalar, `dt=0.35/1024`, and frozen copied-source driver
within each layout. Every timed row is a fresh Julia 1.10.12 process with
pinned workers, a full mesh face/halo check before timing, a minimum 0.2-second
interval, and a final field compared bitwise with a fresh control-source
oracle. The many-block layout ran 128 `advect!` calls per row; the one-block
layout ran 256. There are three independent adjacent control/candidate pairs
per worker count and layout. The table reports medians in nanoseconds per
scalar cell per complete `advect!` call. Ratio intervals are 95% independent-
process percentile bootstraps of control median divided by candidate median;
with only three pairs, wide intervals must not be read as proof of parity.
Candidate efficiency uses the **faster compatible one-worker median** in each
layout, which is the control.

| Grid/layout | Workers | Control → selector ns/cell/`advect!` | Control/selector [95%] | Selector Eₚ [95%] |
|:--|--:|--:|--:|--:|
| 1024², 8×8-cell blocks | 1 | 13.241 → 13.814 | 0.959 [0.940, 0.974] | 0.959 |
| 1024², 8×8-cell blocks | 4 | 5.095 → 5.264 | 0.968 [0.790, 1.015] | 0.629 [0.525, 0.665] |
| 1024², 8×8-cell blocks | 8 | 4.115 → 3.144 | 1.309 [0.972, 1.399] | 0.526 [0.508, 0.534] |
| 1024², one block | 1 | 8.371 → 10.246 | 0.817 [0.814, 0.833] | 0.817 |
| 1024², one block | 4 | 2.127 → 2.705 | 0.786 [0.623, 1.187] | 0.774 [0.614, 0.824] |
| 1024², one block | 8 | 1.580 → 1.752 | 0.902 [0.669, 2.288] | 0.597 [0.580, 0.802] |

The many-block eight-worker paired ratios were 1.367, 1.315, and 0.990;
the one-block eight-worker ratios were 0.669, 1.731, and 1.192. All three
one-block four-worker ratios were below one (0.786, 0.820, 0.902), and the
one-worker regression was consistent across pairs. A small serial cost could
be acceptable for a robust four-/eight-worker complete-step benefit, but
this candidate has no such general result. It remains research-only; no
thread-count- or architecture-specific production policy is justified by
these data.

The independent Julia audit
`benchmark/scaling/stage3/scalar_selector/audit_study.jl` passed both cohorts:
38/38 successful child launches, all 36 timed rows, exact randomized adjacent
job commands, reused compact-suite provenance, locked copied source and driver
hashes, worker placement, elapsed-time normalization, and bitwise full-field
oracle equality. Ignored local receipts are
`benchmark/scaling/results/20261010-scalar-selector-1024b8-fixed-r3/` and
`20261010-scalar-selector-1024b1024-fixed-r3/`; the initial copied-source
validator receipt is in `20261010-scalar-selector-candidate/`. The three-pair
curves are exploratory and fall short of the plan's five-repeat reporting gate.

The fixed carrier allowed the CFL bound to be set before timing. These costs
cover complete `advect!` calls, **not** a dynamic-CFL timestep with
`stable_timestep` charged in every step. No new 2048² result was attempted;
the earlier checked control setup failure remains unresolved as recorded in
the [mesh-construction report](multithreading-stage3-mesh-crash-report.md).
This selector does not establish the Stage 3 performance gate or a full
transport result.

## Next general execution prototype

The next copied-source experiment should reduce full regular-flux-array
accesses inside **each leaf**, including leaves on mixed-level meshes, without
changing face ownership or addition order. First compute owned coarse–fine,
cross-block, periodic-seam, and physical-boundary fluxes from the unchanged
scalar field. Precompute regular faces consumed by leaf-edge cells and tile
seams, with one owner per face: x faces where `j=1` or `j=ny` or `i=1` or
`i=nx−1`; y faces where `i=1` or `i=nx` or `j=1` or `j=ny−1`, plus internal
tile-seam y faces. Deduplicate intersections. Within each deterministic tile,
roll one x-flux row and two y-flux rows while computing every cell's original
ordered integrated sum into `workspace.increment` as temporary storage. Only
**after all sums are complete** should a separate pass read each sum, divide
by its block volume, store the resulting change back into `increment`, and
apply that same unrounded change to the scalar. This matters for Float32
scalars on Float64-volume meshes: updating from the rounded stored increment
can change field bits. The two-pass separation keeps
every upwind face read on the old field and avoids in-place row-dependency
hazards. Handle two-cell-wide or two-cell-high leaves through the all-edge
path. The extra increment-read/update traversal and synchronization must be
measured against the avoided full x/y array accesses; ring operands still
require loads, and cache reuse and write allocation prevent a direct DRAM-byte
claim.

`workspace.increment` interiors and final scalar values have tested exact
behavior and must remain bitwise equal, including mixed-level seams,
boundaries, and cut cells. `workspace.rates` has a tested CFL role. Source
inspection found no package caller, test, or public document that treats
post-call `x_flux` or `y_flux` contents as an output; they are internal scratch
despite being observable fields on an exported workspace type. A prototype
may change their advection representation without a full-array copy solely
to preserve undocumented intermediate contents. Keep cut-cell and CFL paths
unchanged in the first prototype, and give each tile's ring scratch one
workspace-owned writer with a topology-generation check. This is a design
proposal, not a measured speedup.

The local WaterLily reference at `/tmp/twophase-waterlily-reference`, revision
[`2a2b2ba2eeacc885d84e346f743b72a34e0b713f`](https://github.com/WaterLily-jl/WaterLily.jl/blob/2a2b2ba2eeacc885d84e346f743b72a34e0b713f/src/Flow.jl#L38-L52),
separates old-state flux/tendency accumulation from its later state update
(`src/Flow.jl:38–52, 176–195`). It uses a full Cartesian flux array, not the
proposed per-leaf rolling buffer, and its staggered fixed-grid convective/
diffusive method has neither AMR seam ownership nor this operator's packed
four-term reduction order. It is a source-level organization reference, not
a matched performance ceiling.

Leaf blocks are sorted, but mesh face schedules are built by iterating a
coordinate dictionary without sorting its keys. Source therefore does not
guarantee a canonical face-ID and packed-order sequence across fresh
processes; no variation was established in these cohorts. Future studies can
record a logical face-schedule/packed-order hash alongside the full-field
oracle and timings. This uncertainty does not invalidate the accepted paired
results.
