# Stage 3 checked narrow face IDs: 10 October 2026

The general forward-Euler upwind `advect!` workspace now stores ordinary
within-leaf x/y face IDs as `Int32` when a checked face count fits, with a
native-`Int` fallback for larger meshes. The live face-velocity and aperture
reads, residual-face ownership, regular flux calculation, and ordered cell
reduction are unchanged. Focused Julia 1.10.12 suites passed 12,105 checks
at each of p1, p4, and p8; all 90 complete fields in the matched timing
cohort were bitwise equal to the committed control oracle.
The selector uses `Int32` through `typemax(Int32)` faces and native `Int`
beyond that bound; a negative face count is rejected. Focused tests forced
the native-width builder on a small periodic/seam and physical-boundary
mesh, rather than allocating billions of faces. Both widths matched frozen
plain/cut-cell fields and promoted-type CFL rates under live velocity and
aperture mutation, and both rejected a stale workspace after mesh topology
invalidation. The full package suite was not rerun for this change.

The direct ID payload halved. The deduplicated reachable mesh/field/velocity/
workspace graph fell by 8.38 MB (3.10%) at 1024² in one block, 33.54 MB
(3.11%) at 2048² in one block, and 7.34 MB (1.96%) at 1024² in 8×8-cell
blocks. Timing changed less consistently. The five-repeat p1 medians favor
the candidate at 2048² one block and 1024² in small blocks, but do not
establish a gain at 1024² one block. The 2048² p4 median is 1.32× faster,
yet its 95% interval [0.951, 1.353] crosses parity; several control p4
samples occupy a slower mode. The 50%-of-strongest-serial efficiency gate
is supported at p4 for all three fixtures; at p8 it is supported only for
1024² one block, missed for 2048² one block, and inconclusive for 1024²
small blocks. Narrowing the IDs is justified by exact storage saving and
unchanged numerical behavior, without a claim of broad throughput gain.

## Matched complete-step cohort

The control is the **committed block-interior implementation** at Git
revision `5f0081d9b2677ddc676c855f9cb5db4d218165dd`, with advection
SHA-256 `cecb8463cccada2e16d90f5245b506842d9cc90f6df9c380bb69fee673e4f115`.
The candidate SHA-256 is
`047457e224baa1c5b5dfa26376eff71acd61de02b225ca7e13360df6c3184541`;
the embedded operator source is unchanged. This control differs from the
**previous general remap** used in the earlier
[block-interior report](multithreading-stage3-block-interior-report.md),
so its medians must not be substituted for that older baseline.

The table gives median nanoseconds per transported scalar cell and complete
`advect!` step, committed control → narrow IDs. `Eₚ` is the candidate's
strongest-compatible-serial efficiency, with independent-process 95%
bootstrap intervals at p4/p8. The p1 point estimate is one when the
candidate is its own strongest serial reference; p1 self-ratio intervals
are omitted. A gate is supported only when its whole interval is above
0.5, missed only when wholly below, and otherwise inconclusive.

| Grid/layout | p1 control → narrow | p4 control → narrow | Narrow E₄ [95%] | p8 control → narrow | Narrow E₈ [95%] |
|:--|--:|--:|--:|--:|--:|
| 1024², one block | 8.770 → 8.802 | 2.203 → 2.281 | 0.961 [0.902, 1.156], supported | 1.644 → 1.634 | 0.671 [0.651, 0.855], supported |
| 2048², one block | 10.483 → 10.122 | 4.351 → 3.295 | 0.768 [0.646, 0.787], supported | 3.366 → 3.305 | 0.383 [0.371, 0.406], missed |
| 1024², 8×8-cell blocks | 13.256 → 13.021 | 5.279 → 4.937 | 0.659 [0.569, 0.669], supported | 4.755 → 4.322 | 0.377 [0.325, 0.527], inconclusive |

Same-worker control/candidate point ratios range from 0.965 to 1.320 at
p4 and from 1.006 to 1.100 at p8. Every p4/p8 same-worker 95% interval
crosses 1.0, including the apparently large 2048² p4 gain. That control
p4 group's five observations are 3.376, 3.393, 4.351, 4.363, and 4.401
ns/cell-step, so its median is sensitive to which timing mode a small
sample captures. The two p1 gains with intervals wholly above parity are
2048² one block, 1.036× [1.024, 1.054], and 1024² small blocks,
1.018× [1.005, 1.110].

Each source ran five fresh processes at p1/p4/p8 across the three cases:
30 timed Julia children produced 90 accepted fields. The 35 captured
launches also include initial-field and oracle creation; all exited
successfully. The minimum accepted timed interval was 0.327 s, above the
predeclared 0.2 s floor. One child batches the three cases for a source,
worker count, and repeat, and the job/case order is recorded. All source,
initial, oracle, field, timing, and affinity receipts passed revalidation
after completion. Timed trajectories used 256 steps for 1024² and 32 for
2048², identical inputs and `dt=0.35/n` in each source pair. The periodic
carrier was `(0.7, -0.3)` and the public general `advect!` performed the
entire upwind update. Pinned worker placements and machine metadata are in
the ignored local
`benchmark/scaling/results/20261010-narrow-ids-final/` receipts:
`raw.jsonl`, `summary.json`, `summary.csv`, `manifest.json`,
`configuration.json`, `commands.jsonl`, and `summary.png`. The coordinator
used Julia 1.12.7; numerical children used frozen Julia 1.10.12 and its
manifest. The exact original schema-1 control freeze and a clean-checkout
schema-2 recreation recipe are documented in
`benchmark/scaling/stage3/narrow_ids/README.md`.

## Storage, workers, and separately measured CFL

The direct x/y ID arrays have equal lengths in these square fixtures. The
table reports their *combined* payload; `Base.summarysize` reports the
deduplicated reachable graph of `(mesh, field, velocity, workspace)`.
Workspace summary size by itself includes reachable mesh references and
must not be added to the mesh size. These are allocated/reachable bytes,
not RSS, cache occupancy, or measured DRAM traffic.

| Grid/layout | ID type control → narrow | Combined ID bytes control → narrow | Whole graph MiB control → narrow |
|:--|:--|--:|--:|
| 1024², one block | Int64 → Int32 | 16,760,832 → 8,380,416 | 257.71 → 249.72 |
| 2048², one block | Int64 → Int32 | 67,076,096 → 33,538,048 | 1029.41 → 997.43 |
| 1024², 8×8-cell blocks | Int64 → Int32 | 14,680,064 → 7,340,032 | 357.26 → 350.26 |

The exact bytes, source/driver hashes, and individual mesh/workspace
sizes are in ignored `20261010-narrow-ids-storage/id-storage-evidence.json`.
Untimed, source-specific copied probes checked the 2048² one-block case at
p4 and p8. For both control and candidate, **every requested worker ID**
entered regular-flux, residual-flux, and gather callbacks, and all four
one-step fields matched the clean oracle bitwise. Participation does not
establish equal work or speed. These four rows, copied-source patch hashes,
and CPU placements are in ignored
`20261010-narrow-ids-worker/worker-evidence.json` and `probes/`.
The earlier block-interior study supplies broader 18-row worker coverage;
the scheduler/loop topology did not change here. Independently warmed
allocation checks at 64² and 256² remained 0 B per `advect!`/CFL call at
p1 and bounded 2,656/2,064 B at p4/p8, with no growth across those sizes.

The prescribed carrier is constant during these matched trajectories, so
`stable_timestep` was **outside** the `advect!` timer. Its separate
five-process median scan time, in milliseconds, was:

| Grid/layout | p1 control → narrow | p4 control → narrow | p8 control → narrow |
|:--|--:|--:|--:|
| 1024², one block | 10.64 → 10.03 | 3.38 → 3.23 | 2.50 → 2.55 |
| 2048², one block | 48.02 → 46.71 | 18.93 → 15.90 | 14.80 → 14.38 |
| 1024², 8×8-cell blocks | 11.44 → 11.36 | 5.20 → 4.80 | 4.26 → 4.20 |

These scan numbers do not turn the timed result into a complete
dynamic-CFL step. That Stage 3 gate remains open.

## Ordered gather inspection and next work

A separate read-only Julia 1.10.12 generated-code inspection used the
actual Float64 workspace and a representative copy of the gather loop
with a runtime-loaded packed face order. Its optimized IR retains four
dynamic tuple selections; the corresponding LLVM stores four fluxes in
a small stack array and performs four indexed loads and four sequential
additions. The candidate and committed control showed the same pattern.
The actual `advect!` entry lowered to a thin wrapper in this inspection,
so this representative loop is **not** proof of the exact inlined threaded
closure cost or a measured bottleneck. Package-built strict interiors had
four observed order codes corresponding to x-face pair reversal and y-face
pair reversal; any future specialization must validate that invariant and
retain a generic fallback and the exact sequential reduction order.
Typed, LLVM, native, order-histogram, source-hash, and loaded-method-path
receipts are in ignored `20261010-narrow-ids-gather-ir-control/` and
`20261010-narrow-ids-gather-ir-candidate/`.

An isolated copied-source pair-selection prototype subsequently passed
12,105 compact checks plus 222 exact order/type cases on each of the
control and candidate modules (36 for the four observed codes
and 186 generic-fallback cases across 24 permutations). Its candidate
advection SHA-256 was
`c4313f0e08c7f1082e89a6d5dbd93751486f4b1affb7e21972be3d8b332a37d4`;
the loaded method paths and hashes were checked. **It has no accepted
performance result.** In the first fresh-process attempt, the control
1024² and 2048² oracles completed, but the first timed control p1 batch
segfaulted while constructing the 2048² one-block mesh. One fresh identical
retry segfaulted during 2048² oracle mesh construction, after 1024².
Both attempts used identical source, driver, input, and Julia 1.10.12
project hashes.
Both stacks point to unchanged `mesh.jl::_face_dual_volumes` at line 208
and record exit code −11. The ignored `20261010-ordered-gather-pilot/` and
`20261010-ordered-gather-pilot-retry1/` directories preserve the two
`failure-*.json` receipts and copied-source validation receipts; neither
has accepted raw or summary timing rows. The 90-row narrow-ID cohort
succeeded independently, but the crash cause is unknown. The prototype
is not promoted. Before another gather timing attempt, isolate the
control's mesh-construction failure with a minimal source-specific
reproducer and exact memory/topology-phase logging.

The next accepted measurement should charge `stable_timestep` inside every
complete step, with matched dynamic-CFL trajectories and bitwise fields,
before claiming a general timestep gate. The remaining general-upwind
cost warrants isolated experiments on ordered gather/locality and a
per-leaf rolling regular-flux buffer. Such a buffer would need to compute
each owned face once from the old scalar state, preserve y-face values at
worker-strip seams, and keep block-edge incidence and public workspace
semantics intact. This is a source design, not a measured speed claim.
General production Lax–Wendroff numerical work, WY transport, and phase
scalars remain separate deferred tasks. The historical dense toy and LW
pipeline timings in the [earlier Stage 3 report](multithreading-stage3-report.md)
use different methods and trajectories; their absolute cost gap remains
context, not a matched narrow-ID speedup. Overall Stage 3 is not accepted.
