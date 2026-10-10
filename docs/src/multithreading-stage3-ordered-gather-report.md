# Stage 3 ordered-gather follow-up: 10 October 2026

An exploratory copied-source ordered-sum specialization improved the 1024²
one-block general forward-Euler upwind path at one and eight workers, but its
1024² many-block fixture was 42% slower at eight workers by the median of
three paired processes. The candidate is **rejected for production**. The
production operator retains its checked narrow-ID implementation; the focused
regression tests and copied-source benchmark remain for future work. The
large 2048² control setup failed validation before any timing, so there is no
large-grid throughput conclusion.

The candidate selected the two x fluxes and two y fluxes directly for the
four packed face orders built for strict block-interior cells. It then started
at `zero(T)` and added the four terms in their original sequence. Other
packed orders retained the original indexed-tuple helper. The same helper
would serve plain advection, cut-cell advection, and promoted-type CFL sums in
every AMR leaf; it was not a globally uniform transport branch.

The copied-source control uses the checked narrow-ID implementation at source
SHA-256 `047457e224baa1c5b5dfa26376eff71acd61de02b225ca7e13360df6c3184541`.
The measured copied candidate has advection SHA-256
`c4313f0e08c7f1082e89a6d5dbd93751486f4b1affb7e21972be3d8b332a37d4`.
Both copied modules passed the compact frozen-reference suite. The focused
helper check covered all 24 four-face permutations plus two nonpermutation
codes, Float32/Float64 fluxes, Float32/Float64 accumulation, signed zeros, and
cancellation. Existing compact trajectories cover mixed-level AMR, periodic
seams, physical boundaries, cut cells, live face velocities and apertures,
promoted CFL rates, and stale-workspace rejection. Every accepted timed final
field below matched a fresh control-source oracle bitwise. These are numerical
checks; the timed throughput fixture itself is periodic and uniform.

After rejecting the candidate, the restored production advection source has
the original `047457…` hash and passed the focused Julia 1.10.12 suite at
one, four, and eight workers: 12,522 assertions per count, including the 223
new independent ordered-sum assertions. Its ignored local TOML receipts are
in `benchmark/scaling/results/20261010-ordered-gather-production-validation/`.
The complete package suite was not rerun for this research-only comparison.

## Paired complete-advection measurements

The two completed periodic 1024² layouts use the same initial scalar field
within each layout and prescribed constant face velocity `(0.7, -0.3)` in
both variants. The one-block case ran 256 fixed-`dt` calls with
`dt=0.35/1024`; the 8×8-cell-block case ran 64. Each table entry is the
median nanoseconds per scalar cell per complete `advect!` call. The ratio
interval is a 95% independent-process percentile bootstrap of the control
median over the candidate median. `Eₚ` uses the faster compatible one-worker
median, which is the candidate in both layouts. The one-block result pools
three and two adjacent fresh-process pairs into five; the many-block result
pools one and two pairs into three, below the plan's five-repeat gate.

| Grid/layout | Workers | Control → candidate ns/cell/`advect!` | Control/candidate [95%] | Candidate Eₚ [95%] |
|:--|--:|--:|--:|--:|
| 1024², one block, five pairs | 1 | 8.459 → 7.732 | 1.094 [1.064, 1.106] | 1.000 |
| 1024², one block, five pairs | 4 | 2.149 → 2.079 | 1.034 [0.775, 1.485] | 0.930 [0.703, 1.004] |
| 1024², one block, five pairs | 8 | 1.585 → 1.425 | 1.113 [1.076, 1.159] | 0.678 [0.659, 0.698] |
| 1024², 8×8-cell blocks, three pairs | 1 | 13.030 → 12.889 | 1.011 [0.985, 1.032] | 1.000 |
| 1024², 8×8-cell blocks, three pairs | 4 | 7.064 → 4.942 | 1.429 [0.979, 1.476] | 0.652 [0.622, 0.668] |
| 1024², 8×8-cell blocks, three pairs | 8 | 3.151 → 4.467 | 0.705 [0.693, 0.988] | 0.361 [0.354, 0.381] |

The five one-block p4 paired ratios include 0.924 and 1.009, and its
same-worker interval crosses parity; the p4 improvement is inconclusive.
All three many-block p8 paired ratios were below one (0.705, 0.988, 0.693).
That persistent observed regression, although from only three pairs, fails the
candidate's general-path retention check. The many-block p4 control samples
also varied from 5.006 to 7.274 ns/cell, so the apparent p4 gain is
inconclusive. A faster one-block result does not offset the many-block loss.

The read-only audits verified 60/60 successful child launches and all 48
accepted timed rows across the two 1024² layouts, with identical source,
driver, project, input, and full-field oracle hashes within each pair of
cohorts. Every timed field matched its fresh control oracle bitwise. Copied
method paths and source-tree hashes, worker affinity, minimum 0.2-second
timed intervals, sample coverage, and normalization also passed. The Julia
auditors are `benchmark/scaling/stage3/ordered_gather/audit_safe_study.jl`
and `benchmark/scaling/stage3/ordered_gather/audit_combined_study.jl`. Ignored
local receipts are the `20261010-ordered-gather-safe-1024b1024-r3/` and
`-r2/` cohorts, plus `20261010-ordered-gather-safe-1024b8-r1/` and `-r2/`,
under `benchmark/scaling/results/`. The checked driver validates every mesh
face and halo entry before timing; the check is outside the `advect!` interval.

The fixed carrier allowed one pre-timing CFL query, so these measurements do
**not** include `stable_timestep` in every step. They therefore give an
absolute cost for a complete `advect!` call, not for a dynamic-CFL timestep or
a complete flow simulation. The 50% Stage 3 target cannot be declared passed
from these fixtures: the general candidate regressed in many blocks, the
2048² fixture failed before timing, and per-step CFL was outside the accepted
performance interval. A separate 64², two-step, one-worker driver smoke
called `stable_timestep` within each step and produced exactly equal control
and candidate fields and CFL bounds; it is a tooling/numerical check, not a
timing result. Its ignored `20261010-ordered-gather-cfl-smoke/smoke.json`
receipt records both source hashes and the generated-driver hash. Plain
upwind has no reconstruction requirement in this interface-free scope.
Production Lax–Wendroff, WY transport, and phase scalars remain separate
pending work.

## Large-mesh construction status

The earlier copied-source attempt produced no accepted ordered-gather timing
rows after an intermittent 2048² mesh-construction failure. Later diagnostics
located invalid face fields in some processes but did not identify the cause.
This follow-up resumes ordinary numerical optimization with fresh, single-case
children and a full face/halo validation before each timed trajectory. Any
failed setup or receipt stops that case's cohort; an incomplete cohort is not
accepted. Passing a fixture validates that run; it does not resolve the
intermittent construction failure. The earlier evidence remains in the
[mesh-construction report](multithreading-stage3-mesh-crash-report.md).

The follow-up 2048² one-block cohort stopped in its **unchanged control**
oracle child before any timed trajectory. Both copied-module compact suites
passed, then the full mesh scan rejected face 6,163,788 because its axis was
invalid. The oracle child exited 1 intentionally on this validation error.
There is no oracle field, timed row, summary, 2048² ordered-gather speedup, or
2048² efficiency result. Its ignored local receipt is
`benchmark/scaling/results/20261010-ordered-gather-safe-2048b2048-r3/`.
The cause remains unknown; this follow-up did not retry the failed fixture or
change its mesh builder.

## Next general execution experiment

The pair-select specialization should remain research-only. A next low-cost
experiment can select each of the four terms directly from all packed 2-bit
orders, without testing for four canonical codes or dynamically indexing a
tuple. It must retain `zero(T)` and the exact four
sequential additions for all 256 packed codes, including nonpermutations,
and run the same mixed-AMR numerical and one-/many-block performance checks.
This is a hypothesis about branch and stack costs, not an observed speedup.

The larger cost opportunity is a topology-cached rolling regular-flux pipeline
applied **inside each leaf**, including leaves on mixed-level meshes. Compute
each owned coarse–fine, cross-block, periodic-seam, and physical-boundary flux
from the old scalar field first. Also precompute regular fluxes needed by leaf
edge cells and at worker-tile y seams, then synchronize before in-place cell
updates. Each tile can calculate its current x row and a rolling pair of y
rows from the old state, use the packed face order for strict interiors, and
update those cells. A final synchronized pass updates leaf-edge cells from
their original ordered incidence. This avoids writing and rereading full-volume
regular x/y flux arrays while keeping AMR ownership and reduction order.

That design needs per-workspace scratch with a single tile owner, a topology
generation check on its seam schedule, and exact preservation of
`workspace.increment`, whose full interiors are compared with frozen results
in the plain and cut-cell tests. Source inspection found no package caller,
test, or public document that relies on post-call contents of `x_flux` or
`y_flux`; the current advection and CFL methods overwrite them with different
quantities. They are internal scratch in the tested contract, although the
exported workspace type makes its fields observable to external callers.
The in-place update must never overwrite
an old cell value before any neighboring face has read it. Mixed-level,
periodic-seam, physical-boundary, cut-cell, live-velocity/aperture, and
concurrent independent-workspace trajectories need full-field comparison to
the frozen general oracle. The accepted performance comparison should then
charge `stable_timestep` within every complete dynamic-CFL step and report
ns/cell/step and four/eight-worker efficiency separately from the fixed-`dt`
`advect!` diagnostic. This is a proposed experiment, not a performance claim.
