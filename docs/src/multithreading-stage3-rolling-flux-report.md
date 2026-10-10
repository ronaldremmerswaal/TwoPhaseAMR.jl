# Stage 3 copied rolling-flux study: 10 October 2026

The first increment-first rolling-flux candidate preserves the general
upwind operator's numerical result, including mixed-level and uneven tile
fixtures, but its audited 1024² one-block and many-block pilots are slower
than the checked narrow-ID production baseline at every tested worker count.
It is **rejected for production**. A separately frozen row-fused variant passed
the compact/focused fixtures, but its many-block timing cohort stopped during
workspace construction before any candidate timing or matched pair. Its
performance is unknown.

The measured copied control has advection SHA-256
`047457e224baa1c5b5dfa26376eff71acd61de02b225ca7e13360df6c3184541`;
the two-pass candidate has SHA-256
`9383d5b7800300353b2f054efac51b14be48f16aeaabf3eec886f4dc1eeb85b6`.
The candidate changes ordinary `advect!` on every leaf, not a globally uniform
or worker-specific branch. It keeps coarse–fine, cross-block, periodic, and
boundary face ownership and original per-cell reduction order. Cut-cell
advection and `stable_timestep` are unchanged. The full `x_flux`/`y_flux`
arrays remain allocated because those paths use them; ordinary advection
touches only leaf-perimeter and tile-seam slots while a workspace-owned ring
holds other regular fluxes.

The candidate precomputes residual, perimeter, and tile-seam faces from the
old scalar field. Each tile computes its remaining x row and alternating y
rows into disjoint scratch, then stores the original ordered integrated sum
temporarily in `workspace.increment`. After all old-state reads finish, a
second pass divides by block volume, stores the final increment, and updates
the scalar using the same **unrounded promoted change**. This preserves the
original Float32-scalar/Float64-volume arithmetic; updating from the rounded
stored increment would not always preserve field bits. The second traversal
and its synchronization are real costs.

## Numerical and face ownership checks

Copied control and candidate each passed 31,613 assertions at one worker.
The candidate passed 31,625 at each of three, four, seven, and eight workers;
the extra assertions come from thread-dependent tile checks. The frozen
compact suite covers mixed AMR, periodic seams, physical boundaries, cut
cells, promoted CFL, live face values, and stale-workspace rejection. Added
focused checks compare exact scalar and `workspace.increment` fields against
the independent frozen CSR reference on non-square one-block and 2×N/N×2
mixed-level leaves with Float32/Float64 values. The Float32 checks exhibit
cases where rounding the increment before the scalar update changes a bit.
Three- and seven-worker runs exercise partial final column tiles without
changing the validated focused-file hash.

A **separate untimed instrumented copy**, with advection SHA-256
`96cab83e01f6a520a802f123642fd13f12e7a3fa960a460b73ef30f410e37ed3`,
counted common regular-flux helper calls by owned face ID. It passed at one,
four, seven, and eight workers: every regular face was computed exactly once
and every residual face zero times on a non-square one-block case and two
mixed-AMR cases. At eight workers these covered 4,000 regular faces in the
one-block case, and 460 regular plus 206 residual faces in each mixed case.
The instrumented source was not timed. Validation receipts are in ignored
`benchmark/scaling/results/20261010-rolling-flux-candidate/` and
`20261010-rolling-flux-call-count/`.

## Many-block fixed-`dt` pilot

The periodic 1024² fixture has 8×8-cell leaves and prescribed constant
carrier `(0.7, -0.3)`. Each timed child ran 128 complete `advect!` calls at
`dt=0.35/1024`; setup and the full face/halo check were outside the timer.
There are three adjacent fresh-process control/candidate pairs at each worker
count. Values below are medians in nanoseconds per scalar cell per complete
`advect!` call. Ratio intervals are 95% independent-process percentile
bootstraps of control median over candidate median. Candidate efficiency uses
the faster compatible one-worker median, which is the control. Therefore its
reported `E₁` is below one; efficiency against its own slower serial time
would obscure that absolute cost.

| Workers | Control → two-pass ns/cell/`advect!` | Control/two-pass [95%] | Two-pass Eₚ [95%] |
|--:|--:|--:|--:|
| 1 | 12.903 → 20.105 | 0.642 [0.640, 0.650] | 0.642 |
| 4 | 4.959 → 6.192 | 0.801 [0.793, 1.154] | 0.521 [0.519, 0.542] |
| 8 | 3.067 → 5.315 | 0.577 [0.568, 0.967] | 0.303 [0.299, 0.369] |

All three eight-worker paired ratios were below parity (0.569, 0.967,
0.576). The four-worker same-worker interval is wide, yet its observed median
is slower; the consistent serial loss is also large. This is a three-pair
exploratory pilot, below the plan's five-repeat gate, but it is enough to
reject this two-pass candidate as an unconditional general implementation.
The source-level costs provide plausible reasons to investigate, not a
measured attribution: on 8×8 leaves, plain advection still touches 52 of
112 regular full-array slots and 28 of 64 cells use edge CSR, while the new
path adds an integrated-increment write, a final increment read/rewrite,
a separate full-cell update traversal, and synchronization. At 1024² in one
block with eight workers, it touches 39,862 of 2,095,104 regular full-array
slots and adds 98,272 `T` scratch slots across 32 tiles. These are
**array-access and allocation counts**, not measured DRAM bytes or an
explanation inferred from wall times; ring operands still load and
cache/write-allocation effects matter.

The independent read-only Julia audit
`benchmark/scaling/stage3/copied_source/audit_study.jl` passed all 19 child
exits, 18 timed rows, copied source/probe/suite/driver/project/input locks,
randomized adjacent command order, worker placement, time normalization, and
bitwise full-field agreement with a fresh control oracle. The accepted
ignored receipt is
`benchmark/scaling/results/20261010-rolling-flux-1024b8-fixed-r3-retry1/`.
An earlier launch stopped at a suite-provenance harness check before oracle
or timing and left zero accepted rows in
`20261010-rolling-flux-1024b8-fixed-r3/`; the harness check was corrected
and the complete study used a new output directory. This was a tooling stop,
not a numerical failure.

The prescribed carrier allows a CFL bound before timing, so these numbers
cover `advect!`, **not** a dynamic-CFL timestep that charges
`stable_timestep` on every call. No new 2048² case was attempted; the earlier
mesh-construction failure remains unresolved. The many-block pilot cannot
establish the Stage 3 complete-step gate or a full transport result.

## One-block fixed-`dt` diagnostic

The same frozen candidate was compared on a periodic 1024² one-block fixture
with the same carrier and `dt`, using 256 complete `advect!` calls per timed
child and three adjacent fresh-process pairs at each worker count. The
independent audit passed all 19 child exits, 18 bitwise full-field rows,
copied source/probe/suite/driver/project/input locks, randomized command
order, placement, and nanosecond normalization. The accepted ignored receipt
is `benchmark/scaling/results/20261010-rolling-flux-1024b1024-fixed-r3/`.
Values are median ns per scalar cell per complete `advect!`; brackets are
95% independent-process percentile bootstrap intervals. Candidate efficiency
again uses the faster compatible control serial median.

| Workers | Control → two-pass ns/cell/`advect!` | Control/two-pass [95%] | Two-pass Eₚ [95%] |
|--:|--:|--:|--:|
| 1 | 8.472 → 9.591 | 0.883 [0.866, 0.896] | 0.883 |
| 4 | 2.125 → 2.357 | 0.901 [0.600, 0.983] | 0.899 [0.609, 0.924] |
| 8 | 1.585 → 1.785 | 0.888 [0.861, 0.933] | 0.593 [0.591, 0.614] |

All nine paired ratios are below parity, including the variable four-worker
pair (0.964, 0.600, 0.920). Eight-worker efficiency exceeds 50% against the
strongest serial control, but its absolute complete-`advect!` cost is higher.
The three-pair study remains exploratory and cannot establish the five-repeat
Stage 3 gate. Together with the many-block regression, it closes this
two-pass candidate without a thread-count or uniform-layout production
special case.

## Row-fused follow-up: numerical fixtures passed, timing blocked

A separately frozen row-fused copied candidate (advection SHA-256
`d3d8edeb29cef090ca0f4168ba9103f44346425c1d55be40f80905f4e2e21027`)
retains the global
residual/perimeter/tile-seam prepass and the same per-tile ring, then computes
all x and y-up fluxes for row `j` before updating any cell in that row. Its
y-down flux is already cached, and cross-tile y seams are precomputed. Thus
later rows and neighboring tiles do not need old row `j`; mixed-level and
periodic faces were computed in the global prepass. It writes final
`workspace.increment` and updates the scalar from the directly promoted
`change` in the row loop, removing the extra global increment pass/barrier.
This dependency argument is source review supported by finite fixtures, not
a proof for every reachable mesh. Removing an extra pass tests a plausible
cost, not a proven cause of the two-pass wall times. Full-array allocations
remain.

The copied control and fused candidate each passed 31,613 compact/focused
assertions at one worker; fused candidate runs passed 31,625 at each of three,
four, seven, and eight workers, including partial final tiles. A separate
untimed instrumented copy (advection SHA-256
`e2648ac900b3609bc4ae4b5d13f4b604300379bcaf6676cc8267fc371df1aff5`)
passed exactly-once regular-face and zero-residual-face call counts at one,
four, seven, and eight workers. The p8 receipt covers 4,000 regular faces in
one block and 460 regular plus 206 residual faces in each mixed-AMR case.
The frozen validation manifest and count receipts are in ignored
`benchmark/scaling/results/20261010-rolling-flux-fused-candidate/` and
`20261010-rolling-flux-fused-call-count/`. These checks establish exactness
on the tested fixtures, not a complete integration or performance gate.

The fused 1024² many-block fixed-`dt` preflight passed the source, parent,
validator, and copied-control locks. In the attempted fresh-process cohort,
the control oracle and one control p8 timing child completed, but the first
fused p8 child exited with signal 11 while constructing its advection
workspace, after mesh validation and before a field report or timed
`advect!` call. The copied source hashes still matched the configuration.
The independent bounded receipt review found three child launches, one
isolated control raw row, **zero accepted matched timing pairs**, no summary,
and no completed post-run parent receipt. The incomplete cohort is excluded;
its ignored receipt is
`benchmark/scaling/results/20261010-rolling-flux-fused-1024b8-fixed-r3/`.
The failure's cause is unknown, and this receipt does not show that the fused
kernel caused it. No retry or additional throughput experiment was run.

Production remains the checked narrow-ID implementation. Before another
general-AMR throughput comparison, ordinary workspace/setup must complete
dependably under the checked fixture protocol. A successful future study
would still need five independent pairs, a CFL-inclusive complete-step
comparison, and appropriate integration validation; neither rolling variant
clears Stage 3 or the pending general LW work.
