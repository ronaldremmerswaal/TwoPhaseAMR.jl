# Stage 3: general cell-centred transport on uniform meshes

The [10 October 2026 block-interior follow-up](multithreading-stage3-block-interior-report.md)
compares a later general-advection implementation against the previous
general-remap working tree. This report retains its September frozen-source
cohorts and numerical scope.

The selected main-package change makes the **existing general upwind method**
faster without changing its complete fields. In 180 accepted samples, the
remapped workspace was the fastest serial among the compared general-upwind
variants for both layouts at 512², 1024², and 2048². At 2048² in one block, four workers took
6.667 ns per unknown per complete step, versus 11.714 ns for frozen general
upwind. Its strongest-serial parallel efficiency was 0.640 (95% bootstrap
interval 0.625–0.677); at eight workers it fell to 0.335 (0.313–0.355).

The requested **absolute cost comparable to the dense Julia toy is not
achieved**. At 2048², selected general upwind costs 6.667/6.356 ns per
cell-step at four/eight workers; the original dense toy costs 1.766/1.924
and the separate research tile pipeline costs 1.284/1.136. These are cost
references in identical units, not numerical method speedups: general
upwind is a one-step first-order method, while the toy and pipeline perform
three Lax–Wendroff sweeps. The full Stage 3 target remains open because this
study does not deliver a production general LW implementation or an interface
workload. The uniform-only CellField LW prototype stays in research.

## Accepted production matrix

Each size/layout/worker/source configuration has five independent fresh
Julia 1.10.12 processes. A process runs six size/layout cases sequentially,
resetting and warming each field and running GC between cases; cases in one
process may have correlated machine conditions. The fixed step counts are
1024/128/16 at 512²/1024²/2048². Frozen and current use identical input
bytes, `dt=0.35/n`, and steps at each size. Every timed step calls the public
general `advect!`, including owned-face fluxes and CSR gather/update; fixed
CFL selection, setup, pinning, and export are outside the matched timers.
The shortest accepted interval is 0.353 s. All **180 complete output fields
matched the frozen one-worker upwind oracle bitwise across every cell**, and
the frozen one-block and 8×8-cell-block oracle fields matched each other at
each size. Checked outputs were deleted after comparison; inputs, oracles,
hashes, summaries, and command receipts remain in ignored local results.

The table reports median **ns per unknown per complete step**. Each `F → C`
entry is frozen → selected current general upwind at that worker count.
One unknown is one advected scalar cell; prescribed velocity components are
excluded. Efficiency is `Eₚ = T₁ / (p × Tₚ)`, using the strongest compatible
one-worker median for `T₁` and the current median for `Tₚ`. Current is the
strongest serial reference for every size/layout. Brackets are 95%
independent-process bootstrap intervals from
five repeats. An efficiency interval wholly above/below 0.5 supports/misses
the 50% target; one spanning 0.5 is inconclusive.

| Grid | Layout | p1 F → C | p4 F → C | Current E₄ [95%] | p8 F → C | Current E₈ [95%] |
|:--|:--|--:|--:|--:|--:|--:|
| 512² | one block | 15.088 → 10.807 | 5.581 → 2.589 | 1.044 [0.998, 1.116] | 3.378 → 1.410 | 0.958 [0.906, 1.027] |
| 512² | 8×8-cell blocks | 16.500 → 12.746 | 8.043 → 5.045 | 0.632 [0.590, 0.667] | 4.785 → 3.356 | 0.475 [0.438, 0.518] |
| 1024² | one block | 20.040 → 14.331 | 9.234 → 4.501 | 0.796 [0.744, 0.820] | 7.698 → 4.092 | 0.438 [0.427, 0.456] |
| 1024² | 8×8-cell blocks | 23.061 → 14.899 | 11.331 → 7.215 | 0.516 [0.493, 0.557] | 9.371 → 6.513 | 0.286 [0.279, 0.304] |
| 2048² | one block | 26.761 → 17.059 | 11.714 → 6.667 | 0.640 [0.625, 0.677] | 9.857 → 6.356 | 0.335 [0.313, 0.355] |
| 2048² | 8×8-cell blocks | 41.731 → 17.850 | 15.427 → 9.106 | 0.490 [0.471, 0.517] | 11.672 → 8.089 | 0.276 [0.190, 0.288] |

Four-worker efficiency is supported above 50% for both 512² layouts and
the 1024²/2048² one-block layouts; it is inconclusive for the two larger
8×8-cell-block layouts. Eight-worker efficiency is supported only for the
512² one-block layout, inconclusive for its 8×8-cell-block layout, and
misses at 1024²/2048². The 512² one-block E₄ median above one has an
interval that includes one; its cause was not isolated. Per-sample values, MAD,
and confidence intervals are in ignored `summary.json` and `summary.csv`.

The side-by-side table is an **absolute cost comparison only**. General
upwind uses one main-package block; the toy and pipeline use dense LW arrays.
The methods share initial cell averages, carrier, and `dt`, but their final
measurement trajectory lengths differ (main: 128/16 steps at 1024²/2048²;
toy/pipeline: 2048/128). No upwind/LW numerical parity or method speedup is
inferred.

| Grid | Workers | Frozen general upwind | Current general upwind | Dense toy LW | Research pipeline LW |
|:--|--:|--:|--:|--:|--:|
| 1024² | 4 | 9.234 | 4.501 | 0.480 | 0.578 |
| 1024² | 8 | 7.698 | 4.092 | 0.277 | 0.330 |
| 2048² | 4 | 11.714 | 6.667 | 1.766 | 1.284 |
| 2048² | 8 | 9.857 | 6.356 | 1.924 | 1.136 |

## Method and separate diagnostics

The benchmark uses a periodic unit-square mesh, prescribed face speeds
`u=0.7`, `v=-0.3`, Float64, the toy's exact smooth cell averages scattered
into every `CellField` block, and the fixed timestep `0.35/n`. A one-block
layout and a partition into blocks of 8×8 cells cover the same `n²` cells;
the latter has 4,096/16,384/65,536 blocks at 512²/1024²/2048². Every
timed production step calls the public general `advect!` method.
Mesh/workspace construction, worker pinning, fixed-CFL selection, and
full-field export are outside the fixed-trajectory timer; the copied-source
worker checks are separate. The `stable_timestep` cost is separately
reported. Julia 1.10.12 uses the frozen dependency project and a
physical-core spread pin order on the Ryzen 5950X host. Primary production
counts are p1/p4/p8; the research pipeline also measures p2/p16/p32, with
p32 using SMT.

The preimplementation source is frozen under the ignored local
`benchmark/scaling/results/20260930-stage3-production-baseline-freeze/`
directory. Its 92 Julia source files are byte-identical to Git revision
`ace94c888012d7cea00cf9545046a5ce5794325b`; the snapshot can be
recreated from that revision. The exact resolved Julia 1.10 project and
manifest are also local ignored artifacts. A fresh checkout must recreate
that environment or declare a new dependency cohort. The benchmark harness
records source hashes, a local source
snapshot, commands, CPU affinity, complete field hashes, and per-cell
tolerance checks. A frozen one-worker upwind trajectory is the numerical
oracle for every production layout and worker count. The original dense toy
is a useful absolute cost and locality reference in the same
nanoseconds-per-cell-step units, but it advances with a different three-sweep
LW method; no upwind numerical parity or speedup denominator is inferred from
the toy.

## Measured general-path costs

An untimed instrumented copy of each frozen/current source recorded the
workers entering the owned-face flux pass and cell gather callback at 1024².
For both layouts and p1/p4/p8, every requested worker participated in both
passes in all 12 checks. Every instrumented complete field was bitwise
identical to the clean frozen oracle. Production source contains no probe
branch, so the accepted timers are uncontaminated. This proves worker
participation, not equal work distribution.

A second copied-source diagnostic bracketed each complete pass with one
time_ns pair per step at 2048² in one block, over 16 steps. Its fields also
matched the clean oracle. These are single-process **diagnostic wall-time
fractions**, not accepted five-repeat estimates or hardware counters.
Instrumented total times differed from clean medians by at most about 4%.

| Source | Workers | Interior flux share | CSR gather/update share | Boundary share |
|:--|--:|--:|--:|--:|
| Frozen | 4 | 62.3% | 37.7% | ≈0% |
| Frozen | 8 | 68.0% | 32.0% | ≈0% |
| Current | 4 | 66.4% | 33.6% | ≈0% |
| Current | 8 | 64.3% | 35.7% | ≈0% |

Both owned-face fluxes and per-cell CSR gather remain material costs. The
current remap improves cell/flux locality, while live `FaceVelocity` reads
still use original mesh-face order. Strided vertical-face reads and
indirection are plausible remaining costs, but no DRAM-byte or bandwidth
measurement identifies their fraction. The main step has no halo fill or
full-field copy. [WaterLily's pinned local reference](multithreading-waterlily-reference.md)
informed the dense Cartesian, output-owned comparison; its QUICK flux and
published Float32 timings are not LW parity or a hardware ceiling.

The separate warmed `stable_timestep` scan is outside this prescribed-carrier
trajectory, as in the toy. At 1024² in one block, frozen/current median
scans were 255.7/12.3 ms at p1 and 40.3/4.7 ms at p8; at 2048² they were
1138.2/51.8 ms at p1 and 165.4/26.1 ms at p8. The typed current helper
removes the frozen closure's per-cell allocation. In identical Julia 1.10
call contexts at N128 and N512, the warmed frozen CFL call allocated about
160 B/cell; current allocated 0 B at p1 or a fixed 1,904 B at p4. Warm
`advect!` allocated 0 B for both sources at p1 and fixed 2,720/2,448 B for
frozen/current at p4. Native Julia 1.12.7 passed focused numerical tests
and exact-zero warmed allocation assertions. A simulation that recomputes
CFL each step must charge that scan; this fixed-trajectory study does not.

The selected workspace adds one 48-byte descriptor for each owned face,
including its original face ID for live velocity access. A 1024² periodic
fixture has 2,097,152 owned faces, so the increment is exactly 96 MiB.
Single-process `Base.summarysize((mesh, field, velocity, workspace))`
diagnostics estimate the **deduplicated reachable Julia object graph** as
276.975 → 372.975 MiB for one block and 349.006 → 445.006 MiB for
8×8-cell blocks, frozen to current. The mesh graph alone was unchanged at
176.377/232.751 MiB respectively. These sizes are neither process RSS nor
measured memory traffic; separately reported mesh/workspace `summarysize`
values share references and must not be added. Construction times were
measured once and include compilation. The +96 MiB cost is the explicit
storage tradeoff for the observed upwind improvement.

## Correctness and historical general-path pilots

The selected remap preserves the original upwind flux formula, boundary
handling, live velocity reads, and each cell's CSR reduction order. Independent
frozen-reference tests passed at p1/p4/p8: 9,368 plain-scalar and 516
cut-cell checks per process, including one/many blocks, mixed/coarse–fine
meshes, physical boundaries, Float32/64, velocity changes, conservation,
and stale-generation behavior. A separate concurrent-workspace check
passed 128/128 at p4. Existing upwind numerical checks passed 18/18 at
p1/p4/p8. The strict zero-allocation assertions passed on native Julia
1.12; in Julia 1.10 the small caller has bounded preexisting overhead,
so those exact-zero assertions do not pass there. The full package suite
was not rerun. The uniform-only `CellField` LW prototype was moved to
research; it is not a duplicate production kernel or an upwind numerical
reference.

Before final selection, bounded one-process pilots studied compact face
metadata and spatially ordered faces. At 1024² one block and eight workers,
frozen/compact/ordered costs were 7.888/5.457/4.633 ns per cell-step; the
selected remap's independent pilot was 3.964. At 2048² one block and
four/eight workers, the ordered pilot cost 9.273/7.628 against frozen
12.227/10.096. These pilots have separate ignored source snapshots and
receipts; they are not pooled into the five-repeat final matrix.

## Independent research LW pipeline

The separate tile pipeline keeps the toy's public `q` array identity. Every
timed complete step includes precondition checks, halo recomputation,
intermediate writes, and parallel copyback; full-field oracle comparison
and export occur **after** timing. The selected 128×32 tile and original
toy's 512×16 tile used matched input, `dt`, and fixed trajectories at each
size: 2048/128/32 steps at 1024²/2048²/4096². Five fresh pinned processes
per method/worker/size produced 180 accepted samples, all at least 0.372 s.
Every complete field matched the serial toy bitwise, `q` retained its identity,
and every requested worker participated in compute and copyback. The
pipeline study covers p1/2/4/8/16/32; the production primary matrix covers
p1/4/8.

| Grid | Workers | Toy → pipeline ns/cell-step | Same-worker toy/pipeline gain [95%] |
|:--|--:|--:|--:|
| 1024² | 4 | 0.480 → 0.578 | 0.830 [0.770, 1.006] |
| 1024² | 8 | 0.277 → 0.330 | 0.839 [0.760, 0.961] |
| 2048² | 4 | 1.766 → 1.284 | 1.376 [1.316, 1.496] |
| 2048² | 8 | 1.924 → 1.136 | 1.693 [1.481, 1.811] |
| 4096² | 4 | 2.446 → 1.782 | 1.373 [1.258, 1.409] |
| 4096² | 8 | 2.487 → 1.677 | 1.483 [1.457, 1.506] |

The pipeline is slower at 1024² in the medians; the p4 interval includes
parity. At 2048² and 4096² it delivers an absolute same-worker gain with
full copyback charged. At 2048² its own serial-relative E₄/E₈ values are
0.679 [0.648, 0.727] / 0.383 [0.335, 0.406]. At 4096² they are
0.503 [0.461, 0.956] / 0.267 [0.257, 0.508]; a retained 6.814 ns serial
outlier broadens those intervals, and the latter spans the 0.5 gate. The
matched original-toy serial reference gives lower E₄/E₈ values of
0.578/0.327 at 2048² and 0.455/0.242 at 4096². An earlier Stage 2
observable-halo serial implementation was faster in separate cohorts
(2.446 ns at 2048² and 2.674 ns at 4096²), so this study does not establish
the global 5% serial-regression guard. This is a research result, not a
production general-advection integration or a DRAM-traffic proof.

## Interruptions and next design

Two production timing interruptions remain in the receipts. An initial
42-row cohort had a Julia abort before project load; one exact retry
completed and passed numerics, but a 0.184 s 512² interval failed the
predeclared 0.2 s gate. Those rows are diagnostic only, and the complete
180-row cohort uses a longer 512² trajectory. In that cohort one frozen p8
process segfaulted during 2048², 8×8-cell-block halo-donor packing before
the final case's timer. Its five preceding partial fields/reports and stack
receipt remain. A separate fresh process for the failed configuration
passed bitwise field and duration checks, followed by one exact six-case
batch retry that passed all checks; its six rows were appended once. The
cause of either Julia failure is undetermined. The research pipeline
separately retains one pre-timer toy p32 compiler/GC SIGSEGV at 2048²;
its exact retry passed and the failed launch contributes no timing sample.
These are distinct signatures, not evidence of one inferred root cause.

The next production design should address both measured general-path
costs: a regular-interior engine applied within **every** AMR leaf, with
one upwind flux formula and explicit ownership of residual faces for block
interfaces, coarse–fine transitions, physical boundaries, and periodic
seams. Even a seam whose endpoints lie in one large block cannot be treated
as an arithmetic-neighbor interior face. Live carrier mutation must remain
visible; any mirrored or reordered velocity view needs a defined refresh
contract, with its cost charged to a complete step. The existing generic
numerical behavior and mixed-mesh oracle should govern that redesign. The
present descriptor remap improves one bounded architecture but does not
exhaust general algorithms or prove a memory-bandwidth limit.

## Reproduction

The frozen source's 92 Julia files are byte-identical to Git revision
`ace94c888012d7cea00cf9545046a5ce5794325b`. Its ignored local snapshot
and resolved Julia 1.10.12 project/manifest are under
`benchmark/scaling/results/20260930-stage3-production-baseline-freeze/`;
they must be recreated if absent. A newly resolved dependency environment
is a new cohort unless recorded manifest and hashes match. Selected
advection and embedded-operator SHA-256 hashes are
`19f5490c04dd9947504d9e17c42d31077990a8da191b6f9b86f5c5bbddde0018`
and `ced65e7a313c80bd7a72334e8c72c52513be4007a63e176cb8c3dd2333680bd3`.
The tracked procedure is in `benchmark/scaling/stage3/README.md` and its
drivers. Ignored primary raw data, summary, and plot are under
`benchmark/scaling/results/20260930-stage3-upwind-final-matrix-v2/`;
worker, component, and setup receipts are under sibling directories
`20260930-stage3-worker-probe/`, `20260930-stage3-component-profile/`, and
`20260930-stage3-setup-final/`. The standalone main plot is
`main-absolute-efficiency.svg` in the primary result directory, generated by
`benchmark/scaling/stage3/plot_summary.jl`. The pipeline table and plot are
under ignored `benchmark/scaling/results/cache_pipeline-final-analysis-20260930/`.
The reviewable six-worker-count pipeline table and rerun commands are in
`research/UniformLWAdvection/experiments/cache_pipeline/FINDINGS.md` and
its adjacent `README.md`. The uniform-only main-package LW prototype is in
`research/UniformLWAdvection/experiments/main_cell_lw/uniform_cell_lw.jl`.
These are plain local paths so the documentation build does not depend on
ignored assets; generated data and figures are deliberately not tracked.
