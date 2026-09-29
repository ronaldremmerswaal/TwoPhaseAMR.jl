# Stage 2 uniform Lax–Wendroff optimization report

This report records the numerical and primary five-repeat measurement work for
the standalone `UniformLWAdvection` experiment. The core method and selected
standard Polyester backend pass the covered one-to-32-worker correctness
fixtures. Separate affine experiments have unresolved parity events in a
large-grid p24 amortized run and a p1 observable-step tracked/untracked profile
probe; neither affine API is promoted. **The requested 50%
efficiency at four or eight workers is not demonstrated on the mandatory
2048² and 4096² complete-trajectory cases.** The 4096² result misses with
confidence intervals below the target; at 2048² the best observable affine
four-worker interval crosses the target. Review of the supplementary controls
and component profiles does not establish an explained shortfall. Under the
plan's stopping rule this is an **unaccepted Stage 2 performance outcome**
in its failure category. The core numerical and threading results remain
usable within their stated scope. No Stage 3 implementation follows.

The [eight-stage plan](multithreading-optimization-plan.md) defines self
efficiency as `Eₚ = T₁/(p Tₚ)` for one fixed algorithm and workload. It also
requires a separate comparison with the fastest validated one-worker method
at the same numerical settings. The [execution ledger](multithreading-execution.md)
records the earlier baseline, control, and screening stages.

## Numerical contract and validation

The independent toy package in `research/UniformLWAdvection/`
advances Float64 cell averages on a periodic unit square. It stores one
positive-oriented face speed per physical face and applies conservative
Strang-split Lax–Wendroff substeps `x(Δt/2), y(Δt), x(Δt/2)`. Each sweep reads
an immutable source and owns its destination cells. Both a once-stored face
flux and an output-owned, identically oriented recomputed flux have been
validated; the primary timing fixture keeps dense velocity arrays present
with a constant carrier. Analytic error uses translated **cell averages**.
The independently tested variable-carrier extension and its midpoint velocity
refresh are separate from the constant-carrier performance fixture.

G2.1 passes conservation, periodic seams, constant states, negative/zero
velocities, an independent serial face-flux oracle, and smooth translation
order. The final affine candidate passes 16 fresh Julia 1.10 processes over
worker counts 1/2/3/4/7/8/16/32, with 514/514 checks per process and an
observed finest-pair smooth order of about 1.99. The independent
correctness drivers in `benchmark/correctness/` also compare
full toy fields at those counts on square and odd rectangular grids,
both flux policies, four backends, and a smooth time-dependent shear.
All such full-field worker-count comparisons are exact for the checked
trajectories. This does not establish parity for every large-grid SMT
trajectory. A separate 4096², 29-step, 24-worker affine amortized follow-up
halted with a maximum field difference of `3.125383729596365e-9` against
the serial reference, above the unchanged `512ε` bound. Its machine-local
`benchmark/scaling/results/20260929-stage2-smt-baseline110-pinned-spread/failure.json`
and partial samples are retained; no SMT scaling curve or general affine
24/32-worker correctness claim can be made. A bounded follow-up in
`/tmp/twophase-affine-p24-diagnosis/` found no reproducible first divergent
step: six fresh-process exact retries, 20 repeated untracked trajectories,
20 poisoned-scratch odd-grid trials at each of seven and 24 workers with
29 per-step comparisons, and five poisoned/tracked large p24 trajectories
with per-step checks all agreed exactly. Tile and chunk ownership counts
were one. A further frozen 1031×257 p7 partition diagnostic recorded 20
exact trajectories, zero difference after each of 29 passes, and all seven
workers in the tracked pass and copy-back probes; its machine-local log is
`benchmark/scaling/results/20260929-stage2-smt-baseline110-pinned-spread/partition-1031x257-p7-default.log`.
The adjacent `bounded-diagnostics-receipt.json` preserves commands, source
hashes, exits, and artifact hashes.
The original failure remains unexplained and is **not** erased by
clean reruns; the experimental amortized path is not promoted.
The bounded reproduction and code-generation inspection entry points are
documented in `research/UniformLWAdvection/experiments/README-affine-diagnostics.md`;
the latter has not yet established a cause.
Before any future affine SMT screening, make the separate diagnostic driver
archive the first divergent step, cell index and bounded reference/candidate
neighborhood (or complete fields), affinity, and source hashes automatically
*before* it raises on parity. The original failure receipt saved the maximum
difference and stderr but not an error location, which limited the diagnosis.
Do not relax the numerical tolerance or change the frozen final acquisition.
There is a second unresolved event in the 4096² p1 supplementary affine
profile: `profile_affine.jl:254` rejected the **observable one-step**
tracked/untracked probe before the CPU profile was written. Its
machine-local stderr is
`benchmark/scaling/results/20260929-profile-affine-mandatory-baseline110/n4096-p1.stderr.txt`.
The check combines field and clock equality without saving either delta,
so the record does not establish which one failed. It occurred on the p1
serial `Matrix` path and cannot be assigned to SMT alone. A later single
4096² p1 detailed diagnostic with a corrected wrapper reported exact
tracked/untracked fields and clock values for both observable and amortized
probes, with maximum field delta zero; its log is
`benchmark/scaling/results/20260929-profile-affine-mandatory-baseline110/n4096-p1-detailed-repro2.log`.
Its adjacent receipt records the exact command, frozen sources, and output
hashes.
That clean rerun does not identify or erase the original failure. The three completed
2048² affine component profiles remain diagnostic only; further affine
profiling was stopped pending a bounded numerical reproduction. The core
standard path is separate evidence. The final
one-to-16-worker timing matrix's 280 samples have zero full-field and
untimed worker-probe difference from the serial reference, a maximum mass
error of `7.42e-14` against the frozen `512ε` absolute bound, and all requested
workers active on mandatory four/eight-worker passes.

Independent state/workspace pairs also pass concurrent execution checks at
one, four, and eight workers: Polyester recomputed/stored paths pass 18/18
checks per count; affine observable/amortized paths pass 24/24 per count.
The raw receipt and test hashes are in
`/tmp/twophase-concurrency-posttiming-20260929/results.txt`. A workspace
remains exclusive to one in-flight operation. Affine scratch belongs to stable
chunk IDs rather than Julia thread IDs, so task migration does not share it.
The observable fused step retains `state.q` identity through a second
compute-to-copy-back barrier. The amortized experiment exposes its output only
after a complete trajectory, copying back once for odd step counts and
skipping that final copy for even counts; it is not an individually observable
`step!` substitute.

## Reproducible performance matrix

The machine-local artifact
`benchmark/scaling/results/20260929-stage2-final-baseline110-pinned-spread/`
contains `raw.jsonl`, `report-corrected.json`, `report-corrected.md`,
`configuration.json`, and its source/dependency snapshot. These records use
Julia 1.10.12, the resolved dependency snapshot, the frozen sources,
an AMD Ryzen 9 5950X, per-worker physical-core pinning spread over both L3
domains, one BLAS thread, and one GC thread. The matrix contains 56
configurations, each run in five independent fresh processes. Every
configuration uses the same frozen number of complete steps across worker
counts at a given grid size; the workload file records 45,301 / 26,056 /
9,816 / 2,198 / 155 / 29 steps at sizes 128 / 256 / 512 / 1024 / 2048 /
4096. All timed trajectories last at least 0.368 seconds, and the six source,
runner, configuration, and workload hashes match across samples. Each sample
retains an untimed probe of actual worker IDs inside the numerical passes,
separate from the timed work. GC remains enabled in all acceptance timings;
recorded GC time is zero for this matrix. Where throughput is discussed below,
one cell-step means one **complete Strang step** for one cell, not one of its
three directional substeps. The raw-results directory is intentionally ignored
by Git. The curated [figure set](assets/stage2-final-4096.svg) and
[aggregate summary](assets/stage2-primary-summary.json) are portable copies
of local workstation results, not portable CI thresholds. The summary records
SHA-256 digests of its frozen raw/report sources and the generation command.

Earlier Julia 1.12.7 fresh-process work intermittently crashed during
compilation/GC, including a control kernel; successful reruns did not establish
the cause. The final matrix therefore uses a separately resolved, frozen
Julia 1.10.12 environment for both serial and threaded paths. Timings from
the two runtimes are never mixed. The 1.12 behavior remains an environment
investigation, not a diagnosed Julia or package defect.

| Grid and complete-trajectory method | `T₁ / T₄ / T₈`, s | Fixed-method `E₄` [95% CI] | Fixed-method `E₈` [95% CI] |
|:--|:--|:--|:--|
| 512² standard observable | 3.9933 / 1.0925 / 0.6313 | **0.914** [0.892, 0.959] | **0.791** [0.773, 0.817] |
| 2048² standard observable | 2.1076 / 1.2262 / 1.2677 | 0.430 [0.419, 0.452] | 0.208 [0.203, 0.218] |
| 2048² affine observable `(512,8)` | 2.2870 / 1.2521 / 0.7878 | 0.457 [0.423, 0.545] | 0.363 [0.339, 0.405] |
| 2048² affine final-state trajectory `(512,8)` | 1.5984 / 0.8948 / 0.5448 | 0.447 [0.419, 0.450] | 0.367 [0.352, 0.372] |
| 4096² standard observable | 1.5518 / 1.1732 / 1.2173 | 0.331 [0.325, 0.337] | 0.159 [0.157, 0.162] |
| 4096² affine observable `(512,8)` | 1.5063 / 0.9443 / 0.8094 | 0.399 [0.389, 0.421] | 0.233 [0.227, 0.242] |
| 4096² affine final-state trajectory `(512,8)` | 1.1809 / 0.7073 / 0.5090 | 0.417 [0.411, 0.426] | 0.290 [0.260, 0.297] |

The exported [512²](assets/stage2-final-512.svg),
[2048²](assets/stage2-final-2048.svg), and
[4096²](assets/stage2-final-4096.svg) figures show wall time,
fixed-configuration efficiency, and speedup against the compatible strongest
serial method. The [working-set throughput](assets/stage2-final-throughput-working-set.svg)
and [storage/allocation](assets/stage2-final-memory-allocation.svg) figures
use complete cell-steps. All five SVGs are snapshots from 29 September 2026
on the pinned-spread Ryzen 9 5950X / Julia 1.10.12 configuration, with the
exact source and generation hashes in the adjacent
[curated JSON](assets/stage2-primary-summary.json).

The cache-sized 512² standard method passes at both four and eight workers.
The standard 1024² curve also passes, with `E₄=1.205` and `E₈=1.037`;
cache placement and superlinear effects make those numbers unsuitable for
extrapolating to larger arrays. At 2048² no candidate has a demonstrated
50% pass: the affine observable four-worker interval straddles 0.50 and
every other primary point has its interval below 0.50. At 4096² every
candidate's four/eight-worker upper interval is below 0.50. More repetitions
could resolve the inconclusive 2048² point, but cannot make the already
measured 4096² result a pass. The original target therefore remains unmet.

The strongest compatible one-worker reference differs by API scope. For an
**observable step**, the validated serial halo path is fastest at 2048² and
4096²: 1.5901 and 1.3009 seconds. Affine observable `(512,8)` costs 43.8%
and 15.8% more at one worker. The standard method also costs 32.5% and 19.3%
more than that strongest compatible halo reference at one worker, although
it matched the original unfused serial core in the earlier screen. Thus its
simplicity recommendation does not waive the plan's 5% strongest-serial guard.
Even affine observable's 2048² four-worker run is slower than the standard
four-worker run. For a **final-state-only trajectory**, observable
methods are eligible as serial references too, but the fastest measured
reference is amortized `(1024,32)` at 1.4300 and 1.0512 seconds. Amortized
`(512,8)` costs 11.8% and 12.3% more at one worker. The reverse inclusion is
invalid: an amortized-only path cannot establish the serial cost of
individually observable steps. The plan's initial one-worker regression
allowance is 5% beyond noise; these costs require a concrete accepted tradeoff
before an affine policy could replace the simpler path. Candidate self
efficiency does not erase the serial cost or establish an absolute gain.

The warmed timing records support a bounded **numerical-array** storage
model, distinct from process memory and traffic. Standard recompute stores
`q,u,v` plus two intermediate arrays: `40N²` bytes, or 160 MiB at 2048²
and 640 MiB at 4096². The affine fused workspace holds `q,u,v,next`
(`32N²` bytes) plus one approximately 76 KiB scratch set per stable chunk;
at four workers and `(512,8)` this reports 129.19 / 513.19 MiB at
2048² / 4096², and at eight workers 130.39 / 514.39 MiB. The benchmark's
serial reference, reset copy, Julia runtime, and compilation are included
in peak process RSS, so RSS is higher: at 4096² it is 1.53 GiB for standard
four-worker, 1.46 GiB for affine observable four-worker, and 1.52 GiB for
affine amortized eight-worker samples. Median warmed allocations are about
192 bytes/step for standard four-worker at 512² and 193 at 4096²; affine
observable four-worker is about 241 bytes/step at 4096², and amortized
eight-worker about 148. These fixed-size scheduler allocations do not grow
with `N²` in the measured cases. They do not measure physical DRAM traffic
or prove bounded memory over arbitrarily long simulations.

The standard method has a separate, source-matched
**SMT diagnostic** at 24 and 32 requested workers: five fresh processes per
point at 1024², 2048², and 4096², with all requested workers active in each
numerical pass and exact serial parity. Its machine-local summary is
`benchmark/scaling/results/20260929-stage2-smt-standard-baseline110-pinned-spread/summary.json`.
The one- and 16-worker denominators come from the earlier final matrix with
the same source/workload, so these are comparisons across acquisition windows.

| Standard grid | `T₁₆ / T₂₄ / T₃₂`, s | `S₃₂` | `E₃₂=S₃₂/32` | `T₁₆/T₃₂` | `S₃₂/16`, physical-core normalized |
|:--|:--|--:|--:|--:|--:|
| 1024² | 0.3785 / 0.4695 / 0.4485 | 9.76 | 0.305 | 0.844 | 0.610 |
| 2048² | 1.3144 / 1.3700 / 1.4602 | 1.44 | 0.045 | 0.900 | 0.090 |
| 4096² | 1.2852 / 1.3499 / 1.3562 | 1.14 | 0.036 | 0.948 | 0.072 |

The 1024² p32 timing has one large outlier and a wide speedup interval;
neither 24 nor 32 threads gives a repeatable improvement over 16 on these
standard fixtures. `S₃₂/16` describes speedup normalized by the 16 physical
cores and is **not** `E₃₂`. The partial affine SMT attempt is excluded after
its unresolved p24 numerical failure and is not merged with this standard
curve.

Placement is material even with all requested workers active. A separate
30-sample randomized 1024² standard experiment, five fresh processes for
each four/eight-worker placement, is recorded at
`benchmark/scaling/results/20260929-stage2-placement-baseline110/summary.json`.
Within its own `driver.jl` harness, spread-pinned median times are 1.109 /
0.612 seconds at p4/p8, compared with compact-pinned 1.925 / 1.627 and
ordinary-unpinned 1.962 / 1.446. Spread is 1.74×/2.66× faster than compact
and 1.77×/2.36× faster than ordinary at p4/p8. The final scaling matrix uses
the same spread placement policy but a different `screen_layouts.jl` caller;
its absolute 1024² times cannot be mixed with this placement experiment.
Pinning is a benchmark configuration, not a library side effect or an
explanation of all large-grid shortfall.

## Bounded alternatives and usable configuration

The early two-repeat [backend/flux screen](multithreading-execution.md#bounded-toy-policy-screen)
compared optimized serial loops, Julia dynamic tile scheduling, chunked
`@spawn`, and Polyester with stored and recomputed fluxes. Polyester plus
recomputed fluxes and contiguous `(1024,32)` columns led the 1024² screen
without a measurable one-worker loss against the serial core. Stored fluxes
add a face array and six pass launches per Strang step, compared with three
output-owned recompute sweeps; they lost on this complete-step fixture. The
ordinary Julia task backends lost the first large comparison and were not
carried into the expensive final matrix. Persistent tasks were not justified
by the measured launch cost. Padded halos improved 4096² **serial** time by
about 15% in the early two-repeat screen but did not improve its parallel
time there; the final matrix retains halo as a serial reference. The original
fused layout had large serial regressions. Affine indexing plus typed batch
dispatch substantially improved it, but that joint change cannot be
attributed to SIMD alone. Amortizing copy-back changes the observation
contract and is a distinct experiment. A later `(512,8)` tile screen found
faster large-grid parallel trajectories, at the serial costs quantified
above; no worker-dependent tile switch is hidden in a reported curve.

For users who want the validated toy parallel backend now, the conservative
choice is to **explicitly** load optional Polyester and construct:

```julia
using UniformLWAdvection, Polyester

s = State(512, 512)
initialize!(s)
w = Workspace(s; backend=:polyester, flux=:recompute,
              tile=(1024, 32), threshold=16384)
```

This keeps the public toy's mathematical method and individually observable
step contract. The toy's standard-library-only/default API remains unchanged;
the optional backend is a caller choice. It is a sensible cache-workload
policy and a simple large-grid control, **not** a claim that its 2048²/4096²
scaling meets the target. The affine kernels remain separate experiments;
their observable and final-state scopes, serial penalties, and memory costs
must accompany any reported large-grid absolute times. The unexplained p24
amortized and p1 tracked/untracked probe events block promotion of the affine
experiments at any count until diagnosed.
The final matrix
shows no single cross-size/cross-worker affine promotion that satisfies the
serial guard and the 50% large-grid target.

For this **standard observable** configuration, the completed physical-core
matrix and standard-only SMT follow-up give these size-specific operating
points. Each grid has a different frozen step count, so compare times only
within a row; the table is not a weak-scaling curve.

| Grid | Faster of p4/p8, time | Fastest measured standard count, time | Practical recommendation |
|:--|:--|:--|:--|
| 128² | p8, 0.4782 s | p8, 0.4782 s | p4 is within about 1.4% and uses all four of the tile-limited workers; p8 requests eight but only four compute. |
| 256² | p8, 0.4982 s | p8, 0.4982 s | p16 is effectively tied but only eight tiles participate; use p8. |
| 512² | p8, 0.6313 s | p16, 0.4684 s | Use p16 if minimum latency matters, or p8 when reserving cores. |
| 1024² | p8, 0.5274 s | p16, 0.3785 s | Use p16 for minimum latency; p24/p32 regress. |
| 2048² | p4, 1.2262 s | p4, 1.2262 s | Prefer p4; adding workers does not improve this standard path. |
| 4096² | p4, 1.1732 s | p4, 1.1732 s | Prefer p4; p8/p16/SMT are slower. |

These are measured count recommendations for the fixed standard method, not
automatic library dispatch rules. The experimental affine observable path is
absolutely faster at p8 than standard on 2048²/4096², and at p16 on 2048²,
but it keeps the stated serial regression and fails the large-grid efficiency
gate. The final-state-only amortized path is faster still at some counts but
has a different API and an unresolved p24 parity event. The p1 profile-probe
failure adds a separate unresolved affine contract question. Neither experiment
is selected as the package default or represented by this standard table.

The machine-local 64² tiny-work fallback artifact at
`benchmark/scaling/results/20260929-tiny-fallback-baseline110-pinned-spread/summary.json`
uses 98,304 identical steps in five fresh processes at one/four/eight
requested workers. The median times are 0.70576 / 0.70795 / 0.70226 seconds;
the numerical loop uses one worker in every process because the grid is below
the 16,384-cell threshold. The four/eight-worker overhead relative to the
one-worker process is +0.31% / −0.50%, within the declared 5% tiny-work
regression budget. Independent speedup intervals `[0.98383, 1.00635]` and
`[0.97410, 1.01093]` bound the corresponding worst overhead at about 1.64%
and 2.66%, also below 5%. These timings are fallback evidence, not parallel
speedup, so the generic requested-worker efficiency label does not apply.
The long trajectory's reported analytic error is not a convergence study;
the separate numerical fixtures establish accuracy.

## Limits of the performance explanation

The final matrix shows that worker participation is not the missing factor at
the mandatory four/eight-worker points. It does not itself measure DRAM
transactions, worker busy fractions, or the cost of compute, copy-back, and
barriers separately. The affine amortized two-worker path is slower than its
one-worker path by 8.6% at 2048² and 11.7% at 4096², despite actual worker
participation. Source inspection finds a serial `Matrix` loop at one worker
and a Polyester `@batch` path at two or more workers. That batch captures
`q0/out` as `StrideArraysCore.PtrArray`, while velocity and chunk scratch stay
as `Matrix`; the chunk count also grows with worker count. Different pointer
specialization, SIMD/code generation, or scheduling overhead are hypotheses,
not measured causes. The archived Matrix/PtrArray typed and LLVM snapshots
at `benchmark/scaling/results/20260929-stage2-smt-baseline110-pinned-spread/codegen/`
preserve the specializations for later analysis. Both serial `Matrix` and
Polyester `PtrArray` compute-loop IR contain vector arithmetic; the `Matrix`
copy-back IR has vector loads/stores while the inspected `PtrArray` copy-back
IR has no textual vector load/store occurrences. These static differences
do not quantify a timing effect or establish a fault. No change to the
`@simd` loop source itself was found.
At 2048² the arrays exceed one L3 cache, yet tile and pass reuse may retain
some data in cache. A 4096² cold-DRAM traffic estimate therefore cannot
explain its result on its own.

A separate pinned 4096² STREAM control at
`benchmark/scaling/results/20260929-stream4096-baseline110-pinned-spread/summary.json`
uses five fresh processes and 24 complete sweeps per sample. Median times
are 0.3852 / 0.3147 / 0.3250 seconds at one/four/eight workers; all requested
workers participate. Its measured *logical* throughput is about 30.7 GB/s
at four workers for 24 bytes per cell update. If each cold store causes an
additional 8-byte write allocation, the corresponding **estimated** physical
traffic ceiling is 40.9 GB/s. Under similarly conditional cold-store models,
the 4096² standard method's 96 bytes per cell-step imply 39.8 GB/s at four
workers, about 97% of that STREAM estimate. The amortized affine method's
40 bytes per cell-step plus one 24-byte copy-back per cell for the 29-step
odd trajectory imply 39.0 GB/s at eight workers, about 98% of the **matching
eight-worker** STREAM estimate of 39.6 GB/s. Its four-worker
model gives only 28.1 GB/s, about 69%, so this bandwidth comparison alone
does **not** explain every four-worker deficit. Halo overlap, cache reuse,
compiler-generated stores, and actual DRAM transactions are unmeasured; these
ratios are sensitivity estimates, not hardware-counter evidence or a measured
kernel ceiling. The separate pinned 4096² five-point stencil control takes
0.5044 / 0.2738 / 0.2838 seconds at one/four/eight workers over 32 sweeps,
for speedups about 1.84 and 1.78; its 48-byte *logical* model reaches about
94 GB/s at four workers, well above STREAM's logical rate. This illustrates
cache reuse and why logical stencil bytes cannot be equated with DRAM
transactions. The 2048² cache transition still lacks a measured traffic or
latency ceiling for the actual transport kernel.

The completed **standard observable component profiles** cover all six
mandatory grid/count combinations at one, four, and eight workers. They run
the same 155/29-step workloads at 2048²/4096², with three repeated full
trajectories and separately timed `x-half`, `y-full`, and `x-half` passes.
The pass values below are medians in seconds; their sum can differ from the
median full run because the passes and full trajectories are distinct timing
observations. The saved records have zero final-field difference and mass
error, and their untimed probes show all requested compute workers active.
These are supplementary measurements, not additional acceptance samples.

| Standard grid | Workers | Full trajectory, s | First x / y / second x, s | Pass share of their median sum |
|:--|--:|--:|:--|:--|
| 2048² | 1 | 1.988 | 0.603 / 0.717 / 0.670 | 30% / 36% / 34% |
| 2048² | 4 | 1.299 | 0.373 / 0.456 / 0.454 | 29% / 36% / 35% |
| 2048² | 8 | 1.215 | 0.377 / 0.425 / 0.407 | 31% / 35% / 34% |
| 4096² | 1 | 1.618 | 0.555 / 0.475 / 0.583 | 34% / 29% / 36% |
| 4096² | 4 | 1.166 | 0.388 / 0.388 / 0.389 | 33% / 33% / 33% |
| 4096² | 8 | 1.210 | 0.406 / 0.406 / 0.404 | 33% / 33% / 33% |

The artifact is
`benchmark/scaling/results/20260929-profile-standard-mandatory-baseline110/`.
Its configuration specifies **process affinity masks only**; unlike the final
matrix, it does not pin or verify each worker's CPU. It also contains wall-clock
pass timing rather than Julia `Profile` stacks. Its 2048² p8 full time is
slightly below p4, unlike the final matrix; that difference is a reason to
keep acquisition windows and placement scopes separate, not evidence that
the acceptance result changed. At 4096² all three passes take nearly equal
time at four and eight workers; no single directional pass dominates the
large-grid runtime. These timings do not separate arithmetic, memory stalls,
and launch/barrier time within a pass.

The completed **2048² affine component profiles** at one, four, and eight
workers are in
`benchmark/scaling/results/20260929-profile-affine-mandatory-baseline110/`.
They use individually pinned workers, three full-trajectory repeats, an
instrumented compute/copy-back split, and Julia sampling profiles. For the
observable path, median split-clock compute/copy-back times are
`1.893/0.428`, `1.065/0.208`, and `0.622/0.192` seconds at p1/p4/p8.
Copy-back therefore occupies about 18%, 16%, and 24% of the instrumented
trajectory. The final-state-only amortized path copies back once on this odd
155-step workload: its corresponding compute/copy-back times are
`1.594/0.003`, `0.892/0.001`, and `0.533/0.001` seconds. These split-clock
values are instrumented diagnostics; the five-repeat matrix remains the
performance decision. All three completed profile points match their
serial final fields, but the later 4096² p1 observable tracked/untracked
probe failed before its profile completed. A later one-point p1 diagnostic
rerun succeeded, but no complete 4096² affine component curve exists, and
the 2048² profiles cannot justify promoting either affine API.

The original 50% goal can only be replaced for Stage 2 by an explicitly
recorded **explained shortfall** after measured bandwidth/latency controls or
a quantified serial dependency support a credible bound for each mandatory
large size. The pinned STREAM and stencil/gather controls, standard component
profiles, completed 2048² affine profiles, scratch/working-set and allocation
evidence, and standard 24/32-worker SMT diagnostics have now been reviewed.
The affine 4096² component and SMT curves remain incomplete after parity
events. Traffic inferred from array counts or STREAM alone is an estimate,
not a measurement of this kernel's physical DRAM bytes. The 2048² cache
transition and the 4096² four-worker result lack a quantified ceiling that
explains their miss. Therefore G2.3 fails its original target and its
alternative explained-shortfall route. G2.4 does not support an affine
replacement policy. The Stage 2 performance outcome is **unaccepted** under
the plan's stopping rule; this report closes the current investigation
without changing the target or proposing a Stage 3 transfer.

| Stage 2 gate | Reported outcome |
|:--|:--|
| G2.1 serial numerics | Passed for the core periodic LW method, including independent flux oracle, conservation, seams, and approximately 1.99 smooth order for the final affine fixture. |
| G2.2 thread correctness | Core toy/selected standard Polyester backend passed covered worker-count and concurrent-workspace checks. Affine experimental promotion is blocked by the unresolved p24 amortized field parity and p1 observable tracked/untracked field-or-clock event. |
| G2.3 complete-step scaling | **Not met** under the original 50% at p4/p8 target: 2048² has no confirmed pass, and 4096² has clear misses. Supplementary controls and profiles do not establish an explained shortfall for both mandatory sizes. |
| G2.4 simplicity/cost | Bounded warmed allocation and tiny-work fallback pass for the standard path. Against the fastest compatible serial halo reference, standard costs 32.5%/19.3% more at 2048²/4096²; affine `(512,8)` also exceeds the 5% guard. No exception or affine replacement policy is accepted. |

The supported recommendation is the explicit standard Polyester backend and
size-specific worker counts above. Do not promote either affine experiment
or turn its final-state-only trajectory into an observable-step API. Preserve
both affine failure receipts and their frozen sources. If the scaling goal is
revisited, first separate the p1 field/clock probe result and capture the
first divergent p24 cell and step; then use matched per-worker placement and
hardware counters or an equivalent measured traffic ceiling to test the
2048² cache transition and 4096² four-worker shortfall. Further scheduler
variants without that evidence would not resolve the current gate.

## Boundary to later stages

[WaterLily v1.8.0's pinned fixed-grid source](multithreading-waterlily-reference.md)
supports output-owned Cartesian loops, explicit serial baselines, and a
stored-flux control. Its limited QUICK flux, Float32 published CPU figures,
fixed Cartesian metadata, and absence of AMR/VOF/CMDGFM cannot substitute
for this package's conservative face ownership or accuracy gates. If work
later proceeds to Stage 3, keep three numerical scopes in order:

1. Profile the **unchanged upwind** `CellField` path first. Preflight memory
   and setup cost, then freeze equal-global-cell one-large-block and many
   8×8-block fixtures. Time owned-face flux, cell gather, prescribed velocity
   sampling, CFL, and complete transport at one/four/eight workers before
   changing a schedule or storage choice.
2. Add a **separately validated uniform periodic centered LW** scalar method
   matching the toy's cell averages, face orientation, carrier, and Strang
   arithmetic. Start with generic stored oriented face flux and gather, with
   six synchronized passes per step and its own validated serial baseline.
   Prove conservation, seams, smooth order, state observability, and lifecycle
   guards before testing any recomputed or dense specialization. The toy's
   final-state-only fused contract does not transfer into a public step API.
3. Profile the **existing closed-wall WY reversal** with LHF reconstruction,
   nonconstant liquid/gas phase scalars, and constant-one tracers over the
   full reversal interval. Include CFL, prescribed-carrier refresh, geometry,
   reconstruction, and diagnostics. Preserve frozen scalar source snapshots
   and phase amount/bound budgets before changing storage or scheduling.

All three scopes need their own strongest compatible serial comparison and
one-block/many-block report; none inherits a Stage 2 target pass. No Stage 3
production changes are part of the Stage 2 result.

The main package's starting baseline has a distinct projected-gas convergence
near miss and mimetic pressure-fit failure recorded in the
[execution ledger](multithreading-execution.md#baseline-findings-requiring-attention).
They are not toy-package G2 failures; their original numerical thresholds
must remain in force before the corresponding later-stage transport/solver
changes are accepted.
