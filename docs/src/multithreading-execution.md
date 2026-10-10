# Shared-memory optimization execution record

The [plan](multithreading-optimization-plan.md) is being executed from
`f604c46f720dd77f8f51345d27782329c7a27eb9` on 29 September 2026. Its original
research revision is retained in its source links. The requested primary goal
is **50% parallel efficiency at four or eight active compute workers**, with
both results reported. Sixteen and SMT worker counts are diagnostic/stretch
experiments. Numerical tolerances and fixed-workload comparisons are unchanged.
On 30 September 2026 the user explicitly authorized proceeding to Stage 3
while retaining the recorded Stage 2 performance miss.

The [detailed Stage 2 report](multithreading-stage2-report.md) records its
completed numerical checks, five-repeat scaling matrix, serial costs,
supplementary diagnostics, and unaccepted performance decision.

A separate [Julia–Fortran/OpenMP toy comparison](multithreading-julia-fortran-comparison.md)
uses matched numerical work, tile geometry, and worker counts to investigate
the large-grid scaling question. It is a follow-up diagnostic and does not
change the Stage 2 gate. The subsequent
[cache-cliff investigation](multithreading-cache-cliff-investigation.md)
identifies a cache-reuse/traffic transition without establishing a measured
bandwidth ceiling.

The authorized Stage 3 scope initially covers uniform **cell-centred scalar
transport without an interface**: the existing upwind control and a separately
validated research LW path matched to the Julia toy, on one large block and
many uniform blocks. The user's subsequent correction requires production
optimization of the **general** advection path; uniform-only LW prototypes
belong in `research/` and are not a production solution. First compare the
general upwind operator with its frozen baseline, using the LW toy only as a
clearly identified different-method cost reference. Interface/WY work remains
deferred. New comparisons are Julia-only at
the user's request; earlier Fortran evidence remains archival. A separate
Julia experiment tests whether a halo-aware full-step tile pipeline can retain
intermediate data in cache. Advancing this work does not retroactively pass
Stage 2 or the deferred Stage 3 interface gates.

Research source scripts and Markdown reports remain reviewable in Git.
Generated measurements, configuration snapshots, binary fixtures, validation
receipts, and plots remain local ignored artifacts, including generated
research assets under `docs/src/assets`. Reports retain their numerical tables
and describe generated artifacts by local paths; the associated runners
reproduce them. This
storage policy follows the user's clarification on 30 September 2026.

The [Stage 3 transport investigation](multithreading-stage3-report.md)
records the general upwind workspace changes and the independent LW pipeline
experiment. Production retains one general advection method: its cached face
access and private flux ordering also support mixed levels and physical
boundaries. The uniform-only LW implementation is a research artifact, not an
exported production operator. General-upwind comparisons use their own frozen
upwind reference; LW timings remain a different-method absolute cost reference.
The [10 October block-interior follow-up](multithreading-stage3-block-interior-report.md)
compares a later general implementation with that previous remap and records
its gate and storage evidence. The subsequent
[checked narrow face-ID study](multithreading-stage3-narrow-ids-report.md)
compares against the committed block-interior implementation, halves the
regular-ID payload, and reports its own 90 matched fields and efficiency
intervals. It does not close dynamic-CFL or general LW work.
The [large-mesh construction investigation](multithreading-stage3-mesh-crash-report.md)
records intermittent failures before advection in a separate copied-source
ordered-gather experiment. Its diagnostic checkpoints remain inconclusive
about cause. A later [ordered-gather follow-up](multithreading-stage3-ordered-gather-report.md)
uses full mesh validation before each timed child and reports matched 1024²
one-/many-block results. The many-block eight-worker regression led to
rejection of the helper for production. The 2048² control failed validation
before timing. Passing smaller children do not resolve the intermittent setup
failure. The completed narrow-ID cohort remains a separate comparison.
The subsequent [all-order scalar-selector study](multithreading-stage3-scalar-selector-report.md)
passed 14,604 copied-source numerical assertions per variant and 36/36
bitwise timed fields across the two 1024² layouts. Its one-block serial and
four-worker regressions and unstable eight-worker results prevent promotion.
The subsequent [two-pass rolling-flux study](multithreading-stage3-rolling-flux-report.md)
passed exact mixed-AMR field/increment and face-ownership checks, but its
audited three-pair 1024² one-block and many-block fixed-`dt` pilots were
slower at one, four, and eight workers. It was rejected for production. A
separate row-fused copied candidate passed focused numerical and face-call
checks, but its first many-block cohort stopped during workspace construction
before a candidate field or timed step. It has zero accepted matched pairs
and unknown throughput.

## Decisions before experiments

- Start with the installed Julia 1.12.7 binary directly. The `juliaup` launcher tries
  to create a lock outside the writable workspace. This is an environment
  restriction, not a package failure.
- Keep BLAS and GC at one worker during the initial comparisons. Discover
  affinity and physical-core/L3 topology rather than treating 32 logical CPUs
  as 32 physical cores.
- Preserve a source snapshot of the implementation baseline before editing
  numerical code. Keep raw results outside version control; commit runners,
  configurations, and this decision record.
- Freeze toy cache/DRAM fixtures before backend comparisons; treat large
  tree meshes separately because connectivity storage dominates a dense toy's
  storage model.
- Use [WaterLily's pinned source review](multithreading-waterlily-reference.md)
  to inform dense loop, ownership, launch, and multigrid experiments. Its
  Cartesian discretization, Float32 published results, and MPI scaling do not
  substitute for this package's AMR/VOF/CMDGFM validation.

## Gate ledger

| Stage | Evidence and status | Next dependent work |
|:--|:--|:--|
| 1. Measurement | Baseline source captured; complete unit baseline recorded with two pre-existing failures. G1.1 research available. Five-repeat Julia 1.10 control curves and verified worker-pinning comparison recorded. Measurement environment ready for the toy screen. | Preserve pinned placement and frozen workloads through backend comparisons; capture package baselines before each change. |
| 2. Uniform LW toy | G2.1 core numerical tests pass on Julia 1.10/1.12. Core G2.2 fresh-process states agree exactly at 1/2/3/4/7/8/16/32 workers on their covered fixtures; the affine experiment passed 16 earlier processes, 514/514 checks each, with approximately 1.99 smooth order. Later 4096² amortized p24 field parity and p1 tracked/untracked profile checks failed without an established cause, so affine experiments are not promoted. The five-repeat matrix misses G2.3's 50% target on 4096², and the 2048² affine four-worker interval crosses the gate. Reviewed controls and completed component profiles do not establish an explained shortfall. Standard allocation/tiny fallback pass, but standard and affine one-worker times exceed the 5% guard against the strongest compatible serial reference; no exception or affine replacement policy is accepted. Stage 2 performance outcome: **unaccepted**. | Preserve the standard optional backend and failure receipts. Investigate affine parity and obtain a measured large-grid ceiling before revisiting this gate. Stage 3 proceeds under the user's separate authorization, not a Stage 2 pass. |
| 3. Uniform main transport | The earlier general-remap and research LW pipeline studies remain historical. The arithmetic block-interior general-upwind implementation passed 90/90 matched fields against the previous remap. A subsequent checked narrow-ID workspace passed another 90/90 bitwise fields against the **committed block-interior control** and halved the regular-ID payload. In the narrow-ID cohort, p4 strongest-serial efficiency supports 50% on all three declared layouts; p8 supports it at 1024² one block, misses at 2048² one block, and is inconclusive for 1024² in 8×8-cell blocks. Same-worker p4/p8 timing intervals all cross parity. A copied ordered-gather helper passed numerical checks and a five-pair 1024² one-block fixed-`dt` study (p8 1.585→1.425 ns/cell), but three many-block pairs showed a p8 regression (3.151→4.467 ns/cell). An all-order scalar selector passed copied-source numerical checks and 36/36 timed bitwise fields, but regressed in the one-block serial and four-worker pairs and had unstable eight-worker results. The two-pass rolling-flux candidate passed exact numerical and face-ownership checks but regressed at p1/p4/p8 on both audited 1024² layouts. All three remain research-only. A row-fused copy passed focused numerical and face-call checks, but its many-block pilot failed during workspace construction before any candidate field or matched timing; its throughput is unknown. The 2048² control oracle failed mesh validation before timing. Fixed-carrier CFL remains outside the accepted timers, and the general/toy absolute cost gap remains. Full Stage 3 is **not accepted**. | Reestablish dependable ordinary workspace/setup completion under checked general-AMR fixtures before another throughput comparison. Do not infer a cause from this setup failure. Then require five independent pairs, CFL-inclusive complete steps, mixed-AMR costs, and integration validation before any promotion. Uniform-only LW stays in research. Production general LW numerical work and WY plus phase scalars remain deferred; no Stage 4 promotion follows. |
| 4. Frozen AMR | Pending Stage 3 gates. | Ownership, clustered/dispersed work, fixed-trajectory accuracy and scaling. |
| 5. Dynamic AMR | Pending Stage 4 gates. | Replay/live equality, lifecycle and complete regrid cost. |
| 6. Staggered/two-velocity transport | Pending preceding transport gates. | Constant-one compatibility, absent-phase handling, complete-policy time. |
| 7. GMG/CMDGFM | Pending solver prerequisites. | Actual hierarchy sizes, complete solves and coefficient refresh. |
| 8. Full flows | Pending preceding numerical/performance gates. | Public simulation path, mandatory flow families, production diagnostics. |

The original copied-source ordered-gather pilot passed compact numerical
validation, but two fresh control launches exited with a mesh-construction
SIGSEGV before any paired timing row was accepted. Its failure receipts are
preserved in the [narrow-ID report](multithreading-stage3-narrow-ids-report.md).
The [mesh-construction investigation](multithreading-stage3-mesh-crash-report.md)
locates invalid face fields in two later diagnostic processes but has not
identified a cause. Under the user's renewed request for ordinary numerical
optimization, the [follow-up](multithreading-stage3-ordered-gather-report.md)
checks the full mesh before each timed child and stops the case cohort on an
invalid or incomplete receipt. The 1024² one-block and many-block cases pass
full-field comparison
and have paired fixed-`dt` results; the latter shows a persistent eight-worker
regression, so the candidate was rejected for production. These results are
separate from the failed pilot and do not resolve the construction failure or
complete the Stage 3 gate. The follow-up 2048² one-block control oracle failed
its full face scan before timing, with an invalid axis at face 6,163,788; it
supplies zero accepted timing rows. The restored production source passed
12,522 focused assertions at each of one, four, and eight workers. A 64²
two-step per-step-CFL driver smoke matched control and candidate full fields
at one worker but supplies no accepted CFL-inclusive performance curve.

No pending gate is claimed as passed. Performance conclusions will identify
fixture, source revision, exact run configuration, raw artifact, worker count,
median and uncertainty, numerical error, allocation cost, and remaining work.

## Baseline findings requiring attention

The complete starting-revision one-worker unit suite on Julia 1.12.7 finished
with **20,715 passes, two failures, and one skipped optional-extension check**.
The failures are listed below; the suite is not reported as green.

The unmodified eight-worker integration suite exposed a phase-face lifecycle
error when adaptation changes a uniform dual schedule into an AMR schedule.
The original `PhaseFaceScalar` storage fixes the workspace's concrete schedule
type at construction; `_finish_phase_face_transfer!` then cannot assign the
new workspace. This blocks the adaptive vortex, independently of optimization.
A separate correctness repair and focused refine/coarsen regression are needed.
The repair uses a bounded union of uniform/AMR workspace types and dispatches
the transport kernel once on the actual workspace. Its focused lifecycle
regression passes 193 checks at one and four workers; adaptive vortex runs at
resolutions 8 and 16 pass their existing checks after five/six regrids.
The strengthened regression subsequently passes 343 checks at one and four
workers on Julia 1.10.12. It adds nonzero divergence-free transport and phase
amount checks after every topology transition, including an edited mixed mesh
with unchanged leaf count. The retained fifth type parameter describes the
construction-time workspace type; callers must inspect the current `workspace`
to determine its current schedule after adaptation.

The same starting-revision suite reports projected-carrier mixed-mesh gas
amount convergence of `0.8987596192271534`, below its existing `0.9` gate.
The errors at resolutions 8/16/32 are respectively
`0.0003795250407507833`, `0.00020861232361135684`, and
`0.00010917752051775587`. This threshold is retained; a near miss is still
a failed numerical gate. It cannot be attributed to new threading changes.
An independent one-worker rerun reproduces those three errors and the order
exactly, so this particular miss is not caused by worker-count variation.
This is the endpoint order of the absolute drift in `integrate(q, :gas)` for
the centered `PhaseScalar`, over a fourfold resolution increase; it is not a
staggered-gas field error or a projection residual. The two adjacent orders
are `0.8633706901494749` and `0.934148548304832`. The failing carrier is
`ProjectedLiquidVelocity`, which projects the liquid flux and transports an
auxiliary gas carrier; the frozen-flux policy has a different conservation
check. A bounded follow-up will record, after each step of the same 8/16/32
mixed-mesh runs, the signed gas-amount drift, auxiliary-carrier reset
discrepancy, and phase-flux partition defect. Their relationship remains a
diagnostic hypothesis, since the intrinsic gas scalar varies in space.

The starting one-worker unit suite additionally reports a mimetic projection
variant's pressure-commuting residual of `0.9` against `< 0.2`, and a
nonfinite cylinder-example pressure-drag diagnostic. These are recorded
baseline failures, not waived tests or evidence of a new regression. The
CMDGFM pressure strategy remains the solver scope of the optimization plan.

A subsequent read-only trace finds that `examples/cylinder_flow.jl` deliberately
maps pressure drag above 100 to `NaN`; its finite-history test rejects this
sentinel. A Julia 1.10 diagnostic confirms that the offending first positive-time
sample has finite raw drag `1.3827241297174357e6`, all 448 pressure cells finite,
and a converged projection. The coarse impulsive-start fixture is not a physical
drag reference. Removing only the sentinel preserves the actual force history;
the existing example test then passes 5/5. The diagnostic artifact is
`/tmp/twophase-cylinder-startup-diagnostic.toml`.
The mimetic residual is a maximum local pressure-fit
diagnostic over candidate constraints, distinct from KKT solve convergence.
Its failing variant/row and support feasibility remain undiagnosed; no fit
threshold or physical error budget has been changed.

The independent toy's numerical tests also pass on Julia 1.10.12. The package's
Julia 1.12 manifest cannot serve as a Julia 1.10 compatibility environment:
its resolved PrecompileTools uses a runtime interface unavailable on 1.10.
Compatibility checks therefore resolve a separate temporary environment,
leaving the repository's dependency manifest unchanged. Julia 1.10
also rejects an explicit zero interactive pool in `--threads=p,0`; the drivers
use `--threads=p` there and record actual pool sizes.
The main package's four-worker threading smoke passes with those independently
resolved Julia 1.10 dependencies.

The phase-face repair also passes all 39 existing performance/allocation
contracts at four workers. The toy's callback-based variable carrier uses
lazy reusable staging buffers, preserving the previous valid carrier if a
callback throws; warmed refreshes allocate zero bytes at 128² and 512².
This storage is absent from the primary constant-carrier fixture.

The existing-package cross-process driver passes at 1/2/3/4/7/8/16/32 workers
with exact equality of fields, topology keys, CFL steps, phase amounts, and
refinement labels. The toy driver checks all four backends, both flux policies,
square and odd rectangular constant-carrier trajectories, and midpoint-refreshed
variable shear; its full fields also agree exactly across all eight counts.
Each record includes a source fingerprint. These comparisons are correctness
evidence, separate from the performance curves.

An additional post-matrix fresh-process check in
`/tmp/twophase-concurrency-posttiming-20260929/` exercised independent
workspaces on Julia 1.10 at one, four, and eight workers. Polyester stored
and recomputed flux paths passed 18/18 checks at each count; affine observable
and amortized paths passed 24/24 at each count. Its result file records test
source hashes. These checks close the selected toy backends' concurrent
independent-workspace numerical obligation, not their scaling gate.

Several Julia 1.12.7 fresh-process experiments intermittently crashed in
compilation/GC, including a control kernel and a concurrent-workspace test.
Successful reruns do not establish their cause. Finalist measurements use a
separately resolved, pinned Julia 1.10.12 environment after cross-process
correctness validation; baseline and candidate share that environment. Do not
compare its times with Julia 1.12 measurements. The long original-baseline
integration process was resumed at 19:48 UTC after isolated measurements,
as recorded in `/tmp/twophase-scaling-baseline/suspended-process.json`.
After the session crash its PID 281507 is absent, and its log ends during the
vortex run without a final suite summary. That baseline integration run is
**interrupted/incomplete**, not a passing suite result; it was not restarted
for this Stage 2 wrap-up.

## Initial machine controls

The first isolated control run contains 180 fresh-process observations: three
sizes, three worker counts, four controls, and five independent repeats.
Its artifact is `benchmark/scaling/results/20260929-controls-baseline110-spread/`;
the exact runner sources are archived there with the raw observations,
configuration, workload, and summary. It uses Julia 1.10.12, Float64, one BLAS
and one GC worker, and a process affinity mask spread across the two L3 domains
of the Ryzen 9 5950X. This initial mask does **not** pin individual workers.

| Control at 4096² | T₁ / T₄ / T₈ (s) | S₄, 95% interval | E₄ | S₈, 95% interval | E₈ |
|:--|:--|:--|--:|:--|--:|
| Stream | 0.3878 / 0.3127 / 0.3247 | 1.24 [1.23, 1.25] | 31.0% | 1.19 [1.19, 1.21] | 14.9% |
| Five-point stencil | 0.4955 / 0.2721 / 0.2861 | 1.82 [1.80, 1.85] | 45.5% | 1.73 [1.69, 1.76] | 21.6% |
| Fixed random gather | 0.6898 / 0.2901 / 0.2658 | 2.38 [2.32, 2.45] | 59.4% | 2.60 [2.49, 2.67] | 32.4% |

These are median complete control trajectories, with independently bootstrapped
speedup intervals. Each curve fixes its own iteration count across workers;
the table does not compare the absolute work of different controls. All control
oracles have zero error and all requested workers participate in the separate
ownership probe. The bandwidth numbers use logical traffic, not measured DRAM
transactions. The controls neither establish nor waive any CFD scaling gate.

At 2048² the gather reaches E₄=79.9% and E₈=56.5%, while at 512² the stencil
reaches E₄=86.7%. Cache-size and placement effects therefore matter. Several
eight-worker cache/launch observations have large spread; the 512² stream
samples are also shorter than the required 0.2 seconds. Those observations
remain diagnostic. The next comparison uses explicit worker affinity with
recorded CPU assignments before attributing variability to a backend.

The follow-up `20260929-controls-baseline110-pinned-spread/` run contains 60
fresh-process observations for the launch and stencil controls at 512² and
4096², with the same workloads. Every compute worker is bound to a distinct
physical CPU, verified in each raw record. At 512², stencil T₈ changes from
0.915 to 0.262 seconds and its S₈ interval narrows from [0.73, 5.87] to
[5.44, 5.78]; T₄ remains about 0.42 seconds. At 4096² the pinned stencil
still gives E₄=46.1% and E₈=22.2%, consistent with the earlier large-array
plateau. The eight-worker empty-launch median falls from 1.15 to 0.350 seconds
for 57,344 launches, with the large outliers removed. Pinning is therefore the
declared placement for the toy screen and its finalists; the process-mask-only
curve remains separate. Both artifacts archive the resolved Project/Manifest,
runtime checksum receipt, source provenance, and exact harness.

A separate experimental tiled full-Strang LW kernel reduces global temporary
storage using chunk-owned halo scratch. It passes 1,793 checks at both one and
four workers, plus an optional Polyester odd-grid check. Warmed allocation is
independent of grid size (zero for serial; scheduler-sized for task backends).
This is numerical evidence only: the experiment has not been selected or
propagated into the package, and it still needs absolute serial and parallel
timing against the unfused toy.

## Bounded toy policy screen

The artifact `20260929-toy-screen-baseline110-pinned-spread/` contains 76
fresh-process samples, two independent repeats for each policy/count. Every
run advances the same 1024² initial condition through 256 steps to `T=0.0875`.
All report mass error zero, checksum 1,048,576, and translation RMS error
`6.848696974952322e-7`; every requested worker is observed during the separate
instrumented step. The source, runtime, resolved dependencies, and worker
placement are frozen and archived.

| Backend | Flux | Tile | T₁ / T₄ / T₈ (seconds) |
|:--|:--|:--|:--|
| Serial | Recompute | Full grid | 0.5481 / — / — |
| Serial | Stored | Full grid | 0.7918 / — / — |
| Julia dynamic | Recompute | 128×32 | 0.5701 / 0.2973 / 0.2838 |
| Julia dynamic | Recompute | 1024×32 | 0.5649 / 0.2216 / 0.2290 |
| Julia dynamic | Stored | 128×32 | 0.7846 / 0.5952 / 0.5984 |
| Julia dynamic | Stored | 1024×32 | 0.7814 / 0.5246 / 0.5833 |
| Chunked spawn | Recompute | 128×32 | 0.5652 / 0.2528 / 0.2052 |
| Chunked spawn | Recompute | 1024×32 | 0.5730 / 0.1782 / 0.1932 |
| Chunked spawn | Stored | 128×32 | 0.7869 / 0.4695 / 0.4689 |
| Chunked spawn | Stored | 1024×32 | 0.8022 / 0.4430 / 0.4784 |
| Polyester | Recompute | 128×32 | 0.5538 / 0.1491 / 0.0825 |
| Polyester | Recompute | 1024×32 | **0.5418 / 0.1202 / 0.0742** |
| Polyester | Stored | 128×32 | 0.7724 / 0.1945 / 0.1152 |
| Polyester | Stored | 1024×32 | 0.7894 / 0.1277 / 0.0880 |

All one-worker paths use a contiguous full-grid traversal, independent of the
parallel tile choice. The standard finalist is **Polyester, recomputed fluxes,
1024×32 tiles**. Its one-worker time is within noise of the serial control,
and it leads at both four and eight workers. Stored fluxes add memory and
launches without winning this complete-step comparison. The two ordinary task
backends are rejected from the larger finalist matrix. Persistent tasks are
not justified by this result and are not being added.

The screen's S₄≈4.51 and S₈≈7.31 are promising but not acceptance evidence:
there are only two repeats and the fastest trajectories are under 0.2 seconds.
The 40 MiB recompute working set also straddles this CPU's 32 MiB-per-domain
L3 capacity, so a superlinear four-worker result cannot be extrapolated to
DRAM-sized grids. Finalists use longer fixed trajectories, five independent
repeats, and the previously declared size ladder.

The independent padded-halo variant passes 936 parity checks plus eight guards
at both one and four workers, and a four-worker Polyester check. It refreshes
only periodic halos before each sweep and adds no full-grid copy per step.
A bounded follow-up compares it, the standard seam loops, and full-Strang
fusion at 512² and 4096² before selecting the final storage policy. Neither
experimental layout has been propagated into the core toy or main package.

The first bounded layout screen is now recorded in
`benchmark/scaling/results/20260929-layout-screen-baseline110-pinned-spread/`.
It contains 36 independent fresh-process observations: standard, halo, and
original fused layouts at 512² and 4096², one/four/eight pinned physical
workers, and two repeats per point. The respective fixed trajectories use
1,024 and 16 steps. Every sample has zero full-field difference from its
serial numerical reference; the largest mass error is below `1.0e-14`.

| Grid | Layout | T₁ / T₄ / T₈ (seconds) | Serial versus standard | Four/eight-worker absolute result |
|:--|:--|:--|--:|:--|
| 512² | Standard | 0.4291 / 0.1131 / 0.0673 | Reference | Reference |
| 512² | Halo | 0.4405 / 0.1258 / 0.0763 | 2.7% slower | 11% / 13% slower |
| 512² | Fused | 1.6125 / 0.4190 / 0.2288 | 276% slower | 270% / 240% slower |
| 4096² | Standard | 0.8530 / 0.6522 / 0.6745 | Reference | Reference |
| 4096² | Halo | 0.7240 / 0.6548 / 0.6760 | 15% faster | Within 0.4% of standard |
| 4096² | Fused | 1.9250 / 0.6007 / 0.4860 | 126% slower | 8% / 28% faster |

These are medians of only two samples per point, so they rank candidates for
follow-up but do not pass G2.3 or choose a final storage policy. The original
fused layout fails the declared 5% serial-regression budget at both sizes:
its attractive 4096² parallel times and self-speedups come from a much slower
one-worker implementation. Halo improves the large serial trajectory but
does not improve its four/eight-worker time, while losing on the smaller grid.
A second 36-sample paired screen in
`benchmark/scaling/results/20260929-affine-layout-screen-baseline110-pinned-spread/`
compares standard, affine fused observable steps, and affine fused amortized
trajectories at the original `(1024,32)` tile. It uses the same fixed workloads,
worker placement, and two repeats per point; all numerical state and mass
checks pass. Medians in seconds are:

| Grid | Mode | One worker | Four workers | Eight workers |
|:--|:--|--:|--:|--:|
| 512² | Standard | 0.4294 | 0.1137 | 0.0684 |
| 512² | Affine observable steps | 0.5068 | 0.3113 | 0.1644 |
| 512² | Affine amortized trajectory | 0.4302 | 0.2935 | 0.1554 |
| 4096² | Standard | 0.8701 | 0.6483 | 0.6812 |
| 4096² | Affine observable steps | 0.7446 | 0.5007 | 0.4861 |
| 4096² | Affine amortized trajectory | 0.5942 | 0.3566 | 0.3108 |

The affine candidate combines contiguous SIMD indexing with typed batch
dispatch that removes boxing, so this whole-step change cannot be assigned to
SIMD alone. Amortized copy-back is a separate experimental factor: it exposes
the state only after a complete trajectory, and its trajectory timing is not
the time of an individually observable step. The small grid favors standard
transport. The large grid merits final validation of the affine candidate,
but two-repeat screens do not pass the five-repeat performance gate.
An observable-step method also serves a final-state-only trajectory, so the
strongest serial reference for an amortized trajectory can include standard,
halo, and both fused tile choices. An amortized-only implementation cannot
serve as the serial reference for an individually observable-step curve.

A final 24-sample exploratory tile screen in
`benchmark/scaling/results/20260929-affine-tile-screen-baseline110-pinned-spread/`
compares **amortized trajectories only** for `(128,8)` and `(512,8)` tiles.
Its two-repeat medians, again in seconds, are:

| Grid | Tile | One worker | Four workers | Eight workers |
|:--|:--|--:|--:|--:|
| 512² | (128,8) | 0.6060 | 0.3302 | 0.1986 |
| 512² | (512,8) | 0.4625 | 0.3130 | 0.1629 |
| 4096² | (128,8) | 0.8088 | 0.4146 | 0.3141 |
| 4096² | (512,8) | 0.6490 | 0.3818 | 0.2747 |

All 24 tile samples match their serial numerical reference and conserve mass
within the established check. This screen does not establish observable-step
timings for `(512,8)`, actual worker participation inside the fused passes, or
the final efficiency target. The selected Stage 2 candidates are standard
Polyester/recompute at `(1024,32)` on the smaller grid and affine fused
`(512,8)` on the larger grid, with observable-step and amortized modes measured
separately. This screen left G2.3 and G2.4 unresolved; the completed
fresh-process matrix and final gate decision follow below.

## Final Stage 2 matrix: target missed, investigation closed

The frozen artifact
`benchmark/scaling/results/20260929-stage2-final-baseline110-pinned-spread/`
contains 280 observations: five independent Julia 1.10.12 processes for each
of 56 fixed configurations. Its `report-corrected.json` and plots use the
recorded step sequence at each size, pinned physical workers, and a separate
one-worker reference for each numerical/API scope. All source, runner,
configuration, and workload hashes remain fixed. Every timed sample lasts at
least 0.368 seconds; maximum mass error is `7.42e-14`, full-field and tracked
probe differences are exactly zero, and all requested workers participate in
every mandatory four/eight-worker pass. The separate 16-process affine
correctness gate remains the broader count/repetition evidence.

| Grid, complete-trajectory mode | T₁ / T₄ / T₈ (s) | Self E₄, 95% interval | Self E₈, 95% interval |
|:--|:--|:--|:--|
| 512² standard observable | 3.9933 / 1.0925 / 0.6313 | 0.914 [0.892, 0.959], pass | 0.791 [0.773, 0.817], pass |
| 2048² standard observable | 2.1076 / 1.2262 / 1.2677 | 0.430 [0.419, 0.452], fail | 0.208 [0.203, 0.218], fail |
| 2048² affine observable `(512,8)` | 2.2870 / 1.2521 / 0.7878 | 0.457 [0.423, 0.545], inconclusive | 0.363 [0.339, 0.405], fail |
| 2048² affine amortized `(512,8)` | 1.5984 / 0.8948 / 0.5448 | 0.447 [0.419, 0.450], fail | 0.367 [0.352, 0.372], fail |
| 4096² standard observable | 1.5518 / 1.1732 / 1.2173 | 0.331 [0.325, 0.337], fail | 0.159 [0.157, 0.162], fail |
| 4096² affine observable `(512,8)` | 1.5063 / 0.9443 / 0.8094 | 0.399 [0.389, 0.421], fail | 0.233 [0.227, 0.242], fail |
| 4096² affine amortized `(512,8)` | 1.1809 / 0.7073 / 0.5090 | 0.417 [0.411, 0.426], fail | 0.290 [0.260, 0.297], fail |

The table's **self efficiency** uses each fixed configuration's own one-worker
trajectory. For an individually observable step, the fastest validated serial
reference at 2048² and 4096² is the halo variant: 1.5901 and 1.3009 seconds.
The affine observable one-worker cost is respectively 43.8% and 15.8% higher.
The standard observable one-worker cost is also 32.5% and 19.3% higher than
the same compatible halo reference. Its match to the original serial core in
the early screen does not pass this strongest-serial guard.
For a final-state-only amortized trajectory, all observable methods are also
eligible serial references, but the fastest measured reference is the
amortized `(1024,32)` tile: 1.4300 and 1.0512 seconds. The selected
`(512,8)` amortized one-worker cost is respectively 11.8% and 12.3% higher.
An amortized-only method cannot be the reference for individually observable
steps. These serial costs exceed the initial 5% regression budget; no tradeoff
has yet been accepted. Candidate self scaling, absolute benefit against the
strongest compatible serial path, and serial regression are distinct facts.

The 512² cache case passes the 50% gate with standard transport. The mandatory
4096² case decisively misses it at four and eight workers on every measured
candidate; 2048² has one interval crossing the gate and no demonstrated pass.
G2.3 therefore cannot be marked met. Neither an explained shortfall nor a
fixed grid-dependent execution policy follows from these timings alone. The
later 24/32-worker SMT diagnostic was halted on a 4096², 29-step,
`fused_amortized` `(512,8)` p24 parity failure: maximum field difference
`3.125383729596365e-9` exceeds the unchanged `512ε` bound. Its partial
artifact and `failure.json` are in
`benchmark/scaling/results/20260929-stage2-smt-baseline110-pinned-spread/`.
Two earlier 2048² p32 standard/affine-observable samples passed, but they are
neither a full SMT curve nor evidence that the amortized case is correct.
Six fresh exact p24 reruns, 20 repeated untracked trajectories, odd-grid
poisoned-scratch stress at p7/p24, and five large p24 poisoned/per-step/probed
checks failed to reproduce the difference; their logs are in
`/tmp/twophase-affine-p24-diagnosis/`. The original failure remains unresolved
and the affine amortized experiment is not promoted. A later 4096² p1
supplementary `profile_affine.jl` run also failed its observable one-step tracked/untracked
**field-or-clock** equality check at line 254; the preserved stderr does not
separate the two. This rules out describing the unresolved affine problem as
SMT-only. A corrected-wrapper, single p1 detailed rerun subsequently found
exact field and clock equality in both modes; a separate 1031×257 p7
partition diagnostic found zero differences across 20 trajectories and 29
passes. Their archived provenance receipts are
`benchmark/scaling/results/20260929-profile-affine-mandatory-baseline110/n4096-p1-detailed-repro2.receipt.json`
and
`benchmark/scaling/results/20260929-stage2-smt-baseline110-pinned-spread/bounded-diagnostics-receipt.json`.
Neither clean result erases the original failed event. All affine profile
timings are diagnostic, and neither affine API
is promoted. Existing one-to-16-worker
finalist results remain separate. The
affine amortized two-worker time is 8.6% slower than its own one-worker time
at 2048² and 11.7% slower at 4096², despite actual worker participation.
Source inspection shows its one-worker path calls the serial `Matrix` loops,
whereas two or more workers enter Polyester's `@batch` path, whose captured
arrays may use pointer views. Different code generation or scheduling cost is
a hypothesis, not a measured explanation; do not assign all four-worker
shortfall to DRAM bandwidth on that basis. The
supplementary measurements are recorded in the detailed report: pinned STREAM
and stencil/gather controls, the complete six-point standard pass profile,
the three completed 2048² affine compute/copy-back profiles, memory and
allocation models, tiny-work fallback, placement, and standard 24/32-worker
SMT data. The standard pass profile uses process affinity masks, without
verified per-worker pinning, and supplies wall-clock sweep fractions rather
than Julia sampling stacks. The 4096² affine profile and affine SMT curve
remain incomplete after their respective parity events. These results do not
give a measured physical-traffic ceiling or quantified serial dependency for
both mandatory sizes. The original 50% target and the alternative explained
shortfall route both remain unmet; G2.4 does not support an affine replacement
policy. Stage 2 performance is **unaccepted** under the plan's stopping rule.
The current investigation stops here. Reopening the gate requires numerical
diagnosis of the affine events and a matched-placement, measured-ceiling
study of the large-grid shortfall; no Stage 3 implementation is authorized by
these Stage 2 measurements.

## Stage 3 implementation design, explicitly authorized with revised scope

The first main-package control is the existing `CellField` forward-Euler upwind
operator in `src/operators/advection/cell_advection.jl`. Keep its one signed
integrated flux per owned face segment, then gather/update each destination
cell from the immutable incidence schedule. The existing `upwind` and
`upwind_complete` scaling fixtures already separate fixed-`dt` transport from
transport plus CFL selection. Freeze prescribed face velocities, the same
physical step sequence, and periodic boundaries for this control. Compare a
single large block with many uniform leaves of 8×8 cells at equal global cell count;
do not merge the two topology curves. Profile face lookup/flux, cell gather,
CFL, field copies, and scheduler time at one and four/eight workers. The
strongest validated one-worker **upwind** implementation on the same layout is
the serial denominator; the dense LW toy is a cost attribution control, not a
numerical or serial-performance substitute.

The second workload is a research plain scalar LW
prototype. It must reproduce the toy's periodic, cell-average Strang
`x(dt/2), y(dt), x(dt/2)` method, constant carrier, face orientation, and
analytic translation fixture before any parallel scheduling change. Keep it
separate from the public upwind contract until conservation, seam and
constant-state checks, and the finest-pair smooth order of at least 1.8 pass.
Compare toy and main LW only with matched numerical settings and report
indirection, halo, pass, and launch fractions of their complete steps. Use a
validated main-package serial LW implementation as its own scaling denominator.
The `UniformCellLWWorkspace` prototype is retained under
`research/UniformLWAdvection/experiments/main_cell_lw/` for diagnosis only.
Its constant-carrier, uniform-periodic contract does not satisfy the revised
production objective. Production work instead starts with the existing general
`AdvectionWorkspace`: measure owned-face access, flux computation, incidence
gather, and CFL before changing their representation. Any dense interior
traversal must work in every leaf block, including blocks in a mixed-level
mesh, with the same flux formula and conservative owned coarse--fine segments.
Preserve arbitrary prescribed velocities and physical boundaries. A future
general LW numerical extension requires its own review; a globally uniform
dispatch is not accepted as a substitute for a general execution design.

The third workload is the existing uniform WY vortex-reversal path with LHF
reconstruction, `Fromm()` centered liquid/gas phase scalars and constant-one
tracers, peak-velocity CFL 0.4, alternating directional order, and the full
closed-wall `T=1` interval. Its current preparatory `wy_complete` fixture
includes face-carrier sampling, CFL, reconstruction, transport, and declared
diagnostics. Keep the prescribed flow and frozen uniform mesh identical across
worker counts and compare one-block/many-block layouts separately. Reuse the
discrete-streamfunction sampling already in the vortex example; never compare
its time directly to a periodic dense scalar stencil as equal work. Exclude
staggered transport until Stage 6.
In `_advect_wy!`, each call currently copies both intrinsic scalar source
fields and constructs new increment `CellField`s for the scalar positions.
Measure these copy and allocation fractions before changing its scheduler.
Workspace-owned reusable per-position buffers are only a later candidate:
they must preserve frozen sweep input, changing scalar sets, and correct
topology retargeting.

For G3.1, preserve existing scalar/VOF error thresholds and frozen roundoff
budgets; check volume, both phase amounts, constant-one identities, phase
bounds, and reversal geometry/scalar errors across fresh worker processes.
For G3.2, verify equal numerical fields for matched one-block/many-block
layouts and demonstrate that the large-block case has enough owned cell/face
work for four/eight workers. For G3.3, time complete uniform transport with
reconstruction and CFL, report both four/eight-worker efficiency and the
strongest serial reference within each numerical/API scope, and retain an
explained shortfall if the 50% goal is missed. For G3.4, use measured component
fractions and Stage 1 bandwidth/launch controls to explain the toy/main LW
gap. The current pinned launch control's one-worker branch launches no tasks,
so its short duration cannot serve as a parallel scheduler denominator; its
parallel absolute launch cost and the large stencil control are separate
measurements. A fused or recomputed path is accepted only
after proving its own face
ownership, conservation, workspace scratch lifetime, serial regression and
allocation budgets. WaterLily's pinned Cartesian loops inform the dense-loop
control, but its QUICK flux, Float32 timings, and absence of AMR/VOF do not
replace these gates.

### Stage 3 ownership preflight (source audit only)

The existing parallel traversals have disjoint destinations in the inspected
hot paths. `_parallel_block_columns` in `src/threading.jl` partitions each
block into nonoverlapping column ranges, including a single large block.
Upwind advection in `src/operators/advection/cell_advection.jl` writes one flux
slot per owned face, finishes both face passes, and then gathers into one
destination cell per column-range item. Its packed `boundary_entries::BitVector`
is built serially with the incidence schedule and only read by the parallel
gather and CFL loops; no packed-bit writes occur there. The WY passes in
`src/indicator/phase/vof_fluxes.jl` likewise write separate face slots or
cell destinations, with each completed pass preceding its dependent pass or
reconstruction. These are source-level ownership and barrier findings, not a
race-detector result.

The inspected reconstruction caches also perform shared structural mutation
outside parallel regions. `reconstruct!` in
`src/indicator/phase/interface_field.jl` resizes field caches before its
normal/PLIC passes, copies old planes before each height pass, and marks the
geometry current after the passes. `_height_cache!` in
`src/indicator/phase/height_functions.jl` builds its block-key dictionary,
ghost list, and donor lists serially; its later interior and ghost passes
write distinct height-array locations while reading donors. The phase-face
aperture path in `src/two_phase/two_velocity/transport_geometry.jl` resizes
and invalidates arrays before the per-face parallel fill and sets validity
afterward. No lazy dictionary build or shared resize was found inside these
particular parallel loops.

The contract for Stage 3 fixtures is one caller-owned workspace and field set
per concurrent solve, with no concurrent mesh/topology mutation and no
intermediate callback into a partially completed pass. Prescribed-velocity,
inflow, and initialization callbacks can execute on multiple workers and must
be pure or synchronized by the caller; use pure prescribed callbacks in the
fixed fixtures. A single `InterfaceField`, its WY workspace, or its derived
geometry caches cannot be advanced concurrently because those objects reuse
mutable scratch and validity state. Separate solves may run concurrently only
with distinct mutable fields and workspaces.

There is a separate public-API validation gap: the ordinary `PhaseScalar`
constructor creates separate liquid and gas fields, but a caller can manually
construct a scalar whose fields alias each other or another scalar. The WY
duplicate check currently compares only each scalar's `liquid` field object
identity; it does not reject every cross-phase alias. This is not a
demonstrated race for the standard constructors and fixtures, and this audit
does not change production behavior. Numerical parity, actual-worker probes,
and complete-workload performance remain required before any G3 gate passes.

### Next general execution design

The five-repeat fixed-CFL upwind study supports single-block four-worker
efficiency of 0.796 [0.744, 0.820] at 1024² and 0.640 [0.625, 0.677] at
2048², using the strongest compatible serial median in that study. It does
not establish toy-level absolute cost: the 2048² general upwind medians are
6.667 and 6.356 ns per scalar unknown per complete step at four and eight
workers, while the dense toy's different LW method measured 1.766 and
1.924 in its separate final cohort. Full fields matched the frozen upwind
oracle exactly. See the [Stage 3 report](multithreading-stage3-report.md) for
all layouts, uncertainty, required-work boundaries, storage costs, and the
retained runtime-failure receipts.

The current compact face cache and private flux ordering preserve the general
numerical method, but still visit a face record and a cell-incidence list for
regular block interiors. They also retain the original mesh face records.
The next experiment should change that representation, with the existing
general path as its numerical and timing reference:

1. Describe regular interior faces by block dimensions and arithmetic local
   indices. Apply this description in every leaf, including mixed-level meshes.
   Keep explicit owned segments for cross-block, coarse--fine, physical, and
   periodic seam connections. A same-block face is not necessarily an interior
   neighbour: a periodic seam in a single large block also has equal block IDs.
2. Use the same numerical flux formula through both traversals. Store each
   owned integrated flux once and preserve equal-and-opposite contributions.
   Derive regular-cell incidence from the Cartesian layout instead of loading
   a full CSR record. Check the accumulation order and numerical budgets
   explicitly before changing it.
3. Include live face-velocity access in the design. Reordering scalar access
   while gathering velocities from the old face order can retain substantial
   indirection. A dense velocity mirror would require an explicit refresh and
   mutation contract, and its refresh belongs in the complete workload cost.
4. Measure setup cost, retained storage, complete transport, and any required
   CFL scan. Compare one large block with many small blocks, then validate
   mixed-level ownership, physical inflow, periodic seams, cut cells, changing
   velocities, and independent workspaces. Instrument component costs in
   separate diagnostic copies, outside accepted timing samples.
5. Evaluate a general LW extension separately after this execution design is
   sound. The research tile pipeline demonstrates potential for intermediate
   cache reuse; applying it across coarse--fine connections requires correct
   intermediate boundary states at every dependent sweep. Its uniform
   constant-carrier contract is not a production substitute.

This is a testable representation proposal, not evidence that a general AMR
algorithm cannot approach dense-grid performance. The absolute performance
objective remains open until a general implementation demonstrates it.
