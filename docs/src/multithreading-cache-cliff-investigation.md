# Investigation of the uniform LW cache cliff

The [matched Julia–Fortran study](multithreading-julia-fortran-comparison.md)
found a sharp increase in wall-clock time **per scalar unknown per complete
three-sweep timestep** between 1024² and 2048², especially at high worker
counts. This investigation separates the measured fact from candidate cache,
memory-traffic, translation, and scheduling explanations. It does not change
the numerical kernel or the Stage 2 acceptance result.

**Finding:** the unusually large *parallel* slowdown is primarily a
cache-residency/traffic transition on the two-CCD test CPU. Around 1280²–1408²,
the five-array footprint crosses the two CCDs' nominal combined L3
capacity; source-of-fill counters show far more loads supplied from local
memory/IO beyond that range. A smaller one-CCD placement encounters its
slowdown earlier. Larger meshes amortize launch costs, but do not guarantee
better strong scaling when the smaller parallel run benefits from cache
reuse that the larger run loses. This evidence identifies the traffic
transition; it does not measure total DRAM bytes or establish a
kernel-specific bandwidth ceiling.

## What needs explaining

| Workers | Julia 1024² → 2048² | Fortran 1024² → 2048² |
|---:|---:|---:|
| 1 | 2.141 → 3.059 ns, 1.43× | 2.017 → 3.415 ns, 1.69× |
| 2 | 1.074 → 2.184 ns, 2.03× | 0.960 → 2.388 ns, 2.49× |
| 4 | 0.571 → 1.892 ns, 3.31× | 0.490 → 1.924 ns, 3.93× |
| 8 | 0.296 → 1.861 ns, 6.29× | 0.273 → 1.905 ns, 6.97× |
| 16 | 0.206 → 2.029 ns, 9.84× | 0.201 → 2.000 ns, 9.96× |
| 32 (SMT) | 0.213 → 2.390 ns, 11.22× | 0.268 → 2.447 ns, 9.13× |

Thus “about tenfold” describes the 16-worker transition, not the serial
transition. The 16-worker speedups fall from Julia 10.39×/Fortran 10.05×
at 1024² to 1.51×/1.71× at 2048². Both implementations show the cliff,
and their 2048² parallel absolute times overlap. The baseline 1024² and
2048² trajectories use different frozen step counts and physical final
times, so controlled length tests must rule out a trajectory-length effect.

## Cache geometry and first-order models

AMD's [Ryzen 9 5950X architecture diagram](https://gpuopen.com/download/GDC_AMD_Ryzen_Processor_Software_Optimization.pdf)
shows two core complex dies, each with its own 32 MiB L3 cache, rather than
one uniform 64 MiB cache. The local Linux cache topology used by the
benchmark agrees. The spread placement assigns physical workers across
both dies. AMD's same diagram also shows separate CCD-to-I/O-die data paths,
with 32 B/cycle in the read direction and 16 B/cycle in the write direction
per CCD. Thus a spread team changes both available L3 capacity and access
to the external data path. A spread placement win by itself is not proof of
cache residency; a shift in the size knee, coupled with source-of-fill or
traffic evidence, would be stronger. [Stengel et al.'s original stencil ECM study](https://arxiv.org/html/1410.5010v2)
explains why cache layer conditions, write allocation, and traffic at each
memory-hierarchy level can change both single-core time and multicore
saturation. Its measured machine is an older Intel CPU, so its numerical
throughput is not a performance limit for this Ryzen.

The toy stores five active Float64 grids: scalar `q`, velocities `u,v`, and
two scratch fields. Their total sizes are 40 MiB at 1024² and 160 MiB at
2048². Under *ideal* half-grid ownership, each CCD's share is 20 versus
80 MiB. A simple total-footprint threshold solves
`5 × 8 × N² = 2 × 32 MiB`, giving `N≈1295`; this assumes perfectly
partitioned ownership and that all five grids must stay resident between
sweeps. A single sweep actually needs roughly its source, one velocity,
and destination, so a three-grid threshold is `N≈1672` under the same
ideal assumptions. Halo overlap, replacement, shared cache users, and
whether the same worker reuses a tile can move the knee earlier or later.
Neither number is an observed cutoff.

The same ideal arithmetic with only one 32 MiB CCD yields `N≈916` for five
grids and `N≈1182` for three. These are capacity guides for the placement
experiment, not predictions that a benchmark must change speed at those
exact dimensions. Both source implementations keep the same three-sweep
algorithm and `(512,16)` tile geometry across 1024² and 2048²; there is no
algorithm switch at either dimension.

There may be **two reuse distances**, so a broad rise is more plausible
than one perfect step function. The sweep sequence is
`x(q,u→first)`, `y(first,v→second)`, `x(second,u→q)`.
Under an idealized streaming order, a velocity value reused between its
two x sweeps (or from one y sweep to the next step) may have to survive
accesses to all five distinct grids; an intermediate scalar reused in the
next sweep has a shorter, roughly three-grid pass-level distance. With two
32 MiB L3 caches, the five-grid guide is `N≈1295` and the three-grid guide
is `N≈1672`. [AMD describes its last-level cache as
non-inclusive/exclusive](https://www.amd.com/en/resources/product-security/bulletin/amd-sb-7032.html),
so private L2 data might add capacity rather than always duplicating L3.
An optimistic bound adding all 16 × 512 KiB active L2 caches gives 72 MiB,
moving those guides to `N≈1374` and `N≈1774`. Actual ownership, coherence,
eviction, and tile ordering decide whether any of that extra capacity helps;
the bounds are a model to check against timing and fill-source data, not a
fit to a known cutoff.

The fixed `(512,16)` tile holds 8192 scalar cells, or 64 KiB per array.
Three tile-local arrays total about 192 KiB, below each core's 512 KiB L2
shown in AMD's diagram. Yet the code completes one whole-grid sweep before
starting the next. At 2048², a worker processes many tiles per sweep; it
cannot assume its first tile survives in L2 until the next sweep. The
cross-sweep retention question is primarily about the shared L3 and memory
traffic, not the nominal size of one tile.

A conditional cold-stream accounting assigns each sweep one 8-byte source
read, one 8-byte velocity read, an 8-byte destination writeback, and an
8-byte write allocation per cell: `3 × 32 = 96 B` per complete step.
With this **assumed** traffic, 2048² at 16 workers would imply about
`96 B / 2.03 ns ≈ 47 GB/s` for Julia, while 1024² would imply
`96 B / 0.206 ns ≈ 466 GB/s`. The latter cannot represent this desktop's
DRAM stream; cached data or a different effective transfer count must be
involved. The former being of the same order as the earlier 4096² STREAM
control is suggestive, not proof of DRAM saturation. Actual memory bytes
have not been measured for these runs; the later counters measure selected
cache-fill categories, not total bytes. The ECM paper's
separation of core and cache-level transfer costs is more appropriate than
equating source-level load counts with DRAM traffic.

## Falsifiable predictions before follow-up measurements

| Hypothesis | Prediction | Contradicting observation |
|:--|:--|:--|
| Loss of useful L3 residency | A fine size ladder should show a knee over a capacity range, plausibly around 1280–1700 for the idealized three-to-five-grid two-CCD bounds. A one-CCD compact team should encounter the **knee at smaller `N`**, not merely be slower everywhere. Demand fills sourced from local DRAM/IO should rise relative to fills sourced from local L3/peer L2. | Comparable compact/spread knees and no source-of-fill change across the slowdown weaken simple L3-capacity attribution. |
| Saturated external data path after the knee | Normalized time should flatten or worsen as workers are added while measured DRAM bytes/s approaches the machine's *measured same-pattern* ceiling. Reducing transferred bytes should improve large-grid time more than small-grid time. | Low measured DRAM utilization with a persistent cliff points to cache-level, translation, instruction, or synchronization cost. |
| Power-of-two stride, cache-set, or TLB conflict | A nearly equal-work padded leading dimension near 2048 should improve time sharply relative to 2048 exactly, with changed cache/TLB indicators; the fine size ladder may show a narrow spike rather than a broad knee. | Similar times for 2048 and padded/non-power-of-two neighbors, with a broad size knee, weaken conflict-specific attribution. |
| Step-count or process setup artifact | At fixed size, changing 64 to 256 completed steps should materially change normalized steady-state time if warmup or startup dominates. | Stable ns/unknown/step at both lengths weakens this explanation. |
| Julia-specific scheduling cost | At the same size/layout, Julia alone should show the cliff or a much larger increase in per-step scheduling cost. | The matched Fortran OpenMP cliff already weighs against this as the dominant cause; a shared control-kernel knee would weaken it further. |

The serialized follow-up measured a fine-size ladder in both languages,
one- versus two-CCD placement, fixed-size trajectory-length controls,
and Fortran padding and per-worker source-of-fill counters. All measurements
used the same validated three-sweep
numerical contract, fixed tile geometry, verified worker placement, and
per-sample full-field checks where applicable. The Linux [Zen 3 PMU event definitions](https://raw.githubusercontent.com/torvalds/linux/master/tools/perf/pmu-events/arch/x86/amdzen3/memory.json)
distinguish demand data-cache fills from same-node DRAM or I/O (`0x43/0x08`)
and from local L3 or another L2 in the same core complex (`0x43/0x02`).
They separately count hardware-prefetch fills from those sources (`0x5a/0x08`
and `0x5a/0x02`). These are fill counts, **not total DRAM bytes**: writebacks
are absent, and the demand category excludes hardware prefetch. The generic
Linux `cache-misses` counter on Zen 3 [maps to event `0x0964`, an L2
miss](https://raw.githubusercontent.com/torvalds/linux/master/arch/x86/events/amd/core.c),
not an L3 miss. Counter permission failure
would have been recorded as missing evidence; on this host the relevant
own-thread events were available. AMD's
[uProf L3 counter documentation](https://docs.amd.com/r/en-US/57368-uProf-user-guide/1.2.3.-L3-Cache-Performance-Monitor-Counters-L3PMC)
and [IBS metrics guide](https://docs.amd.com/r/en-US/68658-uProf-getting-started-guide/Analyzing-Performance-Using-IBS-Data)
identify further cache and load-latency observations, not collected here.

## Follow-up observations and diagnosis

The first two frozen controls retained the exact three-sweep numerical
kernel, `(512,16)` tiles, input-field/oracle construction, pin checks, and
full-field comparisons from the matched study. Each number below is the
median of three fresh processes. All 66 fine-ladder samples lasted at least
0.342 s, all 48 main placement samples at least 0.484 s, and all 36
lower-size placement samples at least 0.426 s; every sample used its
requested workers in each sweep and reproduced the serial oracle's complete
field exactly. The raw observations and per-point 95% bootstrap intervals
are retained in the local ladder data (`docs/src/assets/cache-cliff/ladder/summary.json`)
and placement data (`docs/src/assets/cache-cliff/placement/summary.json`), with a
separate lower-size placement bracket (`docs/src/assets/cache-cliff/placement-low/summary.json`).
Across all six follow-up phases there were **268 accepted samples**:
66 ladder, 48 main placement, 36 lower-size placement, 40 step-count,
48 PMU, and 30 padding. Every complete field matched its serial oracle
bitwise, every requested worker participated, and worker pinning passed.
The local asset manifest (`docs/src/assets/cache-cliff/asset-manifest.json`)
hashes all exported configurations, raw samples, summaries, validation
receipts, and plots; the local ignored result directory also retains the
full source snapshots and binary field fixtures.

### A. Fine size ladder, 16 workers spread across both CCDs

| Mesh `N²` | Five active grids (MiB) | Julia ns/(cell·step) | Fortran ns/(cell·step) |
|---:|---:|---:|---:|
| 768 | 22.5 | 0.183 | 0.208 |
| 1024 | 40.0 | 0.190 | 0.202 |
| 1152 | 50.6 | 0.234 | 0.261 |
| 1280 | 62.5 | 0.292 | 0.290 |
| 1408 | 75.6 | 0.734 | 0.711 |
| 1536 | 90.0 | 1.020 | 1.008 |
| 1664 | 105.6 | 1.319 | 1.325 |
| 1792 | 122.5 | 1.560 | 1.570 |
| 1920 | 140.6 | 1.777 | 1.799 |
| 2048 | 160.0 | 2.052 | 2.020 |
| 2304 | 202.5 | 2.181 | 2.166 |

The steepest measured segment is 1280² to 1408²: Julia increases 2.51×
and Fortran 2.45× per cell and step. The five-grid footprint crosses from
62.5 to 75.6 MiB, bracketing the two CCDs' aggregate 64 MiB L3 and the
optimistic 72 MiB L3-plus-active-L2 guide. Both dimensions use three
512-wide x tiles and the same kernel. Their tile counts increase only from
240 to 264 (10%) as the y extent grows, much less than the normalized
time increase. A later, continuing rise through 1664²–1792² is compatible
with the shorter three-grid reuse-distance guide, though the timing alone
cannot assign misses to a particular array. The fine ladder also rules out
an explanation requiring *exactly* a power-of-two width: 1408, 1536, and
1664 are non-power-of-two widths and already show the cliff.

*Figure: Full-step time per unknown in the two-language fine-size ladder. Local generated file: `docs/src/assets/cache-cliff/ladder/ladder-p16.svg`.*

### B. One versus two CCDs at eight physical workers

| Mesh `N²` | Julia compact / spread ns | Fortran compact / spread ns | Julia compact/spread | Fortran compact/spread |
|---:|---:|---:|---:|---:|
| 768¹ | 0.266 / 0.253 | 0.295 / 0.279 | 1.05× | 1.06× |
| 896¹ | 0.353 / 0.246 | 0.367 / 0.272 | 1.43× | 1.35× |
| 960¹ | 0.520 / 0.291 | 0.520 / 0.305 | 1.79× | 1.71× |
| 1024 | 0.716 / 0.316 | 0.677 / 0.276 | 2.27× | 2.45× |
| 1280 | 1.570 / 0.394 | 1.529 / 0.399 | 3.98× | 3.83× |
| 1536 | 2.056 / 0.990 | 1.995 / 0.984 | 2.08× | 2.03× |
| 2048 | 2.697 / 1.862 | 2.569 / 1.934 | 1.45× | 1.33× |

¹ The 768²–960² rows are a separately randomized, frozen lower-size
bracket with the same kernel, placement policy, and three repeats.

At the same eight-worker count, compact and spread medians are close at
768²; the compact penalty grows through 896² and 960², then peaks near
1280² before shrinking at 2048². For a compact eight-core team, the
five-grid L3 guide is `N≈916`, or `N≈972` if its eight private 512 KiB L2
caches all add distinct useful capacity. The observed onset around
896²–960² agrees with this broad guide. For a spread team, four cores per
CCD can use two L3 caches and at most 4 MiB of active L2 in the optimistic
model; its size knee occurs later. Placement also changes
CCD-to-I/O-die access, so this table alone cannot separate added L3
capacity from external-path bandwidth. The raw source-of-fill counter
control is designed to distinguish those mechanisms.

This placement result also explains why the headline cliff is much larger
at 16 workers than at one. The one-worker baseline is pinned to CPU 0 and
can use only its CCD's 32 MiB L3; the 1024² five-grid set is already 40 MiB,
above that ideal single-CCD guide. The 16-worker spread team spans both
CCDs, so an ideal balanced partition is only 20 MiB per CCD at 1024² and
can retain much more useful state. At 2048² that partition grows to 80 MiB
per CCD, far beyond either 32 MiB L3. Thus the *relative* parallel speedup
can collapse as the team loses a cache-capacity advantage that serial work
largely lacked already. Actual tile ownership, useful live data, L2,
coherence, and external-path traffic prevent treating those footprints as
measured residency, but the compact/spread and size results support this
mechanism in both languages.

*Figure: Compact versus spread placement at eight physical workers. Local generated file: `docs/src/assets/cache-cliff/placement/placement-p8.svg`.*

*Figure: Lower-size placement bracket at eight physical workers. Local generated file: `docs/src/assets/cache-cliff/placement-low/placement-p8.svg`.*

### C. Matched trajectory lengths

To check whether the baseline's mesh-dependent step counts created the
cliff, each language ran **both** sizes for exactly 64 and 256 complete
steps at 16 spread workers, with five fresh processes per configuration.
All 40 final fields matched their corresponding serial oracle exactly and
all requested workers participated. The values below are ns/(cell·step);
the step-control summary (`docs/src/assets/cache-cliff/steps/summary.json`) retains
the intervals and raw sample-duration flags.

| Steps | 1024² Julia / Fortran | 2048² Julia / Fortran | 2048² ÷ 1024², Julia / Fortran |
|---:|---:|---:|---:|
| 64 | 0.264 / 0.386 | 2.095 / 1.963 | 7.93× / 5.08× |
| 256 | 0.230 / 0.287 | 2.103 / 2.014 | 9.14× / 7.02× |

Every 1024² diagnostic sample is shorter than 0.2 s (0.017–0.081 s),
so those precise ratios are less robust than the long baseline and ladder
samples. The small-grid value does change with trajectory length,
especially in Fortran, but a large matched-length gap persists. At 2048²,
the 64- and 256-step medians are near 2 ns in both codes; all those
samples lasted at least 0.517 s. The differing frozen baseline step counts
therefore contribute to the comparison but are not the primary cause of
the near-tenfold 16-worker gap.

*Figure: Matched 64- and 256-step controls at two mesh sizes. Local generated file: `docs/src/assets/cache-cliff/steps/steps-p16.svg`.*

### D. Source-of-fill counters in the Fortran diagnostic clone

The separately compiled diagnostic keeps the same three conservative
sweeps, logical field, and tile geometry. It counts events on each pinned
OpenMP worker over the complete trajectory in user mode. Two fresh-process
cohorts measure demand fills and hardware-prefetch fills separately; each
also measures instructions, cycles, and Linux's generic `cache-misses`.
All 48 counter samples produced the exact serial-oracle field, passed
worker/affinity checks, and had `time_running/time_enabled = 1.0` for every
event on every worker, so no multiplexing correction was needed. The
counter summary (`docs/src/assets/cache-cliff/pmu/summary.json`) and
raw counters (`docs/src/assets/cache-cliff/pmu/raw.jsonl`) retain every per-worker
value and event encoding. These instrumented times form their own cohort,
separate from the primary wall-time tables.

The following numbers are **fill events per logical scalar unknown per
complete step**, summed over all 16 workers and divided by cells × steps.
The first two columns are demand fills; the last two are hardware-prefetch
fills measured in separate processes.

| Mesh `N²` | Demand from local memory/IO | Demand from internal cache | Hardware prefetch from local memory/IO | Hardware prefetch from internal cache |
|---:|---:|---:|---:|---:|
| 1024 | 0.000562 | 0.025273 | 0.002341 | 0.086000 |
| 1280 | 0.001438 | 0.020001 | 0.065735 | 0.225767 |
| 1408 | 0.003767 | 0.017372 | 0.181396 | 0.206144 |
| 2048 | 0.024175 | 0.023123 | 0.415825 | 0.130183 |

*Figure: Fortran fill origins across the 16-worker mesh ladder. Local generated file: `docs/src/assets/cache-cliff/pmu/pmu-origin-p16.svg`.*

From 1024² to 2048², local-memory/IO **demand** fills rise about 43×
and **hardware-prefetch** fills about 178× per cell and step. The
hardware-prefetch rise is already about 2.76× from 1280² to 1408², the
same interval as the sharp wall-time rise. Summed user-mode instructions
stay near 22.5–22.9 per cell and step across these sizes, so a tenfold
increase in *executed instruction count* is not the cause. Summed cycles
increase, but include time spent in OpenMP waits; they cannot be treated
as pure stencil work or a simple IPC estimate.

Placement changes **where** fills come from, not just the time to serve a
fixed number of fills. At eight workers, 1024² compact versus spread
placement produced 0.150557 versus 0.003824 local-memory/IO
hardware-prefetch fills per cell-step (about 39×), and 0.006745 versus
0.000475 demand fills (about 14×). At 1280² the corresponding
hardware-prefetch values were 0.331128 versus 0.065191 (about 5×).
The separate counter placement rows (`docs/src/assets/cache-cliff/pmu/summary.json`)
therefore weaken a *pure* CCD fabric-throughput explanation in which the
same volume of data simply travels through twice as many paths. Fabric
bandwidth and latency can still contribute to elapsed time.

*Figure: Fortran fill origins under compact and spread eight-worker placement. Local generated file: `docs/src/assets/cache-cliff/pmu/pmu-origin-placement-p8.svg`.*

Linux defines demand `0x0843` and hardware-prefetch `0x085a` as data-cache
fills from DRAM **or I/O connected to the same NUMA node**. Their internal
cache counterparts `0x0243` and `0x025a` mean L3 **or another L2 in the
same core complex**; the categories do not identify an individual array.
They omit dirty writebacks, other fill classes, and any uncounted traffic;
the two source-origin cohorts were not measured simultaneously. Hence the
rates cannot be multiplied by 64 bytes and called measured total DRAM
traffic or used to declare a saturated bandwidth ceiling. The generic
`cache-misses` event is an instruction-plus-data **L2 request miss**, and
indeed does not track the local-memory fill rise monotonically. The
[diagnostic counter notes](../../research/UniformLWAdvection/fortran/diagnostics/README.md)
describe scope, programming, and remaining counter limitations.

### E. Padded leading-dimension control

The diagnostic clone also ran the same logical fields at padded leading
dimensions `N`, `N+8`, and `N+32`, with counters off and five fresh
processes per configuration. Padding changes the physical stride while
preserving every logical value and the three-sweep order. All 30 complete
fields were bitwise identical to the serial oracle, and worker/affinity
checks passed. Packing and unpacking logical fields occurred outside the
timer. Its padding summary (`docs/src/assets/cache-cliff/padding/summary.json`)
reports these ns/(cell·step) medians:

| Mesh `N²` | Unpadded `N` | `N+8` | `N+32` |
|---:|---:|---:|---:|
| 1024 | 0.245 | 0.207 | 0.210 |
| 2048 | 1.979 | 1.945 | 1.942 |

At 2048², padding changes median time by at most about 2%, while the
1024² unpadded samples form two clusters (0.245–0.253 and 0.201–0.205 ns).
Their median does not establish a robust 1024² padding win. The
large 1024²-to-2048² gap survives all three strides. This rejects an
*exact power-of-two stride conflict* as the dominant cliff mechanism,
while not excluding smaller TLB, alignment, or cache-set contributions.
Within the padded-clone `N+8` cohort, the 2048²/1024² normalized-time
ratio remains about 9.38×. The clone and frozen primary binary are
different builds, so their absolute medians remain separate.

*Figure: Fortran padded-leading-dimension control. Local generated file: `docs/src/assets/cache-cliff/padding/padding-p16.svg`.*

### Diagnosis and next decision

The evidence has a consistent order: the 16-worker size knee occurs as
the ideal five-grid footprint crosses the two L3 caches; the compact
one-CCD team slows at a smaller size; placement strongly changes the
source of data-cache fills; and the same cliff appears in both Julia and
Fortran while instruction work remains almost constant in the Fortran
counter cohort. Matched step counts preserve a large gap, and changing
power-of-two stride does not remove it. A loss of useful cache reuse,
followed by much greater external fill traffic, is the best-supported
explanation of the tenfold **p16** normalized-time rise. The exact
per-array reuse history, contribution of each cache level, DRAM bytes,
and fraction of elapsed time limited by memory bandwidth remain unmeasured.

The [optimization plan's performance gate](multithreading-optimization-plan.md)
requires either the original 50% efficiency at four or eight workers on
each mandatory large fixture, or a measured bandwidth/latency ceiling or
quantified serial dependency and an explicit revised goal for an accepted
explained shortfall. This investigation does not supply that ceiling, and
the 2048²/4096² target remains unmet in both codes. Stage 2 therefore
remains **unaccepted** for performance; do not promote this implementation
to Stage 3 on the strength of the cache-cliff diagnosis alone.

The next optimization experiment should target cross-sweep data movement
while preserving the numerical dependencies and full-field parity of
`x(Δt/2)`, `y(Δt)`, `x(Δt/2)`. A bounded candidate is a separate
tile-pipeline prototype: compute the three sweeps for each tile with
halo-aware private intermediate scratch, keep input `q` frozen until all
tiles finish, and write final interiors into a separate output buffer.
It must retain dense velocity reads and each arithmetic expression's order,
then pass whole-field parity before performance is considered. This tests
whether shorter intermediate reuse distance pays for its extra halo work
and output buffer; no speedup is presumed. Quantify the changed traffic with
representative kernel-specific measurements, including writebacks or
equivalent byte accounting, and measure a credible throughput/latency
ceiling for the same access pattern. Then rerun **absolute** full-step
time and `E₄/E₈` on the original mandatory 2048² and 4096² fixtures,
against the fastest compatible serial baseline. Scheduler-only Julia
changes are lower priority because the Fortran/OpenMP code reproduces the
same large-grid loss. Any revised efficiency target must be proposed and
recorded explicitly under the plan rather than inferred from these data.

## Reproducing the controls

The subsequent [Stage 3 investigation](multithreading-stage3-report.md)
records the user-authorized Julia-only tile-pipeline experiment and the
general main-package advection comparison. That follow-up tests whether part
of this cost can be reduced; it does not retroactively accept Stage 2 or
establish a measured memory-bandwidth ceiling. The Fortran controls below
remain archival evidence.

The repository file `benchmark/scaling/cache_cliff/README.md` gives the
six-phase command sequence and the separate counter diagnostic build.
Use a fresh output directory for each phase; the harness refuses to
overwrite an earlier result. The local `/tmp` Julia executable, project,
and depot paths in the recorded commands were specific to this host and
can be replaced with `--julia`, `--project`, and `--depot`. The locally generated
ladder configuration (`docs/src/assets/cache-cliff/ladder/configuration.json`),
PMU configuration (`docs/src/assets/cache-cliff/pmu/configuration.json`), and
source hashes (`docs/src/assets/cache-cliff/ladder/source_sha256.json`) capture the
resolved benchmark settings and numerical implementation. Each phase's
asset directory also contains raw JSONL and a source-hash manifest.
The frozen Julia numerical core, Fortran release kernel, and release
binary were not edited for this investigation; padding and counters live
in a separate diagnostic clone. All plots derive from the exported raw
data, and the asset manifest (`docs/src/assets/cache-cliff/asset-manifest.json`)
records their checksums.
