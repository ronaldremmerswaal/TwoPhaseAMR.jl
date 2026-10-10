# Stage 3 block-interior general upwind: 10 October 2026

The general `advect!` path now computes ordinary faces inside every AMR leaf
from arithmetic neighbours and keeps a compact, explicitly owned residual
path for block edges, periodic seams, and coarse–fine faces. Physical
boundary faces retain their separate flux pass. A matched five-repeat study
of the complete forward-Euler upwind step
found lower median cost than the **previous general remap** in all nine
size/layout/worker groups. Every one of the 90 output fields matched the
frozen one-worker oracle bitwise. This is an improvement to the existing
general method, not a production Lax–Wendroff implementation.

At 1024² in one block, the new path costs 2.947 ns per scalar cell-step at
four workers and 1.693 at eight. Strongest-compatible-serial efficiency is
0.730 [0.681, 0.979] and 0.635 [0.605, 0.665] respectively, so both
intervals support the 50% target. At 2048² in one block, four workers cost
3.806 ns per cell-step with efficiency 0.678 [0.604, 0.764]; eight workers
cost 3.334 but efficiency falls to 0.387 [0.383, 0.391]. The four-worker
configuration is the supported efficiency choice at that size. The
eight-worker median is lower, but these five-process results alone do not
establish a repeatable direct p4-versus-p8 gain. For 1024² partitioned into
8×8-cell blocks, the four-worker efficiency interval [0.498, 0.668]
crosses 0.5 and is inconclusive.

## Matched complete-step results

The entries below are previous general remap → new block interior, in median
**ns per scalar unknown per complete `advect!` step**. Each layout has five
fresh Julia 1.10.12 processes per variant and worker count. `Eₚ` uses the
fastest compatible one-worker median divided by `p` times the new median;
the new path is fastest at one worker in all three layouts. Brackets are
95% independent-process bootstrap intervals. A gate is supported only if
its entire interval is above 0.5, missed if entirely below, and otherwise
inconclusive. The p1 point estimate is one by construction; its intervals
are omitted from this table.

| Grid and layout | p1 previous → new | p4 previous → new | New E₄ [95%] | p8 previous → new | New E₈ [95%] |
|:--|--:|--:|--:|--:|--:|
| 1024², one block | 14.805 → 8.600 | 4.594 → 2.947 | 0.730 [0.681, 0.979], supported | 4.833 → 1.693 | 0.635 [0.605, 0.665], supported |
| 2048², one block | 17.129 → 10.321 | 7.542 → 3.806 | 0.678 [0.604, 0.764], supported | 6.718 → 3.334 | 0.387 [0.383, 0.391], missed |
| 1024², 8×8-cell blocks | 15.504 → 13.307 | 8.118 → 5.194 | 0.640 [0.498, 0.668], inconclusive | 6.759 → 4.592 | 0.362 [0.343, 0.445], missed |

The one-worker same-method gains over the previous remap are 1.72×,
1.66×, and 1.17× in table order. At four workers they are 1.56×, 1.98×,
and 1.56×; at eight, 2.85×, 2.01×, and 1.47×. These ratios compare the
same complete general-upwind method at the same worker count, initial field,
time step, and trajectory length. The shortest accepted timed interval was
0.436 s, above the predeclared 0.2 s floor. The per-sample timings,
dispersion, bootstrap intervals, full-field hashes, affinity checks, and
launch receipts are in the ignored local `benchmark/scaling/results/`
directory `20261010-block-interior-final-julia/` (`summary.json`,
`summary.csv`, `raw.jsonl`, `manifest.json`, `commands.jsonl`, and
`summary.png`). The graph shows the same medians and efficiency intervals.

The periodic unit-square fixture uses Float64 smooth cell averages, a
prescribed live face carrier `(0.7, -0.3)`, and `dt=0.35/n`. One block and
8×8-cell blocks cover the same `n²` scalar cells; velocity components are
not counted as transported unknowns. Matched trajectories run 256 steps at
1024² and 32 at 2048². Every timed call uses the public general `advect!`;
owned residual and regular fluxes, physical-boundary handling, and the cell
gather/update are inside the timer. Setup, pinning, a fixed-carrier CFL scan,
and field export are outside it. One process batches the three cases for a
given variant, worker count, and repeat, so their machine conditions may be
correlated. Process/case order was randomized and recorded. Julia workers
were pinned to distinct physical cores in a spread order on the Ryzen
5950X; the recorded kernel affinity matched the requested CPUs. The tooling
coordinator used Julia 1.12.7; every numerical child used the frozen Julia
1.10.12 project and manifest.

## Correctness, participation, and storage

The new workspace stores dense x/y face IDs and areas for arithmetic
neighbours in each leaf, retaining the original face IDs for **live**
carrier and aperture reads. Residual faces use compact owner metadata;
block-edge cells keep ordered incidence entries. The four regular fluxes
are added in each cell's original face-reduction order. This preserves the
generic owned-face method at coarse–fine and periodic seams rather than
substituting a uniform-only solver. Focused Julia 1.10.12 numerical suites
passed 11,502 checks at each of p1/p4/p8, including frozen-reference plain
and cut-cell comparisons, mutation/closed cases, concurrency, and existing
scalar/threading clients. The complete package suite was not rerun.

Untimed copied-source probes kept the accepted source free of counters.
At 1024² one block, 2048² one block, and 1024² in 8×8-cell blocks, every
requested p4/p8 worker entered the previous remap's owned-flux and gather
passes and the new path's regular-flux, residual-flux, and gather passes.
Each of the 18 one-step probe fields matched the clean oracle bitwise.
Participation establishes that callbacks ran, not that work was balanced.
The source copies, exact patches, hashes, and per-pass worker IDs are in
ignored `20261010-block-interior-diagnostics/probes/` and
`worker-evidence.json` under `benchmark/scaling/results/`.

Independently warmed Julia 1.10.12 allocation checks at 64² and 256² found
0 B per `advect!` and CFL call at p1. At p4 and p8, the bounded per-call
overhead was 2,656 B for `advect!` and 2,064 B for CFL at both sizes; it
does not grow per cell in these checks.

Single-process `Base.summarysize((mesh, field, velocity, workspace))`
measured the **deduplicated reachable Julia object graph** as follows.
The mesh graph is unchanged by the new workspace. Separately reported
workspace sizes include reachable mesh references and must not be added
to the mesh size. Construction timings were single diagnostics that may
include compilation.

| Fixture | Previous reachable graph | New reachable graph | Change |
|:--|--:|--:|--:|
| 1024², one block | 373.0 MiB | 257.7 MiB | −30.9% |
| 2048², one block | 1490.9 MiB | 1029.4 MiB | −31.0% |
| 1024², 8×8-cell blocks | 445.0 MiB | 357.3 MiB | −19.7% |

These are reachable object sizes, not process RSS or measured DRAM traffic.
The exact bytes, mesh/workspace components, and setup timings are in ignored
`20261010-block-interior-diagnostics/setup/setup-evidence.json`.

## Separately measured CFL and component costs

The carrier is prescribed and constant during each matched trajectory, so
its warmed `stable_timestep` scan was **excluded** from the `advect!` timer.
Median scan times in milliseconds over the same five processes are below;
each entry is previous remap → new path. A solver that recomputes CFL each
step must charge this cost. Thus the fixed-trajectory result does not close
the dynamic-CFL part of Stage 3 gate G3.3.

| Grid and layout | p1 CFL | p4 CFL | p8 CFL |
|:--|--:|--:|--:|
| 1024², one block | 12.55 → 10.04 | 4.93 → 3.81 | 5.12 → 2.54 |
| 2048², one block | 53.83 → 47.04 | 28.66 → 16.18 | 27.32 → 15.08 |
| 1024², 8×8-cell blocks | 12.52 → 11.19 | 6.94 → 5.26 | 5.98 → 4.26 |

A second copied-source diagnostic ran the same 2048² one-block field and
32-step trajectory at p4/p8, with one timing bracket around residual plus
regular owned-face flux, one around physical-boundary handling, and one
around ordered cell gather/update. All four fields matched the clean oracle
bitwise. The single-process wall-time shares are diagnostic fractions, not
five-repeat estimates or hardware counters:

| Source | Workers | Residual + regular flux | Gather/update | Physical boundary |
|:--|--:|--:|--:|--:|
| Previous remap | 4 | 62.9% | 37.1% | <0.01% |
| Previous remap | 8 | 63.4% | 36.6% | <0.01% |
| New block interior | 4 | 53.7% | 46.3% | <0.01% |
| New block interior | 8 | 61.2% | 38.8% | <0.01% |

Periodic seams and interblock residual flux are inside the combined flux
pass. The near-zero physical-boundary line on this periodic fixture says
nothing about AMR communication or a general interface workload. Both flux
and gather remain material. The regular flux loop traverses cells with x
fast, while the original mesh-face ordering can make x-face carrier reads
strided. This is a plausible remaining locality cost, not a measured DRAM
traffic attribution. The copied-source timings and field hashes are in
ignored `20261010-block-interior-diagnostics/component/`.

A source-level payload estimate for the one-block fixed-`dt` step counts the
scalar and increment arrays (~16 B/cell), live face velocities (~16), regular
x/y face IDs (~16), regular x/y flux arrays (~16), and packed face order
(~1): about **65 B per scalar cell** of leading touched dense arrays before
periodic seams and edge metadata. The separately allocated CFL `rates`
array is not touched by this fixed-`dt` `advect!` call. For context, the
dense toy's recompute path uses about 40 B/cell of q, u, v, and two scratch
arrays; the research pipeline uses about 32 B/cell of q, u, v, and next,
plus tile scratch. These are code-level payload estimates, distinct from
reachable-object storage above. They are not measured cache residency or
DRAM traffic and do not identify the cause of the remaining timing gap.

## Exploratory tile and next work

A separate copied-source **128×8 regular-flux tile** experiment compared
the current traversal with a tiled traversal at 2048² in one block. This
diagnostic used three fresh-process pairs per p1/p4/p8 configuration
and a **128-step** trajectory, rather than the accepted matrix's 32 steps.
All 18 fields matched its own clean oracle bitwise; every timed interval
was at least 1.595 s. Copied-source compact suites passed 11,474 assertions
each for the control and tile. Median matched control → tile costs were
10.599 → 10.527 ns/cell-step at p1, 3.462 → 3.696 at p4, and
3.458 → 3.060 at p8. All three p8 pairs favored the tile, while two of
three p4 pairs regressed. This mixed one-case diagnostic is not an
acceptance cohort and is not compared directly with the 32-step medians.
The current production traversal remains selected; the tile is retained as
research evidence only. Copied-source patches, frozen hashes, full-field
checks, and all 18 timings are in ignored
`benchmark/scaling/results/20261010-flux-tile-2048-oneblock/`.

The next general-upwind experiments should test a narrower face-ID type
with a checked wide-index fallback (up to ~8 B/cell less leading ID
payload), and inspect ordered-gather code generation and live carrier-read
locality. They must preserve original face ownership, live velocity and
aperture changes, and each cell's exact reduction order. The measured CFL
scan also needs charging in a dynamic-CFL complete-step study; neither
the fixed-`dt` efficiency result nor the tile pilot passes that part of
G3.3. No payload estimate or single-process component fraction promises
a speedup. General LW is a separate numerical addition requiring matched
method parity before performance claims; interface/WY and phase-scalar
transport remain deferred.

## Absolute references, scope, and reproduction

The older [Stage 3 report](multithreading-stage3-report.md) gives the dense
Julia toy LW and separate research pipeline LW reference costs below. These
are absolute costs in common ns-per-cell-step units, not numerical-method
speedups: general upwind is one first-order forward-Euler sweep, while those
LW implementations perform three sweeps,
and their reported trajectories have different lengths. The dense toy
loads both prescribed velocity matrices; the general method uses live
mesh-face values. No upwind/LW field parity is claimed. The absolute
general-path cost gap remains, as do production general LW, interface,
and phase-scalar tasks. Overall Stage 3 is not accepted on this result.

| Grid | Workers | New general upwind | Dense toy LW | Research pipeline LW |
|:--|--:|--:|--:|--:|
| 1024² | 4 | 2.947 | 0.480 | 0.578 |
| 1024² | 8 | 1.693 | 0.277 | 0.330 |
| 2048² | 4 | 3.806 | 1.766 | 1.284 |
| 2048² | 8 | 3.334 | 1.924 | 1.136 |

The comparison baseline is the ignored frozen **previous general-remap
working tree** at `benchmark/scaling/results/20261010-stage3-block-interior-baseline/`.
It is distinct from the September original-Git freeze used in the older
Stage 3 report. Previous/new `cell_advection.jl` SHA-256 hashes are
`19f5490c04dd9947504d9e17c42d31077990a8da191b6f9b86f5c5bbddde0018`
and `cecb8463cccada2e16d90f5245b506842d9cc90f6df9c380bb69fee673e4f115`;
the new `embedded_operators.jl` hash is
`bc9b95f79e4ac353ca5c1236ad138f8c9d5996953053c11a361b9054c86ff33e`.
The ignored cohort manifest records all package sources, drivers, Julia
binaries, both resolved projects, and the Julia coordinator sources. Its
tooling `Project.toml` and `Manifest.toml` SHA-256 hashes are
`fb7860579c9ea3392e7baef81306f4effe011490d814c814aefcf10505308a4a`
and `8eb6d0ccb40211f8f4fa5496bea97d3fb452edc73061cd36696352d4c690d936`.
The tracked reproduction procedure is in
`benchmark/scaling/stage3/block_interior/README.md`. A different source
or dependency resolution is a new cohort.

The project-wide scripting rule in the repository's `AGENTS.md` now specifies Julia for
research, benchmark, orchestration, analysis, and plotting scripts. The 41
active Python entrypoints were replaced by Julia entrypoints using the
explicit `benchmark/scaling/Project.toml` tooling environment with JSON
and CairoMakie; numerical subprocesses retain their frozen Julia 1.10.12
environment. The tooling procedure is in `benchmark/scaling/README.md`.
Saved-data replays in ignored `benchmark/scaling/results/migration-validation-*`
directories matched archived deterministic summaries and fields, with
expected script-hash/path changes; bootstrap draws can differ because
Julia and Python use different random streams, and a few cross-language
figures can differ in rendering/last-bit arithmetic. The Stage 2 runner
replay kept the legacy percentile rank and draw count. Specific local
receipts include `migration-validation-run-summary/` for the 60-row Stage 2
runner replay, `migration-validation-stage2-report/` for saved summary and
figure regeneration, `migration-validation-cross-language/` and
`migration-validation-cache-pipeline-analysis/` for later-stage saved data,
and `migration-validation-block-pilot/` for the new Julia coordinator.
Migration artifacts were written into new ignored directories so historical
receipts and figures were preserved.
