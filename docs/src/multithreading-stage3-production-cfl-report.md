# Stage 3 production CFL-inclusive baseline: 10 October 2026

The unchanged checked narrow-ID general upwind source has a validated
1024² **one-block** fixed-carrier, per-step-CFL baseline. Across five
independent fresh processes at each worker count, its median complete step
costs 17.842, 5.439, and 3.968 ns per cell at one, four, and eight workers.
The four- and eight-worker efficiency intervals both clear 50% **on this
fixture**. The 1024² many-block setup gate stopped at eight workers before
timing, so there is no accepted many-block CFL curve. The 2048² and general
Lax–Wendroff (LW) obligations remain open; Stage 3 is **not accepted**.

The measured source is the frozen production advection SHA-256
`047457e224baa1c5b5dfa26376eff71acd61de02b225ca7e13360df6c3184541`
and tree SHA-256
`0bb8826fb52d143ae0fa5f84e043f4ed4d4feeadb4c7328d6fbda33cbf7095a2`.
It ran under the frozen Julia 1.10.12 project/manifest. Each child constructed
and checked its full periodic mesh, created its own field, velocity, and
workspace, then advanced a fresh input field for 256 timed steps in the
one-block case. The carrier was prescribed and constant `(0.7, -0.3)`, but
each **timed** step called `stable_timestep(velocity, workspace; cfl=0.9)`,
selected `dt = 0.4 * bound`, and called complete `advect!`. Setup, warmup,
pinning, and field export were outside the timer. This is a practical
per-step-CFL transport cost for a fixed carrier, not a changing-velocity
simulation or an isolated CFL kernel measurement.

The one-block setup gate passed separate ordinary source children at one,
four, and eight workers. Its measurement cohort then passed a fresh
one-worker oracle and all 15 timed children (five per worker count). All 15
complete fields matched the oracle bitwise; prescribed-carrier CFL bounds
were exact, and every timed interval met the 0.2-second minimum. The independent
read-only Julia audit checked all 16 child exits, commands, placement,
source/driver/project/input/setup locks, full-field hashes, CFL policy, and
nanoseconds-per-cell-per-complete-step normalization. Values below are
medians; intervals are 95% independent-process percentile bootstraps. The
only compatible serial implementation in this cohort is the same production
source at one worker, so its median is the serial reference in
`Eₚ = T₁/(p × Tₚ)`.

| Workers | ns/cell/complete step | Efficiency [95%] |
|--:|--:|--:|
| 1 | 17.842 | 1.000 |
| 4 | 5.439 | 0.820 [0.663, 0.875] |
| 8 | 3.968 | 0.562 [0.553, 0.661] |

The earlier fixed-`dt` pilots used a different `dt` and trajectory. Their
costs cannot be subtracted from these numbers to estimate isolated CFL
overhead or presented as a matched source comparison. This study measures
one production implementation, with no candidate speedup claim.

The separately predeclared 1024² **8×8-cell-leaf** setup gate passed at one
and four workers, then its eight-worker child exited before producing a
workspace report. It left three command receipts, two setup rows, no passed
gate summary, and **zero timed rows**. The copied production source and tree
hashes still matched the frozen control. No retry or many-block measurement
followed; the cause is unknown. A separate one-off *instrumented* setup
fixture passed for control and row-fused copies with checked target, stable
permutation, and CSR assertions (research commit `f97d4e7`), but it does not
establish dependable uninstrumented setup or explain the failed gate.

The completed one-block setup and timing receipts are ignored raw artifacts
under `benchmark/scaling/results/20261010-production-cfl-1024b1024-setup/`
and `20261010-production-cfl-1024b1024-measure-r5/`. The excluded many-block
receipt is `20261010-production-cfl-1024b8-setup/`. The reviewable Julia
coordinator and independent auditor are
`benchmark/scaling/stage3/copied_source/production_cfl_study.jl` and
`production_cfl_audit.jl`.

The next bounded source-level optimization candidate is to combine each
parallel cell-rate write in `stable_timestep` with a workspace-owned tile
maximum, then reduce only the tile maxima. It must compare the **stored,
rounded** rate, preserve the existing `zero(T)` and final division, and use
the original serial scan for nonfinite or signed-zero exceptional values.
This could remove a separate full-cell maximum traversal, but it is
**unimplemented and unmeasured**. Reliable ordinary workspace setup across
the declared layouts is a prerequisite to interpreting any new throughput
comparison. Dynamic-carrier behavior, mixed-AMR complete-step costs, 2048²,
and production general LW still require separate validation.
