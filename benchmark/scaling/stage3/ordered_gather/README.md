# Copied-source ordered-gather diagnostic

This exploratory diagnostic starts from the frozen narrow-ID advection source
(SHA256 `047457e224baa1c5b5dfa26376eff71acd61de02b225ca7e13360df6c3184541`).
The generator changes only `_cell_advection_ordered_sum`: four face orders
produced by the package's mesh builder select signed x/y fluxes directly, while
every other packed order calls the original generic helper. Both paths start
with `zero(T)` and perform the same four sequential additions. This applies to
all AMR leaves, both plain and cut-cell advection, and promoted CFL sums.

The validation driver explicitly includes each copied `TwoPhaseAMR` module,
checks the loaded method's file and source hash, compares all 24 distinct face
permutations plus two nonpermutation codes across Float32/Float64 signed-zero
and cancellation cases, then runs the full compact frozen-reference suite.
The study gates timing on both copied suites and a fresh control-source oracle.

The original `study.jl` is the historical batched diagnostic. It launched
1024² and 2048² in one process and stopped in the control's 2048² mesh setup;
neither original pilot accepted a timing row. Its commands and failure receipts
remain in the ignored `results/20261010-ordered-gather-pilot*` directories.

The follow-up [safe_study.jl](safe_study.jl) runs one mesh case per child
process. Its copied [checked driver generator](make_checked_driver.jl) scans
every periodic face reference, face axis, geometry, dual volume, and halo donor
before field setup and timing. Each paired run starts from the same copied
initial field, uses the same fixed step count and CPU placement, and compares
the entire final field bitwise with a fresh checked control oracle. Copied
source suites must pass first. Run it from the repository root with Julia
1.12.7 tooling and the frozen Julia 1.10.12 numerical environment:

```sh
JULIA_DEPOT_PATH=/tmp/twophase-scaling-tools-depot:/home/ronald/.julia \
  /home/ronald/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia \
  --startup-file=no --project=benchmark/scaling \
  benchmark/scaling/stage3/ordered_gather/safe_study.jl \
  --case 1024:1024 --steps 256 --repeats 3 \
  --output benchmark/scaling/results/NEW-ordered-gather-case
```

Use a new output directory for each cohort. `--case 1024:8` with 64 steps
exercises the many-block uniform layout; 2048:2048 with 32 steps is supported
but the recorded fresh control oracle failed ordinary mesh validation before
timing (`invalid face axis at 6163788`). Do not count that failed cohort as a
comparison or silently select another setup attempt.

The local 1024² one-block five-pair study consists of a three-repeat cohort
and a separate two-repeat cohort, both with the same checked driver and
256-step trajectory. Its control/candidate median ns per cell advection was
8.459/7.732 at p1, 2.149/2.079 at p4, and 1.585/1.425 at p8. The 1024²
block-8 three-pair diagnostic used 64 steps; its p8 median was 3.151/4.467,
and all three paired p8 control/candidate ratios were below one. The copied
helper was therefore **rejected for unconditional production use** despite
its one-block gain. These are fixed-dt `advect!` timings, not CFL-inclusive
complete-step timings or mixed-AMR scaling results.

Audit a completed one-case cohort or combine the two one-block cohorts with:

```sh
JULIA_DEPOT_PATH=/tmp/twophase-scaling-tools-depot:/home/ronald/.julia \
  /home/ronald/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia \
  --startup-file=no --project=benchmark/scaling \
  benchmark/scaling/stage3/ordered_gather/audit_safe_study.jl \
  benchmark/scaling/results/20261010-ordered-gather-safe-1024b8-r1

JULIA_DEPOT_PATH=/tmp/twophase-scaling-tools-depot:/home/ronald/.julia \
  /home/ronald/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia \
  --startup-file=no --project=benchmark/scaling \
  benchmark/scaling/stage3/ordered_gather/audit_combined_study.jl \
  benchmark/scaling/results/20261010-ordered-gather-safe-1024b1024-r3 \
  benchmark/scaling/results/20261010-ordered-gather-safe-1024b1024-r2
```

The read-only audits check the copied source trees, suite receipts, runtime,
driver, project, input, oracle, commands, every final-field hash, complete row
coverage, and timing summaries. The combined audit reports five-process
medians and bootstrap intervals; the three-pair block-8 result remains
exploratory.

A separate [CFL driver generator](make_cfl_driver.jl) and
[checked wrapper](make_checked_cfl_driver.jl) define a fixed prescribed carrier
with one `stable_timestep` call inside every timed step and
`dt = 0.4 * CFL bound`. [cfl_study.jl](cfl_study.jl) uses that driver with a
fresh control oracle and the same full-field comparison; it is a distinct
trajectory and has no accepted throughput result yet. Its intended CLI is the
same as `safe_study.jl` with a new output directory. A small driver and field
correctness smoke passed at 64², two steps, and p1: both copied sources gave
the same finite full field, bitwise, with `dt = 0.4 * CFL bound`. It makes no
throughput claim. Replay it as:

```sh
JULIA_DEPOT_PATH=/tmp/twophase-scaling-tools-depot:/home/ronald/.julia \
  /home/ronald/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia \
  --startup-file=no --project=benchmark/scaling \
  benchmark/scaling/stage3/ordered_gather/cfl_smoke.jl \
  --output benchmark/scaling/results/NEW-ordered-gather-cfl-smoke
```

The generator and runners are research-only Julia tools; their generated
drivers, source copies, fields, and receipts live under ignored
`benchmark/scaling/results/` directories.

For historical replay, the original batched command was:

```sh
JULIA_DEPOT_PATH=/tmp/twophase-scaling-tools-depot:/home/ronald/.julia \
  /home/ronald/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia \
  --startup-file=no --project=benchmark/scaling \
  benchmark/scaling/stage3/ordered_gather/study.jl \
  --cohort benchmark/scaling/results/20261010-narrow-ids-final \
  --output benchmark/scaling/results/20261010-ordered-gather-pilot
```

That original design specified three adjacent control/candidate pairs at p1,
p4, and p8. Each fresh process would batch the same 1024² one-block trajectory of
1024 steps and 2048² one-block trajectory of 128 steps, producing 18 processes
and 36 timed case rows. Each sample must last at least 0.2 seconds and match
the fresh oracle bitwise. Raw times, hashes, process commands, affinities,
pair order, and copied-suite receipts would be kept in its ignored output.
The control mesh failure stopped this design before those timed rows existed.
