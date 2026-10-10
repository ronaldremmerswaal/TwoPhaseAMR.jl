# Reusable copied-source Stage 3 measurement

[study.jl](study.jl) compares a prepared advection source copy with the frozen
narrow-ID source (`047457e224baa1c5b5dfa26376eff71acd61de02b225ca7e13360df6c3184541`).
It accepts a generator receipt and source-locked validation suite manifest,
then runs one 1024² mesh layout per fresh numerical Julia process. The checked
driver validates every periodic face reference, axis, dual volume, and halo
donor before timing. Each cohort builds a fresh control oracle and compares
every candidate and control final field bitwise against it.

Run from the repository root with a new output directory:

```sh
JULIA_DEPOT_PATH=/tmp/twophase-scaling-tools-depot:/home/ronald/.julia \
  /home/ronald/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia \
  --startup-file=no --project=benchmark/scaling \
  benchmark/scaling/stage3/copied_source/study.jl \
  --candidate-source benchmark/scaling/results/PREPARED-CANDIDATE \
  --candidate-label rolling_flux \
  --probe-receipt benchmark/scaling/results/PREPARED-CANDIDATE/rolling-flux-receipt.json \
  --suite-dir benchmark/scaling/results/PREPARED-CANDIDATE \
  --case 1024:8 --steps 128 --repeats 3 --workload fixed \
  --output benchmark/scaling/results/NEW-COPIED-SOURCE-COHORT
```

Supported layouts are `1024:8` and `1024:1024`. The `fixed` workload times
`advect!` with a fixed prescribed carrier and timestep; `cfl` separately scans
CFL inside every timed step and uses `dt = 0.4 * CFL bound`. The selected
workload and all source, validator, project, runtime, driver, initial-field,
and oracle hashes are recorded and checked between child launches. Every
timed child must last at least 0.2 seconds. A failed setup, field comparison,
short interval, or changed input leaves its command/raw receipts but no
completed summary; a partial cohort is not a performance result.

[audit_study.jl](audit_study.jl) independently validates a completed output
and reports same-worker ratios and strongest-compatible-serial parallel
efficiency with bootstrap intervals:

```sh
JULIA_DEPOT_PATH=/tmp/twophase-scaling-tools-depot:/home/ronald/.julia \
  /home/ronald/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia \
  --startup-file=no --project=benchmark/scaling \
  benchmark/scaling/stage3/copied_source/audit_study.jl \
  benchmark/scaling/results/NEW-COPIED-SOURCE-COHORT
```

The runner and auditor are Julia research tools. Generated sources, fields,
commands, and audit receipts belong under ignored `benchmark/scaling/results/`.
