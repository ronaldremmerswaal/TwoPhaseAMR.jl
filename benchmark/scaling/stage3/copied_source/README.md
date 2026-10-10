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

## Production-only complete-step baseline (prepared, not run)

`production_cfl_study.jl` measures the unchanged narrow-ID production source
without requiring a candidate. It reuses the checked CFL driver: each timed
step calls `stable_timestep` on the prescribed constant carrier, chooses
`dt = 0.4 * bound`, then calls complete `advect!`. This trajectory differs
from the fixed-`dt` studies above. First run one ordinary setup child at each
of 1, 4, and 8 threads; any failed child excludes that layout. Only a passed
setup directory can be supplied to the measurement phase. Each layout needs
a separate new setup and measurement directory; no 2048 case is supported.

```sh
JULIA_DEPOT_PATH=/tmp/twophase-scaling-tools-depot:/home/ronald/.julia \
  /home/ronald/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia \
  --startup-file=no --project=benchmark/scaling \
  benchmark/scaling/stage3/copied_source/production_cfl_study.jl \
  --phase setup --case 1024:8 --steps 128 --repeats 5 \
  --output benchmark/scaling/results/NEW-PRODUCTION-CFL-SETUP

JULIA_DEPOT_PATH=/tmp/twophase-scaling-tools-depot:/home/ronald/.julia \
  /home/ronald/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia \
  --startup-file=no --project=benchmark/scaling \
  benchmark/scaling/stage3/copied_source/production_cfl_study.jl \
  --phase measure --case 1024:8 --steps 128 --repeats 5 \
  --setup-gate benchmark/scaling/results/NEW-PRODUCTION-CFL-SETUP \
  --output benchmark/scaling/results/NEW-PRODUCTION-CFL-MEASURE
```

Repeat with `--case 1024:1024 --steps 256` and distinct directories. The
measurement phase creates a fresh one-thread control oracle and fifteen
independent timed children (five each at 1, 4, and 8 threads). It records all
process outcomes, requires at least 0.2 seconds per timed child, compares
every full field bitwise to the oracle, and reports nanoseconds per unknown
per complete step and efficiency relative to its one-thread median. A
completed summary is required; partial rows after a failure are excluded.
