# Copied-source all-order scalar-selector diagnostic

This research probe starts from the frozen narrow-ID advection source, SHA-256
`047457e224baa1c5b5dfa26376eff71acd61de02b225ca7e13360df6c3184541`.
[make_probe.jl](make_probe.jl) changes only the ordered four-flux sum in a
copied source tree. A balanced scalar `ifelse` selector handles every packed
`UInt8` order while preserving the original zero accumulator and addition
sequence. Production source is unchanged.

[validation_driver.jl](validation_driver.jl) checks all 256 packed orders and
2,304 direct signed-zero/cancellation cases, then runs the compact advection
suite against each copied module. The source-locked control/candidate p1 suite
receipts and generated patch are under ignored
`benchmark/scaling/results/20261010-scalar-selector-candidate/`.

[study.jl](study.jl) runs one periodic 1024² case per fresh Julia process,
checks every face reference, axis, dual volume, and halo donor before timing,
and compares each final field bitwise against a fresh checked control oracle.
It supports separate `fixed` and per-step `cfl` workloads; only `fixed` was
timed for this candidate. Run a new exploratory fixed-dt cohort from the
repository root with:

```sh
JULIA_DEPOT_PATH=/tmp/twophase-scaling-tools-depot:/home/ronald/.julia \
  /home/ronald/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia \
  --startup-file=no --project=benchmark/scaling \
  benchmark/scaling/stage3/scalar_selector/study.jl \
  --case 1024:8 --steps 128 --repeats 3 --workload fixed \
  --suite-dir benchmark/scaling/results/20261010-scalar-selector-candidate \
  --output benchmark/scaling/results/NEW-scalar-selector-case
```

The `--suite-dir` receipt must match the exact validator, compact test, Julia
runtime/project, source trees, and suite report hashes. Without it, the study
runs fresh copied-source suites. Every cohort records process commands,
affinity, source and tooling hashes, oracle and full-field comparisons, raw
times, pair order, and a summary. [audit_study.jl](audit_study.jl) independently
checks a completed cohort:

```sh
JULIA_DEPOT_PATH=/tmp/twophase-scaling-tools-depot:/home/ronald/.julia \
  /home/ronald/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia \
  --startup-file=no --project=benchmark/scaling \
  benchmark/scaling/stage3/scalar_selector/audit_study.jl \
  benchmark/scaling/results/20261010-scalar-selector-1024b8-fixed-r3
```

Both completed three-pair fixed-dt cohorts passed the independent receipt
audit and matched their fresh oracles bitwise. Median ns per cell advection
for control → candidate were:

| Layout | p1 | p4 | p8 |
|:--|--:|--:|--:|
| 1024², block 8, 128 steps | 13.241 → 13.814 | 5.095 → 5.264 | 4.115 → 3.144 |
| 1024², block 1024, 256 steps | 8.371 → 10.246 | 2.127 → 2.705 | 1.580 → 1.752 |

The many-block p8 gain varied across pairs; its same-worker ratio 95% bootstrap
interval included one. The one-block p1 slowdown appeared in all three pairs,
with a ratio interval of 0.814–0.833. The candidate is **rejected for
production promotion**. No five-repeat or CFL-inclusive throughput claim was
made. These uniform periodic cases do not establish mixed-AMR scaling.

All study and audit programs are Julia. Generated copies, fields, and receipts
are under ignored `benchmark/scaling/results/` directories.
