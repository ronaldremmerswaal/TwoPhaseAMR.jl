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

After the narrow-ID five-repeat matrix and diagnostics complete and the
numerical CPU is released, run from the repository root:

```sh
JULIA_DEPOT_PATH=/tmp/twophase-scaling-tools-depot:/home/ronald/.julia \
  /home/ronald/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia \
  --startup-file=no --project=benchmark/scaling \
  benchmark/scaling/stage3/ordered_gather/study.jl \
  --cohort benchmark/scaling/results/20261010-narrow-ids-final \
  --output benchmark/scaling/results/20261010-ordered-gather-pilot
```

The fixed diagnostic has three adjacent control/candidate pairs at p1, p4,
and p8. Each fresh process batches the same 1024² one-block trajectory of
1024 steps and 2048² one-block trajectory of 128 steps, producing 18 processes
and 36 timed case rows. Each sample must last at least 0.2 seconds and match
the fresh oracle bitwise. Raw times, hashes, process commands, affinities,
pair order, and copied-suite receipts are kept in the new ignored output.
The result is exploratory; it does not alter production or establish the
tuple-selection cost as the dominant performance limit.
