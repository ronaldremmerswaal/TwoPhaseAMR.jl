# Checked narrow face-ID comparison

The control is the committed block-interior implementation at revision
`5f0081d9b2677ddc676c855f9cb5db4d218165dd`, extracted from Git into the
ignored `benchmark/scaling/results/20261010-stage3-narrow-ids-control/`
directory before candidate edits. Its advection SHA-256 is
`cecb8463cccada2e16d90f5245b506842d9cc90f6df9c380bb69fee673e4f115`.
The original ignored `freeze.json` (schema 1) records every extracted file
and the frozen numerical Julia 1.10.12 project, main driver, tooling Julia
environment, and executable. It is part of the archived cohort and must
not be overwritten.
The candidate may narrow only the persistent regular x/y face-ID arrays,
using checked `Int32` when face IDs fit and native `Int` otherwise. The
control and candidate must run the same public general `advect!` path.

The Julia 1.12.7 coordinator uses `benchmark/scaling/Project.toml` and its
manifest, with `JULIA_DEPOT_PATH=/tmp/twophase-scaling-tools-depot:/home/ronald/.julia`.
Numerical children use `/tmp/julia-1.10.12/bin/julia`, the copied frozen
`benchmark-project`, and `/tmp/twophase-julia110-depot:/home/ronald/.julia`.
Run one numerical job at a time and keep the candidate source fixed during
each cohort.

On a clean checkout, prepare that Julia 1.10.12 executable and numerical
depot, plus the Julia 1.12.7 tooling environment, first. The exact pinned
numerical [Project.toml](numerical_project/Project.toml) and
[Manifest.toml](numerical_project/Manifest.toml) are tracked here. Then
recreate the committed control in the canonical ignored location:

```sh
JULIA_DEPOT_PATH=/tmp/twophase-scaling-tools-depot:/home/ronald/.julia \
  julia --startup-file=no --project=benchmark/scaling \
  benchmark/scaling/stage3/narrow_ids/recreate_control.jl
```

The recreation script refuses an existing directory; use a fresh checkout
or a separate new directory for a verification smoke. It extracts `src`
and `test` from the exact committed revision, copies the pinned numerical
Project/Manifest, and copies the committed control `src` into the numerical
project for Julia's project layout. It checks the control source hash,
direct source identities, pinned environment hashes, Julia executable,
and a complete per-file manifest while writing and verifying a new schema-2 freeze.
The original schema-1 numerical project held a different historical copy
of `src`; it was not the measured method because the benchmark driver
explicitly loaded each variant through `source=`. Schema 2 records this
layout difference, so a clean-checkout reconstruction is reproducible
without claiming byte identity to the archived environment directory.
Run new cohorts into new ignored output paths; the archived receipts and
manifests remain historical evidence.

After the focused numerical tests pass, use a **new** ignored output
directory for a one-repeat calibration pilot. The default cases are 1024²
in one block, 2048² in one block, and 1024² in 8×8-cell blocks, each at
p1/p4/p8. The step file permits a distinct size-specific trajectory,
initially `{"1024": 256, "2048": 32}`. The pilot is diagnostic only:
it does not classify an efficiency target. If any candidate or control
timed interval is below 0.2 s, increase the step count for that size for
**both** sources before the accepted matrix.

```sh
julia --project=benchmark/scaling benchmark/scaling/stage3/narrow_ids/study.jl pilot \
  --steps-file benchmark/scaling/results/narrow-ids-pilot-steps.json \
  --output benchmark/scaling/results/narrow-ids-pilot
julia --project=benchmark/scaling benchmark/scaling/stage3/narrow_ids/summarize.jl \
  benchmark/scaling/results/narrow-ids-pilot
```

If correctness and the pilot justify the candidate, freeze the final source
and matched step file, then run five fresh repeats with a minimum 0.2 s
timed interval. The matrix has three cases × three worker counts × two
sources × five repeats = 90 accepted full-field samples in 30 sequential
Julia processes. The runner checks bitwise full-field equality against a
committed-control one-worker oracle, source and tooling hashes, reported
CPU affinity, and repeat IDs. Failed or partial launches remain as
receipts and are not accepted samples; resume revalidates all prior rows.

```sh
julia --project=benchmark/scaling benchmark/scaling/stage3/narrow_ids/study.jl matrix \
  --steps-file benchmark/scaling/results/narrow-ids-final-steps.json \
  --repeats 5 --min-seconds 0.2 \
  --output benchmark/scaling/results/narrow-ids-final
julia --project=benchmark/scaling benchmark/scaling/stage3/narrow_ids/summarize.jl \
  benchmark/scaling/results/narrow-ids-final
julia --project=benchmark/scaling benchmark/scaling/stage3/narrow_ids/plot.jl \
  benchmark/scaling/results/narrow-ids-final
```

Summaries report absolute median nanoseconds per scalar cell and complete
step, same-worker control/candidate ratios, and efficiency relative to the
strongest compatible serial median with 95% bootstrap intervals. Compare
only matched trajectories; the prior block-interior cohort is historical
and has separate source, setup, and step counts. `Base.summarysize` setup
checks describe deduplicated reachable Julia objects, not process RSS or
measured DRAM traffic. Direct face-ID array bytes can show the storage
effect separately.

For untimed worker participation and storage after a successful matrix,
make probes from the immutable control and final candidate snapshot. Each
copied probe records exact patched-source hashes and checks its complete
field against the clean oracle.

```sh
julia --project=benchmark/scaling benchmark/scaling/stage3/narrow_ids/make_worker_probe.jl \
  --source benchmark/scaling/results/20261010-stage3-narrow-ids-control \
  --output benchmark/scaling/results/narrow-ids-diagnostics/probes/baseline
julia --project=benchmark/scaling benchmark/scaling/stage3/narrow_ids/make_worker_probe.jl \
  --source benchmark/scaling/results/narrow-ids-final/current-source \
  --output benchmark/scaling/results/narrow-ids-diagnostics/probes/current
julia --project=benchmark/scaling benchmark/scaling/stage3/narrow_ids/large_worker_study.jl \
  benchmark/scaling/results/narrow-ids-final \
  benchmark/scaling/results/narrow-ids-diagnostics
julia --project=benchmark/scaling benchmark/scaling/stage3/narrow_ids/id_storage_study.jl \
  benchmark/scaling/results/narrow-ids-final \
  benchmark/scaling/results/narrow-ids-diagnostics/setup
```

The proportional diagnostics above cover only the 2048² one-block p4/p8
worker callbacks for each source and the three direct x/y ID payload and
deduplicated graph sizes. The original full block-interior participation
study remains the broader scheduler evidence; no new scaling claim follows
from the untimed probe.

Ordered-gather generated code is a separate read-only diagnostic of a
concrete Float64 caller reproducing the production gather loop body with
runtime-loaded order, plus the actual `advect!` entry. The representative
loop is not the exact threaded closure. Its output is evidence about
generated instructions, not a performance result or proof of a bottleneck.
A later complete-step comparison must isolate any
gather change from the narrow-ID candidate. CFL-inclusive timing likewise
requires its own cohort that charges `stable_timestep` inside every timed
step and validates a matched trajectory.

The code inspection script records optimized typed IR, LLVM, and native
code for a copied production gather loop body and the actual `advect!`
entry. It also records the runtime order histogram and exact loaded source
hash. Run it under the **numerical** Julia 1.10.12 project, separate from
the timed matrix:

```sh
JULIA_DEPOT_PATH=/tmp/twophase-julia110-depot:/home/ronald/.julia \
  /tmp/julia-1.10.12/bin/julia --startup-file=no \
  --project=benchmark/scaling/results/20261010-stage3-narrow-ids-control/benchmark-project \
  benchmark/scaling/stage3/narrow_ids/inspect_gather.jl \
  benchmark/scaling/results/20261010-stage3-narrow-ids-control \
  benchmark/scaling/results/narrow-ids-gather-ir-control
```
