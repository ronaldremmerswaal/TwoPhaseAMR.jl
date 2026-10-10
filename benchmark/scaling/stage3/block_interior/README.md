# Block-interior general-upwind study

The baseline is the ignored `benchmark/scaling/results/20261010-stage3-block-interior-baseline/`
snapshot of the **previous general remap**. Both variants use the same Julia
1.10.12 environment and the public `advect!` path in `../main_driver.jl`.
Each sample times complete forward-Euler upwind steps for one scalar cell
unknown; setup, the fixed CFL scan, pinning, and field export are outside the
timer. The runner verifies the full Float64 field against a frozen one-worker
upwind oracle, source hashes, CPU affinity, and a minimum 0.2-second interval
for accepted rows.

Run a new one-repeat pilot after the implementation is frozen and the CPU is
available. The default cases are 1024² in one block, 2048² in one block, and
1024² in 8×8-cell blocks; worker counts are 1, 4, and 8. Pilot timings only
choose fixed step counts that exceed 0.2 seconds in the accepted run.
The coordinator uses native Julia 1.12.7 and the explicit
`benchmark/scaling/Project.toml`/`Manifest.toml` tooling environment; its
numerical children use frozen Julia 1.10.12 and the frozen benchmark project.
For this machine, set
`JULIA_DEPOT_PATH=/tmp/twophase-scaling-tools-depot:/home/ronald/.julia`
and run the native binary at
`/home/ronald/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia`.

```sh
julia --project=benchmark/scaling benchmark/scaling/stage3/block_interior/study.jl pilot \
  --pilot-steps 16 \
  --output benchmark/scaling/results/block-interior-pilot-1
julia --project=benchmark/scaling benchmark/scaling/stage3/block_interior/summarize.jl \
  benchmark/scaling/results/block-interior-pilot-1
```

After selecting matched fixed steps, create a JSON file mapping sizes to
steps, such as `{"1024": 256, "2048": 32}`. Use a **new** output directory
for the accepted five-repeat matrix:

```sh
julia --project=benchmark/scaling benchmark/scaling/stage3/block_interior/study.jl matrix \
  --steps-file benchmark/scaling/results/block-interior-steps.json \
  --repeats 5 --min-seconds 0.2 \
  --output benchmark/scaling/results/block-interior-final
julia --project=benchmark/scaling benchmark/scaling/stage3/block_interior/summarize.jl \
  benchmark/scaling/results/block-interior-final
julia --project=benchmark/scaling benchmark/scaling/stage3/block_interior/plot.jl \
  benchmark/scaling/results/block-interior-final
```

Each fresh Julia process batches the three cases for one variant, worker
count, and repeat. Cases within a process can share machine conditions.
Process and case order are randomized; command receipts preserve the order.
Generated outputs live under ignored `benchmark/scaling/results/`. The runner
refuses to resume a cohort after source, driver, Julia binary, or environment
hash changes. Failed launches and incomplete fields stay available for audit
but do not become accepted rows.

For untimed participation, generate probe copies from the frozen sources:

```sh
julia --project=benchmark/scaling benchmark/scaling/stage3/block_interior/make_worker_probe.jl \
  --source benchmark/scaling/results/20261010-stage3-block-interior-baseline \
  --output benchmark/scaling/results/block-interior-diagnostics/probes/baseline
julia --project=benchmark/scaling benchmark/scaling/stage3/block_interior/make_worker_probe.jl \
  --source benchmark/scaling/results/block-interior-final/current-source \
  --output benchmark/scaling/results/block-interior-diagnostics/probes/current
julia --project=benchmark/scaling benchmark/scaling/stage3/block_interior/worker_probe_study.jl \
  benchmark/scaling/results/block-interior-final \
  benchmark/scaling/results/block-interior-diagnostics
julia --project=benchmark/scaling benchmark/scaling/stage3/block_interior/setup_study.jl \
  benchmark/scaling/results/block-interior-final \
  benchmark/scaling/results/block-interior-diagnostics/setup
```

The probe records which requested worker IDs actually enter each pass.
`Base.summarysize` diagnostics describe the deduplicated reachable Julia
object graph; they are not process RSS or measured memory traffic.

For the bounded 2048², one-block component diagnostic, create separate
copied-source probes and point the profile runner at the completed matrix.
It reads that matrix's fixed step count, input, and clean oracle, then runs
four diagnostic processes at four and eight workers. Its flux bracket
includes both residual and regular owned faces.

```sh
julia --project=benchmark/scaling benchmark/scaling/stage3/make_component_probe.jl \
  --source benchmark/scaling/results/20261010-stage3-block-interior-baseline \
  --output benchmark/scaling/results/block-interior-diagnostics/component/baseline
julia --project=benchmark/scaling benchmark/scaling/stage3/make_component_probe.jl \
  --source benchmark/scaling/results/block-interior-final/current-source \
  --output benchmark/scaling/results/block-interior-diagnostics/component/current
julia --project=benchmark/scaling benchmark/scaling/stage3/component_profile_study.jl \
  benchmark/scaling/results/block-interior-final \
  benchmark/scaling/results/block-interior-diagnostics/component
```

The separate 128×8 copied-source tile study is exploratory. Its default
2048² one-block case uses 128 matched steps and three paired repeats per
worker count; it verifies the full field against a fresh oracle and records
the exact copied-source patch. Use a new ignored output directory:

```sh
julia --project=benchmark/scaling benchmark/scaling/stage3/block_interior/flux_tile_study.jl \
  --cohort benchmark/scaling/results/block-interior-final \
  --output benchmark/scaling/results/block-interior-tile-diagnostic
```
