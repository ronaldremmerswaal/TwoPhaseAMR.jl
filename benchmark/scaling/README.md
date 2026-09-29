# Strong-scaling experiments

The runner requires Python 3.11+; figure generation additionally requires Matplotlib.

This is the Stage 1 measurement runner for the [threading plan](../../docs/src/multithreading-optimization-plan.md). It uses one fresh Julia process per independent observation. The Julia driver compiles and warms each path, restores initial state outside timed trajectories, keeps GC enabled, fixes BLAS and GC workers at one, and times a complete synchronized trajectory. The Python runner randomizes process order within each repeat, stores raw JSONL plus a configuration and frozen workload, and computes median time, median absolute deviation, and a 95% bootstrap interval for self-speedup. Five independent process repeats are required for a reported finalist. The 95% interval does not turn a tiny 128² smoke run into performance evidence.

Use the direct installed Julia binary if `julia` resolves to a launcher that might update between runs. The runner defaults to 1, 2, 4, 8, and 16 workers and 512², 1024², 2048², and 4096² for controls and the standalone toy. The declared primary Stage 2 gate is at least 50% efficiency at **4 or 8** workers on each mandatory 512², 2048², and 4096² toy fixture, with both counts and their confidence intervals reported. 1024² is the cache-transition diagnostic. 128² and 256² measure the small-grid overhead and serial crossover; the current toy threshold is a strict `<16384` cells, so 128² is not automatically serial at multiple workers. Sixteen is diagnostic, and 24/32 test SMT separately; do not reinterpret 32 logical threads as 32 physical cores. Fix a numerical method, backend, flux policy, tile geometry, source revision, workload, and placement across a scaling curve. Only the worker count changes. The toy uses Float64 constant speeds `(0.7,-0.3)`, a fixed `0.35/(0.7nx+0.3ny)` step, smooth analytic cell-average translation, and a Strang directional LW step. Its variable-carrier extension is a separate numerical comparison.

## Baseline and controls

The starting revision can be run from a source snapshot containing its `src/`, `Project.toml`, and `Manifest.toml`; set `--project` to that snapshot. The `configuration.json` source hash identifies uncommitted code, while `project_revision` records Git HEAD. The runner uses `--threads=p,0` on Julia 1.12 onward and the compatible `--threads=p` form on Julia 1.10/1.11, then checks the reported default-pool count. This keeps the interactive pool at zero where the runtime accepts explicit zero. The Julia driver records binary/version, resolved-manifest hash, compute/interactive/GC thread counts, BLAS configuration, CPU affinity, active worker IDs, allocations, GC time, working set, diagnostics, first-call latency, and Linux peak RSS. The Python runner records CPU topology, RAM, OS, governor/boost state, and placement. A process source-hash check aborts a run if source files change while collecting samples. Machine data live in ignored `benchmark/scaling/results/` by default.

```sh
python3 benchmark/scaling/run.py --smoke --julia /path/to/julia
python3 benchmark/scaling/run.py --project /path/to/starting-snapshot \
  --julia /path/to/julia --cases stream stencil gather launch \
  --sizes 512 1024 2048 4096 --threads 1 2 4 8 16 \
  --repeats 5 --seconds 0.25 --placement ordinary
```

`stream` counts logical loads plus a store (24 bytes/cell), `stencil` tests a periodic five-point gather (48 logical bytes/cell), and `gather` uses a fixed random permutation with three source reads (40 logical bytes/cell); these are traffic estimates, not hardware-counter measurements. The gather control is a synthetic proxy for irregular AMR access, not an AMR schedule. `launch` measures an empty synchronized traversal of a fixed 128 items by default (`--launch-items` changes that fixed work). The `worker_probe_items` distribution reports work-item imbalance in an **untimed** probe; it does not measure busy/idle wall time. Active worker IDs for the package use its scheduler, while toy IDs come from an instrumented actual transport step. An active count below the requested count must be reported and explained before making a scaling claim.

`--placement ordinary` inherits the OS affinity mask. `compact` chooses physical cores in one L3 domain first, and `spread` cycles L3 domains; both derive CPU/core/L3 mappings from sysfs. The harness sets a process affinity mask and records it. `--pin-workers` additionally binds each default-pool Julia worker to one unique selected Linux CPU using `sched_setaffinity` inside a static worker loop, then records and verifies that worker's own affinity and current CPU. Use `--placement compact` or `spread` with this option. Pinned and unpinned measurements are separate curves; pinning is an explicit harness setting, never a library side effect. `--profile` saves separate aggregate and per-thread flat Julia `Profile` tables for an untimed copy of each full trajectory; use sample durations long enough to collect stacks. Those per-thread stacks give actual kernel participation evidence, distinct from the scheduler work-item probe, but sampled counts are not wall-time busy fractions.

## Toy screen and finalists

Run one bounded screen at 1024² with two process repeats per configuration. `screen.py` freezes the first run's `workload.json` and passes it to every later policy, so all policies cover the same physical duration. It screens `threads`, `spawn`, and `polyester` backends; `recompute` and `stored` face fluxes; and tile shapes `(128,32)` and `(1024,32)`. Two `serial` flux controls establish absolute serial time. The twelve parallel configurations use 1, 4, and 8 workers. A screen ranks plausible finalists; it cannot pass a performance gate.

```sh
python3 benchmark/scaling/screen.py --julia /path/to/julia \
  --project /path/to/starting-snapshot \
  --output benchmark/scaling/results/toy-screen
```

Compare the time at 4 and 8, mass and translation errors, one-worker slowdown, and the actual active workers. Select at most two viable parallel policies; if one is within timing noise of a simpler one, keep the simpler one. Then run each finalist with `--sizes 128 256 512 1024 2048 4096 --threads 1 2 4 8 16 --repeats 5 --seconds 0.5` and the **same** frozen workload across policies. Run 24/32 as separate exploratory curves with that workload. For every fixture publish all counts, errors, memory, and confidence intervals; use 512/2048/4096 for the declared gate. Inspect actual `time_seconds`; if a sample is below 0.2 s, freeze a longer trajectory before acceptance. The 4096² toy has five to six Float64 grids (at least 640–768 MiB) plus scheduling and runtime storage, so verify available RAM before launching all repeats.

For odd and nondivisible partition stress, run `--cases toy --sizes --shapes 257x193 --threads 1 3 7 --steps 2 --repeats 1`; this is a numerical/worker-participation check, not a timed scaling gate.

The `--steps N` option freezes one step count directly. `--workload path/to/workload.json` reuses a prior count per case and size across algorithms and revisions. Never compare curves with different `steps`, `dt`, or numerical contract. A backend may use different thread-count-dependent partitioning, but its algorithm settings stay fixed. `--threshold`, `--tile-x`, and `--tile` are explicit in `configuration.json`. `--backend polyester` loads the toy's optional Polyester extension, which requests `per=thread` so the worker probe can expose SMT usage.

## Main-package upwind baseline

The package baseline uses the actual `BlockTreeMesh`, `CellField`, `FaceVelocity`, `AdvectionWorkspace`, and `advect!` public workflow. It is **first-order upwind/forward Euler**, so never treat it as the toy's LW solution. `upwind` times fixed-`dt` advection; `upwind_complete` also evaluates the stable timestep each step. Both start each independent trajectory from copied cell fields and report mass and analytic translation error. They exclude velocity changes, AMR, projection, and output. `--block 32` provides many small roots; `--block 512` with 512² gives one root. Run each topology as a separate configuration. The default large ladder is for controls/toy; upwind above 1024² requires `--allow-large-upwind` and a memory preflight because face/incidence schedules are much larger than dense toy arrays.

```sh
python3 benchmark/scaling/run.py --julia /path/to/julia \
  --project /path/to/starting-snapshot --cases upwind upwind_complete \
  --sizes 128 256 512 1024 --block 32 --threads 1 2 4 8 16 \
  --repeats 5 --seconds 0.25
```

Reuse the baseline workload for a candidate revision and preserve the same block size and numerical contract. For one-root stress cases run 512² separately with `--block 512`. Different block sizes have different topology and storage; report them as distinct fixtures rather than merging them into one scaling curve. Use `--profile` at 1 and the best 4/8 count and optionally 16/32. Whole-flow timings and component fractions belong to later stage fixtures; this upwind driver does not imply whole-simulation speedup.

## Analysis

```sh
python3 benchmark/scaling/plot.py benchmark/scaling/results/NAME
python3 benchmark/scaling/compare.py benchmark/scaling/results/BASELINE benchmark/scaling/results/CANDIDATE
```

The plotter writes SVG and PNG wall-time, speedup, efficiency, work-item imbalance, throughput/working-set, and timing/error figures beside `summary.json`. `compare.py` checks equal steps, final time, numerical contract, precision, Julia version, BLAS count, fixture cell count, and placement before computing self-speedup, speedup against the fastest measured serial run, and speedup against the starting revision at one worker. Review numerical tests and error budgets separately before calling a serial implementation *validated*. The runner's mass and control-oracle checks catch gross failures; they do not replace the toy's independent serial reference tests or package integration suite. Profile fractions, worker busy time, solver iterations/residuals, and complete-flow plots require their later-stage drivers rather than guessed data.

## Preparatory uniform WY transport fixture

`wy_complete` follows the existing closed-wall vortex-reversal integration
contract at fixed physical time `T=1`. It creates a circular interface,
nonconstant liquid and gas `PhaseScalar` fields, and a second phase scalar that
is exactly one in both phases. The face carrier is sampled from the discrete
streamfunction at the peak for each step's WY CFL calculation and at the
midpoint for transport; its `cospi(t)` factor reverses the flow. The full
trajectory uses the same precomputed peak-CFL step sequence at every worker
count, including the last partial step. It alternates sweep order and uses
`:lhf` reconstruction with `Fromm()` centered phase transport. The timed path
includes carrier sampling, CFL, reconstruction, WY, phase scalar updates, and
any requested inline diagnostics. Restoring all initial fields, invalidating
derived interface geometry, and creating a fresh WY workspace happen outside
timing before each independent run.

```sh
python3 benchmark/scaling/run.py --julia /path/to/julia \
  --project /path/to/starting-snapshot --cases wy_complete \
  --sizes 64 128 256 --block 32 --threads 1 2 4 8 \
  --repeats 5 --diagnostics summary
```

`--diagnostics summary` reports volume, liquid and gas amounts, the constant-one
identity, fraction bounds, intrinsic extrema, and reversal errors after the
timed trajectory. `--diagnostics sampled --diagnostic-interval 10` additionally
integrates liquid volume every ten steps *inside* the timed trajectory; use a
separate curve and compare only runs with the same mode/interval. The fixture
reports the actual prescribed step count, peak CFL step, and last partial
step. `--block 32` gives many roots and `--block N` for an `N×N` case gives one
root. The suggested 64²/128²/256² ladder is preparatory and subject to a
memory and runtime preflight; values above 256² require `--allow-large-wy`.
This is Stage 3 measurement preparation. Stage 3 performance acceptance waits
for the toy numerical and scaling gates, and no result from this fixture alone
establishes whole-flow speedup.

`wy_kernel` is an optional one-step diagnostic using the same initial fields,
LHF reconstruction, Fromm phase scalar transport, and midpoint carrier as
`wy_complete`. Its fixed peak-CFL step is timed with WY and scalar updates;
carrier sampling and CFL calculation happen during reset outside timing. It
runs exactly one step per independent process, so sub-0.2-second samples are
reported only as kernel diagnostics and cannot pass a complete-trajectory
scaling gate.
