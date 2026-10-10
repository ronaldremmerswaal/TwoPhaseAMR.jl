# Stage 3 Julia general-upwind measurements on uniform meshes

`main_driver.jl` loads either the frozen preimplementation production source or
the current main-package source. Its production workload is the public general
`AdvectionWorkspace` and `advect!` path on a periodic uniform `CellField` mesh.
The same exact Float64 cell averages from the Julia toy are scattered into
each block. The prescribed face carrier is `(0.7, -0.3)`, the fixed timestep is
`0.35/n`, and the timer covers every complete upwind step. Setup, fixed CFL
selection, pinning, probes, and field export are outside the timer. The
separate `profile` mode records a warmed CFL scan and Julia flat sampling
profile; it is a diagnostic, not an accepted timing sample.

`main_study.jl` serializes fresh Julia processes, pins each requested worker
to a physical-core spread order, retains command/source hashes and source
snapshots, creates a frozen one-worker upwind field oracle for each mesh/layout/
trajectory, and compares every complete output field cell by cell. It writes
generated receipts under ignored `benchmark/scaling/results/` directories.
Use distinct output directories for pilot and accepted cohorts. A source change
inside a cohort is rejected. Failed processes write a failure receipt and do
not contribute a timing row.

The general upwind algorithm is first-order forward Euler. The dense Julia
toy and the research-only main-package prototype use three-sweep
Lax–Wendroff. Their absolute cost can be shown in the same units, but toy LW
times are not a numerical-parity or speedup denominator for general upwind.
The original toy and independent cache-preserving pipeline study has its own
driver under `research/UniformLWAdvection/experiments/cache_pipeline/`.

The baseline snapshot is
`benchmark/scaling/results/20260930-stage3-production-baseline-freeze/`.
It is an ignored local artifact. Its 92 Julia source files match Git revision
`ace94c888012d7cea00cf9545046a5ce5794325b` byte for byte; recreate the
baseline source from that revision if the local snapshot is absent. The runner
currently expects Julia 1.10.12 at `/tmp/julia-1.10.12/bin/julia`, a resolved
project at the ignored `frozen-project/` path, and the depot path written in
`main_study.jl`. Copy the recorded `Project.toml`/`Manifest.toml` into that
project and instantiate with the recorded Julia binary before reproduction.
The initializer also reads the local `freeze.json` receipt for the declared
baseline Git revision and Julia executable; retain or recreate that receipt
alongside the frozen source and environment.
If the exact manifest is unavailable, resolve a separate environment and
record its new dependency hashes; that run is a new cohort, not a bitwise
reproduction of the frozen one. The result manifest records the source,
driver, project, Julia binary, and input hashes used in each cohort.

The current runner batches the six size/layout configurations per fresh
`(repeat, worker count, source variant)` Julia process. Every configuration
still receives five independent processes, each with its own field reset,
warmup, GC, timed complete trajectory, and full-field oracle check. Cases
inside one batch share a process and thus may have correlated thermal/cache
conditions; process launch order and per-batch case order are randomized and
retained. The final matrix rejects any timed interval shorter than 0.2 s.

With the local frozen source, Julia binary, resolved project, and depot in
place, create a **new** ignored result directory and run the fixed production
matrix from the repository root with:

```sh
stage3_matrix_dir=benchmark/scaling/results/stage3-upwind-rerun
julia --project=benchmark/scaling benchmark/scaling/stage3/prepare_matrix.jl \
  --output "$stage3_matrix_dir"
julia --project=benchmark/scaling benchmark/scaling/stage3/main_study.jl matrix \
  --variant both --sizes 512 1024 2048 --layouts both \
  --counts 1 4 8 --repeats 5 --min-seconds 0.2 \
  --steps-file "$stage3_matrix_dir/steps.json" \
  --output "$stage3_matrix_dir"
julia --project=benchmark/scaling benchmark/scaling/stage3/summarize.jl \
  "$stage3_matrix_dir"
julia --project=benchmark/scaling \
  benchmark/scaling/stage3/plot_summary.jl "$stage3_matrix_dir"
```

The fixed step counts in `steps.json` are 1024/128/16 for
512²/1024²/2048².
Each frozen/current pair uses the same initial binary field and step count.
The runner writes each launch receipt and accepted sample as it completes;
rerunning the same command resumes missing configurations only after verifying
the complete source manifest. Summary tables use the strongest compatible
one-worker production variant at each mesh/layout as the efficiency reference.
`plot_summary.jl` renders the accepted absolute-time and efficiency curves
with 95% bootstrap intervals to ignored SVG/PNG files in the output directory.
The auxiliary `make_worker_probe.jl`, `worker_probe_study.jl`,
`make_component_probe.jl`, `component_profile_study.jl`, and `setup_study.jl`
reproduce the copied-source participation, component, and resident-size
diagnostics described in `docs/src/multithreading-stage3-report.md`.
`recover_batch.jl` and `recover_mesh_setup.jl` retain the historical recovery
workflow for interrupted cohorts.

The 512/1024 diagnostic prototype LW pilots predate the production scope
correction; their ignored receipts are research evidence only. The one-off
1024 pilot include failure is retained as an unexplained startup failure and
is excluded from accepted timing.

The earlier `20260930-stage3-upwind-final-matrix` directory is an incomplete
diagnostic cohort. It has 42 accepted rows, then a Julia process aborted before
loading the project or entering the timer. One exact retry completed the process
and passed the numerical checks, but its 512² one-block, eight-worker interval
was 0.184 s, below the predeclared 0.2 s minimum. The recovery therefore
appended no rows. The complete cohort above uses a longer 512² trajectory;
no row from the earlier cohort is pooled into its analysis. The abort cause
is undetermined.

The replacement cohort retained a separate Julia SIGSEGV in frozen 2048²,
8×8-cell-block mesh construction at eight workers, before that case's timer.
Its first five partial fields/reports, stack trace, and launch receipt are
preserved locally. An isolated fresh process for that exact configuration
passed full-field and duration checks, followed by one exact failed-batch
command retry whose six fields also passed. The recovery script records both
launches and atomically appended those six validated rows. This failure was
in threaded halo-donor packing setup; its cause remains undetermined.
