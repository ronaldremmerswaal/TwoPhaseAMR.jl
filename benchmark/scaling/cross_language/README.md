# Julia and Fortran OpenMP toy comparison

`study.jl` runs the same periodic, constant-carrier, Float64, cell-average
Lax–Wendroff problem in the unchanged `UniformLWAdvection` Julia core and
`research/UniformLWAdvection/fortran/lw_omp.f90`. Each complete timestep is
`x(dt/2)`, `y(dt)`, `x(dt/2)` with synchronized sweeps and observable `q` at
every step. Both implementations read identical binary initial `q` values,
store dense face-speed arrays, recompute fluxes in three passes, and update
the final field in place. Their untimed worker probes precede timing.

The primary protocol uses square meshes 512, 1024, 2048, 4096, workers
1, 2, 4, 8, 16, 32, tile 512×16, and five independent processes per
configuration. A one-worker run traverses the entire domain contiguously;
parallel runs use 512×16 tiles. A 512² mesh has exactly 32 tiles. On the
16-core / 32-logical-CPU benchmark host, 32 workers use SMT. This layout is
different from the Stage 2 `1024×32` Julia layout; its times must be measured
fresh and must not be spliced into the older curve.

The Julia runner pins each language's workers to the same spread CPU order.
Julia and OpenMP report individual CPU placement and per-pass participation;
both are checked for every sample. OMP placement, dynamic scheduling,
wait policy, and thread count are set explicitly. Pilot timings at 4, 8,
16, and 32 workers in both languages choose one frozen step count per mesh.
Five-replicate paired Julia/Fortran jobs are balanced in language order and
randomized by repeat. The timer covers only full trajectories, never startup,
allocation, initialization, worker probes, output, or field comparison.

Every measured final field is compared cell by cell with a retained Julia
serial full-trajectory reference. The runner records each field SHA256, maximum
error, tolerance failures, clock, mass, source hashes, full command context,
and affinity evidence, then removes matching redundant field dumps. Divergent
dumps remain for diagnosis and abort the matrix. The exported metric is
`1e9 * trajectory_seconds / (nx*ny*steps)` nanoseconds per scalar unknown per
complete timestep. `summarize.jl` writes all 24 paired configurations to
JSON/CSV plus CairoMakie absolute-time and self-speedup plots with bootstrap
95% intervals.

From the repository root, after building the Fortran binary with
`make -C research/UniformLWAdvection/fortran`, run:

```sh
julia --startup-file=no --project=benchmark/scaling benchmark/scaling/cross_language/study.jl \
  --julia /tmp/julia-1.10.12/bin/julia \
  --project /tmp/twophase-compat110 \
  --depot /tmp/twophase-julia110-depot \
  --fortran research/UniformLWAdvection/fortran/build/lw_omp \
  --output benchmark/scaling/results/20260930-julia-fortran-matched
```

The runner uses the Julia tooling environment in
`benchmark/scaling/Project.toml` for JSON processing and CairoMakie plots.
Its `source-snapshot/`, configuration,
workload, fixture hashes, job order, raw JSONL, and summary files preserve
the exact comparison protocol. The temporary Julia project path resolves
Polyester dependencies; the toy's own source is loaded from this repository.

After a completed study, regenerate the summary, CSV, and plots from the
unchanged raw samples with:

```sh
julia --startup-file=no --project=benchmark/scaling benchmark/scaling/cross_language/summarize.jl \
  benchmark/scaling/results/20260930-julia-fortran-matched
julia --startup-file=no --project=benchmark/scaling benchmark/scaling/cross_language/export_assets.jl
```

The summary records the SHA256 of the postprocessing script. The measurement
configuration and `source-snapshot/` retain the numerical code and scripts
used during timing, even if the reporting script is later updated. The export
helper refreshes `docs/src/assets/julia-fortran-matched/asset-manifest.json`,
including the Fortran validation receipts.
