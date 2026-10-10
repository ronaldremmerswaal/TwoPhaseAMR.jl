# Cache-cliff follow-up

This study investigates why the matched Julia and Fortran LW toy codes rise
from about 0.2 ns/unknown/complete step at 1024²/p16 to about 2 ns at
2048²/p16. It keeps the same unmodified kernels, Float64 initial field,
`x(dt/2), y(dt), x(dt/2)` trajectory, tile 512×16, and per-worker CPU pinning
as the [cross-language study](../cross_language/README.md). The three sweeps
are synchronized, and every measured final field is compared with a Julia
serial trajectory for the same mesh and step count.

Run the phases separately, with no concurrent CPU benchmarks:

```sh
julia --startup-file=no --project=benchmark/scaling benchmark/scaling/cache_cliff/cache_study.jl --phase ladder \
  --output benchmark/scaling/results/20260930-cache-cliff/ladder
julia --startup-file=no --project=benchmark/scaling benchmark/scaling/cache_cliff/cache_study.jl --phase placement \
  --workload-from benchmark/scaling/results/20260930-cache-cliff/ladder/workload.json \
  --output benchmark/scaling/results/20260930-cache-cliff/placement
julia --startup-file=no --project=benchmark/scaling benchmark/scaling/cache_cliff/cache_study.jl --phase placement \
  --placement-sizes 768 896 960 \
  --workload-from benchmark/scaling/results/20260930-cache-cliff/ladder/workload.json \
  --output benchmark/scaling/results/20260930-cache-cliff/placement-low
julia --startup-file=no --project=benchmark/scaling benchmark/scaling/cache_cliff/cache_study.jl --phase steps \
  --output benchmark/scaling/results/20260930-cache-cliff/steps
julia --startup-file=no --project=benchmark/scaling benchmark/scaling/cache_cliff/export.jl
```

The fine ladder uses p16 across two 32 MiB L3 domains and three fresh paired
Julia/Fortran process repeats at each of 11 meshes. The placement phases use
p8 on one L3 domain (`compact`) or spread across two domains, with the same
frozen step count per mesh. The short-step phase compares 64 and 256 complete
steps at 1024² and 2048² with five fresh process repeats; its trajectories
may be shorter than 0.2 s and are explicitly diagnostic. Normalized timing
always divides the timed trajectory by `nx*ny*steps`.

Raw JSONL, configuration, source snapshots, serialized job order, binary
initial/oracle fields, summaries, and plots reside in the ignored result
directory. The exporter copies compact evidence to docs assets and hashes
each file. Source snapshots capture the exact script revision for each phase;
later diagnostic script edits do not rewrite earlier raw results.

The separate Fortran diagnostic clone adds `--pad 0|8|32` to change the
physical leading dimension while preserving logical `N×N` input/output, and
`--counters 0|1|2` for off, demand-source events, and hardware-prefetch-source
events. Mode 1 measures Linux hardware cycles/instructions/generic cache
misses plus Zen 3 raw demand local-memory/IO and internal-cache fills. Mode 2
uses the same generic events plus Zen 3 hardware-prefetch local-memory/IO and
internal-cache fills. The raw AMD event selectors are recorded in each sample;
the complete counter-region runs around the full trajectory on every pinned
worker. Per-worker `value`, `time_enabled_ns`, and `time_running_ns` are kept
in raw JSONL. Each worker's count is scaled by enabled/running time, summed,
and divided by `N² × steps`. A sample requires at least 99% event running
time and an exact final-field SHA match with its serial oracle. These counts
indicate the source of cache fills; they are not measured DRAM bytes. Generic
cache misses are not treated as LLC misses.

The controlled diagnostics run with the frozen ladder/placement step counts:

Build the separate clone first with
`make -C research/UniformLWAdvection/fortran/diagnostics`. Its
[diagnostic README](../../../research/UniformLWAdvection/fortran/diagnostics/README.md)
documents padded storage and the AMD event selectors.

```sh
julia --startup-file=no --project=benchmark/scaling benchmark/scaling/cache_cliff/fortran_controls.jl --mode pmu \
  --output benchmark/scaling/results/20260930-cache-cliff/pmu
julia --startup-file=no --project=benchmark/scaling benchmark/scaling/cache_cliff/fortran_controls.jl --mode padding \
  --output benchmark/scaling/results/20260930-cache-cliff/padding
julia --startup-file=no --project=benchmark/scaling benchmark/scaling/cache_cliff/plot_controls.jl \
  benchmark/scaling/results/20260930-cache-cliff
julia --startup-file=no --project=benchmark/scaling benchmark/scaling/cache_cliff/export.jl
```

PMU-on timing is a separate instrumented cohort and is never merged into the
unmodified Julia/Fortran timing curves. The padding comparison uses the
counter-off diagnostic clone for all three padding values.

Each study command requires a new `--output` directory; existing results are
never overwritten. The shown Julia executable, resolved project, depot, and
default result root reflect this benchmark machine. Set `--julia`, `--project`,
`--depot`, `--binary`, and `--result-root` as needed on another host.
