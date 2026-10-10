# Cache-preserving full-step tile experiment

`CachePipeline.jl` is an isolated constant-carrier prototype for the frozen
`UniformLWAdvection` toy. It computes one complete Strang step per output
tile: `x(dt/2)`, `y(dt)`, `x(dt/2)`. It evaluates the same positive-oriented
Float64 Lax–Wendroff faces as the toy and reads the same dense `u` and `v`
arrays. It does not change the package implementation.

The input `State.q` is frozen throughout the tile phase. Each logical chunk
owns a private first-x and second-y scratch pair, including the necessary
periodic halos. Tiles write disjoint portions of `PipelineWorkspace.next`.
After every tile finishes, a second parallel pass copies `next` into the
original `State.q` array. That copy is part of every timed `step!`; the
public `q` object and any aliases to it remain valid after each step. Both
passes use `Polyester.@batch per=thread minbatch=1`; no scratch is indexed by
the executing thread ID.

The first x intermediate spans the output tile plus one x and y cell of
halo. Its frozen input reads extend two x cells and one y cell beyond the
output. The y intermediate spans one x-cell halo, and the final x stage
writes only physical output cells. Each tile uses contiguous x loops through
its physical middle; only values crossing the periodic x seam use scalar
wrapped access. The final x stage peels output `i=1` and SIMD processes the
remaining cells. Face flux arithmetic and dense velocity reads match the
frozen toy.

```julia
using UniformLWAdvection
include("research/UniformLWAdvection/experiments/cache_pipeline/CachePipeline.jl")

s = UniformLWAdvection.State(2048,2048)
UniformLWAdvection.initialize!(s)
w = CachePipeline.PipelineWorkspace(s; tile=(128,32),
                                    batches_per_worker=4, threaded=true)
dt = UniformLWAdvection.stable_timestep(s)
CachePipeline.simulate!(s,w,dt,64) # includes tile work and copyback each step
```

Run validation with the compatible Julia environment:

```sh
JULIA_DEPOT_PATH=/tmp/twophase-julia110-depot:/home/ronald/.julia \
  /tmp/julia-1.10.12/bin/julia --project=/tmp/twophase-compat110 \
  --threads=4 research/UniformLWAdvection/experiments/cache_pipeline/validate.jl
```

Validation compares the complete field *exactly* against the frozen serial
recompute toy at every step, including 1-cell seams, odd and rectangular
grids, uneven tiles, and 1, 2, 3, 4, 7, 8, 16, and 32 steps. It checks public
`q` identity, rejected output aliasing, independent workspaces, mass, smooth
translation order, and warmed allocation. The implementation currently
supports only the toy's constant-carrier mode, and its workspace is exclusive
to one in-flight trajectory. Its scratch size is bounded by tile dimensions
and logical chunk count; the separate `next` array is a full grid.
Fresh validation processes at 1, 3, 8, and 32 Julia workers each passed
1,893 checks for this seam-peel version. Both compute and copyback used every
requested worker on the 1024² probe. Warmed allocation was 0 bytes per step
at one worker and 80 bytes at the other counts at both 128² and 512².

For an untimed participation check, call `track_workers!(w)` before one step,
then `active_workers(w,:compute)` and `active_workers(w,:copyback)`. The
validation script asserts both phases use every requested Julia worker on a
1024² grid. Leave tracking disabled for timed runs.

`timing_driver.jl` measures a fresh pinned Julia process in `toy`,
`pipeline`, or serial `oracle` mode. The timer covers the complete trajectory;
worker probes, allocation sampling, field comparison, and file I/O run
outside it. Each `pipeline` step includes the parallel copyback. Its report
records per-phase worker IDs, worker affinities, numerical array bytes,
allocation bytes, exact full-field comparison to the serial toy oracle, and
source hashes. `pilot.jl` prepares initial and oracle fields, runs paired
toy/pipeline jobs in randomized order, and writes configuration, raw records,
and medians to an ignored result directory. Launch the harness with the
`benchmark/scaling` tooling project; `--julia`, `--project`, and `--depot`
select the separate frozen environment for numerical child processes. For example:

```sh
julia --startup-file=no --project=benchmark/scaling research/UniformLWAdvection/experiments/cache_pipeline/pilot.jl \
  --output benchmark/scaling/results/cache_pipeline-pilot \
  --sizes 1024 2048 --workers 1 4 8 16 --repeats 1
```

The selected research candidate uses the frozen toy's `(512,16)` layout and
the pipeline's `(128,32)` layout. Further pipeline layouts can be supplied with
`--pipeline-tiles 128x32 256x32 512x16`; each comparison uses the same input,
step count, timestep, carrier, and serial oracle. The runner checks a source
hash manifest before every sample and refuses an existing output directory.
It snapshots source files and appends every init, oracle, and sample command,
CPU order, exit status, stdout, stderr, and process wall time to
`commands.jsonl` outside the trajectory timer. Failed launches retain unique
local receipts. For a final cohort, `--min-seconds 0.2` rejects any short
timed sample.

The exploratory seam-peel pilot at 2048² used three fresh pinned processes
per case, 32 full steps per process, and exact full-field comparison to one
serial toy oracle. Median nanoseconds per cell per complete step were:

| Workers | Frozen toy `(512,16)` | Pipeline `(128,32)` | Pipeline `(512,16)` |
| ---: | ---: | ---: | ---: |
| 1 | 3.166 | 3.438 | 3.087 |
| 4 | 1.875 | 1.307 | 1.667 |
| 8 | 1.977 | 1.160 | 1.206 |

At four and eight workers, `(128,32)` was 1.44× and 1.70× faster than the
same-worker toy median. Its one-worker median was slower than toy serial;
the `(512,16)` serial result is too close to toy serial to establish a win.
Earlier toy serial medians were 3.026 and 3.039 ns, and the fastest toy
serial sample in this pilot was 3.001 ns. There is no supported claim of a
serial improvement. Several 32-step samples lasted less than 0.2 seconds,
making this a tile-selection pilot rather than the final scaling study. No
memory-traffic counters were measured. This is research code, with no
production integration.

The five-repeat final results, bootstrap intervals, exact field and worker
checks, and the pre-timer crash receipt are summarized in [FINDINGS.md](FINDINGS.md).
Use `resume.jl --cohort DIR --failed-receipt FILE` only for a matching
signal-11 pre-timer failure receipt. Use
`summarize.jl --cohorts DIR ... --output NEW_DIR` to regenerate the final
summary and CairoMakie plots from accepted raw samples.
Raw records, launch logs, source snapshots, and plots are retained locally in
the ignored directories
`benchmark/scaling/results/cache_pipeline-final-1024-20260930`,
`benchmark/scaling/results/cache_pipeline-final-2048-20260930`,
`benchmark/scaling/results/cache_pipeline-final-4096-20260930`, and
`benchmark/scaling/results/cache_pipeline-final-analysis-20260930`. The
source snapshots in those directories retain the README as it stood during
the timed runs; this navigation paragraph was added afterward.

The final matrix can be reproduced in new, empty output directories with the
same local Julia binary, project, depot, pin placement, seed, and fixed
trajectory lengths:

```sh
julia --startup-file=no --project=benchmark/scaling research/UniformLWAdvection/experiments/cache_pipeline/pilot.jl \
  --output benchmark/scaling/results/cache-pipeline-rerun-1024 \
  --sizes 1024 --workers 1 2 4 8 16 32 --steps 1024:2048 \
  --pipeline-tiles 128x32 --toy-tile 512x16 --placement spread \
  --seed 30092027 --repeats 5 --min-seconds 0.2
julia --startup-file=no --project=benchmark/scaling research/UniformLWAdvection/experiments/cache_pipeline/pilot.jl \
  --output benchmark/scaling/results/cache-pipeline-rerun-2048 \
  --sizes 2048 --workers 1 2 4 8 16 32 --steps 2048:128 \
  --pipeline-tiles 128x32 --toy-tile 512x16 --placement spread \
  --seed 30092027 --repeats 5 --min-seconds 0.2
julia --startup-file=no --project=benchmark/scaling research/UniformLWAdvection/experiments/cache_pipeline/pilot.jl \
  --output benchmark/scaling/results/cache-pipeline-rerun-4096 \
  --sizes 4096 --workers 1 2 4 8 16 32 --steps 4096:32 \
  --pipeline-tiles 128x32 --toy-tile 512x16 --placement spread \
  --seed 30092027 --repeats 5 --min-seconds 0.2
```
