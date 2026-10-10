# Julia and Fortran/OpenMP uniform LW comparison

This follow-up compares the unchanged standard Julia
`UniformLWAdvection` toy with an equivalent Fortran/OpenMP implementation.
It responds to the [Stage 2 finding](multithreading-stage2-report.md) that
strong scaling declined on the largest meshes. The comparison is a separate
experiment: Stage 2 used `(1024,32)` tiles, while both implementations here
use `(512,16)` tiles. Its timings therefore do not replace or merge with the
Stage 2 acceptance matrix.

**Result:** the poor large-grid strong scaling appears in both languages.
At 4096², each reaches its best median with four workers: 2.401 ns per
unknown per complete step in Julia and 2.397 ns in Fortran, with respective
four-worker efficiencies of 33.7% and 34.2%. The absolute times converge
on large meshes despite clear differences on smaller ones. The 50% target
at four or eight workers remains unmet at 2048² and 4096² in both codes.

## Numerical and timing contract

Both programs advance Float64 scalar cell averages on a periodic unit square
with fixed right and upper face speeds `u=0.7`, `v=-0.3`. One complete step is
three conservative Lax–Wendroff sweeps in the order `x(Δt/2)`, `y(Δt)`,
`x(Δt/2)`. Each sweep recomputes the same positively oriented face fluxes,
reads a separate source field, and writes each destination cell once. The
observable scalar field is updated after every complete step. Each program
keeps dense velocity arrays and two intermediate scalar arrays; neither uses
the experimental affine or final-state-only path.

The five active Float64 numerical grids occupy 10, 40, 160, and 640 MiB at
the four sizes, respectively. Both drivers also retain an initial-field
copy outside the timed arithmetic, so total process memory exceeds those
figures. Temporary probe buffers are released before timing.

The initial scalar field is generated once by Julia's analytic cell-average
fixture and supplied as the same binary Float64 field to both programs. The
time increment at an `Nₓ × Nᵧ` mesh is
`Δt = 0.35/(0.7Nₓ + 0.3Nᵧ)`. A mesh uses one frozen step count in every
language and worker configuration. The complete final fields are compared
outside timing with the Julia serial result, alongside mass and the final
clock. Worker participation is probed separately from the timed trajectory.

The primary sizes are `512²`, `1024²`, `2048²`, and `4096²`. Each has
measurements at 1, 2, 4, 8, 16, and 32 requested workers, with five fresh
processes per configuration. Tiles remain `(512,16)` at every size and count:
the smallest grid has 32 nonoverlapping tiles, so it can expose all 32
workers. Julia uses the standard recomputed-flux Polyester backend and
Fortran uses OpenMP; their scheduling mechanisms are part of the comparison.
At more than one worker, both synchronize after each sweep. One-worker
execution traverses each whole sweep serially.

The reported quantity is **wall-clock nanoseconds per scalar unknown per
complete timestep**:

```text
ns / (unknown · step) = 10⁹ × timed trajectory seconds / (Nₓ × Nᵧ × steps).
```

The timer includes all three sweeps and their synchronization on every step.
It excludes compilation, initialization, warmup, output, checks, and worker
probes. The absolute Julia/Fortran time ratio is `T_Julia/T_Fortran` for the
same size and worker count; a value above one means Fortran was faster in
that configuration. Each language's strong-scaling speedup is its own
one-worker median divided by its `p`-worker median. Efficiency divides that
speedup by `p`. Thirty-two workers occupy SMT siblings on the 16-physical-core
test machine; that point is not 32 physical cores.

The Fortran implementation passed its independent scalar-flux oracle on
one-cell, seam, odd-size, nondivisible-tile, negative-speed, and zero-speed
cases. A separate cross-language preflight checked a 512² grid and an odd
1031×257 grid at 32 workers: both programs used all 32 workers in all three
passes and produced matching complete fields. These checks complement the
per-sample full-field parity test in the primary matrix. The Fortran method's
smooth translation order was checked separately after the timing matrix,
using the identical frozen release source and binary. At `N=32,64,128`
and physical time `T=0.25`, RMS errors were `0.00198127`, `0.000499601`,
and `0.000125166`, giving successive observed orders `1.988` and `1.997`.
One- and four-worker final fields were identical. A bounds-checked debug
build also passed the independent scalar oracle; neither post-matrix check
changed the release kernel. The locally retained
convergence receipt (`docs/src/assets/julia-fortran-matched/fortran-validation/convergence.json`)
and debug test receipt (`docs/src/assets/julia-fortran-matched/fortran-validation/debug-oracle-receipt.json`)
record these results and the release hashes.

## Results

All 240 fresh-process samples completed: four meshes × six counts × two
languages × five repeats. The frozen step counts are 9,427, 1,728, 64, and
13 at 512², 1024², 2048², and 4096², respectively. Every timed sample
lasted at least 0.333 s. All 240 final fields were **bitwise identical**
to their mesh's Julia serial oracle, across both languages and all worker
counts. Every untimed sweep probe used every requested worker, including all
32 SMT workers; individual worker CPU affinity was verified for both
programs. Final mean scalar mass stayed within `2×10⁻¹²` of its unit
starting value.

The tables report medians of five independent processes. Brackets are 95%
bootstrap intervals for each median or the paired Julia/Fortran time ratio.
`ns` means nanoseconds per scalar unknown per **complete three-sweep step**.
The ratio is below one where Julia was faster. Speedup and efficiency each
use that language's own one-worker time at the same mesh and fixed layout;
the intervals for those derived metrics are in the
local summary (`docs/src/assets/julia-fortran-matched/summary.json`).

### 512² — 9,427 steps

| Threads | Julia ns (95% CI) | Fortran ns (95% CI) | Julia/Fortran (95% CI) | Speedup Julia / Fortran | Efficiency Julia / Fortran |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.564 [1.542, 1.625] | 1.655 [1.602, 1.932] | 0.945 [0.841, 0.972] | 1.00 / 1.00 | 100.0% / 100.0% |
| 2 | 0.810 [0.795, 0.841] | 0.882 [0.863, 0.910] | 0.919 [0.873, 0.948] | 1.93 / 1.88 | 96.5% / 93.8% |
| 4 | 0.426 [0.419, 0.433] | 0.467 [0.453, 0.497] | 0.913 [0.849, 0.956] | 3.67 / 3.55 | 91.7% / 88.6% |
| 8 | 0.248 [0.247, 0.251] | 0.277 [0.273, 0.282] | 0.896 [0.890, 0.912] | 6.31 / 5.98 | 78.9% / 74.8% |
| 16 | 0.179 [0.173, 0.184] | 0.221 [0.216, 0.232] | 0.812 [0.791, 0.829] | 8.73 / 7.50 | 54.6% / 46.9% |
| 32 | 0.220 [0.192, 0.274] | 0.287 [0.263, 0.348] | 0.769 [0.728, 0.804] | 7.10 / 5.77 | 22.2% / 18.0% |

### 1024² — 1,728 steps

| Threads | Julia ns (95% CI) | Fortran ns (95% CI) | Julia/Fortran (95% CI) | Speedup Julia / Fortran | Efficiency Julia / Fortran |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.141 [1.956, 2.409] | 2.017 [1.909, 2.236] | 1.061 [0.987, 1.224] | 1.00 / 1.00 | 100.0% / 100.0% |
| 2 | 1.074 [0.869, 1.133] | 0.960 [0.938, 1.000] | 1.119 [0.906, 1.169] | 1.99 / 2.10 | 99.7% / 105.1% |
| 4 | 0.571 [0.559, 0.581] | 0.490 [0.481, 0.549] | 1.166 [1.045, 1.170] | 3.75 / 4.12 | 93.7% / 103.0% |
| 8 | 0.296 [0.277, 0.321] | 0.273 [0.260, 0.299] | 1.083 [1.014, 1.183] | 7.23 / 7.38 | 90.4% / 92.2% |
| 16 | 0.206 [0.184, 0.211] | 0.201 [0.195, 0.209] | 1.027 [0.942, 1.037] | 10.39 / 10.05 | 64.9% / 62.8% |
| 32 | 0.213 [0.200, 0.481] | 0.268 [0.234, 0.309] | 0.795 [0.683, 2.030] | 10.05 / 7.53 | 31.4% / 23.5% |

### 2048² — 64 steps

| Threads | Julia ns (95% CI) | Fortran ns (95% CI) | Julia/Fortran (95% CI) | Speedup Julia / Fortran | Efficiency Julia / Fortran |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.059 [2.832, 3.125] | 3.415 [3.311, 3.569] | 0.896 [0.805, 0.926] | 1.00 / 1.00 | 100.0% / 100.0% |
| 2 | 2.184 [2.015, 2.318] | 2.388 [2.311, 2.442] | 0.915 [0.844, 0.978] | 1.40 / 1.43 | 70.0% / 71.5% |
| 4 | 1.892 [1.807, 1.965] | 1.924 [1.862, 1.961] | 0.983 [0.942, 1.016] | 1.62 / 1.78 | 40.4% / 44.4% |
| 8 | 1.861 [1.751, 1.993] | 1.905 [1.868, 1.927] | 0.977 [0.937, 1.034] | 1.64 / 1.79 | 20.5% / 22.4% |
| 16 | 2.029 [1.945, 2.180] | 2.000 [1.949, 2.024] | 1.015 [0.973, 1.091] | 1.51 / 1.71 | 9.4% / 10.7% |
| 32 | 2.390 [2.260, 2.808] | 2.447 [2.318, 3.212] | 0.977 [0.855, 1.123] | 1.28 / 1.40 | 4.0% / 4.4% |

### 4096² — 13 steps

| Threads | Julia ns (95% CI) | Fortran ns (95% CI) | Julia/Fortran (95% CI) | Speedup Julia / Fortran | Efficiency Julia / Fortran |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.239 [3.098, 3.322] | 3.281 [3.251, 3.300] | 0.987 [0.944, 1.015] | 1.00 / 1.00 | 100.0% / 100.0% |
| 2 | 2.572 [2.505, 2.634] | 2.534 [2.477, 2.546] | 1.015 [1.008, 1.034] | 1.26 / 1.29 | 63.0% / 64.7% |
| 4 | 2.401 [2.393, 2.622] | 2.397 [2.373, 2.456] | 1.002 [0.985, 1.105] | 1.35 / 1.37 | 33.7% / 34.2% |
| 8 | 2.484 [2.476, 2.589] | 2.497 [2.471, 2.563] | 0.995 [0.973, 1.048] | 1.30 / 1.31 | 16.3% / 16.4% |
| 16 | 2.609 [2.552, 2.651] | 2.621 [2.610, 2.665] | 0.996 [0.958, 1.016] | 1.24 / 1.25 | 7.8% / 7.8% |
| 32 | 2.779 [2.696, 2.987] | 2.772 [2.764, 3.074] | 1.002 [0.905, 1.081] | 1.17 / 1.18 | 3.6% / 3.7% |

The absolute time plots for 512² (`docs/src/assets/julia-fortran-matched/comparison-512.svg`),
1024² (`docs/src/assets/julia-fortran-matched/comparison-1024.svg`),
2048² (`docs/src/assets/julia-fortran-matched/comparison-2048.svg`), and
4096² (`docs/src/assets/julia-fortran-matched/comparison-4096.svg`) show both codes on
the same log-scaled ns/unknown/step axis at each mesh. Matching
speedup plots for 512² (`docs/src/assets/julia-fortran-matched/speedup-512.svg`),
1024² (`docs/src/assets/julia-fortran-matched/speedup-1024.svg`),
2048² (`docs/src/assets/julia-fortran-matched/speedup-2048.svg`), and
4096² (`docs/src/assets/julia-fortran-matched/speedup-4096.svg`) show each language
relative to its own serial time. The comparison CSV (`docs/src/assets/julia-fortran-matched/comparison.csv`)
contains every median and the summary JSON (`docs/src/assets/julia-fortran-matched/summary.json`)
contains all confidence intervals and validation maxima.

## Interpretation and limitations

The larger grid does not consistently favor stronger scaling. Both programs
scale well through 512² and 1024²: their fastest median is at 16 workers.
At 512², Julia reaches 0.179 and Fortran 0.221 ns/unknown/step at that
count. At 1024² they are nearly equal, 0.206 and 0.201. Efficiency at 16
workers improves from 54.6%/46.9% at 512² to 64.9%/62.8% at 1024².

At 2048², both flatten around four to eight workers: Julia's median drops
only from 1.892 to 1.861 ns/unknown/step from four to eight; Fortran's
from 1.924 to 1.905. Their best measured eight-worker efficiencies are
20.5% and 22.4%, respectively. Fortran's self-speedup is slightly larger
because its one-worker median is slower (3.415 versus Julia's 3.059 ns);
the two parallel absolute times overlap. At 4096², four workers give the
best median for both, 2.401 Julia and 2.397 Fortran ns/unknown/step.
Their self-speedups are 1.35 and 1.37, and efficiencies 33.7% and 34.2%;
more workers make both slower. At no mesh does the 32-worker SMT median beat
the same code's 16-worker median. The 1024² Julia 32-worker estimate has
substantial uncertainty, so that comparison is less precise.

The absolute comparison changes by mesh. Julia is faster throughout the
512² ladder, with Julia/Fortran median ratios from 0.945 at one worker to
0.769 at 32. At 1024², Fortran is faster at four and eight workers, but
the 32-worker ratio interval `[0.683, 2.030]` is too wide to rank the two.
One of the five Julia 32-worker trajectories was substantially slower than
its peers; it passed all validity checks and remains in the analysis.
At 2048², Julia has the faster serial median, while their four- and
eight-worker absolute times are close. At 4096², the ratios remain near one
at every count; the four-worker interval `[0.985, 1.105]` includes parity.
The shared large-grid plateau is therefore **not evidence of a Julia-only
scheduler problem**. It also does not prove that DRAM bandwidth alone is the
limit.

Larger meshes reduce the relative cost of starting each parallel sweep, but
their five-grid working set also moves from 10 MiB at 512² to 640 MiB at
4096². The host has two 32 MiB L3 domains and 16 physical cores. A change
in cache reuse or memory traffic could therefore offset the expected launch
amortization. These are explanations to test against both measured curves,
not an assumed bandwidth ceiling. The matched study changes the fixed tile
layout and uses a different Fortran runtime, so it isolates neither the
language implementation nor the memory system by itself.

Each mesh has a fixed step count but a different physical final time because
`Δt` changes with mesh spacing. Comparisons are within a mesh, at identical
step count and initial field. The curves do not measure convergence order.
The five-process intervals describe this host and run window, with turbo
boost enabled and the `powersave` governor; they are not uncertainty bounds
for other machines. A measured cache/memory traffic ceiling would require
hardware counters or an equivalent direct traffic measurement. The earlier
Stage 2 STREAM control is informative but does not establish such a ceiling
for these full LW timesteps.

The next useful diagnostic is actual cache and memory traffic of these
three-sweep kernels, together with their worker placement and active
working set. That evidence can distinguish a shared memory-system limit
from an implementation-specific cost more directly than another Julia-only
scheduler rewrite. The existing Stage 2 acceptance decision remains
unchanged because this follow-up does not meet the large-grid efficiency
target.

The [cache-cliff investigation](multithreading-cache-cliff-investigation.md)
tests the sharp 1024²-to-2048² transition with finer sizes, placement, and
traffic controls before any later-stage implementation is advanced.

## Reproduction

The run used Julia 1.10.12 with the unchanged standard toy source and GNU
Fortran 13.3.0. The Fortran release build uses
`-O3 -march=native -fopenmp -ffp-contract=off -fno-fast-math -Wall -Wextra -Werror`.
The machine is an AMD Ryzen 9 5950X with 16 physical cores, 32 SMT CPUs, two
32 MiB L3 domains, and 32 GB RAM. Workers were individually pinned in the
same spread-across-L3 CPU order at each count; Julia GC and BLAS were each
limited to one thread. Runs were serialized, with language order randomized
within each repeat and adjacent Julia/Fortran measurements for the same
mesh/count.
Neither power settings nor background user processes were changed for this
study. The recorded machine, cgroup, governor/boost, CPU order, dependency
versions, compiler flags, and source hashes are in the
configuration (`docs/src/assets/julia-fortran-matched/configuration.json`).

On the original host, the frozen workload can be rerun with a fresh output
directory using:

```sh
make -C research/UniformLWAdvection/fortran
julia --startup-file=no --project=benchmark/scaling benchmark/scaling/cross_language/study.jl \
  --julia /tmp/julia-1.10.12/bin/julia \
  --project /tmp/twophase-compat110 \
  --depot /tmp/twophase-julia110-depot \
  --fortran research/UniformLWAdvection/fortran/build/lw_omp \
  --workload benchmark/scaling/results/20260930-julia-fortran-matched/workload.json \
  --output benchmark/scaling/results/reproduction-20260930
```

The `/tmp` Julia project and depot are local resolved environments. Their
original Project and Manifest are preserved in the local source snapshot;
another machine must resolve a compatible environment and will have its own
timings. The command's fresh output path must not already exist.

The maintained implementation and runner are in
`research/UniformLWAdvection/fortran/` and
`benchmark/scaling/cross_language/`. The local ignored directory
`benchmark/scaling/results/20260930-julia-fortran-matched/` preserves the
serial initial/oracle fields and a byte-for-byte source and resolved-project
snapshot. Local raw samples (`docs/src/assets/julia-fortran-matched/raw.jsonl`),
frozen workload (`docs/src/assets/julia-fortran-matched/workload.json`),
field fingerprints (`docs/src/assets/julia-fortran-matched/fixtures.json`),
source hashes (`docs/src/assets/julia-fortran-matched/source_sha256.json`), and
asset manifest (`docs/src/assets/julia-fortran-matched/asset-manifest.json`) are
generated locally and ignored by Git. The benchmark's configuration SHA256 is
`2dc61d18bd61d3ca181bcccd1d5e551746cbca242c65b588d6e9f7bb9a05960f`.
After acquisition, the analysis script was updated only to export confidence
interval columns in the CSV. It regenerated the summary and figures from
the unchanged 240 raw samples; the summary records the analysis-script hash.
