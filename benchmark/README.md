# Benchmarks

This suite measures the current package architecture instead of preserving a
chronology of past optimization experiments. It uses reproducible fixtures,
the typed public API, one BLAS thread, and BenchmarkTools setup blocks so timed
samples exclude fixture construction unless setup itself is the quantity under
test.

The flat suite has five user-oriented groups:

- `setup`: construction of single-phase, one-velocity, and two-velocity flows;
- `stepping`: complete fixed- and mixed-mesh simulation steps, including a
  representative diagnostics configuration;
- `mesh and interface` / `adaptivity`: topology construction, level-set/PLIC
  work, interface-height evaluation, and refinement with field transfer;
- `interface transport`: one-velocity WY and both typed two-velocity transport
  choices on uniform and mixed meshes;
- `linear solvers` / `two-velocity projection`: Poisson application/solve and
  CMDGFM versus mimetic projection.

A separate `latency.jl` probe reports package load, first flow construction,
first-step, and warmed second-step time and allocations. It deliberately runs
in a fresh Julia process because compilation latency cannot be measured inside
an already-loaded BenchmarkTools suite.

## Setup

Create the dedicated environment once from the repository root:

```sh
julia --project=benchmark -e '
    using Pkg
    Pkg.develop(path=".")
    Pkg.instantiate()
'
```

Run the complete suite with:

```sh
julia --project=benchmark --threads=auto benchmark/benchmarks.jl
```

Run the fresh-process latency probe with:

```sh
julia --project=benchmark benchmark/latency.jl
```

For a quick smoke run:

```sh
TWOPHASE_BENCHMARK_SAMPLES=1 \
TWOPHASE_BENCHMARK_SECONDS=0.05 \
julia --project=benchmark benchmark/benchmarks.jl
```

`TWOPHASE_BENCHMARK_OUTPUT=/path/to/results.json` saves raw BenchmarkTools
results. Record the Git revision, Julia version, thread count, and CPU together
with that file; the runner prints the latter three. The documented latency
investigation below is an explicitly identified machine-local snapshot, not a
portable performance contract. Stable allocation and compile-time contracts
belong in `test/performance/`, while this directory is for exploratory
comparisons and regression investigation.

For a refactor comparison, run the same command on the baseline and candidate
revisions in fresh checkouts on the same machine. Compare medians from the
BenchmarkTools suite and both latency-probe records; do not compare results
collected on different Julia versions, thread counts, or CPUs.

## Numerical regression oracle

From the repository root, with the benchmark environment set up as above:

```sh
julia --heap-size-hint=3G -t 1 --project=benchmark benchmark/oracle/record.jl
julia --heap-size-hint=3G -t 1 --project=benchmark benchmark/oracle/check.jl
julia --heap-size-hint=3G -t 1 --project=benchmark benchmark/oracle/perf.jl
```

`record.jl` replaces the committed numerical reference. `check.jl` uses
bitwise SHA-256 comparisons when Julia version and thread count match, and
otherwise compares field norms and final diagnostics with tolerance `1e-10`.
Override with `--mode=bitwise` or `--mode=tolerance` and `--tol=1e-10`.
The tolerance comparison uses this value as both relative and absolute
tolerance. Set `ORACLE_PERTURB=1` to exercise the failure path.

The oracle captures owned interior faces and cell interiors; inactive
two-velocity NaNs participate in the hash but not in the norms. The
performance probe records first-run latency after package/example loading and
the mean time and allocated bytes of ten warm steps in
`oracle/perf_baseline.toml`; compare only on the same machine, Julia version,
and thread count. These snapshots are refactor reference points, not portable
performance guarantees. Re-record only when intentionally changing the
numerical reference or measurement environment.
The bounded 64×64 lid-driven cavity uses a fixed `dt=0.001`, five warm-up
steps, and ten measured steps; its `warm_lid_cavity_64` entry tracks
non-periodic boundary-convection allocation separately from the periodic case.
The Taylor–Green measurement precedes Stokes in the same fresh process, so
Stokes can reuse compilation performed by the Taylor–Green measurement.

## Latency breakdown

Fresh-process probe of where the first run of an example spends its time:

```sh
julia --heap-size-hint=3G -t 1 --project=benchmark benchmark/latency_breakdown.jl stokes3_wave
```

The full investigation (phase tables, inference and invalidation analysis,
PrecompileTools experiment, follow-up proposals) is in
[`latency/README.md`](latency/README.md).

## Extending the suite

Add a benchmark only when it represents a current public workflow or a major
documented numerical layer. Put reusable deterministic fixtures in
`common.jl`, exclude construction with a BenchmarkTools `setup` block, use
`evals=1` for mutating operations, and add the benchmark to an existing group
unless a genuinely new architectural concern has several measurements.
