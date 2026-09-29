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
with that file; the runner prints the latter three. Do not commit machine-local
timing snapshots as package documentation. Stable allocation and compile-time
contracts belong in `test/performance/`, while this directory is for
exploratory comparisons and regression investigation.

For a refactor comparison, run the same command on the baseline and candidate
revisions in fresh checkouts on the same machine. Compare medians from the
BenchmarkTools suite and both latency-probe records; do not compare results
collected on different Julia versions, thread counts, or CPUs.

## Extending the suite

Add a benchmark only when it represents a current public workflow or a major
documented numerical layer. Put reusable deterministic fixtures in
`common.jl`, exclude construction with a BenchmarkTools `setup` block, use
`evals=1` for mutating operations, and add the benchmark to an existing group
unless a genuinely new architectural concern has several measurements.

The separate [strong-scaling runner](scaling/README.md) provides fresh-process
measurements, a frozen workload, topology/affinity metadata, and uncertainty
intervals for the staged multithreading study.
