# Testing and validation

The test tree has three suites: unit tests, integration tests, and performance
contracts. Machine-dependent benchmarks remain outside the test tree.

| Area | Location | Purpose |
| --- | --- | --- |
| unit | `test/runtests.jl` and its component folders | isolated contracts, numerical invariants, and short example smoke tests |
| integration | `test/integration/runtests.jl` | refinement, manufactured-solution, and literature-comparison studies |
| performance contracts | `test/performance` | allocation and hot-kernel regression checks |
| benchmarks | `benchmark` | machine-local timing and allocation measurements |

Run the standard suite with:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

Run the complete integration suite:

```sh
julia --project=. --threads=4 test/integration/runtests.jl
```

CI runs the same integration files as independent jobs so that long numerical
studies execute in parallel. The command above remains the complete local entry
point.

Run performance contracts separately with:

```sh
julia --project=. test/performance/runtests.jl
```

All three runners enable Julia's verbose test timing, so their summaries show
the elapsed time of each component testset without a separate profiling mode.

The single-phase cylinder case is part of the integration suite despite its
long statistically meaningful time window. The benchmark environment and
reproducibility controls are documented in `benchmark/README.md`. Optional
cross-package reference cases are documented in
`test/reference-validation.md` and are not normal package dependencies.

When adding an equation system or model specialization, add focused
constructor/dispatch tests, a regression for every numerical invariant at
risk, and a small public-API example. Keep case-specific factories beside the
study; only broadly reusable test utilities belong in `test/test_utils.jl`.
