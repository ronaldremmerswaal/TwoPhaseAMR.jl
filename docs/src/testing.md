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

Run the complete integration suite, or select one or more named cases:

```sh
julia --project=. --threads=4 test/integration/runtests.jl
julia --project=. --threads=4 test/integration/runtests.jl two-phase-waves
julia --project=. --threads=4 test/integration/runtests.jl single-phase-cylinder
```

Run performance contracts separately with:

```sh
julia --project=. test/performance/runtests.jl
```

The single-phase cylinder case is deliberately part of the integration suite,
but is omitted from routine CI because its statistically meaningful default
time window is expensive. The benchmark environment and reproducibility
controls are documented in `benchmark/README.md`. Optional cross-package
reference cases, including the separately selectable `mimeticfv-reference`
case, are documented in `test/reference-validation.md` and are not normal
package dependencies.

When adding an equation system or model specialization, add focused
constructor/dispatch tests, a regression for every numerical invariant at
risk, and a small public-API example. Keep case-specific factories beside the
study; only broadly reusable test utilities belong in `test/test_utils.jl`.
