# Testing and validation

The test tree follows numerical ownership and separates fast package tests
from long refinement studies and machine-dependent benchmarks.

| Area | Location | Purpose |
| --- | --- | --- |
| unit and regression | `test/mesh`, `test/operators`, `test/indicator`, `test/two_phase`, `test/navier_stokes`, `test/visualization` | isolated contracts and numerical invariants |
| examples | `test/examples` | short, headless smoke runs of public example entry points |
| validation | `test/validation` | compact manufactured/reference checks used by the main suite or integration helpers |
| integration | `test/integration` | longer refinement and literature-comparison studies, sharded in CI |
| performance contracts | `test/performance` | allocation and hot-kernel regression checks |
| benchmarks | `benchmark` | machine-local timing and allocation measurements |

Run the standard suite with:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

Run one integration study directly, preferably with the same thread count as
CI:

```sh
julia --project=. --threads=4 test/integration/two_phase_waves.jl
```

The benchmark environment and reproducibility controls are documented in
`benchmark/README.md`. Optional cross-package reference cases are documented
in `test/reference-validation.md` and are not normal package dependencies.

When adding an equation system or model specialization, add focused
constructor/dispatch tests, a regression for every numerical invariant at
risk, and a small public-API example. Keep case-specific factories beside the
study; only broadly reusable test utilities belong in `test/test_utils.jl`.
