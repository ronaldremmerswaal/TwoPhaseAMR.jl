# KH single-step projection diagnostic

This development diagnostic isolates the failure of a two-velocity pressure
projection after the KH interface geometry changes. It executes exactly one
production step, but records the states after:

1. interface and momentum transport;
2. balanced capillary forcing;
3. pressure projection.

For each projection it reports:

- the jump-space size, normal-constraint rank, and kernel dimension before
  and after interface transport;
- the condition number of `N * inv(K_xi) * N'`;
- velocity changes caused independently by transport, forcing, and projection;
- circulation and sheet-enstrophy changes caused by the projection;
- commuting/fallback/sliver diagnostics from the production solver.

The default comparison contains CMDGFM and every `MimeticProjection` jump
interpolant. Both the production commuting model and its unreduced
`commuting_sliver_epsilon=0` variant are included.

Run the inexpensive one-step comparison from the repository root with:

```sh
julia --project=. development/kh_single_step/kh_single_step_diagnostic.jl
```

To repeat the same single-step experiment from identical initial data with
time-step factors `1`, `1/2`, `1/4`, and `1/8`, use:

```sh
julia --project=. development/kh_single_step/kh_single_step_diagnostic.jl --refine-dt
```

The refinement mode is designed to expose a geometry-triggered projection
impulse. If the constraint defect or its rank changes discontinuously when
the interface moves, the projection correction need not decrease with
`dt`, even though every linear solve converges.

The script can also be included and driven programmatically:

```julia
include("development/kh_single_step/kh_single_step_diagnostic.jl")
using .KHSingleStepDiagnostic

results = KHSingleStepDiagnostic.run_diagnostic(
    methods = (:cmdgfm, :cell, :commuting_unreduced),
    timestep_factors = (1.0, 0.5),
    cells_per_wavelength = 8,
)
```

Every returned entry contains the raw projection statistics and all stage
diagnostics, so additional plots or tables do not require rerunning a full KH
integration test.
