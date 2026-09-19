# Adding an equation system

Equation systems are top-level numerical domains. A future Maxwell model, for
example, belongs in `src/maxwell/`, not in `src/solvers/`; the latter is
reserved for reusable linear-system algorithms.

## Define ownership first

Before adding files, identify four boundaries:

1. immutable physical and numerical configuration;
2. evolving fields owned by the simulation;
3. reusable operator/linear-solver workspace;
4. one completed-step operation and its convergence statistics.

Keep generic mesh fields and spatial operators in `mesh/` and `operators/`.
Put only reusable Krylov, multigrid, and preconditioning machinery in
`solvers/`. Model-specific assembly and stage ordering stay with the equation
system.

## Integrate with package services

- Add includes to the labeled dependency manifest in `src/TwoPhaseAMR.jl` at
  the lowest layer that satisfies all dependencies.
- Expose a compact typed constructor. Do not infer the model from field shape
  or accept unrelated keywords in one unstructured option list.
- Make the workspace reusable across steps and rebuild only the parts whose
  topology or coefficients changed.
- Return typed convergence/statistics data from stepping and linear solves.
- Construct [`SimulationState`](@ref) at completed states so custom
  diagnostics can observe the new system without depending on visualization.
- Keep visualization integration as a subscriber/adapter; the equation-system
  core must load without Makie.

## Verification checklist

Add focused tests for constructor validation and dispatch, isolated operator
identities, one representative completed step, Float32, supported boundary
and AMR paths, and allocation-sensitive kernels. Place long convergence or
literature studies under `test/integration/` or `test/validation/` and add a
separate CI shard when their runtime is unsuitable for the standard suite.

Finally, add a tutorial-level example, API reference entries, numerical-method
explanation with literature citations or explicit citation TODOs, and a
benchmark fixture for construction and steady stepping.
