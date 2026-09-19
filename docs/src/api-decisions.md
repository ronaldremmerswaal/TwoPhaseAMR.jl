# Refactor API decisions

This record captures the naming and compatibility decisions that guide the
refactor.

## Ownership and vocabulary

- `indicator` means a physical phase or solid characteristic function.
- AMR objects are called refinement criteria and live under `mesh/adaptivity`.
- `solvers` contains linear-system algorithms only.
- `NavierStokesSystem` names the configured equation system;
  `FlowSimulation` owns evolving state, workspace, time, and diagnostics.
- `OneVelocityModel` and `TwoVelocityModel` select the formulation explicitly.

The package remains one Julia module. Folders express source ownership and do
not introduce nested user namespaces.

## Typed choices

Physical and numerical choices are values rather than unrelated symbols or
booleans: `CGFM`/`CSF`/`GFM`, `FrozenInterfaceFluxes`/
`ProjectedLiquidVelocity`, and `CMDGFMProjection`/`MimeticProjection`.
`NavierStokesOptions` groups PDE discretization policy and delegates linear
algebra to `LinearSolverOptions` and `MultigridOptions`.

## Compatibility window

The keyword-only high-level `FlowSimulation` constructor translates to the
typed API and warns. It is supported throughout `0.2.x`, with removal no
earlier than `0.3`. The same window applies to compatibility-only high-level
keywords. Low-level constructors and useful type aliases are not scheduled
for removal merely because the high-level path is smaller.

Any future removal requires a release note, an entry in the migration guide,
and a replacement that preserves access to the relevant low-level numerical
building block.
