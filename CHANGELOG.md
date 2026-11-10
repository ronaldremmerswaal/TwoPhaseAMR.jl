# Changelog

## Unreleased

- Shared the stepping controls, force bounds and surface-tension/gravity stages across flow models, and consolidated topology-dependent workspace rebuilding.
- `NavierStokesSystem` is now the concrete equation-system type, and `FlowSimulation` exposes it as `flow.system`; numerical options and physics are retained on the system.
- Two-velocity projection tolerances now compare dimensionless local velocity-jump and residual defects, using the documented local mesh spacing.
- Standalone one-velocity projections require the pressure operator and `velocity_boundary` to have the same open sides; either an open boundary with a Neumann-only operator or an open operator with a mismatched boundary now raises `ArgumentError`.
