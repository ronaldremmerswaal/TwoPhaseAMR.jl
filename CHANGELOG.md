# Changelog

## Unreleased

- `NavierStokesSystem` is now the concrete equation-system type, and `FlowSimulation` exposes it as `flow.system`; numerical options and physics are retained on the system.
- Interface and solid refinement bands protect coarsening families whose parent may be near the boundary, preventing the buffer-induced static-band refine/coarsen cycle.
- Curvature refinement detects mixed liquid–gas cells using the open-fluid conditional fraction, rather than treating liquid-filled solid cut cells as unresolved interfaces.
- Two-velocity projection tolerances now compare dimensionless local velocity-jump and residual defects, using the documented local mesh spacing.
- Standalone one-velocity projections require the pressure operator and `velocity_boundary` to have the same open sides; either an open boundary with a Neumann-only operator or an open operator with a mismatched boundary now raises `ArgumentError`.
