# Changelog

## Unreleased

- Made single-phase composite-grid viscous diffusion dissipative in the dual-volume norm, using shared geometric transition contacts instead of nonsymmetric corrected rows, and gave a split boundary side's wall strips their own half-wall rows, matching the embedded operator (#80).
- `step!(flow, dt)` now performs a complete flow step, including time and counter updates, post-step adaptivity, diagnostics, and the `:step` observer event; `simulate!` repeats these steps and retains run lifecycle events and continuation.
- `FlowSimulation` accepts `force` and stores it as `flow.force`, the default body acceleration for its stepping and simulation calls. Field-level `step!` remains available for the physical solve alone.
- Prevented CG from using asymmetric multigrid V-cycles and selected CG for volume-weighted symmetric embedded viscous systems (#128).
- Shared the stepping controls, force bounds and surface-tension/gravity stages across flow models, and consolidated topology-dependent workspace rebuilding.
- `NavierStokesSystem` is now the concrete equation-system type, and `FlowSimulation` exposes it as `flow.system`; numerical options and physics are retained on the system.
- Fixed premature termination of ILU-preconditioned CMDGFM projection solves and eliminated warm staggered elliptic solver allocations.
- Avoided per-face allocations in the single-phase pressure predictor.
- Interface and solid refinement bands protect coarsening families whose parent may be near the boundary, preventing the buffer-induced static-band refine/coarsen cycle.
- Curvature refinement detects mixed liquid–gas cells using the open-fluid conditional fraction, rather than treating liquid-filled solid cut cells as unresolved interfaces.
- Two-velocity projection tolerances now compare dimensionless local velocity-jump and residual defects, using the documented local mesh spacing.
- Standalone one-velocity projections require the pressure operator and `velocity_boundary` to have the same open sides; either an open boundary with a Neumann-only operator or an open operator with a mismatched boundary now raises `ArgumentError`.
