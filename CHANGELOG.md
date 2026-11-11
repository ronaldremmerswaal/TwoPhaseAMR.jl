# Changelog

## Unreleased

- **Breaking:** removed the legacy mimetic two-velocity pressure projection. `MimeticProjection`, `MimeticProjectionSolver`, `MimeticProjectionWorkspace`, `MimeticProjectionStats` and `mimetic_gradient!` are gone; `CMDGFMProjection` is the only `AbstractTwoVelocityProjection` and still the default. The `ProjectedLiquidVelocity` carrier is unchanged. The `capillary_mimetic` oracle case, the mimetic example variants, tests, documentation and technical-reference sections were removed with it.
- One-phase `step!` now checks the advective bound at the top of the accepted `cfl` range (`cfl = 1`) rather than at `cfl = 1/2`, so `simulate!(…; cfl = 0.8)` no longer stops at its first advection-limited step and any step `stable_timestep` returns is accepted (#121). `simulate!` and the flow-level `stable_timestep` reject a `cfl` outside `(0, 1]` (one phase) or `(0, 1/2]` (two phases) up front, also when a prescribed `dt` makes it unused.
- Made single-phase composite-grid viscous diffusion dissipative in the dual-volume norm, using shared geometric transition contacts instead of nonsymmetric corrected rows, and gave a split boundary side's wall strips their own half-wall rows, matching the embedded operator (#80).
- `step!(flow, dt)` now performs a complete flow step, including time and counter updates, post-step adaptivity, diagnostics, and the `:step` observer event; `simulate!` repeats these steps and retains run lifecycle events and continuation.
- `FlowSimulation` accepts `force` and stores it as `flow.force`, the default body acceleration for its stepping and simulation calls. Field-level `step!` remains available for the physical solve alone.
- Bound WY cut-donor timesteps by the actual swept strips, including hanging faces and frozen phase subfaces; freeze subface ordering with apertures in solid geometry to keep opposing donating regions disjoint, snap round-off phase apertures to an exact single phase, and report an empty donor strip by face and cell instead of returning a zero step (#52, #56).
- Prevented CG from using asymmetric multigrid V-cycles and selected CG for volume-weighted symmetric embedded viscous systems (#128).
- Shared the stepping controls, force bounds and surface-tension/gravity stages across flow models, and consolidated topology-dependent workspace rebuilding.
- `NavierStokesSystem` is now the concrete equation-system type, and `FlowSimulation` exposes it as `flow.system`; numerical options and physics are retained on the system.
- Fixed premature termination of ILU-preconditioned CMDGFM projection solves and eliminated warm staggered elliptic solver allocations.
- Avoided per-face allocations in the single-phase pressure predictor.
- Interface and solid refinement bands protect coarsening families whose parent may be near the boundary, preventing the buffer-induced static-band refine/coarsen cycle.
- Curvature refinement detects mixed liquid–gas cells using the open-fluid conditional fraction, rather than treating liquid-filled solid cut cells as unresolved interfaces.
- Two-velocity projection tolerances now compare dimensionless local velocity-jump and residual defects, using the documented local mesh spacing.
- Standalone one-velocity projections require the pressure operator and `velocity_boundary` to have the same open sides; either an open boundary with a Neumann-only operator or an open operator with a mismatched boundary now raises `ArgumentError`.
