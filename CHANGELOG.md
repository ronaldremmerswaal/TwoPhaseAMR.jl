# Changelog

## Unreleased

- Fixed premature termination of ILU-preconditioned CMDGFM projection solves and eliminated warm staggered elliptic solver allocations.
- Avoided per-face allocations in the single-phase pressure predictor.
- Two-velocity projection tolerances now compare dimensionless local velocity-jump and residual defects, using the documented local mesh spacing.
- Standalone one-velocity projections require the pressure operator and `velocity_boundary` to have the same open sides; either an open boundary with a Neumann-only operator or an open operator with a mismatched boundary now raises `ArgumentError`.
