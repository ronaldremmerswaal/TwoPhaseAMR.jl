# Proof review archive

One row per audited proof. Each folder holds `audit.md` (steps, errors, sharpness, history) and
`scripts/` (Julia checks; `scripts/run_all.jl` reruns them and exits non-zero on failure).

| Label | Claim | Source | Latest audit | Status | Folder |
|---|---|---|---|---|---|
| `eq:setting:patch-metric` | The patch moment tensor $\mathsf Y_v^\top\mathsf N_v$ is symmetric on every vertex patch of a 2:1-graded quadtree with square cells (so the symmetrisation is a no-op and affine pressures are reproduced exactly) | `docs/techref/chapters/03_discrete_setting.tex:256-305` | 2026-11-08 | ⚠️ claim verified; E1 (2:1 balance not guaranteed for some extents), E2 (two-cell periodic direction) | [eq-setting-patch-metric](eq-setting-patch-metric/) |
