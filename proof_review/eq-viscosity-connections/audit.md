# Audit: eq:viscosity:connections — measures and distances of the dual-face connections (single phase)

- Source: `docs/techref/chapters/09_viscosity.tex:38-66`, commit `03bad7a`; code `src/operators/staggered_diffusion.jl` (`_dual_diffusion_connections`)
- Date: 2026-11-09
- Overall status: ⚠️ definition equals the code; the geometric reading "length of the dual face" holds for regular (conforming) connections only, and the code has two undocumented rules (zero-distance drop, error)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-viscosity-connections/scripts/run_all.jl`

## Statement and assumptions

For every pair $(\dminus g,\dplus g)$ joined by dual faces and every axis $t$: $\meas g=\bigl|\sum_{g'}\sum_{f'\in\mathcal F_t}w_{g'f'}|f'|\bigr|$, $d_g=|(x_{\dplus g}-x_{\dminus g})_t|$ (nearest periodic image); connections with $\meas g\le256\,\epsilon\sum|\cdot|$ dropped; "on a conforming patch $\meas g$ is the length of the geometric dual face and $d_g$ the distance of the two unknowns".

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| formula = code (signed net measure per primal-flux axis, absolute value, `256 eps` of the absolute sum) | l. 59-62 | ✅ | read against `_dual_diffusion_connections`; $d_g$ checked against positions with the wrapped difference on all 32 265 connections of 24 component graphs | `code_01_connections.jl` | |
| uniform periodic mesh: $\meas g=d_g=h$, four connections per unknown | l. 63-64 | ✅ | exact | same | |
| conforming patch: $\meas g$ is the dual-face length | l. 63-64 | ✅ | for the 25 070 regular connections (not flagged as refinement interface) $\meas g\,d_g=V_{\dminus g}=V_{\dplus g}$ (defect 0) and for the 8 836 unknowns all of whose connections are regular the connection measures close: $\sum\meas g=2|f|$ along the face axis and $2V_f/|f|$ along the other axis (defect 0) | same | |
| classification flag $=$ non-regular of Ch. 6 (`eq:momentum:regular`) | l. 175-177 | ✅ | recomputed from weights and volumes on a 2-level mesh, 3262 dual faces, no mismatch | same | |
| "for each axis $t$ with a non-vanishing net measure" | l. 60 | ✅ (note) | no pair carried both axes in 24 component graphs (0 of 32 265): one axis per pair in practice | same | the chapter wording admits two |

## Errors and consequences

**F1 (precision, geometric reading).** At non-regular dual faces the net measures are *smaller* than the geometric dual-face lengths: the transverse rule spreads the weights over several partner unknowns and signs cancel; on 4 248 unknowns with a refinement-interface connection the connection measures sum to as little as $2/3$ of the dual-cell perimeter projection (example: unknown with face area 0.5, dual volume 0.375, $\sum\meas g$ along $y$ is 1.0 instead of 1.5). The chapter says correctly "on a conforming patch"; the consequence is that the conservative core $-\dualvolmat^{-1}\viscstiff{k}$ is **not** consistent on non-regular faces, which is the reason for the band (`eq:viscosity:laplacian`).

**F2 (undocumented rules).** (i) The code additionally requires $d_g>256\,\epsilon\,L_t$ for a connection of axis $t$. (ii) If a pair with non-zero absolute measure has no axis left it throws `ArgumentError("degenerate AMR dual diffusion connection")`. This is defect **D1** of `eq-viscosity-cut`: a cut-cell schedule (weights times apertures) produces, at a level transition, a dual face between two vertically adjacent unknowns whose only non-zero weight is on an $x$-flux (distance 0 along $x$). Minimal reproducer in `../eq-viscosity-cut/scripts/defect_01_constructor.jl`.

Replacement (add after "the nearest periodic image is used"):

```latex
; a connection whose distance $\conndist{g}$ along $t$ is below $256\machineeps\domextent{t}$ is dropped, and a dual face
none of whose axes survives is an error (this occurs with a solid across a level transition, \cref{sec:viscosity:single}),
and the geometric statement holds for regular dual faces only: at a non-regular dual face the net measures can be smaller
than the length of the geometric dual face
```

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| $\meas g\,d_g=V$ on regular connections | **sharp** (identity) | defect 0 | `code_01_connections.jl` | none |
| geometric dual-face length at non-regular faces | **not valid** | net measure up to $1/3$ below the geometric length | same | state the restriction |
| one connection per axis | unknown whether two can occur | none in 24 random graphs | same | none |

## Open points
None.

## History
- 2026-11-09: first audit.
