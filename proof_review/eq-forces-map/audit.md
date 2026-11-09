# Audit: eq:forces:map — the force map $\mathsf R$, its identity with the pressure gradient (eq:forces:map-gradient) and the conforming reduction (eq:forces:conforming)

- Source: `docs/techref/chapters/08_forces.tex:137-168` (map 141, map-gradient 153, conforming 160); coefficients 99-136; `docs/techref/chapters/03_discrete_setting.tex:377-415` (weighted/cut gradient), commit `03bad7a`
- Code audited (normative): `src/two_phase/forces.jl` (`_weighted_jump_force!`), `src/operators/elliptic.jl` (`weighted_gradient!`), `src/embedded/operators.jl` (`_cut_cell_weighted_gradient!`, `GradientOperator(geometry)`, `_scaled_inverse_metric`)
- Date: 2026-11-09
- Overall status: ✅ verified (three labels); one precision proposed ("conforming patch")
- Rerun: `julia --project=@claude-symbolics proof_review/eq-forces-map/scripts/run_all.jl`

## Statement and assumptions

**`eq:forces:map`.** $(\mathsf R\mathcal J)_f=(\mathsf C\mathsf K\mathsf C\mathcal J)_f$ without solid; with a solid $(\gamma_f^{1/2}/a_f)(\mathsf W\mathsf K\mathsf W\,\mathsf C\mathcal J)_f$ if $a_f>0$, 0 if $a_f=0$; boundary faces 0; $\mathsf C=\operatorname{diag}\gamma^{1/2}$.
**`eq:forces:map-gradient`.** $\mathfrak S_\gamma p=\mathsf R(\mathcal Dp)$ and $\mathfrak S^{\rm o}p=\mathsf R(\mathcal Dp)$, "evaluated by the same products with the same operator".
**`eq:forces:conforming`.** On a conforming patch ($\mathsf K$ diagonal, entries $1/V_f$): $(\mathsf R\mathcal J)_f=\mathcal J_f/(\bar\rho_fV_f)$ (no solid), $a_f\mathcal J_f/(\bar\rho_fV^{\rm o}_f)$ (solid); at a level transition $\mathsf K$ couples the segments of a vertex patch, so a face without a jump can carry an acceleration.

**Assumptions made explicit.** $\gamma=1/\bar\rho$ or $a^2/\bar\rho$ (eq:forces:coefficient); $\mathsf W^2=V_f/V^{\rm o}_f$ is already contained in the code's `inverse_metric` (`scale .^ 2` multiplies the entries of $\mathsf K$); $V^{\rm o}_f$ replaced by 1 where it vanishes (faces with $a_f=0$, which carry no force); "conforming patch" = the face's row of $\mathsf K$ has a single entry.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| 1. conforming algebra, no solid: $\gamma^{1/2}\tfrac1V\gamma^{1/2}J=J/(\bar\rho V)$ | 160-165 | ✅ verified | symbolic (Symbolics.jl), $\gamma=1/r^2$ | `step01_map_symbolic.jl` (S1) | exact |
| 2. conforming algebra, solid: $(\gamma^{1/2}/a)\,W\tfrac1VW\,\gamma^{1/2}J=aJ/(\bar\rho V^{\rm o})$, $\gamma=a^2/\bar\rho$, $W^2=V/V^{\rm o}$ | 160-165 | ✅ verified | symbolic ($a=s_a^2$, $\bar\rho=r^2$, $W=w$) | S2 | exact |
| 3. two-velocity: $(\mathsf R^{\rm tv}J)_f=J/(\bar\rho V^{\rm o})$, so $\mathsf R^1J=a\mathsf R^{\rm tv}J$ on diagonal rows | (new; used in `eq-forces-two-velocity`) | ✅ verified | symbolic, non-vacuous (the two maps differ symbolically) | S3 | — |
| 4. entry-wise: $\mathsf R^1_{ij}=\sqrt{a_ia_j}\,\mathsf R^{\rm tv}_{ij}=\mathsf M_{ij}a_j/\sqrt{\bar\rho_i\bar\rho_j}$, i.e. $\mathsf R^1=\mathsf A^{1/2}\mathsf R^{\rm tv}\mathsf A^{1/2}$ | (new) | ✅ verified | symbolic, $3\times3$ symmetric $\mathsf M=\mathsf W\mathsf K\mathsf W$, 18 identities | S4 | — |
| 5. code: $K_{ff}=1/V^{\rm o}_f$ at diagonal rows ($=\mathsf W^2/V_f$); $\mathsf K$ symmetric | 160, 148 | ✅ verified | package: 5 fixtures, 480–1162 diagonal rows each | `code_01_conforming.jl` (C1) | $2.2\times10^{-16}$ |
| 6. code: $(\mathsf R\mathcal J)_f=\mathcal J_f/(\rho^{\rm ind}_fV_f)$ / $a_f\mathcal J_f/(\rho^{\rm ind}_fV^{\rm o}_f)$ at diagonal rows, random $\mathcal J$ and independent density | 160-165 | ✅ verified | package `_weighted_jump_force!` | C2 | $4.3\times10^{-16}$ |
| 7. level transition: 78 coupled rows in $x\in[3.75,4.125]$ (composite fixtures); a face without jump carries an acceleration | 166-168 | ✅ verified | package: set $\mathcal J_{f_0}=0$ on a coupled face, other jumps random → $|(\mathsf R\mathcal J)_{f_0}|>10^{-8}$ (3 composite fixtures) | C3 | — |
| 8. $\mathsf R\mathcal J$ (code) equals the independent replica of eq:forces:map with independent $\bar\rho$ for the CSF/CGFM jumps of the drop (uniform, composite, cut) and the gravity jumps (6 planar fixtures) | 141-150 | ✅ verified | independent replica (K from the code, strip-weighted density with own polygon clipping) | C4 | $1.3\times10^{-16}$ |
| 9. $\mathfrak S_\gamma p=\mathsf R(\mathcal Dp)$ and $\mathfrak S^{\rm o}p=\mathsf R(\mathcal Dp)$ | 153-157 | ✅ verified | package, 5 fixtures, random $p$ | `../eq-forces-balance/scripts/code_01_gradient_identity.jl` (B1) | **0.0** (bitwise) |
| 10. boundary faces get zero | 150 | ✅ verified by inspection | `fill!(destination.boundary, 0)` in both routines (forces.jl, operators.jl) | — | — |

## Errors and consequences

No error. Precision (not an error): "conforming patch" is used for faces whose row of $\mathsf K$ is diagonal; in a mesh with a level transition 1162 of 1240 faces are conforming in this sense, the coupled ones are within one coarse cell of the transition. A proposed one-line addition to eq:forces:conforming:

```latex
On a conforming patch, i.e.\ at the faces whose row of $\rieszmat$ has a single entry $1/\dualvol{f}$,
\begin{equation}\label{eq:forces:conforming}
  (\forcemap\fjump{})_f=\frac{\fjump{}_f}{\stagdens_f\,\dualvol{f}}\quad\text{(no solid)},\qquad
  (\forcemap\fjump{})_f=\frac{\aperture{f}\fjump{}_f}{\stagdens_f\,\opendualvol{f}}\quad\text{(solid)} :
\end{equation}
an integrated force per staggered mass.
```
And for the two-velocity section (new fact, from steps 3–4): with the same $\rieszmat$ the one-velocity map and the map of the two-velocity coefficient $\twocoef$ are related by $\forcemap^{1}=\mathsf A^{1/2}\forcemap^{\mathrm{tv}}\mathsf A^{1/2}$, $\mathsf A=\operatorname{diag}\aperture{}$, so $\forcemap^{1}\fjump{}=\aperture{f}\forcemap^{\mathrm{tv}}\fjump{}$ at conforming faces (stated in `eq-forces-two-velocity`).

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| Identities 1–4, 9 | **sharp** (exact identities) | symbolic; package bitwise | `step01`, `code_01` | — |
| "coupled at a level transition" | **sharp** | only the 78 rows of the vertex patches at $x=4$ are coupled; 0 coupled rows on the uniform meshes | `code_01` (C3) | — |
| Formula with a solid needs $V^{\rm o}_f>0$ | **sharp** (true for every face with $a_f>0$ in the fixtures; closed faces carry no force) | C2 over all diagonal rows with $a_f>0$ | `code_01` | state "for $a_f>0$" |
| $\mathsf K$ SPD | not re-proved here | `eq-setting-patch-metric` | — | — |

## Open points

- The replacement of a vanishing $V^{\rm o}_f$ by one depends on the units (issue #42, cited in chapter 3); it does not enter any force since such faces have $a_f=0$ (checked: all 457 faces with both strips empty have $a_f=0$, `eq-forces-stages` D4).

## History

- 2026-11-09: first audit (`step01_map_symbolic.jl`, `code_01_conforming.jl`; identity B1 in `eq-forces-balance`). `run_all.jl`: PASS (symbolic 23 tests; package 5 tests).
