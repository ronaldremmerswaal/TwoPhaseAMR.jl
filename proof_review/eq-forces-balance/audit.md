# Audit: eq:forces:balance — balanced force: the force map of a cell-function jump is the weighted gradient, and the projection returns the state

- Source: `docs/techref/chapters/08_forces.tex:137-212` (map 141, map-gradient 153, balance 169, text 170-212), branch `docs/techref-forces-viscosity`, commit `03bad7a` (working tree `/tmp/techref-ch89`)
- Code audited (normative): `src/two_phase/forces.jl` (`_weighted_jump_force!`), `src/operators/elliptic.jl` (`weighted_gradient!`, `project!`), `src/embedded/operators.jl` (`_cut_cell_weighted_gradient!`, `GradientOperator(geometry)`, `project!`), `src/two_phase/stepping.jl` (`_refresh_pressure_substep!`, `_phase_force_substep!`)
- Date: 2026-11-09
- Overall status: ⚠️ statement and its consequence verified (also the converse, "if and only if"); two stale/imprecise sentences in the text (F1, F2)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-forces-balance/scripts/run_all.jl`

## Statement and assumptions

**Claim (as printed, lines 169-190).** If the integrated jumps are those of a cell function $a$ on every face with $\gamma_f>0$, then $\mathsf R\mathcal J=\mathfrak S_\gamma a$ (or $\mathfrak S^{\rm o}a$ with a solid). Consequently, if the velocity $v$ before the force is discretely (open-)solenoidal and satisfies the boundary conditions, the one-velocity projection of $v+\delta t\,\mathsf R\mathcal J$ returns $v$ and the pressure $a$ "up to its constant null space"; it needs (i) the jumps of one cell function on every open face and (ii) the projection with the operator of the map; "where the condition fails ... for the gravity force on cut faces in the liquid ... no balance is claimed".

**Assumptions made explicit.**
1. $\mathsf R=\mathsf C\,\mathsf K\,\mathsf C$ (no solid, $\mathsf C=\operatorname{diag}\gamma^{1/2}$) or $\operatorname{diag}(\gamma^{1/2}/a)\,\mathsf W\mathsf K\mathsf W\,\operatorname{diag}\gamma^{1/2}$ (solid, rows $a_f=0$ zero); $\mathsf K$ symmetric positive definite (eq:setting:riesz, with the cut-cell scaling $\mathsf W$).
2. $\mathsf S_\gamma a=\mathsf R\mathcal D a$ with the *same* $\mathsf K$, $\gamma$ (definition in eq:setting:weighted-gradient / cut-gradient).
3. "Constant null space" is, with a solid, a function constant on each connected component of the graph (active cells, faces with $\gamma_f>0$); covered cells are components of their own.
4. The projection is the one-velocity `project!` with the same operator object (refreshed once per step).

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| 1. $\mathsf R(\mathcal D a)=\mathsf S_\gamma a$ on every face | 153-157, 169-175 | ✅ verified | package: `_weighted_jump_force!` against `weighted_gradient!`/`_cut_cell_weighted_gradient!` on 5 fixtures (uniform, composite, composite + sloped solid, uniform/composite + disc), 3 random cell functions each | `code_01_gradient_identity.jl` (B1) | difference **0.0** (bitwise: same products, same operator) |
| 2. faces with $\gamma_f=0$ do not enter | 176 | ✅ verified | same, with $10^3$-size garbage on all $\gamma=0$ faces (80-203 faces with a solid) | `code_01_gradient_identity.jl` (B2) | difference 0.0 |
| 3. $\mathsf R$ as printed (map) with $\mathsf K$ read from the code | 141-150 | ✅ verified | independent replica $\operatorname{diag}(\gamma^{1/2}/a)\mathsf K\operatorname{diag}(\gamma^{1/2})J$ | `code_01_gradient_identity.jl` (B3) | $9\times10^{-17}$ |
| 4. Projection of $v+\delta t\,\mathsf R\mathcal D a$ returns $v$ and $p=a$ ($v$ nonzero solenoidal, from a projected random field) | 177-183 | ✅ verified | package: `project!` (tolerance $10^{-12}$), 5 fixtures | `code_02_projection_returns.jl` (P1–P3) | $|v_{\rm new}-v|/(\delta t\max|\mathsf R J|)\le6.5\times10^{-10}$; $\max|\mathsf S(p-a)|/\max|\mathsf Sa|\le6.5\times10^{-10}$ (solver tolerance) |
| 5. **Converse:** $\mathsf R\mathcal J=\mathfrak S a\Rightarrow\mathcal J_f=(\mathcal Da)_f$ for $\gamma_f>0$ ("if and only if", used in eq:forces:hydrostatic) | not stated here, used at 439 | ✅ verified | $\mathsf C\mathsf K\mathsf C|_{\gamma>0}$ is symmetric positive definite (smallest eigenvalue 0.0038–0.004, largest 6–30; $\mathsf K$ symmetric to 0.0) hence injective on the support; single-face perturbation $\delta$ of $\mathcal J$ leaves a best-approximation residual $\propto\delta$ (1.5e-4, 1.5e-3, 1.5e-2 for $\delta=10^{-3},10^{-2},10^{-1}$) | `code_03_iff_injectivity.jl` (C1, C2) | the printed implication is an equivalence |
| 6. Needs the *same* operator in the projection | 186-190 | ✅ verified | package: stale coefficient (density of a displaced drop): leftover **0.32** of $\max|\mathsf R J|$; two-velocity operator ($a/\rho$) on the cut mesh: **1.45**; same operator: $2\times10^{-11}$ | `sharp_01_operator_mismatch.jl` (M1, M2) | the condition is necessary |
| 7. "no balance is claimed ... for the gravity force on cut faces in the liquid" | 191-195 | ❌ stale (F1) | package: after issue #74 gravity on cut liquid faces **is** balanced (consistency $3\times10^{-15}$, 113 cut one-phase faces, `eq-forces-hydrostatic`) | `../eq-forces-hydrostatic/scripts/code_01_planar_state.jl` | replace |
| 8. null space "constant" | 181 | ⚠️ imprecise (F2) | algebra (Step 5): $\mathsf S p=0\iff(\mathcal Dp)_f=0$ on $\{\gamma>0\}$ | — | constant per connected component |

## Errors and consequences

**F1 (stale sentence).** The list of cases without a balance claim names "the gravity force on cut faces in the liquid"; since the fix of issue #74 (open-face mixture density $\bar\rho_f^{\rm a}$) that case *is* balanced — the hydrostatic condition holds exactly with a sloped solid in either phase on uniform and composite meshes (consistency residual $4\times10^{-15}$, acceleration error $1.8\times10^{-14}$). The cases that really are not balanced: a curvature varying from face to face, unresolved faces, a planar state whose interface cells are not reproduced exactly by the PLIC (triple cells, interface in the stencil of a wall; see `eq-forces-hydrostatic`), and the two-velocity projections (see `eq-forces-two-velocity`).

**F2 (null space).** With a solid the kernel of $\mathfrak S$ is the set of cell functions constant on each connected component of the graph (active cells; faces with $\gamma_f>0$), not only the constants of the mesh; the projected *velocity* is unique in any case.

**Proposed replacement** (lines 169-195; macros existing; the iff is new):

```latex
\emph{Balanced force.} Let the jumps be those of a cell function $\avar$,
\begin{equation}\label{eq:forces:balance}
  \fjump{}_f=(\jumpmat\avar)_f\ \text{ for every }f\text{ with }\facecoef_f>0
  \quad\Longleftrightarrow\quad
  \forcemap\fjump{}=\gradw{\facecoef}\avar\ \ (\text{or }\gradopen\avar).
\end{equation}
``$\Rightarrow$'' holds since faces with $\facecoef_f=0$ do not enter~\cref{eq:forces:map};
``$\Leftarrow$'' since the matrix $\coefmat^{1/2}\rieszmat\,\coefmat^{1/2}$ (times $\dualscale$ with a solid) is
positive definite on the faces with $\facecoef_f>0$, hence $\forcemap$ is injective on them. Consequently, if the velocity $\velo$
before the force is discretely solenoidal ($\divh\velo=0$, or $\divopen\velo=0$ with a solid) and satisfies the boundary conditions,
the one-velocity projection of $\velo+\dt\forcemap\fjump{}$ (\cref{ch:projection}) returns $\velo$ and the pressure $\avar$, up to a
function that is constant on each connected component of the graph of the active cells joined by the faces with $\facecoef_f>0$:
$\pres=\avar$ makes the corrected velocity $\velo$, which is solenoidal, and the projected velocity is unique.
The statement is exact up to the tolerance of the linear solve. It needs both conditions: the jumps must be those of one cell
function on every open face (the converse above shows that this is also necessary), and the projection must use the
operator of~\cref{eq:forces:map}, which the one-velocity step guarantees by refreshing it once and using it for both
(\cref{sec:forces:time}); a stale coefficient or the two-velocity coefficient $\twocoef$ leaves $O(1)$ of the force
(measured: 0.3 and 1.4 of $\dt\max_f\abs{(\forcemap\fjump{})_f}$ in the drop fixtures). No balance is claimed for a curvature that varies from
face to face, for a face with unresolved curvature (\cref{sec:forces:curvature}), for gravity when the hydrostatic
condition~\cref{eq:forces:hydrostatic} fails (\cref{sec:forces:gravity}), and for the two-velocity projections
(\cref{sec:forces:two-velocity}).
```

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| Implication ⇒ ($\mathsf R\mathcal D a=\mathsf S a$) | **sharp** (identity, bitwise) | 0.0 on 5 fixtures | `code_01` | — |
| ⇐ (jumps must be those of a cell function on $\{\gamma>0\}$) | **sharp** (iff) | positive definiteness of $\mathsf C\mathsf K\mathsf C|_S$; one wrong face by $10^{-2}$ leaves $9\times10^{-3}$ of the force after projection | `code_03`, `sharp_01` (M3) | state the converse |
| "same operator in the projection" | **sharp** (counterexamples) | stale coefficient 0.32; two-velocity coefficient 1.45 (cut mesh), exactly 0 difference without solid | `sharp_01` (M1, M2) | — |
| $\mathsf K$ SPD is needed for the converse only | **sharp** | ($\mathsf K$ SPD is a design property of eq:setting:patch-metric, see `eq-setting-patch-metric`) | — | — |
| Solver tolerance "exact up to tolerance" | **likely sharp** | leftover $4\times10^{-11}$–$6\times10^{-10}$ at tolerance $10^{-12}$ (density ratio 1000) | `code_02` | — |

## Open points

- The kernel statement (F2) is algebraic; the number of connected components was not counted numerically.
- Equivalence `weighted_gradient!` vs `_cut_cell_weighted_gradient!`: different functions, identical products (checked bitwise through the five fixtures); a future change of one without the other would break eq:forces:map-gradient, which the script `code_01` guards.

## History

- 2026-11-09: first audit (steps 1-8; scripts `code_01`-`code_03`, `sharp_01`). `run_all.jl`: PASS (4 scripts, all testsets).
