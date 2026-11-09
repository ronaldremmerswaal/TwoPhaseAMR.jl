# Audit: eq:forces:two-velocity — the common force increment of the two-velocity step; balance with CMDGFM and mimetic projections (issue #76)

- Source: `docs/techref/chapters/08_forces.tex:520-574` (two-velocity 523, deviation 555), `eq:forces:coefficient` 99-136; issue #76; commit `03bad7a`
- Code audited (normative): `src/two_phase/two_velocity/stepping.jl` (`_phase_force_substep!`, `_add_common_phase_acceleration!`, `step!`), `src/two_phase/projection/common.jl` (`_projection_pressure_operator`, `_projection_gradient_core!`, `_projection_face_data`), `src/two_phase/projection/cmdgfm_projection.jl`, `src/two_phase/projection/mimetic_projection.jl`
- Date: 2026-11-09 (first audit at `03bad7a`; update "After the fix of #76" at `266c7f8`)
- Overall status: ⚠️ (**after the fix: CMDGFM ✅, mimetic still open**; see the section "After the fix") — first audit: formula and the measured numbers verified (all of the chapter's quoted values reproduced); the **explanation is incomplete**: the CMDGFM failures of issue #76 (with a solid on a composite mesh, and the CSF drop on the cut mesh) have one cause, now proved and verified by a counterfactual; the mimetic failure is independent of the map; exact conditions below
- Rerun: `julia --project=@claude-symbolics proof_review/eq-forces-two-velocity/scripts/run_all.jl`

## Statement and assumptions

**Claim.** $v^{\pi,(3)}_f=v^{\pi,(2)}_f+\delta t(\mathsf R\mathcal J^\sigma)_f+\delta t(\mathsf R\mathcal J^g)_f$ for both phases (not at absent values), with the one-velocity map ($\gamma=a^2/\bar\rho$ with a solid); the projection that follows (CMDGFM or mimetic) assembles its own system with the common coefficient $\gamma^{\rm tv}=a/\bar\rho$ and the gradient-jump unknowns $\xi$. "Without a solid $\gamma^{\rm tv}=\gamma$, so the jumps $\mathcal Da$ give the common pressure-gradient part of the projection for $p=a$, and whether the projection returns $a$ with $\xi=0$ is a property of its system. With a solid the maps differ by the factor $a_f$ on cut faces, and the balance fails where a force acts on such faces." Measured: one inviscid step from rest of the planar heavy liquid leaves $2.2\times10^{-7}$ (one velocity), $9\times10^{-11}$ / $1.1\times10^{-10}$ (CMDGFM without solid / solid on the uniform mesh), 0.029 (CMDGFM, solid, composite), 0.017 / 0.66 (mimetic, uniform / composite, also without solid), of $\delta t|g|$; a drop with constant curvature through CMDGFM keeps $1.1\times10^{-12}$ without solid, 0.013 on the cut mesh. "Open (#76)".

**Assumptions made explicit.** Equal phase increments are projected from rest; $\xi$ exists only at resolved mixed faces; the experiments below project the increments directly (projection only; they reproduce the full `simulate!` numbers of the chapter: 0.0292, 0.0299, 0.0175, 0.662, 0.653, 0.0127 vs 0.029, 0.017, 0.66, 0.013).

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| 1. Without solid $\gamma^{\rm tv}=\gamma$ (bitwise) | 530-533 | ✅ verified | package operators (`PoissonOperator` vs `_projection_pressure_operator`) | `code_01_map_relation.jl` (M1) | `face_coefficient ==` |
| 2. With a solid: $\mathsf R^1=\mathsf A^{1/2}\mathsf R^{\rm tv}\mathsf A^{1/2}$ (entry-wise $\sqrt{a_ia_j}$), so $\mathsf R^1\mathcal J=a_f\mathsf R^{\rm tv}\mathcal J$ **only at faces whose row of $\mathsf K$ is diagonal** | 534-536 | ⚠️ the printed "maps differ by the factor $a_f$" is exact at conforming faces only (F1) | symbolic ($3\times3$) + package | `../eq-forces-map/scripts/step01_map_symbolic.jl` (S4), `code_01` (M2) | at the 78 coupled rows the ratio is not $a_f$ (deviation 12 %) |
| 3. Phase gradients of the code are $\text{common}+\zeta^\pi\xi/m$, $\zeta^{\rm l}=-m_{\rm g}$, $\zeta^{\rm g}=m_{\rm l}$, so $\operatorname{grad}^{\rm g}-\operatorname{grad}^{\rm l}=\xi$; for equal phase increments $|v^{\rm l}-v^{\rm g}|=\delta t|\xi|$ after the projection and $v^{\rm l}=v^{\rm g}$ where there is no $\xi$ | (new; code `common.jl`) | ✅ verified | package CMDGFM, 4 hydrostatic fixtures | `sharp_01_condition_cases.jl` (X1) | difference 0.0 |
| 4. In every hydrostatic case $\xi\equiv0$ (also where the rest state is lost), hence CMDGFM = the **one-velocity** cut-cell projection with the coefficient $\gamma^{\rm tv}$ of the common increment | (new) | ✅ verified | package: $\max|\xi|=0$; $\max|v_{\rm CMDGFM}-v_{\rm one\text{-}velocity}(\mathsf R^{\rm tv})|/(\delta t\max|\mathsf R^1J|)$ | X2, X3 | $\le4.4\times10^{-10}$ (four fixtures; $\xi=0$ exactly) |
| 5. **Exact CMDGFM balance condition:** $\exists p$: $\mathsf R^1\mathcal J^g=\mathsf R^{\rm tv}\mathcal Dp$ (follows from 3–4). Without solid $\iff\mathcal J_f=(\mathcal Da)_f$ (the hydrostatic condition); with a solid, sufficient: $a_f=1$ on every face carrying a jump and on every face with $a_g>0$ coupled to it through $\mathsf K$; fails otherwise | 536-545 | ✅ verified | $E=\mathsf R^1J-\mathsf R^{\rm tv}\mathcal Dp^*$ over the six fixtures: zero ($\le2.6\times10^{-11}$) exactly when $\min a>0=1$ on the reach of the jumps; **0.034** for composite + solid in either phase (one cut face with $a=0.8$ at the level transition carries a jump, $\min a=0.8$) | `code_01` (M3, M4) | $E=0$ ⟺ min $a=1$ in both directions over the six fixtures |
| 6. CMDGFM with the increment through $\mathsf R^{\rm tv}$ (counterfactual: `gravity!` called with the two-velocity operator) is balanced in every case | — | ✅ verified | package: leftover $\le1.1\times10^{-10}$ (composite + solid in gas: $1.09\times10^{-10}$; liquid: $4.3\times10^{-11}$) vs 0.029 / 0.030 with the one-velocity map | `code_02_hydrostatic_projection.jl` (P1, P2) | **the coefficient mismatch is the only cause for CMDGFM** |
| 7. Measured numbers (table below) | 545-555 | ✅ verified | package | `code_02` | all six chapter numbers reproduced (projection only) |
| 8. Constant-curvature drop through CMDGFM: balanced without solid ($\le1.1\times10^{-12}$); the 0.013 on the cut mesh is **CSF only** (the figure script uses CSF); CGFM on the same mesh is balanced ($2.5\times10^{-12}$) | 550-555 | ⚠️ the text omits "CSF" and has no explanation ("unexplained" in #76) (F2) | package + $E$: $E\ne0$ at exactly **one** face, $(x,y)=(4.0,1.375)$, $a=0.8$, $J=0$ — a cut face coupled by $\mathsf K$ at the level transition $x=4$ to faces carrying a CSF jump (the CSF smear reaches the row $y=1.5$) | `code_03_drop_tension.jl` | with the two-velocity map: $2.2\times10^{-12}$ |
| 9. Mimetic (default `:joint_smoothed`) is never balanced, with the one- **and** the two-velocity map | 545-555 | ✅ verified | package | `code_02` (P3) | 0.0175 (uniform), 0.662 (composite), with solids 0.0175 / 0.653 / 0.654 / 0.662: **independent of the map** |
| 10. mimetic `:joint_face`: balanced on the uniform mesh without solid ($5.7\times10^{-12}$), not on the composite mesh (0.094) | (new) | ✅ verified | package | `code_02` | the default variant is the worse one |
| 11. interface exactly on a grid line (no mixed cell): CMDGFM balanced ($1.5\times10^{-11}$, $2.2\times10^{-11}$), **default mimetic far worse** (≈1.0 uniform, 1.76 composite) | (new) | ✅ verified | package | `code_02` (P4) | the smoothed-aperture rows do not rely on mixed cells |
| 12. mimetic residual location: uniform — all 32 residual faces touch a mixed cell; composite — 521 residual faces, only 71 touch a mixed cell | (new) | ✅ verified | package | `code_02` (localisation) | the failure spreads on composite meshes |

**Hydrostatic leftover** (projection only; maximum phase velocity $/(\delta t|g|)$; planar interface through (4.3, 4.6), densities (1000, 1)):

| fixture | CMDGFM, one-velocity increment (code) | CMDGFM, two-velocity increment | mimetic (default), one-velocity | mimetic, two-velocity |
|---|---|---|---|---|
| uniform, no solid | $7.3\times10^{-12}$ | = | 0.0175 | = |
| uniform, solid in gas | $1.8\times10^{-11}$ | $1.8\times10^{-11}$ | 0.0175 | 0.0175 |
| uniform, solid in liquid | $2.1\times10^{-11}$ | $3.4\times10^{-11}$ | 0.0175 | 0.0175 |
| composite, no solid | $1.8\times10^{-11}$ | = | 0.662 | = |
| composite, solid in gas | **0.0292** | $1.1\times10^{-10}$ | 0.653 | 0.654 |
| composite, solid in liquid | **0.0299** | $4.3\times10^{-11}$ | 0.662 | 0.662 |

## Errors and consequences

**F1 (cause of the CMDGFM failures; the chapter's "differ by the factor $a_f$ on every cut face" is incomplete).** $\mathsf R^1=\mathsf A^{1/2}\mathsf R^{\rm tv}\mathsf A^{1/2}$. The common pressure gradient of the CMDGFM projection is $\mathsf R^{\rm tv}\mathcal Dp$, and $\xi\equiv0$ for equal phase increments (step 4), so the projection removes exactly the part of $\mathsf R^1\mathcal J$ that lies in the range of $\mathsf R^{\rm tv}\mathcal D$; the rest stays as a velocity. The rest vanishes iff $\mathsf R^1\mathcal J=\mathsf R^{\rm tv}\mathcal Dp$ for a cell function $p$: without a solid iff the hydrostatic condition (eq:forces:hydrostatic); with a solid it is violated by faces with $a_f<1$ that carry a jump **or are coupled to one by $\mathsf K$ at a level transition** — not only by faces carrying a jump. On a uniform mesh with vertical $g$ all faces with a jump have $a_f=1$ (cut faces carry $\mathcal D\phi=0$), hence the balanced uniform solid cases; on the composite mesh a cut face with $a=0.8$ at the level transition carries a jump ($\phi$ differs between the coarse and the fine neighbour) and the balance is lost (3 %).
Consequence for the text: the sentence "the balance fails where a force acts on such faces" must read "on such faces and on the faces coupled to them at a level transition". Fix in the code (not applied, outside this audit): evaluate the two-velocity phase increments with the two-velocity pressure operator, i.e. call `gravity!`/`surface_tension!` with `_projection_pressure_operator(...)` instead of the one-velocity `solver.pressure_solver.operator` (counterfactual of step 6: all CMDGFM cases balanced to $10^{-10}$).

**F2 (drop on the cut mesh).** The 0.013 of the third bullet of #76 is the CSF force on the composite+solid mesh with the same mechanism (step 8): one cut face ($a=0.8$) coupled to CSF jumps at $x=4$; CGFM is balanced. "Where the solid is away from the drop" is true for the jumps but not for the coupling of the metric. Replacement text: see below.

**F3 (mimetic).** The default mimetic projection loses hydrostatic equilibrium **independently of the force map** (steps 9–12), even without a solid and on a uniform mesh; the variant `:joint_face` keeps it on the uniform mesh without solid only. The residual sits at the faces adjacent to mixed cells (uniform) and spreads globally on composite meshes; for an interface on a grid line the default variant is worse (≈1). A derivation of the cause is not possible from the chapter (chapter 10 is a stub); it is a property of the fitted jump rows, not of the force map.

**Proposed replacement** (the paragraph after `eq:forces:two-velocity`, lines 526-554; macros existing, `\forcemap^{\mathrm{tv}}` is the same map with $\twocoef$):

```latex
Without a solid $\twocoef=\facecoef$. For equal phase increments the jump unknown of the CMDGFM projection vanishes ($\gradjump=0$:
its rows are homogeneous and its phase gradients differ by $\gradjump$), so CMDGFM acts as the one-velocity projection with the
coefficient $\twocoef$, and it returns a rest state exactly if there is a cell function $\avar$ with
$\forcemap\fjump{}=\forcemap^{\mathrm{tv}}\jumpmat\avar$. Without a solid this is the condition $\fjump{}=\jumpmat\avar$ of
\cref{eq:forces:balance}. With a solid $\forcemap=\mathsf A^{1/2}\forcemap^{\mathrm{tv}}\mathsf A^{1/2}$, $\mathsf A=\diag{\aperture{}}$, so at
a conforming face $\forcemap\fjump{}=\aperture{f}\,\forcemap^{\mathrm{tv}}\fjump{}$; the two agree if $\aperture{f}=1$ on every face with a
jump and on every face coupled to it through $\rieszmat$, and in general not: at a level transition a cut face with
$\aperture{f}<1$ that carries a jump, or is coupled to one, leaves a velocity. (The sentence with the measured one-step values of the existing text follows here unchanged.) Evaluating
the increments with $\twocoef$ in place of $\facecoef$ removes it (all CMDGFM cases balanced to $10^{-10}$, a check not used by the code).
The drop with constant curvature behaves alike: the CSF force on the cut mesh leaves 0.013 through one cut face with
$\aperture{f}=0.8$ at the level transition that $\rieszmat$ couples to faces carrying a CSF jump; CGFM is balanced ($2.5\times10^{-12}$).
The mimetic projection is not balanced with either map: it leaves 0.017 (uniform) and 0.66 (composite) without solid, 0.09 on the
composite mesh already for the variant \code{joint_face} (balanced to $6\times10^{-12}$ on the uniform mesh), and about 1 for an
interface on a grid line; the loss is a property of its fitted jump rows (\issue{76}).
```

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| "$\gamma^{\rm tv}=\gamma$ without solid" | **sharp** (bitwise) | M1 | `code_01` | — |
| CMDGFM balance iff $\mathsf R^1J=\mathsf R^{\rm tv}\mathcal Dp$ | **sharp** (necessary and sufficient given $\xi=0$) | $\xi\equiv0$ in all cases; $E=0\iff\min a=1$ in the six fixtures; counterfactual with $\mathsf R^{\rm tv}$ balances every case | `sharp_01`, `code_01`, `code_02` | — |
| sufficient condition "$a=1$ on the jumps and their $\mathsf K$-neighbours" | **likely sharp** (6 fixtures: both directions) | M3/M4 | `code_01` | a proof for general $\mathsf K$ is the congruence $\mathsf R^1=\mathsf A^{1/2}\mathsf R^{\rm tv}\mathsf A^{1/2}$ plus the injectivity of $\mathsf R^{\rm tv}$ on its support |
| "the balance fails where a force acts on cut faces" | **not sharp** (too narrow and too broad) | holds also at faces coupled to a jump; does not fail on uniform meshes with solid, where the cut faces carry no jump | `code_01`, `code_02` | replacement above |
| mimetic never balanced | **sharp for the default variant** (6 fixtures, with both maps, grid-aligned interface); not for `:joint_face` on the uniform mesh without solid | `code_02` (P3, P4) | — | open: cause |

## After the fix of #76 (2026-11-09, commit `266c7f8`: PR #81 merged, `497a401`)

**Code change audited.** `src/two_phase/two_velocity/stepping.jl`: `_phase_force_substep!` now evaluates `surface_tension!`/`gravity!` with the pressure operator cached by the selected projection (`_phase_force_pressure_operator!`: CMDGFM `cache.operator`, mimetic `cache.pressure.operator`), i.e. the map $\mathsf R^{\rm tv}$ (coefficient $a/\bar\rho$) when there is a solid; on an uncut mesh the one-velocity operator as before (identical coefficient); `step!` refreshes the one-velocity pressure map only without a solid.

**Scripts updated.** The `:one` rows of `code_02`/`code_03` (and `sharp_01`) keep the *former* increment $\mathsf R^1\mathcal J$ as a counterfactual (they characterise the mechanism, F1); the new `:code` rows use the code's own operator obtained from `_phase_force_pressure_operator!`; `code_01` asserts that this operator equals the two-velocity pressure operator (coefficient, and metric for CMDGFM; coefficient for mimetic); the new `code_04_full_step.jl` runs `simulate!` through the three velocity models. Assertions now: CMDGFM $\le10^{-8}$ in every case, mimetic $>10^{-3}$ (open defect).

| Case (leftover / ($\delta t|g|$)) | before the fix | after the fix (code, projection only) | after the fix (full step, `simulate!`) | assertion |
|---|---|---|---|---|
| CMDGFM, uniform, no solid / gas / liquid | $7\times10^{-12}$ / $1.8\times10^{-11}$ / $2.1\times10^{-11}$ | same (increment unchanged on conforming faces) | $9.1\times10^{-11}$ / $1.3\times10^{-10}$ / $6.3\times10^{-11}$ | $\le10^{-8}$ ✅ |
| CMDGFM, composite, no solid | $1.8\times10^{-11}$ | same | $2.9\times10^{-11}$ | $\le10^{-8}$ ✅ |
| CMDGFM, composite, solid in gas | **0.0292** | $1.09\times10^{-10}$ | $2.9\times10^{-10}$ | $\le10^{-8}$ ✅ |
| CMDGFM, composite, submerged solid | **0.0299** | $4.3\times10^{-11}$ | $1.6\times10^{-10}$ | $\le10^{-8}$ ✅ |
| CMDGFM, CSF drop (constant curvature), composite + sloped solid | **0.0127** | $2.2\times10^{-12}$ | — | $\le10^{-8}$ ✅ (CGFM, uniform, composite: unchanged, $\le1.1\times10^{-12}$) |
| one velocity, six fixtures | $2.2\times10^{-7}$ (uniform) | — | $\le2.2\times10^{-7}$ | $\le10^{-6}$ ✅ |
| mimetic (default), uniform (no solid / gas / liquid) | 0.0175 | 0.0175 | 0.01748 (all three) | $>10^{-3}$ — **open (#76)** |
| mimetic (default), composite (no solid / gas / liquid) | 0.662 / 0.653 / 0.662 | 0.662 / 0.654 / 0.662 | 0.6615 / 0.6537 / 0.6615 | $>10^{-3}$ — **open (#76)** |
| mimetic `:joint_face`, uniform / composite, no solid | $6\times10^{-12}$ / 0.094 | same | — | uniform balanced, composite not |
| interface on a grid line, default mimetic uniform / composite | 0.998 / 1.76 | same | — | $>0.1$ — open |

`run_all.jl`: PASS (97 tests, 3 min 17 s; `eq-forces-two-velocity`).

**Verdicts after the fix.**
- F1 and F2 (CMDGFM, the cut-mesh CSF drop): **resolved by the code**, exactly as the counterfactual of step 6 predicted; the code now realises the equivalence "$\xi\equiv0$, CMDGFM = one-velocity projection with $\gamma^{\rm tv}$, balanced iff the jumps are those of a cell function" with the same $\mathsf R^{\rm tv}$ on both sides.
- F3 (mimetic): **unchanged and open**. The force map was never its cause: it fails identically with the pre-fix and the post-fix operator.
- The new chapter text (two-velocity section of `08_forces.tex`, lines 594-651 at `266c7f8`): checked against these runs. Every number agrees with the macros: `chkStepCmdgfmSolidUniform` $1.3\times10^{-10}$ (mine: max of $1.27\times10^{-10}$, $6.3\times10^{-11}$), `chkStepCmdgfmSolidAmr` $2.9\times10^{-10}$ (mine $2.9\times10^{-10}$, $1.6\times10^{-10}$), `chkStepMimeticSolidAmr` 0.66 (mine 0.654, 0.6615), `chkTensionTwoCmdgfmCut` $2.2\times10^{-12}$ (mine $2.2\times10^{-12}$); the pre-fix values 0.029, 0.030, 0.013 and the post-fix $1.1\times10^{-10}$, $4.3\times10^{-11}$, $2.2\times10^{-12}$ quoted for "projection only" are mine; the congruence $\mathsf R=\mathsf A^{1/2}\mathsf R^{\rm tv}\mathsf A^{1/2}$, "jump unknowns stayed zero", the $a=0.8$ face at the level transition, `:joint_face` 6e-12/0.09 and the grid-line 1.0/1.8 are all reproduced. One wording remark: "the mimetic projection is not balanced with either map" is accurate (rows P3 above). No correction needed.
- Remaining consequence for the code: the scripts assert the open mimetic defect (`@test r > 1e-3`); when #76's mimetic part is fixed these assertions (code_02 P3/P4, code_04 F3) must be inverted.

## Open points

- Cause of the mimetic imbalance (fitted jump rows, `correction_cap`, active-mass set): not derived; chapter 10 is not written. The localisation (uniform: faces adjacent to mixed cells) and the grid-line result (≈1, worse) are the data for it. For the caller: issue #76 first and third bullets are explained (CMDGFM), the second (mimetic) is not due to the map.
- The code fix suggested in F1 changes the semantics of the two-velocity predictor (the increment would then be the gradient of the two-velocity map); momentum-conservation consequences were not examined.

## History

- 2026-11-09 (later): re-audit after the merge of the force-map fix (`266c7f8`): scripts updated (`:code` rows, `code_04_full_step.jl`), CMDGFM assertions now require balance, mimetic assertions kept as the open defect; `run_all.jl` PASS (97 tests).
- 2026-11-09: first audit (`code_01_map_relation.jl`, `code_02_hydrostatic_projection.jl`, `code_03_drop_tension.jl`, `sharp_01_condition_cases.jl`). `run_all.jl`: PASS. The scripts assert the *current* behaviour including the defects of issue #76 (CMDGFM with a solid on a composite mesh fails; mimetic fails): after a code fix the corresponding `@test`s (marked `expected_failure` / `P3` / `P4`) must be inverted.
