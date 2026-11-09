# Audit: eq:forces:stress-balance — sign conventions of the continuous model (with eq:forces:cont-momentum, eq:forces:steady)

- Source: `docs/techref/chapters/08_forces.tex:24-80` (cont-momentum 27, stress-balance 36, steady 54); thesis (2.1b), (2.6), (2.7), (2.9), (4.5), (4.6), (4.8), (7.53)–(7.55); commit `03bad7a`
- Code audited: none for the symbolic part; `src/interface` (cell curvature) and `src/two_phase/forces.jl` for the sign checks
- Date: 2026-11-09
- Overall status: ✅ verified (signs consistent with the discrete conventions and the thesis)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-forces-continuous/scripts/run_all.jl`

## Statement and assumptions

(i) $\rho^\pi(\partial_tv+v\cdot\nabla v)=\nabla\!\cdot S^\pi+\rho^\pi g+\rho^\pi b$, $S=-pI+\mu(\nabla v+\nabla v^\top)$; (ii) $n\cdot[\![S]\!]=\sigma\varkappa n$ with $[\![\cdot]\!]$ = gas minus liquid, $n$ into the gas, $\varkappa=\nabla\!\cdot n$; without viscosity $[\![p]\!]=-\sigma\varkappa$: a convex drop ($\varkappa>0$) has $p^{\rm l}-p^{\rm g}=\sigma\varkappa$; (iii) singular force $\sigma\varkappa\nabla\mathbf 1_{\rm l}=-\sigma\varkappa n\delta$; (iv) steady states (a) $p=p_0+\sigma\varkappa_0\mathbf 1_{\rm l}$, (b) $p=\rho^\pi g\cdot(x-x^I)$, continuous at the planar interface.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| 1. $\nabla\!\cdot(x/|x|)=1/r$ in 2D: convex drop has $\varkappa>0$ | 36-52 | ✅ verified | Symbolics: derivative of $x/\sqrt{x^2+y^2}$ evaluated at 4 points, and the exact identity $2/r-(x^2+y^2)/r^3=1/r$ | `step01_continuous_symbolic.jl` (C1) | — |
| 2. $n\cdot[\![S]\!]=\sigma\varkappa n\iff p^{\rm l}-p^{\rm g}=\sigma\varkappa$, i.e. $[\![p]\!]=-\sigma\varkappa$ | 36-52 | ✅ verified | symbolic ($S=-pI$) | C2 | thesis (2.6), (4.5) agree ($[\![p]\!]=-\sigma\varkappa$) |
| 3. $\nabla\mathbf 1_{\rm l}=-n\delta$; the force $\sigma\varkappa\nabla\mathbf 1_{\rm l}$ points into the liquid | 55-62 | ✅ verified | smoothed indicator $\psi=(1-\tanh(d/\varepsilon))/2$: $\psi'=-\operatorname{sech}^2(d/\varepsilon)/(2\varepsilon)$ checked at 4 points, $\psi(-40)-\psi(40)=1$ | C3 | thesis (4.8): $\lim\nabla\alpha=-\eta\delta$ |
| 4. steady drop: $-\nabla p+\sigma\varkappa_0\nabla\psi=0$ for $p=p_0+\sigma\varkappa_0\psi$ | 63-72 | ✅ verified | symbolic | C4 | consistent with the discrete balance (`eq-forces-st-balance`) |
| 5. steady planar interface: $\nabla(\rho g\cdot x)=\rho g$, $p^\pi=\rho^\pi g\cdot(x-x^I)$ vanishes on the plane in both phases (continuous) | 63-72 | ✅ verified | symbolic | C5 | thesis (7.54)–(7.55) |
| 6. cell curvature of the package is positive and $\approx1/R$ for the drop (mean over mixed cells, 3 % high on the uniform mesh, 1.2 % on the composite mesh) | 40-52 | ✅ verified | package | `code_01_cell_curvature_sign.jl` | 0.4624, 0.4536 vs $1/R=0.4484$ |
| 7. Laplace pressure of the projected force: positive, within 5 % of $\sigma/R$ (real curvature: +1.9 %, +1.0 %), $\sigma/R$ to $5\times10^{-7}$ for constant curvature | 40-52 | ✅ verified | package | same | 0.31996, 0.31715; 0.313901 vs 0.313901 |
| 8. material-constant validation (positive finite $\rho$; $\mu$ both positive or both zero; $\sigma\ge0$; periodic gravity) | 73-81 | ✅ verified | code inspection (`physics.jl:48-84`, `model.jl:34`) and the periodic rejection test (`eq-forces-gravity` G6) | — | single phase: $\mu>0$ strictly (`single_phase/physics.jl:21`) as stated |

## Errors and consequences

No error. All signs of the chapter (gas minus liquid, $n$ into the gas, $\varkappa=\nabla\!\cdot n$, $p^{\rm l}-p^{\rm g}=\sigma\varkappa$, force towards the liquid) are consistent with the discrete conventions of eq:forces:st-jump.

## Sharpness

Not applicable (sign conventions). The only quantitative statements are the 3 %/1 % deviations of the mean cell curvature from $1/R$ (curvature estimator accuracy, not claimed in this chapter).

## Open points

- The viscous term of the stress balance (the footnote in the thesis (2.6)) is outside this chapter.

## History

- 2026-11-09: first audit (`step01_continuous_symbolic.jl`, `code_01_cell_curvature_sign.jl`). `run_all.jl`: PASS (symbolic 17 tests; package 6 tests).
