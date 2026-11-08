# Audit: eq:interface:contact-row — the linearised contact row (4.20) of the constrained GHF fit

- Source: `docs/techref/chapters/04_interface.tex:496-523` (eq. 4.20 at 504-507), commit `1e1886c` on branch `docs/techref-geometry` (chapter unchanged in the working tree; only `proof_review/` is untracked). Thesis: `st_curvature.tex`, "Triple point curvature for arbitrary geometries" (printed eq. (4.97), p. 114).
- Date: 2026-11-08
- Overall status: ⚠️ **the row (4.20) is verified** (algebra = thesis (4.97); the package builds exactly this row; triple point, wall normal and $\cos\theta$ equal an independent evaluation to $4\times10^{-16}$ on 74 triple cells, anisotropic cells included). **One statement is inaccurate**: "added with weight $10^3$" — the code multiplies the row by $10^3$, i.e. weight $10^6$ in the sum (4.15) (numerically immaterial: $\le10^{-5}$ in $\kappa$). "Linearised" should be spelled out (first order in the slope offset); its effect is quantified.
- Rerun: `julia --project=@claude-symbolics proof_review/eq-interface-contact-row/scripts/run_all.jl` (two sequential child processes: Symbolics group under `@claude-symbolics`, package group under the repository environment; flags `--heap-size-hint=3G -t 1`). Last run: all passed.

## Statement and assumptions

In a triple cell the GHF fit (4.15) $\min\sum_jw_j(a_2\tau_j^2+a_1\tau_j+a_0-\nu_j)^2$ in the frame $\tau=\mathbf t\!\cdot\!(\mathbf p-\mathbf o)/h$, $\nu=\boldsymbol\eta\!\cdot\!(\mathbf p-\mathbf o)/h$ ($\mathbf t=(-\eta_2,\eta_1)$, $\mathbf o$ the PLIC midpoint, $h=\min_kh_{c,k}$) gets one more row from the contact condition at the triple point $\mathbf x_*$ (intersection of the interface and solid lines of the cell, if inside the cell), $\tau^*=\mathbf t\!\cdot\!(\mathbf x_*-\mathbf o)/h$, $n_w=\hat S^{-1}s/\lvert\hat S^{-1}s\rvert_2$ the solid normal into the fluid:

$$-2\tau^*(\mathbf t\!\cdot\!n_w)\,a_2-(\mathbf t\!\cdot\!n_w)\,a_1=\cos\theta-\boldsymbol\eta\!\cdot\!n_w\qquad(4.20)$$

"added to (4.15) with weight $10^3$; it is the unnormalised fitted normal $\boldsymbol\eta-(a_1+2a_2\tau^*)\mathbf t$ inserted in (4.3)." If the lines do not meet in the cell the fit is computed without the row; if the fit fails the candidate's curvature is kept. Thesis (4.97): $[1;-B-2A\,S^\top(x_*-x^I_c)]^\top R^\top n^\Gamma=\cos\theta$, $R=[\eta\ S]$, with "a sufficiently large weight".

Assumptions used: seed unit normal $\eta$ (the cell's PLIC normal); $\theta\in(0,\pi)$ measured in the liquid; $n_w$ points into the fluid; fit needs $\ge3$ rows and a smallest singular value $>10^{-10}$ of the largest (the row counts as a row).

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| S1 fitted curve $\nu=a_2\tau^2+a_1\tau+a_0$ has unit normal $(\eta-f't)/\sqrt{1+f'^2}$, $f'=2a_2\tau^*+a_1$; (4.20) is $N\!\cdot\!n_w=\cos\theta$ for the unnormalised $N=\eta-f't$ | 509-511 | ✅ | symbolic (components $\eta=(e_1,e_2)$, $n_w=(w_1,w_2)$) | `step01` | $N\!\perp\!$ tangent, $\lvert N\rvert^2=1+f'^2$ |
| S2 thesis (4.97) $=$ (4.20) | thesis | ✅ | symbolic | `step01` | $R^\top n^\Gamma=(\eta\!\cdot\!n_w,\ t\!\cdot\!n_w)$; the slope is dimensionless, so $\tau^*$ in units of $h$ is the thesis' $S^\top(x_*-x^I_c)$ with $A\to a_2/h$ |
| S3 linearisation: the exact condition $N\!\cdot\!n_w=\cos\theta\sqrt{1+f'^2}$ is replaced by $\cos\theta\cdot1$; the enforced unit normal has $\hat N\!\cdot\!n_w=\cos\theta/\sqrt{1+f'^2}$ (error $-\tfrac12\cos\theta f'^2$), enforced-angle error $\approx\tfrac12f'^2\cot\theta$ | 503, 509-511 | ✅ | symbolic (first and second derivative at $f'=0$) + numeric | `step01` | at $\theta=60^\circ$: $f'=0.05$: $0.041^\circ$; $0.1$: $0.164^\circ$; $0.3$: $1.39^\circ$ |
| S4 a straight interface meeting the wall at $\theta$, seeded with its exact normal, satisfies the row with $a_1=a_2=0$ | — | ✅ | symbolic | `step01` | the row is consistent with exact data |
| S5 the code's row is (4.20) times $\alpha=10^3$ | 500-512; code `height_functions.jl:485-492` | ⚠️ | package | `step01` (S5), `code_01` (B) | weight in the functional $=\alpha^2=10^6$ (see Findings) |
| S6 package triple point, $n_w$, $\cos\theta$ (`_embedded_ghf_constraint`) = independent evaluation (2×2 solve in physical coordinates from the stored planes, $\tau^*=\mathbf t\!\cdot\!(x_*-o)/h$, $n_w=(s_1/h_x,s_2/h_y)/\lvert\cdot\rvert$) | 500-512 | ✅ | package, 40 random walls (slopes $\pm0.05\ldots0.55$), angles $35^\circ$–$135^\circ$, isotropic and anisotropic cells | `code_01` (A) | 74 triple cells with a row: max deviation $4.0\times10^{-16}$; cells where the lines do not meet inside the cell have no row (`nothing`) exactly as stated |
| S7 package fit = independent weighted least squares (4.15)+(4.20) with the row scaled by $10^3$ | 500-523 | ✅ | package, 60 fixtures; the independent solve mirrors the package's point selection (height points, then PLIC midpoints of rings 1, 2, 3, unit weights) and the package's own `_ghf_geometry` reproduces the fit | `code_01` (B) | 111 fits: $\max\lvert\kappa_{\rm code}-\kappa(\text{row}\times10^3)\rvert=0.0$; scaling $\sqrt{10^3}$ (weight $10^3$) differs by up to $1.0\times10^{-5}$ ($\kappa\approx0.25$) |
| S8 the row is needed: without it 87 of the 111 fits have fewer than three rows (cut neighbours give no height points); where a fit without the row exists it changes $\kappa$ by up to 0.33 | 500-523 | ✅ | package | `code_01` (B) | stages used: height points 5, ring-1 centroids 34, ring-2 64, ring-3 8 |
| S9 enforcement: row residual $1.8\times10^{-8}$ (max) at the code's scaling; $\lvert\text{angle}_{10^3}-\text{angle}_{\rm eq.\text{-}constrained}\rvert\le2.8\times10^{-8}$ rad | 512-513 | ✅ | package vs equality-constrained limit (KKT) | `code_01` (C) | residual scales like $\alpha^{-2}$: $1.6\times10^{-2}$ ($\alpha=1$), $1.8\times10^{-5}$ ($\sqrt{10^3}$), $1.8\times10^{-8}$ ($10^3$), $1.8\times10^{-14}$ ($10^6$) |
| S10 enforced exact angle: median error $2.9\times10^{-3}$ rad ($0.17^\circ$), 90 % $6.3\times10^{-2}$ rad ($3.6^\circ$), max 0.58 rad; $\lvert f'\rvert$ median 0.15, 90 % 0.59, max 2.7 | 503 | ✅ (information) | package, 112 fits | `code_01` (C) | dominated by cells with a rough seed normal: median 1.1e-3 rad for $\lvert f'\rvert<0.3$ (84 fits) vs 5.0e-2 rad for $\lvert f'\rvert\ge0.3$ (28 fits) |

## Errors and consequences

No error in (4.20) or in the geometry of the constraint. Findings:

**F1 — the weight.** Source: "added to (4.15) with weight $10^3$". Code (`height_functions.jl:485-492`): `weight = T(1.0e3)` multiplies the row of the matrix and of the right-hand side, while the height rows are scaled by $\sqrt{w_j}$. In the functional $\sum_jw_jr_j^2$ of (4.15) the row therefore enters with weight $\alpha^2=10^6$, not $10^3$. The two readings give curvatures that differ by at most $1.0\times10^{-5}$ on 111 fits ($\kappa\approx0.25$), and the row is enforced to $\le1.8\times10^{-8}$ (weight $10^6$) versus $\le1.8\times10^{-5}$ (weight $10^3$): immaterial numerically, but the text should state what is implemented. (The check in `figures/cutcells.jl` multiplies the row by $10^3$ too, so it agrees with the code by construction; it cannot detect this.) With exactly three rows (two height points) the system is square and the weight plays no role.

**F2 — "linearised".** (4.20) is the first-order (in $f'$) expansion of the exact condition $(N/\lvert N\rvert)\!\cdot\!n_w=\cos\theta$: the fitted unit normal at $\mathbf x_*$ satisfies $\hat N\!\cdot\!n_w=\cos\theta/\sqrt{1+f'^2}$, an angle error $\approx\frac12f'^2\cot\theta$ (S3). In the fixtures $f'$ is not small in a sizeable fraction of triple cells (median 0.15, 90 % 0.59, max 2.7), giving median/90 %/max angle errors of $0.17^\circ/3.6^\circ/33^\circ$. $f'$ is the slope of the fitted curve at $\mathbf x_*$ relative to the seed normal: it contains the genuine turning of the interface between the PLIC midpoint and the triple point (at most $\approx\kappa\cdot0.7h\approx0.2$ for $R\ge3.5$ cells) and the error of the seed normal, so values above 0.3 (28 of 112 fits) must come from the seed. Imposing the exact condition by fixed-point iteration (`sharp_01`) reduces the median angle error to $1.7\times10^{-8}$ rad after four iterations but does **not** improve the curvature (median $\lvert\kappa-1/R\rvert$ $4.0\times10^{-2}\to4.2\times10^{-2}$): on these fixtures ($R=3.5$–$5$ cells) the discretisation error of the fit dominates, so the linearisation is benign for $\kappa$.

Consequence: none for correctness; wording only.

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| row (4.20) $=N\!\cdot\!n_w=\cos\theta$ | **sharp** (an identity) | symbolic; package equals the independent solve exactly | `step01`, `code_01` | — |
| weight "$10^3$" | **imprecise** | implemented $\alpha=10^3$ (weight $10^6$); $\kappa$ differs by $\le1.0\times10^{-5}$ from the weight-$10^3$ reading | `code_01` (B) | say "row multiplied by $10^3$ (weight $10^6$)" |
| row enforced "to ensure that it really is enforced" (thesis) | **sufficient** | residual $1.8\times10^{-8}$ (max over 112 fits); equal to the hard constraint to $2.8\times10^{-8}$ rad | `code_01` (C) | $\alpha=10^6$ would give $10^{-14}$, no benefit |
| linearisation $\sqrt{1+f'^2}\to1$ | **not sharp** (improvement exists) but benign | exact angle by 4 fixed-point iterations ($\mathrm{rhs}_k=\cos\theta\sqrt{1+f_{k-1}'^2}-\eta\!\cdot\!n_w$): median angle error $2.9\times10^{-3}\to1.7\times10^{-8}$ rad; **no** gain in $\kappa$ | `sharp_01` | not recommended: no accuracy benefit observed |
| the row is necessary | **sharp** | 87 of 111 triple fits do not exist without it; where they exist $\kappa$ changes by up to 0.33 | `code_01` (B) | — |
| triple point must lie inside the cell | needed | outside: no row (`nothing`), as stated; the determinant test $\lvert\det\rvert>256\varepsilon\lvert\cdot\rvert\lvert\cdot\rvert$ and the bound $\lvert x\rvert\le h(\frac12+256\varepsilon)$ are the only tolerances | `code_01` (A) | — |

## Open points

- The $\alpha$ in `_ghf_fit` is a magic number; state it in the implementation note ("$10^3$ times the row, i.e. weight $10^6$").
- The 28 fits with $\lvert f'\rvert\ge0.3$ (seed-normal error above ≈0.1 rad, see F2) are a limitation of the seed (WY/GHF, `:lhf_ghf`), not of the row; ch. 4 already documents the WY-line limitation in triple cells.
- Not checked: the thesis' claim for 3-D triple points (the package is 2-D).

## Proposed replacement text

`04_interface.tex:503-513`, replace "the linearised condition~\cref{eq:interface:contact} on the fitted normal is ... added to~\cref{eq:interface:ghf} with weight $10^3$; it is the unnormalised fitted normal ... inserted in~\cref{eq:interface:contact}." by:

```latex
the linearised condition~\cref{eq:interface:contact} on the fitted normal is
\begin{equation}\label{eq:interface:contact-row}
  -2\tcoord^*(\itangent\cdot\wallnormal)\,\parcoef{2}-(\itangent\cdot\wallnormal)\,\parcoef{1}
    =\cos\cangle-\inormal\cdot\wallnormal,
\end{equation}
added to~\cref{eq:interface:ghf} as a row multiplied by $10^3$ (weight $10^6$ in the
sum, the height rows being multiplied by $\sqrt{\fitweight{j}}$). It is the unnormalised
fitted normal $\inormal-(\parcoef{1}+2\parcoef{2}\tcoord^*)\itangent$ inserted
in~\cref{eq:interface:contact}, i.e.\ the exact condition with the factor
$\sqrt{1+(\parcoef{1}+2\parcoef{2}\tcoord^*)^2}$ on its right-hand side replaced by one:
the fitted unit normal at $\triplept$ makes the angle $\cangle'$ with
$\wallnormal$ where $\cos\cangle'=\cos\cangle/\sqrt{1+(\parcoef{1}+2\parcoef{2}\tcoord^*)^2}$, an error
$\approx\frac12(\parcoef{1}+2\parcoef{2}\tcoord^*)^2\cot\cangle$ of the angle. If the two lines
do not meet in the cell the fit is computed without this row; if the fit fails, the
candidate's curvature is kept.
```

## History

- 2026-11-08: first audit (symbolic derivation and linearisation error; package comparison on 74 triple cells for the constraint and 111 fits for the least-squares solve with the independent point selection; weight/scaling analysis; fixed-point enforcement experiment). Outcome: ⚠️ verified; F1 (weight) and F2 (linearisation wording).
