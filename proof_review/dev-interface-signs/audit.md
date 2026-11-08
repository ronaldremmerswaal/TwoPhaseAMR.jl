# Audit: dev:interface:signs — sign conventions of the height-function formulas ($\kappa=\nabla\!\cdot\!\eta$, contact angle, ghost value)

- Source: `docs/techref/chapters/04_interface.tex:20-45` (eqs. 4.1–4.3), `460-475` (eq. 4.18), `477-499` (eq. 4.19), `526-537` (deviation 4.6, `dev:interface:signs`), commit `1e1886c` on branch `docs/techref-geometry` (chapter unchanged in the working tree; only `proof_review/` is untracked). Thesis: `st_curvature.tex` (printed eq. (4.92), §4.7.2), `basic_navier_stokes.tex` (eqs. (2.6), (2.8)), `int_reconstruction.tex` (eq. (3.60)).
- Date: 2026-11-08
- Overall status: ✅ **the techref's conventions and formulas are consistent and verified**; ✅ **the three printed thesis signs are typographical errors** (each contradicts a premise printed in the same paragraph; no single convention makes the printed set consistent); ✅ **the ghost value (4.18) imposes (4.3) exactly at the wall face**. One imprecision in the deviation text (acute-branch sign is wrong in the thesis only for $s=+1$).
- Rerun: `julia --project=@claude-symbolics proof_review/dev-interface-signs/scripts/run_all.jl` (two sequential child processes: Symbolics/ForwardDiff group under `@claude-symbolics`, package group under the repository environment; flags `--heap-size-hint=3G -t 1`). Last run: all passed.
- Label: the folder is named after `dev:interface:signs` (a deviation box); the equations covered are `eq:interface:graph`, `eq:interface:contact`, `eq:interface:contact-right`, `eq:interface:ghf-result`.

## Statement and assumptions

Conventions of the source (C1) $\eta$ points into the gas, $\kappa=\nabla\!\cdot\eta$ (convex liquid: $\kappa>0$, thesis (2.6) with $\llbracket p\rrbracket=-\sigma\kappa$, $\llbracket a\rrbracket=a^{\rm gas}-a^{\rm liq}$); (C2) $n_w$ is the unit wall normal pointing into the fluid, $\theta$ is measured in the liquid, $\eta\!\cdot\!n_w=\cos\theta$ (4.3); (C3) a graph $x_k=H^{\pm k}(x_t)$ has $\eta=(\pm e_k\mp H'e_t)/\sqrt{1+H'^2}$ and $\kappa=\mp H''/(1+H'^2)^{3/2}$ (4.2). Derived: right-angle branch (4.18): ghost value $H_{-1}=H_0+h_{c,k}\cos\theta/(s\sin\theta)$ so that the height function has $\mathrm dH/\mathrm d\xi=-\cos\theta/(s\sin\theta)$ at the wall ($\xi$ along the inward normal, $s=\operatorname{sgn}(\eta_c)_t$), $\kappa_c=-s\,H''/(1+H'^2)^{3/2}$; acute/obtuse branch (4.19): $\kappa_c=-2s\,a_2/(1+\text{slope}^2)^{3/2}$, $s=\operatorname{sgn}\cos\theta$.

Thesis, as printed: (4.92) "$\kappa=\nabla\!\cdot\!\eta=\nabla\!\cdot\!\big(\tfrac{\pm1}{W}[-H_x;-H_y;1]\big)=\pm\frac{(1+H_y^2)H_{xx}+(1+H_x^2)H_{yy}-2H_xH_yH_{xy}}{W^3}$"; right angle: "$\cos\theta=\eta\!\cdot\!n^\Gamma=-H_x/\sqrt{1+H_x^2}$, which yields $\partial H^{+y}/\partial x=1/\tan\theta$" ($n^\Gamma=e_x$); acute (example $H^{+y}$, $n^\Gamma=e_y$): "$\kappa=2A/\sqrt{1+B^2}^{\,3}$".

Questions: (a) are the printed signs typos relative to a consistent convention; (b) does the code's ghost value impose (4.3).

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| S1 $\nabla\!\cdot(\sigma(-H_x,-H_y,1)/W)=-\sigma E$, $E=[(1+H_y^2)H_{xx}+(1+H_x^2)H_{yy}-2H_xH_yH_{xy}]/W^3$ | thesis (4.92) | ✅ (techref) / ❌ (thesis print) | symbolic (chain rule, reduction mod $W^2=1+H_x^2+H_y^2$) + `Symbolics.derivative` of the explicit expression for a general quadratic $H$ (40 draws, $\sigma=\pm1$) | `step01` | the printed upper sign gives $+E$: wrong for the displayed $\eta$ |
| S2 2-D: $\kappa=-\sigma H''/(1+H'^2)^{3/2}$ for $\eta=\sigma(e_k-H'e_t)/\sqrt{1+H'^2}$ ((4.2)) | 29-35 | ✅ | symbolic derivative, 16 grid points | `step01` | "$\mp$ for $H^{\pm k}$" is right |
| S3 convex disc: $\nabla\!\cdot(x/\lvert x\rvert)=1/r>0$; top of a convex drop ($\sigma=+1$, $H''<0$) has $\kappa>0$ | 22-26 | ✅ | symbolic derivative | `step01` | |
| S4 right-angle slope: $\eta\!\cdot\!n_w=\cos\theta\Rightarrow\mathrm dH/\mathrm d\xi=-\cos\theta/(s\sin\theta)$ | 465-466 | ✅ | symbolic, both $s=\pm1$ ($\rho=\sqrt{1+u^2}=1/\sin\theta$) | `step01` | the thesis' printed $+1/\tan\theta$ is the solution for $s=-1$: it contradicts the thesis' own preceding line ($s=+1$) |
| S5 ghost value (4.18) imposes (4.3) at the wall face: $(H_0-H_{-1})/h=-\cos\theta/(s\sin\theta)$ and the normal built from this slope has $\eta\!\cdot\!n_w=\cos\theta$ | 462-475 | ✅ | symbolic + numeric ($s,\psi=\pm1$, 64 draws) | `step01` | exact in the discrete sense: face-centred difference of (ghost, first cell). The slope $(H_1-H_{-1})/2h$ used in $\kappa$ differs by $O(h)$, so the normal inside $\kappa$ satisfies (4.3) to $O(h)$: measured defect $\lvert\eta\!\cdot\!n_w-\cos\theta\rvert$ $=1.0\mathrm e{-1},4.5\mathrm e{-2},2.2\mathrm e{-2},1.1\mathrm e{-2}$ for $m=1,2,4,8$ at $55^\circ$ (same order at $70^\circ$, $90^\circ$, $125^\circ$; `code_01` (C)) |
| S6 acute branch: $\cos\theta=\sigma/\sqrt{1+H'^2}$ forces $\sigma=\operatorname{sgn}\cos\theta$, $H'^2=\tan^2\theta$; $\kappa=-\sigma\,2A/(1+B^2)^{3/2}$ | 477-499 | ✅ | symbolic/numeric | `step01` | thesis' $+2A/\ldots$ is right only for $\sigma=-1$ |
| S7 convention matrix: statements T1–T5 of the thesis versus (G) $\eta\to$ gas and (L) $\eta\to$ liquid | — | ✅ | `ForwardDiff` divergence of a general quadratic $H$, 200 draws | `step02` | G satisfies T2, T4 (premises) and violates T1, T3, T5 (results); L the complement: the thesis mixes the two |
| S8 (4.3) for the exact sessile circle at both contact points: $\eta\!\cdot\!e_y=\cos\theta$, horizontal height function has $\mathrm dH/\mathrm dy=-\cos\theta/(s\sin\theta)$ (not $+\ldots$), $\nabla\!\cdot\eta=+1/R$ | 37-45, 460-475 | ✅ | package-free geometry, 6 angles × 2 contact points | `code_01` (A) | $\theta\in\{20,55,70,90,125,160\}^\circ$ |
| S9 package: contact curvature $\to+1/R$ at six angles, equal to the independent evaluation | 526-537 | ✅ | package, $m=1,2,4,8$ ($R=6m$ cells) | `code_01` (B) | table below; all contact cells have $\kappa>0$ |
| S10 counterfactuals with the printed thesis signs do not converge | 526-537 | ✅ | package | `code_01` (C) | table below |

Table (S9/S10): relative error $\lvert\kappa R-1\rvert$ over all contact cells, $R=6m$ cells, $m=8$ (finest); columns: code's signs; overall sign flipped (printed (4.92)); ghost slope with the printed $+1/\tan\theta$; acute branch with the printed $+2A$.

| $\theta$ | branch | code, $m=1,2,4,8$ | flipped sign | printed ghost slope, $m=1,2,4,8$ | printed $+2A$ |
|---|---|---|---|---|---|
| 20° | acute | 1.1e-1, 5.2e-2, 4.3e-2, 2.2e-2 | 2.00 | — | 2.00 (liquid wedge: $\kappa\to-1/R$) |
| 55° | right | 5.8e-2, 2.2e-2, 9.9e-3, 4.7e-3 | 2.01 | 9.3, 17.7, 34.5, 68.1 (grows like $1/h$) | — |
| 70° | right | 3.0e-2, 1.1e-2, 4.8e-3, 2.2e-3 | 2.00 | 4.7, 9.0, 17.7, 35.2 | — |
| 90° | right | 1.1e-2, 2.6e-3, 6.5e-4, 1.6e-4 (order 2) | 2.00 | = code ($\cos\theta=0$) | — |
| 125° | right | 2.2e-2, 1.4e-2, 8.0e-3, 4.2e-3 | 2.00 | 7.7, 16.1, 32.8, 66.4 | — |
| 160° | acute (gas wedge) | 3.1e-1, 1.0e-1, 3.9e-2, 2.2e-2 | 2.02 | — | **= code** (the printed sign is right for $s=-1$) |

## Errors and consequences

**No error in the techref**: (4.2), (4.3), (4.18), (4.19), (4.16) are consistent with $\kappa=\nabla\!\cdot\eta$, $\eta\to$ gas, $n_w\to$ fluid, $\theta$ in the liquid (S1–S10).

**Thesis errors (typos), all in the results, none in the premises** (printed pages 108–112; sources `st_curvature.tex`):
1. (4.92): the premise displayed in the same line, $\eta=\pm(-H_x,-H_y,1)/W$, gives $\nabla\!\cdot\eta=\mp E$; the printed $\pm$ is wrong (S1; e.g. $H=-x^2/2$, upper sign: $\kappa(0)=+1$, printed formula $-1$).
2. §4.7.2 right angle: the preceding printed line $\cos\theta=-H_x/\sqrt{1+H_x^2}$ gives $H_x=-1/\tan\theta$, not $+1/\tan\theta$ (S4). With the printed slope the ghost value moves to the wrong side and the curvature error grows like $1/h$ (S10).
3. §4.7.2 acute: for the printed example ($H^{+y}$, liquid wedge, $\eta=(-H_x,1)/\sqrt{\cdot}$) $\nabla\!\cdot\eta=-2A/(1+B^2)^{3/2}$; the printed $+2A$ is correct only for the negative principal direction (obtuse angle, $s=-1$).

Consequence of S7: the printed results T1, T3, T5 are exactly what one obtains with $\eta$ pointing into the liquid ($\kappa_L=-\kappa$), whereas the contact-angle premises T2, T4 of the same paragraphs, thesis (2.6) and the figures/tests (convex drops: $\kappa>0$) use $\eta$ into the gas. So the correction is to the three printed results (as the techref does); the techref's wording "The thesis prints the opposite signs" is right for (4.92) and the right-angle slope but, for the acute branch, only for the liquid wedge (suggested text below). No numerical consequence: the code follows the declared convention and its curvature converges to $+1/R$ (S9).

**Ghost value.** (4.18) imposes (4.3) on the discrete height function exactly: the face-centred difference between the ghost cell and the wall cell equals the contact-angle slope, so the wall-face normal satisfies $\eta\!\cdot\!n_w=\cos\theta$ to rounding (S5). It is a Neumann condition in the finite-difference sense; the curvature then uses the centred slope $(H_1-H_{-1})/2h$, for which (4.3) holds to $O(h)$, consistent with the observed first-order (second order at $90^\circ$) convergence.

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| signs of (4.2), (4.18), (4.19) | **sharp** (forced) | any other sign choice fails: flipped overall sign $\to-1/R$ (error 2.00), opposite ghost slope $\to$ error growing like $1/h$ (68 at $m=8$), $+2A$ for $s=+1$ $\to-1/R$ | `code_01` (C) | — |
| orientation $s=\operatorname{sgn}(\eta_c)_t$ / $\operatorname{sgn}\cos\theta$ | **needed** | both contact points and both phase wedges are covered (right/left, acute/obtuse); with the opposite $s$ the curvature has the wrong sign | `code_01` | — |
| ghost value as a Neumann condition at the wall face | **consistent** (the only consistent choice) | $(H_0-H_{-1})/h=u$ makes $H_1-2H_0+H_{-1}=h^2H''+O(h^3)$; imposing $u$ on the centred slope instead ($H_{-1}=H_1-2hu$) would give $2(H_1-H_0-hu)\approx2h^2H''$, twice the second difference; the slope $(H_1-H_{-1})/2h$ in the denominator of $\kappa$ is first-order accurate, consistent with the observed orders 1.1–1.4 (55°, 70°, 125°) and 2 at 90° ($\cos\theta=0$) | `code_01` (B), by hand | — |

## Open points

- Author's confirmation of the three typos (the techref says "pending"); the matrix S7 shows the thesis mixes conventions, which may help the author decide whether to correct the premises or the results (the code/validation figures are on convention G, so the results).
- Not checked by a script: three dimensions (the package is 2-D). By hand, the 3-D right-angle derivation (thesis (4.95)) shows the same pattern: its printed premise $\cos\theta=\eta^\perp/\sqrt{1+(\eta^\perp)^2}$ gives $\eta^\perp=+1/\tan\theta$ (printed: $-1/\tan\theta$), and then, with $\eta^\perp/\cos\vartheta=-\partial H^{+x}/\partial z$, $\partial H^{+x}/\partial z=-1/(\cos\vartheta\tan\theta)$ (printed: $+$).

## Proposed replacement text

`04_interface.tex:526-537` (deviation 4.6), replace the last three sentences ("The thesis prints the opposite signs ... pending the author's confirmation.") by:

```latex
The thesis prints~\thesis{eq.~(4.92)} $\curv=\pm(\dots)/W^3$, whereas the
preceding line, $\curv=\divergence\inormal$ with $\inormal=\pm(-H_x,-H_y,1)/W$, gives
$\mp$; in \S4.7.2 it prints $\mathrm{d}H^{+y}/\mathrm{d}x=1/\tan\cangle$ right after
$\cos\cangle=-H_x/\sqrt{1+H_x^2}$, which gives $-1/\tan\cangle$; and it prints
$\curv=2A/\sqrt{1+B^2}^{\,3}$ for its acute example, whose positive principal direction
gives $-2A/(1+B^2)^{3/2}$ (the printed sign is correct for the negative direction,
$s=-1$, obtuse angles). Each of the three results contradicts a premise printed
in the same paragraph (the three results are those obtained with $\inormal$ pointing
into the liquid, the premises those with $\inormal$ into the gas), so they
are typographical errors (pending the author's confirmation). The code follows the
definitions above: its curvature converges to $1/R>0$ for sessile drops at
$20^\circ$, $55^\circ$, $70^\circ$, $90^\circ$, $125^\circ$ and $160^\circ$, and the
ghost value~\cref{eq:interface:contact-right} makes the discrete wall slope of the
height function exactly $-\cos\cangle/(s\sin\cangle)$.
```

## History

- 2026-11-08: first audit (symbolic chain-rule divergence and explicit `Symbolics.derivative` for a general quadratic height function, right-angle/ghost/acute algebra for $s=\pm1$, convention matrix with `ForwardDiff`, sessile drops at six angles with counterfactuals). Outcome: ✅ techref consistent; three thesis typos confirmed as typos.
