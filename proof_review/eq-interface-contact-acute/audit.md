# Audit: eq:interface:contact-acute — the rationalised contact-parabola coefficient $A_2$ (4.19) and the evaluation at the wall cell

- Source: `docs/techref/chapters/04_interface.tex:473-499` (eq. 4.19) and `539-546` (deviation 4.7, `dev:interface:contact-acute`), commit `1e1886c` on branch `docs/techref-geometry` (the chapter is unchanged in the working tree; only `proof_review/` is untracked). Thesis: `st_curvature.tex`, "Extremely acute and obtuse contact angles in 2D" (printed p. 111–112, eq. (4.94)).
- Date: 2026-11-08
- Overall status: ⚠️ **the formula (4.19) is verified** (identical to the thesis root $A^-$; code = (4.19) to $5\times10^{-16}$ on 104 contact cells and 1303 injected height pairs). Two wording issues (root selection for obtuse angles; sufficiency of $h_2>h_1$) and the **quantified consequence** of the evaluation point (first order, no uniform advantage over the thesis' point, not optimal).
- Rerun: `julia --project=@claude-symbolics proof_review/eq-interface-contact-acute/scripts/run_all.jl` (two sequential child processes: Symbolics group under `@claude-symbolics`, package group under the repository environment; flags `--heap-size-hint=3G -t 1`). Last run: all passed.

## Statement and assumptions

Notation of the source (thesis in brackets): wall cell $c$, wall normal axis $k$, tangential axis $t$, $w=h_{c,t}$ ($h$). The wedge phase $s=\operatorname{sgn}\cos\theta$ (liquid for acute, gas for obtuse $\theta$) is measured by heights from the wall in the next two columns away from the contact point, $h_m=h_{c,k}\bar H_m$ ($m=1,2$; $h_1=H_i-H_*$, $h_2=H_{i+1}-H_*$). With $x_t$ measured from the first column away from the contact point, the parabola $H=a_2x_t^2+a_1x_t+h_1$ ($a_2=A$, $a_1=B$) through $(0,h_1)$, $(w,h_2)$ vanishing with slope $T=\lvert\tan\theta\rvert$ at an unknown point $x_*$ has

$$a_2=\frac{(h_2-h_1)^2-w^2T^2}{w^2\big(h_1+h_2+\sqrt{4h_1h_2+w^2T^2}\big)},\quad a_1=\frac{h_2-h_1}{w}-a_2w,\quad \kappa_c=\frac{-2s\,a_2}{\big(1+(a_1-2a_2w)^2\big)^{3/2}}\qquad(4.19)$$

"required $h_2>h_1>0$. This is the thesis' root with $a_1\tan\theta>0$ [eq. (4.94)] in a rationalised form; the slope is evaluated at the wall cell, $x_t=-w$." Deviation 4.7: the thesis evaluates $\kappa=2a_2/(1+a_1^2)^{3/2}$ (slope at the first column), the code one cell closer to the contact point.

Claims audited: (i) (4.19) equals the thesis root selected by (4.94); (ii) what evaluating at $x_t=-w$ does.

Assumptions actually used (where): $h_1,h_2>0$ and $T>0$ (so $D=4h_1h_2+w^2T^2>0$ and $a_2$ is defined; in the branch $\lvert\theta-\pi/2\rvert>\pi/4$ one has $0<T<1$); $h_2>h_1$ in the code (`H1 > H0 > 0 || continue`) to select the physical configuration; $w$ is the width along the wall; $h_m$ in physical units ($h_{c,k}\bar H_m$), anisotropic cells allowed (verified).

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| S1 elimination of the contact point: $H(x_*)=0$, $H'(x_*)=T\Rightarrow b^2=t^2+4ap$ ($p=h_1$, $a=a_2w^2$, $b=a_1w$, $t=wT$) | thesis | ✅ | symbolic | `step01` | $(2a\xi+b)^2-(b^2-4ap)=4a(a\xi^2+b\xi+p)$ |
| S2 roots $a_2^\pm=(h_1+h_2\pm\sqrt D)/w^2$, $a_1^\pm=-(2h_1\pm\sqrt D)/w$; $a_1^+a_1^-=-4h_1(h_2-h_1)/w^2-T^2<0$ | thesis | ✅ | symbolic (reduction modulo $r^2=D$) | `step01` | all printed thesis formulas reproduced |
| S3 (4.19) $=a_2^-$ (the rationalisation): $((h_2-h_1)^2-w^2T^2)-(h_1+h_2-r)(h_1+h_2+r)=0$ mod $r^2=D$ | 483-492 | ✅ | symbolic + 2000 random 512-bit draws ($<10^{-120}$) | `step01` | $a_1$ of (4.19) is $a_1^-=(r-2h_1)/w$ |
| S4 sign of the roots: $a_1^->0$ iff $4h_1(h_2-h_1)+w^2T^2>0$; $a_1^+<0$ always | 493-495 | ✅ | symbolic + numeric (2000 draws) | `step01` | $h_2>h_1$ is sufficient, not necessary (example $h_1=1,h_2=0.95,T=0.5$) |
| S5 contact point $x_*=-2h_1/(a_1+T)$ for both roots: the minus root puts it before the first column ($x_*<0$), the plus root beyond it ($x_*>0$, impossible for a wall contact since $H(0)=h_1>0$) | — | ✅ | symbolic + 10 007 numeric checks | `step01` | so "the root with $a_1>0$" is the physical one |
| S6 literal reading "root with $a_1\tan\theta>0$" | 493-494 | ⚠️ | S4 | `step01` | true as printed only for acute $\theta$ ($\tan\theta>0$). For obtuse $\theta$ the code measures the **gas** wedge with slope $\lvert\tan\theta\rvert$, $a_1>0$, while $\tan\theta<0$: say "slope $\lvert\tan\theta\rvert$" (replacement below) |
| S7 code = (4.19): pipeline on sessile drops at random $\theta\in(8^\circ,44^\circ)\cup(136^\circ,172^\circ)$, anisotropic cells, both contact points, clipped as (4.17) | 477-499 | ✅ | package | `code_01` | 104 contact cells: max relative deviation $4.6\times10^{-16}$ from (4.19), $6.9\times10^{-16}$ from the thesis root written without rationalisation (256-bit) |
| S8 the branch itself with injected admissible heights, anisotropic cell widths, $s=\pm1$; inadmissible $h_2\le h_1$ rejected | 477-499 | ✅ | package (`_contact_curvature` on a modified height array) | `code_01` | 1303 admissible pairs equal (4.19) to $10^{-11}$; 217 pairs with $h_2\le h_1$ return `nothing` |
| S9 direction "away from the contact point" (rule `-(sgn seed)·s`) | 477-482 | ✅ | package vs the geometry of the exact circle (right/left contact points, acute/obtuse) | `code_01` | not stated in the `.tex`; the geometry-derived columns reproduce the code on all 104 cells |
| S10 slope at $x_t=-w$ is what the code evaluates ($B-2Ah$) | 496-498, 539-546 | ✅ | package | `sharp_01` | the independent value with $x_e=-1$ equals the stored curvature to $10^{-8}$ on every cell |
| S11 evaluation point changes $\kappa$ by the relative amount $6wa_2a_1/(1+a_1^2)+O(w^2)$, i.e. $-3w\kappa a_1\sqrt{1+a_1^2}$ for a drop | — | ✅ | symbolic derivative + numeric (50 + 20 draws) | `step01` | first order in $h$ |
| S12 "rationalised form avoids cancellation" (code comment `height_functions.jl:795`) | — | ⚠️ partly | 512-bit reference, near-straight interfaces | `sharp_02` | removes the cancellation of $h_1+h_2-\sqrt D$ but not the one of the data; gain ≈ 10×, immaterial (see Sharpness) |

## Errors and consequences

No error in (4.19) or in the code. Findings:

**F1 (wording).** "the thesis' root with $a_1\tan\theta>0$" is literally true only for $\tan\theta>0$. The code (and (4.19) as written, with slope $\lvert\tan\theta\rvert$ and $a_1>0$ for both orientations) corresponds to the thesis' root with $B\cdot(\text{slope at the contact point})>0$, the slope being $+\lvert\tan\theta\rvert$ of the wedge phase. In the thesis the conditions are written for the "$+y$" orientation, "$H'(x_*)=\tan\theta$"; for obtuse angles the thesis text uses $\pm\tan\theta$ for the two orientations (S6). Suggested text below.

**F2 (precision).** $h_2>h_1>0$ is sufficient for the physical root but not necessary: $a_1^->0\iff4h_1(h_2-h_1)+w^2T^2>0$. Harmless (the code requires $h_2>h_1$), but the text says "required".

**Consequence of the evaluation point (Deviation 4.7), quantified** (`sharp_01`; sessile drops, $R=6$ base cells, refinement $h=1/m$, 8 positions of the contact point in its cell, both contact points, $\theta\in\{10,20,30,150,160,170\}^\circ$; error $=\max\lvert\kappa-1/R\rvert$ over all contact cells, exact $1/R=0.1667$):

| $\theta$ | thesis point $x_e=0$ ($m=1,2,4,8,16$) | code $x_e=-w$ ($m=1,2,4,8,16$) | observed order thesis / code ($m=8\to16$) | best $x_e/w$ at $m=16$ and its error |
|---|---|---|---|---|
| 10° | 1.2e-2, 6.3e-3, 3.8e-3, 2.1e-3, **1.13e-3** | 4.4e-3, 3.1e-3, 2.0e-3, 1.05e-3, **5.6e-4** | 0.90 / 0.92 | −0.85 → 4.5e-4 |
| 20° | 1.6e-2, 1.3e-2, 7.2e-3, 3.7e-3, **1.84e-3** | 2.4e-2, 1.4e-2, 7.0e-3, 3.7e-3, **1.85e-3** | 1.02 / 1.01 | −0.50 → 9.0e-4 |
| 30° | 3.2e-2, 1.7e-2, 8.3e-3, 4.3e-3, **2.19e-3** | 4.0e-2, 2.3e-2, 1.3e-2, 6.6e-3, **3.37e-3** | 0.97 / 0.96 | −0.35 → 1.2e-3 |
| 150° | 1.7e-2, 1.7e-2, 7.7e-3, 3.7e-3, **2.06e-3** | 1.1e-1, 3.5e-2, 1.6e-2, 7.3e-3, **3.55e-3** | 0.86 / 1.04 | −0.30 → 1.1e-3 |
| 160° | 3.3e-2, 1.5e-2, 7.7e-3, 3.7e-3, **1.91e-3** | 5.3e-2, 1.8e-2, 8.3e-3, 3.9e-3, **1.96e-3** | 0.97 / 1.00 | −0.45 → 9.4e-4 |
| 170° | 2.9e-2, 1.4e-2, 6.9e-3, 3.1e-3, **1.39e-3** | 2.9e-2, 6.9e-3, 2.7e-3, 1.2e-3, **5.96e-4** | 1.14 / 1.06 | −0.85 → 4.7e-4 |

(1) both variants are first-order (observed orders 0.9–1.15 for $m\ge4$); (2) the code's point is not uniformly better: better by 2.0–2.3× at $10^\circ,170^\circ$, equal at $20^\circ,160^\circ$, worse by 1.5–1.7× at $30^\circ,150^\circ$ than the thesis' point; (3) neither is optimal: the minimiser of the error lies between $x_e=-0.30w$ and $-0.85w$ depending on $\theta$, with an error 1.2–3.3× below the code's and 1.9–3.0× below the thesis'. The relative change between the two evaluation points is $6wa_2a_1/(1+a_1^2)\approx-3w\kappa a_1\sqrt{1+a_1^2}$, i.e. $O(h\kappa)$; it cannot change the order. The tests' claim "converges with order at least 0.7" is consistent with these numbers.

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| root choice $a_1>0$ ($x_*<0$) | **sharp** (necessary) | the other root has $x_*>0$ for all admissible data (symbolic sign analysis, 2000 random draws) | `step01` | — |
| $h_2>h_1>0$ | **not sharp** (sufficient only) | $a_1^->0\iff4h_1(h_2-h_1)+w^2T^2>0$; $h_1=1,h_2=0.95,T=0.5$ has $a_1^->0$ | `step01` | state the weaker condition (the code's check can stay) |
| formula (4.19) | **sharp** (an identity) | symbolic + 256/512-bit | `step01`, `code_01` | — |
| rationalisation | **modest** (not a removal of the cancellation) | near-straight interfaces ($h_2=(x_1+w)T+\delta w$): relative error of $a_2$ grows like $\varepsilon/\delta$ for both forms, the rationalised one ≈10× smaller (e.g. $\delta=10^{-10}$: $3.8\times10^{-6}$ vs $4.3\times10^{-7}$); absolute error of the direct form $\le1.07\,\varepsilon(h_1+h_2)/w^2$ ($\approx10^{-13}$ for $w=10^{-3}$) | `sharp_02` | keep (harmless); do not claim "avoids cancellation" in the text |
| first-order accuracy of the contact curvature | **likely sharp** (numerical) | observed order 0.9–1.15 for all six angles and both evaluation points, $m=4\ldots16$ | `sharp_01` | not investigated (first order is what the thesis reports for this closure) |
| evaluation at $x_t=-w$ | **not sharp** | optimum $x_e^*(\theta)\in[-0.85,-0.30]\,w$ (symmetric under $\theta\to\pi-\theta$): error at the code's point is 1.2–3.3× that at the optimum (10°/170°: 1.2–1.3×; 30°/150°: 2.9–3.3×), at the thesis' point 1.9–3.0× | `sharp_01` | $x_t=-\tfrac12w$ halves the error of the code's point for $20^\circ$–$160^\circ$ (and of the thesis' point at $20^\circ,160^\circ$; 1.2–1.3× at $30^\circ,150^\circ$) but is 1.3–1.4× worse than the code's at $10^\circ,170^\circ$ ($7.3\times10^{-4}$ vs $5.6\times10^{-4}$): no single value is optimal for all $\theta$ |

## Open points

- Whether the choice $x_t=-w$ was a deliberate design (curvature "at the wall cell") or a convention: the data show first-order accuracy either way; if accuracy at acute/obtuse angles matters, $x_t=-w/2$ is a one-line change worth benchmarking in the full sessile-drop tests (not done here: the test of `height_functions.jl:369` quotes order ≥0.7).
- For the `literature-scout`: Afkhami–Bussmann (2008) and Afkhami et al. (2009) for height-function contact-angle closures at $\lvert\theta-\pi/2\rvert>\pi/4$ (the thesis cites [17]).
- Not checked: the three-dimensional analogues (the package is 2-D); parabola evaluation in cells where the contact point lies in the neighbouring column.

## Proposed replacement text

`04_interface.tex:493-499`, replace "required $\hphys{2}>\hphys{1}>0$. This is the thesis' root ... whose branch has valid heights determines the curvature." by:

```latex
required $\hphys{2}>\hphys{1}>0$. The parabolas through both heights with the
slope $\abs{\tan\cangle}$ at an unknown zero have
$\parcoef{2}=(\hphys{1}+\hphys{2}\pm\sqrt{D})/\cellwidth{c}{t}^2$,
$D=4\hphys{1}\hphys{2}+\cellwidth{c}{t}^2\tan^2\cangle$~\thesis{\S4.7.2};
\cref{eq:interface:contact-acute} is the minus sign, written without the
subtraction of nearly equal numbers. It is the thesis' root with
$B\tan\cangle>0$~\thesis{eq.~(4.94)}, where the slope of the wedge phase is
$\abs{\tan\cangle}$, i.e.\ the root with $\parcoef{1}>0$: its zero lies at
$\coordk{t}=-2\hphys{1}/(\parcoef{1}+\abs{\tan\cangle})<0$, before the first
column, whereas the other root has it beyond. The slope in the curvature is
evaluated at the wall cell, $\coordk{t}=-\cellwidth{c}{t}$
(\cref{dev:interface:contact-acute}). The first wall (left, right, bottom, top)
whose branch has valid heights determines the curvature.
```

Deviation 4.7 (`04_interface.tex:539-546`), replace by:

```latex
The thesis evaluates the curvature of the contact parabola with the slope at the
first height column, $\curv=2\parcoef{2}/(1+\parcoef{1}^2)^{3/2}$ in the notation
used here (sign as printed, see \cref{dev:interface:signs})~\thesis{\S4.7.2};
the code evaluates the slope at the column of the wall cell, one cell closer to the
contact point~\cref{eq:interface:contact-acute}. The two values differ by the
relative amount $6\cellwidth{c}{t}\parcoef{2}\parcoef{1}/(1+\parcoef{1}^2)$, of
order $\cellwidth{c}{t}\abs{\curv}$. Both are first-order accurate: on
sessile drops with $R/h=96$ at $\cangle=10^\circ,20^\circ,30^\circ,150^\circ,160^\circ,170^\circ$ the largest
curvature errors over the contact cells are $5.6\times10^{-4}$ to $3.6\times10^{-3}$ (code) and
$1.1\times10^{-3}$ to $2.2\times10^{-3}$ (thesis) for $1/R=0.167$, neither
uniformly smaller.
```


## History

- 2026-11-08: first audit (Symbolics reduction of the root structure, package comparison on 104 sessile-drop contact cells with random angles/anisotropy, 1303 injected height pairs, convergence study $m=1\ldots16$ for six angles, rationalisation accuracy in 512-bit). Outcome: ⚠️ formula verified; wording F1/F2; evaluation point quantified.
