# Audit: eq:interface:area — closed-form PLIC area fraction (4.8), reflected branch, and the plane-constant inverse (4.9)

- Source: `docs/techref/chapters/04_interface.tex:136-190` (eqs. 4.8 at 142-155, 4.9 at 158-161 and the checks paragraph 181-190), commit `1e1886c` on branch `docs/techref-geometry` (chapter unchanged in the working tree; only `proof_review/` is untracked). Reference: Scardovelli & Zaleski (2000), cited in the source.
- Date: 2026-11-08
- Overall status: ✅ **verified**: (4.8) is exactly the distribution function of the sum of two uniform variables (symbolic regime-by-regime proof), the code equals it to the rounding bound on 40 000 random draws, the closed-form inverse is accurate to $1.7\times10^{-16}\lvert n\rvert_1$ over the whole range of fractions. One imprecision in the checks paragraph (where and why the bisection loses accuracy) and one remark that needs a qualifier ("the reflected branch avoids cancellation").
- Rerun: `julia --project=@claude-symbolics proof_review/eq-interface-area/scripts/run_all.jl` (two sequential child processes: Symbolics group under `@claude-symbolics`, package group under the repository environment; flags `--heap-size-hint=3G -t 1`). Last run: all passed.

## Statement and assumptions

For a rectangle $\prod_k[\hat r^-_k,\hat r^+_k]$ and the line $\hat n\!\cdot\!\hat\xi\le\hat a$ let $w_k=\lvert\hat n_k\rvert(\hat r^+_k-\hat r^-_k)$, $w_<=\min_kw_k$, $w_>=\max_kw_k$, $z=\hat a-\sum_k\min(\hat n_k\hat r^-_k,\hat n_k\hat r^+_k)$ (plane constant measured from the lowest corner). Then (4.8) $\mathcal A=0$ ($z\le0$), $F(z)$ ($0<z\le\frac12(w_1+w_2)$), $1-F(w_1+w_2-z)$ ($\frac12(w_1+w_2)<z<w_1+w_2$), $1$ ($z\ge w_1+w_2$), with $F(z)=z^2/(2w_<w_>)$ for $z<w_<$, $F(z)=(z-w_</2)/w_>$ for $z\ge w_<$; for $w_<\le\varepsilon w_>$ the code uses $\operatorname{clamp}(z/w_>,0,1)$; "the reflected branch avoids cancellation near full cells". The plane constant is the root of (4.9) found by 54 bisection steps on $[-r,r]$, $r=\frac12\sum_k\lvert\hat n_k\rvert$ (uncut) or on the range of $\hat n\!\cdot\!\hat\xi$ over the vertices of the open polygon (cut cells, clipped-polygon area). Check paragraph: bisection vs the closed-form Scardovelli–Zaleski inverse differ by $\le\chi_{\rm PlaneInverse}$ for $10^{-2}\le C\le1-10^{-2}$ ("near pure cells the plane constant is ill-conditioned, $\partial\hat a/\partial C\sim C^{-1/2}$").

Assumptions used: $\hat n\neq0$; any signs of $\hat n_k$ (handled by the minimum in $z$); any rectangle (not only sub-rectangles of the reference cell; used by the height-value halos (4.10) and strips (7.5)); $\hat n$ unnormalised (the formula is scale-invariant in $(\hat n,\hat a)$).

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| S1 change of variables: $\hat n\!\cdot\!\hat\xi=\sum_k\min(\ldots)+U_1+U_2$, $U_k$ uniform on $[0,w_k]$, so $\mathcal A=P(U_1+U_2\le z)$ | 136-141 | ✅ | argument + numerics (all sign patterns) | `code_01` | |
| S2 regimes of $P(w_1U_1'+\ldots)$: exact shoelace areas $z^2/(2w_1w_2)$ ($z\le w_<$), $(z-w_</2)/w_>$ ($w_<\le z\le w_>$), $1-(w_1+w_2-z)^2/(2w_1w_2)$ ($w_>\le z\le w_1+w_2$) equal the branches of (4.8) incl. the case where the linear regime lies in the reflected branch | 142-153 | ✅ | symbolic (Symbolics shoelace in $w_1,w_2,z$, $w_1\le w_2$ w.l.o.g.) | `step01` | the branch conditions are monotone in $z$; checked at the regime ends |
| S3 reflection: $F(z)+F(w_1+w_2-z)=1$ for the exact areas | 146-148 | ✅ | symbolic | `step01` | |
| S4 continuity and $C^1$ matching at $z=w_<$, $z=w_>$ and at the cut $z=\frac12(w_1+w_2)$ | — | ✅ | symbolic | `step01` | density $1/w_>$ on $[w_<,w_>]$, total mass 1 |
| S5 closed-form inverse: $z=\sqrt{2w_<w_>C}$, $w_>C+w_</2$, $w_<+w_>-\sqrt{2w_<w_>(1-C)}$ with breakpoints $C=w_</(2w_>)$, $1-w_</(2w_>)$ | (check paragraph) | ✅ | symbolic ($F(F^{-1}(C))=C$ on all three regimes, junction values) + 256-bit closed form is the root of the exact polygon area (200 draws, $<10^{-60}$) | `step01`, `code_02` | |
| S6 conditioning: $\mathrm dz/\mathrm dC=\sqrt{w_<w_>/(2C)}$ (corner regime), $w_>$ (linear regime), $\sqrt{w_<w_>/(2(1-C))}$ near full | 181-190 | ✅ | symbolic | `step01` | symmetric about $C=\frac12$ |
| S7 `_plane_fraction` = (4.8) = polygon area | 142-156 | ✅ | package vs 256-bit Sutherland–Hodgman reference, 40 000 draws (random normals of all sign patterns, near-axis-aligned, 45°, exactly aligned; random rectangles incl. far from the origin; plane constants from below to above the cell) | `code_01` | max $\lvert\text{code}-\text{ref}\rvert=1.9\times10^{-13}$ only for rectangles far from the origin; always $\le5.5\,\varepsilon\,(1+\max(\lvert a\rvert,\lvert\text{shift}\rvert)/w)$ (rounding of $z=a-\text{shift}$); a literal transcription of (4.8) from the `.tex` gives exactly the same values as the code ($\max\lvert\cdot\rvert=0$ over the 40 000 draws) |
| S8 degenerate normals $w_<\le\varepsilon w_>$: $\operatorname{clamp}(z/w_>,0,1)$ | 154-156 | ✅ | package | `code_01` | neglected term $w_</(2w_>)\le\varepsilon/2$ |
| S9 branch points, symmetry $F(a)+F(-a)=1$, monotonicity, $F=\frac12$ at the centre line | — | ✅ | package, 2000 random normals | `code_01` | |
| S10 54 bisection steps on $[-r,r]$: area residual $\lvert F(a)-C\rvert\le3.3\times10^{-16}$ for all $C$ from $10^{-14}$ to $1-10^{-14}$ | 157-168 | ✅ | package (134 normals × 17 fractions) | `code_02` | |
| S11 bisection vs closed form $\le10^{-14}$ for $10^{-2}\le C\le1-10^{-2}$ | 181-190 | ✅ (statement) ⚠️ (explanation) | package vs 256-bit truth | `code_02` | max $3.5\times10^{-16}\lvert\hat n\rvert_1$ in that range; the ill-conditioning is not symmetric, see Findings |
| S12 cut cells: $\lvert L_c\rvert=C\lvert O_c\rvert$ to rounding | 158-168 | ✅ | package, cylinder fixture, 20 cut cells × 200 random $(C,\hat n)$, independent clipping and shoelace | `code_03` | max residual $4.4\times10^{-16}$; the intercept stays in the projection range of the polygon; $C=0,1$ return the range ends |
| S13 "the reflected branch avoids cancellation near full cells" | 155-157 | ⚠️ (qualifier) | package vs 256-bit, strongly anisotropic widths | `sharp_01` | true relative to inclusion–exclusion; identical to the explicit third regime |

## Errors and consequences

No error in (4.8), (4.9) or the code. Two points to correct in the text:

**F1 — where the plane constant is ill-conditioned.** The map $C\mapsto\hat a$ has $\partial\hat a/\partial C\sim C^{-1/2}$ at the empty end and $\sim(1-C)^{-1/2}$ at the full end (S6), but the *bisection* loses accuracy only at the **full** end: it evaluates $\mathcal A$ with an absolute rounding error $\varepsilon$, which is amplified by $\partial\hat a/\partial C$. Measured (256-bit truth, 134 normals, error in units of $\lvert\hat n\rvert_1$):

| $C$ | $10^{-14}$ | $10^{-8}$ | $10^{-4}$ | 0.5 | 0.99 | $1-10^{-4}$ | $1-10^{-6}$ | $1-10^{-8}$ | $1-10^{-10}$ | $1-10^{-14}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| bisection | $1.0\mathrm e{-16}$ | $1.1\mathrm e{-16}$ | $1.4\mathrm e{-16}$ | $8.3\mathrm e{-17}$ | $3.5\mathrm e{-16}$ | $2.1\mathrm e{-15}$ | $2.0\mathrm e{-14}$ | $2.0\mathrm e{-13}$ | $2.0\mathrm e{-12}$ | $2.0\mathrm e{-10}$ |
| closed form (Float64) | $9.5\mathrm e{-17}$ | $8.2\mathrm e{-17}$ | $8.8\mathrm e{-17}$ | $1.1\mathrm e{-16}$ | $1.7\mathrm e{-16}$ | $1.1\mathrm e{-16}$ | $1.1\mathrm e{-16}$ | $1.5\mathrm e{-16}$ | $1.0\mathrm e{-16}$ | $9.7\mathrm e{-17}$ |

i.e. bisection error $\approx2\times10^{-17}(1-C)^{-1/2}\lvert\hat n\rvert_1$ near full cells (ratio to $\varepsilon\,\mathrm dz/\mathrm dC$ never above 0.8), at the empty end it is accurate to $10^{-16}$. The closed form is accurate to $\le1.7\times10^{-16}\lvert\hat n\rvert_1$ everywhere. Consequence: none in practice — the fraction floor snaps $1-C\le1.5\times10^{-8}$ to a pure cell, which bounds the bisection error by $\approx2\times10^{-13}\lvert\hat n\rvert_1$; but the stated reason for restricting the comparison to $[10^{-2},1-10^{-2}]$ is only half the story.

**F2 — "the reflected branch avoids cancellation near full cells"** is true relative to the textbook inclusion–exclusion form $[z^2-(z-w_1)_+^2-(z-w_2)_+^2+(z-w_1-w_2)_+^2]/(2w_1w_2)$, which loses $\varepsilon\,w_>/w_<$ (measured $4\times10^{-9}$ for $w_</w_>=10^{-8}$, $7\times10^{-11}$ for $10^{-6}$, $3\times10^{-13}$ for $10^{-4}$); both it and the explicit third regime $1-(w_1+w_2-z)^2/(2w_1w_2)$ that the reflection reproduces equal the 256-bit reference rounded to Float64 in all 100 tested configurations (same arithmetic). So the reflection is a way to write the third regime with the first formula, not an additional accuracy measure.

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| (4.8) closed form | **sharp** (an identity: equals the exact area) | symbolic, 256-bit, 40 000 random draws | `step01`, `code_01` | — |
| $w_<\le\varepsilon w_>$ fall-back | **sharp to $\varepsilon/2$** | neglected term $w_</(2w_>)\le\varepsilon/2$ | `code_01` | — |
| bisection root accuracy | **not sharp near full cells** | error $\approx2\times10^{-17}(1-C)^{-1/2}\lvert\hat n\rvert_1$; $2\times10^{-10}$ at $1-C=10^{-14}$ | `code_02` | uncut cells: use the closed-form inverse (O(1) operations, $\le1.7\times10^{-16}$ everywhere); or bisect on the deficit $1-\mathcal A$ computed in the reflected branch for $C>\frac12$ (not tested) |
| 54 bisection steps | **sufficient** (not more than needed) | bracket $2^{-54}\lvert\hat n\rvert_1\approx5.5\times10^{-17}\lvert\hat n\rvert_1$, error at the empty end $1.0\mathrm e{-16}$ | `code_02` | — |
| $\partial\hat a/\partial C\sim C^{-1/2}$ | **sharp** (corner regime, $C<w_</(2w_>)$) | exact derivative $\sqrt{w_<w_>/(2C)}$; measured amplification matches (ratio $\le0.8$) | `step01`, `code_02` | in the linear regime ($C\in[w_</2w_>,1-w_</2w_>]$) and for grid-aligned normals the map is Lipschitz with constant $w_>$ |
| cut-cell bisection (clipped polygon) | **sharp** | residual $4.4\times10^{-16}$ | `code_03` | — |
| uncut rectangles with any $\hat r^\pm$ | needed | used by (4.10) and (7.5); tested for rectangles far from the origin (rounding of $z$: error $\le8\varepsilon(1+\max(\lvert a\rvert,\lvert\text{shift}\rvert)/w)$) | `code_01` | — |

## Open points

- The closed-form inverse could replace the 54-step bisection in uncut cells (a speed-up, not a correctness issue); cut cells have no closed form.
- For the `literature-scout`: Scardovelli & Zaleski (2000) give the 2-D inverse in the normalised form used here ($m_1,m_2$ with $m_1+m_2=1$); the 3-D case is not needed.

## Proposed replacement text

`04_interface.tex:155-157`: replace "and the reflected branch avoids cancellation near full cells" by

```latex
and the third regime is evaluated by the first through the reflection
$\zdist\mapsto\pwidth{1}+\pwidth{2}-\zdist$ (the inclusion--exclusion form of the
distribution function loses $\machineeps\,\pwidth{>}/\pwidth{<}$)
```

`04_interface.tex:181-185`: replace "(near pure cells the plane constant is ill-conditioned, $\partial\lconst{c}/\partial\condfrac{\liq}{c}\sim(\condfrac{\liq}{c})^{-1/2}$)" by

```latex
(near pure cells the plane constant is ill-conditioned,
$\partial\lconst{c}/\partial\condfrac{\liq}{c}\sim(\condfrac{\liq}{c})^{-1/2}$ and
$(1-\condfrac{\liq}{c})^{-1/2}$; the bisection evaluates $\planearea$ with an
absolute error $\machineeps$, so its root is accurate to
$\machineeps\,(1-\condfrac{\liq}{c})^{-1/2}\abs{\lnormal{c}}_1$ near full cells, at most
$2\times10^{-13}\abs{\lnormal{c}}_1$ for the fractions that survive the floor of
\cref{eq:interface:floor}, while the closed form stays within a few
$10^{-16}\abs{\lnormal{c}}_1$ everywhere)
```

## History

- 2026-11-08: first audit (symbolic regime-by-regime proof of (4.8) and of the inverse; package vs exact polygon clipping on 40 000 random draws, inverse over 17 fractions from $10^{-14}$ to $1-10^{-14}$ against 256-bit truth, cut-cell intercepts; reflected-branch comparison). Outcome: ✅ verified; F1, F2 wording.
