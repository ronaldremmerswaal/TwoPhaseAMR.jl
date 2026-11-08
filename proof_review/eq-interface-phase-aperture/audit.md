# Audit: eq:interface:phase-aperture — phase-aperture partition (4.21)–(4.22) versus the thesis' sum of phase apertures

- Source: `docs/techref/chapters/04_interface.tex:563-598` (eqs. 4.21 at 567-569, 4.22 at 572-583) and the deviation "Open apertures" `07_cutcells.tex:117-125`, commit `1e1886c` on branch `docs/techref-geometry` (chapters unchanged in the working tree; only `proof_review/` is untracked). Thesis: `basic_interface.tex` (printed eq. (2.41)), `basic_geometry.tex` (§2.4, printed $\alpha^{\rm o}:=\{\!\{\alpha\}\!\}$, $a^{\rm o}:=\{\!\{a\}\!\}$ with $\{\!\{\varphi\}\!\}:=\varphi^{\rm g}+\varphi^{\rm l}$).
- Date: 2026-11-08
- Overall status: ✅ **verified.** On every face without a covered cell the code's liquid aperture, gas aperture and open aperture equal the thesis' definition ((2.41) and the sum over phases) to $\le6.7\times10^{-15}$ (4280 interior faces, 384 of them hanging, 524 touching a cut cell; 384 boundary faces), the clamp of (4.22) is never active there, and the only place where the two definitions differ — a covered neighbour — behaves exactly as the deviation box says (the thesis' sum is half of the open trace of the other cell, the code's is zero).
- Rerun: `julia --project=@claude-symbolics proof_review/eq-interface-phase-aperture/scripts/run_all.jl` (two sequential child processes: Symbolics group under `@claude-symbolics`, package group under the repository environment; flags `--heap-size-hint=3G -t 1`). Last run: all passed.
- Label: `eq:interface:phase-aperture`; the trace formula (4.21) is `eq:interface:liquid-trace`, the open aperture (3.6) is `eq:setting:aperture`.

## Statement and assumptions

With $\hat f_c$ the local image of the face segment, the one-sided liquid trace is $a^{\rm liq}_{c,f}=\lvert\hat f_c\cap\hat Q^{\rm liq}_c\rvert/\lvert\hat f_c\rvert$ (solid half-plane applied first; pure cells give 0 or the open trace $a^{\rm o}_{c,f}$). On an interior face (4.22)

$$a^{\rm liq}_f=\begin{cases}0&\text{no open cell of }\mathcal C(f)\text{ has }C^{\rm liq}_c>0,\\ a^{\rm o}_f&\text{no open cell has }C^{\rm liq}_c<1,\\ \operatorname{clamp}\big(\tfrac12\sum_{c\in\mathcal C(f)}a^{\rm liq}_{c,f},\,0,\,a^{\rm o}_f\big)&\text{otherwise,}\end{cases}\qquad a^{\rm gas}_f=a^{\rm o}_f-a^{\rm liq}_f,$$

and on a boundary face $a^{\rm liq}_f=\operatorname{clamp}(a^{\rm liq}_{c_{\rm b}(f)},0,a^{\rm o}_f)$. Here $a^{\rm o}_f$ is the shared open aperture (3.6): $0$ if a cell of $\mathcal C(f)$ is covered, else $\min\{1,\frac12(a^{\rm o}_{c^-,f}+a^{\rm o}_{c^+,f})\}$ from the solid half-planes. Thesis: $a^\pi_f=\frac12\sum_{c\in\mathcal C(f)}\lvert f\cap c^\pi\rvert/\lvert f\rvert$, $a^{\rm o}=a^{\rm liq}+a^{\rm gas}$ (no covered rule, no clamp). Deviation box: "Without clamping and covered neighbours both give the same sum."

Assumptions used: each cell's open part is split by one PLIC line (liquid = open polygon $\cap$ half-plane, gas = open polygon minus liquid); conditional fractions after the floor of (4.3); covered cell = open fraction 0.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| S1 segment clipping primitive: the crossing of the code's `_clip_line_interval` is the root $-k/m$ of $m s+k$; lengths of the kept intervals ($-v_{lo}/m$, $v_{hi}/m$) and of the complement add up to $h-l$ | 567-570 | ✅ | symbolic | `step01` | |
| S2 partition lemma: with liquid $\subset$ open on every face segment, open $=$ liquid $+$ gas and the half-sums over the two cells keep the identity, so $a^{\rm liq}+a^{\rm gas}=\frac12\sum_c a^{\rm o}_{c,f}=a^{\rm o}_f$ when no cell is covered | 572-583; 07:117-125 | ✅ | symbolic (additivity of lengths) + numeric (S3) | `step01`, `code_01` | $a^{\rm liq}_{c,f}\le a^{\rm o}_{c,f}\le1$, so the clamp and the $\min\{1,\cdot\}$ in (3.6) are no-ops up to rounding |
| S3 code $(a^{\rm liq},a^{\rm gas},a^{\rm o})$ = thesis' definition on faces without a covered cell | 563-585 | ✅ | package vs independent evaluation (Sutherland–Hodgman polygons from the stored solid and PLIC planes, Cyrus–Beck segment clipping; nothing of `_face_aperture`/`_phase_face_apertures` reused) on 6 fixtures: a sloped wall under a wavy liquid film (contact lines, triple cells, covered cells) on uniform meshes and on 2:1-graded AMR meshes (hanging faces: half-side traces), isotropic and anisotropic cells ($1.3\times0.7$, $0.8\times1.4$), four wall/film configurations | `code_01` | max deviations $\le6.7\times10^{-15}$ (liquid), $6.2\times10^{-15}$ (gas, $a^{\rm o}$); isotropic fixtures $\le2.2\times10^{-16}$; 4280 interior faces, 384 hanging, 524 touching a cut cell, 363 with both phases |
| S4 $a^{\rm liq}+a^{\rm gas}=a^{\rm o}$ | 572-583 | ✅ | package | `code_01` | holds exactly (0.0) by construction ($a^{\rm gas}:=a^{\rm o}-a^{\rm liq}$) and, independently, $a^{\rm o}$ equals the sum of the thesis' phase apertures |
| S5 the clamp in the third case is inactive away from covered cells | 572-583; 07:123-124 | ✅ | package | `code_01` | 0 faces with $\frac12\sum a^{\rm liq}_{c,f}\notin[0,a^{\rm o}_f]$ beyond $10^{-14}$; max excess $1.1\times10^{-16}$ |
| S6 covered neighbour: code $(a^{\rm o},a^{\rm liq},a^{\rm gas})=(0,0,0)$; thesis' sum $=\frac12$ (open trace of the other cell) | 07:96-100, 117-125 | ✅ | package, 1206 such faces; symbolic S4; grid-aligned boundary example ($y\le3$ solid, liquid above): 8 faces, thesis $\frac12$, code 0 | `code_01`, `code_02` | the clamp is active exactly here ($a^{\rm o}_f=0$); at a grid-aligned wall both cells are pure and the average would be $\frac12$ |
| S7 pure-phase cases give exact zeros | 572-583 | ✅ | package | `code_01` | faces without liquid: $a^{\rm liq}$ is `0.0` (bitwise); without gas: $a^{\rm gas}$ is `0.0` |
| S8 boundary faces: $\operatorname{clamp}(a^{\rm liq}_{c_{\rm b}},0,a^{\rm o}_f)$ and $a^{\rm o}_f=$ trace of the cell | 584 | ✅ | package, 384 boundary faces (7 with both phases) | `code_01` | $\le8.3\times10^{-17}$; the thesis' (2.41) applied literally to a boundary face (one adjacent cell) would halve the trace: the techref's boundary rule is the natural extension |
| S9 hanging faces: traces over a half side of the coarse cell | 07:90-93 | ✅ | package | `code_01` | 384 hanging faces included in S3 |

## Errors and consequences

No error. The deviation box in `07_cutcells.tex` is correct; it can be made exact (replacement text below): the two definitions coincide on every face without a covered cell (clamp and $\min\{1,\cdot\}$ inactive) and differ on a face with a covered cell by exactly $\frac12$ of the open trace of the other cell ($=\frac12$ at a grid-aligned solid boundary, where the other cell is pure).

Remarks (information, no correction needed):
- Rounding-level openings: on 2 of the 4280 faces a phase that is geometrically absent on the face keeps an aperture below $10^{-12}$ (a mixed cell whose PLIC line touches the face segment at an end point; the pure-phase guards of (4.22) only cover pure cells, as written). The text claims exactness only for the pure cases.
- The three-case form of (4.22) is equivalent to the single formula $\operatorname{clamp}(\frac12\sum_ca^{\rm liq}_{c,f},0,a^{\rm o}_f)$ up to the exact zeros of the first two cases (S7); they matter for the guard against a $10^{-17}$ opening of an absent phase.

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| (4.22) equals the thesis' sum on faces without a covered cell | **sharp** (an equality) | $\le6.7\times10^{-15}$ on 4280 faces | `code_01` | — |
| the clamp is needed only at covered neighbours | **sharp** | 0 active elsewhere; always active at the 1206 covered-neighbour faces | `code_01` | — |
| $\min\{1,\cdot\}$ in (3.6) | **redundant** (no-op: traces $\le1$) | $a^{\rm o}\le1$ on all faces | `code_01` | drop it from (3.6) or keep as a guard |
| covered rule | **needed** | thesis' sum leaves $a^{\rm o}=\frac12$ (flux leaks) at grid-aligned solid boundaries | `code_02` | — |
| (4.22), three cases | **equivalent** to one formula + exact zeros | S7 | `code_01` | — |
| boundary rule (no factor $\frac12$) | **needed** | one adjacent cell: the average of the single trace is the trace | `code_01` | — |

## Open points

- None blocking. If desired, add to the implementation note that $a^{\rm gas}_f$ is computed as $a^{\rm o}_f-a^{\rm liq}_f$ (so it can carry a rounding-level residual where the absent phase is not caught by the pure-cell guards).

## Proposed replacement text

`07_cutcells.tex:122-124`, replace the last sentence of the deviation box "Without clamping and covered neighbours both give the same sum." by:

```latex
On a face without a covered cell, $\aperture{f}$ equals the sum of the thesis' phase
apertures, $\half\sum_{c\in\cellset{f}}(\phasetrace{\liq}{c}{f}+\phasetrace{\gas}{c}{f})
=\half\sum_{c\in\cellset{f}}\traceap{c}{f}$ (the clamp
in~\cref{eq:interface:phase-aperture} is then inactive up to round-off). If a cell of
$\cellset{f}$ is covered, the thesis' sum is half of the open trace of the other cell
(one half at a grid-aligned solid boundary), the code's is zero.
```

## History

- 2026-11-08: first audit (symbolic primitive and partition lemma; package vs an independent polygon evaluation of (2.41) on six film/wall fixtures with hanging faces, anisotropic cells, covered neighbours and boundary faces; grid-aligned covered-neighbour example). Outcome: ✅ verified; replacement text for the deviation box.
