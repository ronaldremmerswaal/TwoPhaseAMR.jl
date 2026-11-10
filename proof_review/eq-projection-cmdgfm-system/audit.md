# Audit: eq:projection:cmdgfm-system (group) — CMDGFM projection: staggered normal, transverse/inference faces, β, rows, interpolant, jump rows, unresolved rules, diagonal dominance, system

- Source: `docs/techref/chapters/10_projection.tex:316-556` (staggered-normal 330, transverse 351, beta 361, cmdgfm-row 379, interpolant 389, cmdgfm-jump-row 401, unresolved 426, dominance 443, cmdgfm-system 461; deviations 501, 512), commit `b1e7b5b`; thesis `~/Documents/phd_project/thesis/pt2_poisson.tex` (§7.3, eqs. (7.11)-(7.19))
- Code audited (normative): `src/two_phase/projection/cmdgfm_projection.jl` (`_cmdgfm_face_normal`, `_cmdgfm_jump_stencils`, `_cmdgfm_jump_blocks`, `_assemble_cmdgfm_cached!`, `cmdgfm_jump_interpolant`, `_cmdgfm_normal_velocity_defect`), `cmdgfm_cache.jl` (`_cmdgfm_pressure_jump`, `_cmdgfm_append_gradient_row!`, `_cmdgfm_cell_flux!`, `_cmdgfm_gauge_partition!`)
- Date: 2026-11-09
- Overall status: ✅ verified: an independent transcription of rules 1-5, β and the rows (patches from face coordinates, own face data) reproduces the code's stencil sets exactly on 4 chapter fixtures and 8 stress fixtures (thin slab, wavy interface, interface on a grid line, a second drop), the assembled system to $1.1\times10^{-16}$ and the solution to $2\times10^{-11}$; the dominance claims and β maximality are proved/verified; **thesis weight β does violate weak dominance for $s\ge0.27$ (new, quantifies Deviation 10.1)**; three precision remarks (F1-F3)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-projection-cmdgfm-system/scripts/run_all.jl`

## Statement and assumptions

Cell normals $\eta_c=\mathrm{diag}(h)^{-1}\ell_c/|\cdot|$ of the PLIC lines (zero if $\le\sqrt\epsilon$), weights $\mu_c=C_c(1-C_c)$, staggered normal $\tilde\eta_f=\mathbf N_f/|\mathbf N_f|$, $\mathbf N_f=\mu_{c^-}\eta_{c^-}+\mu_{c^+}\eta_{c^+}$ (or $(C_{c^-}-C_{c^+})\mathbf n_f$ if the weights vanish/cancel), $R_f=\tilde\eta_{f,t}/\tilde\eta_{f,k}$; transverse faces (axis $t$, vertex patches of the end points of $f$, mixed by fraction and open), weights $w_{f,h}=\mu_h/\sum\mu_{h'}$; inference faces $i^{\rm g},i^{\rm l}$ (extremes of $m^{\rm l}/m$ among the axis-$k$ faces of the patches, dominated by their phase), $s_f=m^{\rm l}_{i^{\rm g}}/m_{i^{\rm g}}+m^{\rm g}_{i^{\rm l}}/m_{i^{\rm l}}$, $\beta_f=1$ ($|R_f|\le1$), $(1-s_f)/(|R_f|-s_f)$ ($|R_f|>1$); row $\mathcal N_f=[u]_f+\beta_fR_f\sum w[u]_h+(\beta_f-1)(u^{\rm g}_{i^{\rm g}}-u^{\rm l}_{i^{\rm l}})$; $\mathcal N_f=(\beta_f/\tilde\eta_{f,k})\tilde\eta_f\!\cdot\!\mathfrak J^\beta$ with the thesis' interpolant (7.16); linear row (cmdgfm-jump-row); five unresolved rules in the printed order; $\mathsf B_\xi$ with unit diagonal, $\sum_{h\ne f}|\mathsf B_{\xi,fh}|\le\beta_f|R_f|+(1-\beta_f)s_f\le1$; two-block system (cmdgfm-system).

**Assumptions made explicit.** Rules 4 and 5 apply to non-aligned faces only (the code returns the aligned stencil before them; see F1); walled box; the PLIC data (`normal_x`, `intercept`) and conditional fractions are those verified in the interface chapter audits; the *normal convention* (the PLIC line has the liquid on $\{\xi\!\cdot\!x\le a\}$, so $\xi$ and $\eta_c$ point into the gas, as thesis §2) was checked cell by cell on the uncut cells.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| S1 normal convention: liquid on the side opposite to $\eta_c$ | 323-330 | ✅ verified | clipped area of the reference cell vs $C$ | `code_01_stencils.jl` | 0 mismatches on all uncut mixed cells (24, 39, 18, 28 cells) |
| S2 classification by rules 1-5, resolved set, transverse set and weights, $R_f$, inference faces, $\beta_f$ | 330-377, 411-436 | ✅ verified | independent transcription; patches by endpoint coordinates | `code_01_stencils.jl` | classes: uniform (47 resolved, 24 rule 2, 1 rule 5), composite (70, 43, 6), uniform+solid (35, 14, 1 rule 4, 1 rule 5), composite+solid (51, 27, 1 rule 4, 4 rule 5); sets exactly equal; $\max|\Delta\beta|=0$, $|\Delta R|=0$, $|\Delta w|\le3.9\times10^{-16}$. Stress fixtures (uniform and composite): drop $r=1.3$ (39+20+1, 55+30+4), thin slab (64 aligned resolved, 30/46 by rule 3), wavy interface ($61$ and $67\,\%$ unresolved: rule 2 and rule 5), interface on a grid line (all 16/24 faces aligned and resolved) — all equal |
| S3 dominated inference faces $\Rightarrow0\le s\le1<|R|$, $0\le\beta<1$; clamp and error branch never act | 361-377 | ✅ proved + verified | symbolic ($1-\beta=(|R|-1)/(|R|-s)$), $10^4$ random $(s,|R|)$, and on all resolved rows of the fixtures | `sym_01_beta_dominance.jl`, `code_01_stencils.jl` | |
| S4 dominance: row sum $=\beta|R|+(1-\beta)s\le1$, equality iff $|R|\ge1$; diagonal $=1$; margins of the assembled block | 438-457 | ✅ verified | symbolic + the assembled `jump_block` vs the formula; left inequality equal when all neighbours carry unknowns | `sym_01_beta_dominance.jl`, `code_01_stencils.jl` | tight rows ($|R|\ge1$): 16, 12, 23, 16, of which 8, 6, 8, 7 have margin $\le64\epsilon$ in the assembled block (the chapter's table: 8, 6, 8, 7) — the others miss an unknown neighbour; minimum margin $0$ / $-2.2\times10^{-16}$ (chapter: $-1.1\times10^{-16}$); strict margins on the thin slab (1.0) and the wavy interface ($1.2\times10^{-3}$) |
| S5 β is the largest $\beta\in[0,1]$ with $\beta|R|+(1-\beta)s\le1$ | 449-457 | ✅ proved | monotone in $\beta$ iff $|R|>s$ | `sym_01_beta_dominance.jl` | |
| S6 "a third of the mixed faces are unresolved, almost all by rule 2" | 432-436 | ⚠️ precision (F2) | counts | `code_01_stencils.jl` | $34.7,\,31.4,\,41.2,\,38.6\,\%$ unresolved; rule 2 gives $96,\,88,\,88,\,84\,\%$ of them |
| C1 gauge partition (anchors = smallest index per component; retained = rest) | 281-288 | ✅ verified | BFS | `code_02_system.jl` | equal on 4 fixtures (1 component each); 2 components in `eq-projection-two-velocity` |
| C2 dense $\begin{bsmallmatrix}\mathsf S&\mathsf J_r^\top\operatorname{diag}(a^\Xi)\\\mathsf B_p&\mathsf B_\xi\end{bsmallmatrix}$ and rhs $[-\tfrac{|c^{\rm o}|\mathfrak D^{\rm mix}}{\delta t};\tfrac{\mathcal N(v^*)}{\delta t}]$ = assembled | 458-485 | ✅ verified | independent assembly incl. $\mathsf B_p=(\beta-1)(\mathsf R^{\rm tv}_{i^{\rm g}}-\mathsf R^{\rm tv}_{i^{\rm l}})\mathsf J_r$ | `code_02_system.jl` | $\le1.1\times10^{-16}$ (matrix), $\le2.1\times10^{-16}$ (rhs); sizes $302,709,256,588$ |
| C3 (cmdgfm-row)/(interpolant): $\mathcal N_f=(\beta_f/\tilde\eta_{f,k})\tilde\eta_f\!\cdot\!\mathfrak J^\beta$ with `cmdgfm_jump_interpolant` | 378-410 | ✅ verified | 47 + 70 + 35 + 51 resolved non-aligned faces | `code_02_system.jl` | $\le10^{-12}$ |
| C4 jump row from the update (cmdgfm-jump-row), $\mathcal N_f(v^{n+1})=0$, $\mathfrak D^{\rm mix}(v^{n+1})=0$ on all open cells (one component, zero boundary flux), mixture update cell by cell, phase identities, momentum identity | 401-410, 475-485 | ✅ verified | independent $\mathcal N_f$, flux, $\mathsf S p+\mathsf J^\top a^\Xi\xi$ on the converged solution ($\varepsilon=10^{-10}$) | `code_02_system.jl` | $|\mathcal N_f|\le2.5\times10^{-14}$ (code's identical), $\delta t|\mathfrak D^{\rm mix}|\le9.5\times10^{-11}$, update $\le4.8\times10^{-15}$, mixture update $\le1.1\times10^{-14}$, identities $\le2\times10^{-13}$ |
| C5 the dense solve of the independent system = the code's solution | 458-485 | ✅ verified | LU | `code_02_system.jl` | $\le7\times10^{-11}$ (cond $\le7.4\times10^{4}$) |
| B1 thesis weight: $\sup_x\beta_{\rm th}|R|=0.8728$ ("$0.873$") | 501-510 | ✅ verified | numerical sup of $(3x-2x^2)\sqrt{1-x^2}$; the thesis prints $0.87278\ldots$ | `sym_01_beta_dominance.jl` | |
| B2 "does not control the inference coefficients" | 501-510 | ✅ verified, quantified | with $s=1$: $\sup(\beta_{\rm th}|R|+(1-\beta_{\rm th}))=1.474$ (at $x=0.329$); weak dominance fails for $s\ge0.27$ | `sym_01_beta_dominance.jl` | |
| B3 inference faces of the thesis: one of them is $f$ itself (thesis p. 215) | 512-520 | ✅ verified against the thesis | `pt2_poisson.tex`/thesis §7.3.5: "We let either $h^{\rm g}=f$ or $h^{\rm l}=f$ depending on which of the corresponding masses is largest inside $\omega_f$" | — | |
| E1 the CMDGFM conserves momentum but not kinetic energy | 484-485 | ✅ verified | momentum identity $\le2\times10^{-13}$; Pythagoras defect $8.9\times10^{-3}$ of $E^*$ | `code_02_system.jl`, `eq-projection-kkt/code_02` | |
| T1 equations of the thesis cited: (7.11) weights $\hat C(1-\hat C)/\sum$, (7.13), (7.15) $R_{f,h}=\tilde\eta\!\cdot\!n_h/\tilde\eta\!\cdot\!n_f$, (7.16), (7.17), (7.18)/(7.19) $\beta(x)=3x^2-2x^3$, (4.50)/(4.51) | throughout | ✅ checked against `thesis.pdf` text | — | the chapter's rows (cmdgfm-jump-row) are the thesis (7.17) with $g^\pi=(\mathsf Sp+\Xi^\pi\xi)/m$ (4.50) |

## Errors and consequences

No error in a formula or in the code relative to the text. Precision remarks:

**F1 (rules 4 and 5 and aligned faces).** In the printed list, rules 4-5 read as applying to every mixed face reaching them, but a face with an aligned normal ($|\tilde\eta_{f,k}|\ge1-64\epsilon$) is turned into the "no transverse, no inference" row (face-normal jump removed) *before* rules 4-5, so it is never unresolved by them. This is not a corner case: **an interface on a grid line** (pure cells on both sides; liquid fraction difference supplies the normal) consists exactly of aligned faces with *no* transverse faces (their transverse neighbours lie between pure cells), so rule 4 read literally would make all of its faces (16 on the uniform, 24 on the composite fixture) unresolved, whereas the code resolves them with the row $\mathcal N_f=[u]_f$ (independent transcription: 16/16 and 24/24 aligned-resolved, 0 unresolved). The chapter's own fixtures contain no aligned face, so the table is unaffected. *Proposed replacement of items 4 and 5 of the list (macros existing):*

```latex
\item its normal is not aligned with the face (an aligned normal,
  $\abs{\normalstag_{f,k}}\ge1-64\machineeps$, keeps the row $\normaljump{f}=\jump{\velo}_f$ without
  transverse or inference faces) and it has fewer than two transverse faces, or fewer than one if
  an end point lies on a physical boundary or a transverse face has another length than $f$
  (a level transition), or their mixing weights sum to zero;
\item its normal is not aligned, $\abs{\normratio{f}}>1$, and $\parfaces{f}$ has fewer than two
  faces, or an inference face is not dominated by its phase;
```

**F2 (counts).** "a third of the mixed faces are unresolved, almost all by rule 2" — measured $31$-$41\,\%$ (34.7, 31.4, 41.2, 38.6) and $84$-$96\,\%$ of them by rule 2 (the remaining 1, 2, 6, 5 faces by rules 4, 5; equal to the table). *Proposed replacement:* "On the four drop fixtures of \cref{tab:projection:counts} between $31$ and $41\,\%$ of the mixed faces are unresolved, between $84$ and $96\,\%$ of them by rule~2 (…)".

**F3 (dominance sentence).** "the right one is an equality exactly when $|\normratio{f}|\ge1$" holds for non-aligned rows; an aligned row has no off-diagonal entries (left side 0). Add "for a row with a transverse term". Also state, as verified, that for $|\normratio{f}|<1$ ($\beta_f=1$) the margin is $1-|\normratio{f}|$ when all transverse faces carry unknowns.

**Deviation 10.1 (CMDGFM jump weight), optional quantification:** after "does not control the inference coefficients" add "(with $s_f=1$ the row sum reaches $1.47$; weak dominance fails for $s_f\ge0.27$)".

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| β is the largest weight with weak dominance | **sharp** (maximal) | proved; larger β violates the bound for $|R|>s$ | `sym_01` | — |
| $\beta|R|+(1-\beta)s\le1$ attained exactly when $|R|\ge1$ | **sharp** | tight rows: 16, 12, 23, 16; margin $\ge-1.1\times10^{-16}$ | `code_01` | — |
| dominated inference faces needed (rule 5) | **sharp** (needed for $s\le1$, hence $\beta\ge0$) | without it $s>1$ gives $\beta<0$ / negative denominators; the code classifies the row as unresolved | `sym_01` | — |
| thesis β | **not dominant** for $s\ge0.27$ (counterexample $s=1$: row sum $1.47$) | `sym_01` | — | the code's β |
| solution agreement with the dense solve | **likely sharp** | $2\times10^{-11}$ at tolerance $10^{-10}$ | `code_02` | — |

## Open points

- The unresolved-rule order of the *thesis* (none) cannot be compared: Deviation 10.2 states the thesis has no classification.
- BiCGSTAB/preconditioner statements (block lower-triangular, ILU drop tolerance $10^{-3}$, `jump_drop_tolerance=1e-3`) were read from the code (`cmdgfm_projection.jl:222-250, 769-771`), not tested; iteration counts of the table were not recomputed.

## History

- 2026-11-09: first audit (scripts `sym_01_beta_dominance`, `code_01_stencils` (+ lib; 4 chapter + 8 stress fixtures), `code_02_system`). `run_all.jl`: PASS (1 symbolic + 2 code scripts; 12 fixtures).
