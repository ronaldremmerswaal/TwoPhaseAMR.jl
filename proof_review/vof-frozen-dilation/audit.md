# Audit: vof-frozen-dilation — the dilation of the frozen carrier and the discrete mixture constraint

- Source: `docs/techref/chapters/05_vof_transport.tex:625-631` ("The dilation of a step, $\chi_c\delta t\sum_f o_{c,f}|f|(a^{\rm l}_fu^{\rm l}_f+a^{\rm g}_fu^{\rm g}_f)$, vanishes where the phase velocities satisfy the discrete mixture constraint [thesis (6.19)] on the frozen apertures, which both two-velocity projections impose up to their tolerance"), together with the carrier (5.23) `eq:vof:frozen-flux` (618-624) and the first bullet of Section 5.9 (806-808); commit `ec20986` (branch `docs/techref-vof`, tree clean). Code: `src/two_phase/two_velocity/interface_transport.jl` (`_compute_directional_wy_fluxes!`, `_freeze_phase_apertures!`), `src/vof/fluxes.jl` (dilation in `_advect_wy!`).
- Date: 2026-11-08
- Overall status: ⚠️ **verified with one precision**: the dilation of cell $c$ is $\chi_c\delta t$ times the mixture divergence of $c$ on the frozen apertures, so it vanishes in a cell with $\chi_c=1$ **iff** the constraint holds there; it is absent in cells with $\chi_c=0$ whatever the divergence. "Vanishes where ... satisfy the constraint" is correct; "exactly when the constraint holds" (the summary of the claim) is correct only cell-wise on $\{\chi_c=1\}$. The sentence "which both two-velocity projections impose up to their tolerance" concerns Chapter 10 and was not audited (open point).
- Rerun: `julia --project=@claude-symbolics proof_review/vof-frozen-dilation/scripts/run_all.jl`. Last run: all checks passed (symbolic group and package group).
- Label: none (the sentence sits between `eq:vof:frozen-flux` and `eq:vof:frozen-cfl`); suggested `prop:vof:frozen-dilation`.

## Statement and assumptions

With frozen phase apertures $a^{\rm l}_f,a^{\rm g}_f$ (the phase apertures of the reconstruction at the start of the step, $a^{\rm l}+a^{\rm g}=a^{\rm o}$) and phase velocities $u^\pi_f$, the carrier is $\upsilon^{\rm o}_f=\sum_\pi a^\pi_fu^\pi_f$ (5.23) in both sweeps, so the dilation of a step in $c$ is $\chi_c\delta t\,|c|(\mathfrak D\upsilon^{\rm o})_c=\chi_c\delta t\,m_c$, $m_c:=\sum_{f\in\mathcal F(c)}o_{c,f}|f|(a^{\rm l}_fu^{\rm l}_f+a^{\rm g}_fu^{\rm g}_f)$, the discrete mixture divergence of $c$ [thesis (6.19): $|c|\mathfrak D\langle au\rangle_c=0$]. Conclusion in the source: the dilation "vanishes where" $m_c=0$. Assumptions that should be explicit: (a) $\chi_c$ is the frozen indicator of the **liquid** ($\chi^{\rm l}_c=1$ iff $C^n_c>\tfrac12$); the dilation acts only in these cells; (b) the apertures in $m_c$ are those frozen at the start of the step, not the apertures of the current reconstruction (which change between sweeps); (c) "per step" means the sum of the two sweeps: the dilation of a single sweep does not vanish for a mixture-solenoidal carrier.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| S1 streamfunction face fluxes are exactly solenoidal on composite cells (hanging vertex cancels); used to build mixture-solenoidal carriers | — (test construction) | ✅ | symbolic (coarse cell with two hanging segments and its two fine neighbours) | `step01` S1 | |
| S2 splitting $a^{\rm o}w=a^{\rm l}u^{\rm l}+a^{\rm g}u^{\rm g}$ with arbitrary jump keeps the mixture flux | — | ✅ | symbolic: $u^{\rm l}=w(1+d_{\rm l})$, $u^{\rm g}=w(1+d_{\rm g})$, $a^{\rm l}d_{\rm l}+a^{\rm g}d_{\rm g}=0$ | `step01` S2 | also for $u^{\rm l}u^{\rm g}<0$ |
| S3 dilation $=\chi_c\delta t\,m_c$; zero iff $m_c=0$ when $\chi_c=1$, identically zero when $\chi_c=0$ | 625-631 | ✅ | symbolic S3; code: per-cell dilation of the independent frozen replica vs $\chi_c\delta t\,m_c$ computed from apertures and velocities, 16 steps: $\le2.9\times10^{-17}$ | `step01`, `code_01` (i) | |
| S4 $\upsilon^{\rm o}$ is the carrier divergence and reduces to $\chi\delta t|c|(\mathfrak Du)_c$ for equal velocities with $a^{\rm l}+a^{\rm g}=1$ | 618-624 | ✅ | symbolic S4 | `step01` | links to `eq-vof-frozen-flux` |
| S5 mixture-solenoidal phase velocities with jumps conserve the liquid volume | 625-631, 806-808 | ✅ | package `advect!` (frozen), 16 composite periodic steps (levels 1, 2, anisotropic) with 683 split faces, 183 of them with $u^{\rm l}u^{\rm g}<0$; $\max\lvert m_c\rvert=2.0\times10^{-14}$, $\lvert\sum$dilation$\rvert\le2.5\times10^{-16}$, relative volume change $\le7.2\times10^{-16}$; plus 4 cut steps (open streamfunction, 70 split faces): $\lvert m_c\rvert\le8.9\times10^{-16}$, volume change $\le3.7\times10^{-16}$ | `code_01` (i)(ii) | the step is halved if the frozen subface strip is too small (`vof-boundedness`) |
| S6 the constraint is needed only in $\chi=1$ cells | 806-808 | ✅ | perturbation of both phase velocities on 287 faces between $\chi=0$ cells: constraint violated in 151 $\chi=0$ cells, in 0 $\chi=1$ cells; volume change $3.1\times10^{-16}$ | `code_01` (iii) | |
| S7 a violation in a $\chi=1$ cell changes the volume by exactly its dilation | 625-631 | ✅ | one face next to a $\chi=1$ cell (and a $\chi=0$ cell), liquid velocity $+0.05$: $\Delta V=4.78296008\times10^{-4}$, formula $\delta t\sum_c\chi_co_c|f|a^{\rm l}\cdot0.05=4.78296008\times10^{-4}$, replica $4.78296008\times10^{-4}$ (agreement $3\times10^{-14}$ absolute) | `code_01` (iv) | |
| S8 the dilation of one sweep does not vanish, the two-sweep sum does | 625 ("of a step") | ✅ | single-sweep dilation up to $0.18$ cell volumes, sum $5.6\times10^{-17}$ | `code_01` (v) | |

## Errors and consequences

No error in the formula. Precision to add (P1): the equivalence "dilation zero $\Leftrightarrow$ mixture constraint" is a cell-wise statement on $\{\chi_c=1\}$; because of $\chi_c$ the constraint is irrelevant in cells with $\chi_c=0$ (S6). The total volume is conserved when $m_c=0$ in every $\chi=1$ cell, which is weaker than the constraint in every cell, and the converse fails only through cancellation between different cells (the sum of dilations may vanish without each term vanishing).

Consequence for the phrase "which both two-velocity projections impose up to their tolerance": the projection imposes the constraint on the phase apertures **current at the projection**; these are the frozen ones of the next step only if the reconstruction used by the projection is the one that starts the transport step (no regrid, no re-reconstruction in between). The text says exactly this in the sentence before ("if no regrid intervened they are those of the preceding projection", `05_vof_transport.tex:588-590`). Not verified here (needs the Chapter 10 projection, out of scope).

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| dilation $=\chi_c\delta t\,m_c$ | **sharp** (an identity) | per-cell agreement $2.9\times10^{-17}$ | `code_01` | — |
| vanishing iff constraint, cell-wise on $\chi=1$ | **sharp** | S6/S7 | `code_01` | state the restriction to $\{\chi=1\}$ |
| total volume conservation needs $m_c=0$ in every $\chi=1$ cell | **sufficient, not necessary** for the total | the total only needs $\sum_c\chi_cm_c=0$ | — | say "conserved if" |
| "frozen" apertures | **needed** | the carrier uses the start-of-step apertures in both sweeps; the current ones change after sweep 1 | `code_01` | — |

## Open points

- The statement that both two-velocity projections impose the mixture constraint on the apertures of the next transport step (Chapter 10) was not audited. If the audited chain "projection → transport" is to be documented, a check is: run `step!` for the two-velocity model, then read `_freeze_phase_apertures!` and the divergence $m_c$ of the projected phase velocities and compare with the projection tolerance.
- The frozen carrier can refuse a step at the stable time step even on single-level cut meshes (strip too small behind a subface; also for pure donors): `vof-boundedness`, `code_03`.

## Proposed replacement text

`05_vof_transport.tex:625-631`, replace "The dilation of a step, ..., vanishes where ... (\cref{ch:projection})." by

```latex
\phantomsection\label{prop:vof:frozen-dilation}The dilation of a step in a cell $c$ with $\indicator{\liq}_c=1$ is
$\dt\sum_{f}\orient{c}{f}\meas{f}(\phaseaperture{\liq}{f}\velo^{\liq}_f+\phaseaperture{\gas}{f}\velo^{\gas}_f)$,
$\dt$ times the discrete mixture divergence of $c$ on the frozen apertures~\thesis{eq.~(6.19)}: it
vanishes exactly when the constraint holds in $c$; in a cell with $\indicator{\liq}_c=0$ there is
no dilation whatever the divergence. Of the two sweeps only the sum vanishes. The two-velocity
projections impose the constraint, up to their tolerance, on the phase apertures of the
reconstruction at the time of the projection (\cref{ch:projection}); these are the frozen apertures if no
regrid intervened.
```

**Applied** on 2026-11-08 to `docs/techref/chapters/05_vof_transport.tex` (uncommitted, branch `docs/techref-vof`), new-file lines `646-653`; label `prop:vof:frozen-dilation`. The techref builds with 0 undefined references and `check_links.py` passes.

## History

- 2026-11-08: first audit. Symbolic algebra (9 checks); package frozen carrier vs independent replica on mixture-solenoidal fields with velocity jumps (composite and cut meshes, 33 checks), perturbation tests of the cell-wise condition. Outcome: ⚠️ verified, one precision (cell-wise on $\chi=1$), Chapter 10 link unaudited.
