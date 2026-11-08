# Audit: eq:vof:volume-balance — time-integrated volume balance (5.30)

- Source: `docs/techref/chapters/05_vof_transport.tex:795-816` (eq. (5.30) = `eq:vof:volume-balance` at 799-803, the two conservation bullets 806-815), built on the sweep update (5.6) `eq:vof:sweep` (115-125) and its two-sweep sum (5.8) `eq:vof:step` (145-152); commit `ec20986` (branch `docs/techref-vof`, tree clean). Code: `src/vof/fluxes.jl` (`_advect_wy!`), `src/vof/workspace.jl` (`_check_wy_workspace`).
- Date: 2026-11-08
- Overall status: ✅ **verified.** (5.30) holds to round-off for the three carriers with a dilation as large as $1.4$ cell volumes and snapped volumes up to $0.25$; the two conservation bullets are exactly right (the divergence-free condition is needed on the cells with $\chi=1$ only).
- Rerun: `julia --project=@claude-symbolics proof_review/eq-vof-volume-balance/scripts/run_all.jl`. Last run: all checks passed (symbolic group and package group).
- Label: `eq:vof:volume-balance` (the identity); the bullets have no label.

## Statement and assumptions

$$V^{{\rm l},n+1}-V^{{\rm l},n}=\delta t\sum_c\chi_c\,|c|(\mathfrak D\upsilon^{\rm o})_c+\sum_{j=1}^{2}\sum_c|c^{\rm o}|\,\delta C^{[j]}_c,\qquad V^{\rm l}=\sum_c|c^{\rm o}|C_c=\sum_c|c|\phi_c ,$$

with $\upsilon^{\rm o}$ the carrier of the path (one velocity: $a_fu_f$; frozen: $\sum_\pi a^\pi_fu^\pi_f$; projected: $a_f\widehat u_f$), $\chi_c=[C^n_c>\tfrac12]$ frozen for the step, $\delta C^{[j]}_c$ the clamp/snap change (5.14). "The liquid volumes of the interior faces cancel and the boundary faces carry none; the gas volume changes by the opposite amount; this holds for all three carriers up to rounding." Consequence: each phase volume is conserved to rounding if the boundaries are impermeable/periodic, the carrier is discretely divergence free in every cell with $\chi_c=1$ (one velocity $(\mathfrak D^{\rm o}u)_c=0$; frozen $\sum_fo|f|(a^{\rm l}u^{\rm l}+a^{\rm g}u^{\rm g})=0$; projected $(\mathfrak D^{\rm o}\widehat u)_c=0$) and no fraction is clamped or snapped.

Assumptions used: each owned segment enters its two cells with opposite orientation and one flux value (also at hanging segments and periodic seams); the same carrier $\upsilon^{\rm o}$ and the same frozen $\chi$ in both sweeps; zero boundary velocity (enforced by the code, `stable_timestep` throws otherwise); the reconstruction between sweeps does not change the fractions beyond rounding (it re-applies the floor to already snapped values).

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| S1 sum over cells of the two sweeps (5.6) gives (5.30): flux terms cancel, dilation $\sum_c\chi_c\sum_fo\,O_f$, snaps $\sum|c^{\rm o}|\delta C$ | 799-803 | ✅ | symbolic (coarse cell with two hanging segments and two fine cells, both sweeps, symbolic fluxes/indicators/snaps) | `step01` S1 | orientation conventions $o_{c,f}=+1$ for the minus cell |
| S2 (5.8) cell by cell sums to the same | 145-152 | ✅ | symbolic | `step01` S2 | $|c|(\mathfrak D\upsilon^{\rm o})_c=\sum_f o|f|\upsilon^{\rm o}_f$ |
| S3 gas volume changes by the opposite amount | 803-805 | ✅ | symbolic S3; code: $\lvert\Delta V^{\rm g}+\Delta V^{\rm l}\rvert\le1.5\times10^{-16}$ (relative to the open volume) | `step01`, `code_01` | gas := open − liquid, identity by definition |
| S4 identity for the three carriers incl. non-zero dilation and snaps | 803-805 | ✅ | package `advect!` (volume read from the `fraction` field) vs independent replica (polygon fluxes, update, clamp/snap, package reconstruction): 72 random steps (graded AMR meshes with hanging faces, anisotropic cells, plain and walled with a random wall; one-velocity, frozen, projected with loose tolerance $10^{-3}$; both orders; default floor and floor 0.06) | `code_01` | worst relative residual $4.7\times10^{-16}$ (plain/one-velocity); max $\lvert$dilation$\rvert=1.36$ cell volumes; max $\lvert$snapped volume$\rvert=0.25$; 36 of 72 steps with snaps; fractions equal the replica to $2.8\times10^{-16}$ |
| S5 boundary faces carry no volume | 803-805 | ✅ | code: nonzero boundary velocity is rejected (`ArgumentError`) | `code_01` | the claim is enforced, not merely assumed |
| S6 solenoidal carriers conserve volume to round-off (uniform, composite, cut, film fixtures) | 806-810 | ✅ | docs fixtures with streamfunction / open streamfunction velocities, both orders | `code_02` (i) | `:film` with the open streamfunction needs the stable step reduced by $2^2$ (cut-donor strip limitation, issue #52) |
| S7 divergence-freeness is needed only in cells with $\chi=1$ | 806-808 | ✅ | perturbation on faces whose two cells have $\chi=0$ (more than 100 faces in each of 6 cases): $\lvert\Delta V\rvert\le3.6\times10^{-15}$ (absolute, total volume $\approx 20$); perturbation next to a $\chi=1$ cell: $\Delta V$ = replica dilation to $10^{-13}$, relative size $3.7\times10^{-4}$–$1.4\times10^{-2}$ | `code_02` (ii)(iii) | |
| S8 the dilation of one sweep does not vanish, the sum of the two does | §5.3 (156-159: "cancels over the two sweeps") | ✅ | solenoidal drop fixture: single-sweep dilation up to $0.25$ cell volume, sum $2.8\times10^{-17}$ | `code_02` (iv) | |

## Errors and consequences

No error. The statement is an identity, with the hypothesis structure exactly as the text says. Remarks (no correction needed):

- The two bullets are sufficient, not necessary: only the sum $\sum_c\chi_c|c|(\mathfrak D\upsilon^{\rm o})_c+\sum|c^{\rm o}|\delta C$ enters (5.30), so cell-wise dilations of opposite signs cancel in the total. Say "conserved if" (as written), not "iff".
- The test velocity fields used in the existing checks never needed the reduced step; the cut-donor strip limitation (issue #52) appears already with a solenoidal (open-streamfunction) carrier on the `:film` fixture: $\delta t_{\rm WY}/4$ was the largest halving needed. This does not affect (5.30) (it concerns the existence of the donating depth) but should be kept in mind when quoting "to rounding on all fixtures".

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| (5.30) | **sharp** (an identity) | residual $\le4.7\times10^{-16}$ with dilation $O(1)$ and snaps $O(0.1)$ | `code_01` | — |
| "no fraction clamped or snapped" | **needed** | with snaps the residual of the unmodified identity is the snapped volume (up to 0.25) | `code_01` | state (5.30) with the snap term (as printed) |
| divergence-free in every $\chi=1$ cell | **sharp cell-wise**; not necessary for the total | perturbations on $\chi=0$-only faces do not change the volume; one $\chi=1$ face does, by exactly the dilation | `code_02` | add "(only the sum enters)" if desired |
| boundary faces carry no volume | **enforced** | boundary velocity $\ne0$ is rejected by `_check_wy_workspace` | `code_01` | — |

## Open points

- None blocking. The reconstruction between the sweeps (floor re-applied to already snapped values) is covered by the replica comparison (fractions agree to $2.8\times10^{-16}$), not proved separately.

## Proposed replacement text

None required. Optional sharpening of the last paragraph of Section 5.9 (`05_vof_transport.tex:806-815`): after "if, besides impermeable or periodic boundaries," keep the bullets and add after the list "(these conditions are sufficient: only the sums in~\cref{eq:vof:volume-balance} matter, so dilations of opposite sign in different cells may cancel)".

## History

- 2026-11-08: first audit. Symbolic cell-sum on a composite configuration (4 checks passed); package vs independent replica on 72 random steps for the three carriers with dilation and snaps (all residuals $\le4.7\times10^{-16}$); conservation bullets by solenoidal and perturbed carriers on the docs fixtures (82 checks passed). Outcome: ✅ verified, no correction.
