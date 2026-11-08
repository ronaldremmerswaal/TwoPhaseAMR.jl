# Audit: eq:vof:fluctuation — fluctuation form of the phase-scalar update and its snap terms

- Source: `docs/techref/chapters/05_vof_transport.tex:456-492` (update (5.18) = `eq:vof:scalar` at 464-473, fluctuation form (5.19) = `eq:vof:fluctuation` at 474-483, snap sentences 483-488), commit `ec20986` (branch `docs/techref-vof`, working tree clean). Code: `src/vof/fluxes.jl` (`_advect_wy!`, scalar block and update loop), `src/vof/fluxes.jl` (`_phase_flux_state`).
- Date: 2026-11-08
- Overall status: ⚠️ **form verified, one sentence needs a correction for the gas.** (5.19) is an exact algebraic rearrangement of (5.18); the package's phase scalars after a full step equal an independent evaluation of (5.19) to round-off on 58 random steps (75 840 cell-sweep-phase events); the snap statements are exact for the liquid. The sentence on the snapped volume has the wrong sign for the gas (F1) and one statement can be sharpened (F2).
- Rerun: `julia --project=@claude-symbolics proof_review/eq-vof-fluctuation/scripts/run_all.jl` (two sequential child processes: Symbolics group under `@claude-symbolics`, package group under the repository environment; flags `--heap-size-hint=3G -t 1`). Last run: all checks passed.
- Label: `eq:vof:fluctuation` exists. The sentence on snaps (483-488) has no label of its own.

## Statement and assumptions

For one sweep $j$ along axis $k_j$, cell $c$, phase $\pi\in\{\rm l,g\}$, with open volume $|c^{\rm o}|$, previous fraction $C^{\pi,[j-1]}_c$ and intrinsic value $\varphi^{\pi,[j-1]}_c$, frozen dilation value $\chi^\pi_c\varphi^{\pi,n}_c$, phase volumes $\upsilon^{\pi,[j]}_f$, carrier $\upsilon^{\rm o}_f$ and face states $(\mathfrak F\varphi)_f$:

* (5.18)  $|c^{\rm o}|(C\varphi)^{[j]}=|c^{\rm o}|(C\varphi)^{[j-1]}-\sum_{f\in\mathcal F(c)\cap\mathcal F^{[k_j]}}o_{c,f}\,\delta t|f|\bigl(\upsilon^{\pi,[j]}_f(\mathfrak F\varphi)_f-\chi^\pi_c\varphi^{\pi,n}_c\upsilon^{\rm o}_f\bigr)$.
* (5.19)  $\varphi^{[j]}_c=\varphi^{[j-1]}_c+\dfrac{1}{|c^{\rm o}|C^{\pi,[j]}_c}\sum_f o_{c,f}\delta t|f|\bigl(\upsilon^{\pi,[j]}_f(\varphi^{[j-1]}_c-(\mathfrak F\varphi)_f)+\chi^\pi_c\upsilon^{\rm o}_f(\varphi^{n}_c-\varphi^{[j-1]}_c)\bigr)$ if $C^{\pi,[j]}_c>\varepsilon^C$ (the **snapped** fraction), $\varphi^{[j]}_c=\varphi^{[j-1]}_c$ otherwise.
* Claims: without snapping (5.19) equals (5.18); a snap to a positive fraction leaves the snapped volume $|c^{\rm o}|\delta C^{[j]}_c$ with the value $\varphi^{[j-1]}_c$; a snap to zero removes the amount of the cell; a constant $\varphi$ gives a zero increment (constants preserved exactly, also across level transitions and in cut cells).

Implicit assumptions that should be stated: (a) $\delta C^{[j]}_c$ (`\snapchange{j}{c}`, defined in (5.14) as the change of the **liquid** conditional fraction) is the right symbol only for the liquid — for the gas the snapped volume is $-|c^{\rm o}|\delta C^{[j]}_c$ (F1); (b) for the auxiliary gas carrier (projected-liquid path) the clamp acts at zero only and the snap is to zero (no upper clamp, no snap to 1), the denominator being $C^{\rm g*}$; (c) the face states are those of (5.20), which are upwind values at every face of a cell that is not full of the phase before or after the sweep — this is what makes the removed amount at a snap to zero small (F2).

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| S1 (5.19) ≡ (5.18) without snapping: $C_{\rm pre}\,\varphi^{[j]}=A_{518}$ | 474-483 | ✅ | symbolic (Symbolics.jl; one cell, three faces, arbitrary $o,\upsilon,\upsilon^{\rm o},$ states; denominators kept as symbols, identities polynomial in $1/V$) | `step01` S1 | exact identity |
| S2 snap to a positive fraction: $C_s\varphi^{[j]}-A_{518}=(C_s-C_{\rm pre})\varphi^{[j-1]}$ | 483-485 | ✅ liquid; ⚠️ gas (sign) | symbolic S2; code (340 events, 197 gas): deviation $\le1.8\times10^{-16}$ from $(C_s-C_{\rm pre})\varphi^{[j-1]}$ | `step01`, `code_01` | the snapped *phase* volume carries $\varphi^{[j-1]}$; the phase volume changed is $|c^{\rm o}|(C^\pi_s-C^\pi_{\rm pre})$, i.e. $+\delta C$ liquid, $-\delta C$ gas (F1) |
| S3 snap to zero removes the amount of (5.18) | 485-486 | ✅ | symbolic S3 ($A_{518}=C_{\rm pre}\varphi^{[j-1]}+I$); code (376 events): $\lvert{\rm dev}+A_{518}\rvert\le10^{-13}(1+\lvert A_{518}\rvert)$ | `step01`, `code_01` | removed amount is $O(\varepsilon^C\max\lvert\varphi\rvert)$ (F2): observed $\lvert A_{518}\rvert\le1.37\,\varepsilon^C(1+\lvert\varphi\rvert)$ |
| S4 gas as the physical complement: pre-clamp gas fraction $=1-C_{\rm pre}$, snapped gas volume $=-\delta C$ | 464-483, 5.14 | ✅ | symbolic S4; code (gas events in `code_01`) | `step01`, `code_01` | gas indicator $1-\chi$, gas volume $\upsilon^{\rm o}-\upsilon^{\rm l}$ |
| S5 a phase that is absent and only received does not keep the stale value (weights add up to 1) | implicit in "constants preserved", 474-483 | ✅ | symbolic S5: $\varphi^{[j]}=\sum w_if_i/\sum w_i$ independent of $\varphi^{[j-1]}$ | `step01` | justifies that near-empty cells may keep arbitrary stored values |
| S6 constants: increment exactly zero | 486-488 | ✅ | symbolic S6; code: bitwise equality $3.0$/$7.0$ in all 12 combinations (plain/cut × three carriers × default/large floor, LW scheme, with snaps) | `step01`, `code_01` | holds also with snaps: $0/C_s=0$ |
| S7 code ≡ (5.19) with the snapped denominator, fluxes by independent polygon clipping, face states (5.20) for Upwind/Centred/LW | 474-483, 498-509 | ✅ | package `advect!` vs `VofAudit.scalar_replica` (+ `Docs.replica_fluxes`): 58 steps (2 more refused by the package's strip error), both sweep orders, one-velocity / frozen / projected carriers, graded AMR meshes with hanging faces, random walls, anisotropic cells, default floor and floor 0.06 | `code_01` | amount-weighted difference $\le2.0\times10^{-15}$; intrinsic values $\le3.5\times10^{-13}$ (liquid), $\le4.1\times10^{-11}$ (gas, conditioning $1/C_s$ near the floor) |
| S8 each snap changes the phase volume by at most the floor, each clamp by at most $128\,\epsilon_{\rm mach}$ | 5.5 (cited by the snap sentences) | ✅ | code: largest snap $0.05995\le0.06$, largest clamp $3.4\times10^{-15}\le128\epsilon_{\rm mach}=2.8\times10^{-14}$ | `code_01` | |

Coverage of `code_01`: 75 840 events = 75 124 without snap, 340 snaps to a positive fraction (197 for the gas), 376 snaps to zero.

## Errors and consequences

No error in (5.19). Two wording defects:

**F1 (sign for the gas; notation).** The text writes the snapped volume as $|c^{\rm o}|\delta C^{[j]}_c$ for both phases, but `\snapchange` is the change of the liquid fraction. For the gas (non-auxiliary) the snapped volume is $|c^{\rm o}|\bigl((1-C^{\rm l}_s)-(1-C^{\rm l}_{\rm pre})\bigr)=-|c^{\rm o}|\delta C_c$ (symbolic S4, code: 197 gas events agree with $(C^{\rm g}_s-C^{\rm g}_{\rm pre})\varphi^{{\rm g},[j-1]}$, and *disagree in sign* with the literal reading, see `eq-vof-scalar-balance`). The same sentence is repeated in Section 5.9 (`05_vof_transport.tex:830-833`), see the audit of `eq:vof:scalar-balance`.

**F2 (size of the removed amount).** "A snap to zero removes the amount of the cell" is exact; it can be added that the amount is of order $\varepsilon^C|c^{\rm o}|\max|\varphi|$. Reason: a cell snapped to zero has pre-clamp fraction $\le\varepsilon^C$, so every face of it is "near the interface" in the sense of (5.20) and uses upwind states, and the amount (5.18) is a combination of values of the cell and its donors with weights adding up to the pre-clamp fraction. Observed: $\lvert A_{518}\rvert/(\varepsilon^C(1+\lvert\varphi\rvert))\le1.37$.

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| (5.19) = (5.18) without snapping | **sharp** (an identity) | symbolic S1 (polynomial identity in $1/V$) and code deviation $\le7.6\times10^{-16}$ | `step01`, `code_01` | — |
| snapped volume keeps $\varphi^{[j-1]}$ | **sharp** (an identity: $C_s\varphi^{[j]}-A_{518}=(C_s-C_{\rm pre})\varphi^{[j-1]}$) | S2, deviation $\le1.8\times10^{-16}$ | `step01`, `code_01` | state for the gas with $-\delta C$ |
| "a snap to zero removes the amount" | **sharp**; size $O(\varepsilon^C)$ | removed amount $\le1.37\,\varepsilon^C(1+\lvert\varphi\rvert)$ | `code_01` | add the size remark (F2) |
| constants preserved exactly | **sharp** (bitwise) | 12/12 combinations, with and without snaps | `code_01` | — |
| the guard $C^{[j]}>\varepsilon^C$ | **needed** | for $C_s\le\varepsilon^C$ the division is by a number that can be $0$ (a snapped-to-zero cell): the update is skipped, the stored value stays | `code_01` | — |

## Open points

- (5.20) was exercised only for Upwind, Centred and Lax–Wendroff (the replica implements the near-interface rule and the LW state as printed; they agree with the code), not for the limiter schemes of Table 5.1. The balance identities do not depend on the face state, so the audits of (5.30)/(5.32) cover all eight schemes.
- Two of 60 random steps were refused by the package (`frozen phase-subface flux exceeds the donor open volume`) at the step `stable_timestep` returned: the frozen-carrier strip limitation (see the audit `vof-boundedness`, `code_03`).

## Proposed replacement text

`05_vof_transport.tex:483-488`, replace "Without snapping ... in cut cells." by

```latex
Without snapping~\cref{eq:vof:fluctuation} equals~\cref{eq:vof:scalar}: it is an algebraic
rearrangement. Let $\delta\condfrac{\phase,\splitlevel{j}}{c}\defeq
\condfrac{\phase,\splitlevel{j}}{c}-\preclamp{\phase,\splitlevel{j}}{c}$ be the change of the
phase fraction by clamping and snapping; it is $\snapchange{j}{c}$ for the liquid, $-\snapchange{j}{c}$
for the gas (the complement of the liquid), and for the auxiliary gas carrier
of \cref{sec:vof:projected} the change by the clamp and the snap to zero. If the snapped fraction
exceeds $\ffloor$, the amount after the sweep exceeds~\cref{eq:vof:scalar} by
$\meas{\opencell{c}}\,\delta\condfrac{\phase,\splitlevel{j}}{c}\,\avar^{\phase,\splitlevel{j-1}}_c$,
so a snap to a positive fraction leaves the snapped volume with the value
$\avar^{\phase,\splitlevel{j-1}}_c$. A snap to zero removes the amount that~\cref{eq:vof:scalar}
assigns to the cell; since all faces of such a cell use upwind states, this is of order
$\ffloor\meas{\opencell{c}}\max\abs{\avar}$. A constant $\avar^{\phase}$ gives a zero increment, so
constants are preserved exactly (bitwise), also across level transitions, in cut cells and when
fractions snap.
```

**Applied** on 2026-11-08 to `docs/techref/chapters/05_vof_transport.tex` (uncommitted, branch `docs/techref-vof`), new-file lines `496-509`. The techref builds with 0 undefined references and `check_links.py` passes.

## History

- 2026-11-08: first audit. Symbolic algebra of (5.19) with and without snapping (S1–S6, 10 checks passed); package vs independent replica on 58 random steps (75 840 events), constants check (12 combinations). Outcome: ⚠️ form verified, two wording corrections (F1, F2).
