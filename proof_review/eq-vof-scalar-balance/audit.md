# Audit: eq:vof:scalar-balance — time-integrated phase-scalar balance (5.32), snap and reset terms

- Source: `docs/techref/chapters/05_vof_transport.tex:817-836` (eq. (5.32) = `eq:vof:scalar-balance` at 824-829, snap/reset sentence 830-836), update (5.18) `eq:vof:scalar` at 464-473, resets (5.28)-(5.29) `eq:vof:aux-gas`, `eq:vof:reset` at 733-751; commit `ec20986` (branch `docs/techref-vof`, tree clean). Code: `src/vof/fluxes.jl` (`_advect_wy!`), `src/vof/workspace.jl` (`_reset_auxiliary_gas_carrier!`, `_update_gas_reset_diagnostics!`).
- Date: 2026-11-08
- Overall status: ⚠️ **identity (5.32) verified; the sentence on snaps is wrong in sign for the gas (F1)**, and the reset term can be given explicitly (F2). The balance identity (5.32) holds to $2\times10^{-16}$ (relative to $\sum|c^{\rm o}|(\lvert\varphi^{\rm l}\rvert+\lvert\varphi^{\rm g}\rvert)$) for all eight face-state schemes, liquid and gas, one-velocity and frozen carriers, with a dilation of up to $5\times10^{-3}$ of that scale.
- Rerun: `julia --project=@claude-symbolics proof_review/eq-vof-scalar-balance/scripts/run_all.jl`. Last run: all checks passed (symbolic group and package group).
- Label: `eq:vof:scalar-balance` (the identity); the snap/reset sentence has no label.

## Statement and assumptions

If no fraction is clamped or snapped,
$$\sum_c|c^{\rm o}|(C^\pi_c\varphi^\pi_c)^{n+1}-\sum_c|c^{\rm o}|(C^\pi_c\varphi^\pi_c)^{n}=\delta t\sum_c\chi^\pi_c\varphi^{\pi,n}_c\,|c|(\mathfrak D\upsilon^{\rm o})_c ,$$
"so the frozen dilation values make the dilation of a scalar vanish together with that of the volume. A snap to a positive fraction adds $|c^{\rm o}|\delta C^{[j]}_c\varphi^{\pi,[j-1]}_c$, a snap to zero removes the cell's amount, and the projected carrier adds the reset of the gas amounts." Assumptions used: face-owned states $(\mathfrak F\varphi)_f$ (one value per owned segment, seen with opposite orientation by both cells — true for every scheme of (5.20) including the limiters, which use only face data); frozen dilation weights $\chi^\pi_c\varphi^{\pi,n}_c$ in **both** sweeps; carrier equal to the one used for the fractions; for the gas of the projected carrier $\chi^{\rm g}:=0$ (no dilation term) and amounts $G^*\varphi$ of the auxiliary carrier, reset to $(1-C^{\rm l})$ at the start of each sweep and implicitly at the end of the step (`integrate(q, :gas)` uses the physical complement).

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| S1 cell sum of (5.18) over the two sweeps gives (5.32) | 824-829 | ✅ | symbolic (coarse + 2 fine cells, face-owned states, frozen weights, both sweeps) | `step01` S1 | |
| S2 the dilation weight must be the frozen value $a^n$, not the current $a^{[j-1]}$ | 468-470 ("frozen at the start of the step") | ✅ | symbolic S2: with $a^{[1]}$ in sweep 2 the balance acquires $\sum_c\chi_c(a^{[1]}_c-a^n_c)\sum_{f\in\mathcal F^{[2]}}o\,O_f$ | `step01` | justifies the freezing |
| S3 with snaps: deviation $={\rm dev}_{j,c}$ per cell and sweep, $C_s a_{\rm new}-A_{518}$ | 830-833 | ✅ | symbolic S3 | `step01`, `code_01` (ii) | ${\rm dev}=(C_s-C_{\rm pre})a^{[j-1]}$ (positive snap), $-A_{518}$ (snap to zero), see `eq-vof-fluctuation` |
| S4 projected carrier, gas: aux amounts without dilation + reset terms | 830-833, 733-751 | ✅ | symbolic S4: $P_2-P_0=\sum_j\sum_c|c^{\rm o}|(1-C^{[j]}-G^{*[j]})a^{[j]}$ | `step01`, `code_01` (iii) | the printed "(5.29)" discrepancy has the opposite sign convention ($E^{\rm g}=\sum(G^*-1+C)$): amounts change by $-E$ weighted by $a$ |
| S5 identity (5.32), liquid and gas, all eight schemes, one-velocity and frozen carriers, plain and cut meshes | 824-829 | ✅ | package (`integrate(q,·)` before/after `advect!`) vs. carrier volumes from the definitions: 32 steps (1 refused by the replica's strip error), no snaps; both orders | `code_01` (i) | relative residual $6.2\times10^{-17}$ (liquid), $2.1\times10^{-16}$ (gas); dilation up to $5.4\times10^{-3}$ of the amount scale |
| S6 identity for the liquid scalar of the projected carrier | 824-829 | ✅ | package vs definitions: 7 steps (tolerance $10^{-3}$) | `code_01` (i') | residual $5.4\times10^{-17}$; the dilation of this carrier is small here ($1.1\times10^{-7}$ relative) |
| S7 snap terms: deviation of the unmodified identity equals the predicted corrections | 830-833 | ✅ | 24 steps with floor 0.06 (23 with snaps), Upwind/Centred/LW, one-velocity and frozen carriers: residual with the corrections $\le1.6\times10^{-16}$ | `code_01` (ii) | |
| S8 sign of the snap term for the gas | 830-831 | ❌ as printed (F1) | the printed term read with the liquid $\delta C$ for both phases misses the gas by $2\lvert\text{term}\rvert$: literal-vs-correct difference $2.8\times10^{-3}$ for a term of size $1.4\times10^{-3}$ (relative) | `code_01` (ii) | correct: $\delta C^{\rm g}=-\delta C^{\rm l}$ |
| S9 reset term of the projected carrier | 831-833 | ✅ | 12 steps: residual with the explicit reset $1.1\times10^{-16}$, without $2.5\times10^{-3}$ | `code_01` (iii) | |

## Errors and consequences

**F1 (sign for the gas).** In "A snap to a positive fraction adds $|c^{\rm o}|\delta C^{[j]}_c\varphi^{\pi,[j-1]}_c$" the symbol $\delta C^{[j]}_c$ is the change of the *liquid* fraction (5.14). For $\pi={\rm g}$ the snapped gas volume is $-|c^{\rm o}|\delta C^{[j]}_c$. Consequence: the identity as printed is correct for the liquid scalar and has the wrong sign for the gas scalar (with the non-auxiliary carriers); nothing else in the chapter depends on it. Replacement below.

**F2 (reset term unspecified).** "the projected carrier adds the reset of the gas amounts" is correct, but the sign/convention can be made explicit: the gas amount changes by $\sum_{j=1}^2\sum_c|c^{\rm o}|(1-C^{{\rm l},[j]}_c-G^{*,[j]}_c)\varphi^{{\rm g},[j]}_c$ (the second reset is the implicit one at the end of the step), which is $-E^{\rm g}$ weighted by the intrinsic values; for constant $\varphi^{\rm g}$ it reduces to minus the signed discrepancy (5.29) times $\varphi^{\rm g}$ (numerically confirmed through the explicit formula, not through `phase_flux_diagnostics`).

No error in (5.32) itself.

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| (5.32) | **sharp** (an identity for any face state) | residuals $\le2.1\times10^{-16}$ with non-zero dilation, 8 schemes | `code_01` | — |
| frozen dilation weights | **needed** | symbolic S2: current values would add $\sum_c\chi_c(a^{[1]}-a^n)\,(\text{sweep-2 carrier divergence})$ | `step01` | — |
| "no fraction clamped or snapped" | **needed** | residual of the unmodified identity equals the snap terms; snap terms exact | `code_01` | state the gas sign (F1) |
| reset term of the projected carrier | **needed and exact** | residual drops from $2.5\times10^{-3}$ to $1.1\times10^{-16}$ with the explicit formula | `code_01` | give the formula (F2) |
| "the intrinsic values are not bounded by these identities" | **correct** (amount identities say nothing about extrema) | — | — | — |

## Open points

- The replica of the snap terms covers Upwind, Centred, Lax–Wendroff; for the limiter schemes the identity was verified without snaps (face-state independent), and the snap term formula is face-state independent as well (S3 is symbolic with arbitrary states).
- The projected-carrier liquid check has a small dilation (loose tolerance $10^{-3}$ gives $1.1\times10^{-7}$ relative); the identity is carrier independent and is tested with large dilations for the other two carriers.

## Proposed replacement text

`05_vof_transport.tex:830-833`, replace "A snap to a positive fraction adds ... (\cref{sec:vof:projected})." by

```latex
A snap to a positive fraction adds
$\meas{\opencell{c}}\,\delta\condfrac{\phase,\splitlevel{j}}{c}\,\avar^{\phase,\splitlevel{j-1}}_c$
with $\delta\condfrac{\liq,\splitlevel{j}}{c}=\snapchange{j}{c}$ and
$\delta\condfrac{\gas,\splitlevel{j}}{c}=-\snapchange{j}{c}$, a snap to zero removes the
cell's amount, and the projected carrier adds, for the gas, the resets of the auxiliary
carrier (\cref{sec:vof:projected}),
$\sum_{j=1}^{2}\sum_c\meas{\opencell{c}}\roundpar{1-\condfrac{\liq,\splitlevel{j}}{c}-\auxgas{,\splitlevel{j}}{c}}\avar^{\gas,\splitlevel{j}}_c$
(the second term of the sum is the reset implied by reading the amount through the physical
complement at the end of the step).
```

**Applied** on 2026-11-08 to `docs/techref/chapters/05_vof_transport.tex` (uncommitted, branch `docs/techref-vof`), new-file lines `866-873`; (5.31) is now the new bound of the projected carrier, so this identity is (5.32). The techref builds with 0 undefined references and `check_links.py` passes.

## History

- 2026-11-08: first audit. Symbolic balance on a composite configuration incl. frozen-weight necessity, snaps and resets (5 checks passed); package vs definitions on 32+7 no-snap steps, 24 snap steps and 12 projected-gas steps. Outcome: ⚠️ identity verified; sign correction for the gas (F1), explicit reset formula (F2).
