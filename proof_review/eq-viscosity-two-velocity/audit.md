# Audit: eq:viscosity:two-velocity — two-velocity diffusion substep (mixture, common increment, jump, combined momentum)

- Source: `docs/techref/chapters/09_viscosity.tex:419-446`, commit `03bad7a`
- Date: 2026-11-09
- Overall status: ⚠️ formulas and the four consequences verified; three precisions needed (covered faces of a solid, open dual volume, clamped fraction) and the sign inherited from `eq:viscosity:momentum`
- Rerun: `julia --project=@claude-symbolics proof_review/eq-viscosity-two-velocity/scripts/run_all.jl`

## Statement and assumptions

$\bar u^{[1]}_f=\bigl(\rho^{\rm l}\hat C_fu^{\rm l,[1]}_f+\rho^{\rm g}(1-\hat C_f)u^{\rm g,[1]}_f\bigr)/\bar\rho_f$ (absent phase: product $0$; $0$ on faces without open volume); $\bar u^{[2]}$ from eq. `phase-midpoint` with $\bar u^{[1]}$ and the mixture density; $u^{\phi,[2]}=u^{\phi,[1]}+(\bar u^{[2]}-\bar u^{[1]})$ (absent values stay absent). Consequences claimed: (a) the jump $u^{\rm g}-u^{\rm l}$ is unchanged, (b) the mixture of the new phase velocities is $\bar u^{[2]}$, (c) the combined momentum $\sum_\phi\sum_f\rho^\phi\,\text{stagfrac}^\phi_f\,\text{dualvol}_f\,u^\phi_f$ changes as the one-velocity momentum, (d) $u^{\rm l}=u^{\rm g}$ is diffused as in the one-velocity model. Implicit: absent velocities (NaN) exactly where the phase has zero mass at the face.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| mixture formula, NaN products 0, covered faces 0 | l. 425-432 | ✅ | independent evaluation on every face of three fixtures (walled, cut, periodic), error 0 | `code_01_two_velocity.jl` | NaN appear exactly where `_mark_inactive_phase_velocities!` sets them: fraction $=0$ (liquid) or $=1$ (gas), i.e. zero phase mass, so `_phase_product` returns 0 consistently |
| (a) jump unchanged | l. 438 | ✅ | 168 faces with both phases active, $|\Delta\text{jump}|\le4.4\times10^{-16}$; NaN stay NaN | same | |
| (b) mixture of the new velocities $=\bar u^{[2]}$ | l. 438-439 | ⚠️ | exact ($\le1.4\times10^{-16}$) on all faces with open volume (walled, periodic, cut); **fails on covered faces of a solid**: mixture is 0 by construction, $\bar u^{[2]}$ there is $\le2.6\times10^{-5}\max|\bar u^{[2]}|$ | same | covered unknowns are only penalty-tied (conductance 1, unit algebraic volume, and the shear least-squares rows of the vertex stencils include them), so $\bar u^{[2]}\ne0$ on them. Immaterial for momentum (open volume 0) but the statement "mixture $=\bar u^{[2]}$" holds on faces with open volume |
| (c) combined momentum | l. 440-443 | ⚠️ | with the **open** dual volume: gap $\le10^{-16}$ (all three fixtures, both components); periodic: equals the one-velocity change and $=\mp\mathbf 1^\top\mathsf r$ to $10^{-12}$ (sign as in `eq-viscosity-momentum`); with the **uncut** dual volume on the cut mesh the gap is $1.5\times10^{-6}$ and $1.6\times10^{-7}$ (components 1, 2) | same | the chapter writes $\meas{\omega_f}$ = `\dualvol{f}`; with a solid it must be `\opendualvol{f}`; the mixture uses the clamped fraction $\hat C_f$ while the sum uses `\stagfrac` (unclamped); the difference is round-off only (clamp inactive in all fixtures) |
| (d) $u^{\rm l}=u^{\rm g}$ | l. 444-446 | ✅ | both updated phases bitwise equal; equal to the one-velocity solve to $4.4\times10^{-16}$ (open faces) | same | |

## Errors and consequences

- **F1** (inherited sign): the "changes exactly as the one-velocity momentum (eq. `momentum`)" is right; only the sign inside eq. `momentum` must be corrected.
- **F2** (precision): "the mixture of the new phase velocities is $\bar u^{[2]}$" holds on faces with open volume; on faces of zero open volume the mixture is $0$ and $\bar u^{[2]}$ is not.
- **F3** (precision): combined momentum needs the open dual volume with a solid and the clamped fraction.

Replacement for the last sentences of the paragraph (l. 438-446):

```latex
The velocity jump $\velo^{\gas}_f-\velo^{\liq}_f$ is unchanged and, on the faces with open volume, the mixture of the new phase velocities
is $\onevelo^{\stage{2}}$ (on a covered face the mixture is zero by definition and $\onevelo^{\stage{2}}$ is only the penalty-tied
value of the solve, of the size of the solve tolerance times the coupling). The combined momentum
$\sum_\phase\sum_f\dens^{\phase}\,\clampfrac[\phase]{f}\,\opendualvol{f}\,\velo^{\phase}_f$, with
$\clampfrac[\liq]{f}=\clampfrac{f}$ and $\clampfrac[\gas]{f}=1-\clampfrac{f}$, changes exactly as the one-velocity momentum~\cref{eq:viscosity:momentum}
(the open dual volume is $\dualvol{f}$ without a solid), and a continuous velocity, $\velo^{\liq}=\velo^{\gas}$, is diffused exactly as in the one-velocity model.
```
(`\clampfrac` has one argument in the notation; write `\clampfrac{f}` and `1-\clampfrac{f}` instead of the indexed form, or define `\clampfrac[\phase]{f}`.)

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| (a), (d) | **sharp** (exact identities) | $4\times10^{-16}$ | `code_01_two_velocity.jl` | none |
| (b) | **sharp on open faces**, false on covered faces | covered $\bar u^{[2]}\le2.6\times10^{-5}$ relative; vanishes only if covered unknowns were pinned | same | state the restriction; or pin covered rows exactly (not a doc issue) |
| (c) open volume vs uncut volume | sharp (counterexample: $1.5\times10^{-6}$ relative gap with a solid) | | same | |

## Open points
- Covered-unknown penalty: size of its effect on the fluid velocities is the variable-operator analogue of the covered-row remark of `eq-viscosity-cut`; not quantified beyond $2.6\times10^{-5}$ in $\bar u^{[2]}$.

## History
- 2026-11-09: first audit.
