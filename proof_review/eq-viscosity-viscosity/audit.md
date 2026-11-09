# Audit: eq:viscosity:viscosity — geometric-mean viscosity with the conditional fraction

- Source: `docs/techref/chapters/09_viscosity.tex:208-217`, commit `03bad7a`; code `dynamic_viscosity`, `_phase_dynamic_viscosity`, `_update_dynamic_viscosity!`
- Date: 2026-11-09
- Overall status: ✅ verified
- Rerun: `julia --project=@claude-symbolics proof_review/eq-viscosity-viscosity/scripts/run_all.jl`

## Statement and assumptions
$\mu_c=(\mu^{\rm l})^{C_c}(\mu^{\rm g})^{1-C_c}=\exp(C_c\log\mu^{\rm l}+(1-C_c)\log\mu^{\rm g})$ with the conditional fraction $C_c$ (liquid volume / open volume of the cell); positive finite viscosities.

## Steps
| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| closed form = `dynamic_viscosity` (both written forms) | l. 214-217 | ✅ | 7 fields (walled, cut, periodic fixtures; random walled/periodic with/without disc), $\mu\in\{(0.1,0.002),(10^3,10^{-2})\}$: $\le1.8\times10^{-15}$ relative; $\mu_{\rm g}\le\mu\le\mu_{\rm l}$ | `code_01_geometric_mean.jl` | |
| conditional fraction = liquid volume / open volume | l. 212 | ✅ | `_refresh_conditional_fraction!` (clamp, floor) read; $|C-\text{clamp}(V^{\rm l},0,\phi)/\phi|\le10^{-12}$ on cut cells | same | covered cell: $C=0$ ($\mu=\mu_{\rm g}$); its rows have zero weight |
| vertex viscosity = geometric mean at the open-volume average of $C$ | l. 267-271 (eq. `weights`) | ✅ | $\mu_v$ from the operator's weights/measures vs $\mu_{\rm l}^{\bar C}\mu_{\rm g}^{1-\bar C}$: $\le2.3\times10^{-15}$ | same | $\log\mu$ is affine in $C$ |
| refresh = rebuild (`_update_dynamic_viscosity!`) | l. 306-308 | ✅ | weights equal bitwise, stiffness $\le10^{-12}$ after changing both viscosities | same | |

## Errors and consequences
None.

## Sharpness
| Claim | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| geometric mean | n/a (modelling choice of the thesis) | equals the code | `code_01_geometric_mean.jl` | none |

## Open points
None.

## History
- 2026-11-09: first audit.
