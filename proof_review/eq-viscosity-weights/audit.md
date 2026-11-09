# Audit: eq:viscosity:weights — stress weights and vertex viscosity

- Source: `docs/techref/chapters/09_viscosity.tex:262-280`, commit `03bad7a`; code `_variable_staggered_diffusion` (measures, interpolation), `_update_log_dynamic_viscosity!`
- Date: 2026-11-09
- Overall status: ✅ verified (independent weights to $2\times10^{-15}$ on 18 meshes)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-viscosity-weights/scripts/run_all.jl` (runs the scripts of `eq-viscosity-rows` and `eq-viscosity-viscosity`)

## Statement and assumptions
$\mathsf W=\operatorname{diag}(2|c^{\rm o}|\mu_c,\ |\omega_v|\mu_v,\ \text{cond}_w\mu_{c(w)})$, $|\omega_v|=\sum_{c\ni v}\tfrac14\phi_c|c|$, $\log\mu_v=|\omega_v|^{-1}\sum_{c\ni v,\phi_c>0}\tfrac14\phi_c|c|\log\mu_c$ (geometric mean at the open-volume average of the conditional fractions). Wall rows: the conductance of eq. `cut` times the viscosity of the cut cell; a covered unknown the conductance one and the viscosity of $\cminus f$.

## Steps
| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| normal-row weights $2|c^{\rm o}|\mu_c$ (0 for covered cells) | l. 267-268 | ✅ | independent: relative $\le2\times10^{-15}$ (all 18 meshes); zero weights coincide | `../eq-viscosity-rows/scripts/code_01_rows.jl` | |
| vertex measures $|\omega_v|$; vertices with $|\omega_v|=0$ dropped | l. 269 | ✅ | sum $=$ open volume to $4\times10^{-16}$ | same | |
| $\log\mu_v$ open-volume mean; geometric mean at $\bar C$ | l. 269-271 | ✅ | independent mean $\le2.3\times10^{-15}$; refresh = rebuild | `../eq-viscosity-viscosity/scripts/code_01_geometric_mean.jl` | |
| wall rows: $\text{cond}_w\mu_{c(w)}$; covered: $1\cdot\mu_{\cminus f}$ | l. 272-275 | ✅ | wall weight difference $\le5\times10^{-16}$ (cut fixture, random discs); face and cell of each row checked | `../eq-viscosity-rows/scripts/code_01_rows.jl` | |

## Errors and consequences
None.

## Sharpness
| Claim | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| weights | identity with the code | $10^{-15}$ | `code_01_rows.jl` | none |
| "quadrature" interpretation $|\omega_v|$ (cell-corner rule) | second order for the dissipation integral | observed 2.0 (`eq-viscosity-quadratic`) | `eq-viscosity-quadratic` | none |

## Open points
None.

## History
- 2026-11-09: first audit.
