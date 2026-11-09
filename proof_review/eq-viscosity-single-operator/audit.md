# Audit: eq:viscosity:single-operator — the operator returned by `diffusion!` (single phase)

- Source: `docs/techref/chapters/09_viscosity.tex:125-131`, commit `03bad7a`; code `_apply_staggered_diffusion!`, `diffusion!`
- Date: 2026-11-09
- Overall status: ✅ verified
- Rerun: `julia --project=@claude-symbolics proof_review/eq-viscosity-single-operator/scripts/run_all.jl`

## Statement and assumptions
$\frac{\mu}{\rho}\Bigl((\lapstag k u)_f+V_f^{-1}\sum_{w:\,f(w)=f}\text{cond}_w\,u^{\rm wall}_w\Bigr)$ on the unknowns, zero on the boundary faces; the wall data are those of eq. `walls` at the time passed to `diffusion!`.

## Steps
| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| operator = $\nu(\mathsf Lu+V^{-1}\sum\text{cond}\,u_w)$, value rules, `viscosity` scaling, `time` | l. 125-131 | ✅ | independent assembly from `laplacian`, `volumes`, the wall list and the written-out value rules; 4 boundary specifications (NoSlip(lid(t)), FreeSlip, VelocityBoundaryConditions with Inflow/Outflow/NoSlip/FreeSlip, NoSlip()), $\nu\in\{1,0.37\}$, $t\in\{0,0.8\}$ (32 checks), error $\le10^{-12}$; boundary rates exactly 0 | `code_01_single_operator.jl` | |
| uniform periodic: $\nu\times$ five-point Laplacian | l. 105-106 | ✅ | independent stencil, $10^{-12}$ | same | |

## Errors and consequences
None.

## Sharpness
| Claim | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| equality of the formula | sharp | $10^{-12}$ | `code_01_single_operator.jl` | none |

## Open points
None.

## History
- 2026-11-09: first audit.
