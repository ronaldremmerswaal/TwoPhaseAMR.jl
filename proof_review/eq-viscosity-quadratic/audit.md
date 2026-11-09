# Audit: eq:viscosity:quadratic — the stiffness quadratic form (discrete dissipation)

- Source: `docs/techref/chapters/09_viscosity.tex:283-300`, commit `03bad7a`
- Date: 2026-11-09
- Overall status: ✅ verified (identity, symmetry, positive semidefiniteness, consistency with $\int2\mu\,\varepsilon:\varepsilon$ of order 2)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-viscosity-quadratic/scripts/run_all.jl`

## Statement and assumptions

$\mathsf S=\mathsf E^\top\mathsf W\mathsf E$ with $\mathsf W=\operatorname{diag}(2|c^{\rm o}|\mu_c,\ |\omega_v|\mu_v,\ \text{cond}_w\mu_{c(w)})\ge0$, hence $\mathsf S$ symmetric PSD and
$u^\top\mathsf Su=\sum_{c,k}2\mu_c|c^{\rm o}|(\partial_ku_k)_c^2+\sum_v\mu_v|\omega_v|\bigl((\mathcal D_2u_1)_v+(\mathcal D_1u_2)_v\bigr)^2+\sum_w\text{cond}_w\mu_{c(w)}u_{f(w)}^2$, "the counterpart of $\int2\mu\,\varepsilon:\varepsilon$". Implicit: positive viscosities and weights.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| $u^\top\mathsf Su=\sum_r w_r(\mathsf Eu)_r^2$ with the printed weights | l. 288-293 | ✅ | independent strain map and weights vs the code's stiffness: $\le10^{-12}$ relative on walled, cut (code's wall rows added), periodic fixtures, 5 random $u$ each (15 checks) | `code_01_quadratic_form.jl` | |
| $\mathsf S$ symmetric | l. 285 | ✅ | asymmetry exactly 0 (the code stores `_complete_symmetric_stiffness`) on 6 configurations | same | |
| positive semidefinite | l. 285 | ✅ | $\lambda_{\min}\ge-7\times10^{-17}\lambda_{\max}$; kernel: periodic composite 2 (the two constants), walled no-slip 0, cut 0, x-periodic channel free slip 1, no-slip 0, walled free slip 0 | same | positive definite with a no-slip wall or a solid; the text says "semidefinite" — correct and general |
| consistency with $\int2\mu\,\varepsilon:\varepsilon$ | l. 295-297 | ✅ | $u=\operatorname{curl}[\sin^2(\pi x/L)\sin^2(\pi y/L)]$, $\mu$ constant: relative error $1.3\times10^{-2},3.2\times10^{-3},8.0\times10^{-4}$ (order 1.994, 1.999) on uniform; composite $8.1\times10^{-3},2.0\times10^{-3},5.0\times10^{-4}$ (order 2.00, 2.00) | same | quadrature $\sum_v|\omega_v|$ (cell-corner rule) and central differences are second order, also at hanging vertices; first-order asymmetric shear rows (see `eq-viscosity-shear`) do not spoil the order of the sum |

## Errors and consequences
None.

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| quadratic-form identity | **sharp** (identity) | $10^{-12}$ | `code_01_quadratic_form.jl` | none |
| PSD | **sharp** | kernel dimension 2 on the periodic mesh | same | the kernel list above could be added in one sentence |
| "counterpart" of the dissipation | **second-order** (observed) | orders 1.99-2.00 on uniform and composite meshes | same | the continuous equality is of order 2, not only consistency |

## Open points
- Interface rows (viscosity jump, $\mu_v$ geometric mean) give only $O(h)$ at the interface; not tested (variable-$\mu$ order not claimed).

## History
- 2026-11-09: first audit.
