# Audit: eq:viscosity:stiffness — integrated stiffness of the single-phase operator

- Source: `docs/techref/chapters/09_viscosity.tex:81-92` (formula), `178-200` (cut-cell variant), commit `03bad7a`; code `_diffusion_stiffness`
- Date: 2026-11-09
- Overall status: ✅ verified (assembly, symmetry, positive semidefiniteness, kernel, quadratic-form identity)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-viscosity-stiffness/scripts/run_all.jl`

## Statement and assumptions

$\viscstiff k=\operatorname{sym}\bigl(\sum_g\meas g\,d_g\,\conncoef g{}^\top\conncoef g{}+\sum_w\text{cond}_w\,e_{f(w)}e_{f(w)}^\top\bigr)$; a two-point connection contributes $\meas g(u_+-u_-)^2/d_g$ to $u^\top\viscstiff ku$. Positive measures and conductances.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| assembly from connections + independent stencils + wall list | l. 85-88 | ✅ | independent assembly equals the code's stiffness to $\le2.2\times10^{-16}$ relative on 6 meshes (periodic, walled, random, FreeSlip) | `../eq-viscosity-gradient/scripts/code_01_gradient_stiffness.jl` | |
| $u^\top\viscstiff ku=\sum_g\meas gd_g(\conngrad gu)^2+\sum_w\text{cond}_wu_w^2$ | l. 90-92 | ✅ | random $u$, relative defect $\le2.7\times10^{-15}$; two-point connections reproduce $\meas g(\Delta u)^2/d_g$ | same | |
| $\operatorname{sym}$ | l. 85 | ✅ (note) | a sum of outer products is already symmetric; the code symmetrises $(K+K^\top)/2$ only to remove round-off | `code_01_stiffness.jl` | cosmetic: could read "(symmetric by construction)" |
| symmetric, PSD, kernel | — | ✅ | $\viscstiff k=\viscstiff k^\top$ exactly; $\lambda_{\min}\ge-10^{-13}\lambda_{\max}$; kernel per component: periodic 1 (constants; columns sum to 0), walled no-slip/free slip 0, x-periodic no-slip 0, x-periodic free slip $(1,0)$ | `code_01_stiffness.jl` | |
| wall-free core has constants in its kernel | — | ✅ | removing the wall terms on the walled composite mesh leaves a PSD matrix with $\mathsf S_{\rm core}\mathbf 1=0$ | same | |
| cut-cell variant "symmetric up to the volumes" ($\dualvolmat\lapstag k=-\viscstiff k$) | l. 179-181 | ✅ | uniform mesh with a disc: $|\dualvolmat\lapstag k+\viscstiff k|\le9\times10^{-16}$, symmetric; $\lambda_{\min}/\lambda_{\max}=0.022$ | same | covered rows: isolated, $-1$ (see `eq-viscosity-cut`) |

## Errors and consequences
None. (Statements about the *Laplacian* rows are in `eq-viscosity-laplacian`: the band replaces the stiffness rows at level transitions, so the stiffness is not the operator there.)

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| PSD, symmetric | **sharp** | kernel = constants on periodic meshes; strictly positive with a wall | `code_01_stiffness.jl` | add the kernel list |
| conservation $\mathbf 1^\top\viscstiff k=0$ (periodic) | **sharp** | $10^{-12}$ | same | the chapter does not state it; it is what makes the core conservative |

## Open points
None.

## History
- 2026-11-09: first audit.
