# Audit: eq:viscosity:shear — vertex derivative: pair, wall and least-squares rows

- Source: `docs/techref/chapters/09_viscosity.tex:241-266`, commit `03bad7a`; code `_vertex_derivative`, `_nearby_diffusion_faces!`
- Date: 2026-11-09
- Overall status: ⚠️ selection rule, support, minimality and exactness verified; two wording precisions (affine exactness near walls, asymmetric pairs are first order) and one unreachable branch
- Rerun: `julia --project=@claude-symbolics proof_review/eq-viscosity-shear/scripts/run_all.jl`

## Statement and assumptions
$(\mathcal D_tu_k)_v$ is $0$ at a free-slip wall normal to $t$; otherwise the two-point difference $(u_+-u_-)/(x_+-x_-)_t$ of the nearest unknowns of component $k$ on the line through $v$ along $t$, one on each side within $3\,c_{\max}(v)$ (boundary faces value 0; at a no-slip wall normal to $t$ the wall point with value 0 replaces the missing side); else the least-squares stencil on the at most 16 nearest unknowns within $3\,c_{\max}$, exact for quadratics, minimising $\sum a_j^2(1+|x_j-v|^2/c_{\max}^2)^2$, or exact for affine $p$ if the quadratic system is rank deficient (smallest singular value $<10^{-10}$ of the largest). "Every row is exact for affine velocities; the two-point rows of a symmetric pair also for quadratics."

## Steps
| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| selection: free-slip zero / pair / wall pair / least squares | l. 241-257 | ✅ | whole strain matrices of the replica equal the code's to $\le3\times10^{-15}$ (12 random composite meshes + 3 docs fixtures, NoSlip and FreeSlip), see `eq-viscosity-rows` | `../eq-viscosity-rows/scripts/code_01_rows.jl` | |
| kinds present (8 random meshes, levels 1–2): symmetric pair 6 938, wall pair 316, asymmetric pair 350, least squares (quadratic) 316, **affine fallback 0** | l. 253-256 | ✅ / ⚠️ | classification by the replica | `code_01_shear_stencils.jl` | the rank-deficient fallback never occurred (also on 8 anisotropic meshes with aspect ratios 1:1–1:4); unreachable in the tests, **unverified** |
| support $\le16$ interior unknowns within $3c_{\max}$ (max seen 16, radius $2.69\,c_{\max}$) | l. 249-251 | ✅ | statistics of the LSQ supports | same | |
| minimal weighted norm; exact for quadratics | l. 251-256 | ✅ | KKT minimum-norm solution (independent algebra) vs the row: $2.7\times10^{-15}$ ($\times c_{\max}$); null-space perturbations increase the norm; quadratic defect $6\times10^{-16}$ | same | |
| "every row exact for affine" | l. 257 | ⚠️ | exact ($3\times10^{-15}$) for all rows whose support contains no wall value (7 302 of 7 920 rows); the 618 rows whose support contains wall values (zero, dropped) or is the wall pair are exact only for fields vanishing at the wall | same | the docs affine check excludes cells within 4 coarse cells of the edges for this reason |
| symmetric pair exact for quadratics | l. 257-258 | ✅ | $6\times10^{-16}$; **asymmetric pairs (stretching at a level transition) are exact only for affine**: quadratic defect $|\sum_ja_jx_{t,j}^2|\,c_{\max}=0.25$ (0 if exact), truncation order 1.03 vs 1.99 (symmetric) and 2.0 (LSQ) | same | |

## Errors and consequences
**F1 (wording).** "Every row is exact for affine velocities" holds for rows without wall values in the support; the wall one-sided difference and the rows near walls are exact only for fields that vanish at the wall (no-slip). **F2 (wording).** Asymmetric pairs occur at every level transition (350 of 8 k rows on the test meshes) and are first order; the text should say so, because it affects the order claim of the quadratic form (which nevertheless converges with order 2, `eq-viscosity-quadratic`). Replacement for the last sentence of the paragraph before eq. `weights`:

```latex
Every row that does not contain a wall value is exact for affine velocities; the two-point rows of a
symmetric pair and the least-squares rows are also exact for quadratics, the two-point rows of an asymmetric pair
(one side coarse, one fine: stretching at a level transition) are first-order accurate.
```

## Sharpness
| Claim | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| affine exactness | **sharp** | quadratic fails for asymmetric pairs | `code_01_shear_stencils.jl` | none |
| row truncation | symmetric pair order 2 (1.99), LSQ 2 (2.0), asymmetric pair **1 (1.03)** | refinement $n=16,32,64$ (periodic composite): max errors per kind $0.0397/0.0146/0.0312\to0.0100/0.0036/0.0153\to0.0025/0.0009/0.0075$ | same | replace the asymmetric pair by the LSQ stencil (second order) — not claimed |
| radius $3c_{\max}$, 16 points | sufficient in all tests | max radius ratio 2.69, no failure | same | none |

## Open points
- The affine fallback branch (smallest singular value $<10^{-10}$) is not exercised; a unit test with collinear data would be needed.

## History
- 2026-11-09: first audit.
