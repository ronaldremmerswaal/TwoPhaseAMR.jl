# Audit: eq:viscosity:gradient — derivative across a connection (two-point and widened)

- Source: `docs/techref/chapters/09_viscosity.tex:67-92`, commit `03bad7a`; code `_gradient_stencil`
- Date: 2026-11-09
- Overall status: ✅ verified (selection rule, minimal-norm stencil, affine exactness); remarks on accuracy and orientation
- Rerun: `julia --project=@claude-symbolics proof_review/eq-viscosity-gradient/scripts/run_all.jl`

## Statement and assumptions

$(\conngrad g u)=(u_{\dplus g}-u_{\dminus g})/d_g$ if $|(x_{\dplus g}-x_{\dminus g})_s|\le256\epsilon L_s$ ($s\ne t$), otherwise $\sum_f\conncoef gf u_f$ over the unknowns within two graph rings of the pair, with $\sum_f\conncoef gf\,p(x_f)=\partial_tp$ for $p\in\{1,x,y\}$ and minimal $\sum_f\conncoef gf^2\bigl(|x_f-x_{\dminus g}|^2+\tfrac1{16}\max(d_g,|\Delta_s|)^2\bigr)$.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| two-point exactly when the tangential offset is below $256\epsilon L_s$ | l. 69-75 | ✅ | independent rule on 6 meshes (2 399–3 806 two-point and 62–358 widened connections per mesh), same index sets as the code (5 217 comparisons) | `code_01_gradient_stiffness.jl` | |
| support: two graph rings of the pair | l. 76-79 | ✅ | independent BFS on the connection graph | same | |
| minimal-norm solution (KKT: $a=D^{-1}C^\top(CD^{-1}C^\top)^{-1}e_t$) | l. 78-80 | ✅ | code stencil vs KKT: $\le10^{-10}/d$ on every connection of walled and periodic composite meshes | same | the chapter's weight is $|x_f-x_{\dminus g}|_2^2+\tfrac1{16}\max(\cdot)^2$ = code's $1/\text{weights}$ |
| exact for affine $p$ | l. 77 | ✅ | defect $\le8.3\times10^{-16}$ (all meshes) | same | at a periodic seam the *two-point* stencil runs from `minus` to `plus` against $+t$ (orientation not consistent there): it is the derivative up to a sign, immaterial because only squares enter the stiffness; the widened stencil is solved for $+\partial_t$ |
| accuracy not claimed beyond affine | — | ✅ | quadratic defect $\sum a(x_t)^2=0.4\ldots1.1$ (units of length, i.e. $O(h)$); derivative error at the pair midpoint: widened $0.168,0.060,0.026$ (orders 1.49, 1.2), two-point $0.040,0.010,0.0025$ (order 2) | `sharp_01_widened_order.jl` | |

## Errors and consequences
None. Remark R1: the widened derivative is first-order accurate (affine exact only); the chapter's wording is correct. Remark R2: orientation sign of the two-point stencil at periodic seams.

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| affine exactness | **sharp** | quadratic defect nonzero ($O(h)$) | `code_01_gradient_stiffness.jl` | none |
| widened derivative order | **likely sharp: first order** | observed 1.5 → 1.2 → (asymptotically 1) | `sharp_01_widened_order.jl` | quadratic exactness would require more constraints (6 rows; support of two rings has $\ge10$ unknowns), but the band already supplies quadratic exact rows where it matters |
| two-ring support | sufficient | at least three non-collinear support points are needed for the $1,x,y$ constraints; the code throws "degenerate widened AMR diffusion stencil" otherwise (never observed) | — | none |

## Open points
None.

## History
- 2026-11-09: first audit.
