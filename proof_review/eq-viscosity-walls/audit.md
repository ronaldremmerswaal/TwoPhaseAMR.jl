# Audit: eq:viscosity:walls — wall conductances, wall values and wall types (single phase, no solid)

- Source: `docs/techref/chapters/09_viscosity.tex:110-127`, commit `03bad7a` (after #75); code `_wall_diffusion_connections`, `_include_wall_connection`, `_wall_value`
- Date: 2026-11-09
- Overall status: ✅ verified (conductances, wall points, value rules, wall types; affine profiles exact); remark on the order of the wall-adjacent rows
- Rerun: `julia --project=@claude-symbolics proof_review/eq-viscosity-walls/scripts/run_all.jl`

## Statement and assumptions

For a boundary face $b$ of the cell $c$ and the unknowns $f$ of component $k$ on the sides of $c$ (side fraction $\varpi_{c,f}$): tangential ($\text{axis}(b)\ne k$, every $f$) $\text{cond}=(\varpi\,|b|/2)/|(x_f-x_b)_{\text{axis}(b)}|$; normal ($\text{axis}(b)=k$, the $f$ opposite $b$) $\text{cond}=\varpi\,|b|/h_k(c)$. Wall value at the point where the wall meets the line of $f$: `NoSlip(u)` the value of `u` on tangential, 0 on normal; `FreeSlip` keeps the normal ones only; `Inflow` its function on both; `Outflow`, `ZeroGradientVelocity` none.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| tangential and normal conductances incl. hanging faces ($\varpi=|f|/h_t(c)$) | l. 115-120 | ✅ | independent list from the faces of each boundary cell equals the code's wall connections (face, conductance $10^{-13}$, tangential flag, wall point) on the docs walled composite mesh and 4 random composite meshes (levels 1, 2; walled, x-periodic, y-periodic): 20/20 | `code_01_walls.jl` | |
| wall point $=(x_b,\ y_f)$ / $(x_f,\ y_b)$ | l. 122-123 | ✅ | same comparison includes the point | same | |
| `NoSlip(u)`: $u$ at the point (tangential), 0 (normal); `FreeSlip`; `Inflow`; `Outflow`/`ZeroGradientVelocity` | l. 121-127 | ✅ | exact affine profile $u_1=U_0+(U_1-U_0)y/H$ with moving walls (bottom, top different): $|\text{rate}|\le2.1\times10^{-14}$ on uniform and composite channels; the same field with stationary walls has $|\text{rate}|>10^{-3}$; free slip $u_1=$ const: $\le10^{-11}$; uniform flow with Inflow left, Outflow/ZeroGradient right, free-slip top/bottom: $1.3\times10^{-14}$ | same | |
| rate sampled at the wall data time | — | ✅ | `diffusion!(...; time)` with time-dependent lid (see `eq-viscosity-single-operator`) | `../eq-viscosity-single-operator/scripts/code_01_single_operator.jl` | |
| wall-adjacent rows not quadratic exact | — | ⚠️ not claimed | $u_1=y(H-y)$, $\Delta u=-2$: error $0.5$ at the wall-adjacent row for every $h$ (interior rows $0$): error $=\tfrac12|u_{yy}|$ | same | the flux over the half-width wall layer uses the gradient at $y=h/4$, a first-order wall treatment; with the $\sin\sin$ test function (small $u_{yy}$ at the wall) the observed order is 1 |

## Errors and consequences
None. Remark: the code additionally skips a wall connection of zero distance (`distance > 0 || continue`) and one of zero measure (`measure > 0`), the latter only with a solid.

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| conductances (identity with the code) | **sharp** | equality to $10^{-13}$ | `code_01_walls.jl` | none |
| affine exactness at walls (moving data) | **sharp** | rate 0 to $10^{-14}$; quadratic fails by $\tfrac12u_{yy}$ | same | wall-adjacent rows are first-order consistent; second order would need the quadratic wall stencil $(u_+-u)/h-(u-u_w)/(h/2)$ over the dual width $3h/4$ instead of $h$ |

## Open points
None.

## History
- 2026-11-09: first audit.
