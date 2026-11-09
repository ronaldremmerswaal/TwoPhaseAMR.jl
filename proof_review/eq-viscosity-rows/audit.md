# Audit: eq:viscosity:rows — normal, shear and wall rows of the strain map $\mathsf E$

- Source: `docs/techref/chapters/09_viscosity.tex:221-240`, commit `03bad7a`; code `_variable_staggered_diffusion` (`variable_diffusion_operator.jl`), `_embedded_variable_wall_stresses`
- Date: 2026-11-09
- Overall status: ✅ verified entrywise against a brute-force reimplementation on 18 meshes (walled, periodic, cut, 12 random composite, no-slip and free-slip)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-viscosity-rows/scripts/run_all.jl`

## Statement and assumptions
Normal rows $(\mathsf Eu)_{c,k}=|c^{\rm o}|^{-1}\sum_{f\in\mathcal F_c\cap\mathcal F_k\cap\mathcal F^{\rm int}}o_{c,f}a_f|f|u_f$ for $\phi_c>0$ (boundary faces omitted: zero wall-normal velocity); shear rows $(\mathcal D_2u_1)_v+(\mathcal D_1u_2)_v$ at the corners $v$ with $|\omega_v|>0$; wall rows $(\mathsf Eu)_w=u_{f(w)}$ for each wall or covered unknown. Vertices: cell corners of the leaf cells on the finest lattice, periodic images identified.

## Steps
| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| normal rows | l. 225-228 | ✅ | independent assembly from the faces of each cell: entrywise $\le3\times10^{-15}\max|E|$ on all fixtures | `code_01_rows.jl` | hanging faces included (levels 1, 2) |
| shear rows (pair / wall / least squares) | l. 229-231 | ✅ | whole rows (both components) match the independent rule of eq. `shear`; per-kind analysis in `eq-viscosity-shear` | same | |
| wall rows: one entry 1 at the face; cut kind and covered kind | l. 232-234 | ✅ | independent list (face, weight$\times\mu_c$) equals the code's for cut meshes and random discs with/without periodicity: 272 wall rows (NoSlip) and 177 (FreeSlip, covered only) on the docs cut fixture | same | `FreeSlip` drops the cut kind only |
| vertex set and measures | l. 235-237 | ✅ | sum of vertex measures equals the open volume of the domain to $4\times10^{-16}$; vertex count 592–924 on the meshes | same | |
| "every row is exact for affine velocities" (end of eq. `shear` paragraph) | l. 257-258 | ✅ | normal rows exact; shear rows verified in `eq-viscosity-shear` (exception: rows whose support contains zero wall values, exact only for fields vanishing at the wall) | `../eq-viscosity-shear/scripts/code_01_shear_stencils.jl` | |
| "Only stationary `NoSlip()`/`FreeSlip()` are accepted; moving walls and `VelocityBoundaryConditions` are rejected" | l. 302-304 | ✅ | `ArgumentError` for `NoSlip(f)` and for `VelocityBoundaryConditions`; `NoSlip()` and `FreeSlip()` build | `../eq-viscosity-phase-midpoint/scripts/code_01_phase_midpoint.jl` | |

## Errors and consequences
None. Note: with a solid, rows of grid-aligned solid faces are absent (D2 of `eq-viscosity-cut`): the wall rows exist only for cut cells and covered unknowns.

## Sharpness
| Claim | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| equality with the independent assembly | sharp | $10^{-15}$ | `code_01_rows.jl` | none |
| boundary faces omitted from normal rows | exact but loses wall-normal momentum | see `eq-viscosity-momentum` | — | state in the rows paragraph that this is the zero wall-normal velocity (it is) |

## Open points
None.

## History
- 2026-11-09: first audit.
