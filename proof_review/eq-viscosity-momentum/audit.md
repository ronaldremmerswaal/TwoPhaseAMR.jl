# Audit: eq:viscosity:momentum — momentum conservation of the two-phase midpoint step

- Source: `docs/techref/chapters/09_viscosity.tex:408-418`, commit `03bad7a`
- Date: 2026-11-09
- Overall status: ❌ sign of the residual (same as `eq:viscosity:energy`); hypothesis "periodic mesh without solid" verified and sharpened (free-slip walls conserve tangential momentum, solids and no-slip walls do not)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-viscosity-momentum/scripts/run_all.jl`

## Statement and assumptions

On a periodic mesh without solid, $\mathsf S\,\mathbf 1^{[k]}=0$ for the indicator of the faces of axis $k$; by symmetry of $\mathsf S$, $\mathbf 1^{[k]\top}\mathsf S=0$, hence
$\sum_{f\in\mathcal F_k}\bar\rho_fV_f(u^{[2]}_f-u^{[1]}_f)=\mathbf 1^{[k]\top}\mathsf r$ (printed). Assumptions: $\mathsf M$ diagonal (so $\mathbf 1^\top\mathsf M(u^{[2]}-u^{[1]})$ is the momentum change), $\mathsf S$ symmetric, the sign convention of $\mathsf r$ (see energy audit).

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| 1. constant component has no strain: normal rows $\sum_fo_{c,f}a_f|f|/|c^{\rm o}|=0$, shear rows $=0$ (two-point and least-squares rows are exact for constants) | l. 410 | ✅ | $\max|\mathsf S\mathbf 1^{[k]}|/\max|\mathsf S|\le3.3\times10^{-16}$ on 10 random periodic 2:1 composite meshes (levels 1, 2; random interfaces), both components, loose and tight solves (40 cases) | `code_01_momentum.jl` | |
| 2. $\mathbf 1^\top\mathsf S=0$ "by symmetry" | l. 410-411 | ✅ | $\mathsf S$ exactly symmetric (asymmetry 0), see `eq-viscosity-stiffness` | `code_01_momentum.jl` | |
| 3. $\mathbf 1^\top\mathsf M(u^{[2]}-u^{[1]})=\mathbf 1^\top\mathsf r$ | l. 411-413 | ❌ sign | algebra: $\mathbf 1^\top\mathsf M(u^{[2]}-u^{[1]})=\mathbf 1^\top[\,\mathsf Au^{[2]}-\text{rhs}\,]$; the solver residual is $\text{rhs}-\mathsf Au^{[2]}$, so the printed right-hand side must be $-\mathbf 1^\top\mathsf r$ | `../eq-viscosity-energy/scripts/step01_energy_algebra.jl`, `code_01_momentum.jl` (identity holds to $10^{-20}$ relative with $\mathsf r=\mathsf Au^{[2]}-\text{rhs}$) | |
| 4. "Walls exchange momentum through the wall rows and the one-sided shear differences" | l. 413-414 | ⚠️ incomplete | see sharpness: the normal rows also exchange it (boundary faces omitted) | `code_01_momentum.jl` | |

## Errors and consequences

**F1 (sign)** as in `eq-viscosity-energy`; replace `=(\faceone{k})^{\transpose}\resvec` by `=-(\faceone{k})^{\transpose}\resvec`.
**F2 (wall sentence).** With $x$ periodic and walls in $y$: no-slip $\|\mathsf S\mathbf 1^{[1]}\|/\|\mathsf S\|=1.7\times10^{-2}$, free slip $1.4\times10^{-16}$ (tangential momentum conserved), and for $\mathbf 1^{[2]}$ (normal component) $7.1\times10^{-3}$ for both wall types: the normal rows lose the omitted boundary faces. In the fully walled box $1.6\times10^{-2}$ (no-slip) and $6.5\times10^{-3}$ (free slip) for both components. With a solid (periodic + disc) $0.92$ and $0.15$.

Replacement for the sentence after the identity:

```latex
the momentum is conserved up to the residual. A solid exchanges momentum through its wall rows, a no-slip wall
through the wall-adjacent shear differences, and any wall through the normal rows, which omit the
boundary faces (zero wall-normal velocity); a free-slip wall conserves the momentum of the tangential component on a mesh
that is periodic along it.
```

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| periodic, no solid $\Rightarrow$ exact conservation | **sharp** (equality, round-off $10^{-16}$) | counterexamples: solid ($S\mathbf1\ne0$, 0.92), no-slip walls (0.017), box (0.016) | `code_01_momentum.jl` | add free-slip tangential case (relaxation of the hypothesis, verified: $1.4\times10^{-16}$) |
| symmetric $\mathsf S$ | needed | the single-phase operator is NOT symmetric at level transitions and then $\mathbf 1^\top\mathsf V\mathsf L\neq0$ (`eq-viscosity-laplacian` K4) | — | the statement is for the two-phase operator only |

## Open points
None.

## History
- 2026-11-09: first audit.
