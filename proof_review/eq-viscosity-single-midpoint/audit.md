# Audit: eq:viscosity:single-midpoint — single-phase Crank–Nicolson step (explicit half at $u^n$, wall data at the midpoint)

- Source: `docs/techref/chapters/09_viscosity.tex:350-382` (equation and prose), `383-392` (stopping criterion), `404-407` (last sentence of the identities), commit `03bad7a`; code `src/navier_stokes/implicit_diffusion.jl` (`implicit_midpoint_diffusion!`, `_add_wall_source!`), `src/single_phase/stepping.jl`
- Date: 2026-11-09
- Overall status: ⚠️ equation, time level, solver choice and stopping criterion verified; the closing sentence "has no such identity" is too strong
- Rerun: `julia --project=@claude-symbolics proof_review/eq-viscosity-single-midpoint/scripts/run_all.jl`

## Statement and assumptions
$(\mathsf I-\tfrac12\Delta t\nu\mathsf L)u^{[2]}=u^{[1]}+\tfrac12\Delta t\nu\mathsf Lu^n+\Delta t\nu\,\mathsf V^{-1}\sum_w\text{cond}_w u^{\rm wall}_w(t^n+\tfrac12\Delta t)$ (a CN step whose explicit half is evaluated at the old velocity); solved with CG if $\mathsf V\mathsf L$ is symmetric for every shift (equal volumes and symmetric $\mathsf L$), BiCGSTAB otherwise; stopping $\max_f|r_f|/m_f\le\max(\epsilon\max_f|b_f|/m_f,\text{floor})$, at most 100 iterations.

## Steps
| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| the equation (wall source $=$ half via `diffusion!` at the midpoint + half via `_add_wall_source!`) | l. 355-361 | ✅ | independent sparse direct solve with $u^n\ne u^{[1]}$ and a time-dependent moving wall: package vs independent $7.8\times10^{-16}$ | `code_01_single_midpoint.jl` | |
| wall data at $t^n+\tfrac12\Delta t$ | l. 361-362 | ✅ | the alternatives $t^n$, $t^n+\Delta t$ change the result by $2.7\times10^{-2}$, $1.2\times10^{-2}$ | same | |
| CG iff $\mathsf V\mathsf L$ symmetric for every shift | l. 363-367 | ✅ | uniform periodic and uniform walled: CG; composite walled/periodic: BiCGSTAB | same | uniform walled meshes qualify because all dual volumes are equal; the cut operator (unequal volumes) takes BiCGSTAB |
| stopping criterion and iteration cap | l. 383-392 | ✅ | strong residual $\max|r_f|/V_f\le\text{tol}\max|b_f|/V_f$: $8.6\times10^{-4},8.3\times10^{-7},6.4\times10^{-11}$ for tol $10^{-3},10^{-6},10^{-10}$; `max_cycles=1` with tol $10^{-13}$ throws `ErrorException`; defaults 100 (single), 500 (two phase) in `_navier_stokes_cycle_limit` | same | |
| "has no such identity: its explicit half acts on $u^n$, and its band rows are not conservative" | l. 404-407 | ⚠️ | **momentum** $\sum V(u^{[2]}-u^{[1]})$ is conserved on the uniform periodic mesh to $4\times10^{-16}$ although $u^n\ne u^{[1]}$ (the rate $\mathsf Lu$ conserves for any argument); on the periodic composite mesh $0.9\%$ / $0.35\%$ relative (band). **Energy**: with $u^n=u^{[1]}$ on the uniform periodic mesh the identity $\Delta E=-\tfrac14\Delta t\nu\,w^\top\mathsf Sw$ holds to $1.5\times10^{-16}$; with $u^n\ne u^{[1]}$ it fails by $3.2\%$ of $|\Delta E|+\text{diss}$ | same | see replacement |

## Errors and consequences
**F1 (precision).** The sentence must be split: the momentum identity of the single-phase step holds exactly on uniform periodic meshes (explicit half irrelevant); the energy identity needs $u^n=u^{[1]}$ and a symmetric $\mathsf L$ (uniform mesh), and fails with the band, where $\operatorname{sym}(-\dualvolmat\mathsf L)$ is indefinite and the energy factor of one step reaches $1.1$ (walled) to $2.7$ (periodic) for $\nu\Delta t/h^2\gtrsim1$ (`eq-viscosity-laplacian`).

Replacement for the last sentence of the *Identities* paragraph:

```latex
The single-phase step~\cref{eq:viscosity:single-midpoint} conserves the momentum on a uniform periodic mesh for any
explicit argument ($\mathbf 1^{\transpose}\dualvolmat\lapstag{k}=0$ there), but has no energy balance
of the form~\cref{eq:viscosity:energy}: its explicit half acts on $\velo^{\timelevel{n}}$ (the balance holds only for
$\velo^{\timelevel{n}}=\velo^{\stage{1}}$ and a symmetric $\lapstag{k}$), and the band rows are neither conservative
nor symmetric (energy factor up to $1.1$ to $2.7$ per step for $\visc\dt/(\dens h^2)\gtrsim1$ on the composite fixtures).
```

## Sharpness
| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| step equation | sharp (identity) | $10^{-16}$ | `code_01_single_midpoint.jl` | none |
| midpoint wall data | **needed**: other time levels change the answer by $1$–$3\%$ of the field at $\Delta t=0.2$ | | same | none |
| explicit half at $u^n$ (deviation `dev:viscosity:time-level`) | first-order consistent; with $u^n\ne u^{[1]}$ the energy balance is lost | | same | thesis form (both halves about $u^{[1]}$) restores the energy identity on symmetric meshes |

## Open points
- Impact of the explicit half at $u^n$ on temporal order (CN is second order only for $u^n=u^{[1]}$): not tested in a full step.

## History
- 2026-11-09: first audit.
