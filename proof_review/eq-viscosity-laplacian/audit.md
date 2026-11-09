# Audit: eq:viscosity:laplacian — single-phase Laplacian: conservative core and corrected band

- Source: `docs/techref/chapters/09_viscosity.tex:89-130`, `338-341`, commit `03bad7a`; code `_corrected_laplacian`, `_quadratic_laplacian_stencil`
- Date: 2026-11-09
- Overall status: ⚠️ all statements verified; the consequences of the non-symmetric band (consistency orders, energy, momentum) are not stated and are quantified here
- Rerun: `julia --project=@claude-symbolics proof_review/eq-viscosity-laplacian/scripts/run_all.jl`

## Statement and assumptions

$(\lapstag k u)_f=-V_f^{-1}(\viscstiff ku)_f$ for $f\notin\bandset k$, $\sum_{f'}\bandcoef f{f'}u_{f'}$ for $f\in\bandset k$. $\bandset k$: unknowns of the connections of non-regular dual faces without a wall connection, plus the unknowns within two graph rings reached without passing wall unknowns. A band row uses unknowns within three graph rings, is exact for quadratics ($\sum\bandcoef f{f'}p(x_{f'})=\Delta p(x_f)$), minimises $\sum\bandcoef f{f'}{}^2(|x_{f'}-x_f|^2/\ell_f^2+1/64)$, $\ell_f$ the largest coordinate offset; on a uniform mesh the operator is the five-point Laplacian; in the band the rows are consistent but not conservative and $\dualvolmat\lapstag k$ is not symmetric.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| five-point Laplacian on the uniform periodic mesh | l. 105-107 | ✅ | independent face lookup, $\le10^{-12}$ | `code_01_laplacian.jl` | |
| band set $\bandset k$ | l. 98-103 | ✅ | independent BFS from the non-regular connections excluding wall unknowns: the set of rows that differ from $-V^{-1}S$ equals it on 8 meshes (286, 640, 514, 1072, 304, 644, 324, 946 rows) | same | |
| support three rings; exact for quadratics | l. 101-103 | ✅ | all six monomials, defect $\le1.3\times10^{-14}$; support $\subseteq$ three rings (independent BFS) | same | |
| minimal weighted norm with $\ell_f$, $1/64$ | l. 103-104 | ✅ | KKT minimiser vs code: $\le5.6\times10^{-15}$ ($\times\ell_f^2$) | same | |
| $\dualvolmat\lapstag k$ not symmetric in the band | l. 107-108 | ✅ | 9 020 … 32 242 asymmetric entries per mesh; $\|A-A^\top\|/\|A\|=0.10$–$0.18$ for $A=-\dualvolmat\lapstag k$ | same, `code_02_...` | |
| "consistent" | l. 107-108 | ✅ with orders | truncation error for $u^*=\sin\sin$ on the walled composite mesh, $n=16,32,64$: core rows $7.7\times10^{-4},2.3\times10^{-4},6.1\times10^{-5}$ (order 1.7, 1.9); **band rows $2.1\times10^{-2},1.1\times10^{-2},5.9\times10^{-3}$ (order 0.85, 0.96: first order)**; wall-adjacent rows order 1.0 (for generic quadratics the error is $O(1)$: $\tfrac12u_{yy}$, see `eq-viscosity-walls`); solution error of $(I-\lapstag k)u=(I-\Delta)u^*$: $6.7\times10^{-3},1.3\times10^{-3},2.1\times10^{-4}$ (order 2.4, 2.6) | `code_02_consistency_and_energy.jl` | quadratic exactness gives $O(h)$ truncation, not $O(h^2)$ |
| core rows outside $\bandset k$ near walls | l. 98-103 | ⚠️ | core rows at unknowns with a wall connection are *not* quadratic exact (all inconsistent core rows have a wall connection; none elsewhere) | `code_01_laplacian.jl` | not claimed by the chapter |

## Errors and consequences

No error. Quantified effects of the non-symmetric band (all on the single-phase operator only; the two-phase operator is symmetric by construction):
1. **Spectrum:** real parts $\le0$ (max $-0.31$ walled; $0$ periodic) — the "stable spectrum" of the tests — **but** $\operatorname{sym}(-\dualvolmat\lapstag k)$ is indefinite: $\lambda_{\min}/\lambda_{\max}(\viscstiff k)=-3.5\times10^{-4},-2.1\times10^{-3}$ (walled, components 1, 2) and $-7.6\times10^{-3},-7.0\times10^{-3}$ (periodic composite).
2. **Energy of the Crank–Nicolson step** (the two-phase identity of `eq:viscosity:energy` does not apply): the V-norm energy factor $\|V^{1/2}(I-a\mathsf L)^{-1}(I+a\mathsf L)V^{-1/2}\|^2$ with $a=\tfrac12\Delta t\nu$ and diffusion number $\text{ND}=a\rho(\mathsf L)$: walled composite $1.000,1.001,1.009,1.014$ (ND $=0.1,1,10,100$; component 1) and $1.001,1.007,1.061,1.105$ (component 2); periodic composite $1.001,1.013,1.13,1.68$ ($\text{ND}=0.1,\ldots,100$) $\to2.16$ ($\text{ND}\to\infty$); component 2: $1.002,1.016,1.16,1.86\to2.71$. Growth appears only for $\text{ND}\gtrsim1$, i.e. $\nu\Delta t/h_{\rm fine}^2\gtrsim0.1$.
3. **Momentum:** $\mathbf 1^\top\dualvolmat\lapstag k=0$ on the uniform periodic mesh ($10^{-18}$); on the periodic composite mesh $|\mathbf 1^\top\mathsf V\mathsf Lu|/\mathbf 1^\top\mathsf V|\mathsf Lu|=3.2\times10^{-3},7.7\times10^{-3}$ for random $u$ but $10^{-16}$ for a smooth mode (quadratic exactness gives $\int\Delta u=0$ to the order of the band).

Replacement (add at the end of the paragraph before *Walls*, l. 107-108):

```latex
in the band the rows are consistent (truncation error $O(h)$, not $O(h^2)$) but not conservative, and $\dualvolmat\lapstag{k}$ is not
symmetric: its symmetric part is indefinite (smallest eigenvalue $-10^{-3}$ to $-10^{-2}$ of the largest stiffness eigenvalue on
the composite fixtures), so the midpoint step of \cref{sec:viscosity:time} can raise the mass-weighted energy by up to a factor
$1.1$ (walled) to $2.7$ (periodic) per step for $\visc\dt/(\dens h^2)\gtrsim1$, and the momentum is conserved
only to the accuracy of the band.
```
($h$ the fine cell width; the numbers are for the docs composite fixtures.)

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| quadratic exactness of the band | **sharp as stated** | all six monomials to $10^{-14}$; first order truncation (order 0.85–0.96) | `code_01_laplacian.jl`, `code_02_...` | cubic exactness would give $O(h^2)$ but needs a larger support |
| band excludes unknowns with a wall connection | by construction (the band stencil would reach through the wall) | the rows with a wall connection keep the core row; they are not quadratic exact (wall treatment of first order); no inconsistent core row without a wall connection exists in the 8 meshes | `code_01_laplacian.jl` | quadratic-exact wall-adjacent rows would need a wall-aware band (not claimed) |
| "stable spectrum" | **spectrum true, energy norm not contractive** | factors above | `code_02_...` | restrict to eigenvalues; state the transient growth |

## Open points
- Whether the transient growth ($\le2.7$ per step, stiff limit) matters in flows with $\nu\Delta t/h^2\gtrsim1$ at a periodic level transition was not tested in a full step.

## History
- 2026-11-09: first audit.
