# Audit: eq:viscosity:energy — energy balance of the two-phase implicit midpoint step

- Source: `docs/techref/chapters/09_viscosity.tex:394-407` (statement), `372-379` (the step), commit `03bad7a` (branch `docs/techref-forces-viscosity`; only the audit files are uncommitted)
- Date: 2026-11-09
- Overall status: ❌ the identity is exact but the printed sign of the residual term is wrong for the residual the chapter defines (`\resvec` = solver residual $\rhsvec-\mathsf A\velo^{[2]}$); one-line fix below
- Rerun: `julia --project=@claude-symbolics proof_review/eq-viscosity-energy/scripts/run_all.jl`

## Statement and assumptions

With $\mathsf M=\operatorname{diag}(\bar\rho_f V_f)$ (positive), $\mathsf S=\mathsf E^\top\mathsf W\mathsf E$ (symmetric PSD), $\mathsf A=\mathsf M+\tfrac12\Delta t\,\mathsf S$, the step is $\mathsf A u^{[2]}=(\mathsf M-\tfrac12\Delta t\,\mathsf S)u^{[1]}$ (eq. `phase-midpoint`). Claim (l. 394-407): for every $\Delta t>0$ the mass-weighted kinetic energy $\tfrac12 u^\top\mathsf M u$ does not increase, up to the residual, via
$$\tfrac12 u^{[2]\top}\mathsf M u^{[2]}-\tfrac12 u^{[1]\top}\mathsf M u^{[1]}=-\tfrac14\Delta t\,w^\top\mathsf S w+\tfrac12 w^\top\mathsf r,\qquad w=u^{[1]}+u^{[2]}.$$
Implicit assumptions that the source should state: $\mathsf M$ diagonal and positive, $\mathsf S$ symmetric PSD (eq. `quadratic`), and **the sign convention of $\mathsf r$**.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| 1. $\mathsf r=\mathsf M(u^{[2]}-u^{[1]})+\tfrac12\Delta t\,\mathsf S w$ (defect of eq. `phase-midpoint`) | l. 394-395 | ✅ | random SPD/PSD instances (200), relative defect $3\times10^{-14}$ | `step01_energy_algebra.jl` | requires $\mathsf r:=\mathsf A u^{[2]}-\text{rhs}$ |
| 2. multiply by $w$; $w^\top\mathsf M(u^{[2]}-u^{[1]})=u^{[2]\top}\mathsf Mu^{[2]}-u^{[1]\top}\mathsf Mu^{[1]}$ ($\mathsf M$ symmetric) | l. 395-402 | ✅ | same | `step01_energy_algebra.jl` | |
| 3. sign of the $\mathsf r$ term: $+\tfrac12w^\top\mathsf r$ | l. 401 | ❌ | package solve with a loose tolerance (walled, cut, periodic; $\Delta t=50$): identity defect $\le1.8\times10^{-16}$ with $\mathsf r=\mathsf Au^{[2]}-\text{rhs}$, defect $2\times10^{-5}\ldots7\times10^{-5}$ (relative) with the opposite sign; the solver's own residual is $\text{rhs}-\mathsf Au^{[2]}=-\mathsf r$ (`_true_residual!`, `elliptic_solver.jl:279-283`; agreement $10^{-15}$) | `code_01_midpoint_energy.jl` | `\resvec` is defined in `techref-notation.sty:825` as "Residual vector of a linear solve" and used with that meaning in the stopping criterion (l. 385). With it the printed sign is wrong |
| 4. dissipation $\tfrac14\Delta t\,w^\top\mathsf Sw\ge0$ | l. 402-405 | ✅ | $\mathsf S=\mathsf E^\top\mathsf W\mathsf E$ equals the code's stiffness (independent strain map, 1e-12); $\lambda_{\min}(\mathsf S)\ge-7\times10^{-17}\lambda_{\max}$ | `code_01_midpoint_energy.jl`, `eq-viscosity-quadratic` | |
| 5. "for every $\Delta t$" | l. 403 | ✅ | eigenmode factor $\bigl((1-\tfrac12\Delta t\lambda)/(1+\tfrac12\Delta t\lambda)\bigr)^2\le1$ verified on 8 modes at $\Delta t=0.05$ and $5000$ to $7\times10^{-14}$; 200 random vectors: $\max E_2/E_1=0.9995$ | `sharp_01_energy_rate.jl` | |
| 6. exact-solve energy change equals the dissipation | l. 403-405 | ✅ | six (fixture, $\Delta t$) cases, $|\Delta E+\text{diss}|/\text{diss}\le8\times10^{-13}$ | `code_01_midpoint_energy.jl` | |

## Errors and consequences

**F1 (sign).** With the chapter's residual the printed right-hand side is $+\tfrac12w^\top\mathsf r$; the correct one is $-\tfrac12w^\top\mathsf r$. Equivalently, the printed identity is right for $\mathsf r:=\mathsf Au^{[2]}-\text{rhs}$ (the defect), the negative of the solver residual. The same sign enters eq. `momentum` (see `eq-viscosity-momentum`) and, through it, the last sentence of eq. `two-velocity`. Nothing else changes: $\Delta E\le0$ for an exact solve, the dissipation term, the checks (`midpoint_check` evaluates $\mathsf r=0$).

Replacement (keeps `\resvec` as the solver's residual):

```latex
\emph{Identities.} Let $\resvec=\rhsvec-\roundpar{\massmat+\tfrac12\dt\,\strainmat^{\transpose}\stressw\strainmat}\velo^{\stage{2}}$
be the residual of~\cref{eq:viscosity:phase-midpoint}, with the right-hand side
$\rhsvec=\roundpar{\massmat-\tfrac12\dt\,\strainmat^{\transpose}\stressw\strainmat}\velo^{\stage{1}}$.
Multiplying it by $\velo^{\stage{1}}+\velo^{\stage{2}}$ gives the energy balance
\begin{equation}\label{eq:viscosity:energy}
  \tfrac12(\velo^{\stage{2}})^{\transpose}\massmat\velo^{\stage{2}}
  -\tfrac12(\velo^{\stage{1}})^{\transpose}\massmat\velo^{\stage{1}}
  =-\tfrac14\dt\,(\velo^{\stage{1}}+\velo^{\stage{2}})^{\transpose}\strainmat^{\transpose}\stressw\strainmat\,(\velo^{\stage{1}}+\velo^{\stage{2}})
  -\tfrac12(\velo^{\stage{1}}+\velo^{\stage{2}})^{\transpose}\resvec :
\end{equation}
```
(and, optionally, after "the discrete counterpart of eq. `dissipation`": "the balance is an equality: on an eigenvector of $\massmat^{-1}\strainmat^{\transpose}\stressw\strainmat$ with eigenvalue $\lambda$ the energy is multiplied by $((1-\tfrac12\dt\lambda)/(1+\tfrac12\dt\lambda))^2$, so the step is neutrally stable on the kernel and does not damp stiff modes as $\dt\to\infty$ (no L-stability)").

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| identity (equality) | **sharp** (an identity) | holds to $10^{-16}$ with $\mathsf r\neq0$ (loose solves), so no tighter statement exists | `code_01_midpoint_energy.jl` | none |
| $\Delta E\le0$ for every $\Delta t$ | **sharp** | equality on the kernel of $\mathsf S$ (the two constants of the periodic fixture: factor $1\pm10^{-12}$); factor of the stiffest mode at $\Delta t=5000$ is $0.99685$ (CN is not L-stable) | `sharp_01_energy_rate.jl` | none |
| $\mathsf M$ positive diagonal, $\mathsf S$ PSD | used in step 4/5 only | with a non-symmetric or indefinite $\mathsf S$ (single-phase band, see `eq-viscosity-laplacian`) the identity has $\operatorname{sym}$ and the sign is lost (energy factor up to 2.7) | `eq-viscosity-laplacian` `code_02` | state that the two-phase operator is symmetric by construction ($\mathsf E^\top\mathsf W\mathsf E$) |
| "up to the residual" | sharp | the residual term is first order in the solver defect: $|\tfrac12w^\top\mathsf r|$ with $\|\mathsf r/\mathsf M\|_\infty\le\epsilon_{\rm tol}\|\text{rhs}/\mathsf M\|_\infty$ | — | optionally quote the bound |

## Open points
- Scope: the continuous statement eq. `dissipation` (boundary terms vanish for stationary no-slip/free-slip/periodic, $\partial_t\rho=0$ in the viscous part) was read, not scripted; it is the standard integration by parts.
- Literature (for `literature-scout`): Crank–Nicolson/midpoint L-stability remarks; not needed for correctness.

## History
- 2026-11-09: first audit (algebra, package solves on walled/cut/periodic fixtures, eigenmode sharpness). Outcome: F1 sign.
