# Audit: eq:viscosity:phase-midpoint — one-velocity two-phase implicit midpoint step

- Source: `docs/techref/chapters/09_viscosity.tex:368-393` (equation and prose), commit `03bad7a`; code `src/two_phase/stepping.jl` (`_diffusion_substep!`), `src/two_phase/diffusion/variable_diffusion_solver.jl`
- Date: 2026-11-09
- Overall status: ✅ verified (step equals the direct solve about the transported velocity; SPD; CG; special viscosities)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-viscosity-phase-midpoint/scripts/run_all.jl`

## Statement and assumptions
$(\mathsf M+\tfrac12\Delta t\,\mathsf E^\top\mathsf W\mathsf E)u^{[2]}=\mathsf Mu^{[1]}-\tfrac12\Delta t\,\mathsf E^\top\mathsf W\mathsf Eu^{[1]}$ with $u^{[1]}$ the transported velocity (after body acceleration), $\mathsf M=\operatorname{diag}(\bar\rho_fV_f)$ of the transported fractions (open volumes with a solid), $\mu$ refreshed from the same fractions; SPD matrix solved by CG with the factorised action and the rediscretised multigrid; both viscosities zero: $u^{[2]}=u^{[1]}$, no operator; one zero viscosity rejected; stopping $\max_f|r_f|/m_f\le\max(\epsilon\max_f|b_f|/m_f,\text{floor})$, at most 500 iterations.

## Steps
| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| step = direct solve about $u^{[1]}=$`workspace.convected` ($u^n$ irrelevant) | l. 372-380 | ✅ | `FlowSimulation` on the periodic composite drop, $u^n$ and $u^{[1]}$ random and different: substep vs `(M + dt/2 S) \ (M u1 - dt/2 S u1)`: $8.7\times10^{-13}$; changing `old_velocity` leaves the result unchanged to $10^{-9}$ | `code_01_phase_midpoint.jl` | the repository test "Two-phase diffusion uses the thesis split state" does the same on a uniform mesh |
| residual identity with $\mathsf S=\mathsf E^\top\mathsf W\mathsf E$; strong residual of the package solve | l. 372-375 | ✅ | six cases (walled, cut, periodic; $\Delta t=0.05,50$): strong residual $\le3\times10^{-13}$ | `../eq-viscosity-energy/scripts/code_01_midpoint_energy.jl` | |
| $\mathsf M=\operatorname{diag}(\bar\rho_fV_f)$ with the density of eq. `forces:density` | l. 375-379 | ✅ | `mixture_density` equals $\rho^{\rm l}\hat C+\rho^{\rm g}(1-\hat C)$ with the code's $\hat C$ to $10^{-12}$; `op.volumes == mesh.face_dual_volumes` | `code_01_phase_midpoint.jl` | the density function itself is audited in Ch. 8 |
| SPD, solved by CG | l. 381-384 | ✅ | `isposdef(M + dt/2 S)`; `typeof(krylov) = CgWorkspace` | same | |
| both zero: $u^{[2]}=u^{[1]}$; one zero rejected | l. 384-387 | ✅ | `FlowSimulation` with $(0,0)$ copies the transported velocity bitwise; $(0.1,0)$: `ArgumentError: phase dynamic viscosities must both be positive or both zero` (in `TwoPhase` and in `StaggeredDiffusionOperator`); single-phase $\nu\le0$ rejected; moving `NoSlip(f)` and `VelocityBoundaryConditions` rejected by the variable operator | same | |
| stopping criterion, 500 iterations | l. 388-392 | ✅ | solver stats: converged, `final_linf`$=3.3\times10^{-12}\le$ target $3.8\times10^{-12}$ (tolerance $10^{-12}$ relative to $\max|b_f|/m_f=3.8$); default cap in `_navier_stokes_cycle_limit` | same, `../eq-viscosity-single-midpoint/scripts/code_01_single_midpoint.jl` (single-phase analogue) | |

## Errors and consequences
None. (The identities derived from this step carry the residual sign correction of `eq:viscosity:energy`.)

## Sharpness
| Claim | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| step form | sharp (equality) | $10^{-12}$ | `code_01_phase_midpoint.jl` | none |
| thesis form (both halves about $u^{[1]}$) | consistent with the thesis | | | `dev:viscosity:time-level` already states the single-phase difference |

## Open points
None.

## History
- 2026-11-09: first audit.
