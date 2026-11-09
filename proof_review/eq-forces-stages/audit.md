# Audit: eq:forces:stages — stage order of the forces in the split step, body acceleration, time-step bounds (with eq:forces:density, eq:forces:coefficient, eq:forces:body)

- Source: `docs/techref/chapters/08_forces.tex:82-136` (density 86, coefficient 99), `495-519` (body 499), `575-624` (stages 578); `eq:vof:dt-forces` (`05_vof_transport.tex:425-428`); thesis (4.1), (7.4), Remark 7.1; commit `03bad7a`
- Code audited (normative): `src/single_phase/stepping.jl` (`step!`), `src/two_phase/stepping.jl` (`_refresh_pressure_substep!`, `_phase_force_substep!`, `_shared_velocity_timestep_bound`), `src/two_phase/two_velocity/stepping.jl` (`step!`, `_phase_force_substep!`, `_two_velocity_timestep_bound`), `src/navier_stokes/stages.jl` (`_add_body_force!`), `src/two_phase/forces.jl` (`mixture_density`, `PoissonOperator`), `src/two_phase/projection/common.jl` (`_projection_pressure_operator`), `src/navier_stokes/system.jl` (single phase)
- Date: 2026-11-09
- Overall status: ✅ verified (five labels); the order of the stages is verified by code inspection and by a hydrostatic step with a very large viscosity; the dt bounds by `stable_timestep` and rejection
- Rerun: `julia --project=@claude-symbolics proof_review/eq-forces-stages/scripts/run_all.jl`

## Statement and assumptions

**`eq:forces:density`.** $\bar\rho_f=\rho^{\rm l}\hat C_f+\rho^{\rm g}(1-\hat C_f)$, $\hat C_f=\operatorname{clamp}((\mathcal I C)_f,0,1)$; weights cell widths (no solid) or open strip areas (solid), $\hat C_f=0$ where both vanish.
**`eq:forces:coefficient`.** $\gamma_f=1/\bar\rho_f$ (no solid), $a_f^2/\bar\rho_f$ (solid); $\gamma^{\rm tv}_f=1/\bar\rho_f$ resp. $a_f/\bar\rho_f$; single phase $1/\rho$ ($\times a^2$); rebuilt once per step before the force stage, metric kept.
**`eq:forces:body`.** $v^{(1)}_f\leftarrow v^{(1)}_f+\delta t\,b(x_f,t^n+\tfrac12\delta t)\cdot e_{k_f}$ at the face centres of the interior faces, before the boundary values at $t^n+\tfrac12\delta t$ and the viscous step; both phases in the two-velocity model (not at absent values).
**`eq:forces:stages`.** BC($t^n$) → transport → body force → BC($t^n+\tfrac12\delta t$) → viscous step → BC($t^{n+1}$) → refresh $\gamma$ → $+\delta t\mathsf R\mathcal J^\sigma+\delta t\mathsf R\mathcal J^g$ (surface tension then gravity) → projection; same for one and two velocities; $\sigma=0$/$g=0$ stages skipped (two-velocity: the pressure map is not even refreshed). With $v^n=0$, $v^{(3)}=\delta t\mathsf R\mathcal J^g$.
**Bounds.** $\delta t\le\min\{\delta t_{\rm WY},\delta t_{\rm cap},\delta t_{\rm grav}\}(1+32\varepsilon)$, $h$ the smallest cell width, $\delta t_{\rm cap}=\sqrt{(\rho^{\rm l}+\rho^{\rm g})h^3/(4\pi\sigma)}$ (thesis (4.1)), $\delta t_{\rm grav}=\sqrt{(\rho^{\rm l}+\rho^{\rm g})h/(2|\rho^{\rm l}-\rho^{\rm g}||g|)}$ = (7.4)$\times1/\sqrt2$.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| 1. $\bar\rho_f$ equals the independent strip-weighted value on all faces | 86-97 | ✅ verified | own polygon clipping of the strips ($\{|x|\le\tfrac12\}\times[lo,hi]$ against the solid half-plane), conditional fractions, clamp; 7 fixtures (uniform/composite × none/sloped/disc, planar interface with solid) | `code_01_density_coefficient.jl` (D1) | difference 0.0 |
| 2. the clamp is inactive | 86-97 | ✅ verified | unclamped weighted fraction | D2 | in $[0,1]$ to rounding (convex combination) |
| 3. coefficients bitwise: $\gamma$, $\gamma^{\rm tv}$, single phase | 99-120 | ✅ verified | package operators vs $1/\bar\rho$, $a^2/\bar\rho$, $a/\bar\rho$ | D3 | difference 0 |
| 4. faces with both strips empty: fraction 0, $a=0$ (457 faces) | 91-92 | ✅ verified | the independent volume vanishes only where $a_f=0$ | D4 | (up to clipping round-off of order $10^{-280}$) |
| 5. refresh keeps the metric | 130-135 | ✅ verified | `PoissonOperator(…; template)` returns the template's `gradient_operator` (same object), $\mathsf K$ entries unchanged, coefficient changed; **the refresh overwrites the template's coefficient in place** | D5 | — |
| 6. body acceleration: $v_f\mathrel{+}=\delta t\,b(x_f,y_f,t,k_f)$ at face centres, interior only, boundary values untouched, callback called once per face at the given (midpoint) time; absent phase values skipped | 495-519 | ✅ verified | package `_add_body_force!` with a known callback (also two-velocity with NaNs) | `code_02_stages_and_bounds.jl` (B1) | deviation 0.0 |
| 7. order BC($t^n$)→transport→body→BC(mid)→viscous→BC($t^{n+1}$)→refresh→forces→project | 578-600 | ✅ verified by code | `single_phase/stepping.jl` (lines of `step!`), `two_velocity/stepping.jl` (`step!`) | — | surface tension then gravity in `_phase_force_substep!`; same order in the two-velocity step |
| 8. from rest with gravity the viscous step does not see the force: one step with $\mu=(5000,500)$ ($\nu\delta t/h^2=0.2$) equals the step with $\mu=0$ | 600-612 | ✅ verified | package `simulate!`, one step, uniform planar fixture | S1 | both $2.2039648682470325\times10^{-7}$ of $\delta t|g|$ (identical in all digits) |
| 9. no forces: velocity stays exactly 0 | 612-618 | ✅ verified | package | S2 | 0.0 |
| 10. $\min\{\delta t_{\rm cap},\delta t_{\rm grav}\}$ = `stable_timestep` at rest ($h=0.25$: 4.2167 and 0.35391) | 620-640 | ✅ verified | package | `code_02` (T1) | equal to $10^{-14}$; thesis (4.1): $\sqrt{\{\!\{\rho\}\!\}h^3/(4\pi\sigma)}$, (7.4): $\sqrt{\tfrac{\rho^{\rm l}+\rho^{\rm g}}{|\rho^{\rm l}-\rho^{\rm g}|}\tfrac h{|g|}}$ — the code has the extra factor $1/\sqrt2$ in the gravity bound, as stated |
| 11. `step!` accepts $\delta t$ = bound, rejects $1.01\times$ bound and names the bound | 640-650 | ✅ verified | package | T2 | message "… exceeds capillary stability bound …" (σ only), "… gravity …" (g only) |

## Errors and consequences

No error. Remarks:
- Step 5: the refresh `PoissonOperator(…; template)` mutates and returns its template (it is not a copy); the text's "rebuilt" is fine, but a reader should not expect the previous coefficient to survive.
- In the two-velocity `step!`, the bound check is performed in `_transport_substep!` only `if !geometry_prepared` (direct `step!` and fixed-$\delta t$ `simulate!`); a CFL-driven `simulate!` selects $\delta t\le$ bound itself.

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| density/coefficient formulas | **sharp** (equalities) | 0.0 / bitwise | `code_01` | — |
| "applying forces after the viscous step keeps the hydrostatic predictor" | **likely sharp** | with the large viscosity the leftover is the same to all digits as without viscosity; a counterfactual with the forces before the viscous step was not run | `code_02` (S1) | — |
| dt bound: gravity factor $1/\sqrt2$ | **sharp** (equality with the code) | T1 | `code_02` | whether the $1/\sqrt2$ is needed was not assessed (a stability experiment would be needed) |
| first-order in time, "no global order claimed" | not assessed | text says so itself | — | — |

## Open points

- `\dtcap` is quoted as the thesis' (4.1); the thesis states (4.1) and adds that a safety factor $1/\sqrt2$ is necessary for stability ("which, up to a safety factor $\tfrac1{\sqrt2}$ … corresponds to the oscillation period of a capillary wave"); the code has no extra factor in $\delta t_{\rm cap}$, and the text's "$\dtcap$ is the thesis' bound (4.1)" is consistent with the code; whether the thesis' $\{\!\{\rho\}\!\}$ in (4.1) is $\rho^{\rm l}+\rho^{\rm g}$ was not looked up. (A literature/thesis check for the caller: thesis §4.1 and Brackbill et al. 1992.)

## History

- 2026-11-09: first audit (`code_01_density_coefficient.jl`, `code_02_stages_and_bounds.jl`). `run_all.jl`: PASS (13 tests).
