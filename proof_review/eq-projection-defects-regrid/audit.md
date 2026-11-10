# Audit: eq:projection:defects, eq:projection:regrid — defects compared with the tolerance; projection after a regrid

- Source: `docs/techref/chapters/10_projection.tex:940-988` (defects 946), `989-1031` (regrid 997, deviation 1017), pressure-update subsection 920-939, commit `b1e7b5b`
- Code audited (normative): `src/two_phase/projection/cmdgfm_projection.jl` (`_project_cmdgfm_assembled!`, `_cmdgfm_residual_defects`, `_cmdgfm_normal_velocity_defect`, `_cmdgfm_initial_solution!`), `mimetic_projection.jl` (`project!`, `_local_mimetic_residual_defects`), `src/two_phase/two_velocity/flow_workspace.jl` (`_project_regridded_two_velocity!`), `src/navier_stokes/simulate.jl:160-180`
- Date: 2026-11-09
- Overall status: ✅ formulas verified (all defects, the convergence rules, the dt-scaling of the regrid projection); ⚠️ three omissions in the text: the unit-dependence of the normal defects is the open issue #94 (not cited), the first-guess rule is simplified, and `project!` itself does not throw
- Rerun: `julia --project=@claude-symbolics proof_review/eq-projection-defects-regrid/scripts/run_all.jl`

## Statement and assumptions

(defects) $\mathcal D^V=\delta t\max_{|c^{\rm o}|>0}|\mathfrak D^{\rm mix}(v^{n+1})|$; $\mathcal D^\eta=\max_{f\in\mathcal F_\xi}|\mathcal N_f(v^{n+1})|$ (CMDGFM); $\mathcal D^\eta=\max\{\max_r|(\mathsf N[u^{n+1}])_r|/\varsigma_r,\max_{\rm inactive}|[u^{n+1}]|\}$ over all candidate rows incl. dependent ones (mimetic); CMDGFM converges iff $\mathcal D^V,\mathcal D^\eta$ and the residual defects $\delta t^2\max_{c\in\mathcal C_r}|r_c|/|c^{\rm o}|$, $\delta t\max_{f\in\mathcal F_\xi}|r_f|$ are $\le\varepsilon$ (restarts BiCGSTAB up to 8 times on the true residual with relative tolerance $\min(\varepsilon,\varepsilon/\max\text{defect})$); mimetic converges iff $\mathcal D^V,\mathcal D^\eta,\mathcal D^\tau\le\varepsilon$, $\mathcal D^\tau=\delta t^2\max_f|r_f|/(m_fV_f^{\rm o\,3/2})$ (restarts MINRES until the residual defects of the three block rows are met); (regrid) $v^\pi\leftarrow v^\pi-\delta t_{\rm prop}G^\pi(a,\xi)$ with the potential $a$ a scratch field; matrices independent of the step, right-hand sides $\propto1/\delta t_{\rm prop}$, so $(a,\xi,\mu)\propto1/\delta t_{\rm prop}$ and the corrected velocity is independent of $\delta t_{\rm prop}$; failure to converge throws.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| D1 CMDGFM defects $\mathcal D^V,\mathcal D^\eta$ equal the code's; `converged` $\Leftrightarrow$ all four $\le\varepsilon$ (also for a run stopped by `max_iterations=3`) | 946-975 | ✅ verified | independent $\mathsf N_f$, mixture flux, system residual $r=b-\mathsf Mx$ (dense), 4 fixtures × 2 runs | `code_01_defects_regrid.jl` | e.g. converged: $\mathcal D^V=1.45\times10^{-11}$, $\mathcal D^\eta=2.0\times10^{-14}$, residual defects $1.45\times10^{-11},1.9\times10^{-14}$; stopped early: $6.1\times10^{-5}>\varepsilon$ and `converged=false` |
| D2 mimetic defects: $\mathcal D^V$, $\mathcal D^\eta$ (all candidate rows + inactive jumps), $\mathcal D^\tau$, multiplier-row defect $\delta t\max|r|/\varsigma$; `converged` $\Leftrightarrow(\mathcal D^V,\mathcal D^\eta,\mathcal D^\tau\le\varepsilon)$ | 976-988 | ✅ verified | independent evaluation on the code's solution vector and assembled system, 4 fixtures × 2 variants | `code_01_defects_regrid.jl` | $\mathcal D^\tau$ to 6 digits ($1.8\times10^{-17}\ldots5\times10^{-18}$); $\mathcal D^\eta$ to $10^{-13}$; all rows are independent on the fixtures (candidates = retained), so the maximum over the candidate rows equals that over the retained rows; the inactive-face jumps raise $\mathcal D^\eta$ in one case ($8.9\times10^{-16}$ vs $5.2\times10^{-16}$, uniform+solid, `joint_face`) |
| D3 normal defects carry the velocity unit; the criterion is not unit-free | 986-988 | ⚠️ confirmed (issue #94) | same flow in velocity units $s=10^{-6},1,10^{6}$ ($u\to u/s$, $\delta t\to s\,\delta t$), $\varepsilon=10^{-3}$ | `code_01_defects_regrid.jl` | $\mathcal D^V$ dimensionless ($9\times10^{-8},4.9\times10^{-6},7.8\times10^{-6}$); $\mathcal D^\eta=5.3\times10^{-4},1.6\times10^{-8},2.8\times10^{-14}$; relative accuracy $\mathcal D^\eta/|[u^*]|$ reached: $6.5\times10^{-10},1.9\times10^{-8},3.4\times10^{-8}$ (factor 52) |
| R1 regrid: velocity independent of $\delta t_{\rm prop}$; matrix independent of $\delta t$; $\delta t\cdot$rhs independent; $\mathcal D^V\propto\delta t$ | 997-1016 | ✅ verified | $\delta t$ vs $3\delta t$, $\varepsilon=10^{-13}$, CMDGFM and mimetic, composite ± solid | `code_01_defects_regrid.jl` | velocities $2.4\times10^{-14}$ (CMDGFM), $2.2\times10^{-15}$ (mimetic); matrix difference exactly $0$; $\delta t\cdot b$ to $2.5\times10^{-16}$; $\mathcal D^V$ ratio $3.0000000000$ |
| R2 "A failure to converge throws"; boundary values at the current time, untruncated step, tolerance of the run | 989-1000 | ✅ code reading | `flow_workspace.jl:_project_regridded_two_velocity!` (`stats.converged || throw`), `simulate!` calls it with `proposed`, `current_time`, `dimensionless_tolerance` | — | |
| R3 the one-velocity model has no such projection | 1000-1010 | ✅ code reading | `simulate!` projects only `velocity isa TwoPhaseVelocity`; `src/mesh/adaptivity/adapt.jl` contains no projection | — | |
| P1 first guess: one-velocity from $p^n$; CMDGFM tries zero and $p^n$ (relative to the anchors) each with the jump from the ILU, accepted only if the residual 2-norm drops and the largest residual defect drops by $\ge5\,\%$; mimetic from zero | 920-939 | ⚠️ code reading (W1) | `_cmdgfm_initial_solution!`, `project!` | — | the $5\,\%$ is relative to the *best so far* (the second candidate must beat the first accepted one) |

## Errors and consequences

No formula is wrong. Three text points:

**W1 (first guess).** The rule is "each candidate is compared with the best residual so far", so the $p^n$ candidate has to beat the zero-pressure candidate (if that was accepted) by $5\,\%$, not only the zero solution. *Optional* rewording: "…and starts from the best candidate, accepted only if it lowers the residual 2-norm and lowers the largest residual defect by at least $5\,\%$ relative to the previous best (initially zero), else from zero".

**W2 (issue #94 not cited).** The last sentence of the defects subsection states the unit-dependence (velocities compared with the dimensionless $\varepsilon$) without referring to the open issue. *Proposed replacement:*

```latex
The volume defects are dimensionless, as with one velocity; the normal defects
$\defect{\eta}$ and the CMDGFM jump-row defect are velocities in the units of the problem,
compared with the same number $\presstol$, so the accuracy they ask for depends on the unit
of velocity (the relative accuracy reached differs by a factor $50$ between velocity units that
differ by $10^{12}$, measured by \code{proof_review/eq-projection-defects-regrid/}; \issue{94}).
```

**W3 (`project!` vs `converged`).** `project!` never throws on non-convergence: it returns `converged=false`; `simulate!`/`step!` callers throw ("pressure projection failed to converge"), `_project_regridded_two_velocity!` throws itself. The text of "Defects and tolerance" says "converges if …"; no change needed, but the regrid paragraph's "A failure to converge throws" is the only throw inside the projection stage.

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| (regrid) velocity independent of $\delta t_{\rm prop}$ | **sharp** (to the solver tolerance) | $2\times10^{-14}$ | `code_01` | — |
| (regrid) the untruncated step only sets the gauge of $\mathcal D^V$ | **sharp** | $\mathcal D^V$ exactly $\propto\delta t$ | `code_01` | — |
| the tolerance on $\mathcal D^\eta$ is unit-free | **fails** (issue #94) | factor 52 in accuracy over $10^{12}$ in units | `code_01` | e.g. $\delta t\,|[u\!\cdot\!n]|/h$ as suggested in #94 |
| mimetic convergence includes the pressure-row defect | **not included** by design: converged $=(\mathcal D^V,\mathcal D^\eta,\mathcal D^\tau)$; the pressure residual enters only through the restart loop's defects | code reading | | state in the text that the restart loop checks all three block rows but `converged` checks three of the four quantities (as the text does) |

## Open points

- The clause "including the dependent ones" (maximum over all candidate rows) is verified only as a formula: no fixture has dependent rows (candidates = retained in all 8 cases); the selection itself is verified on synthetic rows in `eq-projection-mimetic-rows/code_03`.
- `_project_regridded_two_velocity!` was read, not executed (the tests `Regrid on a shortened final step` etc. exist in `test/two_phase/two_velocity/transport/two_velocity.jl`).

## History

- 2026-11-09: first audit (script `code_01_defects_regrid`). `run_all.jl`: PASS (1 code script).
