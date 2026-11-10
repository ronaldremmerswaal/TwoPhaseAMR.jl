# Audit: eq:projection:one-system (group) — one-velocity Poisson projection: system, gauge, exact defect identity, stopping rule, energy identity

**Historical audit:** the results and defect descriptions below refer to the
audited pre-fix commit. Issues #95 and #97 subsequently changed the code and
chapter: compatibility and pressure gauges are componentwise, and the
current-iterate roundoff floor is guarded by the initial residual. The linked
scripts now check the updated behavior; the measurements below are not post-fix results.

- Source: `docs/techref/chapters/10_projection.tex:71-192` (one-system 84, one-gauge 101, one-defect 121, one-tolerance 127, one-energy 141; the boundary-value stage 884-918 is audited in `dev-projection-outflow`), commit `b1e7b5b` (branch `docs/techref-projection`, working tree `/tmp/techref-ch10`)
- Code audited (normative): `src/navier_stokes/projection.jl` (`project!`), `src/embedded/operators.jl` (`project!`, `_cut_cell_weighted_gradient!`, `PoissonOperator(geometry)`), `src/solvers/multigrid/{elliptic_solver.jl,cycles.jl,stats.jl}` (`solve!`, `_roundoff_residual_floor!`, `_project_integrated_rhs!`, `_remove_volume_mean!`, `_dimensionless_tolerance`), `src/two_phase/forces.jl` (`PoissonOperator(field[, geometry])`)
- Date: 2026-11-09
- Overall status: ⚠️ all five labelled claims verified (system, defect identity, stopping rule incl. the round-off floor, gauge, orthogonality/energy identity); **two text corrections** (F1 the energy sentence with nonzero boundary values is ambiguous about the side/sign, F2 the disconnected-domain sentence omits that the default preconditioner throws) and two pre-existing **code defects** on domains split by a solid (D1: the default projection throws; D2: a per-component flux imbalance is reported as converged with a diverged pressure)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-projection-one-system/scripts/run_all.jl`

## Statement and assumptions

**Claims (as printed).**
1. (one-system) $\mathsf S p=b$, $b_c=-\tfrac1{\delta t}|c^{\rm o}|(\mathfrak D^{\rm o}v^{*})_c$, $v^{n+1}_f=v^*_f-\delta t(\mathsf R\mathsf Jp)_f$ on interior faces and $v^{n+1}_f=v^*_f$ on boundary faces, with $\mathsf S=\mathsf J^\top\gamma^{1/2}\mathsf K^{\rm o}\gamma^{1/2}\mathsf J$, $\gamma=1/\bar\rho$ (no solid) or $a^2/\bar\rho$ (solid), covered cells identity rows with $p_c=0$.
2. (one-gauge) $b\leftarrow b-\bar b\,|c^{\rm o}|$, $\bar b=\sum b_c/\sum|c^{\rm o}|$, volume mean of $p$ zero; `compatibility=:error` throws iff $|\sum b_c|>256\,\epsilon\max(1,\sum|b_c|)$; $\sum b_c=-\tfrac1{\delta t}\times$ boundary flux of $v^*$; on a domain split by a solid only the global mean is fixed (velocity unique, component constants undetermined).
3. (one-defect) with $r=b-\mathsf Sp$: $|c^{\rm o}|(\mathfrak D^{\rm o}v^{n+1})_c=-\delta t(r_c+\bar b|c^{\rm o}|)$ for every open cell.
4. (one-tolerance) the solve stops when $\delta t^2\max_c|r_c|/|c^{\rm o}|\le\max\{\varepsilon^{p},\delta t^2\epsilon_{\rm fl}\}$, $\epsilon_{\rm fl}=128\epsilon_{\rm mach}\times$ the largest strong magnitude of the terms of $\mathsf Sp$ and of $b$ (at least $128\epsilon_{\rm mach}$); the left-hand side equals $\delta t\max|\mathfrak D^{\rm o}v^{n+1}|$ for $\bar b=0$.
5. (one-energy) $v^\top\mathsf M_\gamma\mathsf R\mathsf Jp=v^\top\mathsf A\mathsf Jp$ with $\mathsf M_\gamma=\mathsf A\gamma^{-1/2}(\mathsf K^{\rm o}_{++})^{-1}\gamma^{-1/2}\mathsf A$; hence $\|v^*\|^2_{\mathsf M}=\|v^{n+1}\|^2_{\mathsf M}+\delta t^2\|\mathsf R\mathsf Jp\|^2_{\mathsf M}$ if $\mathfrak D^{\rm o}v^{n+1}=0$ and the boundary values vanish; $\mathsf M$ is the diagonal of the staggered masses $\bar\rho V$, $\bar\rho V^{\rm o}$, $a\bar\rho V^{\rm o}$ where $\mathsf K^{\rm o}$ is diagonal; "with nonzero boundary values the left-hand side gains $2\delta t\sum_{f\in\mathcal F_{\rm bnd}}s_f|f|a_fv_fp_{c(f)}$".

**Assumptions made explicit.** Walled box (no periodic directions) for the checks; $\mathsf K^{\rm o}$ symmetric positive definite on the faces with $\gamma>0$ (eq:setting:patch-metric, audited elsewhere); boundary values are those *used in the solve* (the values before the final reset of sec:projection:stage — the identity concerns the corrected velocity before the reset, see `dev-projection-outflow`); `tolerance` lies in $(0,1)$ (code: `_dimensionless_tolerance` throws otherwise; not stated in the text).

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| A1 $\mathsf S=\mathsf J^\top\gamma^{1/2}\mathsf K^{\rm o}\gamma^{1/2}\mathsf J$ with the stated $\gamma$ ($\bar\rho$ from an independent staggered fraction) | 76-83, 84-95 | ✅ verified | dense independent assembly vs `operator.stiffness`, 4 chapter fixtures + 2 split-domain fixtures | `code_01_system_defect.jl` | $\le2\times10^{-16}$ relative; coefficient $\gamma$ to $10^{-13}$ |
| A2 $-|c^{\rm o}|\mathfrak D^{\rm o}\mathfrak G_\gamma p=\mathsf Sp$ ("$-\mathsf V\mathfrak D^{\rm o}\mathfrak G=\mathsf S$") | 96-100 | ✅ verified | random $p$; independent open divergence of the package's weighted gradient | `code_01_system_defect.jl` | $\le1.6\times10^{-16}$ |
| A3 update on interior faces with $\mathsf R\mathsf Jp$ (independent $\mathsf R=\operatorname{diag}(\gamma^{1/2}/a)\mathsf K^{\rm o}\operatorname{diag}(\gamma^{1/2})$), boundary faces unchanged | 84-95, 96-100 | ✅ verified | returned velocity vs $v^*-\delta t\,\mathsf R\mathsf J p$ | `code_01_system_defect.jl` | $\le4.6\times10^{-16}$; boundary difference exactly 0 |
| A4 (one-defect) $|c^{\rm o}|\mathfrak D^{\rm o}v^{n+1}=-\delta t(r+\bar b|c^{\rm o}|)$ every open cell | 119-126 | ✅ verified | independent $\mathsf S$, $b$, $\bar b$, $r$ | `code_01_system_defect.jl` | $\le4.1\times10^{-14}$ of $\max|\mathfrak D v^*|$ (all 6 fixtures incl. the split domain); code `removed_rhs_mean` = independent $\bar b$ |
| A5 covered cells: identity row, $p_c=0$, no flux | 91-92 | ✅ verified | 34 / 102 / 32 / 80 covered cells | `code_01_system_defect.jl` | $p_c/\max|p|\le1.4\times10^{-15}$, flux 0 |
| T1 stopping rule $\delta t^2\max|r_c|/|c^{\rm o}|\le\max\{\varepsilon,\delta t^2\epsilon_{\rm fl}\}$ | 127-132 | ✅ verified | independent residual, $\varepsilon=10^{-9},10^{-12}$ | `code_02_tolerance_floor.jl` | e.g. $1.1\times10^{-13}\le10^{-12}$; `converged` agrees |
| T2 floor $=128\epsilon\max(1,\text{strong terms of }\mathsf Sp,\ |b|)$ | 133-137 | ✅ verified | $\varepsilon=10^{-30}$ (unreachable): the code's `target_linf` vs the formula from the final $p$ | `code_02_tolerance_floor.jl` | identical to all printed digits ($1.084435902093914\times10^{-8}$; $5.111672809557269\times10^{-8}$) |
| T3 $\delta t\max|\mathfrak D^{\rm o}v^{n+1}|=\delta t^2\max|r|/|c^{\rm o}|$ ($\bar b=0$) | 137-141 | ✅ verified | independent divergence | `code_02_tolerance_floor.jl` | equal to $10^{-3}$ relative (the difference is $\bar b\approx10^{-15}$) |
| T4 tolerance must lie in $(0,1)$ | — (not in the text) | ✅ code | `ArgumentError` for $0,1,-1,\mathrm{NaN}$ | `code_02_tolerance_floor.jl` | add to the text (F3) |
| G1 volume mean of $p$ is zero | 101-108 | ✅ verified | 4 fixtures | `code_03_gauge.jl` | $\le5\times10^{-17}$ relative to $\sum|c^{\rm o}||p|$ |
| G2 $\bar b=\sum b_c/\sum|c^{\rm o}|$ with integrated $b$; solution solves $\mathsf Sp=b-\bar b|c^{\rm o}|$ | 101-108 | ✅ verified | `solve!` with a right-hand side of mean 0.37 | `code_03_gauge.jl` | removed mean 0.37 to $10^{-14}$, strong residual $\le2\times10^{-9}$ |
| G3 `:error` throws iff $|\sum b_c|>256\epsilon\max(1,\sum|b_c|)$ | 108-111 | ✅ verified | bracketing factors 0.5 and 2 of the threshold | `code_03_gauge.jl` | no throw / throw |
| G4 $\sum_c b_c=-\tfrac1{\delta t}\times$ boundary flux (random boundary values); zero for Inflow+Outflow | 111-114 | ✅ verified | discrete divergence theorem; channel with Inflow/Outflow | `code_03_gauge.jl` | $\le10^{-12}$; removed mean $-8\times10^{-17}$ |
| G5 split domain: 2 null vectors of $\mathsf S$ (= components); only the global mean fixed; velocity unique | 114-118 | ✅ verified with `:ilu` | eigenvalue count; two projections started from pressures differing by $(3\times10^4,-10^4)$ on the two components | `code_03_gauge.jl` | velocity difference $1.3\times10^{-14}$; pressure difference constant per component ($-19704$, $+20296$, spread $4\times10^{-12}$), global mean $10^{-12}$ |
| G6 (not in the text) default `:tree_geometric` on the split domain | — | ❌ defect D1 | CG throws "operator or preconditioner is not symmetric positive definite" | `code_03_gauge.jl`, `repro_01_split_domain.jl` | also `:geometric`; `nothing`, `:ilu` converge |
| Q1-Q3 per-component compatibility: a right-hand side with zero total but unequal component sums | 114-118 | ❌ defect D2 | `solve!` with `:ilu` on the split domain | `code_05_component_flux.jl` | removed mean $6.6\times10^{-17}$ (nothing removed), pressure diverges to $9\times10^{18}$, strong residual $104>|b|$, yet **`converged=true`** (stop level $8.6\times10^{6}$: the round-off floor scales with the iterate); with zero flux per component: converges ($3\times10^{-10}$) |
| Q2 the removed mean is subtracted from every cell proportionally to its volume (all components) | 114-118 | ✅ verified | rhs mean 0.25 | `code_05_component_flux.jl` | $0.2500000000000001$ |
| Q4 `project!` with Inflow into the lower component only + Outflow right (uniform correction balances the total only) | 884-902 | ❌ defect D2 | `project!`, tolerance $10^{-8}$ | `code_05_component_flux.jl` | `converged=true`, $\delta t\max|\mathfrak Dv|=0.83$, $\max|p|=1.3\times10^{19}$ |
| E1 $v^\top\mathsf M_\gamma\mathsf R\mathsf Jp=v^\top\mathsf A\mathsf Jp$ | 141-147 | ✅ verified | dense $\mathsf K^{\rm o}_{++}$ inverse, random $v,p$ | `code_04_energy.jl` | $\le3.3\times10^{-15}$ |
| E2 Pythagoras after `project!` (density ratio 1000; 0.8-47 % of the energy removed) and $\Delta v=-\delta t\,\mathsf R\mathsf Jp$ | 147-156 | ✅ verified | independent norms | `code_04_energy.jl` | $\le6.9\times10^{-16}$ relative; $\Delta v$ to $3.4\times10^{-16}$ |
| E3 diagonal metric: $\mathsf M=\bar\rho V$ (no solid), $\bar\rho V^{\rm o}$, and $a\bar\rho V^{\rm o}$ for $\gamma^{\rm tv}=a/\bar\rho$ | 148-154 | ✅ verified | on all faces whose row of $\mathsf K^{\rm o}$ has a single entry (all on the uniform meshes; 1162/1240 and 971/1037 on the composite ones) | `code_04_energy.jl` | $\le10^{-12}$ |
| E4 nonzero boundary values: $\|v^*\|^2=\|v^{n+1}\|^2+\delta t^2\|\mathsf R\mathsf Jp\|^2+2\delta t\,B$, $B=\sum_{\rm bnd}s_f|f|a_fv^*_fp_{c(f)}$ | 156-158 | ⚠️ identity verified, sentence ambiguous (F1) | Inflow/Outflow channel on a composite mesh; $v^{n+1}$ before the boundary reset | `code_04_energy.jl` | residual $2.4\times10^{-16}$ with $+2\delta tB$ on the right-hand side; the opposite sign misses by $2.5\times10^{-2}$ of $\|v^*\|^2$ ($2\delta tB/\|v^*\|^2=-1.2\times10^{-2}$) |

## Errors and consequences

No error in a formula. Two wording points and one pre-existing defect:

**F1 (one-energy, last sentence).** "the left-hand side of the identity gains the boundary pressure work $2\delta t\sum\ldots$" does not say on which side of the equality the term stands. The verified identity is $\|v^*\|^2_{\mathsf M}=\|v^{n+1}\|^2_{\mathsf M}+\delta t^2\|\mathsf R\mathsf Jp\|^2_{\mathsf M}+2\delta t\,B$ (so $B=\sum_{f\in\mathcal F_{\rm bnd}}s_f|f|a_fv_fp_{c(f)}$ is the work done by the pressure at the boundary; for an inflow with $p>0$, $B<0$ and the correction *adds* energy to the final state), where $v^{n+1}$ is the corrected velocity **before** the boundary values are applied again; the boundary reset changes the Outflow values and the identity is then not an identity of the end-of-step velocity (see `dev-projection-outflow`).

*Proposed replacement of the last sentence of the paragraph (macros existing):*

```latex
With nonzero boundary values the pressure does work at the boundary,
\begin{equation*}
  \norm{\velo^{\stage{3}}}^2_{\kinmetric{\facecoef}}
  =\norm{\velo^{\timelevel{n+1}}}^2_{\kinmetric{\facecoef}}
  +\dt^2\norm{\forcemap\jumpmat\pres}^2_{\kinmetric{\facecoef}}
  +2\dt\sum_{f\in\facesbnd}\fside{f}\meas{f}\aperture{f}\velo_f\pres_{\cbnd{f}},
\end{equation*}
where $\velo^{\timelevel{n+1}}$ is the corrected velocity before the boundary values
are applied again (\cref{sec:projection:stage}) and $\velo_f$ the boundary values of the predictor;
the identity holds to the tolerance of the solve ($2\times10^{-16}$ on the Inflow--Outflow channel of
\code{proof_review/eq-projection-one-system/}).
```

**F2 (one-gauge, disconnected domains).** The sentence "the pressure constant of every component but one is whatever the Krylov iteration converges to" is true for the conjugate gradients without or with an ILU preconditioner (verified: velocity unique to $10^{-14}$, component constants follow the initial pressure), but with the **default** tree-geometric multigrid preconditioner (whose coarse level removes one constant) the solve **throws** (defect D1). Either fix the code (issue #95) or state it. *Proposed replacement of lines 114-118 of the source (the sentence from "On an open domain" to "all components"):*

```latex
On an open domain that a solid splits into several components, $\stiffmat$ has one null
vector per component but only the global mean is fixed: the velocity is unique, and the pressure
constant of every component but one is the one the Krylov iteration started from (\issue{95}).
The default \code{:tree_geometric} preconditioner removes a single constant on its coarsest level
and the iteration then stops with the error that the operator or the preconditioner is not
symmetric positive definite; the \code{:ilu} preconditioner and the unpreconditioned iteration
converge. The prescribed normal flux must vanish on every component separately; only its total is
checked and removed (as $\removedmean$ from every cell), and with a nonzero flux on one component the
solve diverges without reporting it (\issue{95}).
```
(The removed mean is indeed subtracted from every cell proportionally to its volume — verified, Q2 — but this cannot make the components compatible one by one, Q1/Q4.)

**F3 (one-tolerance).** Add "(with $0<\presstol<1$)" after "dimensionless tolerance $\presstol$"; and that `project!` returns `converged=false` rather than throwing (only `simulate!`/`step!` callers throw; `eq:projection:regrid` throws explicitly).

**D1 (code defect, minimal reproducer `scripts/repro_01_split_domain.jl`).** `project!` with `EllipticSolver(op)` (default `:tree_geometric`) on a `GeometryField` whose solid band cuts the box in two: `ErrorException: The linear operator A or the preconditioner M is not symmetric positive definite` (also `:geometric`); `nothing` converges in 55 iterations, `:ilu` in 4. Cause (not investigated beyond the reading): the coarse solve of `TreeGeometricMultigridPreconditioner` is augmented by one constant vector (`coarse_augmented`), whereas $\mathsf S$ has two null vectors. Related to issue #95 ("one-velocity pressure gauge on disconnected domains"); #95 does not mention the throw. Not fixed.

**D2 (code defect, reproducer `scripts/code_05_component_flux.jl`, Q1/Q4).** On a domain split by a solid, `compatibility=:error` and the mean removal only look at the global sum. If the flux of one component does not vanish (e.g. `Inflow` into the lower part and an `Outflow` side on the whole right wall: the uniform outflow correction makes the *total* zero, not the components), the system is inconsistent per component; CG (with `:ilu` or unpreconditioned, the choices that do not throw, D1) diverges to $\max|p|\sim10^{19}$ and the solve **reports `converged=true`** because `_roundoff_residual_floor!` is proportional to $|\mathsf S||p|$ of the diverged iterate (stop level $8.6\times10^{6}$ against a strong residual of $104$ and a right-hand side of size 1). After `project!` the cell volumes change by $\delta t|\mathfrak Dv|=0.83$ per step. Not fixed. The one-velocity text should say that the prescribed flux must vanish *per component* and that this is not checked (issue #95 covers the gauge, not this false convergence).

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| (one-defect) exact cell-wise identity | **sharp** (equality) | residual $\le4.1\times10^{-14}$ of the predictor's divergence, all open cells, composite + solid + split domain | `code_01` | — |
| (one-tolerance) "bounds the fraction of volume gained per step" | **sharp** | $\delta t\max|\mathfrak Dv|=\delta t^2\max|r|/|c^{\rm o}|$ to $b̄\sim10^{-15}$ | `code_02` | — |
| (one-tolerance) floor: when $\varepsilon<\delta t^2\epsilon_{\rm fl}$ the floor is the stop level | **sharp** | $\varepsilon=10^{-30}$: stop level = formula to all digits | `code_02` | state that the solve then stops at the floor, not at $\varepsilon$ |
| (one-energy) Pythagoras $\Leftarrow$ $\mathfrak D^{\rm o}v^{n+1}=0$, vanishing boundary values | **sharp** | without vanishing boundary values the identity fails by exactly $2\delta tB$ (1.2 % of $\|v^*\|^2$ in the channel) | `code_04` | the exact extension E4 replaces the "if" |
| (one-gauge) threshold $256\epsilon\max(1,\sum|b|)$ | **sharp** (as implemented) | throws at $2\times$, not at $0.5\times$ the threshold | `code_03` | — |
| (one-gauge) compatibility is per component | **sharp** | counterexample Q1/Q4: component sums $\pm27$ with zero total diverge | `code_05` | check per component and fail loudly |
| (one-gauge) "only the global mean is fixed" | **sharp** | component constants follow the initial pressure (difference $4\times10^4$ preserved) | `code_03` | per-component anchors as in the two-velocity projections (issue #95) |

## Open points

- The coefficient $\gamma$ and the metric $\mathsf K^{\rm o}$ are those verified in `eq-forces-*`/`eq-setting-patch-metric`; here only their consistent use is checked.
- Literature: none needed.

## History

- 2026-11-09: first audit (scripts `code_01`-`code_05`, `repro_01`). `run_all.jl`: PASS (5 code scripts + reproducer; G6/Q1/Q4 assert the presence of defects D1/D2).
