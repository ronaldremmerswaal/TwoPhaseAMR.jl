# Audit: vof-projected-dilation — dilation bound $\delta t\,\varepsilon^{\widehat p}\sum_{\chi=1}|c^{\rm o}|$ of the projected-liquid carrier

- Source: `docs/techref/chapters/05_vof_transport.tex:817-823` ("A velocity produced by a projection is divergence free only to its solver tolerance ... For the projected carrier every cell with $\chi_c=1$ is active, so the dilation is at most $\delta t\,\varepsilon^{\widehat p}\sum_{\chi_c=1}|c^{\rm o}|$"); stopping rule `05_vof_transport.tex:714-720` ("stops when $\max_{c:C_c>0}|(\mathfrak D^{\rm o}\widehat u)_c|\le\max(\varepsilon^{\widehat p},\cdot)$, the largest strong residual, with $\varepsilon^{\widehat p}=\sqrt{\epsilon_{\rm mach}}$ by default, a round-off floor and at most 200 cycles"); the check sentence 753-768 ($\max|\mathfrak D^{\rm o}\widehat u|=3.0\times10^{-13}$, bound $2.0\times10^{-13}$). Commit `ec20986` (branch `docs/techref-vof`, tree clean). Code: `src/two_phase/two_velocity/transport_workspace.jl` (`_project_liquid_velocity!`), `src/solvers/multigrid/elliptic_solver.jl` (`_solve_krylov_with_stats!`: `target = max(atol, rtol*..., roundoff_floor)`), `src/solvers/multigrid/cycles.jl` (`_roundoff_residual_floor!` $=128\,\epsilon_{\rm mach}\cdot\max(1,\max_c(|A||x|/|c^{\rm o}|),\max_c|b_c|/|c^{\rm o}|)$).
- Date: 2026-11-08
- Overall status: ⚠️ **inequality chain verified, the printed constant is not**: the dilation is at most $\delta t\,\max_{c:C_c>0}|(\mathfrak D^{\rm o}\widehat u)_c|\sum_{\chi=1}|c^{\rm o}|$ and the solver guarantees $\max_c|\cdot|\le\max(\varepsilon^{\widehat p},\epsilon_{\rm fl})$ with the round-off floor $\epsilon_{\rm fl}\ge128\,\epsilon_{\rm mach}$ ($\epsilon_{\rm fl}$ observed up to $2.4\times10^{-11}$, i.e. above $\varepsilon^{\widehat p}=10^{-12}$ in the cut fixtures). The printed bound with $\varepsilon^{\widehat p}$ alone holds only when $\varepsilon^{\widehat p}\ge\epsilon_{\rm fl}$. The bound is tight as an inequality (equality example) but far from sharp for solver output (observed ratio $\le0.021$).
- Rerun: `julia --project=@claude-symbolics proof_review/vof-projected-dilation/scripts/run_all.jl`. Last run: all checks passed (symbolic group and package group).
- Label: none (label now in the source); nearest `eq:vof:projection` (the projection) and `eq:vof:volume-balance` (the identity it feeds).

## Statement and assumptions

By (5.30) with the projected carrier $\upsilon^{\rm o}_f=a_f\widehat u_f$ the liquid volume of a step changes by $\delta t\sum_c\chi_c|c^{\rm o}|(\mathfrak D^{\rm o}\widehat u)_c$ plus snaps. A cell with $\chi_c=1$ has $C^n_c>\tfrac12>0$, hence is *active* in the projection, whose stopping rule bounds $|(\mathfrak D^{\rm o}\widehat u)_c|$ on all active cells by the solver target. Hence $|\Delta V^{\rm l}|\le\delta t\,\varepsilon^{\widehat p}\sum_{\chi=1}|c^{\rm o}|$ if no fraction is snapped.

Hidden assumptions: (a) the divergence in the stopping rule is the **open** divergence $(\mathfrak D^{\rm o}v)_c=\frac1{|c^{\rm o}|}\sum_fo|f|a_fv_f$ (`eq:setting:open-divergence`) of the **final** velocity and the solver's strong residual equals it to round-off; (b) the target is $\max(\varepsilon^{\widehat p},\epsilon_{\rm fl})$ ("$\cdot$" in the text), with $\epsilon_{\rm fl}$ the solver's round-off floor, **not** $\varepsilon^{\widehat p}$; (c) no clamp/snap in the step; (d) the projection and the transport use the same fractions (one projection per `advect!`, deterministic).

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| S1 Gauss on a composite configuration: $\sum_c|c^{\rm o}|\mathfrak D^{\rm o}v=0$, so the signed sum over $\chi=1$ equals minus the sum over $\chi=0$ cells (not controlled by the stopping rule) | 817-823 | ✅ | symbolic | `step01` S1 | why the bound is loose |
| S2 dilation $=\delta t\sum_{\chi=1}|c^{\rm o}|(\mathfrak D^{\rm o}\widehat u)_c$ | 799-803 | ✅ | package: $\Delta V$ vs the dilation computed independently from the projected velocity (sum of $o|f|a\widehat u$): $\le10^{-13}$ on 20 runs (no snaps) | `code_01` (iii) | |
| S3 every cell with $\chi_c=1$ is active ($C>0$) | 818-820 | ✅ | package: all $\chi=1$ cells have $C>0$ on all runs; by definition $C^n>1/2$ | `code_01` (i) | trivial |
| S4 $\max_{\rm active}|\mathfrak D^{\rm o}\widehat u|\le$ solver target; target $=\max(\varepsilon^{\widehat p},\epsilon_{\rm fl})$ | 714-720 | ✅ with the floor | package: 20 runs, tolerances $10^{-2},10^{-4},10^{-8},10^{-12},10^{-15}$, plain and cut: $\max|\mathfrak D^{\rm o}\widehat u|\le$ `stats.target_linf` always; it equals `stats.final_linf` to $\le10^{-12}\max(1,\text{target})$ and the reported `projection_final_divergence` to $10^{-13}$ | `code_01` (ii) | for $\varepsilon^{\widehat p}\ge10^{-8}$: target $=\varepsilon^{\widehat p}$ exactly |
| S5 printed bound $\delta t\varepsilon^{\widehat p}\sum_{\chi=1}|c^{\rm o}|$ | 817-823 | ⚠️ valid iff $\varepsilon^{\widehat p}\ge\epsilon_{\rm fl}$ | same runs: target $=\varepsilon^{\widehat p}$ for $\varepsilon^{\widehat p}\ge10^{-8}$; at $10^{-12}$ the target is $1.48\times10^{-12}$ (plain), $4.2\times10^{-12}$, $3.4\times10^{-12}$ (cut); at $10^{-15}$ it is $4\times10^{2}$–$2.4\times10^{4}$ times $\varepsilon^{\widehat p}$ (max $\lvert\mathfrak D^{\rm o}\widehat u\rvert$ up to $1.6\times10^{-12}$) | `code_01` (iv) | the sum $\sum_{\chi=1}|\text{dilation}_c|$ exceeds the printed bound at $10^{-15}$ |
| S6 the replaced bound $\delta t\,\max(\varepsilon^{\widehat p},\epsilon_{\rm fl})\sum_{\chi=1}|c^{\rm o}|$ | — | ✅ | all runs without snaps: $\lvert\Delta V\rvert\le$ bound; observed $\lvert\Delta V\rvert/\text{bound}\le0.021$ | `code_01` (iv) | |
| S7 the chain $|\sum x_c|\le\sum|x_c|\le\tau\sum|c^{\rm o}|$ is tight | 817-823 | ✅ | equality example: a one-velocity carrier with $(\mathfrak D^{\rm o}u)_c=\tau$ on every $\chi=1$ cell (Poisson solve on the uniform periodic fixture, divergence error $5.6\times10^{-18}$): $\Delta V/(\delta t\,\tau\sum_{\chi=1}|c^{\rm o}|)=0.9999999999999993$ | `code_01` (v), `step01` S3/S4 | |
| S8 docs check: $\max\lvert\mathfrak D^{\rm o}\widehat u\rvert=3.0\times10^{-13}$, $\Delta V=5.3\times10^{-15}\le2.0\times10^{-13}$ | 753-768 | ✅ consistent | the bound there uses the *achieved* divergence, as in the replacement below | docs `chkVofProjected*` | the check is correct; the general sentence is not stated with the achieved value |

## Errors and consequences

**F1 (printed constant).** $\varepsilon^{\widehat p}$ must be replaced by the solver target $\max(\varepsilon^{\widehat p},\epsilon_{\rm fl})$ or, sharper, by the achieved $\max_{C_c>0}|(\mathfrak D^{\rm o}\widehat u)_c|$ (reported as `projection_final_divergence`). Evidence: with $\varepsilon^{\widehat p}=10^{-15}$ the solver stops at a target $\ge4\times10^{-13}$ (up to $2.4\times10^{-11}$ on the cut mesh); with $\varepsilon^{\widehat p}=10^{-12}$ the target is $>\varepsilon^{\widehat p}$ in 3 of 4 runs. The text's stopping rule itself is correct ("$\max(\varepsilon^{\widehat p},\cdot)$"); only the consequence in Section 5.9 drops the floor. For the default $\varepsilon^{\widehat p}=\sqrt{\epsilon_{\rm mach}}=1.5\times10^{-8}$ the floor is irrelevant ($\epsilon_{\rm fl}\ll10^{-8}$ in all runs), so no result of the chapter changes.

**F2 (looseness).** The bound is not sharp for solver output: observed $|\Delta V|\le0.021$ of the (floor-corrected) bound, because the stopping rule bounds only the magnitude while the solver residual has mixed signs; the equality case needs a one-signed divergence $=\tau$ on every $\chi=1$ cell, which holds for the equality example but not for Krylov output.

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| dilation $\le\delta t\,\tau\sum_{\chi=1}|c^{\rm o}|$ with $|\mathfrak D^{\rm o}\widehat u|\le\tau$ on $\chi=1$ cells | **sharp as an inequality** (equality example, ratio $1-7\times10^{-16}$) | equality example | `code_01` (v) | — |
| $\tau=\varepsilon^{\widehat p}$ | **not valid** below the round-off floor; valid above | target $=\varepsilon^{\widehat p}$ for $\ge10^{-8}$, $\gg$ for $10^{-15}$ | `code_01` | use $\max(\varepsilon^{\widehat p},\epsilon_{\rm fl})$ or the achieved value |
| bound for solver output | **not sharp** (observed ratio $\le0.021$) | table in `code_01` | `code_01` | none needed; remark only |
| all $\chi=1$ cells active | **trivially true** ($C>1/2$) | — | — | — |

## Open points

- The round-off floor depends on the problem ($128\epsilon_{\rm mach}\cdot\max(1,\ldots)$); the claim "irrelevant for the default tolerance" was checked on the random fixtures here (largest floor $2.4\times10^{-11}$), not proved.
- The result is for the one projection per `advect!`; the flow-level count (Section 5.9, "in a simulation the volume changes per step by the dilation of that residual") follows by summing.

## Proposed replacement text

`05_vof_transport.tex:819-822`, replace "For the projected carrier every cell with $\indicator{\liq}_c=1$ is active, so the dilation is at most $\dt\projtol\sum_{\indicator{\liq}_c=1}\meas{\opencell{c}}$." by

```latex
For the projected carrier every cell with $\indicator{\liq}_c=1$ is active, so the dilation is at
most $\dt\,\max_{c:\,\condfrac{\liq}{c}>0}\abs{(\divopen\projvelo^{\liq})_c}\sum_{\indicator{\liq}_c=1}\meas{\opencell{c}}$
and, by the stopping rule of \cref{sec:vof:projected},
\begin{equation}\label{eq:vof:projected-dilation-bound}
  \abs{\dt\sum_c\indicator{\liq}_c\meas{c}(\divh\volumeflux^{\open})_c}
  \le\dt\,\max(\projtol,\roundfloor)\sum_{\indicator{\liq}_c=1}\meas{\opencell{c}},
\end{equation}
with $\roundfloor$ the round-off floor of the solver (at least $128\machineeps$). For $\projtol$ above the
floor, in particular for the default, the bound is $\dt\projtol\sum_{\indicator{\liq}_c=1}\meas{\opencell{c}}$.
```

The replacement introduces one new symbol, $\epsilon_{\mathrm{fl}}$ (the solver's round-off floor, `_roundoff_residual_floor!`); no macro for it exists in `techref-notation.sty`. If the symbol is kept, a `\DefineNotation` entry in the group `vof` (\code{code}: `_roundoff_residual_floor!`, thesis: new) is the consistent way to register it.

**Applied** on 2026-11-08 to `docs/techref/chapters/05_vof_transport.tex` (uncommitted, branch `docs/techref-vof`), new-file lines `848-856`; the bound is now the displayed equation (5.31), label `eq:vof:projected-dilation-bound`; the symbol is registered as `\roundfloor` in `techref-notation.sty:544-546`. The techref builds with 0 undefined references and `check_links.py` passes.

## History

- 2026-11-08: first audit. Symbolic chain and equality algebra (9 checks); 20 projected steps (plain and cut meshes, tolerances $10^{-2}\ldots10^{-15}$) against the independent open divergence, the solver's target and the volume change; equality example. Outcome: ⚠️ chain verified, printed constant needs the round-off floor.
