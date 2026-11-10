# Audit: ch:overview — Chapter 1: stage lists of one step, the `simulate!` run sequence, the options table, the thesis-step table

- Source: `docs/techref/chapters/01_overview.tex`: `sec:overview:step` with `tab:overview:thesis` (l. 322–368), `sec:overview:step-one` (l. 370–454), `sec:overview:step-two` (l. 456–529), `sec:overview:run` (l. 531–599), `sec:overview:options` with `tab:overview:options` (l. 601–683); cross-checked against `08_forces.tex` (`eq:forces:stages`, l. 695–770), `09_viscosity.tex` (`dev:viscosity:time-level`), `10_projection.tex` (`dev:projection:incremental`, `dev:projection:regrid`), `13_stability_accuracy.tex` (`eq:stab:selection`, `eq:stab:check`) and the thesis sources `pt2_overview.tex` (§7.2). Commit `ad4f795` (worktree `/tmp/techref-ch1`, branch `docs/techref-overview`; the chapter is pinned to `619ea56`; `git diff 619ea56 HEAD -- src` is empty, so the code is the one of the pin). The sections of the chapter outside these labels (scope, governing equations, discretisation at a glance, limitations table, API table) are not part of this audit.
- Date: 2026-11-10
- Overall status: ⚠️ the stage lists, the run sequence and the thesis table are **correct** (every statement verified on an instrumented run); **one wrong table row** (E1, the Krylov row of `tab:overview:options`) and **five wording defects** with replacement LaTeX (W1–W4), none of which changes a number or an order
- Rerun: `julia --project=docs/techref/figures proof_review/ch-overview/scripts/run_all.jl` (flags for the shared machine are set by the driver: `--heap-size-hint=3G -t 1`, `OPENBLAS_NUM_THREADS=1`, `JULIA_NUM_PRECOMPILE_TASKS=1`; three child processes of 1–4 minutes each; the worktree has no root `Manifest.toml`, so the `docs/techref/figures` project resolves `TwoPhaseAMR` to the working tree)

## Statement and assumptions

Claims audited (each is a statement about what the code does, so the code is the reference; the thesis only for the table):

1. `sec:overview:step-one`: the eight stages of `step!` with a `FaceVelocity` (one phase and two phases with one velocity): start boundary values (and the open divergence kept for `dt < dtref`), transport (check at CFL ½ against the transport and, with two phases, capillary and gravity bounds; one phase: staggered convection of $v^n$ with itself as frozen carrier in two sweeps of alternating order; two phases: WY sweeps of fraction, phase scalars and density-weighted momentum in alternating order), body acceleration then boundary values at $t^n+\tfrac12\delta t$, viscous step (omitted for two phases if both viscosities vanish), phase forces (coefficient rebuilt from $C^{n+1}$, preconditioner refreshed for unequal densities, surface tension then gravity, each only if switched on; nothing for one phase), end-of-step boundary values **after** the forces, old-pressure predictor, projection with the retained share of the inherited divergence.
2. `sec:overview:step-two`: the nine stages of `step!` with a `TwoPhaseVelocity`; in particular boundary values at $t^{n+1}$ **before** the forces, the pressure-map stage (CMDGFM: reconstruction + one-velocity coefficient without a solid, the cached operator with a solid; mimetic: the same maps, only if there are phase forces), the common force increments, the CMDGFM predictor and increment projection versus the absolute mimetic KKT projection, the saved flux for a shortened CMDGFM step, the check skipped when `simulate!` has just evaluated the bounds.
3. `sec:overview:run`: set-up, step size, step, regrid, events, end of `simulate!`; the events `:initial`, `:step`, `:completed`, `:failed`; the stop polling.
4. `tab:overview:options`: the default (first) value of each choice, the admitted values, the legacy markings, and the Krylov method and preconditioner of each system.
5. `tab:overview:thesis`: T1–T5 of the thesis §7.2 against the stages of the code, with the stated deviations.

**Assumptions made explicit** (not all stated in the chapter): (a) $\delta t=\delta t_{\rm ref}$ unless stated; (b) "boundary values at $t$" means `enforce_velocity_boundary!` plus the covered-face zeroing of the solid (`enforce_geometry_no_slip!`); for the open (Outflow/ZeroGradient) sides the predictor variant `_enforce_predictor_velocity_boundary!` extrapolates the change since $v^n$; (c) the diffusion solver gets the step start time $t^n$ and evaluates the walls at the midpoint $t^n+\tfrac12\delta t$ internally (`implicit_midpoint_diffusion!`), consistent with "implicit midpoint"; (d) "stage" means the call of the function the implementation box names.

## Method

**Instrumented run, not reading.** `scripts/_trace.jl` loads a *copy* of the package (module `TracedAMR`) built from the unmodified `src/` through `include(mapexpr, …)`: every top-level definition of a function in the list `TRACED` (40 stage functions, e.g. `step!`, `_transport_substep!`, `_body_force_substep!`, `_enforce_predictor_velocity_boundary!`, `_diffusion_substep!`, `_refresh_pressure_substep!`, `surface_tension!`, `gravity!`, `_pressure_predictor!`, `_project_prepared!`, `_project_two_velocity!`, `adapt!`, `_record_diagnostics!`, `_observe_safely!`, …) gets one extra first statement that appends its name and selected argument values (times, `dt`, sweep axis, flags) to a log. Nothing else is changed (the external packages are those of the real package). The log of one `step!`/`simulate!` is compared with the list of the chapter, in order. The observer of `step02` is a custom `AbstractObserver` that records every event with its `StepInfo`.

Fixtures (all small: 8×8 cells, one block, walls; the regrid cases 4×4 blocks with one refinement level): a single phase with a sine shear flow; a drop ($\rho=(1000,1)$, $\mu=(10^{-2},10^{-3})$, $\sigma=0.07$, $g=(0,-1)$) with one velocity, with two velocities (CMDGFM + frozen fluxes, with and without a solid; projected-liquid carrier; mimetic with and without a solid and forces).

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| S1.1 one-phase order: BC($t^n$) → check → transport → body → BC($t^n+\tfrac12\delta t$) → viscous → [refresh, force: no-ops] → BC($t^{n+1}$, covered faces) → predictor → projection | step-one 1–8 | ✅ verified (instrumented run) | trace of `step!`; times 0.25, +dt/2, +dt recorded | `step01` | the BC at $t^n+\tfrac12\delta t$ is imposed also without a body acceleration (W1) |
| S1.2 sweeps of the single-phase convection alternate (x,x,y,y / y,y,x,x) | item 2 | ✅ verified | `_dispatch_convection!` sweep axes over 4 steps | `step01` | the carrier is $v^n$ in both sweeps (field ≠ carrier object) |
| S1.3 two phases, one velocity: same list plus coefficient refresh and surface tension then gravity, each only if on; $\mu=0$ omits the viscous solve; $\sigma=0$, $g=0$ omit the force calls | items 4, 5 | ✅ verified | 4 force/viscosity configurations | `step01` | |
| S1.4 the coefficient is that of the **transported** fraction $C^{n+1}$, and the preconditioner is refreshed only for unequal densities | item 5 | ✅ verified | drop moved by uniform flow: face coefficient after the step equals $1/\bar\rho$ of the new fraction to round-off and differs from the old one; `_refresh_geometric_multigrid!` called between the coefficient refresh and the force stage for $\rho=(1000,1)$, not for $\rho=(1,1)$ | `step01` | |
| S1.5 WY sweeps of the fraction alternate in order over steps | item 2 | ✅ verified | `_compute_directional_wy_fluxes!` axes (2,1 / 1,2 / 2,1) | `step01` | |
| S1.6 shortened step: the projection receives $\delta t_{\rm ref}\ge\delta t$ | intro, items 1, 8 | ✅ verified | `simulate!` to 0.0025 with `dt=1e-3`: $(\delta t,\delta t_{\rm ref})=(10^{-3},10^{-3}),(10^{-3},10^{-3}),(5\cdot10^{-4},10^{-3})$ | `step01` | the share $1-\delta t/\delta t_{\rm ref}$ itself is audited in Chapter 10 |
| S2.1 two-velocity CMDGFM order: BC($t^n$) → [saved flux] → transport → body → BC($t^n+\tfrac12\delta t$) → viscous → BC($t^{n+1}$) → pressure map → forces → old-pressure predictor ($-\delta t$) → CMDGFM projection | step-two 1–8 | ✅ verified | trace; BC times $t^n$, $t^n+\tfrac12\delta t$, $t^{n+1}$; force increments added with $+\delta t$ (surface tension, then gravity), the predictor with $-\delta t$ | `step01` | the end-of-step BC is **before** the forces here, **after** them with one velocity: as stated |
| S2.2 CMDGFM without a solid reconstructs the transported interface and refreshes the one-velocity coefficient; with a solid it uses the cached operator and does not refresh | item 6 | ✅ verified | `reconstruct!` between `_two_velocity_pressure_operator!` and `_refresh_pressure_substep!`; with a solid `_phase_force_pressure_operator!` follows, no refresh | `step01` | |
| S2.3 saved mixture flux only for CMDGFM and $\delta t<\delta t_{\rm ref}$ | item 1 | ✅ verified | called once, in the shortened step, after the start BC and before the transport; not in the mimetic step | `step01` | |
| S2.4 "the step is checked … unless `simulate!` has just evaluated them" | item 2 | ⚠️ true, but misleading (W2) | frozen carrier with `dt=nothing`: `stable_timestep` before `step!`, transport with `geometry_prepared=true`, no `_check_transport_timestep`; prescribed `dt` or projected-liquid carrier: check inside the step | `step01` | the skip needs both an automatic step **and** the frozen carrier |
| S2.5 mimetic: map only if there are phase forces (one-velocity operator without a solid, cached operator with one), forces common to both phases, no old-pressure predictor, absolute projection | items 6–8 | ✅ verified | 4 configurations (forces / no forces; with / without solid; shortened step) | `step01` | the projection is called with `pressure_prepared = phase_forces` |
| S2.6 projected-liquid carrier: WY sweeps alternate | item 2 | ✅ verified | axes over 4 steps | `step01` | |
| R1 events `:initial`, `:step` after every step, `:completed`; counters, times, `StepInfo.dt` (0, step, …, 0), `wall_time`, `topology_changed`; the procedure only at `:initial` | items 1, 5, 6 | ✅ verified | recording observer on a 3-step run with a shortened last step | `step02` | |
| R2 order of calls: diagnostics before each observer event; stop polled before the first step and after every step; `:completed` diagnostics only if not recorded at the same step | items 1, 5, 6 | ✅ verified | trace of `_record_diagnostics!`, `_observe_safely!`, `_observer_stop_requested`, `step!`; history steps $[0,1,2,3]$ | `step02` | |
| R3 stop polling: stop after step $k$ → events `:initial`, `:step`×$k$, `:completed`, `stopped=true`, final time $k\delta t$; stop before the first step → no step; a throwing poll is logged and means "continue" | items 1, 5 | ✅ verified; ⚠️ the last behaviour is not in the text (W4) | polls $=1+$steps | `step02` | |
| R4 observer exceptions: at `:initial` propagates before the first step (and no `:failed`); at `:step`/`:completed` logged, run continues; `InterruptException` propagates after `:failed` | items 1, 5, 6 | ✅ verified; ⚠️ "no `:failed` at `:initial`" not stated (W4) | | `step02` | |
| R5 `:failed`: a step above the bounds raises `ArgumentError`, not retried, rethrown with the exception in `StepInfo`; an unconverged projection raises `ErrorException`; an observer error at `:failed` does not replace it | items 3, 6 | ✅ verified; ⚠️ an unconverged **viscous** solve and the post-regrid projection also stop the run (W3) | | `step02` | |
| R6 step size: first step equals `stable_timestep` at CFL 0.45; `cfl` scales it; never increases although the bound grows 23×; remainder $<10^{-6}$ of the proposal is absorbed (3 steps, last $\delta t+5\cdot10^{-10}$), a larger one is its own step | item 2 | ✅ verified | | `step02` | |
| R7 retained history: after an automatic run the unmodified proposal and the last step; a prescribed step retains only the last step (proposal $\infty$); standalone `step!` and `project!(flow)` end it; `project!` leaves the pressure unchanged | end, closing paragraph | ✅ verified; ⚠️ "the proposal … retained" needs "after an automatic step" (W5) | `simulation_step` fields | `step02` | |
| R8 regrid: `adapt!(step = completed steps, time = end of step)` → rebuild → [two velocities: projection of the transferred velocities with the unshortened proposal as $\delta t$, at the end-of-step time] → diagnostics → `:step` with `topology_changed`; defect reference $=\delta t_{\rm prop}$; acts at most every `update_interval` steps | item 4 | ✅ verified | one and two velocities; `update_interval=2`: calls with steps 1,2,3,4, acts at 1 and 3 | `step02` | |
| R9 set-up: AMR workspace registered before the first diagnostics; an observer that advances copies is rejected with adaptivity | item 1 | ✅ verified | `_supports_dynamic_topology = false` → `ArgumentError("exact keyframes …")` | `step02` | uses the internal hook that `Recorder(exact_keyframes=true)` overrides |
| O1 defaults, first value of every row: `OneVelocityModel`, `FrozenInterfaceFluxes`, `CMDGFMProjection` (`:block`), `MimeticProjection` (`:joint_smoothed`), `CGFM`, `LaxWendroff`, `:lhf`, no refinement, CFL 0.45 | `tab:overview:options` | ✅ verified | constructors with and without keywords; values admitted and rejected | `step03` | |
| O2 states at level transitions for all 8 schemes (MC for LaxWendroff/Centred, else the scheme itself) | row 7 | ✅ verified | `AdvectionOptions(scheme=s).amr_interface_scheme` | `step03` | an upwind transition scheme with a non-upwind scheme is rejected |
| O3 legacy markings | rows 2, 3, text | ✅ verified | by reading `docs/src/navier-stokes.md`, `two-velocity-advection.md` and the docstrings of `step!` ("legacy mimetic", "legacy carrier"); the three reasons named in the text are the issues #76, #45 and the absolute projection (Chapter 10, 11) | — | |
| O4 Krylov row: "CG for symmetric positive (semi)definite systems, otherwise BiCGSTAB; tree geometric multigrid (`:auto`)" | row 10 | ❌ wrong for the mimetic system (MINRES), the CMDGFM system (own block preconditioner) and the liquid carrier (no preconditioner) | Krylov workspace of every system after one step: pressure CG (preconditioner `:tree_geometric_multigrid`); one-phase viscous CG on a uniform mesh and BiCGSTAB on a composite mesh; two-phase viscous CG; CMDGFM BiCGSTAB; mimetic MINRES; the preconditioner names (`:block`/`:ilu`/none, block Cholesky) by reading `src/two_phase/projection/methods.jl` | `step03` | E1, replacement below |
| O5 caps 100 / 500, block size 16×16, `update_interval` 1, `coarsen_delay` 4 | implementation boxes | ✅ verified | | `step03` | |
| T1 the five thesis steps T1–T5 of §7.2 are the boxes interface advection, momentum transport, diffusion, gravity, pressure Poisson in this order (thesis `pt2_overview.tex`: `step:overview:fraction_advection`, `…momentum_advection`, `…diff`, `…grav`, `…poisson`); T1 is "dimensionally unsplit … donating regions (Owkes 2014)"; T3 is implicit; T4 gravity; T5 contains surface tension in the scaled gradient and the Poisson problem | `tab:overview:thesis` | ✅ verified by reading the thesis source | | — | the thesis' Remark 7.1 (order of steps) is the hydrostatic argument quoted in Chapter 8 |
| T2 code column: stage of the code for each thesis row and the two extra rows (body acceleration; projection after a regrid) | table | ✅ verified | the traces of S1/S2/R8 | `step01`, `step02` | |
| T3 "single phase: explicit half at $v^{(n)}$" (`dev:viscosity:time-level`) | row T3 | ✅ verified | `implicit_midpoint_diffusion!(velocity, convected, old_velocity, …)`: right-hand side $v^{(1)}+\tfrac12\delta t\,\nu L\,v^{n}$ (source, docstring of `implicit_diffusion.jl`) | — | two phases and two velocities use the transported state in both halves, as `dev:viscosity:time-level` says |
| T4 consistency with `eq:forces:stages` (Chapter 8) and Chapter 10 | table, intro | ✅ verified; ⚠️ two defects of Chapter 8 (X1, X2) | S1/S2 traces against `eq:forces:stages` | `step01` | below |

## Errors and consequences

**E1 — `tab:overview:options`, row "Krylov method and preconditioner" (l. 635–637) is wrong for three systems.** The row says "CG for symmetric positive (semi)definite systems, otherwise BiCGSTAB (`tab:solvers:systems`); tree geometric multigrid (`:auto`)". In the code (and in `tab:solvers:systems`): the mimetic KKT system is symmetric **indefinite** and solved by **MINRES** with its own block-Cholesky preconditioner; the CMDGFM system is nonsymmetric, BiCGSTAB with its own block preconditioner (`:block`, `:ilu` or none, row 4); the liquid carrier is CG **without** preconditioner. The sentence "tree geometric multigrid (`:auto`)" holds for the pressure projection and for both viscous steps only. The one-phase viscous step chooses CG or BiCGSTAB by the symmetry test (verified: CG on a uniform mesh, BiCGSTAB on a composite one). Replacement:

```latex
    Krylov method and preconditioner (\apitype{LinearSolverOptions} and \apitype{MultigridOptions})
      & pressure projection and viscous steps: CG, or BiCGSTAB after a symmetry test for the one-phase viscous step, with the tree geometric multigrid (\code{:auto}); the CMDGFM system (BiCGSTAB) and the mimetic system (MINRES) have their own preconditioners (\cref{tab:solvers:systems})
      & \cref{ch:solvers}\\
```

Consequence: none for numbers; the same simplification occurs in the glance paragraph "Linear solvers" of `sec:overview:glance` ("CG for the Poisson projection, … BiCGSTAB for the CMDGFM and MINRES for the mimetic system. Except for the mimetic system and the liquid carrier they are preconditioned by V-cycles"), which is correct, so after the replacement the two statements agree.

**Wording defects (statements true as read in the best sense, misleading as printed):**

**W1 — body acceleration stage (step-one item 3, l. 393–397; step-two item 3, l. 476–478).** The boundary values at $t^n+\tfrac12\delta t$ are imposed in every step, also without a body acceleration (`_body_force_substep!` with `force = nothing` is a no-op, the boundary stage after it is not; verified: `_enforce_predictor_velocity_boundary!(time = t+dt/2)` is called with `force = nothing`). Replacement for item 3 of the one-velocity list:

```latex
\item \emph{Body acceleration.}
  $\velo^{\stage{1}}\leftarrow\velo^{\stage{1}}+\dt\,\bodyacc(\timevar^{\timelevel{n}}+\half\dt)$
  on the interior faces~\cref{eq:forces:body} (nothing is added without a body
  acceleration); then, in every step, the boundary values at
  $\timevar^{\timelevel{n}}+\half\dt$, open faces extrapolating the change
  since $\velo^{\timelevel{n}}$.
```

and for the two-velocity list:

```latex
\item \emph{Body acceleration} on both phases where they are
  present~\cref{eq:forces:body}; then, in every step, the boundary values at
  $\timevar^{\timelevel{n}}+\half\dt$.
```

**W2 — step-two item 2 (l. 467–470).** "The step is checked against the bounds, with the frozen interface fluxes unless `simulate!` has just evaluated them for this state" can be read as "the check evaluates the frozen interface fluxes". The check is skipped only when `simulate!` chose an *automatic* step with the *frozen* carrier (then `step!` receives `geometry_prepared = true`); with a prescribed step or the projected-liquid carrier the step checks the bounds itself (verified on three runs). Replacement of the first sentences:

```latex
\item \emph{Transport.} The step is checked against the bounds unless
  \apifun{simulate!} has just evaluated them for this state (automatic step,
  frozen interface fluxes); that evaluation also reconstructs the interface and
  freezes the phase apertures of the step start (\cref{sec:vof:frozen}). The
  WY sweeps with the selected carrier~\cref{eq:vof:frozen-flux}
```

(the rest of the item unchanged). The same imprecision is in `08_forces.tex`, the paragraph after `eq:forces:stages` ("only a two-velocity step whose step size \apifun{simulate!} has just chosen from these bounds skips the check"): add "with the frozen interface fluxes"; `13_stability_accuracy.tex` states it correctly (X2).

**W3 — run item 3 (l. 549–552).** "A projection that has not converged stops the run with an error" omits the other solves: an unconverged viscous solve (staggered or variable) and the post-regrid two-velocity projection throw `ErrorException` too (verified for the viscous solve: `max_cycles = 0` stops the run in the viscous step, with the message "staggered diffusion solve failed to converge", before the projection is reached). Replacement:

```latex
\item \emph{Step.} One \apifun{step!} as in
  \cref{sec:overview:step-one,sec:overview:step-two}. A linear solve that has
  not converged (viscous step, projection, and the projection after a
  regrid) stops the run with an error; a step that fails a run-time check is
  not retried (\cref{sec:stab:gaps}).
```

**W4 — run items 1, 5, 6 (l. 537–542, 560–565, 566–570): three behaviours that are tested but not stated.** (i) An exception of the observer at `:initial` ends the call and the observer receives no `:failed` event (the `try` block starts after `:initial`). (ii) A throwing `stop_requested` is logged and counted as "continue" (an interrupt propagates). (iii) An error of the observer at `:failed` is dropped, the original exception is rethrown. Replacement of the two sentences that carry them:

```latex
... The diagnostics are recorded and the observer receives the
  event \code{:initial}; an exception there ends the call before the first
  step, without a \code{:failed} event. The observer is asked whether to stop
  (an exception of the poll is logged and counts as no).
```
```latex
... exception of the observer is logged and the run continues, an interrupt
  propagates. ...
\item \emph{End.} ... If the run throws, the observer receives
  \code{:failed} with the exception (an error of the observer there is
  dropped), which is then rethrown.
```

**W5 — run item 6 (l. 566–569).** "the proposal and the last step are retained for a call that continues the run" holds for an automatic step only; with a prescribed step the last step is retained but the proposal is $\infty$ (`13_stability_accuracy.tex` says so). Replacement: "the last step and, after an automatic step, the proposal are retained for a call that continues the run".

**Cross-chapter findings (not in this file):**

- **X1 — `08_forces.tex` `eq:forces:stages`** draws the boundary values at $t^{n+1}$ twice (after $v^{(2)}$ and after $v^{(3)}$) for both models; the code applies them once: after the forces with one velocity, before the forces with two velocities (Chapter 8 says the latter in the text below the display; the one-velocity reading is left to the reader). This is the note already recorded in the history of `proof_review/eq-forces-stages/audit.md` (2026-11-10); Chapter 1's lists are correct and agree with the code.
- **X2 — `08_forces.tex`** "only a two-velocity step whose step size `simulate!` has just chosen from these bounds skips the check": add "with the frozen interface fluxes" (W2).

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| order and content of the stage lists | **sharp** (exact structural equalities) | the traces equal the lists in every traced configuration (one phase; one velocity with the four on/off combinations of viscosity, surface tension and gravity; two velocities: CMDGFM with and without solid, automatic and prescribed step, shortened step, projected-liquid carrier, mimetic with and without forces and solid) | `step01` | — |
| "each force only if on; $\mu=0$ omits the viscous solve" | **sharp** | the four on/off combinations traced | `step01` | — |
| `step!` checks at CFL ½, `simulate!` selects at 0.45 (margin $0.5/0.45$) | **sharp** as a statement; the margin is a design choice | `step02` E: first step equals the bound at 0.45; 0.3 scales by 2/3 | `step02` | — |
| "never increases the step within a run" | **sharp** | bound grows by a factor 23 (viscous decay of the sine flow), step unchanged | `step02` | — |
| "a remainder below $10^{-6}$ of the proposal is absorbed" | **sharp** at the threshold used | $5\cdot10^{-10}$ absorbed (step $\delta t+5\cdot10^{-10}$), $5\cdot10^{-8}$ a step of its own | `step02` | — |
| "acts at most every `update_interval` steps" | **sharp** | interval 2: acts at steps 1 and 3, called at 1–4 | `step02` | — |
| table of defaults / legacy | **sharp** | | `step03` | — |

## Open points

- The thesis equation numbers quoted in `tab:overview:thesis` and the box (§7.2, eq. 7.52 in `dev:viscosity:time-level`) were not re-read; the *order* of T1–T5 and the content of the five boxes were checked against the thesis source (`/home/ronald/Documents/phd_project/thesis/pt2_overview.tex`, boxes `step:overview:*`).
- The two-velocity stages with a solid were traced for CMDGFM and mimetic with a small sloped solid; Outflow/ZeroGradient sides (open boundary extrapolation in `_enforce_predictor_velocity_boundary!`) were not traced.
- The sections of Chapter 1 outside the audited labels were not audited; `docs/techref/check_links.py` was not run.
- A literature search is not needed.

## History

- 2026-11-10: first audit (instrumented run of the package; scripts `step01`–`step03`, driver `run_all.jl`), commit `ad4f795`. Result: ⚠️ one wrong row (E1) and wording defects W1–W5; the stage lists, the run sequence and the thesis table are correct. `run_all.jl`: PASS, 96 checks (`step01_stage_order` 32, `step02_run_sequence` 41, `step03_options_defaults` 23), run after the last edit of the scripts (the first full run had four script errors, all in the expectations of the scripts and not in the chapter: the in-step `_projection_weighted_gradient!` and `_add_common_phase_acceleration!` calls were missing from the expected stage lists, a run to an absolute time was read as relative, a single phase needs a positive viscosity, and the composite test mesh had no refined block).
- 2026-11-10: corrections applied: E1 (Krylov row of `tab:overview:options`, with the liquid carrier unpreconditioned and the one-phase viscous CG/BiCGSTAB choice), W1–W5 in `01_overview.tex`; X1 (`eq:forces:stages` draws the boundary values at $t^{n+1}$ once: after the forces with one velocity, before them with two) and X2 in `08_forces.tex`.
