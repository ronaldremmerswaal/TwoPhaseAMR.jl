# Audit: `api` slice (TwoPhaseAMR.jl)

Scope: `src/navier_stokes/**`, `src/simulation/**`, `src/single_phase/**`, `src/visualization/**`, `ext/*.jl`, `src/TwoPhaseAMR.jl`, `Project.toml`, plus the high-level user path (README Quick start, `docs/src/getting-started.md`, `docs/src/api.md`).
Method: static reading plus single-threaded (`-t 1`, BLAS 1 thread) Julia 1.12.7 scripts in a `/tmp/audit-api/env` environment (package `Pkg.develop`ed, Aqua added). No JET. No tracked file was modified. Scripts and raw output: `/tmp/audit-api/{misuse.jl,misuse2.jl,async.jl,infer.jl,alloc.jl,aqua*.jl,aqua.out,exports.txt,usage.tsv}`.

## 1. Summary

- **API surface:** 231 exported names (142 types, 89 functions). About 13 are pure aliases. 145 of the 231 never appear in README or examples. 23 appear in no test, doc or example. `docs/src/api.md` already says "most applications need ~25 names", but the module still exports all 231.
- **Deprecation:** the documented "actionable deprecation warning" is invisible at default Julia flags (`Base.depwarn` is silent without `--depwarn=yes`; verified). `Project.toml` says `0.1.0-DEV` while the docs promise a window "throughout 0.2.x".
- **Design:** `FlowSimulation` is concretely typed and `step!` infers (good). The weaknesses are the options round-trip (typed options → flat 21-parameter solver → `NamedTuple` and back), about 200 lines of duplicated stepping and rebuild code, no high-level `step!(flow, dt)`, no restart or output, and no CFL guard for single-phase fixed `dt` (two-phase has one).
- **Visualization:** `asynchronous=true` delivers 0 frames during `simulate!` when run on the creating thread (measured: 0 delivered, 20 of 21 dropped). All 11 examples work around it with a `Threads.@spawn`+`fetch` boilerplate. The Makie extension reaches into ~25 private names of the core package, and `visualize` is a single 364-line function.
- **Aqua:** 4 of 11 checks fail: 1 ambiguity (`Diagnostic(::Symbol, ::Symbol)`), 1 stale dep (`SpecialFunctions`), missing `[compat]` for 3 stdlib deps and 2 extras, plus 1 undocumented export (`MimeticProjectionSolver`). Piracy, unbound args, undefined exports and persistent tasks pass.

## 2. Findings table

| ID | Sev | Area | file:line | Evidence | Proposal | Benefit | Effort |
|---|---|---|---|---|---|---|---|
| A1 | high | API surface | src/TwoPhaseAMR.jl:22-110 | 231 exports; 145 absent from README/examples; docs/api.md:34-52 lists ≈25 "standard" names | Export only the standard surface (~45 names, §3.1). Mark advanced building blocks `public` (1.11+) behind a version guard. Document that "public" means "may change in minor versions". | Smaller namespace (no collisions with `Centered`, `Slip`, `LW`, `MC`, `Face`, `Cell`, `interior`, `integrate`); explicit stability contract | M |
| A2 | high | Deprecation | src/navier_stokes/flow_simulation.jl:280, src/two_phase/two_velocity/transport_methods.jl:76 | Verified: default `julia` prints nothing for `FlowSimulation(mesh; density=..)` and for `interface_method=`; `--depwarn=yes` prints the warning | Replace `Base.depwarn` by `Base.depwarn(...; force=true)` or `@warn ... maxlog=1 _id=...` (or `Base.@deprecate`-style, but those are also silent). Alternatively delete the legacy path if the package is pre-release. | The migration docs' promise ("emits an actionable warning") becomes true | S |
| A3 | high | Version policy | Project.toml:4 vs docs/src/api-decisions.md:30, migration.md:5 | Version `0.1.0-DEV`; docs promise "supported throughout 0.2.x, removal no earlier than 0.3" | Decide whether the package has been released. If not, delete all compat aliases and the legacy keyword path now (clean cutover, ≈150 lines + 13 exports). If yes, bump to the version the docs name. | Removes a contradiction that blocks cleanup | S |
| A4 | high | Project hygiene | Project.toml:7-16, 41-52 | Aqua: stale dep `SpecialFunctions` (used only by test/integration/prosperetti_amplitude.jl:2); `Random` used only by test/navier_stokes/two_phase_navier_stokes.jl:3; no `[compat]` for LinearAlgebra, Random, SparseArrays, Test, BenchmarkTools | Move `SpecialFunctions`, `Random` out of `[deps]`. Add stdlib `[compat]` entries (`LinearAlgebra = "1"`, `SparseArrays = "1"`, …). Add `test/Project.toml` (Test, Random, SpecialFunctions, AlgebraicMultigrid, LimitedLDLFactorizations, CairoMakie, Aqua) and drop `[extras]/[targets]`. | Two fewer load-time deps; Aqua clean; CI no longer mutates Project.toml (CI.yml:167 runs `Pkg.add("CairoMakie")` into the main project) | S |
| A5 | high | Visualization async | src/visualization/session_types.jl:297-301, frame_capture.jl:~238 | Measured, `-t 1`, 20 steps, fake viewer: sync = 21/21 frames; `asynchronous=true` = 0 frames before `simulate!` returned, 1 after `wait_visualization!`, `dropped=20`. The `@async` publisher is sticky to the creating thread and `simulate!` never yields | Add `yield()` after `_publish_latest!` in the async branch of `_publish_live_frame!`, and document that the publisher runs on the constructing task's thread. Then remove the per-example `Threads.@spawn`/`fetch` boilerplate (11 files) by letting `simulate!(flow, T; visualization=session)` do it when `asynchronous=true`. | Live view works when `simulate!` is called directly (REPL, docs). ≈10 lines × 11 examples removed | S |
| A6 | med | Misuse safety | src/single_phase/stepping.jl:68-129 | Single-phase fixed `dt=1.0` on a 16×16 Taylor–Green mesh (CFL ≈ 2.5) runs silently, KE 9.87→6.52→5.99→6.23. Two-phase throws "time step exceeds … stability bound" (two_phase/stepping.jl:80-84) | Apply the same `dt <= bound` check to single phase, or warn once. Put the numeric bound in the message (currently none). | Prevents silent garbage on the most common first run | S |
| A7 | med | API completeness | src/navier_stokes/flow_simulation.jl (no `step!(::FlowSimulation)`); docs/src/diagnostics.md:104-112 | The documented manual loop calls 5-arg low-level `step!`, then hand-increments `flow.time`, `flow.steps`; it has no AMR (`adapt!` / `_rebuild_topology_dependent_state!` is only inside `simulate!`) and no `force` persistence | Add `step!(flow, dt; force)` that does time/steps bookkeeping, diagnostics, AMR and returns stats; implement `simulate!` as a loop over it. | The only way to do custom time loops today cannot adapt the mesh | M |
| A8 | med | Restart / output | (absent) | `grep -il "serialize\|checkpoint" src` hits visualization only; no save/restore of a `FlowSimulation`; "continue" works only in-memory | Add `write_checkpoint(path, flow)` / `FlowSimulation(path, …)` (a minimal binary/JLD2-free format via `Serialization` or an extension) and an output-writer callback. | Long runs become restartable | L |
| A9 | med | Config round-trip | src/navier_stokes/options.jl:73-100; workspace.jl:36-76; system.jl:62-84 | `NavierStokesOptions` is flattened to 14 kwargs (`_navier_stokes_keywords`), stored as 14 flat fields in a 21-type-parameter mutable struct, re-expanded by `_navier_stokes_configuration` for `_independent_solver` | Store `options::NavierStokesOptions` and `physics` in the system; delete both translators and the repeated `solver.coarse_size, pre_sweeps, …` kwargs lists. | ≈90 lines removed; one source of truth for defaults | M |
| A10 | med | Duplication | see §3.4 | Identical blocks in single/1V/2V stepping and in the two topology rebuilds | Factor out as in §3.4 | ≈190–230 lines | M |
| A11 | med | Naming | src/navier_stokes/system.jl:94-104 | Preferred name `NavierStokesSystem` is the alias; the real type (printed in errors/`typeof`) is `IncompressibleNavierStokesSolver`; field is `flow.solver` | Rename the struct to `NavierStokesSystem`; rename field `solver`→`system` (docs say "system"). | Types print with the documented name | S |
| A12 | med | Extension boundary | ext/TwoPhaseAMRMakieExt.jl (50 `TwoPhaseAMR.<name>` refs, ≈25 private) | Ext calls `_field_entry`, `_render_source_values`, `_phase_visualization_geometry`, `_visualization_topology`, … | Define a documented backend interface in core (`frame_vertex_values(frame, name)`, `topology(frame)`, …) and make the ext consume only that; or accept that ext is internal and say so. | Backend authors (WGLMakie-only, VTK, headless PNG) can reuse the capture layer; internal refactors stop breaking the ext | M |
| A13 | med | Ext structure | ext/TwoPhaseAMRMakieExt.jl:570-933, 6-62 | `visualize` is 364 lines; `MakieAMRVisualization` has 56 fields and 16 type parameters, many `Ref{Any}`/`Observable` | Split into `_build_controls`, `_build_axes`, `_wire_callbacks`; replace the 16 type parameters with `Any` fields in a UI struct (type-stability is irrelevant for GUI state). | Readability; shorter compile | M |
| A14 | med | Custom extensibility | src/mesh/adaptivity/indicators.jl:2; staggered_diffusion.jl:59-76; flow_simulation.jl:254-259 | Custom refinement criteria: `AbstractRefinementIndicator` is exported but its contract (`_adaptive_refinement_labels!`, …) is undocumented. Custom BCs: `VelocityBoundaryConditions` rejects anything outside a closed `Union`. Body force: only a `force=` kwarg of `simulate!` (not stored in the `FlowSimulation`; absent from README/getting-started). | Document the three extension points with a tested minimal example each; store `force` in the `FlowSimulation` (or `NavierStokesOptions`). | Users can extend without reading internals; forgetting `force=` on a second `simulate!` call no longer drops it | M |
| A15 | low | Aqua | src/simulation/diagnostics/interface.jl:11-15 | Ambiguity `Diagnostic(::Symbol, ::Symbol)` | Add the missing method, or drop the `Diagnostic(measure, name)` do-block form | Aqua ambiguity clean | S |
| A16 | low | Docs | src/two_phase/two_velocity/projection/mimetic_projection.jl (exported) | `MimeticProjectionSolver` has no docstring (`checkdocs = :exports` in docs/make.jl:53 would miss non-exported) | Add docstring or unexport | Aqua undocumented_names clean | S |
| A17 | low | Visualization layers | src/visualization/{session_types,frame_capture,controls,updates}.jl | 24 hand-written `lock(x); try … finally unlock(x) end` sites, 0 uses of `@lock` | Use `Base.@lock` (Julia ≥ 1.7) | ≈70 lines | S |
| A18 | low | Visualization layers | session_types.jl:~100-150, frame_capture.jl:~180-235 | `_live_sampling`/`_diagnostic_sampling` and `_live_frame_due!`/`_diagnostic_due!` are copy-paste twins; `_checkpoint_specification` repeats the vector branch twice | One `_sampling(T; every, dt, name)` and one `_due!(schedule, …)` | ≈50 lines | S |
| A19 | low | Observed perf | whole step | `alloc.jl`: steady-state bytes/step 116 KB (64 cells), 310 KB (256), 1.03 MB (1024), i.e. ≈1 KB/cell/step, linear in N | Hand to the performance slice: something in the single-phase step allocates O(N) arrays per step | Possible GC pressure on large meshes | M |

## 3. Details

### 3.1 Export list (A1, A3)

Counts (`names(TwoPhaseAMR)` minus the module: 231; classification by grep over README, `examples/*.jl`, `docs/src`, `test`, see `/tmp/audit-api/usage.tsv`).

**Pure aliases among the exports** (`const X = Y`):

| Alias | Defined | Target | uses in src / test / examples / docs |
|---|---|---|---|
| `Centered` | operators/advection/staggered_schemes.jl:32 | `Centred` | 17 / 10 / 0 / 2 (counts include `Centred`-adjacent text) |
| `LW` | staggered_schemes.jl:35 | `LaxWendroff` | 24 / 44 / 1 / 19 (it is the default argument in 6+ signatures) |
| `MC` | staggered_schemes.jl:38 | `MonotonizedCentral` | 12 / 32 / 0 / 12 |
| `Slip` | operators/staggered_diffusion.jl:27 | `FreeSlip` | used in examples/sloped_sessile_droplet.jl:70 |
| `CenteredMagnitude` | visualization/types.jl:93 | `CentredMagnitude` | 8 src (the code uses the alias as the primary spelling in sampling.jl:131,139,334, registry.jl:98,133,160) |
| `NavierStokesSimulation` | navier_stokes/flow_simulation.jl:73 | `FlowSimulation` | 0 outside docs |
| `NavierStokesSolver` | navier_stokes/system.jl:94 | `IncompressibleNavierStokesSolver` | used internally at flow_simulation.jl:267; 26 test hits |
| `NavierStokesSystem` | system.jl:104 | same | the documented name is the alias |
| `GeometricMultigridPreconditioner` | solvers/multigrid/cycles.jl:3 | `TreeGeometricMultigridPreconditioner` | 1 test |
| `GeometricMultigridSolver` | cycles.jl:6 | same | **0 uses anywhere** |
| `MultigridStats` | solvers/multigrid/geometric_preconditioner.jl:18 | `LinearSolverStats` | **0 uses** |
| `StaggeredGeometricMultigridPreconditioner` | navier_stokes/implicit_diffusion.jl:249 | `TreeStaggeredGeometricMultigridPreconditioner` | 1 test |
| `StaggeredGeometricMultigridSolver` | implicit_diffusion.jl:253 | same | **0 uses** |
| `AbstractTwoVelocityPressureProjection` | two_phase/two_velocity/projection/methods.jl:5 | `AbstractTwoVelocityProjection` | 1 test |

`AbstractEllipticOperator` (src/TwoPhaseAMR.jl:20) is used only as a supertype at operators/elliptic.jl:202 and :888; no method dispatches on it.

`simulation_state` is exported twice (src/TwoPhaseAMR.jl:63 and :110). It also names two unrelated functions: `simulation_state(flow)` returns a `SimulationState` (flow_diagnostics.jl:2) while `simulation_state(session)` returns a lifecycle `Symbol` (controls.jl:32). Rename the latter to `session_state`.

**Exports with no occurrence in tests, docs or examples (23):** `AMRFaceGeometry AbstractBoundaryCondition AbstractDiagnostic AbstractDiagnosticSchedule AbstractRefinementIndicator AbstractStaggeredAdvectionScheme AbstractWallBoundaryCondition DiagnosticCollection DiagnosticHistory GeometricMultigridSolver LeafBlock MultigridStats PLICSolidGeometry StaggeredGeometricMultigridSolver VisualizationCheckpoint VisualizationDiagnosticSample VisualizationTopology ZeroGradient algebraic_multigrid_preconditioner diagnostic_name enforce_geometry_no_slip! incomplete_cholesky_preconditioner incomplete_lu_preconditioner register_interface!`.
Several of these are user-facing extension hooks (`AbstractDiagnostic`, `diagnostic_name`, `AbstractRefinementIndicator`) that are undocumented, so they are untested promises (see A14).

**Proposed layout**

1. `export` (standard surface, ≈45 names): `BlockTreeMesh MeshAdaptationOptions InterfaceField CellField FaceField FaceVelocity SinglePhase TwoPhase SurfaceTension CGFM CSF GFM OneVelocityModel TwoVelocityModel FrozenInterfaceFluxes ProjectedLiquidVelocity CMDGFMProjection MimeticProjection NavierStokesOptions AdvectionOptions LinearSolverOptions MultigridOptions Adaptivity InterpolationErrorIndicator CurvatureRefinementIndicator VorticityRefinementIndicator InterfaceRefinementIndicator GeometryRefinementIndicator NoSlip FreeSlip Inflow Outflow ZeroGradientVelocity VelocityBoundaryConditions FlowSimulation simulate! add_diagnostic! diagnostic_history print_simulation_summary EveryStep EveryNSteps EveryTimeInterval KineticEnergy PotentialEnergy SurfaceEnergy TotalEnergy LinearMomentum PhaseVolume PhaseMass InterfaceMeasure MaximumVelocity CFLTimeStep DivergenceNorm LeafBlockCount CellCount Diagnostic VisualizationFieldRegistry VisualizationSession visualize register_default_fields!`.
2. `public` (Julia ≥ 1.11; guard for 1.10): everything else that is intentionally supported (fields, operators, workspaces, multigrid, stats, visualization registry/frames, `step!`, `project!`, `adapt!`, `stable_timestep`, scheme types `Upwind`, `Fromm`, `Minmod`, `LaxWendroff`, …). Compat-free idiom:
   ```julia
   @static if VERSION >= v"1.11"
       include_string(@__MODULE__, "public step!, project!, adapt!, …")
   end
   ```
   (`Compat.@compat public …` also works; `Compat` is already in the dependency closure but not a direct dep.) Documenter: `checkdocs = :exports` (docs/make.jl:53) must become `:public` (Documenter ≥ 1.8, [INFERENCE: not run]) or public names silently stop being checked; `@autodocs` with `Private=false` must be re-pointed to an explicit `@docs` list.
3. Delete: all aliases flagged "0 uses" now. `Centered`, `CenteredMagnitude`, `Slip` — keep one spelling (the package already uses British `Centred` in docs headings); do not export the other. `NavierStokesSimulation`, `NavierStokesSolver`, `IncompressibleNavierStokesSolver` go with A3.

Effort/benefit: removing the 14 aliases deletes ≈30 lines of definitions/docstrings and 14 export entries; the real gain is the contract, not the line count.

### 3.2 Concrete typing, dispatch, options (modern practice)

- `FlowSimulation{T,M,V,S,W}` (flow_simulation.jl:46) is concrete (`isconcretetype(typeof(flow))` = true, 5 params). `Base.return_types(step!, …)` → `LinearSolverStats{Float64}`; `simulate!(flow, T)` infers to the `NamedTuple`. Stage dispatch is static: the one-/two-velocity choice dispatches on `FaceVelocity` vs `TwoPhaseVelocity` (stepping.jl twins) and single/two-phase on `solver.two_phase::Nothing|TwoPhaseModel` (a type parameter). Good.
- Where it is not concrete: `NavierStokesWorkspace` fields `advection`, `elliptic`, `two_phase` are `Union{Nothing, X{T,M}}` with `X` missing parameters (workspace.jl:1-11) and `solver.pressure_solver::EllipticSolver{T}` (system.jl). One dynamic dispatch per stage per step; cheap relative to a step, and the comments (stages.jl:20-26) show the authors know it. Not worth changing unless profiling says so.
- `IncompressibleNavierStokesSolver` has 21 type parameters and 22 fields (system.jl:62-84) because options are flattened (A9). Storing `options::NavierStokesOptions` collapses 14 of them.
- Public `step!` carries a private kwarg `_transport_geometry_prepared` (single_phase/stepping.jl:73, two_velocity/stepping.jl:~268) used by `simulate!` to pass state. Move into `workspace` (e.g. `workspace.two_phase.geometry_prepared::Bool`) so the public signature is clean.
- Validation style: `ArgumentError` everywhere, zero `@assert` in `src/` (good). The `isa` checks inside option constructors (options.jl:71-93, linear_options.jl:21-26) duplicate what typed keyword arguments would give; they produce nicer messages though. Fine.
- Positional struct constructors bypass validation: `SinglePhase(1.0, 0.1)` is accepted (observed). For invalid values (e.g. a negative density) [INFERENCE] the same path would also be accepted because the validating constructor is a separate outer keyword method. Make the inner constructor private (`struct X; …; global function …`) or validate in an inner constructor. Same for `TwoPhase`, `SurfaceTension`, `MultigridOptions`.
- Symbol vs types: physics/numerics choices are types (`CGFM()`, `CSF()`, schemes), but several choices are still Symbols checked at runtime: `preconditioner ∈ {:auto,:geometric,:tree_geometric,:amg,:ilu,:ichol}`, `MimeticProjection(variant=:joint_smoothed|:joint_face)` (methods.jl:31-39), `InterfaceField(reconstruction_method=:lhf|:lhf_ghf|:wy)` (interface_field.jl:84), `phase_rendering ∈ {:cell,:plic}`, `compatibility=:project` (projection.jl), session `state`/`mode` Symbols (session_types.jl:~55-60). `krylov=:bogus` is not validated at construction and fails later with `FieldError: type Symbol has no field m` (observed). Proposal: singleton types or `Val`-style enums for the first four (S each), a `@enum` for session state.

### 3.3 Visualization architecture (A5, A12, A13, A17, A18)

- **Layering.** `src/visualization/` is 3 177 lines; the Makie ext 1 062. The backend-neutral layer (registry, sampling, frame capture, checkpoints, session) is justified because it lets non-Makie backends reuse frames, but then it should be a *documented* interface. Today the ext imports ≈25 private names (`TwoPhaseAMR._field_entry`, `_render_source_values`, `_phase_visualization_geometry`, `_visualization_topology`, `_check_phase_rendering`, `_frame_phase_geometry`, `_visualization_session_state`, …). What only the ext needs and could live in `ext/`: `AMRCellGeometry`/`AMRFaceGeometry` triangulation (types.jl), `phase_geometry.jl` (483 lines of PLIC triangulation), the `PLICVisualizationGeometry` family. What must stay in core: registry, sampling, frame/checkpoint capture, session state machine.
- **Task lifecycle.** One `@async` publisher per session (session_types.jl:301), with a capacity-one latest-wins mailbox (checkpoints.jl:31-130). Error propagation is sound: publisher failure sets `state=:failed`, notifies `control_condition`, and is re-thrown on the simulation side via `fetch` in `_check_publisher!`. `update_visualization!` takes `session_lock` for the whole update including frame capture and the synchronous viewer redraw (updates.jl:103-107), so GUI callbacks (`request_pause!`) block while a frame is sampled. There is no cancel/close API: `_pause_at_step_boundary!` waits forever if the viewer is closed while paused ([INFERENCE]; no timeout found). `_initialize_session!` closes the mailbox before checking that the old task is done and then throws (session_types.jl:~385-392), leaving the session half-reset.
- **Starvation (A5, measured).** With `@async`, the publisher cannot run while `simulate!` runs on the same thread (no yield point after `notify`). Evidence from `/tmp/audit-api/async.jl`: `async=false` 21/21 frames in-loop; `async=true`: 0 in-loop, 1 after, 20 dropped. The examples hide this with `fetch(Threads.@spawn run_simulation())` (taylor_green.jl:~62; same 4-line pattern in 11 files).
- `visualize(registry; phase_rendering=…)` mutates its input: `registry.phase_rendering = …` (ext:580).
- `VisualizationSession` has 38 fields (session_types.jl:35-70), 12 typed `Any`, mutated from two tasks; a `@kwdef`-style immutable config + small mutable state struct would shrink the surface.

### 3.4 Duplication (A10), quantified

| Duplicate | Locations | Removable lines |
|---|---|---|
| CFL/capillary/gravity bound: `_timestep_bound` (1V) and `_two_velocity_timestep_bound` (2V), line-for-line identical physics | two_phase/stepping.jl:10-31, two_velocity/stepping.jl:40-58 | ≈19 |
| `_transport_substep_other_policy!` repeats the frozen branch with `advect!` in place of `_advect_prepared_frozen!` | two_velocity/stepping.jl:~84-122 | ≈16 (pass the function) |
| `_phase_force_substep!` 1V and 2V differ only in the "add acceleration" helper | two_phase/stepping.jl:130-160, two_velocity/stepping.jl:~222-258 | ≈25 |
| `step!` prologue (mesh check, `_positive_finite_step`, `_finite_time`, `_dimensionless_tolerance`, `_navier_stokes_cycle_limit`) in single-phase and 2V; same again in `simulate!` and `project!(flow)` | single_phase/stepping.jl:68-85, two_velocity/stepping.jl:~290-312, simulate.jl:38-47, flow_simulation.jl:~470 | ≈25 (one `_step_controls(T, solver, dt, time, tol, max_cycles)`) |
| `project!` with/without `geometry` branch repeated in `step!` and `project!(flow)` | single_phase/stepping.jl:~106-128, flow_simulation.jl:~490-505 | ≈20 (`geometry` positional → `nothing`-dispatch in `project!`) |
| Pressure-solver rebuild block duplicated between single-phase and two-phase topology rebuild (differ in the `PoissonOperator` call) | navier_stokes/workspace.jl:78-135, 188-232 | ≈35 |
| `TwoPhaseFlowWorkspace(…)` 11-keyword list in constructor and rebuild | workspace.jl:~385-400, ~245-265 | ≈15 |
| Options flatten/expand round-trip (A9) | options.jl:73-100, workspace.jl:36-76 | ≈90 |
| `_one_velocity_initializer` / `_two_velocity_initializer` share the same two-callable predicate | flow_simulation.jl:~275-310 | ≈8 |
| Visualization twins (A17/A18) | session_types.jl, frame_capture.jl, controls.jl | ≈120 |
| Aliases/legacy path (A3) | §3.1 | ≈60 |

Total ≈ 190–230 lines in the Navier–Stokes/stepping files and ≈ 120 in visualization; ≈ 60 more if the legacy path goes. No behaviour change is intended by any of them.

### 3.5 Aqua summary (`Aqua.test_all(TwoPhaseAMR)`, Julia 1.12.7, `-t 1`, temp env)

`Test Summary: 7 pass, 4 fail, 11 total`

| Check | Result | Detail |
|---|---|---|
| Method ambiguity | **FAIL (1)** | `Diagnostic(name::Symbol, measure::F)` (interface.jl:11) vs `Diagnostic(measure, name::Symbol)` (interface.jl:15): ambiguous for `(Symbol, Symbol)` |
| Unbound type parameters | pass | |
| Undefined exports | pass | |
| Project.toml vs test/Project.toml | pass (vacuous: no `test/Project.toml`) | |
| Stale dependencies | **FAIL** | `SpecialFunctions` never loaded by `src/` (A4). `Random` is also test-only but was not reported, presumably because it is loaded transitively |
| Compat bounds: `julia` | pass | |
| Compat bounds: deps | **FAIL** | no compat for `LinearAlgebra`, `Random`, `SparseArrays` |
| Compat bounds: extras | **FAIL** | no compat for `BenchmarkTools`, `Test` |
| Compat bounds: weakdeps | pass | |
| Piracy | pass | |
| Persistent tasks | pass | |
| (extra) `test_undocumented_names` | **FAIL** | `MimeticProjectionSolver` |

Not covered: the Makie extension (Makie is not in the temp env). JET was deliberately not run.

### 3.6 Project.toml / test hygiene

- `[extras]`/`[targets]` (Project.toml:41-52) carry `Test`, `AlgebraicMultigrid`, `LimitedLDLFactorizations`, `BenchmarkTools`. `benchmark/Project.toml` already has its own environment, so `[targets] benchmark` is redundant.
- `test/makie_visualization.jl` (needs CairoMakie) is not in `runtests.jl`; CI runs it via `julia --project=. -e 'Pkg.add("CairoMakie")'` (CI.yml:166-169), which edits the checked-out Project.toml. A `test/Project.toml` with CairoMakie, run through `Pkg.test`, fixes both.
- `IncompleteLU` is a hard dependency while `AlgebraicMultigrid` and `LimitedLDLFactorizations` are weakdeps with extensions (cycles.jl:285-335): inconsistent but harmless; the `_optional_extension` error message is good.
- `compat Makie = "0.24"` only; WGLMakie/GLMakie are not constrained by the package, fine.

## 4. High-level API review

Path reviewed: README "Quick start", docs/src/getting-started.md, docs/src/api.md, and the misuse runs in `/tmp/audit-api/misuse*.jl` (8×8 or 16×16 blocks, ≤3 steps).

### 4.1 What works well

- Typed composition `FlowSimulation(mesh, physics; velocity_model, numerics, adaptivity, boundary, diagnostics, dt)` is easy to read and the README quick start runs as written. Constructors validate eagerly, with specific `ArgumentError`s (observed, verbatim): `density must be a (liquid, gas) tuple`, `phase dynamic viscosities must both be positive or both zero`, `gravity must be a two-tuple`, `OneVelocityModel requires one shared velocity initializer`, `interface_transport must be FrozenInterfaceFluxes() or ProjectedLiquidVelocity()`, `mimetic variant must be :joint_smoothed or :joint_face`, `diagnostic :kinetic_energy is already registered`, `unknown diagnostic :nope`, `gravity requires nonperiodic boundaries in its direction`, `FlowSimulation owns the initial time; set flow.time before resuming`.
- `(liquid, gas)` ordering is stated in every docstring and error message. Float32 end to end works (`eltype(f.pressure) == Float32` verified). Invalid velocity initializers produce a message that lists all accepted forms.
- `simulate!` returns `(time, steps, pressure::LinearSolverStats, mesh_updates, total_steps, total_mesh_updates)`; continuing, `flow.time = 0.0` and re-running both work (observed).

### 4.2 Observed behaviour on misuse (all single-threaded, verbatim class and gist)

| Misuse | Behaviour | Verdict |
|---|---|---|
| `FlowSimulation(mesh, phys; velcity=…)` / `boundry=` | `MethodError: no method matching FlowSimulation(::BlockTreeMesh{Float64, TwoPhaseAMR.RootGrid{…RegionTrees.Cell{…}}}, ::SinglePhase; velcity::…)` (~600 chars of type noise) | Rejected, but the message is dominated by type names; no did-you-mean |
| `BlockTreeMesh(blocksize=…)` | MethodError listing the valid kwargs (`got unsupported keyword argument "blocksize"`) | OK (Julia ≥ 1.10 message) |
| `LinearSolverOptions(preconditoner=:auto)` | MethodError, valid kwargs listed | OK |
| `simulate!(flow, t; visualisation=…)` | MethodError citing the **low-level** `simulate!(::FaceVelocity, ::CellField, ::Incompressible…)` method, not the `FlowSimulation` one | Confusing: forwards `options...` blindly (flow_simulation.jl:~420) |
| `FlowSimulation(mesh, SinglePhase; velocity_model=TwoVelocityModel())` | raw `MethodError … velocity_model::TwoVelocityModel{…}` | Rejected, but no explanation that `SinglePhase` has no velocity model |
| `TwoVelocityModel(interface_transport=CSF())` | `ArgumentError: interface_transport must be FrozenInterfaceFluxes() or ProjectedLiquidVelocity()` | Good |
| `TwoPhase` + `SurfaceTension(method=CSF())` + `TwoVelocityModel()` | runs (`CSF` valid in both models, as documented) | OK |
| `OneVelocityModel` + `velocity=(vel, vel)` | `ArgumentError: OneVelocityModel requires one shared velocity initializer` | Good |
| `boundary=:noslip` | `ArgumentError: boundary must be NoSlip() or FreeSlip()` | Message omits `VelocityBoundaryConditions` |
| `T=Float16` | `ArgumentError: FlowSimulation requires T=Float32 or T=Float64` | Good |
| `krylov=:bogus` | `FieldError: type Symbol has no field m` at construction | Bad: leaks internals |
| `preconditioner=:bogus` | `ArgumentError: preconditioner must be :geometric, :amg, :ilu, :ichol, :tree_geometric …` (omits `:auto`) | Good |
| `SinglePhase(density=1.0)` | `UndefKeywordError: dynamic_viscosity` | Good |
| `SinglePhase(density=1, dynamic_viscosity=0)` | `ArgumentError: dynamic_viscosity must be finite and strictly positive` | Good (inviscid single phase unsupported; two-phase allows both-zero) |
| `SinglePhase(1.0, 0.1)` | accepted (positional bypass) | See §3.2 |
| fixed `dt=1` in two-phase | `ArgumentError: time step exceeds two-phase transport or force stability bound` (no value for the bound or a suggested `dt`) | Good intent; add numbers |
| fixed `dt=1`, single phase, CFL≈2.5 | **no error, no warning**; three steps complete | See A6 |
| `simulate!(flow, t_past)` | `ArgumentError: final_time must not precede the initial time` | Good |
| fresh `simulate!(flow, flow.time)` (0 steps) | returns `pressure.converged == false`, residuals `Inf` (zero-initialised stats); README asserts `result.pressure.converged` | Trap for the README pattern when no step runs; return `nothing`/`missing` stats or `converged=true` |
| legacy `FlowSimulation(mesh; density=…)` | no warning at default flags (A2) | Bug against docs |
| `print_simulation_summary(io, flow)` | `MethodError` — only stdout form exists | Add `io` method |
| `add_diagnostic!(flow, :x, (a,b,c,d,e)->1)` (legacy 5-arg form) | `MethodError` at the first evaluation, not at registration | Check arity with `applicable(measure, state)` at registration |
| diagnostic function throws | error surfaces inside `simulate!` (`ErrorException: boom`), unwrapped, no mention of which diagnostic | Wrap with the diagnostic name |

### 4.3 Consistency

- **Conventions that are consistent:** per-phase values are `(liquid, gas)` tuples; velocity callables are `(x, y, time, axis)` while pressure/interface callables are `(x, y)`; scalar `dynamic_viscosity` is accepted for two phases but `density` must be a tuple (asymmetric: the docstring says scalar viscosity only).
- **Unit/dimension story:** no units appear in signatures; `register_field!` has a `units` label only. Consistent with "dimensional or nondimensional" (visualization.md), but `PotentialEnergy` says "potential is zero at the coordinate origin" and single-phase "has no gravity model" (interface.jl) while `SinglePhase` has no `gravity` argument at all; a single-phase user cannot apply gravity except through `force=`.
- **Unsafe defaults:** `BlockTreeMesh` is fully periodic by default (`periodic=(true,true)`, mesh.jl:841) while `FlowSimulation(boundary=NoSlip())` is the default boundary. A user who forgets `periodic=(false,false)` gets a periodic box with the wall condition silently inactive [INFERENCE from lid_driven_cavity.jl:19 passing `periodic=(false,false)`; the baseline TG run is periodic]. Gravity on a periodic mesh is rejected with a good message; viscosity/inflow setups are not. Suggest requiring `periodic` explicitly or logging `periodic` in `print_simulation_summary` (already prints mesh/model).
- **Automatic `dt`/CFL:** `dt=nothing` selects a CFL-limited step (`cfl=0.45` default, simulate.jl:44); `dt` given is a fixed step and is only checked in two-phase (A6). `default_dt` is stored but not documented as "the fixed step reused by every subsequent `simulate!`".
- **Names:** British/American pairs `Centred/Centered`, `CentredMagnitude/CenteredMagnitude` (A1); `solver` (field, argument names throughout) vs `system` (docs); `FlowSimulation` vs `NavierStokesSimulation` alias; `diagnostic_history` returns a detached copy, `DiagnosticCollection` is exported but unused outside.
- **Return values:** `simulate!` returns a `NamedTuple`; the low-level `simulate!` returns the same without `total_*`. The pressure stats type differs between formulations (`LinearSolverStats` vs `CMDGFMProjectionStats`/`MimeticProjectionStats`) and summary code probes with `hasproperty` (flow_diagnostics.jl:~70-85); a common `stats` interface (`converged`, `iterations`, `final_residual`) removes those probes.
- **Fields after a run:** `flow.velocity` (a `FaceVelocity` or a `(liquid, gas)` tuple, NaN-masked for absent phase), `flow.pressure`, `flow.interface`, `flow.time`, `flow.steps`; `propertynames(flow)` lists them and the fields are documented in getting-started.md:~35. No accessor functions (`velocity(flow)`), no cell-centred sampling helpers other than the visualization registry.
- **Restart/continue/checkpoint:** continue in memory works; there is no on-disk checkpoint, no output writer, no snapshot export (A8). "Checkpoints" in `VisualizationSession` are in-memory frames.

### 4.4 Extensibility

| Extension point | State | Gap |
|---|---|---|
| Custom body force | `force=(x,y,time,axis)` kwarg of `simulate!`/`step!`, sampled at the time midpoint (navier-stokes.md:36) | Not part of `FlowSimulation`, not in README/getting-started/how-to; must be re-passed on each `simulate!` call |
| Custom diagnostic | `Diagnostic(name, f)` / `add_diagnostic!(flow, name, f)` with `f(state::SimulationState)`; or subtype `AbstractDiagnostic` + `diagnostic_name`/`diagnostic_value` | Subtype route is undocumented (names appear in no doc); `state` exposes `solver.mesh` internals, no `mesh(state)` helper |
| Custom BC | closed `Union{NoSlip,FreeSlip,Inflow,Outflow,ZeroGradientVelocity}` (staggered_diffusion.jl:62-74); `Inflow(f)` takes a callable | No user-extensible BC; fine if documented as closed |
| Custom refinement criterion | `AbstractRefinementIndicator`, `Adaptivity(criteria=…)`, labels `:refine/:coarsen/:keep` | The required methods are internal (`_adaptive_refinement_labels!`, indicators.jl); no documented contract or example |
| Custom schedule | `AbstractDiagnosticSchedule` subtype | Undocumented |
| Custom visualization backend | `AbstractVisualization` + `update_visualization!`/`_bind_visualization_session!`… | Three of the hooks are underscore-private (interface.jl:29-32) |

Proposal: one "Extending" doc page with a tested, ≤15-line example per row, and promote the underscore hooks that backends must implement to documented names. Effort M.

### 4.5 Comparison with other Julia packages (from memory of their public APIs; not re-verified in this session)

| Package idiom | Adopt? | Concretely |
|---|---|---|
| **Oceananigans** `Simulation(model; Δt, stop_time)` + `run!`; `add_callback!(sim, f, IterationInterval(n)/TimeInterval(t))`; `OutputWriter`s; `Checkpointer` + `pickup` | **Yes, partially.** `FlowSimulation` already is the `Simulation`; `EveryNSteps`/`EveryTimeInterval` already are the schedules. Add `add_callback!(flow, f, schedule)` (generalising `add_diagnostic!`, which only records a value) and a `write_checkpoint`/restart pair plus one output writer. Add an adaptive-`dt` wizard only if `cfl=` is insufficient (it is not). | M each |
| **Oceananigans** model/simulation split | **Already effectively there** (`NavierStokesSystem`+workspace = model). The naming gap (A11) is the only fix. | S |
| **Trixi.jl** `semidiscretize` + OrdinaryDiffEq `solve`, callbacks `AnalysisCallback`, `SaveSolutionCallback`, `SaveRestartCallback`, `AMRCallback`, `StepsizeCallback` | **No for the ODE interface.** Fractional-step with an embedded pressure projection, per-step topology changes and Krylov/multigrid caches do not map to a state-vector RHS. **Yes for callback vocabulary and restart-file concept.** | — |
| **WaterLily.jl** `Simulation(dims, u_BC, L; ν, body)`, `sim_step!(sim; remeasure)`, `sim_time`, small public surface | **Yes for the manual loop.** `step!(flow, dt)` / `sim_time`-style accessors (A7). Not the macro-heavy generic-dimension internals. | S |

## 5. Nits

- README/docs: `visualize` as a keyword name in every example (`run_example(; visualize=false)`) shadows the exported function, which forces `TwoPhaseAMR.visualize(...)` qualification (taylor_green.jl:~46). Rename the keyword `show`/`display`.
- `TwoPhase.density::R <: Tuple` stores `(1000, 1.0)` mixed-type tuples as given; conversion to `T` happens later (system.jl). Convert at construction.
- `FlowSimulation` docstring still mentions `NavierStokesSolver`-era details (`flow_simulation.jl:34-38`) alongside the "compatibility path"; fold into one deprecation note.
- `docs/src/api.md` "Types and constants" autodocs lists every exported type, including the 14 aliases; this undoes the "compact standard API" table above it.
- `_enforce_current_velocity_boundary!` and `_enforce_phase_velocity_boundary!` (single_phase/stepping.jl:1-12, two_velocity/stepping.jl:11-31) differ only in a loop over phases; `_copy_cell_field!`/`_copy_velocity!` live in two files (independent_step.jl:1, stages.jl:3).
- `simulate!` result field `pressure` is named for the stats of the projection; `projection` would be clearer, and the two-velocity stats have no `.cycles`.
- `Base.getproperty(::FlowSimulation, …)` computes `:interface`/`:diffusion_operator` virtually; `Base.propertynames` includes them, fine, but `Base.show` omits the model kind (one/two velocity, single/two phase).
- `LimitedLDLFactorizations`/`AlgebraicMultigrid` extension error text is good; add the same for `Makie` in `visualize` (currently a bare `MethodError` when no backend is loaded: `visualize` is `function visualize end`, interface.jl:10).
- Hand-maintained `src/TwoPhaseAMR.jl` include order has comments only for groups; `simulation/` is included before `visualization/` but `navier_stokes/options.jl` is included mid-way (line 66) while the rest of `navier_stokes/` is last: split `options.jl` into its own folder or move it.
