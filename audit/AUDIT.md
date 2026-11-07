# TwoPhaseAMR.jl — consolidated audit (2026-10-20, `main` @ 157448c)

**Work plan:** the phased, PR-level execution plan derived from this audit is in [WORKPLAN.md](WORKPLAN.md).

Seven read-only audit slices; each has a detailed report with file:line evidence and finding IDs used below:

| Slice | Report | IDs |
|---|---|---|
| Core: mesh, operators, solvers, threading | [audit-core.md](audit-core.md) | C1–C27 |
| Interface / two-phase / projections | [audit-twophase.md](audit-twophase.md) | F1–F24 |
| High-level API, Navier–Stokes orchestration, visualization, extension, Aqua | [audit-api.md](audit-api.md) | A1–A19 |
| Efficiency (measured, Ryzen 9 5950X, Julia 1.12.7, ≤4 threads, ≤16k cells) | [audit-perf.md](audit-perf.md) | P-F1–F10 |
| Technical reference vs implementation, thesis coverage | [audit-techref.md](audit-techref.md) | T-F1–F16, §5 |
| User docs, visual examples | [audit-docs.md](audit-docs.md) | D1–D11, E1–E11, G1 |
| Tests, CI, docs deployment, repo hygiene | [audit-hygiene.md](audit-hygiene.md) | CI-, DOC-, TST-, HYG- |
| `threading` branch (17 commits not on `main`) | [audit-threading.md](audit-threading.md) | §3–§6 |

Spot-checked by the orchestrator against the code: C8/C11 (dead `_residual_norms!`, identical sparse kernels), F3/F5 (`geometry::Any`, `assemble_stiffness` always `true`), T-F1 (`navier-stokes.md:250` "incremental" vs non-incremental `project!`), CI-03 (`cycles.jl:262-268` `applicable` guard), P-F3 (`geometric_preconditioner.jl:679` stops coarsening at `active_level > 0`), D1 (getting-started AMR snippet uses `level = 2` on a `max_level = 0` mesh), CI billing annotation and Pages 404 (via `gh`).

Corrections between slices: the "≈1 KB/cell/step" single-phase allocation (A19) did **not** reproduce (periodic TG: 4.3 KB/step constant; non-periodic: 105–208 KB/step from boundary-convection boxing; two-phase: 1.7 KB/cell, mostly coarse LU storage). `enforce_velocity_boundary!` 43.8 KB/call (C13) applies only to the `Inflow/Outflow/ZeroGradientVelocity` path.

---

## 0. Decisions for the owner

| # | Decision | Options | Recommendation |
|---|---|---|---|
| O1 | CI billing and Pages | A: make repo public (free Actions minutes, free Pages). B: stay private, fix payment method (Free 2 000 min/mo; Pages requires Pro/Team and the site is public anyway). C: B + self-hosted runner for heavy tests. | A if the history is publishable (remove the 4.7 GB leftovers first); else B with Pro. Either way adopt the tiered CI (§2.7), which fits ≈1 100 min/mo. |
| O2 | Version / deprecation policy | `Project.toml` is `0.1.0-DEV`, docs promise compatibility "throughout 0.2.x" (migration.md, api-decisions.md). | Treat as pre-release: delete the 13 aliases, legacy `FlowSimulation(mesh; density=…)` path, `interface_method` shims, migration/api-decisions pages (A2, A3, F21, C9). Start CHANGELOG at the first tag. |
| O3 | Julia 1.10 support | Fix the CHOLMOD `ldiv!` path (CI-03, S) or raise `julia = "1.11"`. | Raising to 1.11 also gives the native `public` keyword (A1); otherwise fix and keep 1.10 via `Compat.@compat public`. |
| O4 | Failing integration assertions | KH/RT (`two_velocity_instabilities.jl:180,195`), capillary dt-monotonicity (`two_phase_waves.jl:341`), cylinder order/Strouhal (`single_phase_cylinder.jl`; no shedding by t=40) fail deterministically. | Numerics triage: these assert properties that do not hold when spatial error dominates or shedding never starts. Record observed rates; rewrite assertions; move to heavy tier until then (CI-04/05/06). |
| O5 | `GFM(conservative=false)` | ~45 lines, only construction-tested at σ=0 (F8). | Delete, or keep with a numerical test if needed for thesis comparisons. |
| O6 | Base-mesh coarsening in tree multigrid | Coarsening stops at root level; a flat base mesh becomes one direct LU (P-F3). | Implement base-level agglomeration (largest single efficiency gain for typical setups). |

---

## 1. Priority roadmap

### P0 — broken or misleading now (all S unless noted)

| Item | Ref |
|---|---|
| CI cannot run (billing) and has never been green (67/67 failed; PRs merged red); no `timeout-minutes`, no `concurrency` (cylinder job 152 min) | CI-01, 02, 07, 08 |
| Julia 1.10: `ldiv!(::CHOLMOD.Factor, ::Vector)` MethodError via `ExternalPreconditioner` (`cycles.jl:257-269`) | CI-03 |
| Docs site not served (Pages off, `/dev/` never deployed); README badge points to 404 | DOC-01 |
| Getting-started AMR snippet fails (`level must be an integer within mesh limits`) | D1 |
| `lid_driven_cavity` default run is one step of dt = 15 (infinite CFL bound at rest); `simulate!` should cap the first step | E1 |
| `lgpi` at `max_level=2`: `diffusion matrix and stiffness sparsity differ` (seen once; repro needed) | E2 |
| Single-phase accepts `dt` above the stability bound silently; two-phase error message has no numbers | A6 |
| Deprecation warnings invisible at default flags (`Base.depwarn`) | A2 |
| `asynchronous=true` visualization delivers 0 frames during `simulate!` (publisher never yields); 11 examples carry a `fetch(Threads.@spawn …)` workaround | A5 |
| Docs call the projection "incremental"; it is non-incremental. Inflow/Outflow BCs exist but docs say all boundaries are impermeable | T-F1, T-F5 |
| `advect!` does not reject aliasing with `workspace.rates` | C25 |
| Aqua: stale `SpecialFunctions`, test-only `Random` in `[deps]`, missing stdlib/extras `[compat]`, `Diagnostic(::Symbol,::Symbol)` ambiguity, `MimeticProjectionSolver` undocumented | A4, A15, A16 |
| 4.7 GB untracked `benchmark/scaling/results` (+ `research/` build artefacts, unreferenced `docs/src/assets/`): one `git add -A` stages 12 554 files | HYG-01..04, D4 |
| **First-run setup time is far too long**: a first run of an example (e.g. `stokes3_wave.jl`) takes 1–2 minutes before the first step (owner report; measured 25.6 s for the README Taylor–Green and 69.5 s for two-phase, `audit-perf.md`; 12–65 s per example, `audit-docs.md` E10). Investigate before optimizing: split time into package load/precompile (`@time_imports`, `--trace-compile`), method compilation per phase (`FlowSimulation` construction, initial AMR passes, first `step!`, visualization) and real setup work (mesh/operator/multigrid construction); check for invalidations (SnoopCompile) and overspecialization (21-parameter solver struct, `NamedTuple` option round-trip, `Val`/kwarg-heavy constructors). Then fix via a PrecompileTools workload (E-1; prototype 25.6 s → 0.8 s) and by reducing type-parameter explosion. Target: first `simulate!` step of any example < 10 s | E-1, A9 |

### P1 — efficiency (measured; see `audit-perf.md`)

| # | Finding | Measurement | Proposal | Expected gain | Effort |
|---|---|---|---|---|---|
| E-1 | Compile latency (see P0 "first-run setup time") | README TG 25.6 s, two-phase 69.5 s to first result; examples 1–2 min (owner) | Profile first (P0 item), then PrecompileTools workload (prototype: 0.8 s / 4.0 s). Natural workloads (`simulate!`, `stable_timestep(flow)`, AMR rebuild) SIGSEGV Julia 1.12.7 LLVM during pkgimage emission → try 1.13, reduce to MWE, report upstream; ship the partial workload meanwhile | 30–60× faster first run | M |
| E-2 | Tree GMG never coarsens the base mesh | flat 128² TG: coarse UMFPACK = 49.6 % of step; same problem as 1 root + 3 levels: 18.4 → 11.5 ms (1.6×); two-phase cost ∝ N^1.5 | Coarsen root blocks (2×2 agglomeration/rediscretization) down to `coarse_size`; interim: build base meshes hierarchically | 1.6× (TG 128²), grows with N | M |
| E-3 | Mimetic projection rebuilds everything per step | 106 ms vs CMDGFM 21 ms; 135 MB/step; ~323 MINRES its | Cache incidence/operator/slices like `CMDGFMPressureCache`; allocate MINRES once; preconditioner `ldiv!` instead of `\`; patch-local scratch in joint fit | ≥3× on mimetic | M |
| E-4 | Per-step coarse refactorization | 1-velocity: diffusion-hierarchy refresh 29.6 % of step, 4.0 of 5.5 MB/step is `lu` storage | `cholesky!`/`lu!` in place (2.9–4.1× / 1.4× in microbenchmarks); reuse symbolic factorization | ~20 % of two-phase step | S–M |
| E-5 | `dt` exact `==` compare triggers full viscous rebuild | +105 ms, 85 MB on truncated last step (16k cells) | Compare with tolerance or refresh numerically only | removes spikes | S |
| E-6 | `reconstruct!` | 19–23 % of two-phase step; ~455 B/mixed cell allocations | Fixed-size returns, stack scratch (F22) | 5–10 % | M |
| E-7 | Poor thread scaling | 4 threads: TG 1.17× (flat), 1.53× (hierarchical); CMDGFM 1.16× | Parallelize serial streaming loops (C20, F11 `_phase_face_average!`, `diffusion!` gather/scatter, residual/RHS loops); E-2 removes the serial UMFPACK bottleneck | 2–3× at 4 threads [INFERENCE] | M |
| E-8 | Boundary convection boxing (`Ref{Any}`) | 105–208 KB/step → 9.8 KB with a function barrier (prototype) | Concrete cache types (C13, F17) | allocation-free non-periodic steps | S |
| E-9 | Topology rebuild after AMR | `EllipticOperator` 218 ms/198 MB, TreeGMG 183 ms/178 MB at 26k cells (≈16× a solve) | Replace `Dict{Tuple{Int,Int},Int}` lookups by CSR search; reuse patterns | 2–5× rebuild [INFERENCE] | M |
| E-10 | Scalars in VOF advection allocate 4 fields per scalar per step | static | Workspace buffers (F12) | O(N) allocs removed | S |

### P1 — code bloat and modern practice

Estimated removable source: ≈ 1 100–1 400 lines (≈ 4 % of `src/`), overlap between slices removed. Main items:

| Item | Lines | Ref |
|---|---|---|
| One tree-coarsening plan shared by the three hierarchy builders (scalar GMG, staggered diffusion, variable diffusion), incl. the duplicated in-loop suffix reuse | ≈ 350–450 | C1, F4 |
| One Krylov-solve-with-stats driver (3 copies), trait-table `_run_krylov!` (4 wrappers), one Jacobi-weight helper (4 inconsistent copies) | ≈ 150 | C2, C3, C4 |
| Shared projection layer for CMDGFM + mimetic (incidence, gauge partition, gradient core, defects, `project!` skeleton); split 1.3k-line files by variant | ≈ 170 | F2, F16 |
| Options round-trip (`NavierStokesOptions` → 14 flat fields → NamedTuple) → store `options` | ≈ 90 | A9 |
| Stepping duplication single / one-velocity / two-velocity (timestep bounds, force substeps, prologue, rebuild blocks) | ≈ 190–230 | A10, F10 |
| Visualization: 24 manual lock/try/finally → `@lock`; sampling/due twins | ≈ 120 | A17, A18 |
| Dead/test-only code: `_residual_norms!`, `_update_diagonal_face_coefficient!`, `assemble_stiffness=false` branch (~100), CMDGFM reference sparse-gradient path (move to `test/`), aliases, `AbstractEllipticOperator` | ≈ 200 | C8–C10, F5, F15, F21 |
| Smaller duplicates: sparse-mul kernels, `_cell_index` ×2, CSR offsets ×6, clip helpers, unit-square literal ×6, BFS ×2, restriction kernels | ≈ 150 | C11, C12, C21–C23, F14 |

Modern-practice items (beyond deletions):

- **Type stability by construction**: `InterfaceField{T,M,G}` instead of `geometry::Any` (removes ~20 hand-written function barriers, F3); concrete caches instead of `Ref{Any}`/`Vector{Any}`/`Dict{Any}` (C13, C17, F17); the two-velocity workspace typed by the velocity model instead of a lazy `Ref{Any}`.
- **Dispatch instead of runtime branching**: transport policy (`isa FrozenInterfaceFluxes`, F6), surface-tension method (`GFM.conservative::Bool`, `CGFM()` is a function, F7), mimetic variant and preconditioner Symbols (F16, F20), scheme ladders duplicating `_canonical_staggered_scheme` (C26, −120 lines). Validate Symbol options at construction (`krylov=:bogus` currently surfaces as `FieldError`).
- **Validation in inner constructors** (`SinglePhase(1.0, 0.1)` bypasses checks); single `MultigridOptions` kwarg instead of 8 copies of defaults (C6); `TwoPhaseModel` built from the validated `TwoPhase` without re-validation and with current keyword names in errors (F9).
- **Structs**: `StiffnessRefresh{T}` instead of 6 positionally splatted fields; `matrix::Nothing` instead of a 0×0 sentinel (C7); rename the real type to `NavierStokesSystem` and field `flow.solver` → `flow.system` (A11).
- **Threading**: one or two launch primitives instead of five, fix wrong docstring, document that user callbacks run inside Polyester workers, add a serial-vs-threaded equivalence test (C15, C16).
- **Layout**: directories matching include blocks (`interface/`, `embedded/`, `vof/`, `two_phase/projection/`, `two_phase/diffusion/`, `visualization/plic_geometry.jl`); split `geometric_preconditioner.jl` (998 lines, 6 topics); keep one module (as `api-decisions.md` states) (F13, C24).

### P1 — high-level API

| Gap | Proposal | Ref |
|---|---|---|
| 231 exports, ~45 needed, 13 aliases, `simulation_state` means two things | Export the standard surface (~45 names listed in A1); mark the rest `public`; rename session `simulation_state` → `session_state` | A1 |
| No `step!(flow, dt)`; manual loops cannot adapt the mesh | Add it (bookkeeping, AMR, diagnostics); implement `simulate!` as a loop over it | A7 |
| No checkpoint/restart/output | `write_checkpoint`/restart pair + an output-writer callback; `add_callback!(flow, f, schedule)` generalizing `add_diagnostic!` (Oceananigans-style vocabulary; not the OrdinaryDiffEq interface) | A8, §4.5 of audit-api |
| `force=` not stored in `FlowSimulation` (dropped on a second `simulate!`) | Store it in the simulation | A14 |
| Undocumented extension contracts (refinement indicators, diagnostics subtypes, schedules, visualization backends with `_`-prefixed hooks) | One "Extending" page with a tested ≤15-line example per extension point | A14, A12 |
| Misuse messages: typo'd `FlowSimulation` keyword → 600-char MethodError; `simulate!` typo cites the low-level method; `TwoVelocityModel` with `SinglePhase` unexplained; 0-step `simulate!` returns `converged=false` | Keyword whitelist with did-you-mean; explicit model-compatibility errors | §4.2 of audit-api |
| Unsafe default: `BlockTreeMesh` periodic by default while default boundary is `NoSlip()` | Require `periodic` explicitly or print it in the summary | §4.3 of audit-api |
| Common stats interface across `LinearSolverStats`/`CMDGFMProjectionStats`/`MimeticProjectionStats` (`hasproperty` probes today) | `converged`, `iterations`, `final_residual` accessors | §4.3 of audit-api |

### P1 — tests

- Group registry (`test/groups.jl`), selected by `ARGS`; CI matrix uses group names (TST-01).
- `test/Project.toml` with Aqua (+ JET non-blocking later); `test/makie/Project.toml` instead of `Pkg.add("CairoMakie")` at CI time (TST-02, TST-03, TST-13).
- Delete wording/implementation pins: `test/diagnostics.jl:89-94`, `test/navier_stokes/high_level_api.jl:37`, `test/two_phase/two_velocity/projection/cmdgfm_projection.jl:431-437`, `test/navier_stokes/two_phase_navier_stokes.jl:20` (TST-04).
- 130 uses of private `_get` → public `getindex(field, ::CellRef)` (TST-05); migrate tests off the deprecated constructor (TST-07); order-dependent extension check in `test/solvers.jl:4-16` (TST-08).
- Examples: AMR (`max_level ≥ 1`) and multi-step smoke runs (would catch E1, E2) (E7); capillary bound and Inflow/Outflow have no numerical test (T-F5, T-F6).

---

## 2. Target documentation, technical reference, examples, CI

### 2.1 Light API documentation (Documenter, GitHub Pages)

Page tree (Diataxis; detailed in `audit-docs.md` §3.3):

```
Home · Getting started (Installation, Quick start, Choosing a model)
Tutorials (Single-phase, Two-phase [capillary wave vs Prosperetti], Adaptive refinement)
How-to (Diagnostics, Numerics & solvers, Live visualization, Static figures, Threads & performance, Extending)
Examples gallery (generated from examples/NN_name/README.md)
Method overview (1 page → PDF chapters) · Scope and limitations
Reference: api/index + 8 grouped pages (mesh-amr, fields-operators, physics-models, navier-stokes,
           interface-geometry, diagnostics, visualization, advanced)
Help: FAQ/troubleshooting, Citing (CITATION.cff, thesis DOI), Changelog
Developer guide: architecture, testing, contributing
```

- Remove from the site: the seven explanation pages (move to the PDF), `migration.md`, `api-decisions.md`, `literature-todos.md` (becomes the PDF's coverage/deviation appendix), `how-to.md` (dissolved). README ≤ 90 lines; one owner per fact (two-phase snippet exists 4×, option matrix 3×, example table 3×) (D7, D8).
- Executed code: only 8 of ~74 blocks run today; make Quick start, tutorials and how-tos `@example` (D2).
- API pages: replace the two `@autodocs` dumps (117 KiB and 102 KiB, warning at 100, error at 200) by `@docs` lists under `## [Title](@id api-…)` headings; function docstring anchors contain signatures (`TwoPhaseAMR.simulate!-Union{…}`) and break on any edit, so the PDF links only to explicit `@id` anchors (D5, T §5.4). `checkdocs = :public` after the export split (D6).
- Tutorial uses internal fields (`flow.solver.diffusion.x.local_to_global`): add a public error helper (D9).

### 2.2 LaTeX technical reference (thesis style)

Main reference: Remmerswaal (2023), *Numerical Modelling of Variability in Liquid Impacts*, PhD thesis, University of Groningen, https://doi.org/10.33612/diss.562133319. Full proposal in `audit-techref.md` §5.

- **Purpose**: (1) method overview in thesis notation; (2) fill what the thesis does not cover; (3) bridge equations ↔ implementation with stable links into the API pages.
- **Coverage map** (23 components; `audit-techref.md` §5.1). Not in the thesis at all, so the tech ref must derive them: block quadtree/2:1/halo/face schedules; level-set and polygon initialisation; tree geometric multigrid + Krylov tolerances; conservative AMR transfer with divergence-preserving prolongation (refinement only) and phase/PLIC transfer; threading. Partly covered: discrete setting on AMR (hanging faces, dual volumes, vertex-patch `K_v`), VOF transport, momentum flux interpolants, cut cells, AMR criteria, mimetic gradient.
- **Deviations from the thesis that must be stated** (`\begin{deviation}` environment + appendix): CMDGFM β = (1−s)/(R−s) (thesis: 3x²−2x³; code keeps the weak diagonal dominance, verified algebraically); mimetic projection as a (p, ξ, λ) KKT system with local constraint fits (thesis: potentials ψ_η, ψ_τ); directional-split WY + LW sweeps (thesis: unsplit donating regions/flux interpolants); single-phase CN explicit half at uⁿ; σ applied as acceleration on the predictor; gravity time-step bound √2 stricter; LHF quality polynomial corrected; not implemented: parabolic/MOF reconstruction, centroid advection, AST surface tension, instability AMR criterion, 3-D, compressibility.
- **Missing content today** (whichever medium): equations for CSF/CGFM/GFM, gravity, curvature chain and face-curvature averaging, cut-cell operators, default mimetic variant, stability limits (advective CFL factor, WY bounds, capillary, gravity; explicit `dt` throws), formal temporal order (single-phase order flagged for derivation-checker), Inflow/Outflow semantics, multigrid smoother/transfer, `pre_sweeps == post_sweeps` requirement for CG.
- **Notation bridge** thesis macro → code symbol (`\cvset{F}` → `mesh.faces`, `\volfrac` → `InterfaceField.fraction`, `\jumpOper` → `cmdgfm_jump_interpolant`, `\gfmgrad` → `cmdgfm_gradient!`/`mimetic_gradient!`, …; `audit-techref.md` §5.2). Rule: no new symbol where a thesis macro exists; AMR additions go in `techref-notation.sty`.
- **Outline** (`docs/techref/`): 01 overview & algorithm box, 02 block-tree AMR, 03 discrete setting on AMR, 04 interface, 05 VOF transport, 06 momentum transport, 07 cut cells, 08 forces, 09 viscosity, 10 projections (one-velocity, CMDGFM, mimetic, BCs), 11 linear solvers, 12 AMR criteria & transfer, 13 stability & accuracy, 14 verification (generated tables, claim → test index); appendices: deviations, notation, API index. Each section ends with an `implnote` box listing `\apitype{}`/`\apifun{}` links.
- **Macros**: vendor a derived `techref-notation.sty` from `preamble.tex:342-617` + `newcommands.tex`, not the whole thesis preamble.
- **Verification tables** must be generated by scripts (the convergence tables in `navier-stokes.md` and `poisson.md` exist in no test or script today, T-F4).

### 2.3 Links between the two

- PDF → API: `\apibase` written by CI per build (`…/dev` on main, `…/vX.Y.Z` on tags; never hard-code `stable` in a PDF); targets are explicit `@id` anchors; a CI script checks every `\apitype/\apifun` target exists in the built HTML.
- API → PDF: PDF published inside the site (`docs/src/assets/techref.pdf` copied before `makedocs`, so each docs version carries its own PDF); deep links via hyperref named destinations (`#nameddest=sec:…`) or page numbers extracted from `.aux` into a `techref_sections.toml` that docstrings reference; CI fails on missing labels.

### 2.4 Examples (owner's layout)

```
examples/
  Project.toml                 TwoPhaseAMR (dev), GLMakie, CairoMakie
  make_all_figures.jl          --only 06,07  --size docs|default
  shared/figures.jl            (later: snapshot_figure from the Makie ext)
  01_taylor_green/  taylor_green.jl  README.md  make_figures.jl  assets/
  02_lid_driven_cavity/ 03_decaying_turbulence/ 04_cylinder_flow/ 05_vortex_reverse/
  06_capillary_wave/ (+ prosperetti_amplitude.jl moved from test/integration)
  07_interfacial_instability/ 08_sessile_droplet/ 09_sloped_sessile_droplet/
  10_stokes3_wave/ 11_lgpi/ 12_manufactured_poisson/
```

- Scripts unchanged and concise (`run_example` API kept). Each `README.md` opens with an **At a glance** table: Physics · Algorithms · Worth varying (keyword = default → effect) · Validated against · Cost (default and docs size, plus first-call compile). Filled examples for 01, 06, 07 in `audit-docs.md` §4.4. Then Setup (→ PDF chapter), Run, Results (figures), Variations, Files.
- Figures: headless CairoMakie works today (PoC PNG/GIF in `/tmp/audit-docs/poc/`, incl. capillary amplitude vs Prosperetti). Add a small static API to the Makie extension (`snapshot_figure`, `plot_diagnostics!`, `record_checkpoints`) so each `make_figures.jl` is ~10 lines (G1). Whole gallery ≈ 5–7 min on one thread → commit assets, regenerate manually or in a scheduled job; docs build copies `examples/*/README.md` + `assets/` into `docs/src/examples/` with link rewriting.
- Fix first: E1 (lid cavity single step), E2 (lgpi), internal `_`-functions in `cylinder_flow.jl`/`manufactured_poisson.jl` (add a public drag diagnostic), GLMakie only works through the global environment (E3). Optional: move the identical ~30-line live-viewer tail into `examples/shared/` (E4); the `visualize` keyword shadows the exported function (E5).

### 2.5 Citing and project metadata

`CITATION.cff` with the thesis as preferred citation and the arXiv papers (2209.14934, 2210.08916) as references (sketch in `audit-hygiene.md` §4.6); `CHANGELOG.md` from the first tag; `TODO_LATER.md` → GitHub issues.

### 2.6 Repository hygiene

`.git/info/exclude` (or move out) `benchmark/scaling/`, `research/`, unreferenced `docs/src/assets/`; delete empty `.agents/`, `.codex/`; parameterise the hard-coded `/home/ronald/...` path in `test/reference-validation.md`. Root `Manifest.toml` is correctly ignored.

### 2.7 CI (fully functional target)

Tiered design (`audit-hygiene.md` §4), ≈1 100 billed min/month:

| Tier | Trigger | Jobs | ≈min |
|---|---|---|---|
| T0 | every PR push, push to main | Runic format; unit tests on Julia `1` (2 threads) + coverage; docs build (no deploy); techref build if `docs/techref/**` changed | 30 |
| T1 | push to main, tags | + unit on LTS/1.10 (or 1.11, see O3); perf allocation contracts pinned to one minor; Makie ext tests; benchmark smoke; **techref PDF → docs/src/assets → Documenter deploy** | +25 |
| T2 | weekly, manual, PR label `run-integration` | integration-fast groups in 3 shards | 45 |
| T3 | monthly/manual, `continue-on-error` until O4 triaged | KH/RT, CI-sized cylinder | 45–200 |

Plus on every job: `timeout-minutes`, `concurrency` with cancel-in-progress on PRs, `julia-actions/cache`, pinned `ubuntu-24.04`; an aggregate `ci-ok` job as the only required check (branch protection); docs preview cleanup on PR close; dependabot (actions) + CompatHelper; `canonical` URL and `size_threshold_warn` in `docs/make.jl`. Failed logs are still retrievable via `gh api repos/…/actions/jobs/<id>/logs` (≈90-day retention).

---

## 3. Not verified

Threading beyond 4 threads; problem sizes > 30k cells; JET (`report_package`) not run (memory on shared machine); Makie extension not covered by Aqua; AMR field-transfer cost for an organic topology change; warm cost of `interfacial_instability`, `cylinder_flow`, full `lgpi`, `manufactured_poisson`; isolated repro of E2; the 30-minute KH group was not rerun locally; tech-ref audit read several thesis sections (cut cells, `int_*`, `st_apert`, `st_csf`, `pt1_momentum`) at heading level only. Julia 1.12.7 crashed in LLVM/GC during compilation in several runs (precompile workloads reproducibly; others not reproducible) — not attributed to the repository.
