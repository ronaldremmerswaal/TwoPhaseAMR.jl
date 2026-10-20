# Audit: hygiene slice (tests, CI, docs deployment, benchmarks, repo hygiene)

Auditor: hygiene agent. Date: 2026-10-20. Repo state: `main` @ 157448c. Read-only; scratch in `/tmp/audit-hygiene/` (downloaded job logs `jl110.log jl1.log instab.log waves.log cyl.log doc.log alljobs.jsonl`).
Evidence tags: **[obs]** = observed (command/log/file:line), **[INF]** = inference.

## 1. Summary

- **CI has never been green** (67/67 `CI` runs failed since 2026-09-13, all PRs #1–#7 merged red, no branch protection) **and cannot run now** (billing block since 2026-09-20 07:10Z). September used **3 309 billed Linux minutes** (CI 3 281 + Docs 28) against 2 000 free on a Free plan; one `single-phase cylinder` job alone burned 125–152 min per run and the old monolithic "Integration" job 165–227 min.
- **Logs are NOT gone**: `gh run view --log-failed` returns nothing, but `gh api repos/ronaldremmerswaal/TwoPhaseAMR.jl/actions/jobs/<job_id>/logs` still works (verified; I saved the four). Default retention is 90 days [INF] → they expire ~2026-12-19; archive them if still needed. All four failures in run 35495856106 are real test failures, not infra: (1) Julia 1.10: `ldiv!(::CHOLMOD.Factor, ::Vector)` MethodError (4 errors, `src/solvers/multigrid/cycles.jl:259-267`); (2) KH instabilities: 3 deterministic assertion failures (30 min); (3) capillary waves: temporal-refinement monotonicity assertion; (4) cylinder: unreachable physics (`pressure_drag_order = -0.88`, Strouhal 40.5/83.5/0.68) asserted only after a 2h32m study that runs at file top level with no `timeout-minutes`.
- **Docs**: `deploydocs` works (previews were pushed to `gh-pages`), but Pages is disabled (404; private repo, needs Pro+), `dev/` was never deployed (all 6 main-push Documentation runs were billing-blocked; the workflow was introduced 2026-09-19), `gh-pages` has no root `index.html`, and README's docs badge links to a non-existent `/dev/`.
- **Hygiene**: 4.7 GB untracked `benchmark/scaling/results` (237 `.bin` = 4.57 GB, 8 241 copied source/test files = 108 MB), leftovers from the `threading` branch which has its own ignore rules; `research/` on `main` = 10 Fortran build artefacts + a `.pyc`, no sources; `.agents/` `.codex/` empty; root `Manifest.toml` is **ignored, not tracked** (good). `Random` and `SpecialFunctions` are in `[deps]` but unused by `src/`+`ext/` (test-only).
- A cheap design (§4) keeps a typical month at ≈1 100 billed min (fits the 2 000 free minutes) while adding Pages deploy + LaTeX tech-ref + required-status aggregation; the decisive owner decision is billing/visibility (§4.5). At the documented $0.006/min, September's overage (≈1 309 min) would have cost ≈ **$8** — [INF] the real blocker is the failed payment method, not the volume.

## 2. CI job table (last real run = 35495856106, PR "Refactor", 2026-09-20, head d6e5febd; wall 152.5 min, billed ≈271 min)

Triggers for `CI.yml`: `push` (master, main) + `pull_request`. For `Documentation.yml`: push main, tags `*`, PR, dispatch.

| Job | Trigger | Last real duration | Last real result | Probable cause / note |
|---|---|---|---|---|
| Runic formatting | push+PR | 1.0 min | success | — (also 17 earlier failures from formatting drift) |
| Julia 1.10 (4 thr), coverage on | push+PR | 14.7 min (unit testset 13m54) | **failure** (12 649 pass, 4 errored) | `MethodError ldiv!(::CHOLMOD.Factor{Float64,Int64}, ::Vector{Float64})` from `mimetic_projection.jl:18/70/133/194` via `ExternalPreconditioner` `ldiv!` (`cycles.jl:259-267`). On 1.10 the generic `ldiv!(Y,A::Factorization,B)` (`factorization.jl:168`) exists so `applicable` returns true, then calls the 2-arg in-place `ldiv!(F, y)` which CHOLMOD lacks before 1.11. I reproduced the same MethodError with a 5-line standalone script on `julia +1.10`. Coverage/Codecov steps were therefore skipped and have never run. |
| Julia 1 (4 thr) | push+PR | 20.5 min (testset 18m01; 13 063 pass) | success | `version: '1'` resolved to **1.13.0** (log), while local default is 1.12.7. 33 `FlowSimulation(...) is deprecated` warnings in the log. |
| Performance contracts (1 thr) | push+PR | 3.6 min | success | allocation contracts only; run on one Julia version (they failed on 1.10/1.11 in the 2026-09-13 runs: `@allocated … == 0`, old_110.log) |
| Makie extension | push+PR | 1.9 min | success | does `Pkg.add("CairoMakie")` into the package project at CI time |
| Benchmark smoke test | push+PR | 3.3 min | success | — |
| Integration / manufactured wave equation | push+PR | 1.4 min | success | |
| Integration / staggered manufactured wave | push+PR | 1.5 | success | |
| Integration / manufactured AMR flow | push+PR | 1.3 | success | |
| Integration / vortex reversal | push+PR | 4.5 | success | |
| Integration / two-velocity transport | push+PR | 2.3 | success | |
| Integration / two-velocity rotation | push+PR | 2.6 | success | |
| Integration / continuous two-velocity vortex reversal | push+PR | 6.4 (10.8 in previous run) | success | |
| Integration / **two-velocity instabilities** | push+PR | **31.8 min** (testset 30m39; 40 min in prior run) | **failure** (207 pass, 3 fail) | `two_velocity_instabilities.jl:195` `all(diff(temporal.errors) .<= 0)` (DR=1000 frozen: errors 0.465,0.505,0.497 / projected 0.674,0.640,0.651) and `:180` `endpoint_rate >= 0.8` (projected DR=1000: 0.493). Values are **bit-identical** to the 2026-09-19 runs (e.g. 0.4929818228808897) ⇒ deterministic, not flaky; the asserted properties do not hold for the scheme/parameters. Never passed in CI (7/7 failed under its old name too). |
| Integration / **capillary and gravity waves** | push+PR | 6.3 min (testset 4m59) | **failure** (135 pass, 1 fail) | `two_phase_waves.jl:341` `all(diff(temporal_errors) .<= 0)` for `model=frozen DR=1`: errors 0.00653, 0.00736, 0.01089 for dt factors 1, .5, .25 (error *grows* when dt shrinks; spatial error dominates). Same family of assertion as KH. 0/16 success history. |
| Integration / floating droplet | push+PR | 2.5 | success | |
| Integration / sessile droplet | push+PR | 1.9 | success | |
| Integration / anisotropic cells | push+PR | 2.5 | success | |
| Integration / **single-phase cylinder** | push+PR | **152.4 min** (125 in prior run; 165–227 as part of the old monolith) | **failure** (10 pass, 1 fail) | see §3.1: `study.pressure_drag_order >= 1.75` evaluated **after** 2h32m; got −0.884. Strouhal (40.5, 83.5, 0.68), max lift (3e-14, 6e-13, 3e-4): n=4,8 never break symmetry within `final_time=40`. No `timeout-minutes` (default 360). |
| Documentation / Build and deploy | PR (success ×9), main push (never succeeded) | 4.0 min | success on PR (preview pushed to `gh-pages/previews/PR6`) | main-push runs: 35496414941 (billing block), 36266374950…36476590381 (billing). Warnings: `api.md`, `api-functions.md` exceed 100 KiB `size_threshold_warn`. |
| All jobs since 2026-09-20 07:10Z | any | 0–5 s | failure | annotation: "The job was not started because recent account payments have failed or your spending limit needs to be increased." (job never starts; unrelated to workflows) |

Billed-minute breakdown (ceil per job, 62 CI runs, sum 3 281): old monolithic Integration 997 (30 %), cylinder 281 (9 %), Julia 1 254, Julia 1.11 225, KH 215, Julia 1.12 178, Julia 1.10 149+120. Per full-matrix run ≈ 270 min; 7 such runs on 2026-09-19/20 (215–279 min each, ≈1 800 min) plus ~14 shorter ones exhausted the quota by the 20th (every push to a PR branch re-ran everything; no `concurrency` cancel).

## 3. Findings table

Severity: H = fix first, M = fix soon, L = nit. Effort S ≤ ½ day, M ≤ 2 days, L > 2 days.

| ID | Sev | Area | Location | Evidence | Proposal | Benefit | Effort |
|---|---|---|---|---|---|---|---|
| CI-01 | H | billing | account settings | runs after 2026-09-20 07:10Z fail in 0–5 s with payment annotation; Sept = 3 309 min vs 2 000 free | Owner decision §4.5 (fix payment/budget, go public, or self-host heavy tier); adopt tiered design to stay < 2 000 min | CI can run at all | S |
| CI-02 | H | merge gating | repo settings | 67/67 CI runs failed; 7/7 PRs merged red; `branches/main/protection` → 404 | After CI is green: branch protection (needs Pro for private) requiring one aggregate job `ci-ok` | red CI stops being ignored | S |
| CI-03 | H | Julia 1.10 | `src/solvers/multigrid/cycles.jl:259-267`; `Project.toml: julia = "1.10"`; README badge "Julia 1.10+" | CI log `jl110.log:569-740`; standalone repro on `julia +1.10`: `applicable(ldiv!,y,F,b)==true` yet `ldiv!(y,F,b)` and `copyto!; ldiv!(F,y)` both MethodError, `F\b` works | In the `ExternalPreconditioner` `ldiv!`: `VERSION < v"1.11" && value isa SparseArrays.CHOLMOD.Factor` → `copyto!(destination, value \ right_hand_side)` (allocating, 1.10 only); or raise compat to 1.11/1.12 and drop 1.10 cell. Add a unit test that exercises 3-arg `ldiv!` on a CHOLMOD factor outside the mimetic path | 1.10 cell green (only failure on that version: 4 errors of 12 653) | S |
| CI-04 | H | cylinder job | `test/integration/single_phase_cylinder.jl:12-46`, `cylinder_reference_problem.jl:267-300`, `CI.yml` | job 152 min, asserts after study; `lift` 3e-14 ⇒ no shedding; `pressure_drag_order=-0.88` | (a) immediately: remove from the on-push matrix, `timeout-minutes`; (b) make failure cheap: move `cylinder_paper_validation` inside the `@testset`, abort per resolution when `converged=false`/no Strouhal; (c) redefine tiers: CI-sized check (n=(4,8), t≈5, check drag/pressure residual vs stored reference ±tol) + manual long validation producing a table artifact, not a hard 1.75 gate; shedding needs perturbation/longer `final_time` (t=40 too short at uniform symmetric start) [INF] | −150 min/run; truthful test | M |
| CI-05 | H | KH/RT instabilities | `test/integration/two_velocity_instabilities.jl:180,195,214,228` | deterministic failures (bit-identical numbers on 09-19 and 09-20); 30–40 min | Owner/numerics triage: either the scheme does not reach the asserted order/monotonic dt behaviour (then it is a result, record it) or thresholds are wrong. Until triaged: move to *heavy* tier with `continue-on-error: true`; replace `all(diff(temporal.errors) .<= 0)` (true only when temporal error dominates) by a check against the observed spatial-limit (error ≤ spatial error·(1+tol)) | stops 30-min red job | M |
| CI-06 | H | waves | `test/integration/two_phase_waves.jl:341` | `capillary temporal refinement … factors=(1.0,0.5,0.25) errors=[0.00653,0.00736,0.01089]` | same: dt-refinement monotonicity is not an invariant when spatial error dominates; assert `error(dt/4) ≤ error(dt) + c·spatial_error` or refine both | green group (135/136 pass already) | S |
| CI-07 | H | no limits | `.github/workflows/*.yml` | `grep timeout-minutes` → none; `concurrency` → none | `timeout-minutes` on every job (10 format, 35 unit, 25 integration-fast, 240 heavy, 15 docs); `concurrency` group per ref with `cancel-in-progress` for PRs | caps worst case at hours→minutes; cancels superseded pushes (7 full-length runs of 215–279 min on 09-19/20) | S |
| CI-08 | H | test tiers | `CI.yml:integration`, `test/integration/runtests.jl` | all 13 integration jobs + perf + makie + bench run on every push AND again on merge push; `Pkg.test` itself is 14–18 min | tier design §4: unit on PR; integration-fast nightly/weekly/label; heavy manual/monthly | ≈270 → ≈30 min per PR push | M |
| DOC-01 | H | Pages | `gh api …/pages` → 404; `gh-pages` tree = `previews/`, `versions.js` only | Documenter deploys fine (doc.log `Deploying: ✔`) but nothing is served; `dev/` never built on main (6/6 main runs blocked); no root `index.html`; README badge → `…github.io/TwoPhaseAMR.jl/dev/` | Enable Pages (source `gh-pages`/root; private repo needs Pro and the site is then *public*), trigger one main deploy, tag `v0.1.0` for `stable` (§4) | docs reachable, stable URLs | S |
| DOC-02 | M | preview cleanup | `Documentation.yml` | previews accumulate in `gh-pages/previews/PRn` | add the standard Documenter `pull_request: types:[closed]` cleanup workflow | gh-pages stays small | S |
| DOC-03 | M | doc size | `docs/make.jl`; log warns `api.md`, `api-functions.md` > 100 KiB | | raise `size_threshold_warn` or split pages | clean log | S |
| DOC-04 | M | doc code never runs | 47 ```` ```julia ```` fences, 0 `@example`/`jldoctest` in `docs/src` (grep); `test/readme.jl` hand-copies the README snippet | snippet drift undetected | convert 5–8 key tutorial snippets to `@example` (small meshes) or add `jldoctest`; keep README snippet test | docs stay true | M |
| DOC-05 | M | tech-ref | none exists (`git ls-files '*.tex' '*.bib'` → 0); `TODO_LATER.md` "Technical report" | | propose `docs/techref/` + CI job §4.4; single `refs.bib` (docs pages carry hand-written reference lists with DOIs, e.g. `two-phase-flow.md:202-230`) | thesis-style PDF published with docs | L (writing) / S (CI) |
| TST-01 | M | group selection | `test/runtests.jl`, `test/integration/runtests.jl`, `test/performance/runtests.jl`, `CI.yml:76-117` | only env var is `TWOPHASE_TEST_PROGRESS`; integration file lists are duplicated in CI matrix (`files:`) and `integration/runtests.jl`; drift not checked | one registry `test/groups.jl` (group → files), selected by `ARGS`/`TWOPHASE_TEST_GROUP`; matrix passes group names; a test asserts every `test/**/*.jl` is registered (or deliberately excluded) | single source of truth, local `julia test/runtests.jl mesh` | S |
| TST-02 | M | deps layout | `Project.toml [extras]/[targets]`; no `test/Project.toml` | `SpecialFunctions` only in `test/integration/prosperetti_amplitude.jl:2`; `Random` only in `test/navier_stokes/two_phase_navier_stokes.jl:3`; both in `[deps]` (`src/`,`ext/` grep: 0 hits) — SpecialFunctions was added to `[deps]` after CI failed "Package SpecialFunctions not found" (old_twophase.log:461); `benchmark = ["BenchmarkTools"]` target is unused (no test uses BenchmarkTools; `benchmark/Project.toml` exists) | create `test/Project.toml` (Test, SpecialFunctions, Random, AlgebraicMultigrid, LimitedLDLFactorizations, Aqua, [JET]) with `[compat]`; drop `[extras]/[targets]`, remove SpecialFunctions & Random from `[deps]`; add `[compat]` for LinearAlgebra/Random/SparseArrays (Aqua) | Aqua clean, lighter package load/precompile, compat honest | S |
| TST-03 | M | Makie job | `CI.yml:makie-extension`, `test/makie_visualization.jl:3` | `Pkg.add("CairoMakie")` mutates the package env at run time; GLMakie also referenced in tests/examples | commit `test/makie/Project.toml` (CairoMakie + path dep), run `julia --project=test/makie test/makie_visualization.jl`; fold into group `makie` | reproducible locally; Makie ext coverage can be merged | S |
| TST-04 | M | wording/implementation pins (owner policy: delete) | `test/diagnostics.jl:89-94`; `test/navier_stokes/high_level_api.jl:37`; `test/two_phase/two_velocity/projection/cmdgfm_projection.jl:425-437`; `test/navier_stokes/two_phase_navier_stokes.jl:20` | `occursin("TwoPhaseAMR simulation summary"…)`, `occursin("steps this run: 3")`, `occursin("FlowSimulation(time=0.0", sprint(show, flow))`, calls private `CMD._solver_failure_message` and asserts `"CMDGFMProjectionStats"`, `"dt=0.001, max_cycles=10"`; `:interface in propertynames(flow)` | delete the text assertions (keep `isnothing(print_simulation_summary(...))`, and `@test_throws` the error type for solver failure); drop the `propertynames` pin (line before already checks `flow.interface === f`) | tests survive message/layout refactors | S |
| TST-05 | M | private-API coupling | 368 refs to 77 distinct `TwoPhaseAMR._*` in 28 files; `_get` alone 130 (`src/mesh/variables.jl:94`) | e.g. `test/indicator/geometry.jl:68`, `height_functions.jl` (70), `visualization/fields.jl` (46), `mesh/amr_connectivity.jl` (39) | give `CellField` a public `Base.getindex(field, ::CellRef)` (or `cell_value`) and migrate the 130 `_get` uses; then review remaining private-symbol tests case by case (some are legitimately unit tests of geometry kernels like `_clip_polygon`, `_polygon_area`) | refactor-proof tests | M |
| TST-06 | M | exported but untested | export list vs `test/**` (regex, 231 names) | no occurrence in any test: `ZeroGradient` (default BC, used internally), `GeometricMultigridSolver`, `StaggeredGeometricMultigridSolver`, `algebraic_multigrid_preconditioner`, `incomplete_cholesky_preconditioner`, `incomplete_lu_preconditioner` (tests go through `preconditioner = :ichol`), `enforce_geometry_no_slip!`, `register_interface!`, `diagnostic_name`, `DiagnosticCollection`, `DiagnosticHistory`, `AMRFaceGeometry`, `PLICSolidGeometry`, `LinearSolverStats`, `MultigridStats`, `EveryStep`, `CentredMagnitude`, `VisualizationCheckpoint/DiagnosticSample/Topology`, plus abstract supertypes (`AbstractBoundaryCondition`, `AbstractDiagnostic`, `AbstractDiagnosticSchedule`, `AbstractRefinementIndicator`, `AbstractStaggeredAdvectionScheme`, `AbstractWallBoundaryCondition`, `AbstractEllipticOperator`, `LeafBlock`) | add one call-site test for each concrete function/solver name; Aqua `undocumented_names`/`check_exports` for the rest; `GeometricMultigridSolver` is an alias (`cycles.jl:3-6`, "Compatibility alias") — per clean-cutover policy delete rather than test | public surface fully exercised | S–M |
| TST-07 | M | deprecated API in tests | CI log jl1.log: 33× `FlowSimulation(mesh; density=…) is deprecated` | `FlowSimulation(` appears 73× in 18 test files | migrate tests to `SinglePhase/TwoPhase`; keep exactly one `@test_logs (:warn, …)` test of the shim; or delete the shim (docs/migration.md promises it through 0.2.x while `version = "0.1.0-DEV"`) | quiet logs | S |
| TST-08 | M | order-dependent test | `test/solvers.jl:4-16` | asserts the extensions are **inactive** — true only if no earlier test in the same process loaded AlgebraicMultigrid/LimitedLDLFactorizations/Makie; `@test_skip _optional_linear_solver_dependencies` (line 47) is a no-op on a Bool | run that check in a fresh subprocess (`run(\`$(Base.julia_cmd()) --project -e …\`)`) or delete; remove the stray `@test_skip` | safe under any group order / ParallelTestRunner | S |
| TST-09 | M | version-fragile allocation contracts | `test/performance/*.jl`; failures on 1.10/1.11 `old_110.log:715` | `@allocated(…) == 0` contracts only pass on the one version/threading used | run `performance-contracts` on pinned minor (e.g. 1.12) rather than floating `'1'`; document | no spurious reds when `'1'` bumps | S |
| TST-10 | L | unit suite is not "fast" | 1.13 log: Mesh 1m13, Indicators 2m22, Operators 2m47, Two-phase 7m42 (transport 2m42, `two_velocity.jl` 3m04), NS 1m49, Examples 1m34 = 18m01 | split `two_velocity/transport/two_velocity.jl` + `two_phase/transport.jl` into group `slow`; give nested per-file timings by making `run_test_file` testsets verbose | PR feedback ≈ 8–10 min | M |
| TST-11 | L | tooling | — | 17.6 k lines of top-level scripts, helpers defined in `Main` (182 helper names, no duplicates, so no collisions today) | **ReTestItems/TestItems: not worth it** (L effort: all files share Main-level helpers; would rewrite 70 files). **ParallelTestRunner**: optional for local runs on the 32-core box once TST-08 is fixed and helpers are per-file self-contained; no CI-minute benefit (minutes are total CPU time). Keep `Test` + group registry | avoid a rewrite | — |
| TST-12 | L | fixtures | `BlockTreeMesh(` 318× in 52 files (`block_size=(4,4)` 22×, `(8,8)` 20×); `InterfaceField(` 192× | fixtures would save little; leave | — | — | — |
| TST-13 | M | Aqua/JET not in CI | no `test/aqua.jl`; earlier Aqua audit: stale deps, missing compat (LinearAlgebra, Random, SparseArrays, Test, BenchmarkTools), `Diagnostic` ambiguity | add `test/aqua.jl` (`Aqua.test_all(TwoPhaseAMR)`) to unit group after TST-02; JET `test_package` as non-blocking job on one version first | catches ambiguities/stale deps | S |
| CI-09 | M | coverage | `CI.yml:51-59`; README Codecov badge | upload only from the 1.10 cell (always failing ⇒ never uploaded); `fail_ci_if_error: true` + `use_oidc` for a private repo untested [INF: needs Codecov GitHub app / token] | move coverage to the unit cell on `'1'`; `fail_ci_if_error: false`; drop `id-token: write` unless OIDC confirmed; merge Makie/ext coverage via `julia-processcoverage` across jobs | coverage actually reported | S |
| CI-10 | L | floating versions | `CI.yml` `version: '1'`, `'1.10'` | `'1'` = 1.13.0 in CI vs 1.12.7 local default | use `lts` (or 1.10) and `1` explicitly; reproduce CI failures with `julia +1.13` (installed: `juliaup status` shows 1.13.1) | repro parity | S |
| CI-11 | L | cache keys | log `Cache key: …;job=integration;…;files=…` | 13 integration entries keep separate ~280 MB caches (same depot) | set `cache-name`/`cache-paths` once for the integration matrix; 10 GB/repo cap | fewer misses/evictions (`No cache found` in cylinder job) | S |
| CI-12 | L | action pinning/updates | `actions/checkout@v7`, etc. | tags only; no `dependabot.yml`; no CompatHelper | `.github/dependabot.yml` (github-actions, weekly), `CompatHelper.yml` (weekly), TagBot only after registration | supply-chain + dep freshness | S |
| CI-13 | L | ubuntu-latest migration | annotation: "`ubuntu-latest` will migrate to Ubuntu 26 beginning October 19, 2026" | already in effect today | pin `ubuntu-24.04` until verified | avoid surprise | S |
| HYG-01 | M | untracked bulk | `benchmark/scaling/` 4.7 GB (results: 237 `.bin` 4.57 GB, 582 json 196 MB, 8 241 copied src/test/benchmark-project files 108 MB; 148 result dirs; 9 554 `.jl` files) | `git status` shows `?? benchmark/scaling/`; `main/.gitignore` has no rule; `threading@4f034c6:.gitignore` does (`/research/*…`, `benchmark/scaling/.gitignore`) so these are leftovers of switching branch; a `git add -A` would stage 12 554 files; every repo-wide grep/Runic/`include` walk also traverses copies (my grep hits `benchmark/scaling/results/*/test/…`) | move `results/` out of the repo (e.g. `~/data/TwoPhaseAMR-scaling/`) or `git worktree` for `threading`; add to `.git/info/exclude`: `/benchmark/scaling/`, `/research/`, `/docs/src/assets/stage*`, `**/__pycache__/` | safe `git add`, fast tooling | S |
| HYG-02 | L | research/ | `research/UniformLWAdvection/{fortran/build,fortran/diagnostics/build,experiments/…/__pycache__}` = 10 binaries/`.pyc` | no sources on main (sources are tracked on `threading`, commit f0a7bd8 "Migrate … to Julia" etc.) | delete from the `main` worktree; research belongs on its branch or its own repo, not in the package | — | S |
| HYG-03 | L | `docs/src/assets/` | untracked: `stage2-*.svg`, `stage2-primary-summary.json`, `cache-cliff/`, `julia-fortran-matched/` (4.5 MB); referenced by no page | Documenter copies everything under `src/` into the site if built locally | move out/ignore | — | S |
| HYG-04 | L | `.agents/`, `.codex/` | both empty dirs (untracked automatically) | | delete or `.git/info/exclude` | — | S |
| HYG-05 | L | `TODO_LATER.md` | tracked, 3.2 KB; line 1 is a bare `# `; includes "CI failure – currently the CI fails for several reasons", AMR criteria AND/NOT, LS-STAG, QR turbulence, tech report plan | convert each section to a GitHub issue (issues enabled, 0 open) with labels `ci`,`numerics`,`docs`; delete the file | trackable, linkable from PRs | S |
| HYG-06 | L | Manifest | `git ls-files Manifest.toml` → empty; `.gitignore` `/Manifest.toml` | correct for a package; local Manifest (Sep 19) is stale | none; (optional) commit `test/Manifest.toml`? no | — | — |
| HYG-07 | L | CITATION/CHANGELOG | absent (`git ls-files`) | thesis DOI 10.33612/diss.562133319 appears in 5 doc pages; `arXiv:2209.14934` (Remmerswaal & Veldman) in `two-phase-flow.md:213` | add `CITATION.cff` (§4.6) and `CHANGELOG.md` (Keep-a-Changelog, start at `0.1.0-DEV`); mention in README | citability, release notes | S |
| HYG-08 | L | stale paths | `test/reference-validation.md:8-13` hard-codes `/home/ronald/Documents/…`; `test/integration/mimeticfv_reference.jl` is intentionally unreferenced (OK) | parameterise with `$MIMETICFV_DIR` | — | S |
| HYG-09 | L | migration doc vs version | `docs/src/migration.md:6-9` "throughout the `0.2.x` series" while `version="0.1.0-DEV"` | align | — | S |

## 4. Proposed CI design

### 4.1 Tiers

| Tier | When | Contents | Budget (billed min) |
|---|---|---|---|
| T0 fast | every PR push + push to main | `format` (1.5), `unit` on Julia `1`, 4 threads, coverage (≈21; shrinks to ≈12 after TST-10), `docs` build w/o deploy (≈4), `techref` build if `docs/techref/**` changed (≈4) | ≈27–31 per push |
| T1 merge | push to main, schedule | + `unit` on `lts`/1.10 (15) + `contracts` (perf 3.6 + makie 2 + bench-smoke 3.3 ≈ 9) + docs *deploy* | +≈25 |
| T2 integration-fast | weekly schedule, `workflow_dispatch`, PR label `run-integration` | the 11 groups that finish < 7 min (sum of job time ≈ 34 min + 11×≈1 min setup) – as 3 balanced shards | ≈45 |
| T3 heavy | monthly schedule / manual only, `continue-on-error: true` until CI-04/05 are triaged | KH/RT instabilities (≈32), cylinder (CI-sized ≈ 10; long study manual) | ≈45–200 |

Monthly estimate: 20 PR pushes × 30 = 600; 4 merges × 55 = 220; 4 weekly T2 × 45 = 180; 1 T3 × ≈100 → **≈1 100 min < 2 000** (Free) — vs 3 309 observed in September. (Billed per job rounded up; Linux 2-core; [INF] private-repo `ubuntu-latest` = 2 vCPU/7 GB, so `JULIA_NUM_THREADS=4` oversubscribes: use 2 for T0/T2.)

### 4.2 Workflow sketch (structure only)

```yaml
# .github/workflows/ci.yml
on:
  pull_request:                      # + types: [opened, synchronize, reopened, labeled]
  push: { branches: [main] }
  schedule: [{cron: '0 3 * * 1'}]    # weekly T2 ; monthly T3 via second cron + `github.event.schedule`
  workflow_dispatch: { inputs: { tier: { type: choice, options: [unit, integration, heavy, all] } } }
concurrency:
  group: ci-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}
permissions: { contents: read }
env: { OPENBLAS_NUM_THREADS: '1', JULIA_NUM_THREADS: '2' }
jobs:
  format:   { timeout-minutes: 10, runs-on: ubuntu-24.04, steps: [checkout, runic-action] }
  unit:                                   # T0 (+ lts cell only when event != pull_request)
    timeout-minutes: 35
    strategy: { fail-fast: false, matrix: { include: [ {julia: '1', cov: true}, {julia: lts, cov: false, merge-only: true} ] } }
    if:  # skip merge-only cells on PRs (use fromJSON matrix or per-step `if`)
    steps: [checkout, setup-julia, julia-actions/cache, buildpkg, runtest{coverage: cov}, processcoverage+codecov (cov only; fail_ci_if_error:false)]
  contracts:                              # T1: perf + makie + bench smoke, 1 job, needs: [] 
    if: github.event_name != 'pull_request' ; timeout-minutes: 25
  integration:                            # T2
    if: schedule || dispatch || contains(github.event.pull_request.labels.*.name,'run-integration')
    timeout-minutes: ${{ matrix.timeout }}   # 25 for shards
    strategy: { fail-fast: false, matrix: { group: [integ-a, integ-b, integ-c] } }
    run: julia --project=. -t2 test/runtests.jl ${{ matrix.group }}      # groups from test/groups.jl
  heavy:                                  # T3
    if: dispatch(tier in [heavy,all]) || monthly cron ; timeout-minutes: 240 ; continue-on-error: true
    matrix: { group: [kh-rt, cylinder-ci] }
  techref:                                # also defined in docs.yml (see 4.3) – reusable workflow `workflow_call`
  ci-ok:                                  # the ONLY required status check
    needs: [format, unit, docs-build]; if: always()
    steps: [ fail if any needed result is failure/cancelled ]

# .github/workflows/docs.yml
on: { push: {branches:[main], tags:['v*']}, pull_request:, workflow_dispatch: }
concurrency: { group: docs-${{ github.ref }}, cancel-in-progress: ${{ github.event_name=='pull_request' }} }
jobs:
  techref:  { timeout-minutes: 15, steps: [checkout, xu-cheng/latex-action@v3 {working_directory: docs/techref, root_file: main.tex, latexmk_use_lualatex?: false, args: -pdf -file-line-error -halt-on-error -interaction=nonstopmode}, upload-artifact techref-pdf] }
  docs:     { needs: techref, timeout-minutes: 20, permissions: {contents: write, pull-requests: write, statuses: write},
              steps: [checkout, setup-julia, cache, download-artifact -> docs/src/assets/TwoPhaseAMR-techref.pdf,
                      Pkg.develop+instantiate docs env, `julia --project=docs docs/make.jl` (deploydocs inside; GITHUB_TOKEN)] }
# .github/workflows/docs-cleanup.yml   on: pull_request: types: [closed]  -> Documenter preview cleanup
# .github/dependabot.yml  github-actions weekly ;  CompatHelper.yml weekly
```

Required code-side changes: `test/runtests.jl` group registry (TST-01); `docs/make.jl`: add `format = Documenter.HTML(…, canonical = "https://ronaldremmerswaal.github.io/TwoPhaseAMR.jl/stable/", size_threshold_warn = 300*2^10)` and a "Technical reference (PDF)" link page; keep `deploydocs(devbranch="main", push_preview=true)`.

### 4.3 Documenter + PDF with stable URLs

- Put the PDF in `docs/src/assets/` at build time ⇒ Documenter versions it with the site: `<site>/dev/assets/TwoPhaseAMR-techref.pdf`, `<site>/stable/assets/…` (after the first `vX.Y.Z` tag), `<site>/vX.Y.Z/assets/…`, `<site>/previews/PRn/assets/…`. API pages link relatively (`[PDF](assets/TwoPhaseAMR-techref.pdf)` works in every version). Canonical citable URL = `/stable/assets/TwoPhaseAMR-techref.pdf`.
- Back-links from the PDF: CI passes the docs version to LaTeX, e.g. `latexmk -pdflatex='pdflatex %O "\def\docsver{dev}\def\gitsha{<sha>}\input{%S}"'` with `\newcommand\apiref[1]{\href{\docsroot/\docsver/api/#1}{…}}`; Documenter anchors are `#TwoPhaseAMR.<name>`. Link implementation to code via `https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/\gitsha/src/...#Lnn` (404 for outsiders while the repo is private).
- Alternative/addition on tags: attach the PDF to the GitHub Release.
- `gh-pages` root gets Documenter's redirect `index.html` only after the first main deploy — trigger one `workflow_dispatch` run after enabling Pages.

### 4.4 Tech-ref location and build

Propose `docs/techref/`: `main.tex`, `preamble.tex`, `sections/*.tex`, `refs.bib` (migrate the hand-written reference lists in `docs/src/*.md` — e.g. `two-phase-flow.md:202-230`, `navier-stokes.md:396-` — into one bib; optionally DocumenterCitations.jl so both outputs share it), `figures/` (TikZ/pgfplots + committed CSV from integration artifacts), `latexmkrc`. No `.tex` exists yet; `TODO_LATER.md` "Technical report" asks for a plan first. CI cost ≈ 3–5 min (latex-action docker pull ≈1 min). Figures that come from convergence studies should be produced by T2/T3 jobs as artifacts and committed/imported deliberately, not regenerated on every push.

### 4.5 Decision the owner must make (billing / visibility)

| Option | Minutes | Pages | Notes |
|---|---|---|---|
| A. Make repo public | unlimited (standard runners, 4 vCPU/16 GB) | free, public | LICENSE (MIT) already present; review history (branches `threading`, `copilot/…`) and drop 4.7 GB leftovers first; Codecov/branch protection free |
| B. Stay private + fix payment method / budget (Free: 2 000 min; Pro $4/mo: 3 000 min) | quota above; overage $0.006/min Linux 2-core | **Pro or Team required** for Pages in private repos; the published site is **public** (private Pages = Enterprise Cloud org only) | Design in §4.1 fits Free quota; branch protection needs Pro |
| C. Private + self-hosted runner on the workstation for T2/T3 | free | as B | resource contention with other work; keep T0 on hosted runners |
Sources: docs.github.com billing/plans pages (read 2026-10-20): Free 2 000 / Pro 3 000 / Team 3 000 min, Linux 2-core $0.006/min, "GitHub Pages in public repositories" on Free, private Pages only via Enterprise Cloud org.

### 4.6 CITATION.cff sketch

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite the thesis below."
title: "TwoPhaseAMR.jl"
authors: [{family-names: Remmerswaal, given-names: Ronald}]
repository-code: "https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl"
license: MIT
preferred-citation:
  type: thesis
  title: "Numerical modelling of variability in liquid impacts"
  authors: [{family-names: Remmerswaal, given-names: R.}]
  institution: {name: "University of Groningen"}
  year: 2023
  doi: "10.33612/diss.562133319"
references:
  - type: article   # Remmerswaal & Veldman, arXiv:2209.14934
    title: "Towards a sharp, structure preserving two-velocity model for two-phase flow: transport of mass and momentum"
    authors: [{family-names: Remmerswaal, given-names: R. A.}, {family-names: Veldman, given-names: A. E. P.}]
    url: "https://arxiv.org/abs/2209.14934"
```
(Add the PDF/tech-ref as a second `references` entry once it exists.)

## 5. Details for H/M items

### 5.1 Why each job of run 35495856106 failed (logs retrieved via `gh api …/actions/jobs/<id>/logs`: ids 106038534100, …207, …251, …289)

1. **Julia 1.10** — see table/CI-03. Julia 1.10.12 runner; `ldiv!` MethodError ×4 in `mimetic_projection.jl` testsets (Three-block projection, AMR interface, embedded geometry, two-velocity NS); everything else (12 649 tests) passes. The `applicable(...)` guard in `cycles.jl:261-265` cannot detect this because the generic Factorization method is applicable. Repro (executed): `julia +1.10 -e 'using SparseArrays,LinearAlgebra; F=cholesky(sparse([2.0 -1 0;-1 2 -1;0 -1 2])); b=ones(3); y=similar(b); applicable(ldiv!,y,F,b) # true; ldiv!(y,F,b) # MethodError; F\b # ok'`.
2. **KH instabilities** — deterministic; 30m39 for 210 tests. Failing: KH DR=1000 `frozen` temporal errors non-monotone (`:195`), KH DR=1000 `projected` endpoint rate 0.493 < 0.8 (`:180`), and `projected` temporal errors `[0.674, 0.640, 0.651]` (`:195`). The earlier 09-19 run additionally failed RT endpoint rate 1.34 < 1.7 (`:214`) and combined KH/RT 0.7946 < 0.8 (`:228`) — thresholds were then edited; remaining ones were not. Whether these also fail on the author's Julia 1.12.7 was not run (30 min).
3. **Capillary/gravity waves** — one failure of 136; `:341` is the same kind of "temporal refinement must not increase error" assertion; for `frozen` DR=1 the error grows with smaller dt because the dt-independent spatial error dominates.
4. **Cylinder** — `single_phase_cylinder.jl` builds `study = cylinder_paper_validation(; cells_per_diameter=(4,8,16), final_time=40, …)` at file top level (line 21), i.e. outside the `@testset` (testset time printed "1.6 s"); 3 uniform single-block runs (56×32, 112×64, 224×128 cells, CFL 0.25, `tolerance=1e-9`, `max_cycles=1000`). Cost ≈ dominated by n=16 (each doubling ≈ 8× work) [INF; per-case time not logged — add timing prints]. Result tuple (log line 359): `pressure_drag=(0.542,0.611,0.738)` vs reference 0.93, `maximum_lift=(2.9e-14, 5.5e-13, 3.0e-4)` vs 0.3, `strouhal=(40.5, 83.5, 0.685)` vs 0.165 ⇒ vortex shedding never develops within t=40 on these grids (symmetric start), so the Strouhal and the order estimate are meaningless; `@test pressure_drag_order >= 1.75` fails with −0.884. No `timeout-minutes`, so the job ran to completion (2h32m). Wasted ≈ 125–152 min × 5 runs + 165–227 min × 5 (monolith) ≈ 25 job-hours.

### 5.2 Test-suite organisation

- No group mechanism: `Pkg.test()` = all unit files (hard-coded `run_test_file` list in `test/runtests.jl`), integration/perf have separate runner scripts, Makie file is run by name in CI, `mimeticfv_reference.jl` is deliberately unreferenced. Orphan check: every `test/**/*.jl` is either included by a runner, CI, or an including file, except `mimeticfv_reference.jl` (documented).
- Helpers: all included into `Main` (no cross-file name collisions found, 182 distinct helper definitions); `test_utils.jl` (8 lines) is conditionally included from two files. Mesh/interface construction repeated hundreds of times but trivially small ⇒ not worth fixtures.
- No timing-based assertions exist (`@elapsed`, `time_ns` grep → none); performance "scaling" contracts compare allocated bytes (`two_phase_performance.jl:199-240`), deterministic but Julia-version sensitive (TST-09).
- Count: 2 081 `@test` lines, 268 `@testset` lines, 13 063 passing assertions on 1.13; 139 `@test_throws` (type-only; good).

### 5.3 Test hygiene fixes (concrete)
Delete: `diagnostics.jl:89-94`, `high_level_api.jl:37`, `cmdgfm_projection.jl:431-437` (and the private `CMD._solver_failure_message` call), `two_phase_navier_stokes.jl:20`. Keep behaviour tests next to them.

## 6. Nits

- `Project.toml`: `[targets] benchmark` is not a Pkg-recognised target; `Test` listed only in `[extras]` without compat.
- `CI.yml`: `permissions: actions: write` on every job (only needed by cache cleanup), `id-token: write` only for Codecov OIDC; `max-parallel: 13` equals matrix size.
- `ubuntu-latest` is floating (see CI-13); actions unpinned to SHAs.
- `Documentation.yml` main trigger lists `branches: [main]` while `CI.yml` lists `[master, main]`.
- README: "Julia 1.10+" badge currently false (CI-03); docs badge target `/dev/` 404 (DOC-01).
- `docs/make.jl` has no `linkcheck`/`doctest`; `checkdocs=:exports` OK (docs build passes 9/9 on PRs).
- `benchmark/README.md` fine; benchmark smoke job is the only guard for `benchmark/` — keep in T1.
- `TODO_LATER.md` line 1 is a stray `# `.
- `.gitignore` ends with `/development/` (that dir was removed in 1d4e514) — harmless.
- Merge-then-push duplication: PR run + push-to-main run of the same tree (e.g. 35496359543 and 35496414940); acceptable if T0 only on push.
